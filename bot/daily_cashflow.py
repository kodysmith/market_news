#!/usr/bin/env python3
"""Daily Cash Flow Bot — 1DTE/3DTE iron condors + funded butterflies.

Runs independently from the main IC bot. Designed for daily consistent income.

Usage:
  python bot/daily_cashflow.py --mode dry-run --once   # Single check, no orders
  python bot/daily_cashflow.py --mode paper --once     # Single check, paper orders
  python bot/daily_cashflow.py --mode paper             # Run on schedule
  python bot/daily_cashflow.py --mode live              # Real money
"""
from __future__ import annotations

import argparse
import logging
import sys
import time
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s %(name)s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("bot.daily_cashflow")


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
@dataclass
class LayerConfig:
    name: str
    dte: int                  # Target DTE
    short_delta: float        # Delta for short strikes
    width: float              # Spread width in dollars
    profit_target_pct: float  # Close at this % of credit captured (0.5 = 50%)
    stop_mult: float          # Close if spread value >= this × credit
    n_contracts: int          # Number of contracts
    entry_time: str           # When to enter ("09:35", "09:40", etc.)
    is_butterfly: bool = False
    fly_wing: float = 0.0
    fly_rounding: float = 25.0
    fly_funder_width: float = 50.0


LAYERS = [
    LayerConfig(
        name="IC3_1DTE",
        dte=1,
        short_delta=0.10,
        width=30.0,
        profit_target_pct=0.50,
        stop_mult=2.0,
        n_contracts=10,
        entry_time="09:35",
    ),
    LayerConfig(
        name="IC6_3DTE",
        dte=3,
        short_delta=0.12,
        width=30.0,
        profit_target_pct=0.40,
        stop_mult=2.0,
        n_contracts=5,
        entry_time="09:40",
    ),
    LayerConfig(
        name="FLY_14DTE",
        dte=14,
        short_delta=0.10,
        width=50.0,  # funder width
        profit_target_pct=0.50,
        stop_mult=2.0,
        n_contracts=3,
        entry_time="09:45",
        is_butterfly=True,
        fly_wing=10.0,
        fly_rounding=25.0,
        fly_funder_width=50.0,
    ),
]

# Safety
MAX_DAILY_LOSS = 30_000       # Stop all new entries if day P&L < -$30K
MAX_MARGIN_TOTAL = 60_000     # Max margin across all daily cashflow positions
IBKR_PAPER_PORT = 7497
IBKR_LIVE_PORT = 7496
IBKR_CLIENT_ID = 30          # Different from main bot (client 10/20)


# ---------------------------------------------------------------------------
# Position tracking (in-memory, persisted to Supabase)
# ---------------------------------------------------------------------------
@dataclass
class OpenPosition:
    layer_name: str
    entry_date: date
    expiration: date
    k_put_short: float
    k_put_long: float
    k_call_short: float
    k_call_long: float
    credit: float            # Net credit received (per unit, not per contract)
    n_contracts: int
    # Butterfly legs (if applicable)
    fly_lower: float = 0.0
    fly_mid: float = 0.0
    fly_upper: float = 0.0
    fly_cost: float = 0.0    # Cost of butterfly (per unit)
    entry_id: str = ""


# ---------------------------------------------------------------------------
# Market data
# ---------------------------------------------------------------------------
def get_snapshot():
    """Get current SPX price and VIX."""
    from bot.market_data import get_market_snapshot
    snap = get_market_snapshot()
    return snap.spx_price, snap.vix_level


def find_expiration(dte: int, from_date: date) -> date | None:
    """Find the next trading day expiration approximately `dte` days out.

    For 0-1 DTE: use tomorrow (or today if 0DTE)
    For 2-3 DTE: find the right date
    For 14 DTE: use the Friday expiration cycle
    """
    target = from_date + timedelta(days=dte)
    # For short DTE, any trading day works (SPX has daily expirations)
    # For 14 DTE, prefer Fridays
    if dte <= 5:
        # Just use calendar day + dte, skip weekends
        d = from_date
        trading_days_ahead = 0
        while trading_days_ahead < dte:
            d += timedelta(days=1)
            if d.weekday() < 5:  # Mon-Fri
                trading_days_ahead += 1
        return d
    else:
        # Use Friday expirations for longer DTE
        from bot.market_data import get_friday_expirations
        fridays = get_friday_expirations(from_date, n=8)
        for f in fridays:
            cal_dte = (f - from_date).days
            if abs(cal_dte - dte) <= 3:
                return f
        return fridays[0] if fridays else None


def compute_strikes(spx: float, vix: float, dte: int, delta: float, width: float):
    """Compute iron condor strikes using BS model."""
    from bot.pricer import compute_strikes as _compute_strikes
    return _compute_strikes(spx, vix, dte, delta, width)


# ---------------------------------------------------------------------------
# Entry logic
# ---------------------------------------------------------------------------
def build_ic_entry(layer: LayerConfig, spx: float, vix: float, today: date) -> OpenPosition | None:
    """Build an iron condor entry for the given layer."""
    expiration = find_expiration(layer.dte, today)
    if expiration is None:
        logger.warning("[%s] No valid expiration found for DTE=%d", layer.name, layer.dte)
        return None

    cal_dte = (expiration - today).days
    if cal_dte <= 0:
        return None

    try:
        k_ps, k_pl, k_cs, k_cl = compute_strikes(spx, vix, cal_dte, layer.short_delta, layer.width)
    except Exception as e:
        logger.warning("[%s] Strike computation failed: %s", layer.name, e)
        return None

    # Compute credit
    from bot.pricer import compute_credit
    credit = compute_credit(spx, vix, cal_dte, k_ps, k_pl, k_cs, k_cl)

    if credit <= 0:
        logger.warning("[%s] Credit <= 0: %.4f", layer.name, credit)
        return None

    # Apply slippage
    slippage = 8 * 0.10 / 2.0  # 8 legs (IC), $0.10 per leg, half for entry
    credit_adj = credit - slippage
    if credit_adj <= 0:
        return None

    logger.info(
        "[%s] IC entry | SPX=%.0f VIX=%.1f DTE=%d | "
        "Put %.0f/%.0f Call %.0f/%.0f | credit=%.2f × %d cts",
        layer.name, spx, vix, cal_dte,
        k_ps, k_pl, k_cs, k_cl, credit_adj, layer.n_contracts,
    )

    return OpenPosition(
        layer_name=layer.name,
        entry_date=today,
        expiration=expiration,
        k_put_short=k_ps,
        k_put_long=k_pl,
        k_call_short=k_cs,
        k_call_long=k_cl,
        credit=credit_adj,
        n_contracts=layer.n_contracts,
    )


def build_fly_entry(layer: LayerConfig, spx: float, vix: float, today: date) -> OpenPosition | None:
    """Build a funded butterfly entry."""
    expiration = find_expiration(layer.dte, today)
    if expiration is None:
        return None

    cal_dte = (expiration - today).days
    if cal_dte <= 0:
        return None

    # Funder: bear call spread (10-delta, $50 wide)
    try:
        _, _, k_cs, k_cl = compute_strikes(spx, vix, cal_dte, layer.short_delta, layer.fly_funder_width)
    except Exception as e:
        logger.warning("[%s] Funder strike computation failed: %s", layer.name, e)
        return None

    from bot.pricer import compute_call_spread_credit
    funder_credit = compute_call_spread_credit(spx, vix, cal_dte, k_cs, k_cl)
    funder_slippage = 4 * 0.10 / 2.0
    funder_credit_adj = funder_credit - funder_slippage
    if funder_credit_adj <= 0:
        return None

    # Butterfly: put butterfly at nearest round strike
    fly_mid = round(spx / layer.fly_rounding) * layer.fly_rounding
    fly_lower = fly_mid - layer.fly_wing
    fly_upper = fly_mid + layer.fly_wing

    # Price the butterfly
    sigma = vix / 100.0
    T = max(cal_dte, 1) / 365.0
    from scripts.optimize_spx_verticals_historical import bs_put_price
    p_lower = bs_put_price(spx, fly_lower, T, sigma)
    p_mid = bs_put_price(spx, fly_mid, T, sigma)
    p_upper = bs_put_price(spx, fly_upper, T, sigma)
    fly_debit = p_lower - 2 * p_mid + p_upper
    fly_slippage = 4 * 0.10 / 2.0
    fly_cost = fly_debit + fly_slippage

    net_credit = funder_credit_adj - fly_cost
    if net_credit < -1.0:  # Allow small debit
        logger.warning("[%s] Net debit too high: %.2f", layer.name, net_credit)
        return None

    logger.info(
        "[%s] Funded butterfly | SPX=%.0f DTE=%d | "
        "Funder: Call %.0f/%.0f credit=%.2f | "
        "Fly: %.0f/%.0f/%.0f cost=%.2f | net=%.2f × %d cts",
        layer.name, spx, cal_dte,
        k_cs, k_cl, funder_credit_adj,
        fly_lower, fly_mid, fly_upper, fly_cost,
        net_credit, layer.n_contracts,
    )

    return OpenPosition(
        layer_name=layer.name,
        entry_date=today,
        expiration=expiration,
        k_put_short=0,
        k_put_long=0,
        k_call_short=k_cs,
        k_call_long=k_cl,
        credit=funder_credit_adj,
        n_contracts=layer.n_contracts,
        fly_lower=fly_lower,
        fly_mid=fly_mid,
        fly_upper=fly_upper,
        fly_cost=fly_cost,
    )


# ---------------------------------------------------------------------------
# Exit logic
# ---------------------------------------------------------------------------
def check_exit(pos: OpenPosition, spx: float, vix: float, layer: LayerConfig) -> tuple[bool, str, float]:
    """Check if position should be closed. Returns (should_exit, reason, current_spread_value)."""
    cal_dte = (pos.expiration - date.today()).days
    if cal_dte <= 0:
        return True, "expiry", 0.0

    from bot.pricer import price_iron_condor, price_call_spread
    sigma = vix / 100.0
    T = max(cal_dte, 1) / 365.0

    # Price the funder (call spread or IC)
    if pos.k_put_short > 0:
        spread_val = price_iron_condor(
            spx, vix, cal_dte,
            pos.k_put_short, pos.k_put_long,
            pos.k_call_short, pos.k_call_long,
        )
    else:
        spread_val = price_call_spread(
            spx, sigma, T,
            pos.k_call_short, pos.k_call_long,
        )

    close_threshold = (1.0 - layer.profit_target_pct) * pos.credit
    stop_threshold = layer.stop_mult * pos.credit

    if spread_val <= close_threshold:
        return True, "profit_target", spread_val
    if spread_val >= stop_threshold:
        return True, "stop_loss", spread_val

    return False, "", spread_val


# ---------------------------------------------------------------------------
# Order execution
# ---------------------------------------------------------------------------
def place_order(pos: OpenPosition, mode: str) -> bool:
    """Place the order via IBKR (or log in dry-run mode)."""
    if mode == "dry-run":
        logger.info("[DRY-RUN] Would place: %s", _format_position(pos))
        return True

    try:
        from bot.order_manager import OrderManager
        om = OrderManager(mode=mode, client_id=IBKR_CLIENT_ID)

        # Build a mock entry order compatible with the order manager
        from bot.entry_engine import EntryOrder
        entry = EntryOrder(
            config_id=pos.layer_name,
            expiration=pos.expiration,
            k_put_short=pos.k_put_short,
            k_put_long=pos.k_put_long,
            k_call_short=pos.k_call_short,
            k_call_long=pos.k_call_long,
            credit=pos.credit,
            n_contracts=pos.n_contracts,
            spx_at_decision=0,
            vix_at_decision=0,
        )

        result = om.place_iron_condor(entry)
        if result.success:
            logger.info("[%s] Order filled at %.4f", pos.layer_name, result.fill_price)
            # Send notification
            try:
                from bot import notifier
                notifier.trade_opened(
                    pos.layer_name, str(pos.expiration),
                    pos.k_put_short, pos.k_put_long,
                    pos.k_call_short, pos.k_call_long,
                    pos.credit, pos.n_contracts, result.fill_price,
                )
            except Exception:
                pass
            return True
        else:
            logger.error("[%s] Order failed: %s", pos.layer_name, result.message)
            return False
    except Exception as e:
        logger.error("[%s] Order execution error: %s", pos.layer_name, e)
        return False


def _format_position(pos: OpenPosition) -> str:
    parts = [f"{pos.layer_name} exp={pos.expiration}"]
    if pos.k_put_short > 0:
        parts.append(f"put {pos.k_put_short:.0f}/{pos.k_put_long:.0f}")
    if pos.k_call_short > 0:
        parts.append(f"call {pos.k_call_short:.0f}/{pos.k_call_long:.0f}")
    parts.append(f"credit={pos.credit:.2f} × {pos.n_contracts}cts")
    if pos.fly_mid > 0:
        parts.append(f"fly {pos.fly_lower:.0f}/{pos.fly_mid:.0f}/{pos.fly_upper:.0f}")
    return " | ".join(parts)


# ---------------------------------------------------------------------------
# State persistence
# ---------------------------------------------------------------------------
def save_position(pos: OpenPosition):
    """Save position to Supabase for tracking."""
    try:
        from bot.state_manager import StateManager
        sm = StateManager()
        sm.insert_position({
            "config_id": pos.layer_name,
            "entry_date": pos.entry_date.isoformat(),
            "expiration": pos.expiration.isoformat(),
            "k_put_short": pos.k_put_short,
            "k_put_long": pos.k_put_long,
            "k_call_short": pos.k_call_short,
            "k_call_long": pos.k_call_long,
            "credit_collected": pos.credit,
            "n_contracts": pos.n_contracts,
            "status": "open",
        })
    except Exception as e:
        logger.warning("Failed to save position to Supabase: %s", e)


# ---------------------------------------------------------------------------
# Main runner
# ---------------------------------------------------------------------------
def run_once(mode: str = "dry-run"):
    """Single pass: check entries, check exits, place orders."""
    today = date.today()
    logger.info("=== Daily Cash Flow Bot | mode=%s | %s ===", mode, today)

    # Get market data
    try:
        spx, vix = get_snapshot()
    except Exception as e:
        logger.error("Failed to get market data: %s", e)
        return

    logger.info("SPX=%.1f  VIX=%.1f", spx, vix)

    # Check entries for each layer
    for layer in LAYERS:
        # Skip butterfly if not the right day (every other Friday)
        if layer.is_butterfly:
            # Only enter on Fridays, every 2 weeks
            if today.weekday() != 4:  # Not Friday
                logger.info("[%s] Skipping — butterfly only enters on Fridays", layer.name)
                continue

        # Build entry
        if layer.is_butterfly:
            pos = build_fly_entry(layer, spx, vix, today)
        else:
            pos = build_ic_entry(layer, spx, vix, today)

        if pos is None:
            logger.info("[%s] No valid entry today", layer.name)
            continue

        # Place order
        success = place_order(pos, mode)
        if success and mode != "dry-run":
            save_position(pos)

    logger.info("=== Daily Cash Flow Bot complete ===")


def run_scheduled(mode: str = "paper"):
    """Run on APScheduler with market-hours checks."""
    try:
        from apscheduler.schedulers.blocking import BlockingScheduler
        from apscheduler.triggers.cron import CronTrigger
    except ImportError:
        logger.error("APScheduler not installed. Run: pip install apscheduler")
        return

    scheduler = BlockingScheduler(timezone="US/Eastern")

    # Entry check at 9:35 AM ET
    scheduler.add_job(
        lambda: run_once(mode),
        CronTrigger(hour=9, minute=35, day_of_week="mon-fri"),
        id="daily_entry",
        name="Daily Cash Flow Entry",
    )

    # Exit checks every 15 min during market hours
    def exit_check():
        try:
            spx, vix = get_snapshot()
            logger.info("Exit check | SPX=%.1f VIX=%.1f", spx, vix)
            # TODO: Load open positions from Supabase, check exits, close if needed
        except Exception as e:
            logger.warning("Exit check failed: %s", e)

    scheduler.add_job(
        exit_check,
        CronTrigger(hour="9-15", minute="0,15,30,45", day_of_week="mon-fri"),
        id="exit_check",
        name="Daily Cash Flow Exit Check",
    )

    logger.info("Scheduler started. mode=%s. Press Ctrl+C to stop.", mode)
    try:
        scheduler.start()
    except (KeyboardInterrupt, SystemExit):
        logger.info("Scheduler stopped.")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Daily Cash Flow Bot")
    parser.add_argument("--mode", choices=["dry-run", "paper", "live"], default="dry-run")
    parser.add_argument("--once", action="store_true", help="Single pass then exit")
    args = parser.parse_args()

    # Safety lock: live mode requires explicit env var
    if args.mode == "live":
        import os
        if os.environ.get("ENABLE_LIVE_TRADING") != "YES":
            logger.error(
                "LIVE TRADING BLOCKED. Set ENABLE_LIVE_TRADING=YES to enable. "
                "This is a safety lock to prevent accidental live execution."
            )
            sys.exit(1)
        logger.warning("!!! LIVE TRADING MODE — REAL MONEY AT RISK !!!")

    if args.once:
        run_once(args.mode)
    else:
        run_scheduled(args.mode)


if __name__ == "__main__":
    main()
