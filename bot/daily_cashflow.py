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

import numpy as np

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
MAX_TOTAL_SPX_RISK = 150_000  # Max total SPX short exposure (sum of all max losses)
MAX_ACCOUNT_RISK_PCT = 0.15   # Never risk more than 15% of account net liq
MAX_CONCURRENT_SPREADS = 6    # Max number of open spreads at once
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
# IBKR live pricing
# ---------------------------------------------------------------------------
def _get_ibkr_option_quote(exp_str: str, strike: float, right: str) -> tuple[float, float] | None:
    """Get real bid/ask from IBKR for one SPX option. Returns (bid, ask) or None."""
    try:
        import ib_insync as ib
        client = ib.IB()
        client.connect('127.0.0.1', IBKR_PAPER_PORT, clientId=IBKR_CLIENT_ID + 5, timeout=8)
        client.reqMarketDataType(3)

        contract = ib.Option('SPX', exp_str, strike, right, 'CBOE', multiplier='100')
        qualified = client.qualifyContracts(contract)
        if not qualified:
            client.disconnect()
            return None

        ticker = client.reqMktData(qualified[0], '', False, False)
        client.sleep(2)
        bid = float(ticker.bid) if ticker.bid and ticker.bid > 0 else None
        ask = float(ticker.ask) if ticker.ask and ticker.ask > 0 else None
        client.cancelMktData(qualified[0])
        client.disconnect()

        if bid is None or ask is None:
            return None
        return (bid, ask)
    except Exception as e:
        logger.debug("IBKR quote failed for %s %s %s: %s", exp_str, strike, right, e)
        return None


def get_real_spread_credit(exp_str: str, short_strike: float, long_strike: float, right: str) -> float | None:
    """Get real credit for a vertical spread from IBKR bid/ask.

    Credit = short_leg_bid - long_leg_ask (worst-case fill).
    Returns None if quotes unavailable.
    """
    try:
        import ib_insync as ib
        client = ib.IB()
        client.connect('127.0.0.1', IBKR_PAPER_PORT, clientId=IBKR_CLIENT_ID + 5, timeout=8)
        client.reqMarketDataType(3)

        short_c = ib.Option('SPX', exp_str, short_strike, right, 'CBOE', multiplier='100')
        long_c = ib.Option('SPX', exp_str, long_strike, right, 'CBOE', multiplier='100')

        sq = client.qualifyContracts(short_c)
        lq = client.qualifyContracts(long_c)
        if not sq or not lq:
            client.disconnect()
            return None

        st = client.reqMktData(sq[0], '', False, False)
        lt = client.reqMktData(lq[0], '', False, False)
        client.sleep(2)

        short_bid = float(st.bid) if st.bid and st.bid > 0 else None
        long_ask = float(lt.ask) if lt.ask and lt.ask > 0 else None

        client.cancelMktData(sq[0])
        client.cancelMktData(lq[0])
        client.disconnect()

        if short_bid is None or long_ask is None:
            return None

        credit = short_bid - long_ask
        return credit if credit > 0 else None
    except Exception as e:
        logger.debug("IBKR spread quote failed: %s", e)
        return None


def get_real_ic_credit(exp_str: str, k_ps: float, k_pl: float, k_cs: float, k_cl: float) -> float | None:
    """Get real IC credit from IBKR: sum of put spread bid and call spread bid."""
    try:
        import ib_insync as ib
        client = ib.IB()
        client.connect('127.0.0.1', IBKR_PAPER_PORT, clientId=IBKR_CLIENT_ID + 5, timeout=8)
        client.reqMarketDataType(3)

        contracts = [
            (k_ps, 'P', 'short_put'), (k_pl, 'P', 'long_put'),
            (k_cs, 'C', 'short_call'), (k_cl, 'C', 'long_call'),
        ]
        quotes = {}
        for strike, right, label in contracts:
            c = ib.Option('SPX', exp_str, strike, right, 'CBOE', multiplier='100')
            q = client.qualifyContracts(c)
            if not q:
                client.disconnect()
                return None
            t = client.reqMktData(q[0], '', False, False)
            client.sleep(1.5)
            quotes[label] = {
                'bid': float(t.bid) if t.bid and t.bid > 0 else None,
                'ask': float(t.ask) if t.ask and t.ask > 0 else None,
            }
            client.cancelMktData(q[0])

        client.disconnect()

        # Credit = sell at bid, buy at ask (worst case)
        sp_bid = quotes['short_put']['bid']
        lp_ask = quotes['long_put']['ask']
        sc_bid = quotes['short_call']['bid']
        lc_ask = quotes['long_call']['ask']

        if any(v is None for v in [sp_bid, lp_ask, sc_bid, lc_ask]):
            return None

        put_credit = sp_bid - lp_ask
        call_credit = sc_bid - lc_ask
        total = put_credit + call_credit
        return total if total > 0 else None
    except Exception as e:
        logger.debug("IBKR IC quote failed: %s", e)
        return None


# ---------------------------------------------------------------------------
# Entry logic
# ---------------------------------------------------------------------------
def build_ic_entry(layer: LayerConfig, spx: float, vix: float, today: date) -> OpenPosition | None:
    """Build an iron condor entry using REAL IBKR prices."""
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

    # Get REAL credit from IBKR bid/ask
    exp_str = expiration.strftime('%Y%m%d')
    real_credit = get_real_ic_credit(exp_str, k_ps, k_pl, k_cs, k_cl)

    if real_credit is not None:
        credit_adj = real_credit
        logger.info("[%s] IBKR real credit: $%.2f (bid/ask)", layer.name, real_credit)
    else:
        # Fallback to BS model if IBKR unavailable
        from bot.pricer import compute_credit
        credit = compute_credit(spx, vix, cal_dte, k_ps, k_pl, k_cs, k_cl)
        slippage = 8 * 0.10 / 2.0
        credit_adj = credit - slippage
        logger.warning("[%s] IBKR quotes unavailable, using BS estimate: $%.2f", layer.name, credit_adj)

    if credit_adj <= 0:
        logger.warning("[%s] Credit <= 0: %.4f", layer.name, credit_adj)
        return None

    # Compute real max loss and EV
    width = layer.width
    max_loss_ct = width * 100 - credit_adj * 100
    p_win = 1.0 - (credit_adj / width)
    ev_ct = p_win * credit_adj * 100 - (1 - p_win) * max_loss_ct

    logger.info("[%s] credit=$%.2f  max_loss=$%.0f/ct  P(win)=%.1f%%  EV=$%.0f/ct",
                layer.name, credit_adj, max_loss_ct, p_win * 100, ev_ct)

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
# GEX regime check
# ---------------------------------------------------------------------------
def check_gex_regime() -> tuple[str, float]:
    """Check GEX regime via Massive API. Returns (regime, net_gex).

    regime: 'POSITIVE_GAMMA' | 'NEGATIVE_GAMMA' | 'UNKNOWN'
    """
    try:
        import json
        config_path = ROOT / "data" / "config.json"
        if not config_path.exists():
            return "UNKNOWN", 0.0
        config = json.load(open(config_path))
        massive_key = config.get("MASSIVE_API_KEY", "")
        if not massive_key:
            return "UNKNOWN", 0.0

        from QuantEngine.gex_calculator import (
            get_option_chain_snapshot, get_spot_price, compute_cockpit_state,
        )
        alphavantage_key = config.get("ALPHAVANTAGE_API_KEY", "")
        fmp_key = config.get("FMP_API_KEY", "")

        spot = get_spot_price("SPX", massive_key, alphavantage_key, fmp_key)
        if not spot:
            return "UNKNOWN", 0.0

        snap = get_option_chain_snapshot(massive_key, "SPX", days_out=30,
                                        spot_price=spot, strike_range=300)
        if not snap or not snap.get("results"):
            return "UNKNOWN", 0.0

        state = compute_cockpit_state(snap, spot, ticker="SPX")
        if not state:
            return "UNKNOWN", 0.0

        regime = state.get("regime", "UNKNOWN")
        net_gex = state.get("net_gex", 0.0)
        return regime, net_gex
    except Exception as e:
        logger.warning("GEX check failed: %s", e)
        return "UNKNOWN", 0.0


# ---------------------------------------------------------------------------
# Bear call spread builder (for negative GEX days)
# ---------------------------------------------------------------------------

BEAR_CALL_LAYER = LayerConfig(
    name="BC_14DTE",
    dte=14,
    short_delta=0.12,
    width=50.0,
    profit_target_pct=0.50,
    stop_mult=2.0,
    n_contracts=6,
    entry_time="10:15",
)


def build_bear_call_entry(layer: LayerConfig, spx: float, vix: float, today: date) -> OpenPosition | None:
    """Build a bear call spread entry using REAL IBKR prices."""
    expiration = find_expiration(layer.dte, today)
    if expiration is None:
        return None

    cal_dte = (expiration - today).days
    if cal_dte <= 0:
        return None

    try:
        _, _, k_cs, k_cl = compute_strikes(spx, vix, cal_dte, layer.short_delta, layer.width)
    except Exception as e:
        logger.warning("[%s] Strike computation failed: %s", layer.name, e)
        return None

    # Get REAL credit from IBKR bid/ask
    exp_str = expiration.strftime('%Y%m%d')
    real_credit = get_real_spread_credit(exp_str, k_cs, k_cl, 'C')

    if real_credit is not None:
        credit_adj = real_credit
        logger.info("[%s] IBKR real credit: $%.2f (bid/ask)", layer.name, real_credit)
    else:
        # Fallback to BS
        from bot.pricer import compute_call_spread_credit
        credit = compute_call_spread_credit(spx, vix, cal_dte, k_cs, k_cl)
        slippage = 4 * 0.10 / 2.0
        credit_adj = credit - slippage
        logger.warning("[%s] IBKR quotes unavailable, using BS estimate: $%.2f", layer.name, credit_adj)

    if credit_adj <= 0.20:
        logger.warning("[%s] Credit too low: %.2f", layer.name, credit_adj)
        return None

    # Real math
    width = layer.width
    max_loss_ct = width * 100 - credit_adj * 100
    cushion = k_cs - spx
    p_win = 1.0 - (credit_adj / width)
    ev_ct = p_win * credit_adj * 100 - (1 - p_win) * max_loss_ct

    logger.info(
        "[%s] Bear call | SPX=%.0f DTE=%d | Call %.0f/%.0f | "
        "credit=$%.2f  max_loss=$%.0f/ct  cushion=%.0fpts(%.1f%%)  P(win)=%.1f%%  EV=$%.0f/ct × %d cts",
        layer.name, spx, cal_dte, k_cs, k_cl,
        credit_adj, max_loss_ct, cushion, cushion/spx*100, p_win*100, ev_ct, layer.n_contracts,
    )

    # Only enter if positive EV or very high probability
    if ev_ct < 0 and p_win < 0.85:
        logger.info("[%s] Skipping — negative EV ($%.0f) and P(win) %.1f%% < 85%%",
                    layer.name, ev_ct, p_win * 100)
        return None

    return OpenPosition(
        layer_name=layer.name,
        entry_date=today,
        expiration=expiration,
        k_put_short=0,
        k_put_long=0,
        k_call_short=k_cs,
        k_call_long=k_cl,
        credit=credit_adj,
        n_contracts=layer.n_contracts,
    )


# ---------------------------------------------------------------------------
# Trade quality filters
# ---------------------------------------------------------------------------
MAX_RISK_REWARD = 10.0       # Reject trades worse than 10:1 R:R (need >91% wins)
MIN_CREDIT_PER_CT = 50.0     # Minimum $50/contract credit (real money, not pennies)
MIN_CUSHION_PCT = 1.5        # Minimum 1.5% distance to short strike
MAX_CONTRACTS = 6            # Default contract size


@dataclass
class TradeCandidate:
    """A potential trade with real IBKR pricing and exact math."""
    name: str
    strategy: str            # "ic" or "bear_call"
    expiration: date
    dte: int
    short_strike: float
    long_strike: float
    right: str               # C, P, or IC
    width: float
    credit: float            # real bid/ask credit per unit
    credit_per_ct: float     # credit × 100
    max_loss_per_ct: float   # (width - credit) × 100
    risk_reward: float       # max_loss / credit
    cushion: float           # distance to short strike (pts)
    cushion_pct: float
    p_win: float             # market-implied
    ev_per_ct: float         # expected value per contract
    # For ICs:
    put_short: float = 0
    put_long: float = 0
    call_short: float = 0
    call_long: float = 0


def scan_opportunities(spx: float, vix: float, is_negative_gex: bool, today: date) -> list[TradeCandidate]:
    """Scan multiple strike/DTE combos using REAL IBKR prices. Returns ranked list."""
    candidates = []

    # DTE targets
    if is_negative_gex:
        # Bear calls only at various DTEs
        dte_targets = [7, 14, 21]
        # Distances to scan (points above spot)
        distances = [100, 150, 200, 250, 300]
    else:
        # ICs at short DTEs
        dte_targets = [1, 3, 7]
        distances = [100, 125, 150, 175, 200]

    width = 50  # $50 wide spreads

    for target_dte in dte_targets:
        expiration = find_expiration(target_dte, today)
        if expiration is None:
            continue
        exp_str = expiration.strftime('%Y%m%d')
        dte = (expiration - today).days

        if is_negative_gex:
            # Scan bear calls at various distances
            for dist in distances:
                short_k = round((spx + dist) / 5) * 5
                long_k = short_k + width

                real_credit = get_real_spread_credit(exp_str, short_k, long_k, 'C')
                if real_credit is None or real_credit <= 0:
                    continue

                credit_ct = real_credit * 100
                max_loss_ct = width * 100 - credit_ct
                rr = max_loss_ct / credit_ct if credit_ct > 0 else 999
                cushion = short_k - spx
                cushion_pct = cushion / spx * 100
                p_win = 1.0 - (real_credit / width)
                ev = p_win * credit_ct - (1 - p_win) * max_loss_ct

                candidates.append(TradeCandidate(
                    name=f"BC_{dte}DTE_{short_k}",
                    strategy="bear_call", expiration=expiration, dte=dte,
                    short_strike=short_k, long_strike=long_k, right='C',
                    width=width, credit=real_credit,
                    credit_per_ct=credit_ct, max_loss_per_ct=max_loss_ct,
                    risk_reward=rr, cushion=cushion, cushion_pct=cushion_pct,
                    p_win=p_win, ev_per_ct=ev,
                    call_short=short_k, call_long=long_k,
                ))
        else:
            # Scan ICs at various distances
            for dist in distances:
                put_short_k = round((spx - dist) / 5) * 5
                put_long_k = put_short_k - width
                call_short_k = round((spx + dist) / 5) * 5
                call_long_k = call_short_k + width

                real_credit = get_real_ic_credit(exp_str, put_short_k, put_long_k,
                                                 call_short_k, call_long_k)
                if real_credit is None or real_credit <= 0:
                    continue

                credit_ct = real_credit * 100
                max_loss_ct = width * 100 - credit_ct
                rr = max_loss_ct / credit_ct if credit_ct > 0 else 999
                cushion = min(spx - put_short_k, call_short_k - spx)
                cushion_pct = cushion / spx * 100
                p_win = 1.0 - (real_credit / width)
                ev = p_win * credit_ct - (1 - p_win) * max_loss_ct

                candidates.append(TradeCandidate(
                    name=f"IC_{dte}DTE_{put_short_k}_{call_short_k}",
                    strategy="ic", expiration=expiration, dte=dte,
                    short_strike=put_short_k, long_strike=call_short_k, right='IC',
                    width=width, credit=real_credit,
                    credit_per_ct=credit_ct, max_loss_per_ct=max_loss_ct,
                    risk_reward=rr, cushion=cushion, cushion_pct=cushion_pct,
                    p_win=p_win, ev_per_ct=ev,
                    put_short=put_short_k, put_long=put_long_k,
                    call_short=call_short_k, call_long=call_long_k,
                ))

    return candidates


# ---------------------------------------------------------------------------
# Main runner
# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
# Portfolio risk check
# ---------------------------------------------------------------------------
def check_portfolio_risk(mode: str) -> tuple[bool, str, dict]:
    """Check if we can safely open new positions.

    Returns (can_trade, reason, stats).
    """
    stats = {
        "net_liq": 0, "total_spx_risk": 0, "n_spreads": 0,
        "unrealized_pnl": 0, "margin_used": 0, "risk_pct": 0,
    }

    if mode == "dry-run":
        return True, "dry-run", stats

    try:
        import ib_insync as ib_mod
        ib = ib_mod.IB()
        ib.connect("127.0.0.1", 7497 if mode == "paper" else 7496,
                   clientId=IBKR_CLIENT_ID + 2, timeout=10)

        # Account net liquidation (for % risk calculation)
        for item in ib.accountSummary():
            if item.tag == "NetLiquidation":
                stats["net_liq"] = float(item.value)

        # Only look at SPX option positions (our responsibility)
        positions = ib.positions()
        short_legs = []
        long_legs = []
        spx_unrealized = 0.0

        # Get portfolio items for SPX-only unrealized P&L
        portfolio = ib.portfolio()
        for item in portfolio:
            ls = item.contract.localSymbol or ""
            if "SPX" in ls or "SPXW" in ls:
                spx_unrealized += item.unrealizedPNL

        for p in positions:
            ls = p.contract.localSymbol or ""
            if "SPX" not in ls and "SPXW" not in ls:
                continue
            if p.position < 0:
                short_legs.append(p)
            elif p.position > 0:
                long_legs.append(p)

        stats["unrealized_pnl"] = spx_unrealized

        # Match spreads and calculate total max loss
        matched_spreads = 0
        total_risk = 0
        used_longs = set()

        for short_p in short_legs:
            s_strike = short_p.contract.strike
            s_right = short_p.contract.right
            s_expiry = short_p.contract.lastTradeDateOrContractMonth
            n_cts = abs(int(short_p.position))

            for i, long_p in enumerate(long_legs):
                if i in used_longs:
                    continue
                if (long_p.contract.right == s_right and
                    long_p.contract.lastTradeDateOrContractMonth == s_expiry and
                    abs(int(long_p.position)) == n_cts):
                    width = abs(long_p.contract.strike - s_strike)
                    if width > 0 and width <= 100:
                        max_loss = width * 100 * n_cts
                        total_risk += max_loss
                        matched_spreads += 1
                        used_longs.add(i)
                        break

        stats["total_spx_risk"] = total_risk
        stats["n_spreads"] = matched_spreads
        stats["risk_pct"] = total_risk / stats["net_liq"] * 100 if stats["net_liq"] > 0 else 0

        ib.disconnect()

        # Risk checks — all SPX-only
        if total_risk >= MAX_TOTAL_SPX_RISK:
            return False, f"SPX risk ${total_risk:,.0f} >= limit ${MAX_TOTAL_SPX_RISK:,.0f}", stats

        if stats["net_liq"] > 0 and total_risk / stats["net_liq"] > MAX_ACCOUNT_RISK_PCT:
            return False, f"SPX risk {stats['risk_pct']:.1f}% > limit {MAX_ACCOUNT_RISK_PCT*100:.0f}% of account", stats

        if matched_spreads >= MAX_CONCURRENT_SPREADS:
            return False, f"{matched_spreads} SPX spreads open >= limit {MAX_CONCURRENT_SPREADS}", stats

        if spx_unrealized < -MAX_DAILY_LOSS:
            return False, f"SPX unrealized ${spx_unrealized:,.0f} < -${MAX_DAILY_LOSS:,.0f} loss limit", stats

        return True, "OK", stats

    except Exception as e:
        logger.warning("Portfolio risk check failed: %s — allowing trade", e)
        return True, f"check failed: {e}", stats


def run_once(mode: str = "dry-run"):
    """Scan opportunities with real IBKR prices, pick the best, execute."""
    today = date.today()
    logger.info("=== Daily Cash Flow Bot | mode=%s | %s ===", mode, today)

    # Get market data
    try:
        spx, vix = get_snapshot()
    except Exception as e:
        logger.error("Failed to get market data: %s", e)
        return

    logger.info("SPX=%.1f  VIX=%.1f", spx, vix)

    # Check portfolio risk before entering
    can_trade, risk_reason, risk_stats = check_portfolio_risk(mode)
    logger.info(
        "RISK CHECK: %s | acct=$%s | SPX risk=$%s (%s%% of acct) | %s spreads | SPX P&L=$%s",
        risk_reason,
        f"{risk_stats['net_liq']:,.0f}" if risk_stats['net_liq'] else "?",
        f"{risk_stats['total_spx_risk']:,.0f}" if risk_stats['total_spx_risk'] else "0",
        f"{risk_stats['risk_pct']:.1f}" if risk_stats['risk_pct'] else "0",
        risk_stats['n_spreads'],
        f"{risk_stats['unrealized_pnl']:,.0f}" if risk_stats['unrealized_pnl'] else "0",
    )

    if not can_trade:
        logger.warning("PORTFOLIO RISK LIMIT HIT: %s — NO NEW ENTRIES TODAY", risk_reason)
        try:
            from bot import notifier
            notifier.risk_alert(f"No new entries: {risk_reason}", risk_stats.get('risk_pct', 0))
        except Exception:
            pass
        logger.info("=== Daily Cash Flow Bot complete (risk blocked) ===")
        return

    # Run quant signals (GARCH, regime, flow, tail risk)
    quant = None
    try:
        from bot.quant_signals import get_all_quant_signals
        from scripts.optimize_spx_verticals_historical import load_spx_vix
        spx_df, _ = load_spx_vix(date.today() - timedelta(days=400), date.today())
        if not spx_df.empty:
            quant = get_all_quant_signals(spx_df, vix)
            if quant['action'] == 'SKIP':
                logger.warning("QUANT SIGNALS: SKIP (danger=%d/10) — no entries today", quant['danger_score'])
                logger.info("=== Daily Cash Flow Bot complete (quant blocked) ===")
                return
            elif quant['action'] == 'HALF_SIZE':
                logger.info("QUANT SIGNALS: HALF SIZE (danger=%d/10)", quant['danger_score'])
    except Exception as e:
        logger.warning("Quant signals failed (non-fatal, proceeding): %s", e)

    # Check GEX regime
    gex_regime, net_gex = check_gex_regime()
    is_negative_gex = gex_regime == "NEGATIVE_GAMMA"
    logger.info("GEX: %s (%.1fB)", gex_regime, net_gex / 1e9 if net_gex else 0)

    # Regime engine: 4-tier system (BEAR/CAUTION/NEUTRAL/BULL)
    regime_state = None
    try:
        from bot.regime_engine import RegimeState, compute_regime_score, update_regime, get_regime_status
        import json as _json
        from scripts.optimize_spx_verticals_historical import load_spx_vix as _load

        _spx_df, _vix_df = _load(date.today() - timedelta(days=400), date.today())
        if not _spx_df.empty:
            _close = _spx_df['Close']
            _sma50 = float(_close.rolling(50).mean().iloc[-1])
            _sma200 = float(_close.rolling(200).mean().iloc[-1])
            _sma200_prev = float(_close.rolling(200).mean().iloc[-21]) if len(_close) > 221 else _sma200

            import yfinance as _yf
            _vix3m_df = _yf.download('^VIX3M', period='5d', progress=False)
            _vix3m = float(_vix3m_df['Close'].iloc[-1]) if not _vix3m_df.empty else None

            _vix_s = _vix_df['Close']
            _v20h = float(_vix_s.rolling(20).max().iloc[-1])
            _v40h = float(_vix_s.rolling(40).max().iloc[-1])

            # Load or create regime state (persist across runs)
            _state_path = ROOT / "data" / "regime_state.json"
            try:
                if _state_path.exists():
                    _saved = _json.load(open(_state_path))
                    regime_state = RegimeState(
                        tier=_saved.get('tier', 'NEUTRAL'),
                        score=_saved.get('score', 0),
                        days_in_tier=_saved.get('days_in_tier', 0),
                        confirmation_days=_saved.get('confirmation_days', 0),
                        history=_saved.get('history', []),
                    )
                else:
                    regime_state = RegimeState()
            except Exception:
                regime_state = RegimeState()

            score = compute_regime_score(spx, _sma50, _sma200, _sma200_prev, vix,
                                         _vix3m, _v20h, _v40h, not is_negative_gex)
            regime_state = update_regime(regime_state, score)
            status = get_regime_status(regime_state)

            # Persist state
            try:
                _json.dump({
                    'tier': regime_state.tier, 'score': regime_state.score,
                    'days_in_tier': regime_state.days_in_tier,
                    'confirmation_days': regime_state.confirmation_days,
                    'history': regime_state.history[-20:],
                }, open(_state_path, 'w'))
            except Exception:
                pass

            logger.info(
                "REGIME: %s (score=%d, %dd in tier, scale=%.0f%%, trend=%s)",
                regime_state.tier, regime_state.score,
                regime_state.days_in_tier, regime_state.position_scale * 100,
                status.get('trend', '?'),
            )

            if regime_state.park_in_sgov:
                logger.info("REGIME BEAR: Park in SGOV. No trades today. Protecting compounding base.")
                logger.info("=== Daily Cash Flow Bot complete (BEAR → SGOV) ===")
                return

    except Exception as e:
        logger.warning("Regime engine failed (non-fatal): %s", e)

    # Determine position scale from regime
    position_scale = regime_state.position_scale if regime_state else 1.0

    # Scan all opportunities with real IBKR prices
    logger.info("Scanning opportunities (scale=%.0f%%)...", position_scale * 100)
    candidates = scan_opportunities(spx, vix, is_negative_gex, today)

    if not candidates:
        logger.info("No tradeable opportunities found (no IBKR quotes available)")

        # Fallback: use the old BS-based approach for butterfly on Fridays
        if today.weekday() == 4:
            for layer in LAYERS:
                if layer.is_butterfly:
                    pos = build_fly_entry(layer, spx, vix, today)
                    if pos:
                        success = place_order(pos, mode)
                        if success and mode != "dry-run":
                            save_position(pos)
        logger.info("=== Daily Cash Flow Bot complete ===")
        return

    # Filter by quality
    filtered = [c for c in candidates
                if c.risk_reward <= MAX_RISK_REWARD
                and c.credit_per_ct >= MIN_CREDIT_PER_CT
                and c.cushion_pct >= MIN_CUSHION_PCT]

    # Log all scanned opportunities
    logger.info("Found %d candidates, %d pass filters (R:R≤%.0f, credit≥$%.0f, cushion≥%.1f%%)",
                len(candidates), len(filtered), MAX_RISK_REWARD, MIN_CREDIT_PER_CT, MIN_CUSHION_PCT)

    for c in sorted(candidates, key=lambda x: x.ev_per_ct, reverse=True)[:10]:
        passes = "PASS" if c in filtered else "FAIL"
        logger.info("  %s | %s | credit=$%.0f  R:R=%.1f:1  cush=%.1f%%  EV=$%.0f  [%s]",
                    c.name, c.strategy, c.credit_per_ct, c.risk_reward,
                    c.cushion_pct, c.ev_per_ct, passes)

    if not filtered:
        logger.info("No opportunities pass quality filters today. Sitting out.")
        logger.info("=== Daily Cash Flow Bot complete ===")
        return

    # Rank by EV and take the best
    filtered.sort(key=lambda x: x.ev_per_ct, reverse=True)
    best = filtered[0]

    logger.info("BEST: %s | credit=$%.0f/ct | R:R=%.1f:1 | cushion=%.0fpts(%.1f%%) | EV=$%.0f/ct",
                best.name, best.credit_per_ct, best.risk_reward,
                best.cushion, best.cushion_pct, best.ev_per_ct)

    # Build position from the best candidate
    pos = OpenPosition(
        layer_name=best.name,
        entry_date=today,
        expiration=best.expiration,
        k_put_short=best.put_short,
        k_put_long=best.put_long,
        k_call_short=best.call_short,
        k_call_long=best.call_long,
        credit=best.credit,
        n_contracts=MAX_CONTRACTS,
    )

    # Place order
    success = place_order(pos, mode)

    # Record in trade tracker
    if success:
        try:
            from bot.trade_tracker import record_entry
            record_entry(
                strategy=best.name,
                expiration=str(best.expiration),
                short_strike=best.short_strike,
                long_strike=best.long_strike,
                right=best.right,
                n_contracts=MAX_CONTRACTS,
                credit_per_ct=best.credit_per_ct,
                spx=spx,
                vix=vix,
                gex_regime=gex_regime,
                source="paper" if mode == "paper" else mode,
            )
        except Exception as e:
            logger.warning("Trade tracker failed: %s", e)

    if success and mode != "dry-run":
        save_position(pos)

    # Also do butterfly on Fridays
    if today.weekday() == 4:
        for layer in LAYERS:
            if layer.is_butterfly:
                fly_pos = build_fly_entry(layer, spx, vix, today)
                if fly_pos:
                    place_order(fly_pos, mode)

    logger.info("=== Daily Cash Flow Bot complete ===")


# ---------------------------------------------------------------------------
# Exit engine — scan positions, close when targets hit
# ---------------------------------------------------------------------------

# Exit rules
PROFIT_TARGET_PCT = 0.50   # close when 50% of credit captured
STOP_LOSS_MULT = 2.0       # close when spread value = 2x entry credit
TIME_STOP_DTE = 1          # close at 1 DTE to avoid gamma risk


def scan_and_close_positions(spx: float, vix: float, mode: str):
    """Scan IBKR positions for SPX spreads and close when targets hit."""
    try:
        import ib_insync as ib_mod
    except ImportError:
        return

    if mode == "dry-run":
        return

    try:
        ib = ib_mod.IB()
        ib.connect("127.0.0.1", 7497 if mode == "paper" else 7496,
                   clientId=IBKR_CLIENT_ID + 1, timeout=10)  # +1 to avoid conflict with entry client
    except Exception as e:
        logger.warning("Exit scan: IBKR connect failed: %s", e)
        return

    try:
        positions = ib.positions()

        # Find SPX option positions and group into spreads
        spx_opts = []
        for p in positions:
            ls = p.contract.localSymbol or ""
            if "SPX" not in ls:
                continue
            spx_opts.append({
                "local_symbol": ls,
                "conId": p.contract.conId,
                "qty": p.position,
                "avg_cost": p.avgCost,
                "contract": p.contract,
                "strike": p.contract.strike,
                "right": p.contract.right,
                "expiry": p.contract.lastTradeDateOrContractMonth,
            })

        if not spx_opts:
            logger.info("Exit scan: no SPX positions")
            ib.disconnect()
            return

        # Group by expiry to find spreads
        from collections import defaultdict
        by_expiry = defaultdict(list)
        for opt in spx_opts:
            by_expiry[opt["expiry"]].append(opt)

        today = date.today()

        for expiry_str, legs in by_expiry.items():
            # Parse expiry
            try:
                exp_date = datetime.strptime(expiry_str, "%Y%m%d").date()
            except (ValueError, TypeError):
                continue

            dte = (exp_date - today).days

            # Find short legs (qty < 0) and their long counterparts
            short_legs = [l for l in legs if l["qty"] < 0]
            long_legs = [l for l in legs if l["qty"] > 0]

            for short in short_legs:
                # Find matching long leg (same right, nearby strike)
                matching_long = None
                for lng in long_legs:
                    if (lng["right"] == short["right"] and
                        lng["expiry"] == short["expiry"] and
                        abs(lng["qty"]) == abs(short["qty"])):
                        # Long leg should be further OTM
                        if short["right"] == "C" and lng["strike"] > short["strike"]:
                            matching_long = lng
                            break
                        elif short["right"] == "P" and lng["strike"] < short["strike"]:
                            matching_long = lng
                            break

                if not matching_long:
                    continue

                n_contracts = abs(int(short["qty"]))
                width = abs(matching_long["strike"] - short["strike"])

                # Estimate entry credit from avg costs
                # short avg_cost = what we received, long avg_cost = what we paid
                entry_credit_per_unit = (short["avg_cost"] - matching_long["avg_cost"]) / 100.0

                if entry_credit_per_unit <= 0:
                    continue

                # Price current spread value
                sigma = vix / 100.0
                T = max(dte, 0.5) / 365.0

                from scripts.optimize_spx_verticals_historical import bs_call_price, bs_put_price

                if short["right"] == "C":
                    short_val = bs_call_price(spx, short["strike"], T, sigma)
                    long_val = bs_call_price(spx, matching_long["strike"], T, sigma)
                else:
                    short_val = bs_put_price(spx, short["strike"], T, sigma)
                    long_val = bs_put_price(spx, matching_long["strike"], T, sigma)

                current_spread = max(short_val - long_val, 0.0)

                # Check exit conditions
                pnl_pct = (entry_credit_per_unit - current_spread) / entry_credit_per_unit
                close_reason = None

                # Dynamic stop loss: use EVT tail risk if available
                effective_stop = STOP_LOSS_MULT
                try:
                    from bot.quant_signals import get_tail_risk_signal
                    from scripts.optimize_spx_verticals_historical import load_spx_vix
                    _spx_df, _ = load_spx_vix(date.today() - timedelta(days=400), date.today())
                    if not _spx_df.empty:
                        _returns = np.log(_spx_df['Close'] / _spx_df['Close'].shift(1)).dropna()
                        _tail = get_tail_risk_signal(_returns)
                        effective_stop = _tail['dynamic_stop_mult']
                except Exception:
                    pass  # fall back to fixed stop

                if current_spread <= entry_credit_per_unit * (1 - PROFIT_TARGET_PCT):
                    close_reason = "PROFIT_TARGET"
                elif current_spread >= entry_credit_per_unit * effective_stop:
                    close_reason = "STOP_LOSS"
                elif dte <= TIME_STOP_DTE:
                    close_reason = "TIME_STOP"

                spread_label = f"{short['right']} {short['strike']:.0f}/{matching_long['strike']:.0f}"
                logger.info(
                    "  %s exp=%s | credit=%.2f spread=%.2f pnl=%.0f%% dte=%d | %s",
                    spread_label, expiry_str,
                    entry_credit_per_unit, current_spread,
                    pnl_pct * 100, dte,
                    close_reason or "HOLD"
                )

                if close_reason:
                    # Close the spread: buy back short, sell long
                    logger.info("  >>> CLOSING %s: %s (%d contracts)", spread_label, close_reason, n_contracts)

                    for leg_info, action in [(short, "BUY"), (matching_long, "SELL")]:
                        order = ib_mod.MarketOrder(
                            action=action,
                            totalQuantity=n_contracts,
                            tif="DAY",
                        )
                        trade = ib.placeOrder(leg_info["contract"], order)
                        ib.sleep(3)

                        if trade.orderStatus.status == "Filled":
                            logger.info("    %s %s filled at %.2f",
                                       action, leg_info["local_symbol"],
                                       trade.orderStatus.avgFillPrice)
                        else:
                            logger.warning("    %s %s status: %s",
                                          action, leg_info["local_symbol"],
                                          trade.orderStatus.status)
                            ib.sleep(10)  # wait longer for fill

                    # Notify
                    pnl_dollars = (entry_credit_per_unit - current_spread) * 100 * n_contracts
                    try:
                        from bot import notifier
                        notifier.trade_closed(
                            config_id=f"DC_{spread_label}",
                            reason=close_reason,
                            pnl=pnl_dollars,
                            cumulative_pnl=0,
                            n_contracts=n_contracts,
                        )
                    except Exception:
                        pass

                    # Remove matched long from list so we don't double-process
                    long_legs.remove(matching_long)

    except Exception as e:
        logger.error("Exit scan error: %s", e, exc_info=True)
    finally:
        try:
            ib.disconnect()
        except Exception:
            pass


def run_scheduled(mode: str = "paper"):
    """Run on APScheduler with market-hours checks.

    Crash-resilient: all jobs wrapped in try/except so one failure
    doesn't kill the scheduler. Two entry windows for redundancy.
    """
    try:
        from apscheduler.schedulers.blocking import BlockingScheduler
        from apscheduler.triggers.cron import CronTrigger
    except ImportError:
        logger.error("APScheduler not installed. Run: pip install apscheduler")
        return

    scheduler = BlockingScheduler(timezone="US/Eastern")

    _entered_today = {"date": None}  # track if we already entered today

    def safe_entry():
        """Entry with crash protection and dedup."""
        try:
            today = date.today()
            if _entered_today["date"] == today:
                logger.info("Already entered today, skipping duplicate")
                return
            run_once(mode)
            _entered_today["date"] = today
        except Exception as e:
            logger.error("Entry job crashed (non-fatal): %s", e, exc_info=True)

    def safe_exit_check():
        """Exit check with crash protection. Scans IBKR positions and closes
        spreads that hit profit target, stop loss, or time stop."""
        try:
            spx, vix = get_snapshot()
            logger.info("Exit check | SPX=%.1f VIX=%.1f", spx, vix)
            scan_and_close_positions(spx, vix, mode)
        except Exception as e:
            logger.warning("Exit check failed (non-fatal): %s", e)

    # Primary entry at 10:15 AM ET (after open settles, tighter spreads)
    scheduler.add_job(
        safe_entry,
        CronTrigger(hour=10, minute=15, day_of_week="mon-fri"),
        id="daily_entry_primary",
        name="Daily Cash Flow Entry (Primary 10:15)",
        misfire_grace_time=1800,  # allow 30 min late execution
    )

    # Backup entry at 11:00 AM ET (in case primary was missed)
    scheduler.add_job(
        safe_entry,
        CronTrigger(hour=11, minute=0, day_of_week="mon-fri"),
        id="daily_entry_backup",
        name="Daily Cash Flow Entry (Backup 11:00)",
        misfire_grace_time=1800,
    )

    # Exit checks every 15 min during market hours
    scheduler.add_job(
        safe_exit_check,
        CronTrigger(hour="9-15", minute="0,15,30,45", day_of_week="mon-fri"),
        id="exit_check",
        name="Daily Cash Flow Exit Check",
        misfire_grace_time=600,
    )

    logger.info("Scheduler started. mode=%s. Entry at 10:15 AM ET (backup 11:00). Press Ctrl+C to stop.", mode)
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
