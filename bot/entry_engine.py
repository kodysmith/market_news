"""Entry decision engine.

Determines whether today is a valid entry day for each strategy config,
computes strikes and credit, and returns an EntryOrder to be executed.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import date, timedelta
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass
class EntryOrder:
    """Fully specified trade to be sent to order_manager."""
    config_id: str
    expiration: date
    k_put_short: float
    k_put_long: float
    k_call_short: float
    k_call_long: float
    credit: float              # model BS credit per contract (dollars/unit × 100)
    n_contracts: int
    spx_at_decision: float
    vix_at_decision: float


def should_enter(
    config,                    # StrategyConfig
    snapshot,                  # MarketSnapshot
    open_positions: list,      # list[Position]
    mode: str = "dry-run",
) -> tuple[bool, str]:
    """Return (True, reason) if we should enter this config today, else (False, reason).

    Mirrors the filter logic in run_filtered_backtest / simulate_enhanced.
    """
    from bot.config import VIX_ABSOLUTE_CAP, MAX_OPEN_IC_TOTAL

    # Hard safety: VIX absolute cap
    if snapshot.vix_level >= VIX_ABSOLUTE_CAP:
        return False, f"VIX {snapshot.vix_level:.1f} >= hard cap {VIX_ABSOLUTE_CAP}"

    # Hard safety: total open IC cap
    if len(open_positions) >= MAX_OPEN_IC_TOTAL:
        return False, f"Open IC cap reached ({len(open_positions)}/{MAX_OPEN_IC_TOTAL})"

    f = config.filters

    # VIX range check
    if f.vix_min is not None and snapshot.vix_level < f.vix_min:
        return False, f"VIX {snapshot.vix_level:.1f} < min {f.vix_min}"
    if f.vix_max is not None and snapshot.vix_level > f.vix_max:
        return False, f"VIX {snapshot.vix_level:.1f} > max {f.vix_max}"

    # Trend filter
    if f.trend_sma_period is not None:
        if getattr(f, 'trend_sma_invert', False):
            # Bearish: enter only when SPX is BELOW SMA
            if snapshot.spx_price > snapshot.sma50:
                return False, f"SPX {snapshot.spx_price:.1f} > SMA50 {snapshot.sma50:.1f} (need below for bearish)"
        else:
            # Bullish: enter only when SPX is strictly > SMA50
            if snapshot.spx_price <= snapshot.sma50:
                return False, f"SPX {snapshot.spx_price:.1f} <= SMA50 {snapshot.sma50:.1f}"

    # RV expansion filter (VD2 only)
    if f.rv_expansion_max is not None and snapshot.rv_ratio > f.rv_expansion_max:
        return False, f"rv_ratio {snapshot.rv_ratio:.3f} > max {f.rv_expansion_max}"

    # Gap down filter
    if f.gap_down_no_entry is not None and snapshot.gap_pct <= f.gap_down_no_entry:
        return False, f"gap {snapshot.gap_pct:.2f}% <= threshold {f.gap_down_no_entry}"

    # News day filter
    if f.skip_news_days and snapshot.is_news_day:
        return False, "news day skip"

    # Concurrent position cap (per config)
    if f.max_concurrent is not None:
        config_positions = [p for p in open_positions if p.config_id == config.config_id]
        if len(config_positions) >= f.max_concurrent:
            return False, (
                f"Max concurrent for {config.config_id}: "
                f"{len(config_positions)}/{f.max_concurrent}"
            )

    return True, "all filters pass"


def _find_target_expiration(dte: int, from_date: date) -> Optional[date]:
    """Find the nearest Friday expiration that is approximately `dte` calendar days away."""
    from bot.market_data import get_friday_expirations

    min_exp = from_date + timedelta(days=dte - 3)
    max_exp = from_date + timedelta(days=dte + 3)
    fridays = get_friday_expirations(from_date, n=12)
    for f in fridays:
        if min_exp <= f <= max_exp:
            return f
    # Fallback: first Friday at least dte days away
    for f in fridays:
        if f >= from_date + timedelta(days=dte - 5):
            return f
    return None


def build_entry(
    config,             # StrategyConfig
    snapshot,           # MarketSnapshot
    n_contracts: int,
    today: Optional[date] = None,
) -> Optional[EntryOrder]:
    """Compute strikes and credit for a new position. Returns None if expiration not found."""
    from bot.config import MIN_ENTRY_DTE
    from bot.pricer import compute_strikes, compute_credit, compute_call_spread_credit

    if today is None:
        today = date.today()

    expiration = _find_target_expiration(config.dte, today)
    if expiration is None:
        logger.error("No valid expiration found for dte=%d from %s", config.dte, today)
        return None

    cal_dte = (expiration - today).days
    if cal_dte < MIN_ENTRY_DTE:
        logger.warning(
            "Expiration %s is only %d days away (min=%d), skipping",
            expiration, cal_dte, MIN_ENTRY_DTE,
        )
        return None

    k_ps, k_pl, k_cs, k_cl = compute_strikes(
        snapshot.spx_price,
        snapshot.vix_level,
        cal_dte,
        config.short_delta,
        config.width,
    )

    if config.strategy == "call_vertical":
        # Bear call spread: only use call side, zero out put strikes
        k_ps, k_pl = 0.0, 0.0
        credit = compute_call_spread_credit(
            snapshot.spx_price, snapshot.vix_level, cal_dte, k_cs, k_cl,
        )
    elif config.strategy == "put_vertical":
        # Bull put spread: only use put side, zero out call strikes
        k_cs, k_cl = 0.0, 0.0
        credit = compute_credit(
            snapshot.spx_price, snapshot.vix_level, cal_dte,
            k_ps, k_pl, k_cs, k_cl,
        )
    else:
        # Iron condor: both sides
        credit = compute_credit(
            snapshot.spx_price, snapshot.vix_level, cal_dte,
            k_ps, k_pl, k_cs, k_cl,
        )

    if credit <= 0:
        logger.warning("[%s] Credit <= 0 (%.4f), skipping", config.config_id, credit)
        return None

    label = config.strategy.replace("_", " ").title()
    logger.info(
        "[%s] %s entry candidate | SPX=%.1f VIX=%.1f DTE=%d | "
        "Put: %.0f/%.0f  Call: %.0f/%.0f | credit=%.4f × %d cts",
        config.config_id, label, snapshot.spx_price, snapshot.vix_level, cal_dte,
        k_ps, k_pl, k_cs, k_cl, credit, n_contracts,
    )

    return EntryOrder(
        config_id=config.config_id,
        expiration=expiration,
        k_put_short=k_ps,
        k_put_long=k_pl,
        k_call_short=k_cs,
        k_call_long=k_cl,
        credit=credit,
        n_contracts=n_contracts,
        spx_at_decision=snapshot.spx_price,
        vix_at_decision=snapshot.vix_level,
    )


def run_entry_checks(
    snapshot,           # MarketSnapshot
    open_positions: list,
    vd2_contracts: int,
    mode: str = "dry-run",
) -> tuple[list[EntryOrder], list[tuple]]:
    """Main entry point called by scheduler at 9:30 AM.

    Returns:
        orders:    EntryOrders to be placed (0, 1, or 2 if both configs qualify)
        decisions: list of (config_id, ok, reason, order_or_None) for publishing
    """
    from bot.config import (
        SWEET_CONFIG, SD1_CONFIG, AG2_CONFIG, VD2_CONFIG,
        BC4_CONFIG, BC2_CONFIG,
        SWEET_CONTRACTS, SD1_CONTRACTS, AG2_CONTRACTS,
        BC4_CONTRACTS, BC2_CONTRACTS,
    )

    orders = []
    decisions = []
    for config, n_contracts in [
        (SWEET_CONFIG, SWEET_CONTRACTS),
        (SD1_CONFIG, SD1_CONTRACTS),
        (AG2_CONFIG, AG2_CONTRACTS),
        (VD2_CONFIG, vd2_contracts),
        (BC4_CONFIG, BC4_CONTRACTS),
        (BC2_CONFIG, BC2_CONTRACTS),
    ]:
        ok, reason = should_enter(config, snapshot, open_positions, mode)
        if ok:
            order = build_entry(config, snapshot, n_contracts)
            if order:
                orders.append(order)
                decisions.append((config.config_id, True, reason, order))
            else:
                decisions.append((config.config_id, False, "no valid expiration found", None))
        else:
            logger.info("[%s] No entry: %s", config.config_id, reason)
            decisions.append((config.config_id, False, reason, None))

    return orders, decisions
