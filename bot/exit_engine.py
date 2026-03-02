"""Exit monitoring engine.

Scans all open positions every 15 minutes during market hours.
Uses BS model for fast exit trigger checks; uses actual IBKR quotes
for the final closing order.
"""
from __future__ import annotations

import logging
import sys
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Optional

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from bot.pricer import price_iron_condor, dte_remaining

logger = logging.getLogger(__name__)


@dataclass
class ExitSignal:
    """Triggered exit decision."""
    reason: str          # profit_target | stop_loss | time_stop | spx_drop | vix_spike | manual
    spread_val: float    # current BS mark of the spread (per unit)
    position_id: str


def _get_config_for_position(position) -> Optional[object]:
    """Return the StrategyConfig matching this position's config_id."""
    from bot.config import ALL_CONFIGS
    for cfg in ALL_CONFIGS:
        if cfg.config_id == position.config_id:
            return cfg
    return None


def check_exit(
    position,           # Position
    snapshot,           # MarketSnapshot
    today: Optional[date] = None,
) -> Optional[ExitSignal]:
    """Evaluate all exit rules for one position. Returns ExitSignal or None (hold).

    Mirrors the exit logic in simulate_enhanced() in evaluate_enhanced_cashflow.py.
    """
    if today is None:
        today = date.today()

    config = _get_config_for_position(position)
    if config is None:
        logger.warning("Unknown config_id: %s", position.config_id)
        return None

    er = config.exit_rules
    dte = dte_remaining(position.expiration, today)

    # Reprice at current market conditions
    spread_val = price_iron_condor(
        snapshot.spx_price,
        snapshot.vix_level,
        dte,
        position.k_put_short,
        position.k_put_long,
        position.k_call_short,
        position.k_call_long,
    )

    credit = position.credit_collected

    # 1. Profit target: spread has decayed to (1 - target_pct) × credit
    if spread_val <= (1 - er.profit_target_pct) * credit:
        return ExitSignal("profit_target", spread_val, position.id)

    # 2. Stop loss: spread exceeded stop_mult × credit
    if er.stop_mult and spread_val >= er.stop_mult * credit:
        return ExitSignal("stop_loss", spread_val, position.id)

    # 3. Time stop: DTE <= threshold (SWEET config)
    if er.time_stop_dte is not None and dte <= er.time_stop_dte:
        return ExitSignal("time_stop", spread_val, position.id)

    # 4. SPX cumulative drop from entry
    if er.spx_drop_from_entry_pct is not None and position.spx_entry > 0:
        drop = (snapshot.spx_price - position.spx_entry) / position.spx_entry * 100
        if drop <= -er.spx_drop_from_entry_pct:
            return ExitSignal("spx_drop", spread_val, position.id)

    # 5. VIX spike from entry
    if er.vix_spike_exit_pct is not None and position.vix_entry > 0:
        vix_change_pct = (snapshot.vix_level - position.vix_entry) / position.vix_entry * 100
        if vix_change_pct >= er.vix_spike_exit_pct:
            return ExitSignal("vix_spike", spread_val, position.id)

    return None  # Hold


def scan_all_exits(
    open_positions: list,
    snapshot,               # MarketSnapshot
    mode: str = "dry-run",
) -> list[ExitSignal]:
    """Scan all open positions and return list of exit signals to act on."""
    signals = []
    for pos in open_positions:
        signal = check_exit(pos, snapshot)
        if signal:
            credit = pos.credit_collected
            pnl_pct = (credit - signal.spread_val) / credit * 100 if credit > 0 else 0
            logger.info(
                "[%s] Exit signal: %s | spread_val=%.4f | credit=%.4f | pnl=%.1f%% | mode=%s",
                pos.config_id, signal.reason, signal.spread_val, credit, pnl_pct, mode,
            )
            signals.append(signal)
        else:
            dte = dte_remaining(pos.expiration)
            spread_val = price_iron_condor(
                snapshot.spx_price,
                snapshot.vix_level,
                dte,
                pos.k_put_short, pos.k_put_long,
                pos.k_call_short, pos.k_call_long,
            )
            pnl_pct = (pos.credit_collected - spread_val) / pos.credit_collected * 100 \
                if pos.credit_collected > 0 else 0
            logger.debug(
                "[%s] Hold | DTE=%d | spread=%.4f | pnl=%.1f%%",
                pos.config_id, dte, spread_val, pnl_pct,
            )
    return signals
