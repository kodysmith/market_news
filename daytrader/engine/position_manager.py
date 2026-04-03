"""Layer 4: Position Manager — manage stops, scale-out, and trailing.

Tracks open positions and processes each 1-min bar to determine exits.
Implements the 1/3-at-1R, 1/3-at-2R, trail-1/3 scale-out schedule.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional

import pandas as pd

from daytrader.config import (
    EXIT_SCALE_1_R, EXIT_SCALE_2_R, EXIT_TRAIL_BUFFER,
    EXIT_TIME_STOP_MIN, COMMISSION_PER_SHARE, SLIPPAGE_PER_SHARE,
)

logger = logging.getLogger("daytrader.positions")


@dataclass
class ScaleLevel:
    """One scale-out level."""
    r_target: float
    shares: int
    filled: bool = False
    fill_price: float = 0.0
    fill_time: str = ""


@dataclass
class Position:
    """An open day trading position."""
    symbol: str
    direction: str          # "long"
    entry_price: float
    stop_price: float
    initial_shares: int
    remaining_shares: int
    risk_per_share: float
    entry_time: str
    pattern: str
    catalyst_score: int = 0
    vwap_at_entry: float = 0.0
    atr_at_entry: float = 0.0

    # Scale-out tracking
    scale_levels: list[ScaleLevel] = field(default_factory=list)
    trailing_stop: float = 0.0
    breakeven_set: bool = False

    # Realized P&L from closed portions
    realized_pnl: float = 0.0
    closed_shares: int = 0

    # Status
    is_open: bool = True
    exit_reason: str = ""
    exit_time: str = ""
    final_pnl: float = 0.0

    def __post_init__(self):
        if not self.scale_levels:
            third = self.initial_shares // 3
            remainder = self.initial_shares - third * 2
            self.scale_levels = [
                ScaleLevel(r_target=EXIT_SCALE_1_R, shares=third),
                ScaleLevel(r_target=EXIT_SCALE_2_R, shares=third),
                ScaleLevel(r_target=0.0, shares=remainder),  # runner (trail)
            ]
            self.trailing_stop = self.stop_price

    @property
    def current_r(self) -> float:
        """Current R-multiple based on last known price (not tracked here)."""
        return 0.0

    def target_price(self, r_mult: float) -> float:
        """Price at a given R-multiple."""
        if self.direction == "long":
            return self.entry_price + r_mult * self.risk_per_share
        else:
            return self.entry_price - r_mult * self.risk_per_share


class PositionManager:
    """Manages all open positions and processes bar updates."""

    def __init__(self):
        self.positions: dict[str, Position] = {}  # symbol → Position
        self.closed_trades: list[dict] = []

    @property
    def open_count(self) -> int:
        return len(self.positions)

    def open_position(self, signal) -> Optional[Position]:
        """Open a new position from an entry signal."""
        if signal.symbol in self.positions:
            logger.warning("Already have position in %s, skipping", signal.symbol)
            return None

        pos = Position(
            symbol=signal.symbol,
            direction=signal.direction,
            entry_price=signal.entry_price,
            stop_price=signal.stop_price,
            initial_shares=signal.shares,
            remaining_shares=signal.shares,
            risk_per_share=signal.risk_per_share,
            entry_time=signal.timestamp,
            pattern=signal.pattern,
            catalyst_score=getattr(signal, "catalyst_score", 0),
            vwap_at_entry=signal.vwap,
            atr_at_entry=signal.atr,
        )
        self.positions[signal.symbol] = pos
        logger.info("OPENED %s %s %d shares at $%.2f (stop $%.2f)",
                    pos.direction.upper(), pos.symbol, pos.initial_shares,
                    pos.entry_price, pos.stop_price)
        return pos

    def process_bar(self, symbol: str, bar: pd.Series, bar_time: str) -> list[dict]:
        """
        Process a 1-min bar for a position. Returns list of fill events.

        Each fill event: {"action": "sell", "shares": N, "price": P, "reason": str}
        """
        if symbol not in self.positions:
            return []

        pos = self.positions[symbol]
        if not pos.is_open:
            return []

        fills = []
        bar_high = float(bar["High"])
        bar_low = float(bar["Low"])
        bar_close = float(bar["Close"])

        # 1. Check stop loss (always first — safety)
        if pos.direction == "long" and bar_low <= pos.stop_price:
            fills.append(self._close_all(pos, pos.stop_price, "stop_loss", bar_time))
            return fills

        # 2. Check trailing stop (for runner portion)
        if pos.breakeven_set and pos.direction == "long" and bar_low <= pos.trailing_stop:
            # Only close the remaining shares (runner)
            runner_shares = pos.remaining_shares
            if runner_shares > 0:
                fills.append(self._close_partial(
                    pos, runner_shares, pos.trailing_stop, "trail_stop", bar_time
                ))
                if pos.remaining_shares <= 0:
                    self._finalize(pos, bar_time)
            return fills

        # 3. Check scale-out levels
        for level in pos.scale_levels:
            if level.filled or level.shares <= 0:
                continue
            if level.r_target <= 0:
                continue  # runner — handled by trailing stop

            target = pos.target_price(level.r_target)
            if pos.direction == "long" and bar_high >= target:
                fills.append(self._close_partial(
                    pos, level.shares, target, f"scale_{level.r_target}R", bar_time
                ))
                level.filled = True
                level.fill_price = target
                level.fill_time = bar_time

                # After 1R hit: move stop to breakeven
                if level.r_target == EXIT_SCALE_1_R and not pos.breakeven_set:
                    pos.stop_price = pos.entry_price
                    pos.trailing_stop = pos.entry_price
                    pos.breakeven_set = True
                    logger.info("  %s: stop moved to BREAKEVEN at $%.2f",
                               symbol, pos.entry_price)

        # 4. Update trailing stop (ratchet up on higher lows)
        if pos.breakeven_set and pos.direction == "long":
            new_trail = bar_low - EXIT_TRAIL_BUFFER
            if new_trail > pos.trailing_stop:
                pos.trailing_stop = new_trail

        # 5. Check time stop (no 1R within N minutes)
        if not pos.breakeven_set:
            entry_dt = pd.to_datetime(pos.entry_time)
            current_dt = pd.to_datetime(bar_time)
            try:
                elapsed = (current_dt - entry_dt).total_seconds() / 60
                if elapsed >= EXIT_TIME_STOP_MIN:
                    fills.append(self._close_all(pos, bar_close, "time_stop", bar_time))
                    return fills
            except Exception:
                pass

        # Check if fully closed
        if pos.remaining_shares <= 0:
            self._finalize(pos, bar_time)

        return fills

    def force_close_all(self, prices: dict[str, float], reason: str, bar_time: str):
        """Force close all positions (EOD or kill switch)."""
        for symbol in list(self.positions.keys()):
            pos = self.positions[symbol]
            if pos.is_open and pos.remaining_shares > 0:
                price = prices.get(symbol, pos.entry_price)
                self._close_all(pos, price, reason, bar_time)

    def _close_partial(self, pos: Position, shares: int, price: float,
                       reason: str, bar_time: str) -> dict:
        """Close a portion of the position."""
        shares = min(shares, pos.remaining_shares)
        if pos.direction == "long":
            gross = (price - pos.entry_price) * shares
        else:
            gross = (pos.entry_price - price) * shares

        commission = shares * COMMISSION_PER_SHARE * 2
        slippage = shares * SLIPPAGE_PER_SHARE
        net = gross - commission - slippage

        pos.realized_pnl += net
        pos.closed_shares += shares
        pos.remaining_shares -= shares

        logger.info("  %s: SELL %d shares at $%.2f (%s) P&L: $%.0f",
                    pos.symbol, shares, price, reason, net)

        return {
            "action": "sell",
            "symbol": pos.symbol,
            "shares": shares,
            "price": price,
            "reason": reason,
            "time": bar_time,
            "pnl": round(net, 2),
        }

    def _close_all(self, pos: Position, price: float,
                   reason: str, bar_time: str) -> dict:
        """Close entire remaining position."""
        result = self._close_partial(pos, pos.remaining_shares, price, reason, bar_time)
        self._finalize(pos, bar_time)
        return result

    def _finalize(self, pos: Position, bar_time: str):
        """Mark position as fully closed and archive it."""
        pos.is_open = False
        pos.exit_time = bar_time
        pos.final_pnl = pos.realized_pnl

        r_mult = pos.final_pnl / (pos.risk_per_share * pos.initial_shares) \
            if pos.risk_per_share * pos.initial_shares > 0 else 0

        trade_record = {
            "symbol": pos.symbol,
            "direction": pos.direction,
            "pattern": pos.pattern,
            "entry_price": pos.entry_price,
            "stop_price": pos.stop_price,
            "initial_shares": pos.initial_shares,
            "entry_time": pos.entry_time,
            "exit_time": pos.exit_time,
            "exit_reason": pos.exit_reason or "scaled_out",
            "gross_pnl": round(pos.realized_pnl + pos.initial_shares * COMMISSION_PER_SHARE * 2, 2),
            "net_pnl": round(pos.final_pnl, 2),
            "r_multiple": round(r_mult, 2),
            "catalyst_score": pos.catalyst_score,
            "vwap_at_entry": pos.vwap_at_entry,
            "atr_at_entry": pos.atr_at_entry,
        }
        self.closed_trades.append(trade_record)

        logger.info("CLOSED %s %s: P&L $%.0f (%.2fR) via %s",
                    pos.direction.upper(), pos.symbol,
                    pos.final_pnl, r_mult, pos.exit_reason or "scaled_out")

        # Remove from active positions
        if pos.symbol in self.positions:
            del self.positions[pos.symbol]

    def get_daily_pnl(self) -> float:
        """Total realized P&L for today."""
        return sum(t["net_pnl"] for t in self.closed_trades)

    def get_trade_count(self) -> int:
        """Number of completed trades today."""
        return len(self.closed_trades)
