"""Layer 5: Risk Governor — enforce all risk limits and adaptive sizing.

The risk governor is the final authority on whether a trade can be taken.
It cannot be overridden by any other layer.
"""
from __future__ import annotations

import logging
from datetime import datetime, timedelta
from typing import Optional

from daytrader.config import (
    ACCOUNT_SIZE, RISK_PER_TRADE, MAX_RISK_DOLLARS,
    RISK_MAX_DAILY_LOSS_PCT, RISK_MAX_WEEKLY_LOSS_PCT,
    RISK_MAX_MONTHLY_DD_PCT, RISK_MAX_TRADES_PER_DAY,
    RISK_MAX_CONSECUTIVE_LOSSES, RISK_PAUSE_MINUTES,
    RISK_ADAPTIVE_LOOKBACK, RISK_ADAPTIVE_FULL_THRESHOLD,
    RISK_ADAPTIVE_REDUCE_THRESHOLD, RISK_ADAPTIVE_MINIMAL_THRESHOLD,
    ENTRY_MAX_CONCURRENT,
)

logger = logging.getLogger("daytrader.risk")


class RiskGovernor:
    """Enforces all risk limits. The kill switch for the trading system."""

    def __init__(self, account_size: float = ACCOUNT_SIZE):
        self.account_size = account_size
        self.max_daily_loss = account_size * RISK_MAX_DAILY_LOSS_PCT
        self.max_weekly_loss = account_size * RISK_MAX_WEEKLY_LOSS_PCT
        self.max_monthly_dd = account_size * RISK_MAX_MONTHLY_DD_PCT

        # State
        self.daily_pnl: float = 0.0
        self.weekly_pnl: float = 0.0
        self.monthly_pnl: float = 0.0
        self.monthly_peak: float = 0.0
        self.trades_today: int = 0
        self.consecutive_losses: int = 0
        self.last_loss_time: Optional[datetime] = None
        self.paused_until: Optional[datetime] = None
        self.killed: bool = False  # daily kill switch
        self.recent_results: list[float] = []  # last N trade P&Ls

        # Sector tracking (prevent correlated positions)
        self.open_sectors: set[str] = set()

    def can_trade(self, open_positions: int = 0) -> tuple[bool, str]:
        """
        Check if a new trade is allowed. Returns (allowed, reason).

        This is the FIRST check before any entry signal is evaluated.
        """
        if self.killed:
            return False, "Daily kill switch activated"

        now = datetime.now()

        # Check pause (after consecutive losses)
        if self.paused_until and now < self.paused_until:
            remaining = (self.paused_until - now).total_seconds() / 60
            return False, f"Paused for {remaining:.0f} more minutes after {RISK_MAX_CONSECUTIVE_LOSSES} consecutive losses"

        # Daily loss limit
        if self.daily_pnl <= -self.max_daily_loss:
            self.killed = True
            return False, f"Daily loss limit hit: ${self.daily_pnl:,.0f} (max -${self.max_daily_loss:,.0f})"

        # Weekly loss limit
        if self.weekly_pnl <= -self.max_weekly_loss:
            return False, f"Weekly loss limit hit: ${self.weekly_pnl:,.0f}"

        # Monthly drawdown limit
        monthly_dd = self.monthly_peak - (self.monthly_peak + self.monthly_pnl)
        if monthly_dd >= self.max_monthly_dd:
            return False, f"Monthly drawdown limit: ${monthly_dd:,.0f}"

        # Max trades per day
        if self.trades_today >= RISK_MAX_TRADES_PER_DAY:
            return False, f"Max trades per day reached: {self.trades_today}/{RISK_MAX_TRADES_PER_DAY}"

        # Max concurrent positions
        if open_positions >= ENTRY_MAX_CONCURRENT:
            return False, f"Max concurrent positions: {open_positions}/{ENTRY_MAX_CONCURRENT}"

        return True, "OK"

    def get_risk_budget(self) -> float:
        """
        Get the current risk budget per trade, adjusted for recent performance.

        Returns dollar amount of maximum risk for the next trade.
        """
        base_risk = MAX_RISK_DOLLARS

        # Adaptive sizing based on recent results
        if len(self.recent_results) >= RISK_ADAPTIVE_LOOKBACK:
            recent = self.recent_results[-RISK_ADAPTIVE_LOOKBACK:]
            wins = sum(1 for r in recent if r > 0)
            win_rate = wins / len(recent)

            if win_rate >= RISK_ADAPTIVE_FULL_THRESHOLD:
                multiplier = 1.0  # full size
            elif win_rate >= RISK_ADAPTIVE_REDUCE_THRESHOLD:
                multiplier = 0.75  # reduced
            elif win_rate >= RISK_ADAPTIVE_MINIMAL_THRESHOLD:
                multiplier = 0.50  # half size
            else:
                multiplier = 0.25  # minimal — something is wrong

            adjusted = base_risk * multiplier
            if multiplier < 1.0:
                logger.info("Adaptive sizing: %.0f%% (win rate %.0f%% over last %d trades)",
                           multiplier * 100, win_rate * 100, len(recent))
            return adjusted

        # Learning phase (< 20 trades): half size
        if len(self.recent_results) < RISK_ADAPTIVE_LOOKBACK:
            return base_risk * 0.5

        return base_risk

    def record_trade(self, pnl: float, sector: str = ""):
        """Record a completed trade result."""
        self.daily_pnl += pnl
        self.weekly_pnl += pnl
        self.monthly_pnl += pnl
        self.trades_today += 1
        self.recent_results.append(pnl)

        # Update monthly peak
        if self.monthly_pnl > self.monthly_peak:
            self.monthly_peak = self.monthly_pnl

        # Consecutive losses tracking
        if pnl < 0:
            self.consecutive_losses += 1
            self.last_loss_time = datetime.now()

            if self.consecutive_losses >= RISK_MAX_CONSECUTIVE_LOSSES:
                self.paused_until = datetime.now() + timedelta(minutes=RISK_PAUSE_MINUTES)
                logger.warning("PAUSED: %d consecutive losses. Paused until %s",
                             self.consecutive_losses, self.paused_until.strftime("%H:%M"))
        else:
            self.consecutive_losses = 0

        # Remove sector from tracking
        if sector and sector in self.open_sectors:
            self.open_sectors.discard(sector)

        logger.info("Risk state: daily P&L $%.0f, trades %d, consec losses %d",
                    self.daily_pnl, self.trades_today, self.consecutive_losses)

    def check_sector(self, sector: str) -> bool:
        """Check if a sector is already occupied by an open position."""
        if not sector:
            return True
        if sector in self.open_sectors:
            logger.info("Sector %s already occupied, skipping", sector)
            return False
        return True

    def register_sector(self, sector: str):
        """Register that we have an open position in this sector."""
        if sector:
            self.open_sectors.add(sector)

    def reset_daily(self):
        """Reset daily counters (call at start of each trading day)."""
        self.daily_pnl = 0.0
        self.trades_today = 0
        self.consecutive_losses = 0
        self.paused_until = None
        self.killed = False
        self.open_sectors.clear()
        logger.info("Daily risk counters reset")

    def reset_weekly(self):
        """Reset weekly counters (call at start of each week)."""
        self.weekly_pnl = 0.0

    def reset_monthly(self):
        """Reset monthly counters (call at start of each month)."""
        self.monthly_pnl = 0.0
        self.monthly_peak = 0.0

    def status(self) -> dict:
        """Get current risk status for logging/display."""
        allowed, reason = self.can_trade()
        return {
            "can_trade": allowed,
            "reason": reason,
            "daily_pnl": round(self.daily_pnl, 2),
            "weekly_pnl": round(self.weekly_pnl, 2),
            "trades_today": self.trades_today,
            "consecutive_losses": self.consecutive_losses,
            "risk_budget": round(self.get_risk_budget(), 2),
            "killed": self.killed,
            "paused_until": str(self.paused_until) if self.paused_until else None,
            "recent_win_rate": round(
                sum(1 for r in self.recent_results[-20:] if r > 0) / max(len(self.recent_results[-20:]), 1) * 100, 1
            ) if self.recent_results else 0,
        }
