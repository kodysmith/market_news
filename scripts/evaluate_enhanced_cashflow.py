"""Shared strategy dataclasses and utilities used by both the backtester
and the live bot (bot/config.py, bot/market_data.py, bot/entry_engine.py,
bot/exit_engine.py).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from typing import Optional


# ---------------------------------------------------------------------------
# Strategy building blocks
# ---------------------------------------------------------------------------

@dataclass
class EntryFilters:
    """Conditions that must be satisfied before entering a new position."""
    vix_min: Optional[float] = None          # Skip if VIX is below this
    vix_max: Optional[float] = None          # Skip if VIX is above this
    trend_sma_period: Optional[int] = None   # Skip if SPX <= SMA(n); None = no check
    rv_expansion_max: Optional[float] = None # Skip if rv_ratio > this (VD2 expansion filter)
    gap_down_no_entry: Optional[float] = None  # Skip if gap_pct <= this (negative = gap-down)
    skip_news_days: bool = False             # Skip on FOMC / CPI / NFP dates
    max_concurrent: Optional[int] = None    # Per-config concurrent position cap


@dataclass
class ExitRules:
    """Conditions that trigger closing an existing position."""
    profit_target_pct: float = 0.50         # Close when spread decays to (1 - pct) × credit
    stop_mult: Optional[float] = 2.0        # Close when spread reaches mult × credit
    time_stop_dte: Optional[int] = None     # Close when DTE falls to this value
    spx_drop_from_entry_pct: Optional[float] = None  # Close on SPX drop of this % from entry
    vix_spike_exit_pct: Optional[float] = None       # Close if VIX rises this % from entry


@dataclass
class StrategyConfig:
    """Complete specification of one iron condor strategy variant."""
    config_id: str
    strategy: str               # e.g. 'iron_condor'
    dte: int                    # Target calendar days to expiration at entry
    short_delta: float          # Absolute delta of the short strikes (e.g. 0.15)
    width: float                # Spread width in SPX points (e.g. 50)
    exit_rules: ExitRules = field(default_factory=ExitRules)
    filters: EntryFilters = field(default_factory=EntryFilters)


# ---------------------------------------------------------------------------
# News / macro calendar
# ---------------------------------------------------------------------------

# Known high-impact macro dates (FOMC, CPI, NFP) for 2025-2026.
# Add to this list as new dates are confirmed.
_MACRO_DATES: set[date] = {
    # 2025 FOMC
    date(2025, 1, 29), date(2025, 3, 19), date(2025, 5, 7),
    date(2025, 6, 18), date(2025, 7, 30), date(2025, 9, 17),
    date(2025, 10, 29), date(2025, 12, 10),
    # 2025 CPI release dates (approx)
    date(2025, 1, 15), date(2025, 2, 12), date(2025, 3, 12),
    date(2025, 4, 10), date(2025, 5, 13), date(2025, 6, 11),
    date(2025, 7, 11), date(2025, 8, 13), date(2025, 9, 10),
    date(2025, 10, 9), date(2025, 11, 13), date(2025, 12, 10),
    # 2026 FOMC
    date(2026, 1, 28), date(2026, 3, 18), date(2026, 4, 29),
    date(2026, 6, 17), date(2026, 7, 29), date(2026, 9, 16),
    date(2026, 10, 28), date(2026, 12, 9),
    # 2026 CPI release dates (approx)
    date(2026, 1, 14), date(2026, 2, 11), date(2026, 3, 11),
    date(2026, 4, 9),  date(2026, 5, 13), date(2026, 6, 10),
    date(2026, 7, 10), date(2026, 8, 12), date(2026, 9, 9),
    date(2026, 10, 14), date(2026, 11, 12), date(2026, 12, 9),
}


def get_news_dates(start: date, end: date) -> set[date]:
    """Return the set of high-impact macro dates between start and end (inclusive)."""
    return {d for d in _MACRO_DATES if start <= d <= end}
