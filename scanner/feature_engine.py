"""
Feature Engine: pure functions for gap%, pm rel volume, liquidity.
Deterministic for backtest/audit.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from scanner.market_data import PremarketSnapshot


@dataclass
class ScannerFeatures:
    """Computed features for one symbol."""
    symbol: str
    gap_pct: float
    pm_rel_volume: Optional[float] = None  # None if no baseline or no pm_volume
    prev_day_dollar_vol: Optional[float] = None
    pm_range_pct: Optional[float] = None


def compute_gap_pct(snap: PremarketSnapshot) -> float:
    """Gap % vs prior close: (pm_last - prior_close) / prior_close."""
    if snap.prior_close is None or snap.prior_close <= 0:
        return 0.0
    return (snap.pm_last - snap.prior_close) / snap.prior_close


def compute_pm_rel_volume(
    pm_volume: Optional[float],
    avg_pm_vol_20d: Optional[float],
) -> Optional[float]:
    """Premarket volume / 20d avg premarket volume. None if either missing."""
    if pm_volume is None or avg_pm_vol_20d is None or avg_pm_vol_20d <= 0 or pm_volume <= 0:
        return None
    return pm_volume / avg_pm_vol_20d


def compute_prev_day_dollar_vol(prior_close: float, prev_day_volume: Optional[float]) -> Optional[float]:
    if prior_close is None or prior_close <= 0 or prev_day_volume is None or prev_day_volume <= 0:
        return None
    return prior_close * prev_day_volume


def compute_pm_range_pct(snap: PremarketSnapshot) -> Optional[float]:
    """(pm_high - pm_low) / pm_last. None if high/low missing."""
    if snap.pm_high is None or snap.pm_low is None or snap.pm_last is None or snap.pm_last <= 0:
        return None
    return (snap.pm_high - snap.pm_low) / snap.pm_last


def compute_features(
    snap: PremarketSnapshot,
    avg_pm_vol_20d: Optional[float] = None,
) -> ScannerFeatures:
    """Compute all features for one symbol."""
    gap_pct = compute_gap_pct(snap)
    pm_rel = compute_pm_rel_volume(snap.pm_volume, avg_pm_vol_20d)
    prev_dollar = compute_prev_day_dollar_vol(snap.prior_close, snap.prev_day_volume)
    pm_range = compute_pm_range_pct(snap)
    return ScannerFeatures(
        symbol=snap.symbol,
        gap_pct=gap_pct,
        pm_rel_volume=pm_rel,
        prev_day_dollar_vol=prev_dollar,
        pm_range_pct=pm_range,
    )
