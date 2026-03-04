"""
Options Data Service (V2): Massive adapter for options signals.
Called only for survivors after gap/relvol/liquidity filter to limit cost.
V1: stub returning empty signals.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional


@dataclass
class OptionsSignals:
    """Normalized options signals for one symbol."""
    symbol: str
    call_put_volume_ratio: Optional[float] = None
    oi_change: Optional[float] = None
    iv_change: Optional[float] = None
    unusual_sweeps_count: Optional[int] = None
    options_heat_score: Optional[float] = None


def get_options_signals(symbol: str, config: Dict[str, Any]) -> OptionsSignals:
    """
    V1: Return empty signals.
    V2: Call Massive options snapshot, aggregate to call_put_ratio, IV change, etc.
    """
    return OptionsSignals(symbol=symbol)
