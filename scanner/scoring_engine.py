"""
Scoring Engine: weighted score + reasons generator -> PremarketTradeIdea.
V1: gap, relvol, liquidity only; no options.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from scanner.feature_engine import ScannerFeatures
from scanner.market_data import PremarketSnapshot


@dataclass
class PremarketTradeIdea:
    """Stable contract for scanner output (API + UI)."""
    symbol: str
    asOf: str  # ET timestamp ISO
    # Premarket
    pm_last: float
    pm_change_pct: Optional[float] = None
    pm_gap_pct: float = 0.0
    pm_volume: Optional[float] = None
    pm_rel_volume: Optional[float] = None
    pm_high: Optional[float] = None
    pm_low: Optional[float] = None
    pm_range_pct: Optional[float] = None
    # Liquidity
    prev_day_dollar_vol: Optional[float] = None
    prev_day_atr_pct: Optional[float] = None
    spread_estimate: Optional[float] = None
    shares_outstanding: Optional[float] = None
    market_cap: Optional[float] = None
    # Catalyst (V3)
    news_headlines: List[str] = field(default_factory=list)
    earnings_flag: Optional[bool] = None
    earnings_time: Optional[str] = None
    # Options (V2)
    call_put_volume_ratio: Optional[float] = None
    oi_change: Optional[float] = None
    iv_change: Optional[float] = None
    unusual_sweeps_count: Optional[int] = None
    options_heat_score: Optional[float] = None
    # Scores
    gap_score: float = 0.0
    relvol_score: float = 0.0
    liquidity_score: float = 0.0
    options_score: float = 0.0
    halt_risk_score: float = 0.0
    total_score: float = 0.0
    # Explainability
    reasons: List[str] = field(default_factory=list)
    playbook: Optional[str] = None
    suggested_levels: Optional[Dict[str, Any]] = None

    def to_payload(self) -> Dict[str, Any]:
        """Full payload for DB/API (JSON-serializable)."""
        return {
            "symbol": self.symbol,
            "asOf": self.asOf,
            "pm_last": self.pm_last,
            "pm_change_pct": self.pm_change_pct,
            "pm_gap_pct": self.pm_gap_pct,
            "pm_volume": self.pm_volume,
            "pm_rel_volume": self.pm_rel_volume,
            "pm_high": self.pm_high,
            "pm_low": self.pm_low,
            "pm_range_pct": self.pm_range_pct,
            "prev_day_dollar_vol": self.prev_day_dollar_vol,
            "prev_day_atr_pct": self.prev_day_atr_pct,
            "spread_estimate": self.spread_estimate,
            "shares_outstanding": self.shares_outstanding,
            "market_cap": self.market_cap,
            "news_headlines": self.news_headlines,
            "earnings_flag": self.earnings_flag,
            "earnings_time": self.earnings_time,
            "call_put_volume_ratio": self.call_put_volume_ratio,
            "oi_change": self.oi_change,
            "iv_change": self.iv_change,
            "unusual_sweeps_count": self.unusual_sweeps_count,
            "options_heat_score": self.options_heat_score,
            "gap_score": self.gap_score,
            "relvol_score": self.relvol_score,
            "liquidity_score": self.liquidity_score,
            "options_score": self.options_score,
            "halt_risk_score": self.halt_risk_score,
            "total_score": self.total_score,
            "reasons": self.reasons,
            "playbook": self.playbook,
            "suggested_levels": self.suggested_levels,
        }


def _gap_score(gap_pct: float, cap_at: float = 0.20) -> float:
    """0-40; saturates after cap_at (e.g. 20%)."""
    if gap_pct <= 0:
        return 0.0
    pct = min(gap_pct, cap_at)
    return 40.0 * (pct / cap_at)


def _relvol_score(rel_vol: Optional[float], log_scale: bool = True) -> float:
    """0-30. 5x = strong, 10x = monster."""
    if rel_vol is None or rel_vol <= 0:
        return 0.0
    if log_scale:
        import math
        # log2(5) ~ 2.3, log2(10) ~ 3.3 -> map to 0-30
        x = math.log2(rel_vol + 1)
        return min(30.0, x * 10.0)
    return min(30.0, rel_vol * 3.0)


def _liquidity_score(prev_day_dollar_vol: Optional[float], threshold_10m: float = 10_000_000) -> float:
    """0-15. Higher dollar vol = better."""
    if prev_day_dollar_vol is None or prev_day_dollar_vol <= 0:
        return 0.0
    if prev_day_dollar_vol >= threshold_10m * 5:
        return 15.0
    return 15.0 * min(1.0, prev_day_dollar_vol / threshold_10m)


def _halt_risk_penalty(gap_pct: float, pm_range_pct: Optional[float], tiny_float: bool = False) -> float:
    """0-20 penalty (subtracted). High gap + huge range + tiny float."""
    penalty = 0.0
    if gap_pct >= 0.15 and pm_range_pct is not None and pm_range_pct >= 0.10:
        penalty += 10.0
    if tiny_float:
        penalty += 10.0
    return min(20.0, penalty)


def _generate_reasons(
    snap: PremarketSnapshot,
    feats: ScannerFeatures,
    gap_s: float,
    relvol_s: float,
    liq_s: float,
    halt_penalty: float,
) -> List[str]:
    reasons = []
    if feats.gap_pct >= 0.05:
        reasons.append(f"Gap {feats.gap_pct * 100:.1f}% vs prior close")
    if feats.pm_rel_volume is not None and feats.pm_rel_volume >= 2.0:
        reasons.append(f"Premarket volume {feats.pm_rel_volume:.1f}× normal")
    if feats.prev_day_dollar_vol is not None and feats.prev_day_dollar_vol >= 5_000_000:
        reasons.append(f"High liquidity: ${feats.prev_day_dollar_vol / 1e6:.1f}M dollar volume yesterday")
    if halt_penalty > 0:
        reasons.append("Caution: halt risk elevated")
    return reasons


def score_and_build_idea(
    snap: PremarketSnapshot,
    feats: ScannerFeatures,
    as_of_et: str,
    avg_pm_vol_20d: Optional[float] = None,
) -> PremarketTradeIdea:
    """Compute scores and build PremarketTradeIdea with reasons. V1: no options."""
    gap_s = _gap_score(feats.gap_pct)
    relvol_s = _relvol_score(feats.pm_rel_volume)
    liq_s = _liquidity_score(feats.prev_day_dollar_vol)
    options_s = 0.0  # V1
    halt_penalty = _halt_risk_penalty(feats.gap_pct, feats.pm_range_pct, tiny_float=False)
    total = gap_s + relvol_s + liq_s + options_s - halt_penalty
    total = max(0.0, total)

    pm_change_pct = None
    if snap.prior_close and snap.prior_close > 0:
        pm_change_pct = (snap.pm_last - snap.prior_close) / snap.prior_close * 100.0

    reasons = _generate_reasons(snap, feats, gap_s, relvol_s, liq_s, halt_penalty)
    playbook = "gap-and-go / first pullback" if feats.gap_pct >= 0.05 else None
    suggested_levels = None
    if snap.pm_high is not None or snap.pm_low is not None:
        suggested_levels = {}
        if snap.pm_high is not None:
            suggested_levels["pm_high"] = snap.pm_high
        if snap.pm_low is not None:
            suggested_levels["pm_low"] = snap.pm_low

    return PremarketTradeIdea(
        symbol=snap.symbol,
        asOf=as_of_et,
        pm_last=snap.pm_last,
        pm_change_pct=pm_change_pct,
        pm_gap_pct=feats.gap_pct,
        pm_volume=snap.pm_volume,
        pm_rel_volume=feats.pm_rel_volume,
        pm_high=snap.pm_high,
        pm_low=snap.pm_low,
        pm_range_pct=feats.pm_range_pct,
        prev_day_dollar_vol=feats.prev_day_dollar_vol,
        gap_score=gap_s,
        relvol_score=relvol_s,
        liquidity_score=liq_s,
        options_score=options_s,
        halt_risk_score=halt_penalty,
        total_score=round(total, 4),
        reasons=reasons,
        playbook=playbook,
        suggested_levels=suggested_levels,
    )


def rank_ideas(ideas: List[PremarketTradeIdea], top_n: int = 25) -> List[PremarketTradeIdea]:
    """Sort by total_score desc and assign rank; return top_n."""
    sorted_ideas = sorted(ideas, key=lambda x: x.total_score, reverse=True)
    for i, idea in enumerate(sorted_ideas[:top_n], start=1):
        # Mutate rank into payload later when writing to DB
        pass
    return sorted_ideas[:top_n]
