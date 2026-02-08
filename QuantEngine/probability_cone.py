"""
Probability Cone & HPH/HPL — "From now to close" range and high/low/close probabilities.

v1: Monte Carlo GBM (drift=0), ATM IV → remaining-day sigma, gamma tilt (depin risk).
Outputs: cone percentiles (Panel A), HPH/HPL + P50/P80 (Panel B), close mode + credible intervals (Panel C).

Log inputs/outputs for calibration: "When model said 80% band, did price stay in it ~80% of the time?"
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

# Trading minutes per year: 252 days * 390 min/day
TRADING_MINUTES_PER_YEAR = 252 * 390


@dataclass
class ProbabilityConeInputs:
    """Inputs used for the run (for logging/calibration)."""
    spot: float
    iv_annual: float  # ATM IV as decimal (e.g. 0.20)
    minutes_to_close: int
    gamma_negative: bool
    depin_risk: float  # 0-100
    put_wall: Optional[float] = None
    call_wall: Optional[float] = None


@dataclass
class ConePoint:
    """Percentiles at one future time bucket (minutes from now)."""
    minutes_from_now: float
    p10: float
    p25: float
    p50: float
    p75: float
    p90: float


@dataclass
class ProbabilityConeResult:
    """Full result for Panel A/B/C."""
    inputs: ProbabilityConeInputs
    cone: List[ConePoint]  # Panel A: percentiles per time bucket
    # Panel B
    high_distribution: List[Tuple[float, float]]  # (bucket_center, count) for histogram
    low_distribution: List[Tuple[float, float]]
    hph: float  # highest-probability high (mode of highs)
    hpl: float  # highest-probability low (mode of lows)
    p50_high: float
    p80_high: float
    p50_low: float
    p80_low: float
    # Panel C
    close_distribution: List[Tuple[float, float]]  # (bucket_center, count)
    close_mode: float
    close_p25: float
    close_p75: float
    close_p10: float
    close_p90: float
    sigma_eff: float
    n_paths: int
    step_minutes: int


def _remaining_day_sigma(iv_annual: float, minutes_to_close: int) -> Tuple[float, float]:
    """T (years) and sigma_T (vol for remaining day)."""
    if minutes_to_close <= 0:
        return 0.0, 0.0
    T = minutes_to_close / TRADING_MINUTES_PER_YEAR
    sigma_T = iv_annual * math.sqrt(T)
    return T, sigma_T


def _gamma_tilt(
    sigma_T: float,
    gamma_negative: bool,
    depin_risk: float,
    k1: float = 0.3,
    k2: float = 0.2,
) -> float:
    """Widen cone (fatter tails) for negative gamma; tighten for positive."""
    depin_risk = max(0.0, min(100.0, depin_risk))
    if gamma_negative:
        sigma_eff = sigma_T * (1.0 + k1 * depin_risk / 100.0)
    else:
        sigma_eff = sigma_T * (1.0 - k2 * (1.0 - depin_risk / 100.0))
    return max(sigma_T * 0.5, min(sigma_T * 2.0, sigma_eff))


def _bucket_size(spot: float) -> float:
    """Histogram bucket size: ~$0.50 for stocks, $1 for indices."""
    if spot >= 500:
        return 1.0
    if spot >= 100:
        return 0.5
    return 0.25


def _histogram(values: List[float], bucket_size: float, center_around: float) -> List[Tuple[float, float]]:
    """Return list of (bucket_center, count) sorted by bucket_center."""
    if not values:
        return []
    min_bucket = min(values) // bucket_size * bucket_size
    max_bucket = max(values) // bucket_size * bucket_size + bucket_size
    buckets: Dict[float, int] = {}
    b = min_bucket
    while b <= max_bucket:
        buckets[b] = 0
        b += bucket_size
    for v in values:
        b = round(v / bucket_size) * bucket_size
        buckets[b] = buckets.get(b, 0) + 1
    out = [(c, cnt) for c, cnt in sorted(buckets.items()) if cnt > 0]
    return out


def _mode_from_histogram(hist: List[Tuple[float, float]]) -> float:
    """Bucket center with max count."""
    if not hist:
        return 0.0
    return max(hist, key=lambda x: x[1])[0]


def _percentile(sorted_values: List[float], p: float) -> float:
    """p in [0,100]. Linear interpolation."""
    if not sorted_values:
        return 0.0
    n = len(sorted_values)
    i = (p / 100.0) * (n - 1)
    lo = int(math.floor(i))
    hi = min(lo + 1, n - 1)
    if lo == hi:
        return sorted_values[lo]
    w = i - lo
    return sorted_values[lo] * (1 - w) + sorted_values[hi] * w


def run_monte_carlo(
    spot: float,
    sigma_eff: float,
    minutes_to_close: int,
    n_paths: int = 5000,
    step_minutes: int = 5,
    seed: Optional[int] = None,
) -> Tuple[List[List[Tuple[float, float]]], List[float], List[float], List[float]]:
    """
    GBM (drift=0), lognormal. Timestep = step_minutes.
    Returns:
        path_prices: list of paths; each path = list of (t_min, price)
        closes: list of close prices (one per path)
        max_prices: list of max price from now to close (one per path)
        min_prices: list of min price from now to close (one per path)
    """
    if seed is not None:
        random.seed(seed)
    # sigma per step (in log space): sigma_eff is for full remaining period
    # So per-step vol: sigma_step = sigma_eff * sqrt(step_minutes / minutes_to_close)
    n_steps = max(1, minutes_to_close // step_minutes)
    sigma_per_step = sigma_eff * math.sqrt(1.0 / n_steps) if n_steps else sigma_eff

    closes: List[float] = []
    max_prices: List[float] = []
    min_prices: List[float] = []
    path_prices: List[List[Tuple[float, float]]] = []

    for _ in range(n_paths):
        price = spot
        path: List[Tuple[float, float]] = [(0.0, price)]
        max_p = price
        min_p = price
        for step in range(1, n_steps + 1):
            z = random.gauss(0.0, 1.0)
            # lognormal: price *= exp(-0.5*sigma^2 + sigma*z)
            price = price * math.exp(-0.5 * sigma_per_step * sigma_per_step + sigma_per_step * z)
            t_min = step * step_minutes
            path.append((float(t_min), price))
            max_p = max(max_p, price)
            min_p = min(min_p, price)
        closes.append(price)
        max_prices.append(max_p)
        min_prices.append(min_p)
        path_prices.append(path)

    return path_prices, closes, max_prices, min_prices


def compute_cone_from_paths(
    path_prices: List[List[Tuple[float, float]]],
    time_bucket_minutes: int = 10,
) -> List[ConePoint]:
    """At each time bucket, compute P10/P25/P50/P75/P90 of price across paths."""
    if not path_prices:
        return []
    # Collect prices at each time step (by step index)
    n_steps = len(path_prices[0])
    cone: List[ConePoint] = []
    for step_idx in range(n_steps):
        t_min = path_prices[0][step_idx][0]
        prices = [path_prices[p][step_idx][1] for p in range(len(path_prices))]
        prices.sort()
        cone.append(ConePoint(
            minutes_from_now=t_min,
            p10=_percentile(prices, 10),
            p25=_percentile(prices, 25),
            p50=_percentile(prices, 50),
            p75=_percentile(prices, 75),
            p90=_percentile(prices, 90),
        ))
    return cone


def compute_probability_range(
    spot: float,
    iv_annual: float,
    minutes_to_close: int,
    gamma_negative: bool = True,
    depin_risk: float = 50.0,
    put_wall: Optional[float] = None,
    call_wall: Optional[float] = None,
    n_paths: int = 5000,
    step_minutes: int = 5,
    time_bucket_minutes: int = 10,
    k1: float = 0.3,
    k2: float = 0.2,
    seed: Optional[int] = None,
) -> ProbabilityConeResult:
    """
    Full computation for Panel A/B/C. "From now to close" only.

    Returns cone percentiles, HPH/HPL, close mode and credible intervals.
    """
    inputs = ProbabilityConeInputs(
        spot=spot,
        iv_annual=iv_annual,
        minutes_to_close=max(1, minutes_to_close),
        gamma_negative=gamma_negative,
        depin_risk=depin_risk,
        put_wall=put_wall,
        call_wall=call_wall,
    )
    T, sigma_T = _remaining_day_sigma(iv_annual, inputs.minutes_to_close)
    sigma_eff = _gamma_tilt(sigma_T, gamma_negative, depin_risk, k1=k1, k2=k2)

    path_prices, closes, max_prices, min_prices = run_monte_carlo(
        spot, sigma_eff, inputs.minutes_to_close, n_paths=n_paths, step_minutes=step_minutes, seed=seed
    )
    cone = compute_cone_from_paths(path_prices, time_bucket_minutes)

    bucket = _bucket_size(spot)
    high_hist = _histogram(max_prices, bucket, spot)
    low_hist = _histogram(min_prices, bucket, spot)
    close_hist = _histogram(closes, bucket, spot)

    hph = _mode_from_histogram(high_hist)
    hpl = _mode_from_histogram(low_hist)
    closes_sorted = sorted(closes)
    close_mode = _mode_from_histogram(close_hist)
    p50_high = _percentile(max_prices, 50)
    p80_high = _percentile(max_prices, 80)
    p50_low = _percentile(min_prices, 50)
    p80_low = _percentile(min_prices, 20)  # P20 low = 80% above this

    return ProbabilityConeResult(
        inputs=inputs,
        cone=cone,
        high_distribution=high_hist,
        low_distribution=low_hist,
        hph=hph,
        hpl=hpl,
        p50_high=p50_high,
        p80_high=p80_high,
        p50_low=p50_low,
        p80_low=p80_low,
        close_distribution=close_hist,
        close_mode=close_mode,
        close_p25=_percentile(closes_sorted, 25),
        close_p75=_percentile(closes_sorted, 75),
        close_p10=_percentile(closes_sorted, 10),
        close_p90=_percentile(closes_sorted, 90),
        sigma_eff=sigma_eff,
        n_paths=n_paths,
        step_minutes=step_minutes,
    )


def result_to_api_dict(result: ProbabilityConeResult) -> Dict[str, Any]:
    """Serialize for API response (Panel A/B/C)."""
    return {
        "inputs": {
            "spot": result.inputs.spot,
            "iv_annual": result.inputs.iv_annual,
            "minutes_to_close": result.inputs.minutes_to_close,
            "gamma_negative": result.inputs.gamma_negative,
            "depin_risk": result.inputs.depin_risk,
            "put_wall": result.inputs.put_wall,
            "call_wall": result.inputs.call_wall,
        },
        "sigma_eff": result.sigma_eff,
        "n_paths": result.n_paths,
        "step_minutes": result.step_minutes,
        "cone": [
            {
                "minutes_from_now": p.minutes_from_now,
                "p10": p.p10,
                "p25": p.p25,
                "p50": p.p50,
                "p75": p.p75,
                "p90": p.p90,
            }
            for p in result.cone
        ],
        "high_distribution": [{"bucket": b, "count": c} for b, c in result.high_distribution],
        "low_distribution": [{"bucket": b, "count": c} for b, c in result.low_distribution],
        "hph": result.hph,
        "hpl": result.hpl,
        "p50_high": result.p50_high,
        "p80_high": result.p80_high,
        "p50_low": result.p50_low,
        "p80_low": result.p80_low,
        "close_distribution": [{"bucket": b, "count": c} for b, c in result.close_distribution],
        "close_mode": result.close_mode,
        "close_p25": result.close_p25,
        "close_p75": result.close_p75,
        "close_p10": result.close_p10,
        "close_p90": result.close_p90,
    }
