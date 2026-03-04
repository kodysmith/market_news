"""Black-Scholes pricing and delta utilities for SPX vertical spreads.

Used by bot/pricer.py. Risk-free rate is baked in as 0 (negligible for
short-duration SPX options given the credit-spread focus).

Public API:
    bs_put_price(S, K, T, sigma)  -> float
    bs_call_price(S, K, T, sigma) -> float
    strike_for_delta(S, T, target_delta, sigma, is_put) -> float | None
"""
from __future__ import annotations

import math
from typing import Optional

_SQRT_2PI = math.sqrt(2 * math.pi)
_R = 0.0   # risk-free rate (baked in)


def _norm_pdf(x: float) -> float:
    return math.exp(-0.5 * x * x) / _SQRT_2PI


def _norm_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2)))


def _d1(S: float, K: float, T: float, sigma: float) -> float:
    return (math.log(S / K) + (_R + 0.5 * sigma ** 2) * T) / (sigma * math.sqrt(T))


def bs_call_price(S: float, K: float, T: float, sigma: float) -> float:
    """European call price via Black-Scholes (r=0 baked in)."""
    if T <= 0:
        return max(S - K, 0.0)
    d1 = _d1(S, K, T, sigma)
    d2 = d1 - sigma * math.sqrt(T)
    return S * _norm_cdf(d1) - K * math.exp(-_R * T) * _norm_cdf(d2)


def bs_put_price(S: float, K: float, T: float, sigma: float) -> float:
    """European put price via Black-Scholes (r=0 baked in)."""
    if T <= 0:
        return max(K - S, 0.0)
    d1 = _d1(S, K, T, sigma)
    d2 = d1 - sigma * math.sqrt(T)
    return K * math.exp(-_R * T) * _norm_cdf(-d2) - S * _norm_cdf(-d1)


def _call_delta(S: float, K: float, T: float, sigma: float) -> float:
    if T <= 0:
        return 1.0 if S > K else 0.0
    return _norm_cdf(_d1(S, K, T, sigma))


def _put_delta(S: float, K: float, T: float, sigma: float) -> float:
    """Absolute value of put delta (positive number, 0–1)."""
    return _call_delta(S, K, T, sigma) - 1.0   # negative; caller takes abs if needed


def friday_expirations(start: date, end: date) -> list[date]:
    """Return all Fridays between start and end (inclusive)."""
    from datetime import timedelta
    # Find first Friday on or after start
    days_ahead = (4 - start.weekday()) % 7   # 4 = Friday
    current = start + timedelta(days=days_ahead)
    result = []
    while current <= end:
        result.append(current)
        current += timedelta(days=7)
    return result


def strike_for_delta(
    S: float,
    T: float,
    target_delta: float,
    sigma: float,
    is_put: bool,
) -> Optional[float]:
    """Find the strike K such that |delta| == target_delta using bisection.

    target_delta is a positive fraction (e.g. 0.15 for 15-delta).
    Returns None if no solution is found within the search range.
    """
    if T <= 0 or sigma <= 0:
        return None

    # For a put: |delta_put| = 1 - N(d1). Target strike is OTM, so K < S.
    # For a call: delta_call = N(d1). Target strike is OTM, so K > S.
    if is_put:
        lo, hi = S * 0.5, S * 1.0   # put strikes below spot
        def f(K):
            return abs(_put_delta(S, K, T, sigma)) - target_delta
    else:
        lo, hi = S * 1.0, S * 1.5   # call strikes above spot
        def f(K):
            return _call_delta(S, K, T, sigma) - target_delta

    f_lo, f_hi = f(lo), f(hi)
    if f_lo * f_hi > 0:
        # Widen search range
        if is_put:
            lo = S * 0.3
        else:
            hi = S * 2.0
        f_lo, f_hi = f(lo), f(hi)
        if f_lo * f_hi > 0:
            return None

    for _ in range(60):
        mid = (lo + hi) / 2.0
        f_mid = f(mid)
        if abs(f_mid) < 1e-6:
            break
        if f_lo * f_mid < 0:
            hi, f_hi = mid, f_mid
        else:
            lo, f_lo = mid, f_mid

    return (lo + hi) / 2.0
