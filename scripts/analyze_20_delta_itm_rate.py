#!/usr/bin/env python3
"""Estimate how often 20-delta puts expire in the money: empirical ITM rate vs theoretical 20%.

Uses historical SPX close (yfinance). For each expiration, entry = 7 trading days before;
finds strike K where put delta = -0.20 (Black-Scholes), then checks if spot at expiry < K.
Reports empirical ITM %, expected 20%, and a simple confidence interval.
"""
from __future__ import annotations

import argparse
import sys
from datetime import date, datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

R = 0.05


def _d1(S: float, K: float, T: float, sigma: float) -> float:
    if T <= 0 or sigma <= 0 or S <= 0 or K <= 0:
        return float("nan")
    return (np.log(S / K) + (R + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))


def bs_put_delta(S: float, K: float, T: float, sigma: float) -> float:
    d1_val = _d1(S, K, T, sigma)
    if np.isnan(d1_val):
        return float("nan")
    return float(norm.cdf(d1_val) - 1.0)


def strike_for_put_delta(S: float, T: float, target_delta: float, sigma: float) -> float | None:
    """Find strike K such that put delta = target_delta (e.g. -0.20). Binary search."""
    if T <= 0 or sigma <= 0 or S <= 0:
        return None
    # Put delta: K low (deep ITM) -> delta near 0; K high (OTM) -> delta near -1.
    # So higher K -> more negative delta. We want target_delta = -0.20.
    lo, hi = S * 0.80, S * 1.005  # 20-delta put is slightly OTM (K < S)
    for _ in range(80):
        mid = (lo + hi) / 2.0
        d = bs_put_delta(S, mid, T, sigma)
        if np.isnan(d):
            return None
        if abs(d - target_delta) < 1e-5:
            return mid
        if d > target_delta:  # delta too high (e.g. -0.1) -> need more negative -> raise K
            lo = mid
        else:  # d < target_delta -> lower K
            hi = mid
    return (lo + hi) / 2.0


def load_spx(start: date, end: date) -> pd.Series:
    try:
        import yfinance as yf
    except ImportError:
        raise RuntimeError("yfinance required: pip install yfinance")
    t = yf.Ticker("^GSPC")
    hist = t.history(start=start.isoformat(), end=(end + timedelta(days=15)).isoformat())
    if hist.empty:
        return pd.Series(dtype=float)
    hist.index = pd.to_datetime(hist.index).tz_localize(None).normalize()
    close = hist["Close"]
    close.index = close.index.date
    return close


def get_trading_days(start: date, end: date) -> list[date]:
    try:
        from scripts.trading_calendar import get_trading_days_index
        idx = get_trading_days_index(start, end)
        return [pd.Timestamp(ts).date() for ts in idx]
    except Exception:
        out = []
        d = start
        while d <= end:
            if d.weekday() < 5:
                out.append(d)
            d += timedelta(days=1)
        return out


def friday_expirations(start: date, end: date) -> list[date]:
    out = []
    d = start
    while d <= end:
        if d.weekday() == 4:
            out.append(d)
        d += timedelta(days=1)
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description="Empirical 20-delta put ITM rate vs theoretical 20%.")
    parser.add_argument("--start", default="2015-01-01", help="Start date YYYY-MM-DD")
    parser.add_argument("--end", default="2024-12-31", help="End date YYYY-MM-DD")
    parser.add_argument("--delta", type=float, default=0.20, help="Target put delta (absolute, e.g. 0.20)")
    parser.add_argument("--iv", type=float, default=0.18, help="Assumed IV for strike calculation")
    parser.add_argument("--dte", type=int, default=7, help="Trading days to expiration at entry")
    args = parser.parse_args()
    start = datetime.strptime(args.start, "%Y-%m-%d").date()
    end = datetime.strptime(args.end, "%Y-%m-%d").date()

    trading = get_trading_days(start, end + timedelta(days=400))
    expirations = [e for e in friday_expirations(start, end) if e in trading]
    if not expirations:
        print("No expirations in range.", file=sys.stderr)
        return 1

    spot = load_spx(start, end + timedelta(days=30))
    if spot.empty:
        print("No SPX data.", file=sys.stderr)
        return 1

    target_delta = -abs(args.delta)
    results = []
    for exp in expirations:
        try:
            pos = trading.index(exp)
        except ValueError:
            continue
        if pos < args.dte:
            continue
        entry_date = trading[pos - args.dte]
        if entry_date not in spot.index or exp not in spot.index:
            continue
        S_entry = float(spot.loc[entry_date])
        S_exp = float(spot.loc[exp])
        if S_entry <= 0:
            continue
        T = args.dte / 252.0
        K = strike_for_put_delta(S_entry, T, target_delta, args.iv)
        if K is None:
            continue
        itm = S_exp < K
        results.append({
            "entry_date": entry_date,
            "exp_date": exp,
            "spot_entry": S_entry,
            "spot_expiry": S_exp,
            "strike": K,
            "itm": itm,
        })

    if not results:
        print("No valid periods.", file=sys.stderr)
        return 1

    n = len(results)
    itm_count = sum(1 for r in results if r["itm"])
    pct = 100.0 * itm_count / n
    expected_pct = 100.0 * abs(target_delta)  # 20% for 20-delta

    # Wilson score interval for proportion
    z = 1.96
    p = itm_count / n
    denom = 1 + z**2 / n
    center = (p + z**2 / (2 * n)) / denom
    margin = z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / denom
    lo = max(0, 100 * (center - margin))
    hi = min(100, 100 * (center + margin))

    print("=" * 60)
    print("20-DELTA PUT: EMPIRICAL ITM RATE VS THEORETICAL")
    print("=" * 60)
    print(f"Period: {args.start} to {args.end}")
    print(f"Expirations: weekly Fridays, entry = {args.dte} trading days before")
    print(f"Strike: K such that put_delta = {target_delta} (BS, IV = {args.iv})")
    print()
    print(f"  N periods:        {n}")
    print(f"  ITM count:       {itm_count}")
    print(f"  Empirical ITM %: {pct:.1f}%")
    print(f"  Theoretical:     {expected_pct:.0f}% (delta = {abs(target_delta):.2f})")
    print(f"  95% CI:          [{lo:.1f}%, {hi:.1f}%]")
    print()
    if abs(pct - expected_pct) <= 5:
        print("  -> Consistent with 20-delta ~ 20% ITM (within 5%).")
    else:
        print(f"  -> Difference from theoretical: {pct - expected_pct:+.1f}%")
    print()
    print("  Note: Delta ~ risk-neutral prob of finishing ITM. Empirical rate can")
    print("  be lower (equity risk premium / positive drift) or vary with IV regime.")
    print("=" * 60)

    # By year
    by_year: dict[int, list[bool]] = {}
    for r in results:
        y = r["entry_date"].year
        by_year.setdefault(y, []).append(r["itm"])
    print()
    print("  By year (ITM %):")
    for y in sorted(by_year.keys()):
        arr = by_year[y]
        pct_y = 100.0 * sum(arr) / len(arr)
        print(f"    {y}: {pct_y:.1f}%  (n={len(arr)})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
