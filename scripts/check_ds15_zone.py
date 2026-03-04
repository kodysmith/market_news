#!/usr/bin/env python3
"""
Check if today's (or given date's) market data meets the 02_IC21_CLOSE_1DTE_DS15
"zone" criteria for 1.5x size.

Zone criteria (all must be true):
  - VIX between 16 and 24
  - rv_ratio (5d realized vol / 10d) ≤ 1.05
  - GEX score ≥ 2
  - Pre-market gap ≥ -0.3%

Uses yfinance for SPX, VIX, VIX3M, VIX9D. Run with no args for "latest available"
(e.g. last trading day); use --as-of for a specific date.

Usage:
  python scripts/check_ds15_zone.py
  python scripts/check_ds15_zone.py --as-of 2025-12-15
"""
from __future__ import annotations

import argparse
import sys
from datetime import date, datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Zone thresholds (from 02_IC21_CLOSE_1DTE_DS15 config)
ZONE_VIX_MIN = 16.0
ZONE_VIX_MAX = 24.0
ZONE_RV_MAX = 1.05
ZONE_GEX_MIN = 2
ZONE_GAP_MIN = -0.3  # %


def load_spx_vix(start: date, end: date) -> tuple[pd.DataFrame, pd.DataFrame]:
    import yfinance as yf

    end_pad = (end + timedelta(days=10)).isoformat()
    spx = yf.Ticker("^GSPC").history(start=start.isoformat(), end=end_pad)
    vix = yf.Ticker("^VIX").history(start=start.isoformat(), end=end_pad)
    for df in (spx, vix):
        if not df.empty:
            df.index = pd.to_datetime(df.index).tz_localize(None).normalize()
    return spx, vix


def load_gex_data(start: date, end: date) -> tuple[pd.Series, pd.Series]:
    import yfinance as yf

    pad_start = (start - timedelta(days=30)).isoformat()
    pad_end = (end + timedelta(days=10)).isoformat()
    vix3m_df = yf.Ticker("^VIX3M").history(start=pad_start, end=pad_end)
    vix9d_df = yf.Ticker("^VIX9D").history(start=pad_start, end=pad_end)
    vix3m = vix3m_df["Close"].copy() if not vix3m_df.empty else pd.Series(dtype=float)
    vix9d = vix9d_df["Close"].copy() if not vix9d_df.empty else pd.Series(dtype=float)
    for s in (vix3m, vix9d):
        if not s.empty:
            s.index = pd.to_datetime(s.index).tz_localize(None).normalize()
            s.index = s.index.date
    return vix3m, vix9d


def compute_gex_score_for_date(
    d: date,
    spx_close: pd.Series,
    vix_close: pd.Series,
    vix3m: pd.Series,
    rv_20d: pd.Series,
    dates_sorted: list,
) -> int:
    score = 0
    vix_val = float(vix_close.get(d, np.nan))
    if np.isnan(vix_val):
        return 0
    # Signal 1: contango (VIX < VIX3M)
    vix3m_val = float(vix3m.get(d, np.nan)) if not vix3m.empty else np.nan
    if not np.isnan(vix3m_val) and vix3m_val > 0 and vix_val < vix3m_val:
        score += 1
    else:
        vix_sma = vix_close.rolling(20).mean()
        try:
            sma_val = float(vix_sma.get(d, np.nan))
            if not np.isnan(sma_val) and vix_val < sma_val:
                score += 1
        except Exception:
            pass
    # Signal 2: VIX > realized vol
    rv_val = float(rv_20d.get(d, np.nan)) if d in rv_20d.index else np.nan
    if not np.isnan(rv_val) and rv_val > 0 and vix_val > rv_val:
        score += 1
    # Signal 3: VIX 5-day change ≤ 15%
    try:
        idx = dates_sorted.index(d)
        if idx >= 5:
            d_5ago = dates_sorted[idx - 5]
            vix_5ago = float(vix_close.get(d_5ago, np.nan))
            if not np.isnan(vix_5ago) and vix_5ago > 0:
                pct_chg = (vix_val - vix_5ago) / vix_5ago * 100
                if pct_chg <= 15:
                    score += 1
    except (ValueError, IndexError):
        pass
    return score


def main() -> int:
    parser = argparse.ArgumentParser(description="Check DS15 zone criteria from live data")
    parser.add_argument(
        "--as-of",
        type=str,
        default=None,
        help="Date to check (YYYY-MM-DD). Default: latest date in data.",
    )
    args = parser.parse_args()

    end = date.today()
    start = end - timedelta(days=60)
    spx_df, vix_df = load_spx_vix(start, end)
    if spx_df.empty or vix_df.empty:
        print("Could not load SPX/VIX data.", file=sys.stderr)
        return 1

    spx_df.index = spx_df.index.date
    vix_df.index = vix_df.index.date
    spx_close = spx_df["Close"].copy()
    spx_close.index = spx_df.index
    vix_close = vix_df["Close"].copy()
    vix_close.index = vix_df.index
    spx_open = spx_df["Open"].copy()
    spx_open.index = spx_df.index

    if args.as_of:
        try:
            as_of = datetime.strptime(args.as_of, "%Y-%m-%d").date()
        except ValueError:
            print(f"Invalid --as-of {args.as_of}", file=sys.stderr)
            return 1
    else:
        as_of = max(d for d in spx_close.index if d in vix_close.index)

    if as_of not in spx_close.index or as_of not in vix_close.index:
        print(f"No data for {as_of}. Try --as-of with a recent trading date.", file=sys.stderr)
        return 1

    vix_val = float(vix_close[as_of])
    spot = float(spx_close[as_of])

    log_ret = np.log(spx_close / spx_close.shift(1))
    rv_5d = log_ret.rolling(5).std() * np.sqrt(252) * 100
    rv_10d = log_ret.rolling(10).std() * np.sqrt(252) * 100
    rv_ratio_series = rv_5d / rv_10d
    rv_val = float(rv_ratio_series.get(as_of, np.nan)) if as_of in rv_ratio_series.index else np.nan
    if np.isnan(rv_val):
        rv_val = 1.0

    rv_20d = log_ret.rolling(20).std() * np.sqrt(252) * 100
    vix3m, _ = load_gex_data(start, end)
    dates_sorted = sorted(spx_close.index)
    gex_score = compute_gex_score_for_date(
        as_of, spx_close, vix_close, vix3m, rv_20d, dates_sorted
    )

    prev_close = spx_close.shift(1).get(as_of)
    open_today = spx_open.get(as_of)
    if prev_close is not None and open_today is not None and not np.isnan(prev_close) and prev_close > 0:
        gap_pct = (float(open_today) - float(prev_close)) / float(prev_close) * 100.0
    else:
        gap_pct = np.nan

    vix_ok = ZONE_VIX_MIN <= vix_val <= ZONE_VIX_MAX
    rv_ok = not np.isnan(rv_val) and rv_val <= ZONE_RV_MAX
    gex_ok = gex_score >= ZONE_GEX_MIN
    gap_ok = not np.isnan(gap_pct) and gap_pct >= ZONE_GAP_MIN
    zone_met = vix_ok and rv_ok and gex_ok and gap_ok

    print(f"Date: {as_of}")
    print(f"SPX close: {spot:.2f}")
    print()
    print("02_IC21_CLOSE_1DTE_DS15 zone criteria (1.5x size when all pass):")
    print(f"  1. VIX in [{ZONE_VIX_MIN}, {ZONE_VIX_MAX}]     : {vix_val:.2f}  -> {'PASS' if vix_ok else 'FAIL'}")
    print(f"  2. rv_ratio (5d/10d vol) ≤ {ZONE_RV_MAX}          : {rv_val:.3f}  -> {'PASS' if rv_ok else 'FAIL'}")
    print(f"  3. GEX score ≥ {ZONE_GEX_MIN}                     : {gex_score}    -> {'PASS' if gex_ok else 'FAIL'}")
    print(f"  4. Gap % ≥ {ZONE_GAP_MIN}%                        : {gap_pct:.2f}%  -> {'PASS' if gap_ok else 'FAIL'}")
    print()
    if zone_met:
        print("  -> ZONE MET: use 1.5x size for 02_IC21_CLOSE_1DTE_DS15.")
    else:
        print("  -> ZONE NOT MET: use 1x size (or skip DS15 sizing).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
