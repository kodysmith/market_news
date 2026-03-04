#!/usr/bin/env python3
"""Export labeled training data from the cashflow backtest.

Produces one row per trade with entry-day features and outcome labels (pnl, win, exit_reason)
for use in classification, regression, or sequence models.

Usage:
  python scripts/export_cashflow_training_data.py
  python scripts/export_cashflow_training_data.py --symbols SPX,GLD --output data/training.parquet --format parquet
"""
from __future__ import annotations

import argparse
import sys
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import yfinance as yf

from scripts.asset_data import SYMBOL_TO_TICKER, load_asset_data
from scripts.evaluate_enhanced_cashflow import (
    get_backtest_trades,
    compute_gex_proxy,
    compute_rolling_features_for_entry,
    compute_entry_features_extended,
    build_extended_feature_series,
)

# Rolling feature column names (lookback-only; must match compute_rolling_features_for_entry keys)
ROLLING_FEATURE_NAMES = [
    "ret_5d_lb", "ret_10d_lb", "ret_21d_lb",
    "vix_5d_mean_lb", "vix_5d_max_lb", "vix_21d_mean_lb", "vix_21d_max_lb",
    "gap_5d_mean_lb", "gap_5d_std_lb", "rv_ratio_5d_mean_lb",
    "spot_vs_sma10_lb", "spot_vs_sma21_lb",
]

# Extended feature columns (vol trend, term structure, regime, interactions)
EXTENDED_FEATURE_NAMES = [
    "vix_change_1d", "vix_change_5d", "vix_percentile_252d",
    "rv_5", "rv_10", "rv_20", "rv_trend", "iv_rv_spread",
    "vix9d_vix_ratio", "term_structure_contango",
    "spot_vs_sma50", "spot_vs_sma200", "atr_pct_14",
    "gap_abs_z", "gap_direction", "regime",
    "vix_at_entry_x_rv_ratio_at_entry", "gex_score_at_entry_x_vix_change_5d",
    "spot_vs_sma21_lb_x_vix_change_5d",
]


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Export cashflow backtest trades as labeled training data (one row per trade).",
    )
    parser.add_argument("--start", type=str, default="2015-01-01")
    parser.add_argument("--end", type=str, default="2025-12-31")
    parser.add_argument("--symbol", type=str, default="SPX", help="Single asset symbol.")
    parser.add_argument(
        "--symbols",
        type=str,
        help="Comma-separated symbols (e.g. SPX,XSP,GLD,SLV). Overrides --symbol if set.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "data" / "cashflow_training_data.csv",
        help="Output path for the training dataset.",
    )
    parser.add_argument(
        "--format",
        type=str,
        choices=["csv", "parquet"],
        default="csv",
        help="Output format.",
    )
    args = parser.parse_args()

    if args.symbols:
        symbols = [s.strip().upper() for s in args.symbols.split(",") if s.strip()]
    else:
        symbols = [args.symbol.strip().upper()]

    for s in symbols:
        if s not in SYMBOL_TO_TICKER:
            print(f"Unknown symbol: {s}. Supported: {list(SYMBOL_TO_TICKER.keys())}", file=sys.stderr)
            return 1

    start = datetime.strptime(args.start, "%Y-%m-%d").date()
    end = datetime.strptime(args.end, "%Y-%m-%d").date()
    start_padded = start - timedelta(days=30)
    # SPX needs longer history for vix_percentile_252d and extended series
    start_padded_spx = start - timedelta(days=400)
    args.output.parent.mkdir(parents=True, exist_ok=True)

    frames: list[pd.DataFrame] = []
    for symbol in symbols:
        print(f"Loading {symbol}...", file=sys.stderr)
        load_start = start_padded_spx if symbol == "SPX" else start_padded
        price_df, iv_series = load_asset_data(symbol, load_start, end)
        if price_df.empty:
            print(f"No data for {symbol}. Skipping.", file=sys.stderr)
            continue

        gex_scores = None
        if symbol == "SPX":
            print("  Computing GEX proxy...", file=sys.stderr)
            gex_scores = compute_gex_proxy(price_df, iv_series, start, end)

        enable_es_hedge = symbol in ("SPX", "XSP")
        trades = get_backtest_trades(
            price_df,
            iv_series,
            start,
            end,
            gex_scores=gex_scores,
            enable_es_hedge=enable_es_hedge,
        )
        if trades.empty:
            continue
        trades["symbol"] = symbol

        if symbol == "SPX":
            spx_close = price_df["Close"].copy()
            spx_close.index = pd.to_datetime(spx_close.index).tz_localize(None).normalize()
            spx_close.index = spx_close.index.date
            spx_open = price_df["Open"].copy()
            spx_open.index = pd.to_datetime(spx_open.index).tz_localize(None).normalize()
            spx_open.index = spx_open.index.date
            prev_close = spx_close.shift(1)
            gap_pct = (spx_open - prev_close) / prev_close * 100.0
            vix_close = iv_series.copy()
            if not vix_close.empty:
                vix_close.index = pd.to_datetime(vix_close.index).tz_localize(None).normalize()
                vix_close.index = vix_close.index.date
            log_ret = np.log(spx_close / spx_close.shift(1))
            rv_5d = log_ret.rolling(5).std() * np.sqrt(252) * 100
            rv_10d = log_ret.rolling(10).std() * np.sqrt(252) * 100
            rv_ratio_series = rv_5d / rv_10d

            vix9d_series = None
            vix3m_series = None
            pad_start = (load_start - timedelta(days=10)).isoformat()
            pad_end = (end + timedelta(days=10)).isoformat()
            vix3m_df = yf.Ticker("^VIX3M").history(start=pad_start, end=pad_end)
            vix9d_df = yf.Ticker("^VIX9D").history(start=pad_start, end=pad_end)
            if not vix3m_df.empty:
                vix3m_series = vix3m_df["Close"].copy()
                vix3m_series.index = pd.to_datetime(vix3m_series.index).tz_localize(None).normalize()
                vix3m_series.index = pd.Index(vix3m_series.index.date)
            if not vix9d_df.empty:
                vix9d_series = vix9d_df["Close"].copy()
                vix9d_series.index = pd.to_datetime(vix9d_series.index).tz_localize(None).normalize()
                vix9d_series.index = pd.Index(vix9d_series.index.date)

            optional_series = build_extended_feature_series(
                spx_close, vix_close, gap_pct, log_ret,
                price_df=price_df, vix9d=vix9d_series, vix3m=vix3m_series,
            )
            print("  Enriching SPX with rolling and extended features...", file=sys.stderr)
            for col in ROLLING_FEATURE_NAMES + EXTENDED_FEATURE_NAMES:
                trades[col] = np.nan
            for i, row in trades.iterrows():
                ed = row["entry_date"]
                iv_at_entry = row.get("iv_entry")
                gex_score = row.get("gex_score_at_entry")
                rv_ratio_at_entry = rv_ratio_series.loc[ed] if ed in rv_ratio_series.index else None
                feats = compute_entry_features_extended(
                    ed, spx_close, vix_close, gap_pct, rv_ratio_series,
                    optional_series=optional_series,
                    iv_at_entry=float(iv_at_entry) if pd.notna(iv_at_entry) else None,
                    gex_score_at_entry=gex_score if pd.notna(gex_score) else None,
                    rv_ratio_at_entry=float(rv_ratio_at_entry) if rv_ratio_at_entry is not None and pd.notna(rv_ratio_at_entry) else None,
                )
                for k, v in feats.items():
                    trades.at[i, k] = v
        else:
            for col in ROLLING_FEATURE_NAMES + EXTENDED_FEATURE_NAMES:
                trades[col] = np.nan

        frames.append(trades)

    if not frames:
        print("No trades produced.", file=sys.stderr)
        return 1

    out = pd.concat(frames, ignore_index=True)

    # Tail-loss and bad-quantile labels (predict damage, not just loss)
    out["tail_1x"] = (out["pnl"] <= -1.0 * out["entry_credit"]).astype(int)
    out["tail_1_5x"] = (out["pnl"] <= -1.5 * out["entry_credit"]).astype(int)
    q10_by_config = out.groupby("config_id")["pnl"].transform(lambda x: x.quantile(0.10))
    out["bad_q10"] = (out["pnl"] <= q10_by_config).astype(int)

    # Derived calendar features
    entry_dt = pd.to_datetime(out["entry_date"])
    out["entry_year"] = entry_dt.dt.year
    out["entry_month"] = entry_dt.dt.month
    out["entry_day_of_week"] = entry_dt.dt.dayofweek

    if args.format == "csv":
        out.to_csv(args.output, index=False)
    else:
        out.to_parquet(args.output, index=False)

    print(f"Wrote {args.output}: {len(out)} rows, {len(out.columns)} columns", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
