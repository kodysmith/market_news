#!/usr/bin/env python3
"""Sweep entry-filter threshold to balance income vs drawdown.

Runs the SPX backtest with the trained XGBoost entry filter at multiple
P(loss) thresholds. Use the table to pick a threshold that keeps income high
while reducing max drawdown.

Goal: increase income by reducing drawdown — skip only the riskiest entries.

Usage:
  python scripts/sweep_entry_filter_threshold.py
  python scripts/sweep_entry_filter_threshold.py --model data/entry_filter_xgboost_spx.joblib --start 2015-01-01 --end 2024-12-31
"""
from __future__ import annotations

import argparse
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.asset_data import load_asset_data
from scripts.evaluate_enhanced_cashflow import get_backtest_trades, compute_gex_proxy


def portfolio_metrics(trades: pd.DataFrame) -> dict:
    """Compute portfolio-level total PnL, max drawdown, n_trades, win_rate, Calmar."""
    if trades.empty:
        return {
            "total_pnl": 0.0,
            "n_trades": 0,
            "win_rate_pct": 0.0,
            "max_drawdown": 0.0,
            "calmar": 0.0,
        }
    total_pnl = float(trades["pnl"].sum())
    n_trades = len(trades)
    win_rate_pct = 100.0 * (trades["pnl"] > 0).sum() / n_trades
    # Equity curve: order by exit_date, cumsum pnl
    by_date = trades.sort_values("exit_date")
    cum = by_date["pnl"].cumsum()
    peak = cum.cummax()
    drawdown = peak - cum
    max_drawdown = float(drawdown.max()) if len(drawdown) else 0.0
    years = (by_date["exit_date"].max() - by_date["exit_date"].min()).days / 365.25 if len(by_date) > 1 else 1.0
    ann_return = total_pnl / years if years > 0 else 0.0
    calmar = ann_return / max_drawdown if max_drawdown > 0 else (ann_return if ann_return > 0 else 0.0)
    return {
        "total_pnl": total_pnl,
        "n_trades": n_trades,
        "win_rate_pct": win_rate_pct,
        "max_drawdown": max_drawdown,
        "calmar": calmar,
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Sweep entry-filter threshold to find best income vs drawdown trade-off.",
    )
    parser.add_argument("--start", type=str, default="2015-01-01")
    parser.add_argument("--end", type=str, default="2025-12-31")
    parser.add_argument(
        "--model",
        type=Path,
        default=ROOT / "data" / "entry_filter_xgboost_spx.joblib",
        help="Path to trained entry filter joblib.",
    )
    parser.add_argument(
        "--thresholds",
        type=str,
        default="none,0.5,0.55,0.6,0.65,0.7,0.75,0.8",
        help="Comma-separated thresholds (use 'none' for no filter).",
    )
    args = parser.parse_args()

    if not args.model.exists():
        print(f"Model not found: {args.model}. Train with scripts/train_entry_filter_xgboost.py", file=sys.stderr)
        return 1

    start = datetime.strptime(args.start, "%Y-%m-%d").date()
    end = datetime.strptime(args.end, "%Y-%m-%d").date()

    print("Loading SPX data...", file=sys.stderr)
    price_df, iv_series = load_asset_data("SPX", start, end)
    if price_df.empty:
        print("No SPX data.", file=sys.stderr)
        return 1
    gex_scores = compute_gex_proxy(price_df, iv_series, start, end)

    thresh_strs = [s.strip().lower() for s in args.thresholds.split(",") if s.strip()]
    results: list[dict] = []

    for th in thresh_strs:
        if th == "none":
            model_path = None
            threshold = 0.5  # ignored
            label = "no filter"
        else:
            try:
                threshold = float(th)
            except ValueError:
                continue
            model_path = args.model
            label = f"P(loss)>{threshold}"

        trades = get_backtest_trades(
            price_df,
            iv_series,
            start,
            end,
            gex_scores=gex_scores,
            enable_es_hedge=True,
            entry_filter_model_path=model_path,
            entry_filter_threshold=threshold,
        )
        m = portfolio_metrics(trades)
        m["threshold"] = label
        results.append(m)

    if not results:
        print("No results.", file=sys.stderr)
        return 1

    # Baseline = no filter
    base = next((r for r in results if r["threshold"] == "no filter"), results[0])
    base_pnl = base["total_pnl"]
    base_dd = base["max_drawdown"]

    print("\n" + "=" * 72, file=sys.stderr)
    print("Entry filter threshold sweep (goal: keep income high, cut drawdown)", file=sys.stderr)
    print("=" * 72, file=sys.stderr)
    print(f"{'Threshold':<18} {'Total PnL':>12} {'PnL %':>8} {'Max DD':>10} {'DD %':>8} {'Trades':>8} {'Win%':>6} {'Calmar':>8}", file=sys.stderr)
    print("-" * 72, file=sys.stderr)

    for r in results:
        pnl_pct = 100.0 * r["total_pnl"] / base_pnl if base_pnl != 0 else 0.0
        dd_pct = 100.0 * r["max_drawdown"] / base_dd if base_dd > 0 else 0.0
        print(
            f"{r['threshold']:<18} ${r['total_pnl']:>10,.0f} {pnl_pct:>7.1f}% ${r['max_drawdown']:>8,.0f} {dd_pct:>7.1f}% {r['n_trades']:>8} {r['win_rate_pct']:>5.1f}% {r['calmar']:>8.2f}",
            file=sys.stderr,
        )

    # Suggest: threshold with best Calmar among those that keep >= 85% of baseline PnL
    filtered = [r for r in results if r["threshold"] != "no filter" and r["total_pnl"] >= 0.85 * base_pnl]
    if filtered:
        best = max(filtered, key=lambda x: x["calmar"])
        print("-" * 72, file=sys.stderr)
        print(f"Suggested (best Calmar with >=85% of baseline PnL): {best['threshold']}", file=sys.stderr)
    print("", file=sys.stderr)

    # Also print to stdout for piping
    df = pd.DataFrame(results)
    df.to_csv(sys.stdout, index=False)
    return 0


if __name__ == "__main__":
    sys.exit(main())
