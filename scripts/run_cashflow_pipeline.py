#!/usr/bin/env python3
"""Multi-asset cashflow pipeline: run enhanced backtest per symbol and write HTML reports.

Usage:
  python scripts/run_cashflow_pipeline.py --symbol GLD
  python scripts/run_cashflow_pipeline.py --symbols SPX,XSP,GLD,SLV --start 2020-01-01 --end 2024-12-31
"""
from __future__ import annotations

import argparse
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.asset_data import SYMBOL_TO_TICKER, load_asset_data
from scripts.evaluate_enhanced_cashflow import (
    run_backtest_and_report,
    compute_gex_proxy,
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run enhanced cashflow backtest for one or more assets and write HTML reports.",
    )
    parser.add_argument(
        "--symbol",
        type=str,
        help="Single asset symbol (e.g. SPX, XSP, GLD, SLV).",
    )
    parser.add_argument(
        "--symbols",
        type=str,
        help="Comma-separated symbols (e.g. SPX,XSP,GLD,SLV). Overrides --symbol if set.",
    )
    parser.add_argument("--start", type=str, default="2015-01-01")
    parser.add_argument("--end", type=str, default="2025-12-31")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "data",
        help="Directory for HTML reports (and optional trades CSV).",
    )
    parser.add_argument(
        "--save-trades",
        action="store_true",
        help="Write trades CSV per asset to output-dir (enhanced_cashflow_trades_{symbol}.csv).",
    )
    parser.add_argument(
        "--entry-filter-model",
        type=Path,
        default=None,
        help="Path to XGBoost entry filter joblib (SPX only). Enables baseline vs filtered comparison in report.",
    )
    parser.add_argument(
        "--entry-filter-threshold",
        type=float,
        default=0.5,
        help="P(loss) threshold for entry filter when --entry-filter-model is set (default 0.5).",
    )
    parser.add_argument(
        "--entry-filter-log",
        type=Path,
        default=None,
        help="When using entry filter (SPX), append per-candidate log to this CSV.",
    )
    args = parser.parse_args()

    if args.symbols:
        symbols = [s.strip().upper() for s in args.symbols.split(",") if s.strip()]
    elif args.symbol:
        symbols = [args.symbol.strip().upper()]
    else:
        parser.error("Provide --symbol or --symbols")
        return 1

    for s in symbols:
        if s not in SYMBOL_TO_TICKER:
            print(f"Unknown symbol: {s}. Supported: {list(SYMBOL_TO_TICKER.keys())}", file=sys.stderr)
            return 1

    start = datetime.strptime(args.start, "%Y-%m-%d").date()
    end = datetime.strptime(args.end, "%Y-%m-%d").date()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    for symbol in symbols:
        print(f"\n--- {symbol} ---", file=sys.stderr)
        price_df, iv_series = load_asset_data(symbol, start, end)
        if price_df.empty:
            print(f"No data for {symbol}. Skipping.", file=sys.stderr)
            continue

        # GEX: only SPX has VIX term-structure proxy; others use pass-all (None)
        gex_scores = None
        if symbol == "SPX":
            print("Computing GEX regime proxy...", file=sys.stderr)
            gex_scores = compute_gex_proxy(price_df, iv_series, start, end)

        enable_es_hedge = symbol in ("SPX", "XSP")
        report_path = args.output_dir / f"enhanced_cashflow_results_{symbol}.html"
        trades_csv = None
        if args.save_trades:
            trades_csv = args.output_dir / f"enhanced_cashflow_trades_{symbol}.csv"

        entry_filter_path = None
        entry_filter_log_path = None
        if symbol == "SPX" and args.entry_filter_model is not None and args.entry_filter_model.exists():
            entry_filter_path = args.entry_filter_model
            if args.entry_filter_log is not None:
                entry_filter_log_path = args.entry_filter_log

        code = run_backtest_and_report(
            price_df,
            iv_series,
            start,
            end,
            report_path,
            gex_scores=gex_scores,
            enable_es_hedge=enable_es_hedge,
            report_title=symbol,
            trades_csv=trades_csv,
            verbose=True,
            entry_filter_model_path=entry_filter_path,
            entry_filter_threshold=args.entry_filter_threshold,
            entry_filter_log_path=entry_filter_log_path,
        )
        if code != 0:
            return code

    if len(symbols) > 1:
        index_path = args.output_dir / "index.html"
        with open(index_path, "w") as f:
            f.write("<!DOCTYPE html><html><head><meta charset='utf-8'><title>Cash Flow Reports</title></head><body>\n")
            f.write("<h1>Enhanced Cash Flow — Multi-Asset</h1>\n<ul>\n")
            for s in symbols:
                f.write(f'  <li><a href="enhanced_cashflow_results_{s}.html">{s}</a></li>\n')
            f.write("</ul></body></html>\n")
        print(f"\nWrote index: {index_path}", file=sys.stderr)

    return 0


if __name__ == "__main__":
    sys.exit(main())
