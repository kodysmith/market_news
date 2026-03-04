#!/usr/bin/env python3
"""Time-chunk experiments for XGBoost entry filter: train on one period, test on another.

Runs forward-expanding, forward-rolling, and backward train/test windows to check
for overfitting and regime sensitivity. Uses the same enriched SPX training data
and model setup as train_entry_filter_xgboost.py.

Usage:
  python scripts/train_entry_filter_chunks.py --mode forward_expanding
  python scripts/train_entry_filter_chunks.py --mode backward --output-csv data/chunk_results.csv
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Match train_entry_filter_xgboost.py (lookback _lb + extended vol/regime/interactions)
DEFAULT_FEATURE_COLUMNS = [
    "spot_entry", "vix_at_entry", "gap_pct_at_entry", "gex_score_at_entry", "rv_ratio_at_entry",
    "entry_year", "entry_month", "entry_day_of_week",
    "dte", "short_delta", "width", "profit_target", "stop_mult", "iv_entry", "entry_credit",
    "ret_5d_lb", "ret_10d_lb", "ret_21d_lb",
    "vix_5d_mean_lb", "vix_5d_max_lb", "vix_21d_mean_lb", "vix_21d_max_lb",
    "gap_5d_mean_lb", "gap_5d_std_lb", "rv_ratio_5d_mean_lb",
    "spot_vs_sma10_lb", "spot_vs_sma21_lb",
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
        description="Train XGBoost entry filter in time chunks; test forward and backward.",
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=ROOT / "data" / "cashflow_training_data.csv",
        help="Enriched training CSV or Parquet (SPX).",
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["forward_expanding", "forward_rolling", "backward", "all"],
        default="all",
        help="forward_expanding | forward_rolling | backward | all.",
    )
    parser.add_argument(
        "--target",
        type=str,
        choices=["loss", "big_loss", "tail_1x", "tail_1_5x", "bad_q10"],
        default="loss",
    )
    parser.add_argument(
        "--loss-threshold",
        type=float,
        default=500.0,
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=None,
        help="If set, write results table to this path.",
    )
    parser.add_argument(
        "--no-scale-pos-weight",
        action="store_true",
        help="Disable scale_pos_weight (same as main script).",
    )
    parser.add_argument(
        "--include-config",
        action="store_true",
        help="Include config_id (label-encoded) as feature.",
    )
    parser.add_argument(
        "--no-entry-year",
        action="store_true",
        dest="no_entry_year",
        help="Exclude entry_year from the feature set.",
    )
    args = parser.parse_args()

    if not args.input.exists():
        print(f"Input not found: {args.input}", file=sys.stderr)
        return 1

    try:
        import numpy as np
        import xgboost as xgb
        from sklearn.metrics import (
            accuracy_score,
            balanced_accuracy_score,
            f1_score,
            precision_score,
            recall_score,
            roc_auc_score,
        )
        from sklearn.preprocessing import LabelEncoder
    except ImportError as e:
        print(f"Missing dependency: {e}", file=sys.stderr)
        return 1

    if args.input.suffix.lower() == ".parquet":
        df = pd.read_parquet(args.input)
    else:
        df = pd.read_csv(args.input)
    df["entry_date"] = pd.to_datetime(df["entry_date"])
    df = df.sort_values("entry_date").reset_index(drop=True)

    spx = df[df["symbol"] == "SPX"].copy()
    if spx.empty:
        print("No SPX rows.", file=sys.stderr)
        return 1
    if "entry_year" not in spx.columns:
        spx["entry_year"] = pd.to_datetime(spx["entry_date"]).dt.year

    feature_cols = [c for c in DEFAULT_FEATURE_COLUMNS if c in spx.columns]
    if args.no_entry_year and "entry_year" in feature_cols:
        feature_cols = [c for c in feature_cols if c != "entry_year"]
    if args.include_config and "config_id" in spx.columns and "config_id" not in feature_cols:
        feature_cols = feature_cols + ["config_id"]
        le = LabelEncoder()
        spx["config_id"] = le.fit_transform(spx["config_id"].astype(str))

    if args.target == "loss":
        spx["target"] = (~spx["win"]).astype(int)
    elif args.target == "big_loss":
        spx["target"] = (spx["pnl"] < -abs(args.loss_threshold)).astype(int)
    elif args.target in ("tail_1x", "tail_1_5x", "bad_q10"):
        if args.target not in spx.columns:
            print(f"Target '{args.target}' not in data. Re-export training data.", file=sys.stderr)
            return 1
        spx["target"] = spx[args.target].astype(int)
    else:
        print(f"Unknown target: {args.target}", file=sys.stderr)
        return 1

    spx = spx.dropna(subset=feature_cols)
    if spx.empty:
        print("No rows after dropna.", file=sys.stderr)
        return 1

    # Define (train_start, train_end, test_start, test_end) by year
    def run_window(train_start: int, train_end: int, test_start: int, test_end: int, label: str) -> dict:
        train_df = spx[(spx["entry_year"] >= train_start) & (spx["entry_year"] <= train_end)]
        test_df = spx[(spx["entry_year"] >= test_start) & (spx["entry_year"] <= test_end)]
        if train_df.empty or test_df.empty:
            return {"label": label, "train_n": len(train_df), "test_n": len(test_df), "error": "empty split"}
        X_train = train_df[feature_cols]
        y_train = train_df["target"]
        X_test = test_df[feature_cols]
        y_test = test_df["target"]
        n_pos = int(y_train.sum())
        n_neg = len(y_train) - n_pos
        scale_pos_weight = (n_neg / n_pos) if n_pos > 0 and not args.no_scale_pos_weight else 1.0
        model = xgb.XGBClassifier(
            n_estimators=400,
            max_depth=6,
            learning_rate=0.03,
            subsample=0.8,
            colsample_bytree=0.8,
            min_child_weight=3,
            reg_alpha=0.1,
            reg_lambda=1.0,
            scale_pos_weight=scale_pos_weight,
            random_state=42,
            eval_metric="auc",
        )
        model.fit(X_train, y_train, verbose=False)
        y_prob = model.predict_proba(X_test)[:, 1]
        y_pred = (y_prob >= 0.5).astype(int)
        try:
            auc = roc_auc_score(y_test, y_prob)
        except Exception:
            auc = float("nan")
        return {
            "label": label,
            "train_start": train_start,
            "train_end": train_end,
            "test_start": test_start,
            "test_end": test_end,
            "train_n": len(train_df),
            "test_n": len(test_df),
            "test_accuracy": accuracy_score(y_test, y_pred),
            "test_balanced_acc": balanced_accuracy_score(y_test, y_pred),
            "test_auc": auc,
            "test_f1": f1_score(y_test, y_pred, zero_division=0),
            "test_precision": precision_score(y_test, y_pred, zero_division=0),
            "test_recall": recall_score(y_test, y_pred, zero_division=0),
        }

    windows: list[tuple[int, int, int, int, str]] = []

    if args.mode in ("forward_expanding", "all"):
        windows.extend([
            (2015, 2019, 2020, 2021, "forward_expand_2015_2019_test_2020_2021"),
            (2015, 2020, 2021, 2022, "forward_expand_2015_2020_test_2021_2022"),
            (2015, 2021, 2022, 2023, "forward_expand_2015_2021_test_2022_2023"),
            (2015, 2022, 2023, 2025, "forward_expand_2015_2022_test_2023_2025"),
        ])
    if args.mode in ("forward_rolling", "all"):
        windows.extend([
            (2015, 2017, 2018, 2019, "rolling_2015_2017_test_2018_2019"),
            (2017, 2019, 2020, 2021, "rolling_2017_2019_test_2020_2021"),
            (2019, 2021, 2022, 2023, "rolling_2019_2021_test_2022_2023"),
            (2020, 2022, 2023, 2025, "rolling_2020_2022_test_2023_2025"),
        ])
    if args.mode in ("backward", "all"):
        windows.extend([
            (2020, 2025, 2015, 2019, "backward_train_2020_2025_test_2015_2019"),
            (2018, 2025, 2015, 2017, "backward_train_2018_2025_test_2015_2017"),
        ])

    results: list[dict] = []
    for train_start, train_end, test_start, test_end, label in windows:
        r = run_window(train_start, train_end, test_start, test_end, label)
        results.append(r)

    # Print table
    print("\nTime-chunk experiment results (SPX entry filter)", file=sys.stderr)
    print("=" * 100, file=sys.stderr)
    fmt = "{:<45} {:>8} {:>8} {:>8} {:>8} {:>8} {:>8}"
    print(fmt.format("Window", "TrainN", "TestN", "Acc", "BalAcc", "AUC", "F1"), file=sys.stderr)
    print("-" * 100, file=sys.stderr)
    for r in results:
        if "error" in r:
            print(fmt.format(r["label"], r["train_n"], r["test_n"], "-", "-", "-", "-"), file=sys.stderr)
            continue
        print(fmt.format(
            r["label"],
            r["train_n"],
            r["test_n"],
            f"{r['test_accuracy']:.3f}",
            f"{r['test_balanced_acc']:.3f}",
            f"{r['test_auc']:.3f}" if not pd.isna(r["test_auc"]) else "N/A",
            f"{r['test_f1']:.3f}",
        ), file=sys.stderr)
    print("=" * 100, file=sys.stderr)

    if args.output_csv:
        out_df = pd.DataFrame(results)
        out_df.to_csv(args.output_csv, index=False)
        print(f"Wrote {args.output_csv}", file=sys.stderr)

    return 0


if __name__ == "__main__":
    sys.exit(main())
