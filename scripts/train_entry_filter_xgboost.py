#!/usr/bin/env python3
"""Train XGBoost entry-risk model (P(loss) or P(big_loss)) on SPX cashflow training data.

Uses enriched training data with entry-day and rolling features. Saves model and
feature importance for use as an entry filter in the backtest.

Usage:
  python scripts/train_entry_filter_xgboost.py --input data/cashflow_training_data.csv
  python scripts/train_entry_filter_xgboost.py --target tail_1x --date-level --calibrate --val-year 2023
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Default feature columns (entry-day + lookback rolling + extended vol/regime/interactions).
# spot_entry removed: absolute SPX level encodes calendar year (temporal leakage).
# spot_vs_sma50 and spot_vs_sma200 capture the same structural signal without leakage.
DEFAULT_FEATURE_COLUMNS = [
    "vix_at_entry", "gap_pct_at_entry", "gex_score_at_entry", "rv_ratio_at_entry",
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

# Config-specific columns excluded from market-level aggregation.
# These vary per strategy config; market features are identical across configs on the same date.
CONFIG_SPECIFIC_COLUMNS = {
    "dte", "width", "short_delta", "profit_target", "stop_mult", "iv_entry", "entry_credit",
}


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Train XGBoost entry-risk model on SPX cashflow training data.",
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=ROOT / "data" / "cashflow_training_data.csv",
        help="Path to enriched training CSV or Parquet.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "data",
        help="Directory for saved model and importance file.",
    )
    parser.add_argument(
        "--target",
        type=str,
        choices=["loss", "big_loss", "tail_1x", "tail_1_5x", "bad_q10"],
        default="tail_1x",
        help="Target: loss, big_loss, tail_1x (pnl<=-1x credit), tail_1_5x, bad_q10 (worst 10%% by config).",
    )
    parser.add_argument(
        "--loss-threshold",
        type=float,
        default=500.0,
        help="PnL threshold for big_loss target (default 500).",
    )
    parser.add_argument(
        "--test-year",
        type=int,
        default=2024,
        help="Treat entries with entry_year >= this as test set (time-based split).",
    )
    parser.add_argument(
        "--test-frac",
        type=float,
        default=None,
        help="If set, use last test_frac of rows by entry_date as test (overrides --test-year).",
    )
    parser.add_argument(
        "--features",
        type=str,
        default=None,
        help="Comma-separated feature column names. Default: all entry + rolling columns.",
    )
    parser.add_argument(
        "--tune-threshold",
        action="store_true",
        help="Find decision threshold that maximizes accuracy on test (saved with model).",
    )
    parser.add_argument(
        "--val-year",
        type=int,
        default=None,
        help="If set, use this year as validation for early stopping (e.g. 2023).",
    )
    parser.add_argument(
        "--scale-pos-weight",
        action="store_true",
        default=True,
        help="Use scale_pos_weight for class imbalance (default True).",
    )
    parser.add_argument(
        "--no-scale-pos-weight",
        action="store_false",
        dest="scale_pos_weight",
        help="Disable scale_pos_weight.",
    )
    parser.add_argument(
        "--include-config",
        action="store_true",
        help="Include config_id (label-encoded) as a feature; can improve accuracy if strategy type predicts loss.",
    )
    parser.add_argument(
        "--no-entry-year",
        action="store_true",
        dest="no_entry_year",
        help="Exclude entry_year from the feature set (production: avoid year leakage).",
    )
    parser.add_argument(
        "--date-level",
        action="store_true",
        help=(
            "Collapse to one row per entry_date before training (eliminates pseudo-replication "
            "from 99 configs sharing the same market features on the same date)."
        ),
    )
    parser.add_argument(
        "--date-label-threshold",
        type=float,
        default=0.20,
        help=(
            "Fraction of configs with tail loss required to label a date as bad "
            "(only used with --date-level, default 0.20)."
        ),
    )
    parser.add_argument(
        "--calibrate",
        action="store_true",
        help=(
            "Wrap fitted XGBoost model with isotonic probability calibration "
            "(CalibratedClassifierCV). Calibration data is val-year if provided, "
            "otherwise the last 20%% of training chronologically."
        ),
    )
    args = parser.parse_args()

    if not args.input.exists():
        print(f"Input not found: {args.input}", file=sys.stderr)
        return 1

    try:
        import joblib
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
    except ImportError as e:
        print(f"Missing dependency: {e}. Install xgboost, scikit-learn, joblib.", file=sys.stderr)
        return 1

    if args.input.suffix.lower() == ".parquet":
        df = pd.read_parquet(args.input)
    else:
        df = pd.read_csv(args.input)
    df["entry_date"] = pd.to_datetime(df["entry_date"])
    df = df.sort_values("entry_date").reset_index(drop=True)

    spx = df[df["symbol"] == "SPX"].copy()
    if spx.empty:
        print("No SPX rows in input.", file=sys.stderr)
        return 1
    if "entry_year" not in spx.columns:
        spx["entry_year"] = pd.to_datetime(spx["entry_date"]).dt.year
    print(f"SPX rows: {len(spx)}", file=sys.stderr)

    # Resolve feature columns before potential date-level aggregation
    if args.features:
        feature_cols = [c.strip() for c in args.features.split(",") if c.strip()]
    else:
        feature_cols = [c for c in DEFAULT_FEATURE_COLUMNS if c in spx.columns]
    if args.no_entry_year and "entry_year" in feature_cols:
        feature_cols = [c for c in feature_cols if c != "entry_year"]
    if args.include_config and "config_id" in spx.columns and "config_id" not in feature_cols:
        feature_cols = feature_cols + ["config_id"]
    missing = [c for c in feature_cols if c not in spx.columns]
    if missing:
        print(f"Missing feature columns (will be skipped): {missing}", file=sys.stderr)
        feature_cols = [c for c in feature_cols if c in spx.columns]
    if not feature_cols:
        print("No feature columns available.", file=sys.stderr)
        return 1

    config_encoder = None
    if "config_id" in feature_cols:
        from sklearn.preprocessing import LabelEncoder
        config_encoder = LabelEncoder()
        spx["config_id"] = config_encoder.fit_transform(spx["config_id"].astype(str))

    # Build raw target column (needed for date-level aggregation or direct use)
    target_col = "target"
    if args.target == "loss":
        spx[target_col] = (~spx["win"]).astype(int)
    elif args.target == "big_loss":
        spx[target_col] = (spx["pnl"] < -abs(args.loss_threshold)).astype(int)
    elif args.target in ("tail_1x", "tail_1_5x", "bad_q10"):
        if args.target not in spx.columns:
            print(
                f"Target column '{args.target}' not in data. "
                "Re-export with export_cashflow_training_data.py.",
                file=sys.stderr,
            )
            return 1
        spx[target_col] = spx[args.target].astype(int)
    else:
        print(f"Unknown target: {args.target}", file=sys.stderr)
        return 1

    # --- Date-level aggregation (eliminates pseudo-replication) ---
    if args.date_level:
        market_cols = [c for c in feature_cols if c not in CONFIG_SPECIFIC_COLUMNS]
        if not market_cols:
            print("No market-level feature columns remain after removing config-specific cols.", file=sys.stderr)
            return 1

        print(
            f"[date-level] Collapsing {len(spx)} rows → date-level "
            f"(market cols: {len(market_cols)}, label threshold: {args.date_label_threshold})",
            file=sys.stderr,
        )

        feat_agg = spx.groupby("entry_date")[market_cols].first()
        label_agg = (
            spx.groupby("entry_date")[target_col].mean() >= args.date_label_threshold
        ).astype(int)
        label_agg.name = target_col

        date_df = feat_agg.join(label_agg).reset_index()
        date_df["entry_year"] = date_df["entry_date"].dt.year

        spx = date_df
        feature_cols = market_cols
        print(
            f"[date-level] Date-level rows: {len(spx)}, "
            f"target mean: {spx[target_col].mean():.3f}",
            file=sys.stderr,
        )

    spx = spx.dropna(subset=feature_cols)
    if spx.empty:
        print("No rows left after dropping NaN in features.", file=sys.stderr)
        return 1
    print(
        f"Rows after dropna: {len(spx)}, "
        f"target mean (positive class): {spx[target_col].mean():.3f}",
        file=sys.stderr,
    )

    # --- Train / val / test split ---
    if args.test_frac is not None:
        n = len(spx)
        n_test = max(1, int(n * args.test_frac))
        train_df = spx.iloc[: n - n_test]
        test_df = spx.iloc[-n_test:]
        val_df = None
    else:
        test_df = spx[spx["entry_year"] >= args.test_year]
        train_val_df = spx[spx["entry_year"] < args.test_year]
        if args.val_year is not None:
            val_df = train_val_df[train_val_df["entry_year"] == args.val_year]
            train_df = train_val_df[train_val_df["entry_year"] != args.val_year]
        else:
            val_df = None
            train_df = train_val_df

    if train_df.empty or test_df.empty:
        print("Train or test set empty; adjust --test-year or --test-frac.", file=sys.stderr)
        return 1
    if args.val_year is not None and (val_df is None or val_df.empty):
        print("Validation set empty; disable --val-year or choose another year.", file=sys.stderr)
        return 1
    print(
        f"Train: {len(train_df)}, "
        f"Val: {len(val_df) if val_df is not None else 0}, "
        f"Test: {len(test_df)}",
        file=sys.stderr,
    )

    X_train = train_df[feature_cols]
    y_train = train_df[target_col]
    X_test = test_df[feature_cols]
    y_test = test_df[target_col]

    # Class imbalance: weight positive class so model doesn't just predict majority
    scale_pos_weight = 1.0
    if args.scale_pos_weight:
        n_pos = int(y_train.sum())
        n_neg = len(y_train) - n_pos
        scale_pos_weight = (n_neg / n_pos) if n_pos > 0 else 1.0
        print(f"scale_pos_weight: {scale_pos_weight:.2f} (positive class = loss)", file=sys.stderr)

    # --- Calibration data setup ---
    # If calibrating without a val year, carve last 20% of training chronologically.
    X_calib: pd.DataFrame | None = None
    y_calib: pd.Series | None = None
    if args.calibrate and val_df is None:
        n_tr = len(train_df)
        n_calib = max(1, int(n_tr * 0.20))
        calib_df = train_df.iloc[-n_calib:]
        train_df = train_df.iloc[: n_tr - n_calib]
        X_calib = calib_df[feature_cols]
        y_calib = calib_df[target_col]
        X_train = train_df[feature_cols]
        y_train = train_df[target_col]
        print(
            f"[calibrate] Carved {n_calib} rows from training for calibration "
            f"(remaining train: {len(train_df)})",
            file=sys.stderr,
        )

    fit_kw: dict = {}
    early_stopping = None
    if val_df is not None and not val_df.empty:
        X_val = val_df[feature_cols]
        y_val = val_df[target_col]
        fit_kw["eval_set"] = [(X_val, y_val)]
        fit_kw["verbose"] = False
        early_stopping = 40
        if args.calibrate:
            X_calib = X_val
            y_calib = y_val

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
        early_stopping_rounds=early_stopping,
    )
    model.fit(X_train, y_train, **fit_kw)

    # --- Isotonic calibration ---
    final_model = model
    if args.calibrate:
        if X_calib is None or y_calib is None:
            print("[calibrate] WARNING: no calibration data available; skipping calibration.", file=sys.stderr)
        else:
            from sklearn.calibration import CalibratedClassifierCV
            print(f"[calibrate] Fitting isotonic calibration on {len(X_calib)} rows.", file=sys.stderr)
            calibrated = CalibratedClassifierCV(estimator=model, cv="prefit", method="sigmoid")
            calibrated.fit(X_calib, y_calib)
            final_model = calibrated

    y_prob = final_model.predict_proba(X_test)[:, 1]

    # --- Walk-forward AUC summary (always printed) ---
    print("\nWalk-forward AUC by year (expanding window):", file=sys.stderr)
    all_years = sorted(spx["entry_year"].unique())
    train_years_prior = [y for y in all_years if y < args.test_year]
    for yr in all_years:
        yr_mask = spx["entry_year"] == yr
        yr_df = spx[yr_mask]
        if len(yr_df) < 10:
            continue
        X_yr = yr_df[feature_cols]
        y_yr = yr_df[target_col]
        if y_yr.nunique() < 2:
            print(f"  {yr}: AUC=N/A (single class)", file=sys.stderr)
            continue
        try:
            yr_prob = final_model.predict_proba(X_yr)[:, 1]
            yr_auc = roc_auc_score(y_yr, yr_prob)
            tag = " [test]" if yr >= args.test_year else ""
            print(f"  {yr}: AUC={yr_auc:.4f}{tag}", file=sys.stderr)
        except Exception as exc:
            print(f"  {yr}: AUC=ERROR ({exc})", file=sys.stderr)
    print("", file=sys.stderr)

    # Optionally find threshold: maximize balanced accuracy (better for imbalanced data)
    best_threshold = 0.5
    if args.tune_threshold:
        best_ba = 0.0
        best_acc = 0.0
        for t in np.arange(0.25, 0.76, 0.05):
            y_pred_t = (y_prob >= t).astype(int)
            ba = balanced_accuracy_score(y_test, y_pred_t)
            acc = accuracy_score(y_test, y_pred_t)
            if ba > best_ba:
                best_ba = ba
                best_acc = acc
                best_threshold = float(t)
        print(
            f"Tuned threshold: {best_threshold:.2f} "
            f"(balanced_acc={best_ba:.4f}, accuracy={best_acc:.4f})",
            file=sys.stderr,
        )
    y_pred = (y_prob >= best_threshold).astype(int)

    print("Test metrics:", file=sys.stderr)
    print(f"  Accuracy:  {accuracy_score(y_test, y_pred):.4f}", file=sys.stderr)
    print(f"  Balanced: {balanced_accuracy_score(y_test, y_pred):.4f}", file=sys.stderr)
    print(f"  Precision: {precision_score(y_test, y_pred, zero_division=0):.4f}", file=sys.stderr)
    print(f"  Recall:   {recall_score(y_test, y_pred, zero_division=0):.4f}", file=sys.stderr)
    print(f"  F1:       {f1_score(y_test, y_pred, zero_division=0):.4f}", file=sys.stderr)
    try:
        auc = roc_auc_score(y_test, y_prob)
        print(f"  ROC-AUC:  {auc:.4f}", file=sys.stderr)
    except Exception:
        print("  ROC-AUC:  N/A (single class?)", file=sys.stderr)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    model_path = args.output_dir / "entry_filter_xgboost_spx.joblib"
    payload = {
        "model": final_model,
        "feature_cols": feature_cols,
        "target": args.target,
        "target_name": args.target,   # both keys for backwards-compat with evaluate_enhanced_cashflow.py
        "best_threshold": best_threshold,
    }
    if config_encoder is not None:
        payload["config_encoder"] = config_encoder
    joblib.dump(payload, model_path)
    print(f"Saved model to {model_path}", file=sys.stderr)

    imp_source = model  # use base XGBoost for importances even when calibrated
    imp = pd.DataFrame({
        "feature": feature_cols,
        "importance": imp_source.feature_importances_,
    }).sort_values("importance", ascending=False)
    imp_path = args.output_dir / "entry_filter_xgboost_spx_importance.csv"
    imp.to_csv(imp_path, index=False)
    print(f"Saved importance to {imp_path}", file=sys.stderr)

    return 0


if __name__ == "__main__":
    sys.exit(main())
