# XGBoost entry filter model (SPX)

The entry filter is an **XGBoost binary classifier** trained to predict P(loss) (or P(big loss)) from entry-day and rolling features. It is used to **skip high-risk entries** in the cashflow backtest to reduce drawdowns (“do not lose money”).

## Goal: reduce drawdown, keep income high

- **Train for big losses**: Use `--target big_loss --loss-threshold 500` (or 1000) so the model predicts *large* drawdowns, not every small loss. You then skip only when P(big loss) is high, keeping most trades and income.
- **Sweep the threshold**: Run `scripts/sweep_entry_filter_threshold.py` to backtest at several P(loss) thresholds (e.g. no filter, 0.5, 0.6, 0.7, 0.8). The table shows total PnL, max drawdown, trade count, and Calmar. Pick a threshold where income stays high (e.g. ≥85% of baseline PnL) and max drawdown drops; the script suggests one (best Calmar among those keeping ≥85% of baseline PnL).
- **Use in production**: Run the pipeline or backtest with `entry_filter_model_path` and `entry_filter_threshold` set to your chosen value.

## Target variable

- **`loss`** (default): binary target = `(win == False)` (any losing trade).
- **`big_loss`**: binary target = `(pnl < -X)` with configurable threshold (e.g. -500). Use to **reduce drawdown while keeping income high** — skip only when P(big loss) is high.
- **`tail_1x`**: 1 if pnl ≤ −1× entry_credit, else 0 (predict tail damage).
- **`tail_1_5x`**: 1 if pnl ≤ −1.5× entry_credit, else 0.
- **`bad_q10`**: 1 if pnl ≤ that config’s 10% quantile of pnl, else 0 (worst 10% by config).

## Features

The model uses the same features as in the enriched training data. **Lookback-only** rolling features use the `_lb` suffix (leakage-safe naming). Extended features include vol trend, term structure, regime, and interactions.

- **Entry-day**: `spot_entry`, `vix_at_entry`, `gap_pct_at_entry`, `gex_score_at_entry`, `rv_ratio_at_entry`, `entry_year`, `entry_month`, `entry_day_of_week`, `dte`, `short_delta`, `width`, `profit_target`, `stop_mult`, `iv_entry`, `entry_credit`.
- **Rolling (lookback, _lb)**: `ret_5d_lb`, `ret_10d_lb`, `ret_21d_lb`, `vix_5d_mean_lb`, `vix_5d_max_lb`, `vix_21d_mean_lb`, `vix_21d_max_lb`, `gap_5d_mean_lb`, `gap_5d_std_lb`, `rv_ratio_5d_mean_lb`, `spot_vs_sma10_lb`, `spot_vs_sma21_lb`.
- **Extended**: `vix_change_1d`, `vix_change_5d`, `vix_percentile_252d`, `rv_5`, `rv_10`, `rv_20`, `rv_trend`, `iv_rv_spread`, `vix9d_vix_ratio`, `term_structure_contango`, `spot_vs_sma50`, `spot_vs_sma200`, `atr_pct_14`, `gap_abs_z`, `gap_direction`, `regime` (0–3), and interactions (`vix_at_entry_x_rv_ratio_at_entry`, `gex_score_at_entry_x_vix_change_5d`, `spot_vs_sma21_lb_x_vix_change_5d`).

Use **`--no-entry-year`** to exclude `entry_year` from the production feature set (avoids year leakage). Feature list is saved with the model and used at prediction time so backtest and training stay aligned.

## Generating enriched training data

Export includes rolling features for SPX when you run:

```bash
python scripts/export_cashflow_training_data.py
# Or with custom range/output:
python scripts/export_cashflow_training_data.py --start 2015-01-01 --end 2024-12-31 --output data/cashflow_training_data.csv
```

The export script loads price/IV with a 30-day padded start so rolling windows have history. SPX rows get the rolling columns; other symbols get NaN. See [CASHFLOW_TRAINING_DATA.md](CASHFLOW_TRAINING_DATA.md) for the full schema.

## Training the model

```bash
# Default: target=loss, test set = 2024+
python scripts/train_entry_filter_xgboost.py --input data/cashflow_training_data.csv

# Big-loss target with $500 threshold
python scripts/train_entry_filter_xgboost.py --target big_loss --loss-threshold 500

# Tail-risk target (skip when P(tail_1x) is high)
python scripts/train_entry_filter_xgboost.py --target tail_1x --no-entry-year

# Custom test split (last 20% by time)
python scripts/train_entry_filter_xgboost.py --test-frac 0.2 --output-dir data
```

**Outputs:**

- **Model**: `data/entry_filter_xgboost_spx.joblib` (contains `model`, `feature_cols`, `target`).
- **Feature importance**: `data/entry_filter_xgboost_spx_importance.csv`.

The script filters to SPX only, drops rows with NaN in feature columns, and uses a time-based train/test split by default.

## Using the model as an entry filter in the backtest

When running the backtest, pass the saved model path and a P(loss) threshold. Trades with **P(loss) > threshold** are skipped.

**From code** (e.g. custom script or notebook):

```python
from datetime import date
from pathlib import Path
from scripts.asset_data import load_asset_data
from scripts.evaluate_enhanced_cashflow import get_backtest_trades, compute_gex_proxy

start = date(2020, 1, 1)
end = date(2024, 12, 31)
price_df, iv_series = load_asset_data("SPX", start, end)
gex_scores = compute_gex_proxy(price_df, iv_series, start, end)

# Unfiltered
trades = get_backtest_trades(price_df, iv_series, start, end, gex_scores=gex_scores, enable_es_hedge=True)

# Filtered: skip when P(loss) > 0.5
model_path = Path("data/entry_filter_xgboost_spx.joblib")
trades_filtered = get_backtest_trades(
    price_df, iv_series, start, end,
    gex_scores=gex_scores, enable_es_hedge=True,
    entry_filter_model_path=model_path,
    entry_filter_threshold=0.5,
)
# Compare: len(trades) vs len(trades_filtered), drawdown, win rate, etc.
```

**Implementation detail:** `run_filtered_backtest` in [scripts/evaluate_enhanced_cashflow.py](../scripts/evaluate_enhanced_cashflow.py) accepts `entry_filter_model_path` and `entry_filter_threshold`. It loads the model once, then at each candidate entry builds the same feature vector (entry-day + lookback rolling + extended vol/term structure/regime/interactions via `compute_entry_features_extended` when VIX9D/VIX3M are available) and skips the trade if the model’s P(loss) exceeds the threshold. Use **`--entry-filter-log path/to/log.csv`** (evaluator CLI) to append one row per candidate entry (entry_date, config_id, p_loss, decision, key features) for auditing.

## See comparison in the backtest report

Run the SPX backtest with the entry filter to get a **baseline vs filtered** comparison in the same HTML report:

- **Evaluator:** `python scripts/evaluate_enhanced_cashflow.py --entry-filter-model data/entry_filter_xgboost_spx.joblib --entry-filter-threshold 0.5` — report: `data/enhanced_cashflow_report.html` (comparison table and portfolio baseline vs filtered chart).
- **Pipeline:** `python scripts/run_cashflow_pipeline.py --symbol SPX --entry-filter-model data/entry_filter_xgboost_spx.joblib --entry-filter-threshold 0.5` — report: `data/enhanced_cashflow_results_SPX.html`.

## Threshold sweep (income vs drawdown)

```bash
python scripts/sweep_entry_filter_threshold.py
# Custom range and model:
python scripts/sweep_entry_filter_threshold.py --start 2015-01-01 --end 2024-12-31 --model data/entry_filter_xgboost_spx.joblib
```

Prints a table: for each threshold (no filter, 0.5, 0.55, 0.6, …), total PnL, % of baseline PnL, max drawdown, % of baseline DD, trade count, win rate, Calmar. Use it to choose a threshold that keeps income high and cuts drawdown.

## Time-chunk experiments (overfitting check)

To test whether the model generalizes across time (and to avoid overfitting to one period), run train/test in chunks:

```bash
python scripts/train_entry_filter_chunks.py --mode all --no-scale-pos-weight --output-csv data/entry_filter_chunk_results.csv
```

**Modes:**

- **forward_expanding**: Train on 2015–Y1, test on Y2–Y3; then expand train to 2015–Y2, test on Y4–Y5; etc. Checks that the model trained on past data holds up on future data.
- **forward_rolling**: Fixed-length train windows (e.g. 2015–2017, test 2018–2019; 2017–2019, test 2020–2021). Tests stability across rolling windows.
- **backward**: Train on recent years (e.g. 2020–2025), test on older (2015–2019). If test metrics drop a lot, the model is regime-dependent.

The script prints a table (and optionally writes CSV) with train_n, test_n, test accuracy, balanced accuracy, AUC, F1, precision, recall per window. Use it to see if performance is stable across time or if one period is overfit.

## Dependencies

- `xgboost`, `scikit-learn`, `joblib` (see [requirements.txt](../requirements.txt)).

## Related

- [CASHFLOW_TRAINING_DATA.md](CASHFLOW_TRAINING_DATA.md) — training data schema and rolling columns.
- [scripts/export_cashflow_training_data.py](../scripts/export_cashflow_training_data.py) — export with SPX rolling enrichment.
- [scripts/train_entry_filter_xgboost.py](../scripts/train_entry_filter_xgboost.py) — train script.
- [scripts/evaluate_enhanced_cashflow.py](../scripts/evaluate_enhanced_cashflow.py) — backtest and `run_filtered_backtest` / `get_backtest_trades`.
- [scripts/sweep_entry_filter_threshold.py](../scripts/sweep_entry_filter_threshold.py) — sweep thresholds to balance income vs drawdown.
- [scripts/train_entry_filter_chunks.py](../scripts/train_entry_filter_chunks.py) — time-chunk experiments (forward/backward) to check overfitting.
