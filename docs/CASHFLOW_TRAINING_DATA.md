# Cashflow training data

The dataset produced by `scripts/export_cashflow_training_data.py` is a **labeled table** of backtest trades (one row per trade) for use in classification, regression, or sequence models. Architecture-agnostic: use with pandas, sklearn, PyTorch, XGBoost, etc.

## Generating the dataset

```bash
# Default: SPX, 2015–2025, CSV to data/cashflow_training_data.csv
python scripts/export_cashflow_training_data.py

# Multi-asset, Parquet
python scripts/export_cashflow_training_data.py --symbols SPX,XSP,GLD,SLV --output data/cashflow_training.parquet --format parquet

# Custom range
python scripts/export_cashflow_training_data.py --start 2020-01-01 --end 2024-12-31 --output data/train_2020_2024.csv
```

## Schema: features vs labels

### Feature columns (inputs)

| Column | Type | Description |
|--------|------|-------------|
| `entry_date` | date | Calendar date of entry. |
| `spot_entry` | float | Underlying price at entry. |
| `vix_at_entry` | float | VIX (or IV proxy) at entry (e.g. percent). |
| `gap_pct_at_entry` | float | Pre-market gap % at entry (open vs prior close). |
| `gex_score_at_entry` | int | GEX regime proxy at entry (0–3; SPX only has real GEX). |
| `rv_ratio_at_entry` | float | Short-term / medium-term realized vol ratio at entry. |
| `entry_year` | int | Year of entry (derived). |
| `entry_month` | int | Month of entry (1–12). |
| `entry_day_of_week` | int | Day of week (0=Mon, 4=Fri). |
| `config_id` | str | Strategy config id (e.g. 02_IC21_CLOSE_1DTE). |
| `strategy` | str | Strategy type (e.g. iron_condor). |
| `dte` | int | Days to expiration at entry (trading days). |
| `short_delta` | float | Short option delta used at entry. |
| `width` | float | Spread width (points). |
| `profit_target` | float | Take-profit fraction of credit. |
| `stop_mult` | float | Stop-loss multiple of credit. |
| `iv_entry` | float | IV used for pricing at entry. |
| `entry_credit` | float | Net credit (after slippage) at entry. |
| `symbol` | str | Asset (SPX, XSP, GLD, SLV); present when multiple symbols exported. |

#### Rolling features (SPX only; lookback-only, `_lb` suffix)

When exporting with the default script, **SPX** rows are enriched with rolling/window features from the same backtest OHLC/series. All rolling features are **lookback-only** (data up to and including entry date); names use the `_lb` suffix to avoid leakage. Other symbols get NaN for these columns.

| Column | Type | Description |
|--------|------|-------------|
| `ret_5d_lb` | float | Log return over 5 trading days ending on entry date. |
| `ret_10d_lb` | float | Log return over 10 trading days. |
| `ret_21d_lb` | float | Log return over 21 trading days. |
| `vix_5d_mean_lb` | float | Mean VIX over 5 trading days. |
| `vix_5d_max_lb` | float | Max VIX over 5 trading days. |
| `vix_21d_mean_lb` | float | Mean VIX over 21 trading days. |
| `vix_21d_max_lb` | float | Max VIX over 21 trading days. |
| `gap_5d_mean_lb` | float | Mean pre-market gap % over 5 days. |
| `gap_5d_std_lb` | float | Std of gap % over 5 days. |
| `rv_ratio_5d_mean_lb` | float | Mean short/medium realized-vol ratio over 5 days. |
| `spot_vs_sma10_lb` | float | Close at entry / 10-day SMA of close. |
| `spot_vs_sma21_lb` | float | Close at entry / 21-day SMA of close. |

#### Extended features (SPX only; vol trend, term structure, regime, interactions)

SPX rows also get the following when VIX9D/VIX3M and sufficient history are available:

| Column | Type | Description |
|--------|------|-------------|
| `vix_change_1d` | float | VIX % change over 1 day (lookback). |
| `vix_change_5d` | float | VIX % change over 5 days (lookback). |
| `vix_percentile_252d` | float | VIX level percentile over trailing 252 trading days. |
| `rv_5`, `rv_10`, `rv_20` | float | Annualized realized vol (5/10/20d, %). |
| `rv_trend` | float | rv_5 − rv_20 (short vs long vol). |
| `iv_rv_spread` | float | iv_entry − rv_20 at entry. |
| `vix9d_vix_ratio` | float | VIX9D/VIX at entry. |
| `term_structure_contango` | int | 1 if VIX < VIX3M at entry, else 0. |
| `spot_vs_sma50`, `spot_vs_sma200` | float | Close / 50- or 200-day SMA. |
| `atr_pct_14` | float | 14-day ATR as % of close. |
| `gap_abs_z` | float | Gap % at entry standardized vs last 20 gaps. |
| `gap_direction` | float | Sign of gap (+1 / −1 / 0). |
| `regime` | int | Deterministic regime 0–3 (calm / elevated / spiking / stress). |
| `vix_at_entry_x_rv_ratio_at_entry` | float | Interaction. |
| `gex_score_at_entry_x_vix_change_5d` | float | Interaction. |
| `spot_vs_sma21_lb_x_vix_change_5d` | float | Interaction. |

### Label columns (targets)

| Column | Type | Description |
|--------|------|-------------|
| `pnl` | float | Realized P&L of the trade ($). |
| `win` | bool | True if pnl > 0. |
| `tail_1x` | int | 1 if pnl ≤ −1× entry_credit, else 0 (tail loss). |
| `tail_1_5x` | int | 1 if pnl ≤ −1.5× entry_credit, else 0. |
| `bad_q10` | int | 1 if pnl ≤ config’s 10% quantile of pnl, else 0. |
| `exit_reason` | str | How the trade closed (e.g. expiry, tp_60, sl_2, vix_spike_…). |
| `exit_date` | date | Date of exit. |
| `exp_date` | date | Option expiration date. |

You can train on `loss`, `big_loss`, `tail_1x`, `tail_1_5x`, or `bad_q10`; `pnl` and `exit_reason` remain for analysis.

## Usage notes

- **Filtering by strategy**: Use `config_id` or `strategy` to train on a subset (e.g. only 02_IC21_CLOSE_1DTE or only iron_condor).
- **Multi-asset**: When `symbol` is present, you can include it as a categorical feature or split by symbol for separate models.
- **Time series**: Sort by `entry_date` (and optionally `config_id` or `symbol`) to build sequences for RNN/Transformer-style models.
- **Simple baseline**: e.g. predict `win` from `vix_at_entry` and `gap_pct_at_entry` with a logistic regression or small tree to establish a baseline before more complex models.

## Related

- [scripts/evaluate_enhanced_cashflow.py](../scripts/evaluate_enhanced_cashflow.py) — backtest and strategy definitions.
- [scripts/run_cashflow_pipeline.py](../scripts/run_cashflow_pipeline.py) — multi-asset HTML reports.
- [CASHFLOW_PIPELINE.md](CASHFLOW_PIPELINE.md) — pipeline usage.
- [ENTRY_FILTER_MODEL.md](ENTRY_FILTER_MODEL.md) — XGBoost entry-risk model (P(loss)) and how to train and use it as an entry filter.
