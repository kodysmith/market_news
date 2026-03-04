# Multi-asset cashflow pipeline

The pipeline runs the enhanced cashflow backtest (iron condors, entry/exit rules, filters) for one or more assets and writes HTML reports so you can compare strategies across symbols.

## Usage

```bash
# Single asset (default date range 2015-01-01 to 2025-12-31)
python3 scripts/run_cashflow_pipeline.py --symbol GLD

# Multiple assets, custom range
python3 scripts/run_cashflow_pipeline.py --symbols SPX,XSP,GLD,SLV --start 2020-01-01 --end 2024-12-31

# Save trades CSV per asset
python3 scripts/run_cashflow_pipeline.py --symbols SPX,GLD --output-dir data --save-trades
```

## Arguments

| Argument | Description |
|----------|-------------|
| `--symbol` | Single asset (SPX, XSP, GLD, SLV). |
| `--symbols` | Comma-separated list; overrides `--symbol` if set. |
| `--start` | Start date (YYYY-MM-DD). Default: 2015-01-01. |
| `--end` | End date (YYYY-MM-DD). Default: 2025-12-31. |
| `--output-dir` | Directory for HTML reports. Default: `data/`. |
| `--save-trades` | Write trades CSV per asset to `output-dir` as `enhanced_cashflow_trades_{symbol}.csv`. |

## Output

- **Per asset**: `{output_dir}/enhanced_cashflow_results_{symbol}.html` — same structure as the single-asset enhanced cashflow report (equity curves, Sharpe/Sortino, drawdowns, heatmaps, summary table).
- **Index** (when multiple symbols): `{output_dir}/index.html` — simple list of links to each asset report.

## Data and behavior per asset

- **SPX**: Price from ^GSPC, IV from VIX. GEX regime proxy (VIX term structure + RV + momentum) is computed. ES hedge (gap-down and weekend) is enabled.
- **XSP**: Price from XSP, IV from VIX (same index). No GEX proxy (pass-all). ES hedge enabled.
- **GLD / SLV**: Price from yfinance. IV is 20-day realized volatility (annualized, as percent). No GEX proxy. ES hedge disabled (not index products).

GLD and SLV reports use realized vol for Black–Scholes and for “VIX-style” filters (e.g. vix_max); there is no futures hedge.

## See also

- [scripts/evaluate_enhanced_cashflow.py](../scripts/evaluate_enhanced_cashflow.py) — standalone SPX evaluator (same strategies, single asset).
- [scripts/asset_data.py](../scripts/asset_data.py) — symbol→ticker map and `load_asset_data(symbol, start, end)`.
