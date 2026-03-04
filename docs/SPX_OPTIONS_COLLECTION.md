# SPX Options Data Collection

Script to generate data files for SPX options over a configurable window (30–90 days) for backtesting and GEX/pinning analysis.

## Get the data first

You need SPX options files in `data/spx_options/` before running the GEX 7DTE backtest or premium timing analysis.

**One-time (current snapshot):**

```bash
# From repo root – uses MASSIVE_API_KEY from data/config.json
python scripts/collect_spx_options.py --source massive
```

This writes `data/spx_options/spx_options_YYYY-MM-DD.parquet` for today. You can then run the backtest **without** `--start`/`--end` (or with a range that includes that date).

**Historical (e.g. all of 2025):** You can use **Massive flat files** (minute aggregates) to build daily snapshots:

- **Source:** Massive.com → File Browser → Options → Minute Aggregates (`us_options_opra/minute_aggs_v1`). Daily `.csv.gz` files with columns: `ticker`, `volume`, `open`, `close`, `high`, `low`, `window_start`, `transactions`. Plan recency/history depends on your Options plan (e.g. 2 years Starter, 4 years Developer, all history Advanced).
- **Script:** [scripts/download_massive_flatfiles.py](../scripts/download_massive_flatfiles.py) reads those files (from a local directory or a base URL), filters to SPX options (`O:SPX...`), aggregates to end-of-day (last close per contract), and writes `spx_options_YYYY-MM-DD.parquet` into `data/spx_options/`.

  ```bash
  # After downloading flat files for 2025 into a directory:
  python scripts/download_massive_flatfiles.py --start 2025-01-01 --end 2025-12-31 --input-dir /path/to/minute_aggs --output-dir data/spx_options -v
  ```

  If your plan provides a direct download URL template, use `--base-url` and ensure `MASSIVE_API_KEY` is set.

- Alternatively, run **collect_spx_options.py daily** (cron) so you accumulate one snapshot per day over time.

## Script

- **Path**: [scripts/collect_spx_options.py](../scripts/collect_spx_options.py)

## Data sources

1. **Massive (default)**  
   - Fetches the **current** SPX options snapshot (I:SPX) from the Massive API.  
   - Writes one file per run: `spx_options_YYYY-MM-DD.parquet` (and optionally CSV).  
   - Massive does not provide historical as-of-date snapshots. To build a **history** over 30–90 days, run the script **daily** (e.g. via cron); over time you accumulate one file per day.

2. **Tradier**  
   - Fetches options **by expiration**: all SPX expirations in `[today - days, today + 30]`.  
   - For each expiration, requests the chain from Tradier and combines into a single file: `spx_options_expirations_YYYY-MM-DD.parquet` (and optionally CSV).  
   - Gives one snapshot per expiration (not per calendar day), useful for structure (GEX by expiry, max pain, etc.).

3. **Massive flat files (minute aggregates)**  
   - Daily downloadable S3 files: `us_options_opra/minute_aggs_v1`, one `.csv.gz` per day (OHLCV per minute per option).  
   - Use [scripts/download_massive_flatfiles.py](../scripts/download_massive_flatfiles.py) with `--input-dir` (after downloading from Massive File Browser) or `--base-url` to produce `spx_options_YYYY-MM-DD.parquet` for a date range (e.g. all of 2025).  
   - Plan history: 2 years (Starter), 4 years (Developer), all history (Advanced). Updated at 11a ET with previous day.

## Usage

From the repo root:

```bash
# Current snapshot from Massive (default), write Parquet to data/spx_options
python scripts/collect_spx_options.py

# Same, but also write CSV
python scripts/collect_spx_options.py --format both

# By-expiration from Tradier, 90-day lookback, include past expirations
python scripts/collect_spx_options.py --source tradier --days 90

# Tradier, 30-day window, only future expirations
python scripts/collect_spx_options.py --source tradier --days 30 --no-expired

# Custom output directory
python scripts/collect_spx_options.py --output-dir /path/to/spx_options
```

### Options

| Option        | Default           | Description |
|---------------|-------------------|-------------|
| `--days`      | 90                | 30 or 90. For Massive: metadata/naming. For Tradier: expiration window `[today - days, today + 30]`. |
| `--output-dir`| `data/spx_options`| Directory for output files. |
| `--source`    | massive           | `massive` or `tradier`. |
| `--format`    | parquet           | `parquet`, `csv`, or `both`. |
| `--no-expired`| off               | Tradier only: exclude expirations that are already in the past. |

## Configuration

- **Massive**: `MASSIVE_API_KEY` in [data/config.json](../data/config.json) (same as GEX), or env `MASSIVE_API_KEY`.
- **Tradier**: `TRADIER_API_KEY` in [data/config.json](../data/config.json), or env `TRADIER_API_KEY`.

## Cron example (Massive, daily history)

Run once per day to accumulate daily snapshots:

```bash
# Every day at 18:05 ET (after regular session), collect SPX snapshot
5 18 * * 1-5 cd /path/to/MarketNews && . venv/bin/activate && python scripts/collect_spx_options.py --source massive
```

Adjust path and timezone to your environment. Over 30–90 days you will have 30–90 files under `data/spx_options/`.

## Output layout

- **Massive**: `{output_dir}/spx_options_YYYY-MM-DD.parquet` (and `.csv` if `--format csv` or `both`).
- **Tradier**: `{output_dir}/spx_options_expirations_YYYY-MM-DD.parquet` (and `.csv` if requested).

## Normalized schema

All outputs use the same columns for GEX/backtest compatibility:

- `underlying` (SPX)
- `strike`
- `expiration_date`
- `contract_type` (call / put)
- `open_interest`
- `implied_volatility`
- `snapshot_date`
- `option_symbol` (when available)
- `shares_per_contract` (default 100 for SPX)
- `bid`, `ask`, `mid` (optional; when available from API for realistic backtest pricing; Massive provides via `last_quote`, Tradier when returned in chain)

Dates are stored as date type in Parquet and YYYY-MM-DD in CSV. Downstream you can build a Massive-style “snap” from these rows and run [QuantEngine/gex_calculator.py](../QuantEngine/gex_calculator.py) (e.g. `calculate_gex`, `compute_cockpit_state`) for historical GEX and pinning.

## GEX 7DTE backtest

The script [scripts/backtest_gex_7dte.py](../scripts/backtest_gex_7dte.py) uses this dataset to backtest a vertical put spread: when GEX is solidly positive, sell a 7-**trading-session** DTE ~20-delta put spread; P&L uses chain mids (or IV fallback), slippage/commissions, and stop-loss/settlement. The backtest uses **trading days only** (NYSE calendar via `pandas_market_calendars`) and **real expirations from the chain** (no synthetic Fridays). For a day-by-day backtest with many trades, daily Massive snapshots are required (run the collector daily via cron).

**Standard backtesting library:** Pass `--vectorbt` to report stats via [VectorBT](https://vectorbt.dev/) (same library used in `market_news_app/backtesting_v2`). Trades are converted to a daily PnL series → equity curve → daily returns; metrics (Sharpe, Sortino, drawdown, etc.) come from VectorBT’s returns accessor (`.vbt.returns(freq='d').stats()`) so the GEX 7DTE strategy is evaluated the same way as the rest of the app. Use `--initial-capital` for sensible return percentages. Requires `pip install vectorbt`.
