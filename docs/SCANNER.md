# Premarket Gap Scanner

Daily premarket gap scanner that ranks small/mid-cap candidates (gap + relative volume + liquidity) and feeds the **Trade Ideas** page (Premarket tab).

## Setup

1. **Supabase**: Run [scripts/scanner_schema.sql](../scripts/scanner_schema.sql) in the Supabase SQL editor to create `scan_runs`, `scanner_trade_ideas`, and `symbol_baselines`.

2. **Env**: Set `SUPABASE_URL` and `SUPABASE_SECRET_KEY` (or `SUPABASE_SERVICE_ROLE_KEY`) in `.env` so the runner and API can read/write scanner tables.

3. **Config** (`data/config.json` or env):
   - `SCANNER_UNIVERSE_LIMIT` – max symbols to scan per run (default 500).
   - `SCANNER_MIN_GAP` – minimum gap fraction (e.g. 0.05 = 5%).
   - `SCANNER_TOP_N` – number of ideas to store per run (default 25).
   - `SCANNER_MIN_PRICE` / `SCANNER_MAX_PRICE` – price filter (default 1–30).
   - `FMP_API_KEY`, `ALPHAVANTAGE_API_KEY` – for prior close and fallback quotes.
   - `SCANNER_USE_IBKR` – set to `false` to skip IBKR and use FMP/AV/yfinance only.
   - `IBKR_HOST`, `IBKR_PORT`, `IBKR_CLIENT_ID` – used when IBKR is enabled.

4. **Universe**: Either add `data/scanner_universe.txt` (one symbol per line) or use `data/sec_universe.json` (from `scripts/fetch_sec_universe.py`).

## Running the scanner

```bash
# From repo root (venv activated)
python scripts/run_premarket_scanner.py

# Limit symbols (e.g. for testing)
python scripts/run_premarket_scanner.py --limit 100

# Dry run (no DB write)
python scripts/run_premarket_scanner.py --dry-run
```

## API

- `GET /trade-ideas/scanner/today` – today’s ranked ideas (payloads), plus `stale_data`, `as_of`.
- `GET /trade-ideas/scanner/<date>` – ideas for date (YYYY-MM-DD).
- `GET /trade-ideas/scanner/today/<symbol>` – single-idea detail for symbol.

API reads from Supabase; no Redis in V1.

## Cron (premarket)

Run during premarket (e.g. 07:00–09:25 ET every 5 min):

```cron
*/5 7-9 * * 1-5 cd /path/to/MarketNews && . venv/bin/activate && python3 scripts/run_premarket_scanner.py >> logs/scanner.log 2>&1
```

Or a single run at 09:25 ET:

```cron
25 9 * * 1-5 cd /path/to/MarketNews && . venv/bin/activate && python3 scripts/run_premarket_scanner.py >> logs/scanner.log 2>&1
```

## Flutter

Trade Ideas screen has two tabs: **Options** (ticker-scoped options ideas) and **Premarket**. The Premarket tab calls `GET /trade-ideas/scanner/today` and shows the ranked list with score, gap %, rel vol, and reasons. Label: “idea generator, not guaranteed PnL.”

## Historical scanner and training data

To generate a **short list of assets per day** for backtesting (and later ML training), run the scanner logic on **historical** dates using the same Polygon cached data as the Warrior backtest.

### 1. Run historical scanner

Produces a CSV of (date, symbol, gap_pct, rel_vol, total_score, rank) for each day in range. Uses gap + relvol filters and scanner-style scoring (gap_score + relvol_score + liquidity_score).

**Data source** (`--source`): `polygon` (default), `yfinance`, or `ibkr`. Use `yfinance` or `ibkr` if Polygon returns 403 (past historical entitlements).

```bash
# Polygon (requires POLYGON_API_KEY; may 403 for old dates)
python scripts/run_historical_scanner.py --start 2022-01-01 --end 2024-12-31 \
  --output data/scanner_candidates.csv --top-n 50 --gap 0.05 --relvol 3.0 --scan-all

# yfinance (free, no API key; requires symbol list)
python scripts/run_historical_scanner.py --start 2022-01-01 --end 2024-12-31 \
  --source yfinance --symbols-csv data/sec_universe.json --output data/scanner_candidates.csv --top-n 50

# IBKR (TWS or Gateway must be running; requires symbol list)
python scripts/run_historical_scanner.py --start 2024-01-01 --end 2024-12-31 \
  --source ibkr --symbols symbols.txt --output data/scanner_candidates.csv --ibkr-port 7497
```

Options: `--gap`, `--relvol`, `--relvol-lookback-days`, `--top-n`, `--min-price`, `--max-price`, `--symbols` / `--symbols-csv` (or `--scan-all` for polygon only).

### 2. Backtest only those candidates

Run the Warrior gap+pullback backtest restricted to the scanner list. This fetches 1-min bars only for symbols that passed the historical scanner each day.

```bash
python scripts/backtest_warrior_gap_pullback.py --start 2022-01-01 --end 2024-12-31 \
  --scanner-candidates data/scanner_candidates.csv \
  --output-dir data/backtest_scanner
```

Output: `trades.csv` (and `summary.txt`) with one row per trade: symbol, day, gap_pct, rel_vol, entry, stop, exit, outcome, pnl_r, etc.

### 3. Use trades as ML training data

`trades.csv` is your training dataset: each row has **features** (gap_pct, rel_vol, symbol, day) and **labels** (outcome, pnl_r). You can add a binary target (e.g. `win = (outcome == 'TP')` or `profitable = (pnl_r > 0)`) and train a model to predict outcome or PnL from scanner-style features. Optionally join back to `scanner_candidates.csv` for extra columns (total_score, prev_day_dollar_vol, rank).

**Pipeline summary:** Historical scanner → `scanner_candidates.csv` → Backtest with `--scanner-candidates` → `trades.csv` → ML training.

## V2 / V3

- V2: Massive options for top 200 candidates; options_score and reasons; suggested levels (pm_high, pm_low).
- V3: News/catalyst, offering detection, link to Warrior gap backtest.
