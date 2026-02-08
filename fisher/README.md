# Fisher Company Valuation Pipeline

Automated Fisher score (Philip Fisher’s “Common Stocks and Uncommon Profits”) from SEC EDGAR, market data, and optional transcripts. Data lives in Postgres (Supabase); the app exposes scores via `/fisher/*` and the Flutter Fisher screen.

## Prerequisites

- Python 3.10+
- Postgres (Supabase or local): run `supabase_schema_fisher.sql` after the main schema.
- Env: `DATABASE_URL` or `SUPABASE_DB_URL` (Postgres connection string).
- S&P 500 list: `data/sp500_constituents.json` (constituents with `ticker`, `name`, `sector`, `industry`, `cik`).

### Local Postgres (testing)

To use a local Postgres in Docker instead of Supabase:

```bash
./scripts/setup_local_postgres.sh
export DATABASE_URL="postgresql://postgres:postgres@localhost:5433/marketnews"
```

Then run the pipeline (e.g. limit 5 companies for a quick test): see “How to run jobs” below.

## Package layout

- **fisher/config.py** – Config paths, SEC user-agent, DB URL, `get_sp500_constituents()`.
- **fisher/db.py** – Postgres connection, `company_id_by_ticker`, `ensure_company`.
- **fisher/edgar_watcher.py** – EDGAR submissions watcher; ensures `fisher_filing` for 10-K/10-Q; calls XBRL parser.
- **fisher/xbrl_parser.py** – SEC Company Facts API → `fisher_financial_fact` (canonical fact names).
- **fisher/market_updater.py** – Yahoo daily OHLCV → `fisher_market_bar_daily`; peer sets by sector → `fisher_peer_set`.
- **fisher/scoring_engine.py** – Feature extraction and Fisher point scoring (MVP-1: points 1, 3, 5, 6, 9, 13, 14, partial 15) → `fisher_score_snapshot`.

## How to run jobs

### 1. EDGAR + Company Facts (filings and financial facts)

From repo root:

```bash
python -c "
from fisher.db import get_connection
from fisher.edgar_watcher import run_edgar_watcher
with get_connection() as conn:
    run_edgar_watcher(conn)
"
```

**Cadence:** Every 1–6 hours (cron or scheduler). Rate-limited to SEC guidelines (User-Agent required).

### 2. Market data (daily bars + peer sets)

```bash
python -c "
from fisher.market_updater import run_market_updater
result = run_market_updater()  # optional: days=365, update_peers=True
print(result)  # {'bars': N, 'peer_sets': M}
"
```

**Cadence:** Daily after market close.

### 3. Scoring (snapshots)

```bash
python -c "
from fisher.scoring_engine import run_scoring_job
n = run_scoring_job()  # optional: company_ids=[...]
print('Snapshots written:', n)
"
```

**Cadence:** On every new 10-K/10-Q ingested, or nightly over companies with new filings.

## Pipeline order

1. Ensure schema is applied and `data/sp500_constituents.json` exists.
2. Run **EDGAR watcher** so `fisher_company`, `fisher_filing`, and `fisher_financial_fact` are populated.
3. Run **market updater** for `fisher_market_bar_daily` and `fisher_peer_set`.
4. Run **scoring job** to write `fisher_score_snapshot`.

After that, the API (`GET /fisher/snapshot`, `/fisher/delta`, `/fisher/evidence`, `/fisher/universe`) and Flutter Fisher screen will show data for the universe.

## Running for a single name or set of names

- **Script (recommended):**  
  `python3 scripts/run_fisher_for_tickers.py AAPL`  
  `python3 scripts/run_fisher_for_tickers.py AAPL MSFT GOOGL`  
  Optional: `--universe sec` (default) or `--universe sp500`. Tickers must exist in the chosen universe list (`data/sp500_constituents.json` or `data/sec_universe.json`).

- **One-liners (EDGAR only, then scoring):**  
  ```bash
  python3 -c "from fisher.edgar_watcher import run_watcher; print(run_watcher(tickers=['AAPL']))"
  python3 -c "
  from fisher.db import get_connection, company_id_by_ticker
  from fisher.scoring_engine import run_scoring_job
  with get_connection() as c:
      ids = [company_id_by_ticker(c, t) for t in ['AAPL','MSFT']]
  print(run_scoring_job(company_ids=[i for i in ids if i]))
  "
  ```

## High-growth universe (Yahoo / Alpha Vantage) → Fisher

To get a list of top high-growth + profitable companies from Yahoo (or Alpha Vantage) and then run Fisher + growth tooling on them:

1. **Build high-growth list (Yahoo, no API key):**  
   `python3 scripts/fetch_high_growth_universe.py --top 100 --source yahoo`  
   Scans S&P 500 (or `--universe sec` after `fetch_sec_universe.py`), computes revenue YoY and profit YoY, keeps profitable only, ranks by revenue growth, writes `data/high_growth_universe.json`. Options: `--limit-scan 300`, `--min-revenue-growth 10`, `--source alphavantage` (set `ALPHAVANTAGE_API_KEY`; free tier ~25 req/day).

2. **Run Fisher on that list:**  
   `python3 -c "from fisher.edgar_watcher import run_watcher; print(run_watcher(universe='high_growth'))"`  
   Then: `python3 -c "from fisher.scoring_engine import run_scoring_job; print(run_scoring_job())"`

3. **Scan for high growth + profitable (Fisher categories):**  
   `python3 scripts/scan_growth_profitable.py -n 30`

## Full-market scan (queue, monthly or manual)

Scan every stock in the SEC universe with the Fisher pipeline. Run on a local server; data is written to the same Fisher database.

**Prerequisites:** `data/sec_universe.json` (run `scripts/fetch_sec_universe.py`), `DATABASE_URL` set, and the queue table (included in `supabase_schema_fisher.sql`; for existing DBs run `scripts/fisher_scan_queue_schema.sql`).

1. **Enqueue** (queue all SEC tickers):  
   `python3 scripts/enqueue_fisher_scan.py`  
   For a **monthly full rescan**, clear and re-enqueue:  
   `python3 scripts/enqueue_fisher_scan.py --reset`

2. **Worker** (process queue in batches):  
   `python3 scripts/run_fisher_scan_worker.py`  
   Runs until the queue is empty. Options:  
   - `--batch 50` (default; tickers per batch; respect SEC rate limits)  
   - `--once` — process one batch and exit (for cron)  
   - `--delay 60` — seconds to sleep after each batch  

**Monthly (cron):**  
- First of month: `enqueue_fisher_scan.py --reset` then start the worker (e.g. in a screen/tmux session or as a service).  
- Or run the worker with `--once` and schedule it every hour: `0 * * * * ... run_fisher_scan_worker.py --once --batch 50`

**Manual:** Run enqueue (without `--reset` to add only new tickers), then run the worker in the foreground or background.

## Finding high scorers (hidden gems)

- **Top scorers (among already-scored companies):**  
  `python3 scripts/top_fisher_scores.py`  
  Options: `-n 20` (top 20), `--min-score 7.0` (only 7+), `--limit 200`.

- **Broader universe (beyond S&P 500):**  
  1. Fetch SEC list: `python3 scripts/fetch_sec_universe.py` → writes `data/sec_universe.json`.  
  2. Run EDGAR on that list:  
     `python3 -c "from fisher.edgar_watcher import run_watcher; print(run_watcher(universe='sec', limit=500))"`  
  3. Run scoring: `python3 -c "from fisher.scoring_engine import run_scoring_job; print(run_scoring_job())"`  
  4. List top scorers: `python3 scripts/top_fisher_scores.py -n 30 --min-score 7`.

## Config

- **SEC User-Agent:** Set `SEC_USER_AGENT` in env or use default in `fisher/config.py`.
- **Database:** `DATABASE_URL` or `SUPABASE_DB_URL`.
- **S&P 500 list:** Edit `data/sp500_constituents.json` or refresh from a public source (e.g. Wikipedia, SPDR); see plan for optional `fisher/scripts/refresh_cik.py`.

## API

See [docs/FISHER_API.md](../docs/FISHER_API.md) for the `/fisher/*` request/response contract.
