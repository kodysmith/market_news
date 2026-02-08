# Fisher Full-Market Scan

Scan every stock in the SEC universe with the Fisher scoring pipeline. Designed for monthly runs or manual triggers on a local server; results are written to the same Fisher Postgres database used by the app.

## Clone and run setup (recommended)

On a **new server**, clone the repo, add `.env` with `DATABASE_URL`, then run the single setup script. Cron will run the worker hourly and enqueue monthly so the scan runs continuously.

```bash
git clone <repo-url> MarketNews && cd MarketNews
# Add database URL (required)
echo 'DATABASE_URL=postgresql://user:pass@host:5432/marketnews' >> .env
# Or: cp env_template.txt .env && edit .env

./setup.sh
```

`setup.sh` will:

1. Create a Python venv (`.venv`) and install `requirements.txt`
2. Fetch SEC universe, create `fisher_scan_queue` table, and enqueue all SEC tickers
3. Install cron: **hourly** worker (one batch) and **monthly** (1st) enqueue reset

After setup, the scan runs automatically. Logs: `logs/fisher_worker.log`, `logs/fisher_enqueue.log`. To run the worker manually: `.venv/bin/python3 scripts/run_fisher_scan_worker.py --batch 50`.

## One-time setup (manual)

If you prefer not to use cron or already have a server:

From repo root with `DATABASE_URL` set:

```bash
export DATABASE_URL="postgresql://user:pass@host:5433/marketnews"
python3 scripts/setup_fisher_full_scan.py
```

This will: fetch the SEC universe (`data/sec_universe.json`), create the `fisher_scan_queue` table if missing, and print the worker command. To also enqueue all tickers:

```bash
python3 scripts/setup_fisher_full_scan.py --reset   # clear queue and enqueue all
# or
python3 scripts/setup_fisher_full_scan.py --enqueue # add only tickers not in queue
```

Then run the worker (see Commands below).

## Prerequisites

- **SEC universe:** Run `python3 scripts/fetch_sec_universe.py` (or use `setup_fisher_full_scan.py` above) to create `data/sec_universe.json`.
- **Database:** `DATABASE_URL` (or `SUPABASE_DB_URL`) set. The queue table is created by `setup_fisher_full_scan.py` or:
  - New installs: `supabase_schema_fisher.sql` already includes `fisher_scan_queue`.
  - Existing DBs: run `scripts/fisher_scan_queue_schema.sql` once.

## Workflow

```mermaid
flowchart LR
  A[fetch_sec_universe.py] --> B[sec_universe.json]
  B --> C[enqueue_fisher_scan.py]
  C --> D[fisher_scan_queue]
  D --> E[run_fisher_scan_worker.py]
  E --> F[EDGAR + scoring]
  F --> G[fisher_company, snapshot, etc.]
```

1. **Enqueue** — Fill the queue with all SEC tickers (or reset and re-fill for monthly).
2. **Worker** — Process pending rows in batches: EDGAR + company facts, then Fisher scoring. Mark each row done or failed.

## Commands

### Enqueue

```bash
export DATABASE_URL="postgresql://user:pass@host:5433/marketnews"

# Add only tickers not already in the queue (manual / incremental)
python3 scripts/enqueue_fisher_scan.py

# Clear queue and re-enqueue all SEC tickers (monthly full rescan)
python3 scripts/enqueue_fisher_scan.py --reset
```

### Worker

```bash
# Process until queue empty (default batch 50)
python3 scripts/run_fisher_scan_worker.py

# One batch then exit (for cron)
python3 scripts/run_fisher_scan_worker.py --once --batch 50

# Optional: delay between batches (seconds)
python3 scripts/run_fisher_scan_worker.py --batch 50 --delay 60
```

## Monthly run (cron)

Option A — run worker in a long-lived session (e.g. screen/tmux) after enqueue:

```bash
# First of month (example)
python3 scripts/enqueue_fisher_scan.py --reset
python3 scripts/run_fisher_scan_worker.py --batch 50
```

Option B — cron runs worker every hour with `--once`; enqueue monthly:

```cron
# Every hour: process one batch
0 * * * * cd /path/to/MarketNews && DATABASE_URL=... python3 scripts/run_fisher_scan_worker.py --once --batch 50

# First of month at 00:00: reset and enqueue (then hourly cron drains queue)
0 0 1 * * cd /path/to/MarketNews && DATABASE_URL=... python3 scripts/enqueue_fisher_scan.py --reset
```

## Queue table

- **fisher_scan_queue:** `ticker` (unique), `status` (pending | processing | done | failed), `source`, `queued_at`, `started_at`, `completed_at`, `error_text`.
- Worker claims `pending` rows with `FOR UPDATE SKIP LOCKED`, processes them, then sets `done` or `failed`.

## Rate limits

The EDGAR watcher already sleeps between requests (SEC guidelines). Use `--batch 50` (or smaller) and optional `--delay` to avoid hammering the SEC. Full SEC universe is large (10k+ tickers); monthly scan can take many hours depending on batch size and delay.
