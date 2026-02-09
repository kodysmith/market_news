# Run Fisher pipeline manually (populate Supabase)

Use this when you have **no Fisher valuation data** in the database and want to populate it once by hand.

## 1. One-time: Supabase schema

In the **Supabase SQL Editor**, run (in order):

1. **Fisher tables:** `supabase_schema_fisher.sql` (creates `fisher_company`, `fisher_score_snapshot`, `fisher_scan_queue`, etc.)
2. **Queue only** (if the Fisher schema already has the rest): `scripts/fisher_scan_queue_schema.sql`

## 2. Env on the machine that runs the scripts

**Preferred (avoids DB host DNS issues):** Use **SUPABASE_URL** and **SUPABASE_SECRET_KEY** in `.env` (Project Settings → API). Scripts then use the Supabase REST API; no direct Postgres host needed. **Fallback:** Use **SUPABASE_DB_URL** or **DATABASE_URL** (Postgres URI; use pooler if Direct fails):

- For fallback only: set **`SUPABASE_DB_URL`** or **`DATABASE_URL`** to your Supabase **Postgres URI**  
  (Supabase → Project Settings → Database → Connection string; use **Session** or **Transaction** pooler if “Direct” fails.)
Load `.env` if you use it (e.g. `set -a && source .env && set +a` or copy the vars into your shell).

## 3. SEC universe (if missing)

From repo root:

```bash
cd /path/to/MarketNews
python3 scripts/fetch_sec_universe.py
```

This creates/updates `data/sec_universe.json`. Needed before enqueue.

## 4. Setup queue and enqueue tickers

From repo root with `SUPABASE_DB_URL` or `DATABASE_URL` set:

```bash
# Create queue table (if not already) and enqueue all SEC tickers
python3 scripts/setup_fisher_full_scan.py --reset
```

`--reset` clears the queue and enqueues every ticker from the SEC universe. For a **small test set** (e.g. 20 tickers), you can enqueue by hand (see “Small test run” below).

## 5. Run the worker (process the queue)

From repo root:

```bash
# Process until queue is empty (batch 50 per run)
python3 scripts/run_fisher_scan_worker.py --batch 50

# Or one batch and exit (good for testing)
python3 scripts/run_fisher_scan_worker.py --once --batch 20
```

The worker:

1. **Claims** `pending` rows from `fisher_scan_queue` (marks them `processing`)
2. Runs **EDGAR watcher** (filings + company facts → `fisher_company`, `fisher_filing`, `fisher_financial_fact`)
3. Runs **Fisher scoring** → writes metrics to `fisher_score_snapshot` (points, category_scores, total_score)
4. **Marks** queue rows `done` or `failed`

The Flutter app reads `fisher_company` and `fisher_score_snapshot` from Supabase, so once the worker finishes, metrics are available in the app.

**Enqueue and process immediately:** Run the worker right after enqueueing so queued items get processed in the same run:

```bash
# Enqueue (or --reset) then process one batch
python3 scripts/enqueue_fisher_scan.py --process-after 1

# Enqueue then process until queue empty
python3 scripts/enqueue_fisher_scan.py --process-after 0

# Enqueue then process 5 batches (e.g. 250 tickers at batch 50)
python3 scripts/enqueue_fisher_scan.py --process-after 5 --batch 50
```

**Auto-trigger when something is queued:** Run the worker on a schedule so new queue rows get picked up without a separate enqueue run:

```bash
# Cron: every 5 minutes, process one batch (anything enqueued gets processed within 5 min)
*/5 * * * * cd /path/to/MarketNews && python3 scripts/run_fisher_scan_worker.py --once --batch 50
```

Full SEC universe is large; processing can take hours. Use `--batch 20` or `50` and optional `--delay 60` to be nice to SEC.

## 6. Small test run (a few tickers)

To fill only a few tickers (e.g. AAPL, MSFT, GOOGL) without the full SEC list:

1. Apply the Supabase schema (step 1).
2. Ensure `SUPABASE_DB_URL` or `DATABASE_URL` is set.
3. Enqueue specific tickers. If you don’t have a dedicated script, you can insert into `fisher_scan_queue` from the Supabase SQL Editor:

   ```sql
   INSERT INTO fisher_scan_queue (ticker, status, source)
   VALUES ('AAPL', 'pending', 'manual'), ('MSFT', 'pending', 'manual'), ('GOOGL', 'pending', 'manual')
   ON CONFLICT (ticker) DO NOTHING;
   ```

4. Run the worker:

   ```bash
   python3 scripts/run_fisher_scan_worker.py --batch 10
   ```

## 7. Check that data is there

- **Supabase Table Editor:** Open `fisher_company` and `fisher_score_snapshot`; you should see rows after the worker runs.
- **CLI (from repo root):**
  ```bash
  python3 scripts/list_fisher_scores.py
  # or
  python3 scripts/top_fisher_scores.py
  ```

## Migrate Fisher data from local to Supabase

When shutting down the local Postgres and moving everything to Supabase, copy Fisher data with:

**Option A – Direct (one host can reach both DBs)**

```bash
# .env: DATABASE_URL = local Postgres, SUPABASE_DB_URL = Supabase Postgres
python3 scripts/migrate_fisher_local_to_supabase.py
```

**Option B – Dump then load (when this machine can’t reach Supabase)**

1. On the machine with local DB (and no Supabase DNS):

   ```bash
   # .env: DATABASE_URL = local Postgres
   python3 scripts/migrate_fisher_local_to_supabase.py --dump-to fisher_dump
   ```

2. Copy the `fisher_dump/` directory to a host that can reach Supabase (e.g. server, cloud shell).

3. On that host, with repo and `.env` containing `SUPABASE_DB_URL`:

   ```bash
   python3 scripts/migrate_fisher_local_to_supabase.py --load-from fisher_dump
   ```

The script copies all Fisher tables in FK order and upserts by `id`, so re-runs are safe.

## Troubleshooting

- **“could not translate host name” / connection refused:** Use the **pooler** connection string (Session or Transaction) from Supabase, not the direct DB host.
- **“Set DATABASE_URL or SUPABASE_DB_URL”:** Scripts read from env; set one of these (or put them in `.env` and source it).
- **Empty queue / worker does nothing:** Run `setup_fisher_full_scan.py --reset` (or insert rows into `fisher_scan_queue`); worker only processes `status = 'pending'`.
- **SEC / rate limits:** Use a smaller `--batch` (e.g. 20) and `--delay 60` if you hit rate limits.
