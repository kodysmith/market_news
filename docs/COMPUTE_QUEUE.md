# On-demand compute job queue

When the app needs data for a symbol that isn’t in the precomputed set (e.g. top 10), it can enqueue a job: the app inserts a row into Supabase; the private server claims it, runs the same API logic (GEX, valuation, cockpit, probability, trade_ideas), and writes the result back. The app shows “We’ll process that now…” and polls until the result is ready.

## Data from Supabase (primary source)

When `SUPABASE_URL` and `SUPABASE_ANON_KEY` are set, the app uses Supabase as the primary data source.

- **Fisher** – Snapshot, delta, evidence, and universe are read from `fisher_company` and `fisher_score_snapshot`. Growth-profitable still uses the API (or a future Supabase view/RPC).
- **GEX and Cockpit (Supabase-only)** – Core symbols (SPX, XSP, SPY, NDX) are precomputed every 5 minutes into `compute_result_cache`. The Flutter app reads only from this cache table; no API and no enqueue. Configure `GEX_CORE_SYMBOLS` in `data/config.json`; run `scripts/run_gex_cockpit_precompute.py` every 5 minutes via cron (see below).
- **Probability, Valuation, on-demand** – Optional: use `compute_job_queue` and `getCachedOrEnqueue`; worker writes results. If Supabase is not configured, those services return null.

## Precompute cache (GEX / Cockpit every 5 min)

To keep core symbols up to date without an always-on API:

1. **Schema**: Run [scripts/compute_result_cache_schema.sql](../scripts/compute_result_cache_schema.sql) in the Supabase SQL editor (creates `compute_result_cache` with symbol, task_type, result, updated_at).
2. **Core symbols**: In `data/config.json`, set `GEX_CORE_SYMBOLS` to `["SPX", "XSP", "SPY", "NDX"]` (or your list). The precompute script and Flutter app use this list.
3. **Cron**: On a machine with **`SUPABASE_URL`** and **`SUPABASE_SECRET_KEY`** set (or fallback `SUPABASE_DB_URL`/`DATABASE_URL`), and with the localhost API running (or `API_BASE_URL` pointing at it), run every 5 minutes:

```bash
*/5 * * * * cd /path/to/repo && python3 scripts/run_gex_cockpit_precompute.py
```

The script calls the existing localhost API endpoints (`/gex/calculate?ticker=X`, `/cockpit/state?ticker=X`) to get GEX and cockpit computations (the API uses its own data sources; no direct massive.com calls from the script). It then upserts the JSON responses into `compute_result_cache`.

**Preferred (no Postgres URI, no IPv6/password issues):** Set **`SUPABASE_URL`** and **`SUPABASE_SECRET_KEY`** (secret key from Project Settings → API; see [Understanding API keys](https://supabase.com/docs/guides/api/api-keys)). The script uses the [Supabase Python client](https://supabase.com/docs/reference/python/initializing) to upsert; the table must already exist (run `compute_result_cache_schema.sql` once in the Supabase SQL editor).

**Fallback:** Set **`SUPABASE_DB_URL`** (or `DATABASE_URL`) to your Supabase Postgres URI if you don’t use the client. The script must write to the same database the app reads from. If you set only `DATABASE_URL` to a local Postgres URL, the app (which reads via the Supabase client) will see no rows.

Set `API_BASE_URL` (default `http://localhost:5000`) and optionally `API_SECRET_KEY` if your API requires an x-api-key header.

## Schema (Supabase)

Run in Supabase SQL editor (after your main schema):

- [scripts/compute_job_queue_schema.sql](../scripts/compute_job_queue_schema.sql) – queue for optional on-demand jobs.
- [scripts/compute_result_cache_schema.sql](../scripts/compute_result_cache_schema.sql) – cache for precomputed GEX/cockpit (one row per symbol, task_type). Required for Supabase-only GEX/cockpit.

`compute_job_queue`: id, symbol, task_type, status, requested_at, claimed_at, completed_at, result (JSONB), error_text. RLS: anon INSERT/SELECT; service_role UPDATE.

`compute_result_cache`: symbol, task_type, result (JSONB), updated_at. RLS: anon SELECT; service_role full. Filled by `run_gex_cockpit_precompute.py`.

## Private server worker

On the machine that runs the heavy API (same codebase, same `data/config.json` and API keys):

1. Set `DATABASE_URL` or `SUPABASE_DB_URL` to your Supabase Postgres URI (same as Fisher).
2. Run:

```bash
python scripts/run_compute_worker.py --once   # process one job and exit
python scripts/run_compute_worker.py --loop   # keep polling until queue empty
```

The worker uses the same Postgres connection as Fisher, claims one row from `compute_job_queue`, runs the corresponding Flask handler in-process (valuation, GEX, cockpit, probability, trade_ideas), and updates the row with `result` and `status = 'done'` (or `failed` and `error_text`). No HTTP; it reuses the API logic via `test_request_context`.

## Flutter app

1. In the app’s `.env` (e.g. `market_news_app/.env`), set:
   - `SUPABASE_URL` – your Supabase project URL
   - `SUPABASE_ANON_KEY` – anon/public key

2. Use [ComputeQueueService](market_news_app/lib/services/compute_queue_service.dart):
   - `ComputeQueueService.isAvailable` – true if Supabase was initialized.
   - `ComputeQueueService.enqueueAndWait(symbol: 'AAPL', taskType: ComputeTaskType.gex)` – inserts a job, polls until done, returns `ComputeJobResult(ok: true, result: {...})` or error.
   - Or `enqueue()` then `getJobResult(jobId)` for manual polling / Realtime.

## Flow

1. User requests a symbol that isn’t in the precomputed cache.
2. App shows “We’ll process that now…” and calls `enqueueAndWait(symbol, taskType)`.
3. App inserts a row into `compute_job_queue` (status = pending).
4. Private server runs `run_compute_worker.py` (once or loop); it claims the row, runs the task, writes `result` and sets status = done.
5. App’s poll sees status = done and returns the result to the UI.

All data stays in Supabase; the server only needs outbound access to Supabase (no public URL).
