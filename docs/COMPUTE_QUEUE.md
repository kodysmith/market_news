# On-demand compute job queue

When the app needs data for a symbol that isn’t in the precomputed set (e.g. top 10), it can enqueue a job: the app inserts a row into Supabase; the private server claims it, runs the same API logic (GEX, valuation, cockpit, probability, trade_ideas), and writes the result back. The app shows “We’ll process that now…” and polls until the result is ready.

## Schema (Supabase)

Run in Supabase SQL editor (after your main schema):

- [scripts/compute_job_queue_schema.sql](../scripts/compute_job_queue_schema.sql)

This creates `compute_job_queue` with columns: `id`, `symbol`, `task_type`, `status` (pending | processing | done | failed), `requested_at`, `claimed_at`, `completed_at`, `result` (JSONB), `error_text`. RLS allows anon to INSERT and SELECT; service_role can UPDATE (claim and set result).

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
