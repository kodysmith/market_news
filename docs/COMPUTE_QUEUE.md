# On-demand compute job queue

When the app needs data for a symbol that isn’t in the precomputed set (e.g. top 10), it can enqueue a job: the app inserts a row into Supabase; the private server claims it, runs the same API logic (GEX, valuation, cockpit, probability, trade_ideas), and writes the result back. The app shows “We’ll process that now…” and polls until the result is ready.

## Data from Supabase (primary source)

When `SUPABASE_URL` and `SUPABASE_ANON_KEY` are set, the app uses Supabase as the primary data source.

- **Fisher** – Snapshot, delta, evidence, and universe are read from `fisher_company` and `fisher_score_snapshot`. Growth-profitable still uses the API (or a future Supabase view/RPC).
- **GEX and Cockpit (Supabase-only)** – Core symbols (SPX, XSP, SPY, NDX) are precomputed every 5 minutes into `compute_result_cache`. The Flutter app reads only from this cache table; no API and no enqueue. Configure `GEX_CORE_SYMBOLS` in `data/config.json`; run `scripts/run_gex_cockpit_precompute.py` every 5 minutes via cron (see below).
- **Probability, Valuation, on-demand** – Optional: use `compute_job_queue` and `getCachedOrEnqueue`; worker writes results. If Supabase is not configured, those services return null.
- **Trade Ideas** – [TradeIdeasService](market_news_app/lib/services/trade_ideas_service.dart) tries **published ideas** first: `fetchPublishedByDate()` reads from `trade_ideas_published` for today/yesterday (ET). If that returns rows, the app shows them with “Valid: YYYY-MM-DD” and does not call the API. If empty, it then tries `getCachedResultFromTable(symbol, trade_ideas)` and `getCachedOrEnqueue(symbol, trade_ideas)` (compute worker / cache). If Supabase is not configured or all return nothing, the app falls back to `GET $apiBaseUrl/trade-ideas/allowed`. **The Flask API does not need to be on the internet** for published ideas: only the cron script needs to reach it (e.g. localhost).

## Precompute cache (GEX / Cockpit every 5 min)

To keep core symbols up to date without an always-on API:

1. **Schema**: Run [scripts/compute_result_cache_schema.sql](../scripts/compute_result_cache_schema.sql) in the Supabase SQL editor (creates `compute_result_cache` with symbol, task_type, result, updated_at).
2. **Core symbols**: In `data/config.json`, set `GEX_CORE_SYMBOLS` to `["SPX", "XSP", "SPY", "NDX"]` (or your list). The precompute script and Flutter app use this list.
3. **Cron**: On a machine with **`SUPABASE_URL`** and **`SUPABASE_SECRET_KEY`** set (or fallback `SUPABASE_DB_URL`/`DATABASE_URL`), and with the localhost API running (or `API_BASE_URL` pointing at it), run every 5 minutes. Activate the repo venv so dependencies are available:

```bash
*/5 * * * * cd /path/to/repo && . venv/bin/activate && python3 scripts/run_gex_cockpit_precompute.py
```

The script calls the existing localhost API endpoints (`/gex/calculate?ticker=X`, `/cockpit/state?ticker=X`, `/trade-ideas/allowed?ticker=X&max_ideas=3&timeframe=all`) to get GEX, cockpit, and trade ideas for each core symbol. It then upserts the JSON responses into `compute_result_cache`. **Trade ideas** for core symbols are therefore precomputed every 5 minutes, so the Trade Ideas page reads from cache for SPX, SPY, XSP, NDX without enqueueing a job.

**Preferred (no Postgres URI, no IPv6/password issues):** Set **`SUPABASE_URL`** and **`SUPABASE_SECRET_KEY`** (secret key from Project Settings → API; see [Understanding API keys](https://supabase.com/docs/guides/api/api-keys)). The script uses the [Supabase Python client](https://supabase.com/docs/reference/python/initializing) to upsert; the table must already exist (run `compute_result_cache_schema.sql` once in the Supabase SQL editor).

**Fallback:** Set **`SUPABASE_DB_URL`** (or `DATABASE_URL`) to your Supabase Postgres URI if you don’t use the client. The script must write to the same database the app reads from. If you set only `DATABASE_URL` to a local Postgres URL, the app (which reads via the Supabase client) will see no rows.

Set `API_BASE_URL` (default `http://localhost:5000`) and optionally `API_SECRET_KEY` if your API requires an x-api-key header.

## Trade ideas published (today / yesterday)

Trade ideas can be **published to Supabase** when SPX (or core symbols) meet entry criteria, so the app can show “Unlocked” ideas **without calling a public API**. Same pattern as GEX precompute: a cron job on the same server as the API writes to Supabase; the app reads by date.

1. **Schema**: Run [scripts/migrations/002_trade_ideas_published.sql](../scripts/migrations/002_trade_ideas_published.sql) in the Supabase SQL editor (creates `trade_ideas_published` with `valid_date`, `as_of_et`, `symbol`, `idea_id`, `payload`).
2. **Publisher script**: [scripts/publish_trade_ideas.py](../scripts/publish_trade_ideas.py) calls the **local** API (`API_BASE_URL`, e.g. `http://localhost:5000`) at `/trade-ideas/allowed?ticker=SYMBOL&max_ideas=5&timeframe=all` for each core symbol (from `GEX_CORE_SYMBOLS` or default SPX, XSP, SPY, NDX). It flattens unlocked ideas and upserts into `trade_ideas_published` with `valid_date` = today ET, `as_of_et` = now ET.
3. **Cron**: On the same machine where the API runs (so the script can reach localhost), run every 5 minutes like GEX:

```bash
*/5 * * * * cd /path/to/repo && . venv/bin/activate && python3 scripts/publish_trade_ideas.py
```

4. **App**: The Trade Ideas screen calls `TradeIdeasService.fetchPublishedByDate()` first. If there are rows for today or yesterday (ET), it shows those and displays **“Valid: &lt;valid_date&gt;”** (and optional as_of_et). No API call and no enqueue. If the table is empty, the app falls back to cache/enqueue or the public API.

**Requirements**: `SUPABASE_URL`, `SUPABASE_SECRET_KEY` (or `SUPABASE_SERVICE_ROLE_KEY`), and `API_BASE_URL` pointing at the local API. The Flask API does not need to be exposed to the internet.

## Schema (Supabase)

Run in Supabase SQL editor (after your main schema):

- [scripts/compute_job_queue_schema.sql](../scripts/compute_job_queue_schema.sql) – queue for optional on-demand jobs.
- [scripts/compute_result_cache_schema.sql](../scripts/compute_result_cache_schema.sql) – cache for precomputed GEX/cockpit (one row per symbol, task_type). Required for Supabase-only GEX/cockpit.

`compute_job_queue`: id, symbol, task_type, status, requested_at, claimed_at, completed_at, result (JSONB), error_text. RLS: anon INSERT/SELECT; service_role UPDATE.

`compute_result_cache`: symbol, task_type, result (JSONB), updated_at. RLS: anon SELECT; service_role full. Filled by `run_gex_cockpit_precompute.py`.

- [scripts/migrations/002_trade_ideas_published.sql](../scripts/migrations/002_trade_ideas_published.sql) – `trade_ideas_published`: valid_date, as_of_et, symbol, idea_id, payload (JSONB). RLS: anon SELECT; service_role full. Filled by `publish_trade_ideas.py`.

## Private server worker

On the machine that runs the heavy API (same codebase, same `data/config.json` and API keys):

1. Set **SUPABASE_URL** and **SUPABASE_SECRET_KEY** in repo `.env` (preferred; worker then writes to Supabase and also upserts to `compute_result_cache`). Or set **SUPABASE_DB_URL** / **DATABASE_URL** to your Supabase Postgres URI.
2. Run:

```bash
python scripts/run_compute_worker.py --once   # process one job and exit
python scripts/run_compute_worker.py --loop   # keep polling until queue empty (or use Realtime when enabled)
python scripts/run_compute_worker.py --daemon # same as --loop
```

**Polling and after-hours:** When the queue is empty, the worker sleeps longer outside market hours (Mon–Fri 8:30–17:00 ET). Use `--poll` (default 30s) for sleep during market hours and `--poll-after-hours` (default 300s) for outside that window.

**Realtime (push):** When using the Supabase client, the worker can subscribe to Supabase Realtime for `INSERT` on `compute_job_queue`. When a new job is enqueued, the worker wakes immediately instead of waiting for the next poll. To enable this, add the queue table to the Realtime publication once in the Supabase SQL editor:

```sql
ALTER PUBLICATION supabase_realtime ADD TABLE compute_job_queue;
```

If Realtime is not enabled for the table (or the `realtime` package is missing), the worker falls back to timeout-based polling using the same market/after-hours intervals. The `realtime` package is included with the `supabase` Python dependency.

The worker uses **SUPABASE_URL + SUPABASE_SECRET_KEY** (preferred) or Postgres URI. It claims one row from `compute_job_queue`, runs the corresponding Flask handler in-process (valuation, GEX, cockpit, probability, trade_ideas), updates the row with `result` and `status = 'done'` (or `failed` and `error_text`), and **also upserts into `compute_result_cache`** when using the Supabase client so the app’s cache read sees the result without running the precompute script. No HTTP; it reuses the API logic via `test_request_context`.

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

## SPY / core symbol GEX stale?

The **compute worker** only processes jobs that are in `compute_job_queue`. It does **not** proactively refresh SPY (or SPX, XSP, NDX). To keep core symbol GEX and cockpit data fresh:

1. **Run the precompute script every 5 minutes** on the same server where the API and worker run (activate venv so deps are available):
   ```bash
   */5 * * * * cd /path/to/repo && . venv/bin/activate && python3 scripts/run_gex_cockpit_precompute.py
   ```
2. **The API must be running** on that server (or set `API_BASE_URL` to where it runs). The precompute script calls `GET /gex/calculate?ticker=SPY` (and cockpit) for each core symbol and upserts into `compute_result_cache`.
3. **Core symbols** come from `data/config.json` → `GEX_CORE_SYMBOLS` or default `["SPX", "XSP", "SPY", "NDX"]`.

Alternatively run the precompute in a loop (e.g. in a second terminal): `scripts/run_gex_precompute_loop.sh` (see script for usage).

## GEX regime flip alert (SPX, SPY, XSP)

When the precompute script runs, it detects **intraday GEX regime flips** (positive to negative gamma or vice versa) for SPX, SPY, and XSP. When a flip is detected it logs an alert and sends an **FCM push** to the topic `market_alerts`, so the phone can buzz even when the app is in background or dozing.

- **App:** On start, the app subscribes to the FCM topic `market_alerts` (no token storage). A background message handler is registered so notifications are handled when the app is in background or terminated.
- **Server:** The precompute script sends one FCM message to topic `market_alerts` (title and body) when a flip is detected. To enable send, set one of these in repo root `.env` to the **absolute path** of your Firebase service account JSON (Firebase Console → Project Settings → Service accounts → Generate new private key):
  - `GOOGLE_APPLICATION_CREDENTIALS=/path/to/firebase-service-account.json`
  - or `FIREBASE_SERVICE_ACCOUNT_JSON=/path/to/firebase-service-account.json`
  If unset, the script still logs the alert but does not send FCM. The repo `requirements.txt` includes `firebase-admin` for this.
- **Storage:** Last regime per symbol is stored in `data/gex_regime_last.json` on the server. Optional: use a Supabase table instead by running [scripts/gex_regime_snapshot_schema.sql](../scripts/gex_regime_snapshot_schema.sql) and extending the precompute script to read/write that table.
- **Alert symbols:** SPX, SPY, XSP (configurable via `GEX_REGIME_ALERT_SYMBOLS` in the script).

### Morning 6:20 AM PST alert (SPX negative GEX)

Run [scripts/run_morning_gex_alert.py](../scripts/run_morning_gex_alert.py) once at 6:20 AM PST (Mon–Fri). If SPX is negative gamma, it sends one FCM: "Be ready for fast moves at open." The script only runs when the current time is 6:18–6:25 PST; outside that window it exits silently. **The API must be running** at that time (same server as cron or set `API_BASE_URL`). Cron examples:

- Server in **ET**: `20 9 * * 1-5` (9:20 AM ET = 6:20 AM PST)
- Server in **PST**: `20 6 * * 1-5`

See [docs/ALERTS_CRON.md](ALERTS_CRON.md) for a single place to copy cron entries.

### Additional alerts (precompute script)

When the precompute runs during market hours it also sends FCM (and logs to the notification dataset) for:

- **Flip approach:** SPX approaching flip line toward negative, or at flip line (one notification per stage; state in `data/gex_flip_alert_state.json`).
- **Pinning:** SPX price at/near call wall or put wall (state in `data/gex_pin_alert_state.json`).

### Notification log (backtesting)

Every sent notification is appended to **`data/notification_log.jsonl`** (one JSON object per line: `ts`, `type`, `symbol`, and optional context). Use this dataset for backtesting (e.g. correlate alert times with price/vol). Types: `morning_negative`, `regime_flip`, `flip_approaching`, `flip_at`, `pin_call`, `pin_put`.

## Data not in Supabase?

- **Workers must write to the same Supabase project the app reads from.** Use **SUPABASE_URL + SUPABASE_SECRET_KEY** in repo `.env` so Fisher and compute workers use the Supabase client (no direct Postgres host). If you set only **SUPABASE_DB_URL** or **DATABASE_URL** to a local or different DB, workers write there and the app (reading via Supabase) sees nothing.
- **Tables must exist.** In Supabase SQL editor, run: `supabase_schema_fisher.sql` and `fisher_scan_queue_schema.sql` for Fisher; `compute_job_queue_schema.sql` and `compute_result_cache_schema.sql` for compute. RLS allows `service_role` to write; anon can read where documented.
- **Fisher:** Enqueue tickers (`scripts/enqueue_fisher_scan.py` or `setup_fisher_full_scan.py --reset`), then run `scripts/run_fisher_scan_worker.py --batch 50`. Data appears in `fisher_company` and `fisher_score_snapshot` as each (sub-)batch completes.
- **Compute cache:** The compute worker upserts to `compute_result_cache` when it finishes a job (Supabase client only). Alternatively run `scripts/run_gex_cockpit_precompute.py` every 5 min to fill the cache for core symbols.
- **Trade ideas tab:** For core symbols (SPX, XSP, SPY, NDX), trade ideas are precomputed every 5 minutes by `run_gex_cockpit_precompute.py`, so the app reads from cache. For other symbols the app enqueues a job; run the compute worker (`scripts/run_compute_worker.py --loop`) on a machine that can run the trade-ideas API so those jobs complete.
