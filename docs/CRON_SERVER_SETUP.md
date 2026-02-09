# Cron jobs on the server (publishing data to Supabase)

Use this after you’ve moved the repo to the server and want to run the cron jobs that publish GEX/cockpit and news to Supabase.

**Market window:** Scripts run only during **8:30 AM – 5:00 PM Eastern, Mon–Fri** (1 hour before market open through 1 hour after close), **or** when there are **pending jobs in the compute queue** (so on-demand requests are processed anytime). Outside that window they exit without doing work. Requires Python 3.9+ (or `pytz` installed) for correct Eastern time.

## 1. One-time setup on the server

### 1.1 Repo and env

- Clone/copy the repo to the server (e.g. `/home/you/MarketNews`).
- In the **repo root**, create or edit `.env` with at least:
  - `SUPABASE_URL=https://YOUR_PROJECT.supabase.co`
  - `SUPABASE_SECRET_KEY=your_secret_key` (from Supabase → Project Settings → API)
  - `API_BASE_URL=http://localhost:5000` (if the Flask API runs on this server; otherwise the URL of the machine that runs the API)

The precompute script calls the API at `API_BASE_URL` for `/gex/calculate` and `/cockpit/state`. So either run the API on this server (same machine) or point `API_BASE_URL` to another host.

### 1.2 Python and dependencies

From the repo root:

```bash
cd /path/to/MarketNews
python3 -m venv venv
./venv/bin/pip install -r requirements.txt
```

Ensure `supabase` is installed so the scripts use the Supabase client (no Postgres URI needed).

### 1.3 Supabase schema (once per project)

In the **Supabase SQL editor**, run (if not already done):

- `scripts/compute_result_cache_schema.sql` (for GEX/cockpit cache)
- `scripts/news_feed_schema.sql` (for news feed)

### 1.4 Optional: run the API on this server

If GEX/cockpit precompute should call an API on the same server:

```bash
cd /path/to/MarketNews
./venv/bin/python -m apis.start_apis
# or: ./venv/bin/gunicorn -w 1 -b 0.0.0.0:5000 "apis.app_factory:create_app()"
```

Run that in a process manager (systemd, screen, etc.) so it stays up. Then keep `API_BASE_URL=http://localhost:5000` in `.env`.

---

## 2. Test the scripts manually

From the repo root:

```bash
cd /path/to/MarketNews

# GEX + cockpit → compute_result_cache (needs API reachable at API_BASE_URL)
./venv/bin/python scripts/run_gex_cockpit_precompute.py

# News → news_feed (reads data/news.json or fetches API_BASE_URL/news.json)
./venv/bin/python scripts/publish_news_to_supabase.py
```

Fix any missing env or API errors before adding cron.

---

## 3. Add cron jobs

Edit crontab:

```bash
crontab -e
```

Add these lines (replace `/path/to/MarketNews` with your repo path, e.g. `/home/you/MarketNews`):

```cron
# GEX + Cockpit: every 5 minutes (app reads from compute_result_cache)
*/5 * * * * cd /path/to/MarketNews && ./venv/bin/python scripts/run_gex_cockpit_precompute.py >> /path/to/MarketNews/logs/precompute.log 2>&1

# News: once daily at 08:00 (app reads from news_feed)
0 8 * * * cd /path/to/MarketNews && ./venv/bin/python scripts/publish_news_to_supabase.py >> /path/to/MarketNews/logs/news_publish.log 2>&1
```

Create the log directory so cron can write:

```bash
mkdir -p /path/to/MarketNews/logs
```

Use your actual path everywhere (e.g. `/home/kody/MarketNews`).

---

## 4. Optional: compute queue worker

If the app uses the **compute job queue** (on-demand jobs for non-core symbols), run a worker on this server so jobs get processed:

```bash
./venv/bin/python scripts/run_compute_worker.py --loop
```

Run that under systemd/supervisor/screen so it stays up. No need to add it to cron; it polls the queue continuously.

## 5. Fisher scan worker (no market window)

The **Fisher scan** worker (`run_fisher_scan_worker.py`) is **not** gated by market window. It processes `fisher_scan_queue` whenever you run it and can run 24/7 (long-running, ongoing process). Only GEX/cockpit precompute and news publish use the market-window check.

---

## 6. Summary

| Job | Schedule | Script | Writes to | When it actually runs |
|-----|----------|--------|-----------|------------------------|
| GEX + Cockpit | Every 5 min | `run_gex_cockpit_precompute.py` | `compute_result_cache` | Market window (8:30–17:00 ET, Mon–Fri) **or** when queue has pending jobs |
| News | Daily 08:00 | `publish_news_to_supabase.py` | `news_feed` | Market window only (8:30–17:00 ET, Mon–Fri) |

Env required: `SUPABASE_URL`, `SUPABASE_SECRET_KEY`, and (for precompute) `API_BASE_URL` pointing at a running API.

Cron can stay “every 5 min” / “daily 08:00”; the scripts no-op outside the market window (and precompute also runs when someone has enqueued a job).
