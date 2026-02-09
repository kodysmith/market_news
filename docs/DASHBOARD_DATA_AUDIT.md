# Dashboard Data Audit: App Screens vs Data Sources

This doc lists each app screen, where it gets data today, and what’s missing when the API isn’t running. Goal: **publish all dashboard data to Supabase** so the app can run API-free and we have data for backtesting/correlation.

## Current App Screens and Data Sources

| Screen | Data source today | When API is off |
|--------|--------------------|------------------|
| **Cockpit** | Supabase `compute_result_cache` (GEX/cockpit) + API `cockpit/events` | Cockpit/GEX OK; events may fail if API off |
| **Intelligence** | API `quantengine/recommendations` | Fails |
| **Trade Ideas** | API `trade-ideas/allowed` | Fails |
| **News** | API `news.json` (reads `data/news.json` + yfinance earnings) | Fails |
| **Range** | Supabase cache/queue or API `probability/range` | OK if Supabase + worker or cache |
| **Fisher & Val** | Supabase `fisher_company`, `fisher_score_snapshot`; API for growth-profitable | Snapshot OK; growth-profitable fails |

## Endpoints Used by the App

| Endpoint | Used by | In Supabase? |
|----------|---------|---------------|
| GEX/cockpit | Cockpit | Yes – `compute_result_cache` (precompute script) |
| `cockpit/events` | Cockpit (events section) | No – API only |
| `quantengine/recommendations` | Intelligence | No |
| `trade-ideas/allowed` | Trade Ideas | No – heavy compute, could cache or queue |
| `news.json` | News | No – **add `news_feed` table + publish job** |
| `probability/range` | Range | Yes – cache/queue |
| Fisher snapshot/delta/evidence | Fisher & Val | Yes – `fisher_company`, `fisher_score_snapshot` |
| `fisher/growth-profitable` | Fisher & Val | No – API only |
| `report.json` | Main nav (when no Supabase) | Skipped when Supabase configured |

## Gaps When Running “Supabase-Only” (No API)

1. **News** – App calls `GET /news.json`. No table yet → add `news_feed` and a job that publishes from `data/news.json` (or API) to Supabase.
2. **Cockpit events** – Still from API; could add `cockpit_events` table and publish from same precompute or a cron.
3. **Intelligence** – From API; could add `intelligence_recommendations` table and a publish job or Edge Function.
4. **Trade Ideas** – From API; already heavy; could cache results in `compute_result_cache` (task_type `trade_ideas`) or a dedicated table + worker.
5. **Fisher growth-profitable** – From API; could add table + scan job that writes to Supabase.

## Recommended Direction: “Publish Everything to DB”

1. **News (first)**  
   - Table: `news_feed` (id, headline, source, url, summary, published_date, category, created_at).  
   - Job: run periodically (e.g. same cron as precompute); read `data/news.json` (or call local `news.json` for enrichment) and upsert into `news_feed`.  
   - App: NewsScreen reads from Supabase when configured, else API.

2. **Cockpit events**  
   - Table: `cockpit_events_cache` or extend precompute to write events JSON per symbol.  
   - Job: same precompute script or separate cron that calls API `cockpit/events` and writes to Supabase.  
   - App: Cockpit reads events from Supabase when available.

3. **Intelligence**  
   - Table: e.g. `intelligence_recommendations` (latest JSON or structured columns).  
   - Job: cron or Edge Function that calls QuantEngine/recommendations and writes to Supabase.  
   - App: Intelligence screen reads from Supabase when available.

4. **Trade Ideas**  
   - Already have `compute_result_cache` with task_type `trade_ideas` and `compute_job_queue`.  
   - Ensure worker (or a cron that runs the trade-ideas logic) writes results to cache/queue; app already can use getCachedOrEnqueue.

5. **Fisher growth-profitable**  
   - Table: e.g. `fisher_growth_profitable` (scan output).  
   - Job: run scan script, write to Supabase.  
   - App: Fisher & Val reads from Supabase when available.

## Supabase Edge Functions vs Cron + Scripts

- **Cron + Python/Node script**: Runs on your machine or a server; calls API or reads files, writes to Supabase. Good for news (read `data/news.json`), precompute (GEX/cockpit), Fisher scan.
- **Supabase Edge Function**: Runs on Supabase; can call external APIs (e.g. news provider) and write to Supabase. Good if you want no self-hosted cron (e.g. invoke via cron job that hits the function URL, or scheduled invocation). For news, either a scheduled Edge Function that fetches news and writes to `news_feed`, or keep a simple “publish from file” script and run it via cron.

## Files Added / Updated in This Pass

- **docs/DASHBOARD_DATA_AUDIT.md** (this file)
- **scripts/news_feed_schema.sql** – table for news (run once in Supabase SQL editor)
- **scripts/publish_news_to_supabase.py** – publish news to Supabase (from `data/news.json` or API `news.json`)
- **market_news_app**: NewsScreen reads from Supabase `news_feed` when configured, else API

### News: one-time setup and cron

1. Run **scripts/news_feed_schema.sql** in the Supabase SQL editor.
2. Add news to DB: run `python scripts/publish_news_to_supabase.py` (uses same `.env`: SUPABASE_URL, SUPABASE_SECRET_KEY). Script reads from `data/news.json` if present, else fetches from API `news.json`.
3. Optional cron (e.g. daily): `0 8 * * * cd /path/to/repo && ./venv/bin/python scripts/publish_news_to_supabase.py` so the feed stays updated. The app shows the latest 100 rows by `created_at`; running the script repeatedly appends (no replace-all unless you truncate `news_feed` in SQL).
