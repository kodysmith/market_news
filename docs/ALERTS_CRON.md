# Alert cron entries

Run these from the repo root with venv activated. Ensure `API_BASE_URL` and (for FCM) `GOOGLE_APPLICATION_CREDENTIALS` or `FIREBASE_SERVICE_ACCOUNT_JSON` are set in `.env`.

## Precompute (GEX / cockpit / trade ideas every 5 min)

Runs during market window (8:30–17:00 ET, Mon–Fri) or when there are pending compute jobs.

```cron
*/5 * * * * cd /path/to/MarketNews && . venv/bin/activate && python3 scripts/run_gex_cockpit_precompute.py >> /path/to/MarketNews/logs/precompute.log 2>&1
```

## Morning 6:20 AM PST alert (SPX negative GEX)

Sends one FCM at 6:20 AM PST if SPX is negative gamma. Script exits silently outside 6:18–6:25 PST. API must be running at that time.

**Server in Eastern time (9:20 AM ET = 6:20 AM PST):**

```cron
20 9 * * 1-5 cd /path/to/MarketNews && . venv/bin/activate && python3 scripts/run_morning_gex_alert.py >> /path/to/MarketNews/logs/morning_alert.log 2>&1
```

**Server in Pacific time:**

```cron
20 6 * * 1-5 cd /path/to/MarketNews && . venv/bin/activate && python3 scripts/run_morning_gex_alert.py >> /path/to/MarketNews/logs/morning_alert.log 2>&1
```

Replace `/path/to/MarketNews` with your repo path. Create `logs/` if needed: `mkdir -p /path/to/MarketNews/logs`.
