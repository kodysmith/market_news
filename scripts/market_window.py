"""
Market window and queue checks for cron scripts.
Run precompute/news only during market window (1 hr before open to 1 hr after close, ET, weekdays)
OR when there are pending jobs in compute_job_queue (so on-demand requests get processed).
"""

from __future__ import annotations

import os
from datetime import datetime

# US Eastern: market open 9:30 AM, close 4:00 PM. Window: 1 hr before to 1 hr after = 8:30 AM - 5:00 PM ET.
WINDOW_START_HOUR = 8
WINDOW_START_MINUTE = 30
WINDOW_END_HOUR = 17
WINDOW_END_MINUTE = 0

try:
    from zoneinfo import ZoneInfo
    ET = ZoneInfo("America/New_York")
except ImportError:
    try:
        import pytz
        ET = pytz.timezone("America/New_York")
    except ImportError:
        ET = None  # fallback: assume local time is ET (cron server in ET)


def is_market_window(now: datetime | None = None) -> bool:
    """
    True if now (or current time in ET) is within the run window:
    Mon–Fri, 8:30 AM – 5:00 PM Eastern (1 hr before open through 1 hr after close).
    """
    if now is None:
        now = datetime.now(ET) if ET else datetime.now()
    if ET:
        if now.tzinfo is None:
            now = now.replace(tzinfo=ET)
        else:
            now = now.astimezone(ET)
    # Monday = 0, Sunday = 6
    if now.weekday() >= 5:
        return False
    hour, minute = now.hour, now.minute
    start_minutes = WINDOW_START_HOUR * 60 + WINDOW_START_MINUTE
    end_minutes = WINDOW_END_HOUR * 60 + WINDOW_END_MINUTE
    now_minutes = hour * 60 + minute
    return start_minutes <= now_minutes < end_minutes


def has_pending_compute_jobs() -> bool:
    """
    True if Supabase compute_job_queue has any row with status = 'pending'.
    Requires SUPABASE_URL and SUPABASE_SECRET_KEY (or SUPABASE_SERVICE_ROLE_KEY) in env.
    """
    url = (os.environ.get("SUPABASE_URL") or "").strip().rstrip("/")
    key = (os.environ.get("SUPABASE_SECRET_KEY") or os.environ.get("SUPABASE_SERVICE_ROLE_KEY") or "").strip()
    if not url or not key:
        return False
    try:
        from supabase import create_client
        client = create_client(url, key)
        resp = client.table("compute_job_queue").select("id").eq("status", "pending").limit(1).execute()
        rows = getattr(resp, "data", None) or []
        return len(rows) > 0
    except Exception:
        return False


def should_run_scheduled_publish(check_queue: bool = True) -> bool:
    """
    True if we should run the scheduled publish (precompute/news):
    - In market window (8:30–17:00 ET, Mon–Fri), OR
    - There are pending compute jobs (someone enqueued; check_queue=True only).
    """
    if is_market_window():
        return True
    if check_queue and has_pending_compute_jobs():
        return True
    return False
