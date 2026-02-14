#!/usr/bin/env python3
"""
Run at 6:20 AM PST (9:20 ET). If SPX is negative gamma, send one FCM: "Be ready for fast moves at open."
Requires API running (GET /cockpit/state?ticker=SPX) and .env with GOOGLE_APPLICATION_CREDENTIALS (or FIREBASE_SERVICE_ACCOUNT_JSON).

Cron (server in ET): 20 9 * * 1-5
Cron (server in PST): 20 6 * * 1-5
"""

from __future__ import annotations

import os
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

# Load .env from repo root (same as precompute)
_env_file = ROOT / ".env"
if _env_file.exists():
    try:
        with open(_env_file) as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    k, _, v = line.partition("=")
                    v = v.strip()
                    if (v.startswith("'") and v.endswith("'")) or (v.startswith('"') and v.endswith('"')):
                        v = v[1:-1]
                    if k and k not in os.environ:
                        os.environ[k] = v
    except Exception:
        pass

# Import after env is set
import run_gex_cockpit_precompute as precompute


def _is_morning_window_pst() -> bool:
    """True if current time is 6:18–6:25 PST, Mon–Fri (avoid duplicate cron runs)."""
    try:
        from zoneinfo import ZoneInfo
        pst = ZoneInfo("America/Los_Angeles")
    except ImportError:
        try:
            import pytz
            pst = pytz.timezone("America/Los_Angeles")
        except ImportError:
            return True  # no tz: run anyway
    now = datetime.now(pst)
    if now.weekday() >= 5:  # Sat/Sun
        return False
    return now.hour == 6 and 18 <= now.minute <= 25


def main() -> int:
    if not _is_morning_window_pst():
        return 0

    base_url = precompute.get_api_base_url()
    api_key = precompute.get_api_key()
    result = precompute.fetch_from_api(base_url, "/cockpit/state?ticker=SPX", api_key)
    if result is None or result.get("error"):
        return 0

    regime = result.get("regime") or {}
    is_positive = regime.get("is_positive")
    label = (regime.get("label") or "").upper()
    if is_positive is True or "POSITIVE" in label:
        return 0

    spot = regime.get("spot")
    title = "SPX negative gamma"
    body = "Be ready for fast moves at open."
    ok = precompute.send_fcm_to_topic(
        precompute.FCM_TOPIC_MARKET_ALERTS,
        title=title,
        body=body,
        data={"symbol": "SPX", "type": "morning_negative"},
    )
    if ok:
        precompute.log_notification_sent(
            "morning_negative", "SPX",
            {"spot": spot, "regime": "negative"},
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
