#!/usr/bin/env python3
"""
Send one test FCM notification to the market_alerts topic.
Run from repo root so .env (and GOOGLE_APPLICATION_CREDENTIALS) is loaded by the precompute module.

  python scripts/test_fcm.py

Ensure the app has been opened at least once (subscribes to market_alerts) and is in background
to see the notification.
"""

import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import run_gex_cockpit_precompute as precompute

def main() -> int:
    # Show why FCM might skip (precompute uses logger.debug for skip reasons)
    import logging
    logging.getLogger("run_gex_cockpit_precompute").setLevel(logging.DEBUG)

    # Diagnose credentials path so user sees the exact cause if it fails
    path = (os.environ.get("GOOGLE_APPLICATION_CREDENTIALS") or os.environ.get("FIREBASE_SERVICE_ACCOUNT_JSON") or "").strip()
    if not path:
        print("No GOOGLE_APPLICATION_CREDENTIALS or FIREBASE_SERVICE_ACCOUNT_JSON in .env")
    else:
        path = os.path.expanduser(path)
        if not os.path.isabs(path):
            path = str((ROOT / os.path.normpath(path)).resolve())
        print("Credentials path:", path)
        print("File exists:", os.path.isfile(path))
        if os.path.isfile(path):
            try:
                with open(path) as f:
                    data = json.load(f)
                if data.get("type") != "service_account":
                    print(
                        "Key file must be a Firebase service account JSON (type: 'service_account'). "
                        "Get it from Firebase Console → Project settings → Service accounts → Generate new private key."
                    )
            except (json.JSONDecodeError, OSError):
                pass

    ok = precompute.send_fcm_to_topic(
        precompute.FCM_TOPIC_MARKET_ALERTS,
        "Test",
        "FCM is working",
        None,
    )
    if ok:
        print("FCM test sent successfully. Check your device for a 'Test' notification.")
        return 0
    print("FCM test failed. Check credentials (GOOGLE_APPLICATION_CREDENTIALS in .env) and logs above.")
    return 1

if __name__ == "__main__":
    sys.exit(main())
