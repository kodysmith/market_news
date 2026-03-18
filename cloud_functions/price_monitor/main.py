"""
Cloud-based price monitor — runs on GCP Cloud Functions.

Fetches prices via Yahoo Finance HTTP API (no IBKR needed),
checks alert thresholds from Supabase, sends FCM push notifications.

Triggered by Cloud Scheduler every 1 minute during market hours.

Deploy:
  gcloud functions deploy price-monitor \
    --runtime python312 \
    --trigger-http \
    --allow-unauthenticated \
    --region us-east1 \
    --memory 256MB \
    --timeout 30s \
    --set-env-vars SUPABASE_URL=...,SUPABASE_KEY=...,FCM_TOPIC=market_alerts

Cloud Scheduler (every 1 min, M-F 9:25-16:20 ET):
  gcloud scheduler jobs create http price-monitor-trigger \
    --schedule="25-80 9-16 * * 1-5" \
    --time-zone="America/New_York" \
    --uri="https://REGION-PROJECT.cloudfunctions.net/price-monitor" \
    --http-method=POST \
    --attempt-deadline=30s
"""

import json
import os
import time
from datetime import datetime, timezone, timedelta

import firebase_admin
from firebase_admin import credentials, messaging
from supabase import create_client

# ─── Config ──────────────────────────────────────────────────────────────────

SUPABASE_URL = os.environ.get("SUPABASE_URL", "")
SUPABASE_KEY = os.environ.get("SUPABASE_KEY", "")
FCM_TOPIC = os.environ.get("FCM_TOPIC", "market_alerts")

# Yahoo Finance quote endpoint — real-time, no API key needed
YF_QUOTE_URL = "https://query1.finance.yahoo.com/v8/finance/chart/{symbol}?interval=1m&range=1d"

# Ticker mapping: our names → Yahoo Finance symbols
TICKER_MAP = {
    "^GSPC": "^GSPC",     # SPX
    "^VIX": "^VIX",       # VIX
    "SPX": "^GSPC",
    "VIX": "^VIX",
    "SCHD": "SCHD",
    "TLT": "TLT",
    "GLD": "GLD",
    "JEPI": "JEPI",
    "SGOV": "SGOV",
    "VGIT": "VGIT",
    "EURUSD": "EURUSD=X",
    "USDJPY": "USDJPY=X",
    "GBPUSD": "GBPUSD=X",
    "USDMXN": "USDMXN=X",
    "GBPJPY": "GBPJPY=X",
    "AUDJPY": "AUDJPY=X",
}

ET = timezone(timedelta(hours=-4))  # EDT (adjust to -5 for EST)


# ─── Price Fetching ──────────────────────────────────────────────────────────

def fetch_price(ticker: str) -> float | None:
    """Fetch current price from Yahoo Finance HTTP API."""
    import urllib.request

    yf_symbol = TICKER_MAP.get(ticker, ticker)
    url = YF_QUOTE_URL.format(symbol=yf_symbol)

    try:
        req = urllib.request.Request(url, headers={
            "User-Agent": "Mozilla/5.0",
        })
        with urllib.request.urlopen(req, timeout=10) as resp:
            data = json.loads(resp.read())

        result = data["chart"]["result"][0]
        meta = result["meta"]

        # regularMarketPrice is real-time during market hours
        price = meta.get("regularMarketPrice")
        if not price:
            # Fallback to last close
            price = meta.get("previousClose")

        return float(price) if price else None

    except Exception as e:
        print(f"Price fetch failed for {ticker}: {e}")
        return None


def fetch_prices(tickers: list[str]) -> dict[str, float]:
    """Fetch prices for multiple tickers."""
    prices = {}
    for ticker in set(tickers):
        price = fetch_price(ticker)
        if price is not None:
            prices[ticker] = price
    return prices


# ─── Alert Logic ─────────────────────────────────────────────────────────────

def check_alert(alert: dict, price: float) -> bool:
    """Check if an alert condition is triggered."""
    condition = alert["condition"]
    threshold = float(alert["threshold"])

    if condition == "below":
        return price < threshold
    elif condition == "above":
        return price > threshold
    elif condition == "between":
        upper = float(alert.get("threshold_upper", threshold))
        return threshold <= price <= upper
    elif condition == "outside":
        upper = float(alert.get("threshold_upper", threshold))
        return price < threshold or price > upper
    return False


def is_in_cooldown(alert: dict) -> bool:
    """Check if alert is still in cooldown period."""
    last = alert.get("last_triggered_at")
    if not last:
        return False

    if isinstance(last, str):
        last = datetime.fromisoformat(last.replace("Z", "+00:00"))

    cooldown = timedelta(minutes=alert.get("cooldown_minutes", 15))
    return datetime.now(timezone.utc) - last < cooldown


def is_expired(alert: dict) -> bool:
    """Check if alert has passed its expiry date."""
    expiry = alert.get("expiry")
    if not expiry:
        return False
    if isinstance(expiry, str):
        from datetime import date
        expiry = date.fromisoformat(expiry)
    from datetime import date
    return date.today() > expiry


# ─── FCM Notifications ──────────────────────────────────────────────────────

_firebase_initialized = False

def init_firebase():
    global _firebase_initialized
    if _firebase_initialized:
        return

    # Try GOOGLE_APPLICATION_CREDENTIALS env var first (set automatically on GCP)
    # Falls back to FIREBASE_SERVICE_ACCOUNT_JSON for local testing
    sa_json = os.environ.get("FIREBASE_SERVICE_ACCOUNT_JSON")
    if sa_json:
        cred = credentials.Certificate(json.loads(sa_json))
        firebase_admin.initialize_app(cred)
    else:
        # On GCP, default credentials work automatically
        firebase_admin.initialize_app()

    _firebase_initialized = True


def send_push(title: str, body: str, data: dict = None):
    """Send FCM push notification to topic."""
    try:
        init_firebase()
        message = messaging.Message(
            notification=messaging.Notification(title=title, body=body),
            data={k: str(v) for k, v in (data or {}).items()},
            topic=FCM_TOPIC,
        )
        messaging.send(message)
        print(f"FCM sent: {title} — {body}")
    except Exception as e:
        print(f"FCM failed: {e}")


# ─── Market Hours Check ─────────────────────────────────────────────────────

def is_market_hours() -> bool:
    """Check if US equity market is open (with buffer for pre/post)."""
    now = datetime.now(ET)
    if now.weekday() >= 5:  # Weekend
        return False
    # Monitor from 9:25 to 16:20 ET (5 min buffer each side)
    market_start = now.replace(hour=9, minute=25, second=0)
    market_end = now.replace(hour=16, minute=20, second=0)
    return market_start <= now <= market_end


# ─── Main Handler ────────────────────────────────────────────────────────────

def price_monitor(request):
    """Cloud Function entry point (HTTP trigger)."""

    # Skip outside market hours (Cloud Scheduler handles this too, but double-check)
    if not is_market_hours():
        return json.dumps({"status": "outside_market_hours"}), 200

    # Connect to Supabase
    if not SUPABASE_URL or not SUPABASE_KEY:
        return json.dumps({"error": "SUPABASE_URL/KEY not set"}), 500

    sb = create_client(SUPABASE_URL, SUPABASE_KEY)

    # Fetch active alerts
    result = sb.table("price_alerts").select("*").eq("enabled", True).execute()
    alerts = result.data

    if not alerts:
        return json.dumps({"status": "no_active_alerts"}), 200

    # Determine which tickers we need prices for
    tickers = list(set(a["ticker"] for a in alerts))
    prices = fetch_prices(tickers)

    if not prices:
        return json.dumps({"error": "no_prices_fetched"}), 500

    # Store price snapshots
    snapshots = [
        {"ticker": t, "price": float(p), "source": "yahoo"}
        for t, p in prices.items()
    ]
    try:
        sb.table("price_snapshots").insert(snapshots).execute()
    except Exception as e:
        print(f"Snapshot insert failed: {e}")

    # Check each alert
    triggered = []
    for alert in alerts:
        ticker = alert["ticker"]
        price = prices.get(ticker)

        if price is None:
            continue

        if is_expired(alert):
            # Auto-disable expired alerts
            sb.table("price_alerts").update({"enabled": False}).eq("id", alert["id"]).execute()
            continue

        if not check_alert(alert, price):
            continue

        if is_in_cooldown(alert):
            continue

        # TRIGGERED
        severity = alert.get("severity", "warning")
        name = alert["name"]
        msg = alert.get("message") or f"{ticker} at ${price:,.2f}"
        strategy = alert.get("strategy", "")

        title_prefix = {"critical": "🚨", "warning": "⚠️", "info": "ℹ️"}.get(severity, "")
        title = f"{title_prefix} {name}"
        body = f"{msg}\n{ticker}: ${price:,.2f}"
        if strategy:
            body += f"\nStrategy: {strategy}"

        send_push(title, body, {
            "ticker": ticker,
            "price": str(price),
            "alert_id": str(alert["id"]),
            "severity": severity,
        })

        # Update alert state
        sb.table("price_alerts").update({
            "last_triggered_at": datetime.now(timezone.utc).isoformat(),
            "trigger_count": (alert.get("trigger_count", 0) or 0) + 1,
        }).eq("id", alert["id"]).execute()

        triggered.append({"name": name, "ticker": ticker, "price": price})

    return json.dumps({
        "status": "ok",
        "prices": {t: float(p) for t, p in prices.items()},
        "alerts_checked": len(alerts),
        "alerts_triggered": len(triggered),
        "triggered": triggered,
    }), 200


# ─── Local testing ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    # For local testing without Cloud Functions framework
    import sys

    # Load .env if exists (try multiple paths)
    env_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../.env")
    if not os.path.exists(env_file):
        env_file = os.path.join(os.getcwd(), ".env")
    if os.path.exists(env_file):
        with open(env_file) as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    k, v = line.split("=", 1)
                    os.environ.setdefault(k.strip(), v.strip())

    class FakeRequest:
        pass

    result, status = price_monitor(FakeRequest())
    print(json.dumps(json.loads(result), indent=2))
