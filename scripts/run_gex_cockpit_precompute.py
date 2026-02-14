#!/usr/bin/env python3
"""
Precompute GEX and cockpit for core symbols (SPX, XSP, SPY, NDX) and upsert into compute_result_cache.
Calls the existing localhost API endpoints (no direct massive.com); writes results via Supabase client or Postgres.

Usage:
  python scripts/run_gex_cockpit_precompute.py

Preferred (no Postgres URI, no IPv6/password issues):
  - SUPABASE_URL (https://xxx.supabase.co) and SUPABASE_SECRET_KEY (secret key from Project Settings -> API).
  - The script uses the Supabase Python client (create_client) to upsert; same credentials as the app.

Fallback (direct Postgres):
  - SUPABASE_DB_URL or DATABASE_URL: Postgres connection string (only if Supabase client not used).

Also:
  - API_BASE_URL (default http://localhost:5000): local API must be running for GEX/cockpit.
  - Core symbols: data/config.json GEX_CORE_SYMBOLS or default [SPX, XSP, SPY, NDX].
  - Run compute_result_cache_schema.sql once in Supabase SQL editor so the table exists.
"""

from __future__ import annotations

import json
import logging
import os
import sys
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from urllib.request import Request, urlopen
from urllib.error import HTTPError, URLError

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

# Load repo root .env so SUPABASE_DB_URL / DATABASE_URL are set (e.g. for cron)
# Values may be quoted; strip one layer of " or ' so passwords with ! etc. work when quoted.
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

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

# Default core symbols for GEX/cockpit (plan: SPX, XSP, SPY, NDX)
DEFAULT_CORE_SYMBOLS = ["SPX", "XSP", "SPY", "NDX"]

# Symbols to alert on when GEX regime flips intraday (positive <-> negative gamma)
GEX_REGIME_ALERT_SYMBOLS = ["SPX", "SPY", "XSP"]

# Flip approach: "at flip" when distance to flip line < this (points)
FLIP_AT_THRESHOLD_PTS = 1.0

# Pinning: alert when spot within this many points of call/put wall; clear state when beyond LEAVE
PIN_WALL_DISTANCE_PTS = 2
PIN_WALL_LEAVE_PTS = 3

# Default localhost API (GEX/cockpit computations go through this API, not massive.com directly)
DEFAULT_API_BASE_URL = "http://localhost:5000"


def _regime_file() -> Path:
    """Path to JSON file storing last GEX regime per symbol (for flip detection)."""
    return ROOT / "data" / "gex_regime_last.json"


def _load_last_regimes() -> dict[str, str]:
    """Load last known regime per symbol. Returns e.g. {'SPY': 'POSITIVE', 'SPX': 'NEGATIVE'}."""
    path = _regime_file()
    if not path.exists():
        return {}
    try:
        with open(path) as f:
            data = json.load(f)
        return {str(k).upper(): str(v).upper() for k, v in (data or {}).items()}
    except Exception as e:
        logger.warning("Could not load regime file %s: %s", path, e)
        return {}


def _save_last_regime(symbol: str, regime: str) -> None:
    """Update stored last regime for symbol. Merges with existing file."""
    path = _regime_file()
    path.parent.mkdir(parents=True, exist_ok=True)
    current = _load_last_regimes()
    current[symbol.upper()] = regime.upper()
    try:
        with open(path, "w") as f:
            json.dump(current, f, indent=2)
    except Exception as e:
        logger.warning("Could not save regime file %s: %s", path, e)


def _extract_regime_from_cockpit(cockpit_result: dict) -> str | None:
    """Extract POSITIVE or NEGATIVE from cockpit state. Returns None if unknown/transition."""
    regime_data = cockpit_result.get("regime") or {}
    if isinstance(regime_data, dict):
        is_pos = regime_data.get("is_positive")
        if is_pos is True:
            return "POSITIVE"
        if is_pos is False:
            return "NEGATIVE"
        label = (regime_data.get("label") or "").upper()
        if "POSITIVE" in label:
            return "POSITIVE"
        if "NEGATIVE" in label:
            return "NEGATIVE"
    return None


def _flip_alert_state_file() -> Path:
    return ROOT / "data" / "gex_flip_alert_state.json"


def _load_flip_alert_state() -> dict[str, str | None]:
    path = _flip_alert_state_file()
    if not path.exists():
        return {}
    try:
        with open(path) as f:
            data = json.load(f)
        return {str(k).upper(): (v if v is None else str(v)) for k, v in (data or {}).items()}
    except Exception as e:
        logger.warning("Could not load flip alert state %s: %s", path, e)
        return {}


def _save_flip_alert_state(symbol: str, state: str | None) -> None:
    path = _flip_alert_state_file()
    path.parent.mkdir(parents=True, exist_ok=True)
    current = _load_flip_alert_state()
    current[symbol.upper()] = state
    try:
        with open(path, "w") as f:
            json.dump(current, f, indent=2)
    except Exception as e:
        logger.warning("Could not save flip alert state %s: %s", path, e)


def _compute_flip_alert_state(cockpit_result: dict) -> str:
    """Return one of: positive, approaching, at_flip, negative."""
    current = _extract_regime_from_cockpit(cockpit_result)
    if current == "NEGATIVE":
        return "negative"
    if current != "POSITIVE":
        return "positive"
    regime = cockpit_result.get("regime") or {}
    in_transition = regime.get("transition") is True
    distance_to_flip = regime.get("distance_to_flip")
    if distance_to_flip is None and regime.get("spot") is not None and regime.get("flip_line") is not None:
        distance_to_flip = regime["spot"] - regime["flip_line"]
    dist_pts = abs(float(distance_to_flip)) if distance_to_flip is not None else float("inf")
    if in_transition and dist_pts < FLIP_AT_THRESHOLD_PTS:
        return "at_flip"
    if in_transition:
        return "approaching"
    return "positive"


def _maybe_send_flip_approach_alert(symbol: str, cockpit_result: dict) -> None:
    """If SPX (or symbol) entered approaching/at_flip, send one FCM and log. Reset state when positive."""
    current = _compute_flip_alert_state(cockpit_result)
    last = _load_flip_alert_state().get(symbol)
    regime = cockpit_result.get("regime") or {}
    spot = regime.get("spot")
    flip_line = regime.get("flip_line")
    if current == "positive":
        _save_flip_alert_state(symbol, None)
        return
    if current == "negative":
        return  # handled by on_gex_regime_flip
    if current == last:
        return
    if current == "approaching":
        title = f"{symbol} approaching flip line toward negative"
        body = "SPX approaching flip line toward negative exposure."
        ok = send_fcm_to_topic(FCM_TOPIC_MARKET_ALERTS, title=title, body=body, data={"symbol": symbol, "type": "flip_approaching"})
        if ok:
            log_notification_sent("flip_approaching", symbol, {"spot": spot, "flip_line": flip_line, "regime": "positive"})
            _save_flip_alert_state(symbol, "approaching")
    elif current == "at_flip":
        title = f"{symbol} at flip line"
        body = "SPX at flip line."
        ok = send_fcm_to_topic(FCM_TOPIC_MARKET_ALERTS, title=title, body=body, data={"symbol": symbol, "type": "flip_at"})
        if ok:
            log_notification_sent("flip_at", symbol, {"spot": spot, "flip_line": flip_line, "regime": "positive"})
            _save_flip_alert_state(symbol, "at_flip")


def _pin_alert_state_file() -> Path:
    return ROOT / "data" / "gex_pin_alert_state.json"


def _load_pin_alert_state() -> dict[str, str | None]:
    path = _pin_alert_state_file()
    if not path.exists():
        return {}
    try:
        with open(path) as f:
            data = json.load(f)
        return {str(k).upper(): (v if v is None else str(v)) for k, v in (data or {}).items()}
    except Exception as e:
        logger.warning("Could not load pin alert state %s: %s", path, e)
        return {}


def _save_pin_alert_state(symbol: str, which: str | None) -> None:
    path = _pin_alert_state_file()
    path.parent.mkdir(parents=True, exist_ok=True)
    current = _load_pin_alert_state()
    current[symbol.upper()] = which
    try:
        with open(path, "w") as f:
            json.dump(current, f, indent=2)
    except Exception as e:
        logger.warning("Could not save pin alert state %s: %s", path, e)


def _maybe_send_pin_alert(symbol: str, cockpit_result: dict) -> None:
    """If spot at call or put wall, send one FCM per wall type; clear state when spot leaves both walls."""
    regime = cockpit_result.get("regime") or {}
    structure = cockpit_result.get("structure") or {}
    primary = structure.get("primary_walls") or {}
    spot = regime.get("spot")
    call_wall = primary.get("call")
    put_wall = primary.get("put")
    if spot is None or (call_wall is None and put_wall is None):
        return
    spot = float(spot)
    dist_call = abs(spot - float(call_wall)) if call_wall is not None else float("inf")
    dist_put = abs(spot - float(put_wall)) if put_wall is not None else float("inf")
    last = _load_pin_alert_state().get(symbol)
    if dist_call > PIN_WALL_LEAVE_PTS and dist_put > PIN_WALL_LEAVE_PTS:
        _save_pin_alert_state(symbol, None)
        return
    if dist_call <= PIN_WALL_DISTANCE_PTS and last != "call":
        title = f"{symbol} pinning at call wall"
        body = "Price at/near call wall – potential resistance."
        ok = send_fcm_to_topic(FCM_TOPIC_MARKET_ALERTS, title=title, body=body, data={"symbol": symbol, "type": "pin_call"})
        if ok:
            log_notification_sent("pin_call", symbol, {"spot": spot, "call_wall": call_wall, "put_wall": put_wall})
            _save_pin_alert_state(symbol, "call")
    elif dist_put <= PIN_WALL_DISTANCE_PTS and last != "put":
        title = f"{symbol} pinning at put wall"
        body = "Price at/near put wall – potential support."
        ok = send_fcm_to_topic(FCM_TOPIC_MARKET_ALERTS, title=title, body=body, data={"symbol": symbol, "type": "pin_put"})
        if ok:
            log_notification_sent("pin_put", symbol, {"spot": spot, "call_wall": call_wall, "put_wall": put_wall})
            _save_pin_alert_state(symbol, "put")


# FCM topic for push notifications (app subscribes to this; server sends here)
FCM_TOPIC_MARKET_ALERTS = "market_alerts"

# Notification log (append-only JSONL for backtesting)
def _notification_log_file() -> Path:
    return ROOT / "data" / "notification_log.jsonl"


def log_notification_sent(alert_type: str, symbol: str, context: dict | None = None) -> None:
    """Append one notification record to data/notification_log.jsonl for backtesting."""
    path = _notification_log_file()
    path.parent.mkdir(parents=True, exist_ok=True)
    ctx = {k: v for k, v in (context or {}).items() if v is not None}
    rec = {
        "ts": datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z",
        "type": alert_type,
        "symbol": symbol,
        **ctx,
    }
    try:
        with open(path, "a") as f:
            f.write(json.dumps(rec) + "\n")
            f.flush()
    except Exception as e:
        logger.warning("Could not write notification log: %s", e)


def _ensure_fcm_initialized() -> bool:
    """Lazy-init Firebase Admin from service account JSON. Returns True if ready to send."""
    try:
        import firebase_admin
        from firebase_admin import credentials
    except ImportError:
        logger.debug("firebase_admin not installed; FCM send skipped")
        return False
    if getattr(firebase_admin, "_apps", None) and firebase_admin._apps:
        return True
    path = (os.environ.get("GOOGLE_APPLICATION_CREDENTIALS") or os.environ.get("FIREBASE_SERVICE_ACCOUNT_JSON") or "").strip()
    if not path:
        logger.debug("FCM: no service account path (GOOGLE_APPLICATION_CREDENTIALS or FIREBASE_SERVICE_ACCOUNT_JSON); skip")
        return False
    path = os.path.expanduser(path)
    if not os.path.isabs(path):
        path = str((ROOT / os.path.normpath(path)).resolve())
    if not os.path.isfile(path):
        logger.debug("FCM: credentials file not found at %s; skip", path)
        return False
    try:
        with open(path) as f:
            service_account_info = json.load(f)
        cred = credentials.Certificate(service_account_info)
        firebase_admin.initialize_app(cred)
        logger.debug("FCM initialized from %s", path)
        return True
    except Exception as e:
        logger.warning("FCM init failed: %s", e)
        return False


def send_fcm_to_topic(
    topic: str,
    title: str,
    body: str,
    data: dict | None = None,
) -> bool:
    """Send a notification to an FCM topic. Returns True on success. Data values must be strings."""
    if not _ensure_fcm_initialized():
        return False
    try:
        from firebase_admin import messaging
        msg_data = {str(k): str(v) for k, v in (data or {}).items()}
        android = messaging.AndroidConfig(
            priority="high",
            notification=messaging.AndroidNotification(
                channel_id="market_alerts",
                priority="max",
                default_sound=True,
                default_vibrate_timings=True,
            ),
        )
        message = messaging.Message(
            notification=messaging.Notification(title=title, body=body),
            data=msg_data,
            topic=topic,
            android=android,
        )
        messaging.send(message)
        logger.info("FCM sent to topic %s: %s", topic, title)
        return True
    except Exception as e:
        logger.warning("FCM send failed: %s", e)
        return False


def on_gex_regime_flip(
    symbol: str, old_regime: str, new_regime: str,
    spot: float | None = None, flip_line: float | None = None,
) -> None:
    """
    Called when GEX regime flips intraday. Logs and sends FCM push to market_alerts topic if configured.
    """
    msg = (
        f"GEX regime flip: {symbol} {old_regime} -> {new_regime}. "
        + ("Consider theta collection." if new_regime == "NEGATIVE" else "Consider directional.")
    )
    logger.warning("[ALERT] %s", msg)
    title = f"GEX: {symbol} regime flip"
    ok = send_fcm_to_topic(
        FCM_TOPIC_MARKET_ALERTS,
        title=title,
        body=msg,
        data={"symbol": symbol, "new_regime": new_regime},
    )
    if ok:
        log_notification_sent(
            "regime_flip", symbol,
            {"old_regime": old_regime, "new_regime": new_regime, "spot": spot, "flip_line": flip_line},
        )


def get_core_symbols() -> list[str]:
    """Load GEX_CORE_SYMBOLS from data/config.json or return default."""
    config_path = ROOT / "data" / "config.json"
    if config_path.exists():
        try:
            with open(config_path) as f:
                config = json.load(f)
            symbols = config.get("GEX_CORE_SYMBOLS")
            if symbols and isinstance(symbols, list):
                return [str(s).strip().upper() for s in symbols if s]
        except Exception as e:
            logger.warning("Could not load GEX_CORE_SYMBOLS from config: %s", e)
    return list(DEFAULT_CORE_SYMBOLS)


def get_api_base_url() -> str:
    """API base URL from env or config (localhost API that performs GEX/cockpit)."""
    url = os.environ.get("API_BASE_URL", "").strip().rstrip("/")
    if url:
        return url
    config_path = ROOT / "data" / "config.json"
    if config_path.exists():
        try:
            with open(config_path) as f:
                config = json.load(f)
            url = (config.get("API_BASE_URL") or "").strip().rstrip("/")
            if url:
                return url
        except Exception:
            pass
    return DEFAULT_API_BASE_URL


def get_api_key() -> str:
    """Optional x-api-key for localhost API (from env or config)."""
    key = os.environ.get("API_SECRET_KEY", "").strip()
    if key:
        return key
    config_path = ROOT / "data" / "config.json"
    if config_path.exists():
        try:
            with open(config_path) as f:
                config = json.load(f)
            return (config.get("API_SECRET_KEY") or "").strip()
        except Exception:
            pass
    return ""


def _sanitize_for_json(obj):
    """Return a JSON-serializable copy of obj (handles float nan/inf, datetime, etc.)."""
    if obj is None:
        return None
    if isinstance(obj, (str, int, bool)):
        return obj
    if isinstance(obj, float):
        if obj != obj:  # NaN
            return None
        if obj == float("inf"):
            return None
        if obj == float("-inf"):
            return None
        return obj
    if isinstance(obj, (list, tuple)):
        return [_sanitize_for_json(x) for x in obj]
    if isinstance(obj, dict):
        return {str(k): _sanitize_for_json(v) for k, v in obj.items()}
    if hasattr(obj, "isoformat"):  # datetime, date
        return obj.isoformat()
    return str(obj)


def fetch_from_api(base_url: str, path: str, api_key: str) -> dict | None:
    """GET base_url + path; return JSON dict or None on error."""
    url = f"{base_url.rstrip('/')}{path}"
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["x-api-key"] = api_key
    try:
        req = Request(url, headers=headers, method="GET")
        with urlopen(req, timeout=120) as resp:
            data = resp.read().decode()
            if not data:
                logger.warning("API %s: empty response body", url)
                return None
            out = json.loads(data)
            if not isinstance(out, dict):
                logger.warning("API %s: response is not a JSON object (got %s)", url, type(out).__name__)
                return None
            return out
    except HTTPError as e:
        body = e.read().decode() if e.fp else ""
        logger.warning("API %s: HTTP %s %s", url, e.code, body[:200] if body else "")
        return None
    except (URLError, json.JSONDecodeError, OSError) as e:
        logger.warning("API %s: %s", url, e)
        return None


def get_supabase_client():
    """Return Supabase client if SUPABASE_URL and SUPABASE_SECRET_KEY (or legacy SUPABASE_SERVICE_ROLE_KEY) are set, else None."""
    url = (os.environ.get("SUPABASE_URL") or "").strip().rstrip("/")
    key = (os.environ.get("SUPABASE_SECRET_KEY") or os.environ.get("SUPABASE_SERVICE_ROLE_KEY") or "").strip()
    if not url or not key:
        return None
    try:
        from supabase import create_client
        return create_client(url, key)
    except ImportError:
        logger.warning("supabase package not installed; pip install supabase to use client instead of Postgres URL.")
        return None


def get_cache_db_url() -> str | None:
    """Postgres URL for cache: prefer SUPABASE_DB_URL so the app can read the cache."""
    url = os.environ.get("SUPABASE_DB_URL") or os.environ.get("DATABASE_URL")
    if url and "REGION" in url.upper():
        raise ValueError(
            "SUPABASE_DB_URL contains placeholder 'REGION'. "
            "Copy the full Postgres URI from Supabase: Project Settings → Database → Connection string (URI). "
            "The host will look like aws-0-us-east-1.pooler.supabase.com (use your actual region, e.g. us-east-1)."
        )
    return url


@contextmanager
def get_cache_connection():
    """Context manager for Postgres connection to the cache DB (prefers SUPABASE_DB_URL)."""
    try:
        import psycopg2
        from psycopg2.extras import RealDictCursor
    except ImportError:
        raise RuntimeError("Install psycopg2-binary for database support.")
    url = get_cache_db_url()
    if not url:
        raise RuntimeError("Set SUPABASE_DB_URL or DATABASE_URL in .env")
    try:
        conn = psycopg2.connect(url)
    except Exception as e:
        err = str(e).lower()
        if "port" in err and "integer" in err:
            raise RuntimeError(
                "Connection URL parse error (often caused by ! in password). "
                "In .env use URL-encoded password: ! -> %21 (e.g. !! -> %21%21). "
                "Example: postgresql://postgres:MyPass%21%21@db.xxx.supabase.co:5432/postgres"
            ) from e
        if "could not translate host name" in err or "nodename nor servname" in err:
            raise RuntimeError(
                "Cannot resolve Supabase direct DB host (db.xxx.supabase.co). "
                "Use the Supabase API instead: pip install supabase, then set SUPABASE_URL and SUPABASE_SECRET_KEY in .env. "
                "The script will then use the Supabase client and avoid direct Postgres connection."
            ) from e
        raise
    conn.cursor_factory = RealDictCursor
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def compute_result_cache_exists(conn) -> bool:
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT 1 FROM information_schema.tables
            WHERE table_schema = 'public' AND table_name = 'compute_result_cache'
            """
        )
        return cur.fetchone() is not None


def _strip_sql_comments(segment: str) -> str:
    lines = []
    for line in segment.splitlines():
        stripped = line.strip()
        if stripped.startswith("--") or not stripped:
            continue
        lines.append(line)
    return "\n".join(lines).strip()


def ensure_compute_result_cache(conn) -> None:
    """Create compute_result_cache table if it does not exist."""
    if compute_result_cache_exists(conn):
        return
    schema_path = ROOT / "scripts" / "compute_result_cache_schema.sql"
    if not schema_path.exists():
        raise RuntimeError(f"Schema file not found: {schema_path}. Run it in Supabase SQL editor.")
    sql = schema_path.read_text()
    raw_statements = [s.strip() for s in sql.split(";") if s.strip()]
    statements = []
    for s in raw_statements:
        stmt = _strip_sql_comments(s)
        if stmt:
            statements.append(stmt)
    with conn.cursor() as cur:
        for stmt in statements:
            try:
                cur.execute(stmt)
            except Exception as e:
                msg = str(e).lower()
                if "already exists" in msg or "duplicate" in msg:
                    continue
                raise
    conn.commit()
    logger.info("Created compute_result_cache table.")


def upsert_cache_pg(conn, symbol: str, task_type: str, result: dict) -> None:
    """Upsert one row using psycopg2 connection."""
    with conn.cursor() as cur:
        cur.execute(
            """
            INSERT INTO compute_result_cache (symbol, task_type, result, updated_at)
            VALUES (%s, %s, %s, NOW())
            ON CONFLICT (symbol, task_type) DO UPDATE
            SET result = EXCLUDED.result, updated_at = NOW()
            """,
            (symbol.upper(), task_type, json.dumps(result)),
        )
    conn.commit()


def _row_for_cache(symbol: str, task_type: str, result: dict) -> dict:
    """Build one cache row with JSON-safe result. Include updated_at so Supabase upsert refreshes it."""
    return {
        "symbol": symbol.upper(),
        "task_type": task_type,
        "result": _sanitize_for_json(result) or {},
        "updated_at": datetime.utcnow().isoformat() + "Z",
    }


def upsert_cache_supabase_batch(supabase, rows: list[dict]) -> None:
    """Upsert multiple rows into compute_result_cache. Raises on API error."""
    if not rows:
        return
    try:
        resp = (
            supabase.table("compute_result_cache")
            .upsert(rows, on_conflict="symbol,task_type")
            .execute()
        )
        if getattr(resp, "count", None) is not None:
            logger.debug("Supabase upsert response count: %s", resp.count)
    except Exception as e:
        err_msg = str(e)
        if hasattr(e, "message"):
            err_msg = getattr(e, "message", err_msg)
        if hasattr(e, "details"):
            err_msg = f"{err_msg} details={e.details}"
        if hasattr(e, "body"):
            err_msg = f"{err_msg} body={e.body}"
        logger.error("Supabase upsert failed: %s", err_msg)
        raise RuntimeError(f"Supabase upsert failed: {err_msg}") from e


def get_db_hint(url: str) -> str:
    """Hint for user: ensure URL is Supabase if app reads from Supabase."""
    if not url:
        return ""
    u = url.lower()
    if "supabase" in u or "supabase.co" in u:
        return " (Supabase – app can read cache)"
    if "localhost" in u or "127.0.0.1" in u:
        return " (local Postgres – app will NOT see cache unless it also uses this DB)"
    return ""


def main() -> int:
    # Only run during market window (8:30–17:00 ET, Mon–Fri) or when someone enqueued a job
    try:
        from market_window import should_run_scheduled_publish
        if not should_run_scheduled_publish(check_queue=True):
            logger.info("Outside market window (8:30–17:00 ET, Mon–Fri) and no pending queue jobs; skipping.")
            return 0
    except ImportError:
        pass

    supabase = get_supabase_client()
    db_url = get_cache_db_url() if not supabase else None

    if supabase:
        logger.info("Using Supabase Python client (SUPABASE_URL + SUPABASE_SECRET_KEY)")
    elif db_url:
        which = "SUPABASE_DB_URL" if os.environ.get("SUPABASE_DB_URL") else "DATABASE_URL"
        hint = get_db_hint(db_url)
        logger.info("Using %s%s", which, hint)
        if not os.environ.get("SUPABASE_DB_URL"):
            logger.info(
                "Tip: Add SUPABASE_URL and SUPABASE_SECRET_KEY to .env (Project Settings -> API) "
                "to use the Supabase client and avoid Postgres URI / IPv6 / password encoding issues."
            )
    else:
        logger.error(
            "Set SUPABASE_URL and SUPABASE_SECRET_KEY (preferred), or SUPABASE_DB_URL/DATABASE_URL. "
            "See docstring and docs/COMPUTE_QUEUE.md."
        )
        return 1

    symbols = get_core_symbols()
    base_url = get_api_base_url()
    api_key = get_api_key()
    logger.info("Core symbols: %s; API: %s", symbols, base_url)

    if not supabase:
        with get_cache_connection() as conn:
            ensure_compute_result_cache(conn)

    cache_rows: list[dict] = []  # for batch Supabase upsert
    upsert_count = 0

    def do_upsert(symbol: str, task_type: str, result: dict) -> None:
        nonlocal upsert_count
        if supabase:
            cache_rows.append(_row_for_cache(symbol, task_type, result))
        else:
            safe = _sanitize_for_json(result) if isinstance(result, dict) else {}
            with get_cache_connection() as conn:
                upsert_cache_pg(conn, symbol, task_type, safe or result)
        upsert_count += 1

    # GEX: GET /gex/calculate?ticker=SYMBOL
    for symbol in symbols:
        try:
            result = fetch_from_api(
                base_url,
                f"/gex/calculate?ticker={symbol}",
                api_key,
            )
            if result is None:
                logger.warning("GEX %s: skipping (API returned no data or request failed)", symbol)
                continue
            if result.get("error"):
                logger.warning("GEX %s: skipping (API error: %s)", symbol, result.get("error"))
                continue
            do_upsert(symbol, "gex", result)
            logger.info("Upserted %s gex", symbol)
        except Exception as e:
            logger.exception("Failed %s gex: %s", symbol, e)

    # Cockpit: GET /cockpit/state?ticker=SYMBOL (and GEX regime flip detection for SPX, SPY, XSP)
    last_regimes = _load_last_regimes()
    for symbol in symbols:
        try:
            result = fetch_from_api(
                base_url,
                f"/cockpit/state?ticker={symbol}",
                api_key,
            )
            if result is None:
                logger.warning("Cockpit %s: skipping (API returned no data or request failed)", symbol)
                continue
            if result.get("error"):
                logger.warning("Cockpit %s: skipping (API error: %s)", symbol, result.get("error"))
                continue
            do_upsert(symbol, "cockpit", result)
            logger.info("Upserted %s cockpit", symbol)
            # GEX regime flip and flip-approach alerts (SPX, SPY, XSP)
            if symbol in GEX_REGIME_ALERT_SYMBOLS:
                regime_data = result.get("regime") or {}
                spot = regime_data.get("spot")
                flip_line = regime_data.get("flip_line")
                current = _extract_regime_from_cockpit(result)
                if current:
                    previous = last_regimes.get(symbol)
                    if previous is not None and previous != current:
                        on_gex_regime_flip(symbol, previous, current, spot=spot, flip_line=flip_line)
                        if current == "NEGATIVE":
                            _save_flip_alert_state(symbol, "negative")
                    last_regimes[symbol] = current
                    _save_last_regime(symbol, current)
                # Flip approach: approaching / at flip (one notification per stage)
                if symbol == "SPX":
                    _maybe_send_flip_approach_alert(symbol, result)
                # Pinning: at call or put wall
                if symbol == "SPX":
                    _maybe_send_pin_alert(symbol, result)
        except Exception as e:
            logger.exception("Failed %s cockpit: %s", symbol, e)

    # Trade ideas: GET /trade-ideas/allowed for core symbols (so app reads from cache)
    for symbol in symbols:
        try:
            result = fetch_from_api(
                base_url,
                f"/trade-ideas/allowed?ticker={symbol}&max_ideas=3&timeframe=all",
                api_key,
            )
            if result is None:
                logger.warning("Trade ideas %s: skipping (API returned no data or request failed)", symbol)
                continue
            if result.get("error"):
                logger.warning("Trade ideas %s: skipping (API error: %s)", symbol, result.get("error"))
                continue
            do_upsert(symbol, "trade_ideas", result)
            logger.info("Upserted %s trade_ideas", symbol)
        except Exception as e:
            logger.exception("Failed %s trade_ideas: %s", symbol, e)

    # Flush batch to Supabase
    if supabase and cache_rows:
        try:
            upsert_cache_supabase_batch(supabase, cache_rows)
            logger.info("Supabase batch upserted %s rows", len(cache_rows))
        except Exception as e:
            logger.exception("Supabase batch upsert failed: %s", e)
            return 1

    # Verify row count in cache
    try:
        if supabase:
            resp = supabase.table("compute_result_cache").select("symbol").execute()
            n = len(resp.data) if getattr(resp, "data", None) else 0
        else:
            with get_cache_connection() as conn:
                with conn.cursor() as cur:
                    cur.execute("SELECT COUNT(*) FROM compute_result_cache")
                    n = cur.fetchone()[0]
        logger.info("Precompute done. Rows in compute_result_cache: %s", n)
        if n == 0:
            logger.warning(
                "No rows in cache. Check: (1) API is running at %s and returns 200, "
                "(2) Table compute_result_cache exists (run compute_result_cache_schema.sql in Supabase SQL editor).",
                base_url,
            )
    except Exception as e:
        logger.warning("Could not verify cache row count: %s", e)

    return 0


if __name__ == "__main__":
    sys.exit(main())
