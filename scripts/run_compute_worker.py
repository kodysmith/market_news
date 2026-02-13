#!/usr/bin/env python3
"""
Process compute_job_queue: claim pending jobs, run the corresponding API logic, write result to Supabase.

Run on the private server (same machine as the heavy API). Loads .env from repo root. Uses SUPABASE_URL + SUPABASE_SECRET_KEY (preferred) or DATABASE_URL/SUPABASE_DB_URL.
Reuses Flask API handlers in-process via test_request_context so no HTTP or code duplication.

Usage:
  python scripts/run_compute_worker.py           # run one job and exit
  python scripts/run_compute_worker.py --once    # same as default, one job
  python scripts/run_compute_worker.py --loop    # drain queue then exit
  python scripts/run_compute_worker.py --daemon   # run forever, sleep when empty (Ctrl+C to stop)
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import threading
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

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

def _is_supabase_client(conn) -> bool:
    return conn is not None and hasattr(conn, "table")


# Task type -> (path_template, view_callable)
# Path template uses {symbol}. View is the Flask view function.
def _get_app_and_routes():
    from apis.app_factory import create_app
    app = create_app()
    with app.app_context():
        from apis.routes_valuation import valuation_calculate
        from apis.routes_gex import calculate_gex
        from apis.routes_cockpit import cockpit_state, trade_ideas_allowed
        from apis.routes_probability import probability_range
    return app, {
        "valuation": ("/valuation/calculate?ticker={symbol}&store=false", valuation_calculate),
        "gex": ("/gex/calculate?ticker={symbol}", calculate_gex),
        "cockpit": ("/cockpit/state?ticker={symbol}", cockpit_state),
        "probability": ("/probability/range?ticker={symbol}", probability_range),
        "trade_ideas": ("/trade-ideas/allowed?ticker={symbol}", trade_ideas_allowed),
    }


def compute_job_queue_exists(conn) -> bool:
    if _is_supabase_client(conn):
        try:
            conn.table("compute_job_queue").select("id").limit(1).execute()
            return True
        except Exception:
            return False
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT 1 FROM information_schema.tables
            WHERE table_schema = 'public' AND table_name = 'compute_job_queue'
            """
        )
        return cur.fetchone() is not None


def _strip_sql_comments(segment: str) -> str:
    """Remove leading comment lines so CREATE TABLE etc. are not skipped."""
    lines = []
    for line in segment.splitlines():
        stripped = line.strip()
        if stripped.startswith("--") or not stripped:
            continue
        lines.append(line)
    return "\n".join(lines).strip()


def ensure_compute_job_queue(conn) -> None:
    """Create compute_job_queue table if it does not exist (Postgres only). With Supabase client, table must exist."""
    if compute_job_queue_exists(conn):
        return
    if _is_supabase_client(conn):
        raise RuntimeError(
            "Table compute_job_queue not found. Run scripts/compute_job_queue_schema.sql in Supabase SQL editor."
        )
    schema_path = ROOT / "scripts" / "compute_job_queue_schema.sql"
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
    logger.info("Created compute_job_queue table.")


def claim_one(conn):
    """Claim one pending job; return (id, symbol, task_type) or (None, None, None)."""
    if _is_supabase_client(conn):
        r = (
            conn.table("compute_job_queue")
            .select("id, symbol, task_type")
            .eq("status", "pending")
            .order("requested_at")
            .limit(1)
            .execute()
        )
        rows = getattr(r, "data", None) or []
        if not rows:
            return None, None, None
        job_id, symbol, task_type = rows[0]["id"], rows[0]["symbol"], rows[0]["task_type"]
        now = datetime.now(timezone.utc).isoformat()
        up = (
            conn.table("compute_job_queue")
            .update({"status": "processing", "claimed_at": now})
            .eq("id", job_id)
            .eq("status", "pending")
            .execute()
        )
        updated = getattr(up, "data", None) or []
        if not updated:
            return None, None, None
        return job_id, symbol, task_type
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT id, symbol, task_type
            FROM compute_job_queue
            WHERE status = 'pending'
            ORDER BY requested_at
            LIMIT 1
            FOR UPDATE SKIP LOCKED
            """
        )
        row = cur.fetchone()
    if not row:
        return None, None, None
    job_id, symbol, task_type = row["id"], row["symbol"], row["task_type"]
    with conn.cursor() as cur:
        cur.execute(
            """
            UPDATE compute_job_queue
            SET status = 'processing', claimed_at = NOW()
            WHERE id = %s
            """,
            (job_id,),
        )
    conn.commit()
    return job_id, symbol, task_type


def _upsert_result_cache(client, symbol: str, task_type: str, result: dict) -> None:
    """Upsert one row into compute_result_cache so the app's cache read sees it. Supabase client only."""
    try:
        now = datetime.now(timezone.utc).isoformat()
        row = {
            "symbol": symbol.upper(),
            "task_type": task_type,
            "result": result,
            "updated_at": now,
        }
        client.table("compute_result_cache").upsert(row, on_conflict="symbol,task_type").execute()
        logger.debug("Upserted cache %s %s", symbol, task_type)
    except Exception as e:
        logger.warning("Upsert compute_result_cache failed (table may not exist): %s", e)


def mark_done(conn, job_id, result: dict, symbol: str = "", task_type: str = ""):
    now = datetime.now(timezone.utc).isoformat()
    payload = {"status": "done", "result": result, "completed_at": now}
    if _is_supabase_client(conn):
        conn.table("compute_job_queue").update(payload).eq("id", job_id).execute()
        if symbol and task_type:
            _upsert_result_cache(conn, symbol, task_type, result)
        return
    with conn.cursor() as cur:
        cur.execute(
            """
            UPDATE compute_job_queue
            SET status = 'done', result = %s, completed_at = NOW()
            WHERE id = %s
            """,
            (json.dumps(result), job_id),
        )
    conn.commit()


def mark_failed(conn, job_id, error_text: str):
    now = datetime.now(timezone.utc).isoformat()
    payload = {"status": "failed", "error_text": (error_text[:5000] if error_text else None), "completed_at": now}
    if _is_supabase_client(conn):
        conn.table("compute_job_queue").update(payload).eq("id", job_id).execute()
        return
    with conn.cursor() as cur:
        cur.execute(
            """
            UPDATE compute_job_queue
            SET status = 'failed', error_text = %s, completed_at = NOW()
            WHERE id = %s
            """,
            (error_text[:5000] if error_text else None, job_id),
        )
    conn.commit()


def run_task(app, routes, symbol: str, task_type: str) -> dict:
    """Run the API handler for task_type and symbol; return JSON-serializable result."""
    if task_type not in routes:
        return {"error": f"Unknown task_type: {task_type}"}
    path_tpl, view = routes[task_type]
    path = path_tpl.format(symbol=symbol)
    with app.test_request_context(path):
        resp = view()
        data = resp.get_json()
        if resp.status_code != 200:
            return data if isinstance(data, dict) else {"error": str(data)}
        return data


def process_one(conn, app, routes) -> bool:
    """Claim one job, run it, write result. Return True if a job was processed."""
    job_id, symbol, task_type = claim_one(conn)
    if job_id is None:
        return False
    logger.info("Processing job %s: %s %s", job_id, task_type, symbol)
    try:
        result = run_task(app, routes, symbol, task_type)
        mark_done(conn, job_id, result, symbol=symbol, task_type=task_type)
        logger.info("Job %s done", job_id)
    except Exception as e:
        logger.exception("Job %s failed: %s", job_id, e)
        mark_failed(conn, job_id, str(e))
    return True


def _sleep_interval_seconds(args) -> tuple[float, str]:
    """Return (seconds, label) for idle sleep: use poll during market hours, poll_after_hours otherwise."""
    try:
        from market_window import is_market_window
        in_window = is_market_window()
    except ImportError:
        in_window = True
    if in_window:
        return args.poll, "market hours"
    return args.poll_after_hours, "after hours"


def _realtime_timeout_seconds(args) -> float:
    """Return timeout for Realtime wait: shorter during market hours, longer after hours."""
    try:
        from market_window import is_market_window
        in_window = is_market_window()
    except ImportError:
        in_window = True
    return args.poll if in_window else args.poll_after_hours


def _run_realtime_listener(wake_event: threading.Event, wss_url: str, api_key: str) -> None:
    """Run async Realtime listener in current thread; on INSERT into compute_job_queue call wake_event.set()."""
    import asyncio
    try:
        from realtime import AsyncRealtimeClient
    except ImportError:
        logger.warning("realtime package not installed; pip install realtime for push-based wake-up. Using poll.")
        return
    async def _listen() -> None:
        socket = AsyncRealtimeClient(wss_url, api_key)
        channel = socket.channel("compute_job_queue_worker")
        channel.on_postgres_changes(
            "INSERT",
            schema="public",
            table="compute_job_queue",
            callback=lambda payload: wake_event.set(),
        )
        await channel.subscribe()
        # Keep running until process exits (daemon thread)
        while True:
            await asyncio.sleep(60)
    try:
        asyncio.run(_listen())
    except Exception as e:
        logger.warning("Realtime listener stopped: %s", e)


def _loop_with_realtime(get_supabase_client, get_connection, app, routes, args) -> None:
    """Loop/daemon mode using Supabase Realtime: wake on INSERT, else wait with timeout (market/after-hours)."""
    if not get_supabase_client():
        raise RuntimeError("Realtime requires Supabase client (SUPABASE_URL + SUPABASE_SECRET_KEY)")
    url = (os.environ.get("SUPABASE_URL") or "").strip().rstrip("/")
    key = (os.environ.get("SUPABASE_SECRET_KEY") or os.environ.get("SUPABASE_SERVICE_ROLE_KEY") or "").strip()
    if not url or not key:
        logger.warning("SUPABASE_URL or SUPABASE_SECRET_KEY not set; Realtime disabled.")
        raise RuntimeError("Realtime requires SUPABASE_URL and SUPABASE_SECRET_KEY")
    wss_url = url.replace("https://", "wss://", 1).replace("http://", "ws://", 1)
    if not wss_url.startswith("ws"):
        wss_url = "wss://" + url.split("://", 1)[-1]
    wss_url = wss_url.rstrip("/") + "/realtime/v1"
    wake_event = threading.Event()
    try:
        from realtime import AsyncRealtimeClient
    except ImportError:
        raise RuntimeError("realtime package not installed; pip install realtime for push-based wake-up")
    listener_thread = threading.Thread(
        target=_run_realtime_listener,
        args=(wake_event, wss_url, key),
        daemon=True,
        name="realtime-listener",
    )
    listener_thread.start()
    logger.info("Realtime: listening for INSERT on compute_job_queue; wake timeout by market hours.")
    while True:
        wake_event.clear()
        timeout = _realtime_timeout_seconds(args)
        wake_event.wait(timeout=timeout)
        with get_connection() as conn:
            while process_one(conn, app, routes):
                pass


def main():
    ap = argparse.ArgumentParser(description="Process compute_job_queue (valuation, GEX, cockpit, etc.)")
    ap.add_argument("--loop", action="store_true", help="Run forever; when queue empty, sleep --poll s then check again (Ctrl+C to stop)")
    ap.add_argument("--daemon", action="store_true", help="Same as --loop (run forever, poll when empty)")
    ap.add_argument("--poll", type=float, default=30.0, help="Seconds to sleep when queue empty during market hours (default 30)")
    ap.add_argument("--poll-after-hours", type=float, default=300.0, help="Seconds to sleep when queue empty outside market hours (default 300)")
    ap.add_argument("--once", action="store_true", help="Process one job and exit (default)")
    args = ap.parse_args()

    from fisher.db import get_connection, get_supabase_client
    from fisher.config import get_database_url
    if not get_supabase_client() and not get_database_url():
        print("Set SUPABASE_URL + SUPABASE_SECRET_KEY, or DATABASE_URL/SUPABASE_DB_URL.", file=sys.stderr)
        return 1
    app, routes = _get_app_and_routes()

    with get_connection() as conn:
        ensure_compute_job_queue(conn)

    if args.loop or args.daemon:
        try:
            _loop_with_realtime(get_supabase_client, get_connection, app, routes, args)
            return 0
        except RuntimeError as e:
            logger.info("Realtime not used: %s; using poll loop.", e)
        except Exception as e:
            logger.warning("Realtime failed: %s; using poll loop.", e)
        poll_sec, poll_label = _sleep_interval_seconds(args)
        logger.info(
            "Loop mode: when queue empty, sleeping %s s (%s); Ctrl+C to stop",
            poll_sec, poll_label,
        )
        while True:
            with get_connection() as conn:
                if not process_one(conn, app, routes):
                    sleep_sec, label = _sleep_interval_seconds(args)
                    logger.info("Queue empty; sleeping %.0f s (%s)", sleep_sec, label)
                    time.sleep(sleep_sec)
                    continue

    with get_connection() as conn:
        if process_one(conn, app, routes):
            print("Processed one job.")
        else:
            print("No pending jobs.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
