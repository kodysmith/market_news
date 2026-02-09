#!/usr/bin/env python3
"""
Enqueue SEC-universe tickers into fisher_scan_queue for Fisher full-market scan.

Run fetch_sec_universe.py first to create data/sec_universe.json.
Monthly: use --reset to clear queue and re-enqueue all tickers.
Manual: run without --reset to add only tickers not already in the queue.

Use --process-after to run the worker after enqueueing so queued items are processed
immediately (claim from queue → EDGAR + facts → Fisher scoring → write to DB for Flutter).
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

# Load repo root .env so DATABASE_URL / SUPABASE_DB_URL are set
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

from fisher.config import get_sec_universe, get_database_url
from fisher.db import get_connection, get_supabase_client


def main() -> int:
    ap = argparse.ArgumentParser(description="Enqueue SEC tickers for Fisher full-market scan")
    ap.add_argument(
        "--reset",
        action="store_true",
        help="Clear queue and re-enqueue all SEC tickers (use for monthly full rescan)",
    )
    ap.add_argument(
        "--process-after",
        metavar="N",
        nargs="?",
        const=1,
        type=int,
        default=None,
        help="After enqueueing, run the worker: N batches (default 1), or 0 = until queue empty",
    )
    ap.add_argument(
        "--batch",
        type=int,
        default=50,
        help="Batch size for --process-after (default 50)",
    )
    ap.add_argument(
        "--delay",
        type=float,
        default=0,
        help="Seconds to sleep after each batch when using --process-after (default 0)",
    )
    args = ap.parse_args()

    if not get_database_url() and not get_supabase_client():
        print("Set SUPABASE_URL and SUPABASE_SECRET_KEY, or DATABASE_URL/SUPABASE_DB_URL.", file=sys.stderr)
        return 1

    constituents = get_sec_universe()
    if not constituents:
        print(
            "SEC universe is empty. Run: python3 scripts/fetch_sec_universe.py",
            file=sys.stderr,
        )
        return 1

    tickers = [c.get("ticker") for c in constituents if c.get("ticker")]
    tickers = [t.upper().strip() for t in tickers if t]
    if not tickers:
        print("No tickers in SEC universe.", file=sys.stderr)
        return 1

    client = get_supabase_client()
    if client is not None:
        if args.reset:
            # Delete all queue rows (Supabase requires a filter; use .neq('id', '00000000-0000-0000-0000-000000000000'))
            r = client.table("fisher_scan_queue").select("id").limit(1000).execute()
            ids = [row["id"] for row in (getattr(r, "data", None) or [])]
            while ids:
                for id_ in ids:
                    client.table("fisher_scan_queue").delete().eq("id", id_).execute()
                r = client.table("fisher_scan_queue").select("id").limit(1000).execute()
                ids = [row["id"] for row in (getattr(r, "data", None) or [])]
            print("Queue cleared.")
        for t in tickers:
            try:
                client.table("fisher_scan_queue").insert({"ticker": t, "status": "pending", "source": "sec"}).execute()
            except Exception:
                pass  # allow duplicate tickers
        r = client.table("fisher_scan_queue").select("id", count="exact").eq("status", "pending").execute()
        pending = getattr(r, "count", None) or len(getattr(r, "data", None) or [])
    else:
        with get_connection() as conn:
            with conn.cursor() as cur:
                if args.reset:
                    cur.execute("DELETE FROM fisher_scan_queue")
                    conn.commit()
                    print("Queue cleared.")

                for t in tickers:
                    cur.execute(
                        "INSERT INTO fisher_scan_queue (ticker, status, source) VALUES (%s, 'pending', 'sec')",
                        (t,),
                    )
                conn.commit()
                cur.execute("SELECT count(*) FROM fisher_scan_queue WHERE status = 'pending'")
                pending = cur.fetchone()["count"]

    print(f"Queue has {pending} pending tickers (SEC universe: {len(tickers)}).")

    if args.process_after is not None:
        worker_script = ROOT / "scripts" / "run_fisher_scan_worker.py"
        if not worker_script.exists():
            print("Worker script not found.", file=sys.stderr)
            return 1
        worker_cmd = [sys.executable, str(worker_script), "--batch", str(args.batch), "--delay", str(args.delay)]
        if args.process_after == 0:
            print("Running worker until queue empty...")
            result = subprocess.run(worker_cmd, cwd=str(ROOT), env=os.environ.copy())
        else:
            print(f"Running worker for {args.process_after} batch(es)...")
            result = subprocess.run(worker_cmd + ["--once"], cwd=str(ROOT), env=os.environ.copy())
            for _ in range(args.process_after - 1):
                if result.returncode != 0:
                    return result.returncode
                result = subprocess.run(worker_cmd + ["--once"], cwd=str(ROOT), env=os.environ.copy())
        if result.returncode != 0:
            return result.returncode

    else:
        print("Run: python3 scripts/run_fisher_scan_worker.py [--batch N] [--once]")
        print("Or enqueue and process now: python3 scripts/enqueue_fisher_scan.py --process-after 1")
    return 0


if __name__ == "__main__":
    sys.exit(main())
