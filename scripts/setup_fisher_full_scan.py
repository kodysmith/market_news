#!/usr/bin/env python3
"""
One-time setup for Fisher full-market scan on a server.

Run from repo root with DATABASE_URL set. This script:
1. Fetches SEC universe (data/sec_universe.json) if missing or empty
2. Applies fisher_scan_queue schema to the database if not already applied
3. Optionally enqueues all SEC tickers (--enqueue or --reset)

Usage:
  export DATABASE_URL="postgresql://..."
  python3 scripts/setup_fisher_full_scan.py
  python3 scripts/setup_fisher_full_scan.py --enqueue    # also enqueue (no reset)
  python3 scripts/setup_fisher_full_scan.py --reset      # clear queue and enqueue all
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


def need_sec_universe() -> bool:
    from fisher.config import get_sec_universe
    constituents = get_sec_universe()
    return not constituents


def run_fetch_sec_universe() -> bool:
    print("Fetching SEC universe (data/sec_universe.json)...")
    r = subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "fetch_sec_universe.py")],
        cwd=str(ROOT),
        capture_output=True,
        text=True,
    )
    if r.returncode != 0:
        print(r.stderr or r.stdout, file=sys.stderr)
        return False
    print(r.stdout or "Done.")
    return True


def queue_table_exists(conn) -> bool:
    from fisher.db import get_supabase_client, _is_supabase_client
    if _is_supabase_client(conn):
        try:
            conn.table("fisher_scan_queue").select("id").limit(1).execute()
            return True
        except Exception:
            return False
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT 1 FROM information_schema.tables
            WHERE table_schema = 'public' AND table_name = 'fisher_scan_queue'
            """
        )
        return cur.fetchone() is not None


def apply_queue_schema(conn) -> bool:
    from fisher.db import _is_supabase_client
    if _is_supabase_client(conn):
        print("When using Supabase client, run scripts/fisher_scan_queue_schema.sql in Supabase SQL editor.")
        return True
    schema_path = ROOT / "scripts" / "fisher_scan_queue_schema.sql"
    if not schema_path.exists():
        print(f"Schema file not found: {schema_path}", file=sys.stderr)
        return False
    sql = schema_path.read_text()
    # Run each statement (split by ;, skip empty and comments)
    statements = [
        s.strip() for s in sql.split(";")
        if s.strip() and not s.strip().startswith("--")
    ]
    with conn.cursor() as cur:
        for stmt in statements:
            if not stmt:
                continue
            try:
                cur.execute(stmt)
            except Exception as e:
                msg = str(e).lower()
                if "already exists" in msg or "duplicate" in msg:
                    continue
                raise
    conn.commit()
    return True


def run_enqueue(reset: bool) -> bool:
    args = [sys.executable, str(ROOT / "scripts" / "enqueue_fisher_scan.py")]
    if reset:
        args.append("--reset")
    print("Enqueueing SEC tickers...")
    r = subprocess.run(args, cwd=str(ROOT))
    return r.returncode == 0


def main() -> int:
    ap = argparse.ArgumentParser(description="Setup Fisher full-market scan (SEC universe + queue table)")
    ap.add_argument("--enqueue", action="store_true", help="Enqueue all tickers after setup (no reset)")
    ap.add_argument("--reset", action="store_true", help="Clear queue and enqueue all (for monthly rescan)")
    args = ap.parse_args()

    from fisher.config import get_database_url
    from fisher.db import get_supabase_client
    if not get_database_url() and not get_supabase_client():
        print("Set SUPABASE_URL and SUPABASE_SECRET_KEY, or DATABASE_URL/SUPABASE_DB_URL.", file=sys.stderr)
        return 1

    # 1. SEC universe
    if need_sec_universe():
        if not run_fetch_sec_universe():
            return 1
    else:
        print("SEC universe already present (data/sec_universe.json).")

    # 2. Queue table
    from fisher.db import get_connection
    with get_connection() as conn:
        if queue_table_exists(conn):
            print("Queue table fisher_scan_queue already exists.")
        else:
            print("Applying fisher_scan_queue schema...")
            apply_queue_schema(conn)
            print("Queue table created.")

    # 3. Optional enqueue
    if args.reset or args.enqueue:
        if not run_enqueue(reset=args.reset):
            return 1

    print("")
    print("Setup complete. Run the worker:")
    print("  python3 scripts/run_fisher_scan_worker.py --batch 50")
    print("Or one batch (e.g. cron):")
    print("  python3 scripts/run_fisher_scan_worker.py --once --batch 50")
    return 0


if __name__ == "__main__":
    sys.exit(main())
