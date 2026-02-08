#!/usr/bin/env python3
"""
Enqueue all SEC-universe tickers into fisher_scan_queue for full-market Fisher scan.

Run fetch_sec_universe.py first to create data/sec_universe.json.
Monthly: use --reset to clear queue and re-enqueue all tickers.
Manual: run without --reset to add only tickers not already in the queue.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from fisher.config import get_sec_universe, get_database_url
from fisher.db import get_connection


def main() -> int:
    ap = argparse.ArgumentParser(description="Enqueue SEC tickers for Fisher full-market scan")
    ap.add_argument(
        "--reset",
        action="store_true",
        help="Clear queue and re-enqueue all SEC tickers (use for monthly full rescan)",
    )
    args = ap.parse_args()

    if not get_database_url():
        print("Set DATABASE_URL or SUPABASE_DB_URL.", file=sys.stderr)
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

    with get_connection() as conn:
        with conn.cursor() as cur:
            if args.reset:
                cur.execute("DELETE FROM fisher_scan_queue")
                conn.commit()
                print("Queue cleared.")

            cur.executemany(
                """
                INSERT INTO fisher_scan_queue (ticker, status, source)
                VALUES (%s, 'pending', 'sec')
                ON CONFLICT (ticker) DO NOTHING
                """,
                [(t,) for t in tickers],
            )
            conn.commit()
            cur.execute("SELECT count(*) FROM fisher_scan_queue WHERE status = 'pending'")
            pending = cur.fetchone()["count"]

    print(f"Queue has {pending} pending tickers (SEC universe: {len(tickers)}).")
    print("Run: python3 scripts/run_fisher_scan_worker.py [--batch N] [--once]")
    return 0


if __name__ == "__main__":
    sys.exit(main())
