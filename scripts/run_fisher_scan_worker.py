#!/usr/bin/env python3
"""
Process fisher_scan_queue in batches: EDGAR + facts, then Fisher scoring.

Run after enqueue_fisher_scan.py. Processes pending tickers in batches (SEC rate limits).
Use --once to process one batch and exit (e.g. cron every hour). Use default --loop until queue empty.

No market-window check: this is a long-running queue process and can run 24/7 (unlike GEX/cockpit precompute).
"""
from __future__ import annotations

import argparse
import logging
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

# Load repo root .env so DATABASE_URL / SUPABASE_DB_URL are set (e.g. for cron)
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

from fisher.db import get_connection, company_id_by_ticker, _is_supabase_client, claim_batch_supabase, mark_done_supabase
from fisher.edgar_watcher import run_watcher
from fisher.scoring_engine import run_scoring_job

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


def claim_batch(conn, batch_size: int):
    """Claim up to batch_size pending rows; return list of (id, ticker). Works with Supabase client or Postgres conn."""
    if _is_supabase_client(conn):
        return claim_batch_supabase(conn, batch_size)
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT id, ticker
            FROM fisher_scan_queue
            WHERE status = 'pending'
            ORDER BY queued_at
            LIMIT %s
            FOR UPDATE SKIP LOCKED
            """,
            (batch_size,),
        )
        rows = list(cur.fetchall())
    if not rows:
        return []
    ids = [r["id"] for r in rows]
    with conn.cursor() as cur:
        cur.execute(
            """
            UPDATE fisher_scan_queue
            SET status = 'processing', started_at = NOW()
            WHERE id = ANY(%s::uuid[])
            """,
            (ids,),
        )
    conn.commit()
    return [(r["id"], r["ticker"]) for r in rows]


def mark_done(conn, ids, error_text=None):
    """Mark queue rows done or failed. Works with Supabase client or Postgres conn."""
    if _is_supabase_client(conn):
        mark_done_supabase(conn, ids, error_text)
        return
    with conn.cursor() as cur:
        if error_text:
            cur.execute(
                """
                UPDATE fisher_scan_queue
                SET status = 'failed', completed_at = NOW(), error_text = %s
                WHERE id = ANY(%s::uuid[])
                """,
                (error_text[:2000], ids),
            )
        else:
            cur.execute(
                """
                UPDATE fisher_scan_queue
                SET status = 'done', completed_at = NOW(), error_text = NULL
                WHERE id = ANY(%s::uuid[])
                """,
                (ids,),
            )
    conn.commit()


def process_batch(batch_size: int, delay_after_batch: float) -> int:
    """Process one batch; return number processed, or 0 if queue empty."""
    with get_connection() as conn:
        batch = claim_batch(conn, batch_size)
    if not batch:
        return 0

    ids = [b[0] for b in batch]
    tickers = [b[1] for b in batch]

    try:
        logger.info("Watcher: %s tickers", len(tickers))
        run_watcher(tickers=tickers, universe="sec")
        with get_connection() as conn:
            company_ids = []
            for t in tickers:
                cid = company_id_by_ticker(conn, t)
                if cid:
                    company_ids.append(cid)
        if company_ids:
            logger.info("Scoring: %s companies", len(company_ids))
            run_scoring_job(company_ids=company_ids)
        with get_connection() as conn:
            mark_done(conn, ids)
        logger.info("Batch done: %s tickers", len(tickers))
        if delay_after_batch > 0:
            time.sleep(delay_after_batch)
        return len(tickers)
    except Exception as e:
        logger.exception("Batch failed: %s", e)
        with get_connection() as conn:
            mark_done(conn, ids, error_text=str(e))
        return len(tickers)  # still consumed from queue (marked failed)


def main() -> int:
    ap = argparse.ArgumentParser(description="Process Fisher scan queue (EDGAR + scoring)")
    ap.add_argument("--batch", type=int, default=50, help="Tickers per batch (default 50)")
    ap.add_argument(
        "--once",
        action="store_true",
        help="Process one batch and exit (for cron)",
    )
    ap.add_argument(
        "--delay",
        type=float,
        default=0,
        help="Seconds to sleep after each batch (default 0)",
    )
    args = ap.parse_args()

    try:
        if args.once:
            n = process_batch(args.batch, args.delay)
            print(f"Processed {n} tickers.")
            return 0
        total = 0
        while True:
            n = process_batch(args.batch, args.delay)
            if n == 0:
                break
            total += n
        print(f"Queue empty. Processed {total} tickers total.")
        return 0
    except RuntimeError as e:
        print(str(e), file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
