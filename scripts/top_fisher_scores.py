#!/usr/bin/env python3
"""List companies with the highest Fisher total_score (hidden gems)."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from fisher.db import get_connection


def main() -> None:
    ap = argparse.ArgumentParser(description="Top Fisher scorers (hidden gems)")
    ap.add_argument("-n", "--top", type=int, default=20, help="Show top N (default 20)")
    ap.add_argument("--min-score", type=float, default=None, help="Minimum total_score (e.g. 7.0)")
    ap.add_argument("--limit", type=int, default=100, help="Max rows to consider (default 100)")
    args = ap.parse_args()

    with get_connection() as conn:
        with conn.cursor() as cur:
            # Latest snapshot per company, then order by total_score DESC (hidden gems first)
            cur.execute("""
                WITH latest AS (
                    SELECT DISTINCT ON (company_id)
                        company_id, snapshot_at, total_score
                    FROM fisher_score_snapshot
                    ORDER BY company_id, snapshot_at DESC
                )
                SELECT c.ticker, c.name, c.sector, l.snapshot_at, l.total_score
                FROM latest l
                JOIN fisher_company c ON c.id = l.company_id
                WHERE l.total_score IS NOT NULL
                ORDER BY l.total_score DESC NULLS LAST, c.ticker
                LIMIT %s
            """, (args.limit,))
            rows = cur.fetchall()

    if not rows:
        print("No Fisher score snapshots found. Run the pipeline first.")
        return

    # Apply min_score filter if set
    if args.min_score is not None:
        rows = [r for r in rows if (r.get("total_score") or 0) >= args.min_score]
    rows = rows[: args.top]

    print(f"{'Ticker':<10} {'Name':<32} {'Sector':<20} {'Score':>6}  Snapshot")
    print("-" * 85)
    for r in rows:
        ticker = (r.get("ticker") or "")[:10]
        name = (r.get("name") or "")[:31]
        sector = (r.get("sector") or "")[:19]
        score = r.get("total_score")
        score_str = f"{score:.2f}" if score is not None else ""
        at = r.get("snapshot_at")
        snapshot_str = at.strftime("%Y-%m-%d") if at else ""
        print(f"{ticker:<10} {name:<32} {sector:<20} {score_str:>6}  {snapshot_str}")
    print("-" * 85)
    print(f"Top {len(rows)} by Fisher total_score (0-10 scale; 7+ = strong, 5-7 = ok, <5 = weak)")


if __name__ == "__main__":
    main()
