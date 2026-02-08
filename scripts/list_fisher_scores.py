#!/usr/bin/env python3
"""Print companies that have Fisher score snapshots and their latest total_score."""
from __future__ import annotations

import sys
from pathlib import Path

# Ensure repo root on path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from fisher.db import get_connection


def main() -> None:
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT c.ticker, c.name, s.snapshot_at, s.total_score
                FROM fisher_score_snapshot s
                JOIN fisher_company c ON c.id = s.company_id
                ORDER BY s.snapshot_at DESC, c.ticker
            """)
            rows = cur.fetchall()
    if not rows:
        print("No Fisher score snapshots found. Run the pipeline first (edgar_watcher, market_updater, scoring_job).")
        return
    print(f"{'Ticker':<10} {'Name':<35} {'Snapshot':<22} {'Score':>6}")
    print("-" * 75)
    for r in rows:
        ticker = (r.get("ticker") or "")[:10]
        name = (r.get("name") or "")[:34]
        at = r.get("snapshot_at")
        snapshot_str = at.strftime("%Y-%m-%d %H:%M") if at else ""
        score = r.get("total_score")
        score_str = f"{score:.2f}" if score is not None else ""
        print(f"{ticker:<10} {name:<35} {snapshot_str:<22} {score_str:>6}")
    print("-" * 75)
    print(f"Total: {len(rows)} snapshot(s)")


if __name__ == "__main__":
    main()
