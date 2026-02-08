#!/usr/bin/env python3
"""Scan for companies with high growth and profitable (Fisher category_scores: growth + financials)."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from fisher.db import get_connection


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Companies with high growth and profitability (Fisher category scores)"
    )
    ap.add_argument("-n", "--top", type=int, default=30, help="Show top N (default 30)")
    ap.add_argument(
        "--min-growth",
        type=float,
        default=6.0,
        help="Minimum growth category score 0-10 (default 6; 7+ = strong revenue growth)",
    )
    ap.add_argument(
        "--min-financials",
        type=float,
        default=6.0,
        help="Minimum financials category score 0-10 (default 6; margins/profitability)",
    )
    ap.add_argument(
        "--order",
        choices=("composite", "total", "growth", "financials"),
        default="composite",
        help="Sort by: composite=growth+financials, total=Fisher total, or single category (default composite)",
    )
    args = ap.parse_args()

    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute("""
                WITH latest AS (
                    SELECT DISTINCT ON (company_id)
                        company_id, snapshot_at, total_score, category_scores
                    FROM fisher_score_snapshot
                    ORDER BY company_id, snapshot_at DESC
                )
                SELECT
                    c.ticker,
                    c.name,
                    c.sector,
                    l.snapshot_at,
                    l.total_score,
                    (l.category_scores->>'growth')::float AS growth,
                    (l.category_scores->>'financials')::float AS financials
                FROM latest l
                JOIN fisher_company c ON c.id = l.company_id
                WHERE (l.category_scores->>'growth')::float IS NOT NULL
                  AND (l.category_scores->>'financials')::float IS NOT NULL
                  AND (l.category_scores->>'growth')::float >= %s
                  AND (l.category_scores->>'financials')::float >= %s
                ORDER BY c.ticker
            """, (args.min_growth, args.min_financials))
            rows = cur.fetchall()

    # Sort by chosen metric
    def _key(r):
        g = r.get("growth") or 0
        f = r.get("financials") or 0
        t = r.get("total_score") or 0
        if args.order == "composite":
            return (g + f, t)
        if args.order == "total":
            return (t, g + f)
        if args.order == "growth":
            return (g, f)
        return (f, g)

    rows = sorted(rows, key=_key, reverse=True)

    if not rows:
        print(
            f"No companies with growth>={args.min_growth} and financials>={args.min_financials}. "
            "Try lower --min-growth / --min-financials or run the Fisher pipeline first."
        )
        return

    rows = rows[: args.top]
    print(
        f"High growth + profitable (growth>={args.min_growth}, financials>={args.min_financials}, "
        f"ordered by {args.order})"
    )
    print(f"{'Ticker':<10} {'Name':<28} {'Sector':<18} {'Growth':>6} {'Fin':>6} {'Total':>6}  Snapshot")
    print("-" * 95)
    for r in rows:
        ticker = (r.get("ticker") or "")[:10]
        name = (r.get("name") or "")[:27]
        sector = (r.get("sector") or "")[:17]
        g = r.get("growth")
        f = r.get("financials")
        t = r.get("total_score")
        gs = f"{g:.2f}" if g is not None else ""
        fs = f"{f:.2f}" if f is not None else ""
        ts = f"{t:.2f}" if t is not None else ""
        at = r.get("snapshot_at")
        snapshot_str = at.strftime("%Y-%m-%d") if at else ""
        print(f"{ticker:<10} {name:<28} {sector:<18} {gs:>6} {fs:>6} {ts:>6}  {snapshot_str}")
    print("-" * 95)
    print(f"Growth = revenue growth + R&D (0-10). Financials = margins + profitability (0-10). Total = Fisher total.")


if __name__ == "__main__":
    main()
    sys.exit(0)
