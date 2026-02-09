#!/usr/bin/env python3
"""List companies with the highest Fisher total_score (hidden gems)."""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

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

from fisher.db import get_connection, get_supabase_client


def _format_at(at) -> str:
    if at is None:
        return ""
    if hasattr(at, "strftime"):
        return at.strftime("%Y-%m-%d")
    if isinstance(at, str) and "T" in at:
        return at.split("T")[0]
    return str(at)[:10]


def main() -> None:
    ap = argparse.ArgumentParser(description="Top Fisher scorers (hidden gems)")
    ap.add_argument("-n", "--top", type=int, default=20, help="Show top N (default 20)")
    ap.add_argument("--min-score", type=float, default=None, help="Minimum total_score (e.g. 7.0)")
    ap.add_argument("--limit", type=int, default=100, help="Max rows to consider (default 100)")
    args = ap.parse_args()

    client = get_supabase_client()
    if client is not None:
        r = client.table("fisher_score_snapshot").select(
            "company_id, snapshot_at, total_score, fisher_company(ticker, name, sector)"
        ).order("snapshot_at", desc=True).limit(min(args.limit * 3, 500)).execute()
        raw = getattr(r, "data", None) or []
        # Latest per company_id (first occurrence after ordering by snapshot_at desc)
        seen = set()
        rows = []
        for row in raw:
            cid = row.get("company_id")
            if cid in seen or row.get("total_score") is None:
                continue
            seen.add(cid)
            fc = row.get("fisher_company") or {}
            rows.append({
                "ticker": fc.get("ticker"),
                "name": fc.get("name"),
                "sector": fc.get("sector"),
                "snapshot_at": row.get("snapshot_at"),
                "total_score": row.get("total_score"),
            })
        rows.sort(key=lambda x: (-(x.get("total_score") or 0), x.get("ticker") or ""))
        rows = rows[: args.limit]
    else:
        with get_connection() as conn:
            with conn.cursor() as cur:
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
        snapshot_str = _format_at(r.get("snapshot_at"))
        print(f"{ticker:<10} {name:<32} {sector:<20} {score_str:>6}  {snapshot_str}")
    print("-" * 85)
    print(f"Top {len(rows)} by Fisher total_score (0-10 scale; 7+ = strong, 5-7 = ok, <5 = weak)")


if __name__ == "__main__":
    main()
