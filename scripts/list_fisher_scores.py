#!/usr/bin/env python3
"""Print companies that have Fisher score snapshots and their latest total_score."""
from __future__ import annotations

import os
import sys
from pathlib import Path

# Ensure repo root on path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

# Load repo root .env (same as worker) so we query the same DB
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
        return at.strftime("%Y-%m-%d %H:%M")
    if isinstance(at, str) and "T" in at:
        return at.replace("T", " ")[:16]
    return str(at)[:16]


def main() -> None:
    client = get_supabase_client()
    if client is not None:
        r = client.table("fisher_score_snapshot").select(
            "snapshot_at, total_score, fisher_company(ticker, name)"
        ).order("snapshot_at", desc=True).execute()
        rows = getattr(r, "data", None) or []
        # Flatten fisher_company embed
        out = []
        for row in rows:
            fc = row.get("fisher_company") or {}
            out.append({
                "ticker": (fc.get("ticker") or "")[:10],
                "name": (fc.get("name") or "")[:34],
                "snapshot_at": row.get("snapshot_at"),
                "total_score": row.get("total_score"),
            })
        rows = out
    else:
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
        snapshot_str = _format_at(r.get("snapshot_at"))
        score = r.get("total_score")
        score_str = f"{score:.2f}" if score is not None else ""
        print(f"{ticker:<10} {name:<35} {snapshot_str:<22} {score_str:>6}")
    print("-" * 75)
    print(f"Total: {len(rows)} snapshot(s)")


if __name__ == "__main__":
    main()
