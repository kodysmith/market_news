#!/usr/bin/env python3
"""
Run Fisher scoring only: write fisher_score_snapshot for companies that already have EDGAR data.

Use when fisher_company + fisher_filing + fisher_financial_fact are already in the DB
(e.g. after the scan worker has run the EDGAR watcher but you want to backfill snapshots,
or after migrating data without snapshots).

Requires: SUPABASE_URL + SUPABASE_SECRET_KEY (or DATABASE_URL) in repo root .env.
"""
from __future__ import annotations

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

from fisher.scoring_engine import run_scoring_job


def main() -> int:
    print("Running Fisher scoring for all companies in fisher_company (with filings/facts)...")
    n = run_scoring_job()
    print(f"Wrote {n} Fisher score snapshot(s) to fisher_score_snapshot.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
