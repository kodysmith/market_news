#!/usr/bin/env python3
"""Run Fisher pipeline (EDGAR + scoring) for one or more tickers."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from fisher.db import get_connection, company_id_by_ticker
from fisher.edgar_watcher import run_watcher
from fisher.scoring_engine import run_scoring_job


def main() -> int:
    ap = argparse.ArgumentParser(description="Run Fisher EDGAR + scoring for given tickers")
    ap.add_argument("tickers", nargs="+", help="Tickers to process (e.g. AAPL MSFT GOOGL)")
    ap.add_argument(
        "--universe",
        choices=("sp500", "sec"),
        default="sec",
        help="Universe to resolve tickers from (default: sec; use sec after fetch_sec_universe.py)",
    )
    args = ap.parse_args()
    tickers = [t.strip().upper() for t in args.tickers if t.strip()]
    if not tickers:
        print("No tickers provided", file=sys.stderr)
        return 1

    print("EDGAR: fetching filings + company facts for", tickers)
    watcher_result = run_watcher(tickers=tickers, universe=args.universe)
    print("EDGAR result:", watcher_result)

    with get_connection() as conn:
        company_ids = []
        for t in tickers:
            cid = company_id_by_ticker(conn, t)
            if cid:
                company_ids.append(cid)
            else:
                print(f"  Skip scoring: {t} not in fisher_company (not in {args.universe} list?)", file=sys.stderr)
    if not company_ids:
        print("No companies to score.", file=sys.stderr)
        return 1

    print("Scoring", len(company_ids), "companies")
    n = run_scoring_job(company_ids=company_ids)
    print("Snapshots written:", n)
    return 0


if __name__ == "__main__":
    sys.exit(main())
