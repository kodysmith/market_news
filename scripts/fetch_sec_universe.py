#!/usr/bin/env python3
"""Fetch SEC company_tickers.json and save as data/sec_universe.json for Fisher pipeline."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parent.parent
OUT_PATH = ROOT / "data" / "sec_universe.json"
SEC_URL = "https://www.sec.gov/files/company_tickers.json"
USER_AGENT = "MarketNews Fisher Pipeline (contact@example.com)"


def main() -> None:
    print("Fetching SEC company tickers...")
    r = requests.get(SEC_URL, headers={"User-Agent": USER_AGENT}, timeout=60)
    r.raise_for_status()
    raw = r.json()
    # raw is dict keyed by index: { "0": { "cik_str": 320193, "ticker": "AAPL", "title": "Apple Inc." }, ... }
    constituents = []
    seen = set()
    for key, item in raw.items():
        ticker = (item.get("ticker") or "").strip().upper()
        if not ticker or ticker in seen:
            continue
        seen.add(ticker)
        cik = item.get("cik_str")
        if cik is None:
            continue
        name = (item.get("title") or ticker).strip()
        constituents.append({
            "ticker": ticker,
            "name": name,
            "cik": str(cik).strip(),
            "sector": None,
            "industry": None,
        })
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    out = {"source": "SEC company_tickers.json", "constituents": constituents}
    with open(OUT_PATH, "w") as f:
        json.dump(out, f, indent=2)
    print(f"Saved {len(constituents)} companies to {OUT_PATH}")
    print("Run Fisher pipeline with universe='sec' to score these (e.g. limit=500 for a batch).")


if __name__ == "__main__":
    main()
    sys.exit(0)
