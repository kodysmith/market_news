#!/usr/bin/env python3
"""
Fetch top N high-growth + profitable companies using Yahoo (yfinance) or Alpha Vantage.

Scans a universe (S&P 500 or SEC list), computes revenue YoY and profit YoY from income
statements, ranks by growth (and requires profitable = positive net income), writes
data/high_growth_universe.json. Then run Fisher + growth tooling on that list.
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

# Yahoo (yfinance) - no API key, reasonable rate limits
def _growth_from_yahoo(ticker: str) -> dict | None:
    import yfinance as yf
    try:
        t = yf.Ticker(ticker)
        inc = t.income_stmt
        if inc is None or inc.empty or len(inc.columns) < 2:
            return None
        cols = list(inc.columns[:2])
        rev_growth = None
        profit_growth = None
        net_curr = None
        for rev_label in ["Total Revenue", "Operating Revenue", "Revenues"]:
            if rev_label in inc.index:
                row = inc.loc[rev_label][cols].dropna()
                if len(row) >= 2:
                    curr, prev = float(row.iloc[0]), float(row.iloc[1])
                    if prev and prev != 0:
                        rev_growth = round((curr - prev) / prev * 100, 2)
                break
        if "Net Income" in inc.index:
            row = inc.loc["Net Income"][cols].dropna()
            if len(row) >= 1:
                net_curr = float(row.iloc[0])
            if len(row) >= 2:
                curr, prev = float(row.iloc[0]), float(row.iloc[1])
                if prev is not None and prev != 0:
                    profit_growth = round((curr - prev) / abs(prev) * 100, 2)
        info = t.info or {}
        name = (info.get("longName") or info.get("shortName") or ticker)
        sector = info.get("sector") or ""
        industry = info.get("industry") or ""
        return {
            "ticker": ticker,
            "name": name,
            "sector": sector,
            "industry": industry,
            "revenue_growth_yoy_pct": rev_growth,
            "profit_growth_yoy_pct": profit_growth,
            "net_income_latest": net_curr,
            "profitable": net_curr is not None and net_curr > 0,
        }
    except Exception as e:
        return {"ticker": ticker, "error": str(e), "profitable": False}


# Alpha Vantage - needs API key, 25 req/day free; use for single-ticker or small batch
def _growth_from_alphavantage(ticker: str, api_key: str) -> dict | None:
    import requests
    url = "https://www.alphavantage.co/query"
    params = {
        "function": "INCOME_STATEMENT",
        "symbol": ticker,
        "apikey": api_key,
    }
    try:
        r = requests.get(url, params=params, timeout=15)
        r.raise_for_status()
        data = r.json()
        reports = data.get("annualReports") or []
        if len(reports) < 2:
            return None
        reports = sorted(reports, key=lambda x: x.get("fiscalDateEnding", ""), reverse=True)[:2]
        r0, r1 = reports[0], reports[1]
        rev0 = _float(r0.get("totalRevenue"))
        rev1 = _float(r1.get("totalRevenue"))
        net0 = _float(r0.get("netIncome"))
        net1 = _float(r1.get("netIncome"))
        rev_growth = ((rev0 - rev1) / rev1 * 100) if rev1 and rev1 != 0 else None
        profit_growth = ((net0 - net1) / abs(net1) * 100) if net1 is not None and net1 != 0 else None
        return {
            "ticker": ticker,
            "name": r0.get("reportedCurrency") or ticker,
            "sector": "",
            "industry": "",
            "revenue_growth_yoy_pct": round(rev_growth, 2) if rev_growth is not None else None,
            "profit_growth_yoy_pct": round(profit_growth, 2) if profit_growth is not None else None,
            "net_income_latest": net0,
            "profitable": net0 is not None and net0 > 0,
        }
    except Exception as e:
        return {"ticker": ticker, "error": str(e), "profitable": False}


def _float(v):
    if v is None:
        return None
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def main() -> int:
    import argparse
    ap = argparse.ArgumentParser(description="Build high-growth + profitable universe (Yahoo or Alpha Vantage)")
    ap.add_argument("--top", type=int, default=100, help="Top N companies to keep (default 100)")
    ap.add_argument("--source", choices=("yahoo", "alphavantage"), default="yahoo", help="Data source (default yahoo)")
    ap.add_argument("--universe", choices=("sp500", "sec"), default="sp500", help="Ticker universe to scan (default sp500)")
    ap.add_argument("--limit-scan", type=int, default=None, help="Max tickers to fetch (default: all in universe)")
    ap.add_argument("--min-revenue-growth", type=float, default=0, help="Min revenue YoY %% (default 0)")
    ap.add_argument("--profitable-only", action="store_true", default=True, help="Require net income > 0 (default True)")
    ap.add_argument("--out", default=None, help="Output path (default data/high_growth_universe.json)")
    args = ap.parse_args()

    out_path = Path(args.out) if args.out else ROOT / "data" / "high_growth_universe.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Load universe
    if args.universe == "sec":
        path = ROOT / "data" / "sec_universe.json"
        if not path.exists():
            print("Run scripts/fetch_sec_universe.py first.", file=sys.stderr)
            return 1
        with open(path) as f:
            constituents = json.load(f).get("constituents", [])
    else:
        path = ROOT / "data" / "sp500_constituents.json"
        if not path.exists():
            print("Missing data/sp500_constituents.json", file=sys.stderr)
            return 1
        with open(path) as f:
            constituents = json.load(f).get("constituents", [])

    tickers = [c.get("ticker") for c in constituents if c.get("ticker")]
    if args.limit_scan:
        tickers = tickers[: args.limit_scan]
    print(f"Scanning {len(tickers)} tickers via {args.source} (profitable_only={args.profitable_only}, min_revenue_growth={args.min_revenue_growth}%)")

    # Build ticker -> cik from existing lists for Fisher
    cik_by_ticker = {c.get("ticker"): c.get("cik") for c in constituents if c.get("ticker") and c.get("cik")}

    results = []
    api_key = os.environ.get("ALPHAVANTAGE_API_KEY", "") if args.source == "alphavantage" else None
    if args.source == "alphavantage" and not api_key:
        print("Set ALPHAVANTAGE_API_KEY for Alpha Vantage (free tier ~25 req/day).", file=sys.stderr)
    for i, ticker in enumerate(tickers):
        if args.source == "yahoo":
            row = _growth_from_yahoo(ticker)
        else:
            row = _growth_from_alphavantage(ticker, api_key or "")
            time.sleep(0.25)  # rate limit
        if row and "error" not in row:
            row["cik"] = cik_by_ticker.get(ticker)
            results.append(row)
        if (i + 1) % 50 == 0:
            print(f"  {i+1}/{len(tickers)}")

    # Filter: profitable and min revenue growth
    if args.profitable_only:
        results = [r for r in results if r.get("profitable")]
    results = [r for r in results if (r.get("revenue_growth_yoy_pct") or 0) >= args.min_revenue_growth]

    # Rank by revenue growth (then profit growth)
    results.sort(
        key=lambda r: (
            r.get("revenue_growth_yoy_pct") or -999,
            r.get("profit_growth_yoy_pct") or -999,
        ),
        reverse=True,
    )
    top = results[: args.top]

    # Output same shape as sp500_constituents for Fisher
    constituents_out = []
    for r in top:
        constituents_out.append({
            "ticker": r["ticker"],
            "name": r.get("name") or r["ticker"],
            "sector": r.get("sector") or "",
            "industry": r.get("industry") or "",
            "cik": r.get("cik"),
            "revenue_growth_yoy_pct": r.get("revenue_growth_yoy_pct"),
            "profit_growth_yoy_pct": r.get("profit_growth_yoy_pct"),
        })

    out_data = {
        "source": f"high_growth scan ({args.source}, universe={args.universe})",
        "updated": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "constituents": constituents_out,
    }
    with open(out_path, "w") as f:
        json.dump(out_data, f, indent=2)
    print(f"Wrote top {len(constituents_out)} to {out_path}")
    print("Next: run Fisher on this list with universe='high_growth', then scripts/scan_growth_profitable.py")
    return 0


if __name__ == "__main__":
    sys.exit(main())
