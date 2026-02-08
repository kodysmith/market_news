"""
SEC EDGAR filing watcher.

For each S&P 500 company (with CIK), fetches submissions and companyfacts,
ensures fisher_filing rows for 10-K/10-Q, and populates financial_fact via xbrl_parser.
Run every 1-6 hours (cron or scheduler).
"""

from __future__ import annotations

import logging
import time
from datetime import datetime
from typing import Any

import requests

from fisher.config import SEC_USER_AGENT, get_sp500_constituents, get_sec_universe, get_high_growth_universe
from fisher.db import get_connection, ensure_company, company_id_by_ticker
from fisher import xbrl_parser

logger = logging.getLogger(__name__)

SEC_SUBMISSIONS = "https://data.sec.gov/submissions/CIK{cik}.json"
SEC_COMPANYFACTS = "https://data.sec.gov/api/xbrl/companyfacts/CIK{cik}.json"
RATE_LIMIT_SEC = 0.2  # SEC asks for max 10 requests/second; be conservative


def _cik_pad(cik: str | int) -> str:
    """Zero-pad CIK to 10 digits."""
    if cik is None:
        return ""
    s = str(cik).strip()
    return s.zfill(10) if s.isdigit() else s


def _fetch_json(url: str) -> dict | None:
    """GET JSON with SEC User-Agent."""
    headers = {"User-Agent": SEC_USER_AGENT, "Accept": "application/json"}
    try:
        r = requests.get(url, headers=headers, timeout=30)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        logger.warning("Fetch %s: %s", url[:80], e)
        return None


def _recent_filings(submissions: dict, form_types: tuple = ("10-K", "10-Q")) -> list[dict]:
    """Extract recent 10-K/10-Q from submissions JSON."""
    filings = submissions.get("filings") or {}
    recent = filings.get("recent") or {}
    forms = recent.get("form") or []
    filing_dates = recent.get("filingDate") or []
    report_dates = recent.get("reportDate") or []
    accession_numbers = recent.get("accessionNumber") or []
    out = []
    for i, form in enumerate(forms):
        if form not in form_types:
            continue
        fd = filing_dates[i] if i < len(filing_dates) else None
        rd = report_dates[i] if i < len(report_dates) else None
        accn = accession_numbers[i] if i < len(accession_numbers) else None
        if fd and rd:
            out.append({"form": form, "filing_date": fd, "report_date": rd, "accession_number": accn})
    return out[:200]  # limit to last 200 relevant filings


def ensure_filings_for_company(conn: Any, company_id: Any, cik: str, ticker: str) -> int:
    """Fetch submissions, ensure fisher_filing rows for 10-K/10-Q; return count of new filings."""
    cik_padded = _cik_pad(cik)
    if not cik_padded:
        return 0
    url = SEC_SUBMISSIONS.format(cik=cik_padded)
    data = _fetch_json(url)
    if not data:
        return 0
    time.sleep(RATE_LIMIT_SEC)
    recent = _recent_filings(data)
    added = 0
    with conn.cursor() as cur:
        for f in recent:
            form_type = f["form"]
            filing_date = f["filing_date"]
            period_end = f["report_date"]
            url_edgar = (
                f"https://www.sec.gov/Archives/edgar/data/{cik.lstrip('0')}/{f['accession_number'].replace('-', '')}/{f['accession_number']}.txt"
                if f.get("accession_number") else None
            )
            try:
                cur.execute(
                    """
                    INSERT INTO fisher_filing (company_id, form_type, filing_date, period_end, url)
                    VALUES (%s, %s, %s::date, %s::date, %s)
                    ON CONFLICT (company_id, form_type, period_end) DO NOTHING
                    """,
                    (company_id, form_type, filing_date, period_end, url_edgar),
                )
                if cur.rowcount:
                    added += 1
            except Exception as e:
                logger.debug("Insert filing %s %s: %s", form_type, period_end, e)
    return added


def run_watcher(
    tickers: list[str] | None = None,
    limit: int | None = None,
    universe: str = "sp500",
) -> dict[str, Any]:
    """
    Load constituents, ensure companies and filings, then fetch companyfacts and populate financial_fact.
    If tickers is set, only process those; else all constituents. limit caps number of companies.
    universe: "sp500" (default), "sec" (SEC list), "high_growth" (data/high_growth_universe.json).
    """
    if universe == "sec":
        constituents = get_sec_universe()
    elif universe == "high_growth":
        constituents = get_high_growth_universe()
    else:
        constituents = get_sp500_constituents()
    if tickers:
        constituents = [c for c in constituents if c.get("ticker") in tickers]
    if limit:
        constituents = constituents[:limit]
    stats = {"companies": 0, "filings_added": 0, "facts_added": 0, "errors": []}
    with get_connection() as conn:
        for c in constituents:
            ticker = (c.get("ticker") or "").upper()
            if not ticker:
                continue
            cik = c.get("cik")
            if not cik:
                stats["errors"].append(f"{ticker}: no CIK")
                continue
            cik_str = str(cik).strip()
            try:
                company_id = ensure_company(
                    conn,
                    ticker,
                    name=c.get("name"),
                    cik=cik_str,
                    sector=c.get("sector"),
                    industry=c.get("industry"),
                )
                stats["companies"] += 1
                added = ensure_filings_for_company(conn, company_id, cik_str, ticker)
                stats["filings_added"] += added
                time.sleep(RATE_LIMIT_SEC)
                # Fetch companyfacts and populate financial_fact
                facts_added = xbrl_parser.sync_company_facts(conn, company_id, cik_str)
                stats["facts_added"] += facts_added
            except Exception as e:
                stats["errors"].append(f"{ticker}: {e}")
                logger.warning("Watcher %s: %s", ticker, e)
    return stats


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    import sys
    limit = int(sys.argv[1]) if len(sys.argv) > 1 else 3
    result = run_watcher(limit=limit)
    print(result)
