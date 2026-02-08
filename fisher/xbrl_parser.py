"""
XBRL / SEC Company Facts → fisher_financial_fact.

Fetches data.sec.gov/api/xbrl/companyfacts/CIK{cik}.json and maps
US-GAAP (and dei) concepts to canonical fact_name; inserts into fisher_financial_fact.
"""

from __future__ import annotations

import logging
import time
from typing import Any

import requests

from fisher.config import SEC_USER_AGENT

logger = logging.getLogger(__name__)

SEC_COMPANYFACTS = "https://data.sec.gov/api/xbrl/companyfacts/CIK{cik}.json"
RATE_LIMIT_SEC = 0.2

# Map SEC concept names to our canonical fact_name (for scoring)
CANONICAL_MAP = {
    "Revenue": "revenue",
    "Revenues": "revenue",
    "SalesRevenueNet": "revenue",
    "GrossProfit": "gross_profit",
    "OperatingIncomeLoss": "operating_income",
    "NetIncomeLoss": "net_income",
    "NetIncomeLossAvailableToCommonStockholdersBasic": "net_income_common",
    "EntityCommonStockSharesOutstanding": "shares_outstanding",
    "StockBasedCompensation": "sbc",
    "PaymentsForRepurchaseOfCommonStock": "buybacks",
    "LongTermDebt": "debt_long_term",
    "LongTermDebtAndCapitalLeaseObligations": "debt_long_term",
    "InterestExpense": "interest_expense",
    "InterestPaid": "interest_expense",
    "ResearchAndDevelopmentExpense": "rd_expense",
    "SellingGeneralAndAdministrativeExpense": "sga_expense",
    "OperatingExpenses": "opex",
    "CashAndCashEquivalentsAtCarryingValue": "cash",
    "Cash": "cash",
    "PropertyPlantAndEquipmentNet": "ppe_net",
    "CapitalExpenditures": "capex",
    "NetCashProvidedByUsedInOperatingActivities": "operating_cash_flow",
    "NetCashProvidedByUsedInInvestingActivities": "investing_cash_flow",
    "Assets": "assets",
    "Liabilities": "liabilities",
    "LiabilitiesAndStockholdersEquity": "liabilities",
}


def _cik_pad(cik: str) -> str:
    s = str(cik).strip().lstrip("0") or "0"
    return s.zfill(10) if s.isdigit() else str(cik)


def _fetch_company_facts(cik: str) -> dict | None:
    cik_padded = _cik_pad(cik)
    url = SEC_COMPANYFACTS.format(cik=cik_padded)
    headers = {"User-Agent": SEC_USER_AGENT, "Accept": "application/json"}
    try:
        r = requests.get(url, headers=headers, timeout=60)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        logger.warning("Company facts %s: %s", cik, e)
        return None


def sync_company_facts(conn: Any, company_id: Any, cik: str) -> int:
    """
    Fetch companyfacts for CIK, resolve filing_id by (company_id, form_type, period_end),
    insert/upsert fisher_financial_fact. Returns number of fact rows written.
    """
    data = _fetch_company_facts(cik)
    if not data:
        return 0
    time.sleep(RATE_LIMIT_SEC)
    facts = data.get("facts") or {}
    # Flatten: we need (form, end) -> list of (fact_name, unit, value)
    by_period: dict[tuple[str, str], list[tuple[str, str, float]]] = {}
    for taxonomy, concepts in facts.items():
        for concept_name, concept_data in (concepts or {}).items():
            canonical = CANONICAL_MAP.get(concept_name)
            if not canonical:
                continue
            units = concept_data.get("units") or {}
            for unit_type, values in units.items():
                unit = "USD" if unit_type == "USD" else "shares" if unit_type == "shares" else str(unit_type)
                for v in values:
                    form = v.get("form") or ""
                    if form not in ("10-K", "10-Q"):
                        continue
                    end = v.get("end")
                    val = v.get("val")
                    if end and val is not None:
                        try:
                            key = (form, end)
                            by_period.setdefault(key, []).append((canonical, unit, float(val)))
                        except (TypeError, ValueError):
                            pass
    written = 0
    with conn.cursor() as cur:
        for (form_type, period_end), fact_list in by_period.items():
            cur.execute(
                "SELECT id FROM fisher_filing WHERE company_id = %s AND form_type = %s AND period_end = %s::date",
                (company_id, form_type, period_end),
            )
            row = cur.fetchone()
            if not row:
                continue
            filing_id = row["id"]
            for fact_name, unit, value in fact_list:
                cur.execute(
                    "DELETE FROM fisher_financial_fact WHERE filing_id = %s AND fact_name = %s AND period = %s::date",
                    (filing_id, fact_name, period_end),
                )
                cur.execute(
                    """
                    INSERT INTO fisher_financial_fact (filing_id, fact_name, value, unit, period)
                    VALUES (%s, %s, %s, %s, %s::date)
                    """,
                    (filing_id, fact_name, value, unit, period_end),
                )
                written += 1
    return written
