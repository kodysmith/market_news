"""
Fisher scoring engine MVP-1: SEC-only points 1, 3, 5, 6, 9, 13, 14, partial 15.

Reads from fisher_financial_fact and fisher_segment_fact; writes fisher_score_snapshot.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime
from decimal import Decimal
from typing import Any

from fisher.db import get_connection

logger = logging.getLogger(__name__)

# All 15 Fisher points (Common Stocks and Uncommon Profits); display all, score implemented only
ALL_15_POINTS = (1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15)
# MVP-1 points (SEC/financials only) — used for total_score and category_scores
MVP1_POINTS = (1, 3, 5, 6, 9, 13, 14, 15)
VERSION = 1


def _stub_point() -> dict:
    """Return point payload for unimplemented points (score N/A)."""
    return {
        "score": None,
        "confidence": 0.0,
        "evidence": ["N/A - not yet automated"],
        "feature_values": {},
    }


def _decimal_to_float(v: Any) -> float | None:
    if v is None:
        return None
    if isinstance(v, (int, float)):
        return float(v)
    if isinstance(v, Decimal):
        return float(v)
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def get_latest_facts(conn: Any, company_id: Any) -> tuple[dict[str, float], dict[str, float] | None]:
    """
    Get financial facts for latest filing (current) and prior filing (for YoY).
    Returns (current_facts, prior_facts). prior_facts may be None.
    """
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT f.id AS filing_id, f.period_end, f.form_type
            FROM fisher_filing f
            WHERE f.company_id = %s AND f.form_type IN ('10-K', '10-Q')
            ORDER BY f.period_end DESC NULLS LAST
            LIMIT 2
            """,
            (company_id,),
        )
        rows = list(cur.fetchall())
    if not rows:
        return {}, None
    current_filing_id = rows[0]["filing_id"]
    prior_filing_id = rows[1]["filing_id"] if len(rows) > 1 else None

    def facts_for_filing(filing_id: Any) -> dict[str, float]:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT fact_name, value, unit FROM fisher_financial_fact WHERE filing_id = %s",
                (filing_id,),
            )
            out = {}
            for r in cur.fetchall():
                val = _decimal_to_float(r["value"])
                if val is not None:
                    # Take first value per fact (in case of duplicates)
                    if r["fact_name"] not in out:
                        out[r["fact_name"]] = val
            return out

    current = facts_for_filing(current_filing_id)
    prior = facts_for_filing(prior_filing_id) if prior_filing_id else None
    return current, prior


def extract_features(current: dict[str, float], prior: dict[str, float] | None) -> dict[str, float | None]:
    """Derive features for scoring from current and prior period facts."""
    rev = current.get("revenue") or 0
    gross = current.get("gross_profit") or 0
    op_inc = current.get("operating_income") or 0
    net_inc = current.get("net_income") or current.get("net_income_common") or 0
    shares = current.get("shares_outstanding")
    sbc = current.get("sbc") or 0
    buybacks = current.get("buybacks")  # often negative (outflow)
    debt = current.get("debt_long_term") or 0
    interest = current.get("interest_expense") or 0
    cash = current.get("cash") or 0
    ocf = current.get("operating_cash_flow") or 0
    capex = current.get("capex")  # often negative
    rd = current.get("rd_expense") or 0
    sga = current.get("sga_expense") or 0
    opex = current.get("opex") or 0

    rev_prior = (prior or {}).get("revenue") or 0
    shares_prior = (prior or {}).get("shares_outstanding") if prior else None

    gross_margin = (gross / rev * 100) if rev and rev > 0 else None
    op_margin = (op_inc / rev * 100) if rev and rev > 0 else None
    revenue_growth_yoy = None
    if rev and rev_prior and rev_prior > 0:
        revenue_growth_yoy = ((rev - rev_prior) / rev_prior) * 100
    rd_pct_revenue = (rd / rev * 100) if rev and rev > 0 else None
    sga_pct_revenue = (sga / rev * 100) if rev and rev > 0 else None
    sbc_pct_revenue = (sbc / rev * 100) if rev and rev > 0 else None
    shares_yoy_pct = None
    if shares and shares_prior and shares_prior > 0:
        shares_yoy_pct = ((shares - shares_prior) / shares_prior) * 100
    interest_coverage = (op_inc / interest) if interest and interest > 0 else None
    debt_to_cash = (debt / cash) if cash and cash > 0 else None
    capex_abs = abs(capex) if capex is not None else None
    fcf = (ocf - capex_abs) if ocf is not None and capex_abs is not None else (ocf if ocf else None)

    return {
        "revenue": rev or None,
        "revenue_growth_yoy_pct": revenue_growth_yoy,
        "gross_margin_pct": gross_margin,
        "operating_margin_pct": op_margin,
        "rd_pct_revenue": rd_pct_revenue,
        "sga_pct_revenue": sga_pct_revenue,
        "sbc_pct_revenue": sbc_pct_revenue,
        "shares_outstanding": shares,
        "shares_yoy_pct": shares_yoy_pct,
        "buybacks": buybacks,
        "debt_long_term": debt or None,
        "interest_expense": interest or None,
        "interest_coverage": interest_coverage,
        "cash": cash or None,
        "debt_to_cash": debt_to_cash,
        "operating_cash_flow": ocf or None,
        "capex": capex,
        "fcf": fcf,
        "net_income": net_inc or None,
    }


def score_point_1(features: dict[str, float | None]) -> dict:
    """Sales growth potential (revenue growth)."""
    growth = features.get("revenue_growth_yoy_pct")
    fv = {"revenue_growth_yoy_pct": growth}
    if growth is None:
        return {"score": 5.0, "confidence": 0.0, "evidence": [], "feature_values": fv}
    # Scale: >15% = 9-10, >5% = 7-8, >0 = 5-6, negative = 2-4
    if growth >= 15:
        score = 9.0 + min(1, (growth - 15) / 20)
    elif growth >= 5:
        score = 7.0 + (growth - 5) / 5
    elif growth >= 0:
        score = 5.0 + growth / 5
    else:
        score = max(0, 5 + growth / 5)
    return {"score": min(10, max(0, score)), "confidence": 0.9, "evidence": ["SEC revenue YoY"], "feature_values": fv}


def score_point_3(features: dict[str, float | None]) -> dict:
    """R&D effectiveness (R&D as % revenue — sector-dependent; higher for tech)."""
    rd_pct = features.get("rd_pct_revenue")
    fv = {"rd_pct_revenue": rd_pct}
    if rd_pct is None:
        return {"score": 5.0, "confidence": 0.0, "evidence": [], "feature_values": fv}
    # Simple: meaningful R&D (e.g. 2-15%) scores well; 0 might be non-R&D company
    if 2 <= rd_pct <= 20:
        score = 5.0 + rd_pct / 4  # 2% -> 5.5, 10% -> 7.5
    elif rd_pct > 20:
        score = 8.0
    else:
        score = 5.0  # unknown or no R&D
    return {"score": min(10, max(0, score)), "confidence": 0.85, "evidence": ["SEC R&D expense"], "feature_values": fv}


def score_point_5(features: dict[str, float | None]) -> dict:
    """Profit margin (gross and operating margin)."""
    gross = features.get("gross_margin_pct")
    op = features.get("operating_margin_pct")
    fv = {"gross_margin_pct": gross, "operating_margin_pct": op}
    if gross is None and op is None:
        return {"score": 5.0, "confidence": 0.0, "evidence": [], "feature_values": fv}
    score = 5.0
    if gross is not None:
        score += (gross - 30) / 20 if gross > 0 else -1  # 30% = 5, 50% = 6
    if op is not None:
        score += (op - 10) / 15 if op > 0 else -0.5  # 10% = 5, 25% = 6
    score = (score + 5) / 2 if (gross is not None and op is not None) else score
    return {"score": min(10, max(0, score)), "confidence": 0.9, "evidence": ["SEC margins"], "feature_values": fv}


def score_point_6(features: dict[str, float | None], prior_margins: tuple[float | None, float | None] | None) -> dict:
    """Maintain or improve margins (trend). prior_margins = (prior_gross, prior_op)."""
    gross = features.get("gross_margin_pct")
    op = features.get("operating_margin_pct")
    fv = {"gross_margin_pct": gross, "operating_margin_pct": op, "prior_margins": prior_margins}
    if prior_margins is None or (prior_margins[0] is None and prior_margins[1] is None):
        return {"score": 5.0, "confidence": 0.3, "evidence": [], "feature_values": fv}
    pg, po = prior_margins
    score = 5.0
    if gross is not None and pg is not None and pg != 0:
        score += (gross - pg) * 2  # improving = higher
    if op is not None and po is not None and po != 0:
        score += (op - po) * 2
    return {"score": min(10, max(0, score)), "confidence": 0.85, "evidence": ["SEC margin trend"], "feature_values": fv}


def score_point_9(features: dict[str, float | None]) -> dict:
    """Depth of management (partial: SBC / governance proxy)."""
    sbc_pct = features.get("sbc_pct_revenue")
    fv = {"sbc_pct_revenue": sbc_pct}
    if sbc_pct is None:
        return {"score": 5.0, "confidence": 0.2, "evidence": [], "feature_values": fv}
    # Moderate SBC can indicate alignment; very high may be excess
    if 0.5 <= sbc_pct <= 3:
        score = 6.5
    elif sbc_pct < 0.5:
        score = 5.0
    else:
        score = 5.0 - (sbc_pct - 3) / 5
    return {"score": min(10, max(0, score)), "confidence": 0.4, "evidence": ["SEC SBC"], "feature_values": fv}


def score_point_13(features: dict[str, float | None]) -> dict:
    """Dilution: shares YoY, SBC % revenue, buybacks."""
    shares_yoy = features.get("shares_yoy_pct")
    sbc_pct = features.get("sbc_pct_revenue")
    buybacks = features.get("buybacks")
    fv = {"shares_yoy_pct": shares_yoy, "sbc_pct_revenue": sbc_pct, "buybacks": buybacks}
    score = 5.0
    # Lower share growth = better (buybacks or stable)
    if shares_yoy is not None:
        if shares_yoy <= -2:
            score += 2  # net buyback
        elif shares_yoy <= 0:
            score += 1
        elif shares_yoy > 5:
            score -= 2
        elif shares_yoy > 2:
            score -= 1
    if sbc_pct is not None and sbc_pct > 5:
        score -= 1
    if buybacks is not None and buybacks < 0:  # negative = cash outflow = buyback
        score += 0.5
    return {"score": min(10, max(0, score)), "confidence": 0.9, "evidence": ["SEC shares, SBC, buybacks"], "feature_values": fv}


def score_point_14(features: dict[str, float | None]) -> dict:
    """Cash and debt management: interest coverage, debt/cash."""
    interest_cov = features.get("interest_coverage")
    debt_to_cash = features.get("debt_to_cash")
    fv = {"interest_coverage": interest_cov, "debt_to_cash": debt_to_cash, "cash": features.get("cash"), "debt_long_term": features.get("debt_long_term")}
    score = 5.0
    if interest_cov is not None:
        if interest_cov >= 5:
            score += 2
        elif interest_cov >= 2:
            score += 1
        elif interest_cov < 1 and interest_cov > 0:
            score -= 2
    if debt_to_cash is not None:
        if debt_to_cash < 1 and debt_to_cash >= 0:
            score += 1  # more cash than debt
        elif debt_to_cash > 3:
            score -= 1
    return {"score": min(10, max(0, score)), "confidence": 0.9, "evidence": ["SEC cash, debt, interest"], "feature_values": fv}


def score_point_15_partial(features: dict[str, float | None]) -> dict:
    """Moat (partial): margin level as proxy for pricing power."""
    gross = features.get("gross_margin_pct")
    op = features.get("operating_margin_pct")
    fv = {"gross_margin_pct": gross, "operating_margin_pct": op}
    if gross is None and op is None:
        return {"score": 5.0, "confidence": 0.2, "evidence": [], "feature_values": fv}
    score = 5.0
    if gross is not None and gross > 40:
        score += 1.5
    elif gross is not None and gross > 25:
        score += 0.5
    if op is not None and op > 15:
        score += 1.5
    elif op is not None and op > 8:
        score += 0.5
    return {"score": min(10, max(0, score)), "confidence": 0.5, "evidence": ["SEC margins as moat proxy"], "feature_values": fv}


def compute_prior_margins(prior_facts: dict[str, float] | None) -> tuple[float | None, float | None]:
    if not prior_facts:
        return (None, None)
    rev = prior_facts.get("revenue") or 0
    gross = (prior_facts.get("gross_profit") or 0) / rev * 100 if rev else None
    op = (prior_facts.get("operating_income") or 0) / rev * 100 if rev else None
    return (gross, op)


def score_company(conn: Any, company_id: Any) -> dict | None:
    """
    Compute Fisher point scores for one company; return snapshot payload (points, category_scores, total_score)
    or None if no filings.
    """
    current, prior = get_latest_facts(conn, company_id)
    if not current:
        return None
    features = extract_features(current, prior)
    prior_margins = compute_prior_margins(prior)

    point_handlers = {
        1: lambda: score_point_1(features),
        3: lambda: score_point_3(features),
        5: lambda: score_point_5(features),
        6: lambda: score_point_6(features, prior_margins),
        9: lambda: score_point_9(features),
        13: lambda: score_point_13(features),
        14: lambda: score_point_14(features),
        15: lambda: score_point_15_partial(features),
    }
    points = {}
    for pid in ALL_15_POINTS:
        points[str(pid)] = point_handlers[pid]() if pid in point_handlers else _stub_point()

    scores_only = [points[str(p)]["score"] for p in MVP1_POINTS]
    total_score = round(sum(scores_only) / len(scores_only), 2) if scores_only else None
    category_scores = {
        "growth": (points["1"]["score"] + points["3"]["score"]) / 2,
        "financials": (points["5"]["score"] + points["6"]["score"] + points["14"]["score"]) / 3,
        "capital_allocation": (points["9"]["score"] + points["13"]["score"]) / 2,
        "moat": points["15"]["score"],
    }
    return {
        "points": points,
        "category_scores": category_scores,
        "total_score": total_score,
    }


def write_snapshot(conn: Any, company_id: Any, payload: dict) -> None:
    """Insert one row into fisher_score_snapshot."""
    # JSON-serialize for JSONB; ensure dicts are JSON-serializable
    def _serialize(obj: Any) -> Any:
        if isinstance(obj, (Decimal,)):
            return float(obj)
        if isinstance(obj, dict):
            return {k: _serialize(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [_serialize(x) for x in obj]
        return obj

    points = _serialize(payload["points"])
    category_scores = _serialize(payload["category_scores"])
    total = payload["total_score"]
    with conn.cursor() as cur:
        cur.execute(
            """
            INSERT INTO fisher_score_snapshot (company_id, snapshot_at, points, category_scores, total_score, version)
            VALUES (%s, NOW(), %s::jsonb, %s::jsonb, %s, %s)
            """,
            (company_id, json.dumps(points), json.dumps(category_scores), total, VERSION),
        )


def run_scoring(conn: Any, company_ids: list[Any] | None = None) -> int:
    """
    Score all companies (or given company_ids) with filings; write fisher_score_snapshot.
    Returns number of snapshots written.
    """
    if company_ids is None:
        with conn.cursor() as cur:
            cur.execute("SELECT id FROM fisher_company")
            company_ids = [r["id"] for r in cur.fetchall()]
    written = 0
    for cid in company_ids:
        try:
            payload = score_company(conn, cid)
            if payload:
                write_snapshot(conn, cid, payload)
                written += 1
        except Exception as e:
            logger.warning("Score company %s: %s", cid, e)
    return written


def run_scoring_job(company_ids: list[Any] | None = None) -> int:
    """Entry point: get connection, run scoring, return count."""
    with get_connection() as conn:
        return run_scoring(conn, company_ids=company_ids)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    n = run_scoring_job()
    print("Snapshots written:", n)
