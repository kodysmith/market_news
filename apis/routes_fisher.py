"""
Fisher score API: snapshot, delta, evidence, universe.

GET /fisher/snapshot?ticker=...
GET /fisher/delta?ticker=...
GET /fisher/evidence?ticker=...&point_id=...
GET /fisher/universe
"""

from __future__ import annotations

from datetime import date, datetime
from decimal import Decimal
from flask import Blueprint, jsonify, request

from fisher.config import get_sp500_constituents
from fisher.db import get_connection, company_id_by_ticker

bp = Blueprint("fisher", __name__, url_prefix="/fisher")


def _serialize_value(v):
    if isinstance(v, (Decimal,)):
        return float(v)
    if isinstance(v, (date, datetime)):
        return v.isoformat()
    return v


def _serialize_snapshot(row: dict) -> dict:
    """Convert DB row to JSON-safe snapshot payload."""
    points = row.get("points") or {}
    cat = row.get("category_scores") or {}
    # Recursively serialize point dicts (score, confidence, evidence, feature_values)
    def ser_pt(pt):
        if isinstance(pt, list):
            return [ser_pt(x) for x in pt]
        if not isinstance(pt, dict):
            return _serialize_value(pt)
        return {k: ser_pt(v) if isinstance(v, (dict, list)) else _serialize_value(v) for k, v in pt.items()}
    out = {
        "company_id": str(row["company_id"]) if row.get("company_id") else None,
        "ticker": row.get("ticker"),
        "snapshot_at": row["snapshot_at"].isoformat() if row.get("snapshot_at") else None,
        "total_score": _serialize_value(row.get("total_score")),
        "version": row.get("version"),
        "points": {k: ser_pt(v) for k, v in points.items()},
        "category_scores": {k: _serialize_value(v) for k, v in cat.items()},
    }
    return out


@bp.route("/snapshot")
def fisher_snapshot():
    """
    Get latest Fisher score snapshot for a ticker.

    Query params:
        ticker: Stock ticker (required)

    Returns:
        JSON: total_score, confidence, category_scores, points (score, confidence, evidence, feature_values per point).
    """
    ticker = request.args.get("ticker")
    if not ticker:
        return jsonify({"error": "ticker parameter is required"}), 400
    ticker = ticker.upper().strip()
    try:
        with get_connection() as conn:
            cid = company_id_by_ticker(conn, ticker)
            if not cid:
                return jsonify({"error": "ticker not in Fisher universe", "ticker": ticker}), 404
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT s.id, s.company_id, c.ticker, s.snapshot_at, s.points, s.category_scores, s.total_score, s.version
                    FROM fisher_score_snapshot s
                    JOIN fisher_company c ON c.id = s.company_id
                    WHERE s.company_id = %s
                    ORDER BY s.snapshot_at DESC
                    LIMIT 1
                    """,
                    (cid,),
                )
                row = cur.fetchone()
            if not row:
                return jsonify({"error": "no snapshot for ticker", "ticker": ticker}), 404
            return jsonify(_serialize_snapshot(dict(row)))
    except RuntimeError as e:
        return jsonify({"error": str(e)}), 503
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@bp.route("/delta")
def fisher_delta():
    """
    What changed since last quarter: point deltas, new filings.

    Query params:
        ticker: Stock ticker (required)

    Returns:
        JSON: current_snapshot_at, previous_snapshot_at, point_deltas (point_id -> score_delta), new_filings (optional).
    """
    ticker = request.args.get("ticker")
    if not ticker:
        return jsonify({"error": "ticker parameter is required"}), 400
    ticker = ticker.upper().strip()
    try:
        with get_connection() as conn:
            cid = company_id_by_ticker(conn, ticker)
            if not cid:
                return jsonify({"error": "ticker not in Fisher universe", "ticker": ticker}), 404
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT s.snapshot_at, s.points
                    FROM fisher_score_snapshot s
                    WHERE s.company_id = %s
                    ORDER BY s.snapshot_at DESC
                    LIMIT 2
                    """,
                    (cid,),
                )
                rows = cur.fetchall()
            rows = [dict(r) for r in rows]
            if len(rows) < 2:
                return jsonify({
                    "ticker": ticker,
                    "current_snapshot_at": rows[0]["snapshot_at"].isoformat() if rows else None,
                    "previous_snapshot_at": None,
                    "point_deltas": {},
                    "message": "Only one or zero snapshots; no delta available.",
                })
            current = rows[0]
            previous = rows[1]
            curr_pts = current.get("points") or {}
            prev_pts = previous.get("points") or {}
            point_deltas = {}
            for pid in set(list(curr_pts.keys()) + list(prev_pts.keys())):
                c_score = curr_pts.get(pid, {}).get("score") if isinstance(curr_pts.get(pid), dict) else None
                p_score = prev_pts.get(pid, {}).get("score") if isinstance(prev_pts.get(pid), dict) else None
                if c_score is not None and p_score is not None:
                    point_deltas[str(pid)] = round(float(c_score) - float(p_score), 2)
                elif c_score is not None:
                    point_deltas[str(pid)] = round(float(c_score), 2)
                elif p_score is not None:
                    point_deltas[str(pid)] = -round(float(p_score), 2)
            return jsonify({
                "ticker": ticker,
                "current_snapshot_at": current["snapshot_at"].isoformat(),
                "previous_snapshot_at": previous["snapshot_at"].isoformat(),
                "point_deltas": point_deltas,
            })
    except RuntimeError as e:
        return jsonify({"error": str(e)}), 503
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@bp.route("/evidence")
def fisher_evidence():
    """
    Evidence list and excerpts for a given point.

    Query params:
        ticker: Stock ticker (required)
        point_id: Fisher point id 1-15 (required)

    Returns:
        JSON: point_id, evidence (list), score, confidence, feature_values.
    """
    ticker = request.args.get("ticker")
    point_id = request.args.get("point_id")
    if not ticker:
        return jsonify({"error": "ticker parameter is required"}), 400
    if not point_id:
        return jsonify({"error": "point_id parameter is required"}), 400
    ticker = ticker.upper().strip()
    try:
        with get_connection() as conn:
            cid = company_id_by_ticker(conn, ticker)
            if not cid:
                return jsonify({"error": "ticker not in Fisher universe", "ticker": ticker}), 404
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT s.points
                    FROM fisher_score_snapshot s
                    WHERE s.company_id = %s
                    ORDER BY s.snapshot_at DESC
                    LIMIT 1
                    """,
                    (cid,),
                )
                row = cur.fetchone()
            if not row or not row.get("points"):
                return jsonify({"error": "no snapshot for ticker", "ticker": ticker}), 404
            points = row["points"]
            pt = points.get(str(point_id)) or points.get(point_id)
            if not pt or not isinstance(pt, dict):
                return jsonify({"error": "point_id not in snapshot", "ticker": ticker, "point_id": point_id}), 404
            evidence = pt.get("evidence") or []
            score = _serialize_value(pt.get("score"))
            confidence = _serialize_value(pt.get("confidence"))
            feature_values = pt.get("feature_values") or {}
            if isinstance(feature_values, dict):
                feature_values = {k: _serialize_value(v) for k, v in feature_values.items()}
            return jsonify({
                "ticker": ticker,
                "point_id": point_id,
                "score": score,
                "confidence": confidence,
                "evidence": evidence,
                "feature_values": feature_values,
            })
    except RuntimeError as e:
        return jsonify({"error": str(e)}), 503
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@bp.route("/universe")
def fisher_universe():
    """
    List of tickers in the Fisher universe (S&P 500 or config).

    Returns:
        JSON: tickers (list of strings).
    """
    try:
        constituents = get_sp500_constituents()
        tickers = [c.get("ticker") for c in constituents if c.get("ticker")]
        return jsonify({"tickers": tickers})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@bp.route("/growth-profitable")
def fisher_growth_profitable():
    """
    High growth + profitable companies (latest snapshot per company, filtered by category scores).

    Query params:
        min_growth: minimum growth category score 0-10 (default 6)
        min_financials: minimum financials category score 0-10 (default 6)
        limit: max rows (default 100)

    Returns:
        JSON: companies (list of {ticker, name, sector, growth, financials, total_score, snapshot_at}).
    """
    min_growth = request.args.get("min_growth", type=float, default=6.0)
    min_financials = request.args.get("min_financials", type=float, default=6.0)
    limit = request.args.get("limit", type=int, default=100)
    try:
        with get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
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
                    ORDER BY (l.category_scores->>'growth')::float + (l.category_scores->>'financials')::float DESC NULLS LAST,
                             l.total_score DESC NULLS LAST,
                             c.ticker
                    LIMIT %s
                    """,
                    (min_growth, min_financials, limit),
                )
                rows = cur.fetchall()
        companies = []
        for r in rows:
            r = dict(r)
            companies.append({
                "ticker": r.get("ticker"),
                "name": r.get("name") or r.get("ticker"),
                "sector": r.get("sector") or "",
                "growth": _serialize_value(r.get("growth")),
                "financials": _serialize_value(r.get("financials")),
                "total_score": _serialize_value(r.get("total_score")),
                "snapshot_at": r["snapshot_at"].isoformat() if r.get("snapshot_at") else None,
            })
        return jsonify({"companies": companies})
    except RuntimeError as e:
        return jsonify({"error": str(e)}), 503
    except Exception as e:
        return jsonify({"error": str(e)}), 500
