"""
Scanner API: serve premarket gap scanner results from Supabase.
GET /trade-ideas/scanner/today
GET /trade-ideas/scanner/<date>
GET /trade-ideas/scanner/today/<symbol>
"""
from __future__ import annotations

import os
from datetime import datetime

from flask import Blueprint, jsonify, request

bp = Blueprint("scanner", __name__, url_prefix="/trade-ideas/scanner")


def _get_supabase():
    url = (os.environ.get("SUPABASE_URL") or "").strip().rstrip("/")
    key = (os.environ.get("SUPABASE_SECRET_KEY") or os.environ.get("SUPABASE_SERVICE_ROLE_KEY") or "").strip()
    if not url or not key:
        return None
    try:
        from supabase import create_client
        return create_client(url, key)
    except ImportError:
        return None


def _today_et_date_str() -> str:
    try:
        from zoneinfo import ZoneInfo
        tz = ZoneInfo("America/New_York")
    except ImportError:
        try:
            import pytz
            tz = pytz.timezone("America/New_York")
        except ImportError:
            return datetime.utcnow().strftime("%Y-%m-%d")
    return datetime.now(tz).strftime("%Y-%m-%d")


@bp.route("/today")
def scanner_today():
    """
    GET /trade-ideas/scanner/today
    Returns today's ranked scanner ideas (payloads) and optional stale_data flag.
    """
    supabase = _get_supabase()
    if not supabase:
        return jsonify({"error": "Scanner backend not configured", "ideas": [], "stale_data": False}), 503

    date_str = _today_et_date_str()
    try:
        # Get runs for today (as_of_et date part = date_str)
        runs = supabase.table("scan_runs").select("id,as_of_et").order("as_of_et", desc=True).limit(50).execute()
        run_ids = []
        for row in (runs.data or []):
            as_of = row.get("as_of_et") or ""
            if as_of.startswith(date_str):
                run_ids.append(row["id"])
        if not run_ids:
            return jsonify({"ideas": [], "stale_data": False, "as_of": None})

        # Latest run for today
        run_id = run_ids[0]
        rows = (
            supabase.table("scanner_trade_ideas")
            .select("symbol,rank,total_score,payload")
            .eq("run_id", run_id)
            .order("rank")
            .execute()
        )
        data = rows.data or []
        ideas = [r["payload"] for r in data]
        as_of = None
        if data and runs.data:
            as_of = runs.data[0].get("as_of_et")
        return jsonify({"ideas": ideas, "stale_data": False, "as_of": as_of})
    except Exception as e:
        return jsonify({"error": str(e), "ideas": [], "stale_data": False}), 500


@bp.route("/<date_str>")
def scanner_by_date(date_str: str):
    """
    GET /trade-ideas/scanner/2024-01-15
    Returns ranked scanner ideas for the given date (YYYY-MM-DD).
    """
    if len(date_str) != 10 or date_str[4] != "-" or date_str[7] != "-":
        return jsonify({"error": "Invalid date; use YYYY-MM-DD"}), 400

    supabase = _get_supabase()
    if not supabase:
        return jsonify({"error": "Scanner backend not configured", "ideas": []}), 503

    try:
        runs = supabase.table("scan_runs").select("id,as_of_et").order("as_of_et", desc=True).limit(200).execute()
        run_id = None
        for row in (runs.data or []):
            as_of = row.get("as_of_et") or ""
            if as_of.startswith(date_str):
                run_id = row["id"]
                break
        if not run_id:
            return jsonify({"ideas": [], "date": date_str})

        rows = (
            supabase.table("scanner_trade_ideas")
            .select("symbol,rank,total_score,payload")
            .eq("run_id", run_id)
            .order("rank")
            .execute()
        )
        ideas = [r["payload"] for r in (rows.data or [])]
        return jsonify({"ideas": ideas, "date": date_str})
    except Exception as e:
        return jsonify({"error": str(e), "ideas": []}), 500


@bp.route("/today/<symbol>")
def scanner_today_symbol(symbol: str):
    """
    GET /trade-ideas/scanner/today/AAPL
    Returns the single scanner idea for symbol from today's run (detail view).
    """
    symbol = (symbol or "").upper().strip()
    if not symbol or len(symbol) > 10:
        return jsonify({"error": "Invalid symbol"}), 400

    supabase = _get_supabase()
    if not supabase:
        return jsonify({"error": "Scanner backend not configured"}), 503

    date_str = _today_et_date_str()
    try:
        runs = supabase.table("scan_runs").select("id,as_of_et").order("as_of_et", desc=True).limit(50).execute()
        run_id = None
        for row in (runs.data or []):
            as_of = (row.get("as_of_et") or "")[:10]
            if as_of == date_str:
                run_id = row["id"]
                break
        if not run_id:
            return jsonify({"idea": None, "symbol": symbol})

        row = (
            supabase.table("scanner_trade_ideas")
            .select("payload")
            .eq("run_id", run_id)
            .eq("symbol", symbol)
            .maybe_single()
            .execute()
        )
        idea = (row.data or {}).get("payload") if row.data else None
        return jsonify({"idea": idea, "symbol": symbol})
    except Exception as e:
        return jsonify({"error": str(e)}), 500
