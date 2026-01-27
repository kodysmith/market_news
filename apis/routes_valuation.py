from __future__ import annotations

from flask import Blueprint, jsonify, request

# Ensure shared path setup runs (QuantEngine/utils on sys.path)
from . import shared as _shared  # noqa: F401

from QuantEngine.intrinsic_value_calculator import calculate_intrinsic_value, batch_calculate
from QuantEngine.intrinsic_value_database import (
    store_valuation,
    get_history,
    add_to_watchlist,
    remove_from_watchlist,
    get_watchlist,
    check_alerts,
    get_recent_alerts,
    acknowledge_alert,
    get_divergence_scan,
)

bp = Blueprint("valuation", __name__)


# ===============================
# Intrinsic Value / Valuation Endpoints
# ===============================

@bp.route("/valuation/calculate")
def valuation_calculate():
    """
    Calculate intrinsic value using all 6 methods.

    Query params:
        ticker: Stock ticker symbol (required)
        store: Whether to store the result (default: true)

    Returns:
        JSON with all valuations, composite value, and divergence percentage
    """
    ticker = request.args.get("ticker")
    if not ticker:
        return jsonify({"error": "ticker parameter is required"}), 400

    store = request.args.get("store", "true").lower() == "true"

    try:
        result = calculate_intrinsic_value(ticker.upper())

        if "error" in result and not result.get("valuations"):
            return jsonify(result), 500

        # Store in database if requested
        if store and result.get("valuations"):
            store_valuation(result)
            # Check for alerts
            alerts = check_alerts(result)
            if alerts:
                result["alerts"] = alerts

        return jsonify(result)

    except Exception as e:
        return jsonify({"error": f"Valuation calculation failed: {str(e)}"}), 500


@bp.route("/valuation/history")
def valuation_history():
    """
    Get historical intrinsic value data for a ticker.

    Query params:
        ticker: Stock ticker symbol (required)
        days: Number of days of history (default: 365)

    Returns:
        JSON array of historical valuation records
    """
    ticker = request.args.get("ticker")
    if not ticker:
        return jsonify({"error": "ticker parameter is required"}), 400

    days = request.args.get("days", 365, type=int)

    try:
        history = get_history(ticker.upper(), days)
        return jsonify(
            {
                "ticker": ticker.upper(),
                "days": days,
                "records": len(history),
                "history": history,
            }
        )

    except Exception as e:
        return jsonify({"error": f"Failed to get history: {str(e)}"}), 500


@bp.route("/valuation/divergence")
def valuation_divergence():
    """
    Get current divergence from intrinsic value with detailed breakdown.

    Query params:
        ticker: Stock ticker symbol (required)

    Returns:
        JSON with current price, each valuation method, and divergence
    """
    ticker = request.args.get("ticker")
    if not ticker:
        return jsonify({"error": "ticker parameter is required"}), 400

    try:
        # Calculate fresh values
        result = calculate_intrinsic_value(ticker.upper())

        if "error" in result and not result.get("valuations"):
            return jsonify(result), 500

        # Format response focused on divergence
        valuations = result.get("valuations", {})
        composite = result.get("composite", {})

        divergence_breakdown = {}
        for method, data in valuations.items():
            if data.get("applicable") and data.get("value"):
                method_divergence = ((result["current_price"] - data["value"]) / data["value"]) * 100
                divergence_breakdown[method] = {
                    "intrinsic_value": data["value"],
                    "divergence_pct": round(method_divergence, 2),
                    "confidence": data.get("confidence", 0),
                }

        return jsonify(
            {
                "ticker": ticker.upper(),
                "company_name": result.get("company_name"),
                "current_price": result.get("current_price"),
                "composite_intrinsic_value": composite.get("value"),
                "composite_divergence_pct": composite.get("divergence_pct"),
                "verdict": composite.get("verdict"),
                "breakdown": divergence_breakdown,
                "timestamp": result.get("timestamp"),
            }
        )

    except Exception as e:
        return jsonify({"error": f"Failed to calculate divergence: {str(e)}"}), 500


@bp.route("/valuation/scan")
def valuation_scan():
    """
    Batch scan multiple tickers for intrinsic value.

    Query params:
        tickers: Comma-separated list of ticker symbols (required)
        threshold: Divergence threshold percentage (default: 20)

    Returns:
        JSON array of valuation results sorted by divergence
    """
    tickers_param = request.args.get("tickers")
    if not tickers_param:
        return jsonify({"error": "tickers parameter is required"}), 400

    tickers = [t.strip().upper() for t in tickers_param.split(",") if t.strip()]
    threshold = request.args.get("threshold", 20, type=float)

    if not tickers:
        return jsonify({"error": "No valid tickers provided"}), 400

    try:
        results = batch_calculate(tickers)

        # Store results and check alerts
        for result in results:
            if result.get("valuations"):
                store_valuation(result)
                check_alerts(result)

        # Filter by threshold and sort by divergence
        filtered = []
        for r in results:
            if r.get("composite", {}).get("divergence_pct") is not None:
                div = r["composite"]["divergence_pct"]
                if abs(div) >= threshold:
                    filtered.append(
                        {
                            "ticker": r["ticker"],
                            "company_name": r.get("company_name"),
                            "current_price": r.get("current_price"),
                            "composite_value": r["composite"].get("value"),
                            "divergence_pct": div,
                            "verdict": r["composite"].get("verdict"),
                        }
                    )

        # Sort by divergence (most undervalued first)
        filtered.sort(key=lambda x: x["divergence_pct"])

        return jsonify(
            {
                "tickers_requested": len(tickers),
                "results_with_data": len([r for r in results if r.get("valuations")]),
                "exceeding_threshold": len(filtered),
                "threshold_pct": threshold,
                "results": filtered,
            }
        )

    except Exception as e:
        return jsonify({"error": f"Batch scan failed: {str(e)}"}), 500


@bp.route("/valuation/watchlist", methods=["GET", "POST", "DELETE"])
def valuation_watchlist():
    """
    Manage the valuation watchlist.

    GET: Get all tickers in watchlist with latest valuations
    POST: Add ticker to watchlist (requires 'ticker' in JSON body)
    DELETE: Remove ticker from watchlist (requires 'ticker' in JSON body)
    """
    if request.method == "GET":
        try:
            watchlist = get_watchlist()
            return jsonify({"count": len(watchlist), "watchlist": watchlist})
        except Exception as e:
            return jsonify({"error": f"Failed to get watchlist: {str(e)}"}), 500

    elif request.method == "POST":
        data = request.get_json() or {}
        ticker = data.get("ticker")
        if not ticker:
            return jsonify({"error": "ticker is required"}), 400

        try:
            # Optionally calculate fresh values
            result = calculate_intrinsic_value(ticker.upper())
            company_name = result.get("company_name")
            sector = result.get("sector")

            # Add to watchlist
            alert_threshold = data.get("alert_threshold", 20.0)
            success = add_to_watchlist(ticker.upper(), company_name, sector, alert_threshold)

            if success:
                # Store the valuation
                if result.get("valuations"):
                    store_valuation(result)

                return jsonify(
                    {
                        "success": True,
                        "ticker": ticker.upper(),
                        "company_name": company_name,
                        "message": f"Added {ticker.upper()} to watchlist",
                    }
                )
            else:
                return jsonify({"error": "Failed to add to watchlist"}), 500

        except Exception as e:
            return jsonify({"error": f"Failed to add to watchlist: {str(e)}"}), 500

    elif request.method == "DELETE":
        data = request.get_json() or {}
        ticker = data.get("ticker")
        if not ticker:
            return jsonify({"error": "ticker is required"}), 400

        try:
            success = remove_from_watchlist(ticker.upper())
            if success:
                return jsonify({"success": True, "message": f"Removed {ticker.upper()} from watchlist"})
            else:
                return jsonify({"error": "Failed to remove from watchlist"}), 500

        except Exception as e:
            return jsonify({"error": f"Failed to remove from watchlist: {str(e)}"}), 500


@bp.route("/valuation/alerts")
def valuation_alerts():
    """
    Get recent valuation alerts.

    Query params:
        limit: Maximum number of alerts to return (default: 50)

    Returns:
        JSON array of recent alerts
    """
    limit = request.args.get("limit", 50, type=int)

    try:
        alerts = get_recent_alerts(limit)
        return jsonify({"count": len(alerts), "alerts": alerts})

    except Exception as e:
        return jsonify({"error": f"Failed to get alerts: {str(e)}"}), 500


@bp.route("/valuation/alerts/<int:alert_id>/acknowledge", methods=["POST"])
def acknowledge_valuation_alert(alert_id: int):
    """Acknowledge a valuation alert"""
    try:
        success = acknowledge_alert(alert_id)
        if success:
            return jsonify({"success": True, "message": f"Alert {alert_id} acknowledged"})
        else:
            return jsonify({"error": "Failed to acknowledge alert"}), 500
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@bp.route("/valuation/undervalued")
def valuation_undervalued():
    """
    Get stocks that are significantly undervalued.

    Query params:
        threshold: Divergence threshold percentage (default: 20)

    Returns:
        JSON array of undervalued stocks from database
    """
    threshold = request.args.get("threshold", 20, type=float)

    try:
        results = get_divergence_scan(threshold)
        # Filter for undervalued only (negative divergence)
        undervalued = [r for r in results if r.get("divergence_pct", 0) < 0]

        return jsonify({"threshold_pct": threshold, "count": len(undervalued), "stocks": undervalued})

    except Exception as e:
        return jsonify({"error": f"Failed to scan: {str(e)}"}), 500

