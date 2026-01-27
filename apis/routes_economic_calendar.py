from __future__ import annotations

from flask import Blueprint, jsonify, request

bp = Blueprint("economic_calendar", __name__)


# Economic Calendar Endpoints
@bp.route("/economic-calendar")
def get_economic_calendar():
    """Get economic calendar events for market news app calendar tab"""
    try:
        import sys
        from pathlib import Path

        quant_engine_path = Path(__file__).parent / "QuantEngine"
        if str(quant_engine_path) not in sys.path:
            sys.path.insert(0, str(quant_engine_path))

        from data_broker import QuantBotDataBroker

        broker = QuantBotDataBroker("QuantEngine/quantbot_data.db")

        # Get query parameters
        impact_filter = request.args.get("impact")  # high, medium, low
        limit = int(request.args.get("limit", 50))

        calendar_events = broker.get_economic_calendar(limit=limit, impact_filter=impact_filter)

        return jsonify(
            {
                "events": calendar_events,
                "total_count": len(calendar_events),
                "impact_filter": impact_filter,
                "source": "FRED Economic Data",
            }
        )

    except Exception as e:
        return jsonify({"error": f"Failed to get economic calendar: {e}"}), 500


@bp.route("/economic-calendar/upcoming")
def get_upcoming_economic_events():
    """Get upcoming economic events within next few days"""
    try:
        import sys
        from pathlib import Path

        quant_engine_path = Path(__file__).parent / "QuantEngine"
        if str(quant_engine_path) not in sys.path:
            sys.path.insert(0, str(quant_engine_path))

        from data_broker import QuantBotDataBroker

        broker = QuantBotDataBroker("QuantEngine/quantbot_data.db")

        # Get days ahead parameter (default 7 days)
        days_ahead = int(request.args.get("days", 7))

        upcoming_events = broker.get_upcoming_economic_events(days_ahead=days_ahead)

        return jsonify(
            {
                "events": upcoming_events,
                "days_ahead": days_ahead,
                "total_count": len(upcoming_events),
                "source": "FRED Economic Data",
            }
        )

    except Exception as e:
        return jsonify({"error": f"Failed to get upcoming economic events: {e}"}), 500


@bp.route("/economic-calendar/high-impact")
def get_high_impact_events():
    """Get high impact economic events"""
    try:
        import sys
        from pathlib import Path

        quant_engine_path = Path(__file__).parent / "QuantEngine"
        if str(quant_engine_path) not in sys.path:
            sys.path.insert(0, str(quant_engine_path))

        from data_broker import QuantBotDataBroker

        broker = QuantBotDataBroker("QuantEngine/quantbot_data.db")

        # Get high impact events
        high_impact_events = broker.get_economic_calendar(limit=20, impact_filter="high")

        return jsonify(
            {
                "events": high_impact_events,
                "impact_level": "high",
                "total_count": len(high_impact_events),
                "source": "FRED Economic Data",
            }
        )

    except Exception as e:
        return jsonify({"error": f"Failed to get high impact events: {e}"}), 500

