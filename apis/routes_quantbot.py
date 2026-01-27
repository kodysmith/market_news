from __future__ import annotations

from flask import Blueprint, jsonify, request

bp = Blueprint("quantbot", __name__)


# QuantBot Data Broker Endpoints
@bp.route("/quantbot/opportunities")
def get_quantbot_opportunities():
    """Get QuantBot trading opportunities from database"""
    try:
        import sys
        from pathlib import Path

        # Add QuantEngine to path
        quant_engine_path = Path(__file__).parent / "QuantEngine"
        if str(quant_engine_path) not in sys.path:
            sys.path.insert(0, str(quant_engine_path))

        from data_broker import QuantBotDataBroker

        broker = QuantBotDataBroker("QuantEngine/quantbot_data.db")
        opportunities = broker.get_opportunities(limit=20)

        return jsonify(
            {
                "opportunities": opportunities,
                "total_count": len(opportunities),
                "source": "QuantBot Database",
            }
        )

    except Exception as e:
        return jsonify({"error": f"Failed to get QuantBot opportunities: {e}"}), 500


@bp.route("/quantbot/market-analysis")
def get_market_analysis():
    """Get latest market analysis from QuantBot"""
    try:
        import sys
        from pathlib import Path

        quant_engine_path = Path(__file__).parent / "QuantEngine"
        if str(quant_engine_path) not in sys.path:
            sys.path.insert(0, str(quant_engine_path))

        from data_broker import QuantBotDataBroker

        broker = QuantBotDataBroker("QuantEngine/quantbot_data.db")
        analysis = broker.get_latest_market_analysis()

        if analysis:
            return jsonify(analysis)
        else:
            return jsonify({"error": "No market analysis available"}), 404

    except Exception as e:
        return jsonify({"error": f"Failed to get market analysis: {e}"}), 500


@bp.route("/quantbot/news")
def get_quantbot_news():
    """Get enhanced news feed from QuantBot database"""
    try:
        import sys
        from pathlib import Path

        quant_engine_path = Path(__file__).parent / "QuantEngine"
        if str(quant_engine_path) not in sys.path:
            sys.path.insert(0, str(quant_engine_path))

        from data_broker import QuantBotDataBroker

        broker = QuantBotDataBroker("QuantEngine/quantbot_data.db")
        news_items = broker.get_news_feed(limit=30)

        # Convert to format expected by news.json
        formatted_news = []
        for item in news_items:
            formatted_news.append(
                {
                    "headline": item["headline"],
                    "source": item["source"],
                    "url": item["url"] or "",
                    "summary": item["summary"] or "",
                    "sentiment": item["sentiment"] or "neutral",
                    "tickers": item["tickers"] or [],
                    "type": item["type"] or "news",
                    "impact": item["impact"] or "medium",
                    "published_date": item["published_date"] or "",
                }
            )

        return jsonify(formatted_news)

    except Exception as e:
        return jsonify({"error": f"Failed to get QuantBot news: {e}"}), 500


@bp.route("/quantbot/signals")
def get_trading_signals():
    """Get trading signals from QuantBot"""
    try:
        import sys
        from pathlib import Path

        quant_engine_path = Path(__file__).parent / "QuantEngine"
        if str(quant_engine_path) not in sys.path:
            sys.path.insert(0, str(quant_engine_path))

        from data_broker import QuantBotDataBroker

        broker = QuantBotDataBroker("QuantEngine/quantbot_data.db")
        signals = broker.get_trading_signals(limit=20)

        return jsonify(
            {"signals": signals, "total_count": len(signals), "source": "QuantBot Database"}
        )

    except Exception as e:
        return jsonify({"error": f"Failed to get trading signals: {e}"}), 500


@bp.route("/quantbot/status")
def get_quantbot_status():
    """Get QuantBot system status"""
    try:
        import sys
        from pathlib import Path

        quant_engine_path = Path(__file__).parent / "QuantEngine"
        if str(quant_engine_path) not in sys.path:
            sys.path.insert(0, str(quant_engine_path))

        from data_broker import QuantBotDataBroker

        broker = QuantBotDataBroker("QuantEngine/quantbot_data.db")
        stats = broker.get_database_stats()

        # Check if QuantBot is running (look for recent data)
        opportunities = broker.get_opportunities(limit=1)
        is_active = len(opportunities) > 0

        return jsonify(
            {
                "is_active": is_active,
                "database_stats": stats,
                "last_opportunity": opportunities[0] if opportunities else None,
                "source": "QuantBot Database",
            }
        )

    except Exception as e:
        return jsonify({"error": f"Failed to get QuantBot status: {e}"}), 500

