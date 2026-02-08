from __future__ import annotations

from flask import Blueprint, jsonify

bp = Blueprint("index", __name__)


@bp.route("/")
def index():
    """API index with endpoint documentation"""
    endpoints = {
        "status": "Market News API is running",
        "endpoints": {
            "report": "/report.json",
            "news": {
                "json": "/news.json",
                "intelligence": "/news/intelligence",
                "status": "/news/status",
            },
            "quantbot": {
                "opportunities": "/quantbot/opportunities",
                "market_analysis": "/quantbot/market-analysis",
                "news": "/quantbot/news",
                "signals": "/quantbot/signals",
                "status": "/quantbot/status",
            },
            "economic_calendar": {
                "all": "/economic-calendar",
                "upcoming": "/economic-calendar/upcoming",
                "high_impact": "/economic-calendar/high-impact",
            },
            "gex": {
                "calculate": "/gex/calculate?ticker=SPY",
                "summary": "/gex/summary",
                "tickers": "/gex/tickers",
                "price_comparison": "/gex/price-comparison?ticker=SPY",
                "max_pain": "/gex/max-pain?ticker=SPY&dte=0",
            },
            "valuation": {
                "calculate": "/valuation/calculate?ticker=AAPL",
                "history": "/valuation/history?ticker=AAPL&days=365",
                "divergence": "/valuation/divergence?ticker=AAPL",
                "scan": "/valuation/scan?tickers=AAPL,MSFT,GOOGL&threshold=20",
                "watchlist": "/valuation/watchlist",
                "alerts": "/valuation/alerts",
                "undervalued": "/valuation/undervalued?threshold=20",
            },
            "cockpit": {
                "state": "/cockpit/state?ticker=SPY",
                "tickers": "/cockpit/tickers",
                "events": "/cockpit/events?days=3",
            },
            "depin": {
                "risk": "/depin/risk?ticker=SPY",
                "state": "/depin/state?ticker=SPY",
                "history": "/depin/history?ticker=SPY&limit=100",
            },
            "fisher": {
                "snapshot": "/fisher/snapshot?ticker=AAPL",
                "delta": "/fisher/delta?ticker=AAPL",
                "evidence": "/fisher/evidence?ticker=AAPL&point_id=1",
                "universe": "/fisher/universe",
                "growth_profitable": "/fisher/growth-profitable?min_growth=6&min_financials=6&limit=100",
            },
        },
    }
    return jsonify(endpoints)

