from __future__ import annotations

import os

from flask import Blueprint, jsonify, request

bp = Blueprint("news", __name__)


@bp.route("/news.json")
def get_news_json():
    """Serve news JSON file (updated by news bot) with earnings added"""
    try:
        import json
        from datetime import datetime, timedelta, date

        # Load base news file
        news_path = os.path.join(os.getcwd(), "data/news.json")
        if not os.path.exists(news_path):
            # Try parent directory
            news_path = os.path.join(
                os.path.dirname(os.path.dirname(__file__)), "data", "news.json"
            )

        news_data = []
        if os.path.exists(news_path):
            try:
                with open(news_path, "r") as f:
                    news_data = json.load(f)
                    if not isinstance(news_data, list):
                        news_data = []
            except (json.JSONDecodeError, ValueError):
                news_data = []

        # Add upcoming earnings to news
        try:
            config_path = os.path.join(
                os.path.dirname(os.path.dirname(__file__)), "data", "config.json"
            )
            config = {}
            if os.path.exists(config_path):
                with open(config_path, "r") as f:
                    config = json.load(f)

            # Get earnings from cockpit events (reuse that logic)
            spy_top_holdings = [
                "AAPL",
                "MSFT",
                "NVDA",
                "AMZN",
                "GOOGL",
                "GOOG",
                "META",
                "TSLA",
                "BRK.B",
                "JPM",
                "V",
                "MA",
                "UNH",
                "HD",
                "PG",
                "JNJ",
                "XOM",
                "CVX",
                "ABBV",
                "MRK",
                "PFE",
                "KO",
                "PEP",
                "COST",
                "WMT",
                "BAC",
                "WFC",
                "NFLX",
                "ADBE",
                "CRM",
                "ORCL",
                "INTC",
                "AMD",
                "QCOM",
            ]

            today = datetime.now().date()
            end_date = today + timedelta(days=14)  # Next 2 weeks

            # Try yfinance for earnings
            try:
                import yfinance as yf

                print(f"[news.json] Fetching earnings from yfinance...")
                earnings_added = 0
                for ticker in spy_top_holdings[:15]:  # Top 15 SPY holdings
                    try:
                        stock = yf.Ticker(ticker)
                        calendar = stock.calendar
                        if (
                            calendar
                            and isinstance(calendar, dict)
                            and "Earnings Date" in calendar
                        ):
                            earnings_dates = calendar["Earnings Date"]
                            if isinstance(earnings_dates, list) and len(earnings_dates) > 0:
                                next_earnings = earnings_dates[0]
                                if isinstance(next_earnings, datetime):
                                    report_date = next_earnings.date()
                                elif isinstance(next_earnings, date):
                                    report_date = next_earnings
                                elif isinstance(next_earnings, str):
                                    report_date = datetime.strptime(
                                        next_earnings.split()[0], "%Y-%m-%d"
                                    ).date()
                                else:
                                    continue

                                if today <= report_date <= end_date:
                                    # Check if already added
                                    if not any(
                                        n.get("headline", "").startswith(
                                            f"{ticker} Earnings"
                                        )
                                        for n in news_data
                                    ):
                                        news_data.insert(
                                            0,
                                            {
                                                "headline": f"{ticker} Earnings Report - {report_date.isoformat()}",
                                                "summary": f"{ticker} will report earnings on {report_date.isoformat()}.",
                                                "source": "Earnings Calendar",
                                                "url": f"https://www.google.com/search?q={ticker}+earnings+{report_date.isoformat()}",
                                                "published_date": report_date.isoformat(),
                                                "category": "earnings",
                                            },
                                        )
                                        earnings_added += 1
                    except Exception:
                        continue
                print(f"[news.json] Added {earnings_added} earnings to news")
            except Exception as e:
                print(f"[news.json] Error adding earnings from yfinance: {e}")
                import traceback

                traceback.print_exc()

        except Exception as e:
            print(f"[news.json] Error enhancing news with earnings: {e}")
            import traceback

            traceback.print_exc()

        return jsonify(news_data)
    except FileNotFoundError:
        return jsonify({"error": "data/news.json not found"}), 404
    except Exception as e:
        print(f"[news.json] General error: {e}")
        import traceback

        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


@bp.route("/news/intelligence")
def get_news_intelligence():
    """Get news directly from intelligence bot database with filters"""
    try:
        import sys
        from pathlib import Path

        # Add news_bot to path
        news_bot_path = Path(__file__).parent.parent / "news_bot"
        if str(news_bot_path) not in sys.path:
            sys.path.insert(0, str(news_bot_path.parent))

        from news_bot.database import NewsDatabase
        from news_bot import config as news_config

        # Get query parameters
        importance = request.args.get("importance", type=int)  # Filter by importance
        limit = int(request.args.get("limit", 50))

        with NewsDatabase(news_config.DB_PATH) as db:
            articles = db.get_recent_articles_for_export(limit=limit)

        # Filter by importance if specified
        if importance:
            if importance >= 4:
                articles = [a for a in articles if a.get("impact") in ["critical", "high"]]

        return jsonify(
            {"articles": articles, "total_count": len(articles), "source": "News Intelligence Bot"}
        )

    except Exception as e:
        return jsonify({"error": f"Failed to get news intelligence: {e}"}), 500


@bp.route("/news/status")
def get_news_bot_status():
    """Get news bot service status"""
    try:
        import json
        from pathlib import Path

        status_file = Path(__file__).parent.parent / "data" / "news_bot_status.json"

        if status_file.exists():
            with open(status_file, "r") as f:
                status = json.load(f)
            return jsonify(status)
        else:
            return jsonify({"error": "News bot status not available"}), 404

    except Exception as e:
        return jsonify({"error": f"Failed to get news bot status: {e}"}), 500

