from flask import Flask, send_from_directory, jsonify, request
import os
import sys
from flask_cors import CORS
import subprocess
import time

# Add parent directory to path for QuantEngine imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from QuantEngine.intrinsic_value_calculator import IntrinsicValueCalculator, calculate_intrinsic_value, batch_calculate
from QuantEngine.intrinsic_value_database import (
    store_valuation, get_history, get_latest_valuation,
    add_to_watchlist, remove_from_watchlist, get_watchlist,
    check_alerts, get_recent_alerts, acknowledge_alert, get_divergence_scan
)
from QuantEngine.decision_cockpit import DecisionCockpit, get_cockpit_state
from QuantEngine.volatility_analyzer import VolatilityAnalyzer, analyze_volatility

app = Flask(__name__)
CORS(app)  # This will allow all domains. For production, restrict origins!

REPORT_PATH = 'data/report.json'
REFRESH_INTERVAL = 30 * 60  # 30 minutes in seconds

def is_report_stale():
    if not os.path.exists(REPORT_PATH):
        return True
    mtime = os.path.getmtime(REPORT_PATH)
    return (time.time() - mtime) > REFRESH_INTERVAL

@app.route('/report.json')
def get_report_json():
    try:
        # Try to find report.json in current directory or parent
        report_path = REPORT_PATH
        if not os.path.exists(report_path):
            # Try parent directory (apis/../data/report.json)
            parent_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data', 'report.json')
            if os.path.exists(parent_path):
                report_path = parent_path
            else:
                # If report doesn't exist, skip regeneration and return error
                return jsonify({
                    "error": "report.json not found",
                    "message": "Report file not available. Some features may not work."
                }), 404
        
        # Only try to regenerate if explicitly stale (skip for now to avoid errors)
        # if is_report_stale():
        #     try:
        #         subprocess.run(['python3', 'generate_report.py'], check=True, cwd=os.path.dirname(os.path.dirname(__file__)))
        #     except (subprocess.CalledProcessError, FileNotFoundError):
        #         pass  # Continue with existing file if regeneration fails
        
        # Serve the file
        abs_path = os.path.abspath(report_path)
        directory = os.path.dirname(abs_path)
        filename = os.path.basename(abs_path)
        return send_from_directory(directory, filename)
    except FileNotFoundError:
        return jsonify({"error": "report.json not found"}), 404
    except Exception as e:
        return jsonify({"error": f"Failed to serve report: {str(e)}"}), 500

@app.route('/news.json')
def get_news_json():
    """Serve news JSON file (updated by news bot) with earnings added"""
    try:
        import json
        from datetime import datetime, timedelta, date
        
        # Load base news file
        news_path = os.path.join(os.getcwd(), "data/news.json")
        if not os.path.exists(news_path):
            # Try parent directory
            news_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data', 'news.json')
        
        news_data = []
        if os.path.exists(news_path):
            try:
                with open(news_path, 'r') as f:
                    news_data = json.load(f)
                    if not isinstance(news_data, list):
                        news_data = []
            except (json.JSONDecodeError, ValueError):
                news_data = []
        
        # Add upcoming earnings to news
        try:
            config_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data', 'config.json')
            config = {}
            if os.path.exists(config_path):
                with open(config_path, 'r') as f:
                    config = json.load(f)
            
            # Get earnings from cockpit events (reuse that logic)
            spy_top_holdings = ['AAPL', 'MSFT', 'NVDA', 'AMZN', 'GOOGL', 'GOOG', 'META', 'TSLA', 'BRK.B', 'JPM', 
                               'V', 'MA', 'UNH', 'HD', 'PG', 'JNJ', 'XOM', 'CVX', 'ABBV', 'MRK', 'PFE', 'KO', 'PEP',
                               'COST', 'WMT', 'BAC', 'WFC', 'NFLX', 'ADBE', 'CRM', 'ORCL', 'INTC', 'AMD', 'QCOM']
            
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
                        if calendar and isinstance(calendar, dict) and 'Earnings Date' in calendar:
                            earnings_dates = calendar['Earnings Date']
                            if isinstance(earnings_dates, list) and len(earnings_dates) > 0:
                                next_earnings = earnings_dates[0]
                                if isinstance(next_earnings, datetime):
                                    report_date = next_earnings.date()
                                elif isinstance(next_earnings, date):
                                    report_date = next_earnings
                                elif isinstance(next_earnings, str):
                                    report_date = datetime.strptime(next_earnings.split()[0], '%Y-%m-%d').date()
                                else:
                                    continue
                                
                                if today <= report_date <= end_date:
                                    # Check if already added
                                    if not any(n.get('headline', '').startswith(f'{ticker} Earnings') for n in news_data):
                                        news_data.insert(0, {
                                            'headline': f'{ticker} Earnings Report - {report_date.isoformat()}',
                                            'summary': f'{ticker} will report earnings on {report_date.isoformat()}.',
                                            'source': 'Earnings Calendar',
                                            'url': f'https://www.google.com/search?q={ticker}+earnings+{report_date.isoformat()}',
                                            'published_date': report_date.isoformat(),
                                            'category': 'earnings'
                                        })
                                        earnings_added += 1
                    except Exception as ticker_err:
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

@app.route('/news/intelligence')
def get_news_intelligence():
    """Get news directly from intelligence bot database with filters"""
    try:
        import sys
        from pathlib import Path
        
        # Add news_bot to path
        news_bot_path = Path(__file__).parent.parent / 'news_bot'
        if str(news_bot_path) not in sys.path:
            sys.path.insert(0, str(news_bot_path.parent))
        
        from news_bot.database import NewsDatabase
        from news_bot import config as news_config
        
        # Get query parameters
        importance = request.args.get('importance', type=int)  # Filter by importance
        limit = int(request.args.get('limit', 50))
        
        with NewsDatabase(news_config.DB_PATH) as db:
            articles = db.get_recent_articles_for_export(limit=limit)
        
        # Filter by importance if specified
        if importance:
            if importance >= 4:
                articles = [a for a in articles if a.get('impact') in ['critical', 'high']]
        
        return jsonify({
            'articles': articles,
            'total_count': len(articles),
            'source': 'News Intelligence Bot'
        })
    
    except Exception as e:
        return jsonify({"error": f"Failed to get news intelligence: {e}"}), 500

@app.route('/news/status')
def get_news_bot_status():
    """Get news bot service status"""
    try:
        import json
        from pathlib import Path
        
        status_file = Path(__file__).parent.parent / 'data' / 'news_bot_status.json'
        
        if status_file.exists():
            with open(status_file, 'r') as f:
                status = json.load(f)
            return jsonify(status)
        else:
            return jsonify({"error": "News bot status not available"}), 404
    
    except Exception as e:
        return jsonify({"error": f"Failed to get news bot status: {e}"}), 500

# QuantBot Data Broker Endpoints
@app.route('/quantbot/opportunities')
def get_quantbot_opportunities():
    """Get QuantBot trading opportunities from database"""
    try:
        import sys
        from pathlib import Path

        # Add QuantEngine to path
        quant_engine_path = Path(__file__).parent / 'QuantEngine'
        if str(quant_engine_path) not in sys.path:
            sys.path.insert(0, str(quant_engine_path))

        from data_broker import QuantBotDataBroker

        broker = QuantBotDataBroker("QuantEngine/quantbot_data.db")
        opportunities = broker.get_opportunities(limit=20)

        return jsonify({
            'opportunities': opportunities,
            'total_count': len(opportunities),
            'source': 'QuantBot Database'
        })

    except Exception as e:
        return jsonify({"error": f"Failed to get QuantBot opportunities: {e}"}), 500

@app.route('/quantbot/market-analysis')
def get_market_analysis():
    """Get latest market analysis from QuantBot"""
    try:
        import sys
        from pathlib import Path

        quant_engine_path = Path(__file__).parent / 'QuantEngine'
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

@app.route('/quantbot/news')
def get_quantbot_news():
    """Get enhanced news feed from QuantBot database"""
    try:
        import sys
        from pathlib import Path

        quant_engine_path = Path(__file__).parent / 'QuantEngine'
        if str(quant_engine_path) not in sys.path:
            sys.path.insert(0, str(quant_engine_path))

        from data_broker import QuantBotDataBroker

        broker = QuantBotDataBroker("QuantEngine/quantbot_data.db")
        news_items = broker.get_news_feed(limit=30)

        # Convert to format expected by news.json
        formatted_news = []
        for item in news_items:
            formatted_news.append({
                'headline': item['headline'],
                'source': item['source'],
                'url': item['url'] or '',
                'summary': item['summary'] or '',
                'sentiment': item['sentiment'] or 'neutral',
                'tickers': item['tickers'] or [],
                'type': item['type'] or 'news',
                'impact': item['impact'] or 'medium',
                'published_date': item['published_date'] or ''
            })

        return jsonify(formatted_news)

    except Exception as e:
        return jsonify({"error": f"Failed to get QuantBot news: {e}"}), 500

@app.route('/quantbot/signals')
def get_trading_signals():
    """Get trading signals from QuantBot"""
    try:
        import sys
        from pathlib import Path

        quant_engine_path = Path(__file__).parent / 'QuantEngine'
        if str(quant_engine_path) not in sys.path:
            sys.path.insert(0, str(quant_engine_path))

        from data_broker import QuantBotDataBroker

        broker = QuantBotDataBroker("QuantEngine/quantbot_data.db")
        signals = broker.get_trading_signals(limit=20)

        return jsonify({
            'signals': signals,
            'total_count': len(signals),
            'source': 'QuantBot Database'
        })

    except Exception as e:
        return jsonify({"error": f"Failed to get trading signals: {e}"}), 500

@app.route('/quantbot/status')
def get_quantbot_status():
    """Get QuantBot system status"""
    try:
        import sys
        from pathlib import Path

        quant_engine_path = Path(__file__).parent / 'QuantEngine'
        if str(quant_engine_path) not in sys.path:
            sys.path.insert(0, str(quant_engine_path))

        from data_broker import QuantBotDataBroker

        broker = QuantBotDataBroker("QuantEngine/quantbot_data.db")
        stats = broker.get_database_stats()

        # Check if QuantBot is running (look for recent data)
        opportunities = broker.get_opportunities(limit=1)
        is_active = len(opportunities) > 0

        return jsonify({
            'is_active': is_active,
            'database_stats': stats,
            'last_opportunity': opportunities[0] if opportunities else None,
            'source': 'QuantBot Database'
        })

    except Exception as e:
        return jsonify({"error": f"Failed to get QuantBot status: {e}"}), 500

# Economic Calendar Endpoints
@app.route('/economic-calendar')
def get_economic_calendar():
    """Get economic calendar events for market news app calendar tab"""
    try:
        import sys
        from pathlib import Path

        quant_engine_path = Path(__file__).parent / 'QuantEngine'
        if str(quant_engine_path) not in sys.path:
            sys.path.insert(0, str(quant_engine_path))

        from data_broker import QuantBotDataBroker

        broker = QuantBotDataBroker("QuantEngine/quantbot_data.db")

        # Get query parameters
        impact_filter = request.args.get('impact')  # high, medium, low
        limit = int(request.args.get('limit', 50))

        calendar_events = broker.get_economic_calendar(limit=limit, impact_filter=impact_filter)

        return jsonify({
            'events': calendar_events,
            'total_count': len(calendar_events),
            'impact_filter': impact_filter,
            'source': 'FRED Economic Data'
        })

    except Exception as e:
        return jsonify({"error": f"Failed to get economic calendar: {e}"}), 500

@app.route('/economic-calendar/upcoming')
def get_upcoming_economic_events():
    """Get upcoming economic events within next few days"""
    try:
        import sys
        from pathlib import Path

        quant_engine_path = Path(__file__).parent / 'QuantEngine'
        if str(quant_engine_path) not in sys.path:
            sys.path.insert(0, str(quant_engine_path))

        from data_broker import QuantBotDataBroker

        broker = QuantBotDataBroker("QuantEngine/quantbot_data.db")

        # Get days ahead parameter (default 7 days)
        days_ahead = int(request.args.get('days', 7))

        upcoming_events = broker.get_upcoming_economic_events(days_ahead=days_ahead)

        return jsonify({
            'events': upcoming_events,
            'days_ahead': days_ahead,
            'total_count': len(upcoming_events),
            'source': 'FRED Economic Data'
        })

    except Exception as e:
        return jsonify({"error": f"Failed to get upcoming economic events: {e}"}), 500

@app.route('/economic-calendar/high-impact')
def get_high_impact_events():
    """Get high impact economic events"""
    try:
        import sys
        from pathlib import Path

        quant_engine_path = Path(__file__).parent / 'QuantEngine'
        if str(quant_engine_path) not in sys.path:
            sys.path.insert(0, str(quant_engine_path))

        from data_broker import QuantBotDataBroker

        broker = QuantBotDataBroker("QuantEngine/quantbot_data.db")

        # Get high impact events
        high_impact_events = broker.get_economic_calendar(limit=20, impact_filter='high')

        return jsonify({
            'events': high_impact_events,
            'impact_level': 'high',
            'total_count': len(high_impact_events),
            'source': 'FRED Economic Data'
        })

    except Exception as e:
        return jsonify({"error": f"Failed to get high impact events: {e}"}), 500

# GEX (Gamma Exposure) Endpoints
@app.route('/gex/calculate', methods=['POST', 'GET'])
def calculate_gex():
    """Calculate GEX for a given ticker"""
    try:
        import sys
        from pathlib import Path
        import json

        # Add QuantEngine to path
        quant_engine_path = Path(__file__).parent.parent / 'QuantEngine'
        if str(quant_engine_path) not in sys.path:
            sys.path.insert(0, str(quant_engine_path))

        from gex_calculator import (
            calculate_gex as calc_gex,
            get_option_chain_snapshot,
            get_spot_price
        )

        # Get parameters
        if request.method == 'POST':
            data = request.get_json()
            ticker = data.get('ticker', 'SPY').upper()
            massive_api_key = data.get('massive_api_key', '')
            alphavantage_api_key = data.get('alphavantage_api_key', '')
            spot_price = data.get('spot_price')  # Optional manual override
        else:
            ticker = request.args.get('ticker', 'SPY').upper()
            massive_api_key = request.args.get('massive_api_key', '')
            alphavantage_api_key = request.args.get('alphavantage_api_key', '')
            spot_price = request.args.get('spot_price', type=float)

        # Load from config if keys not provided
        config_path = Path(__file__).parent.parent / 'data' / 'config.json'
        if config_path.exists():
            with open(config_path, 'r') as f:
                config = json.load(f)
            if not massive_api_key:
                massive_api_key = config.get('MASSIVE_API_KEY', '')
            if not alphavantage_api_key:
                alphavantage_api_key = config.get('ALPHAVANTAGE_API_KEY', '')

        if not massive_api_key:
            return jsonify({"error": "Massive API key is required"}), 400

        # Configuration for symmetric strike selection
        # Strategy: Fetch wide range from API, then filter locally to ±20 strikes
        API_FETCH_RANGE = 60        # Points to fetch from API (wider than needed)
        NUM_STRIKES_EACH_SIDE = 20  # ±20 strikes around ATM (41 total)
        STRIKE_INCREMENT = 1.0      # $1 strikes for SPY

        # === STEP 1: Get spot price FIRST (Yahoo/Alpha Vantage) ===
        if not spot_price:
            spot_price = get_spot_price(ticker, massive_api_key, alphavantage_api_key)
            if not spot_price:
                return jsonify({
                    "error": "Could not fetch spot price. Please provide spot_price parameter.",
                    "ticker": ticker
                }), 400

        # === STEP 2: Fetch WIDE range from Massive API ===
        # Get more than we need, then filter locally for symmetric ±20
        snap = get_option_chain_snapshot(
            massive_api_key, 
            ticker, 
            limit=250,
            spot_price=spot_price,
            strike_range=API_FETCH_RANGE
        )

        if not snap.get("results"):
            return jsonify({"error": f"No options data found for {ticker}"}), 404

        # === STEP 3: Calculate GEX with symmetric ±20 strike filter ===
        df, agg, metrics, skipped = calc_gex(
            snap,
            spot_price,
            massive_api_key,
            alphavantage_api_key,
            num_strikes_each_side=NUM_STRIKES_EACH_SIDE,
            strike_increment=STRIKE_INCREMENT
        )

        if df.empty:
            return jsonify({
                "error": "No computable contracts found",
                "ticker": ticker,
                "skipped": skipped
            }), 400

        # Calculate breakdown
        call_gex = float(df[df["type"] == "call"]["gex"].sum())
        put_gex = float(df[df["type"] == "put"]["gex"].sum())

        # Prepare chart annotations for frontend visualization
        chart_annotations = {
            "spot_price": spot_price,
            "flip_line": metrics.get("flip_line"),
            "put_wall": metrics.get("put_wall"),
            "call_wall": metrics.get("call_wall")
        }

        # Get diagnostics from metrics
        diagnostics = metrics.get("diagnostics", {})
        
        # Format response with cumulative_gex included in gex_by_strike
        return jsonify({
            "ticker": ticker,
            "spot_price": spot_price,
            "metrics": metrics,
            "breakdown": {
                "call_gex": call_gex,
                "put_gex": put_gex,
                "total_contracts": len(df),
                "skipped_contracts": skipped,
                "call_contracts": len(df[df["type"] == "call"]),
                "put_contracts": len(df[df["type"] == "put"])
            },
            "gex_by_strike": agg.to_dict('records'),
            "chart_annotations": chart_annotations,
            "diagnostics": {
                "calls_in_target": diagnostics.get("calls_in_target", 0),
                "puts_in_target": diagnostics.get("puts_in_target", 0),
                "is_symmetric": diagnostics.get("is_symmetric", False),
                "target_strikes": NUM_STRIKES_EACH_SIDE * 2 + 1,
                "num_strikes_each_side": NUM_STRIKES_EACH_SIDE,
                "api_fetch_range": API_FETCH_RANGE,
                "strike_range": f"±{NUM_STRIKES_EACH_SIDE} strikes around {spot_price:.0f}",
                "missing_from_api": diagnostics.get("missing_from_api", 0),
                "returned_but_unusable": diagnostics.get("returned_but_unusable", 0)
            },
            "timestamp": time.time()
        })

    except Exception as e:
        import traceback
        return jsonify({
            "error": f"Failed to calculate GEX: {str(e)}",
            "traceback": traceback.format_exc()
        }), 500


@app.route('/gex/tickers')
def get_gex_tickers():
    """Get list of supported GEX tickers from config"""
    try:
        import json
        from pathlib import Path

        config_path = Path(__file__).parent.parent / 'data' / 'config.json'
        if config_path.exists():
            with open(config_path, 'r') as f:
                config = json.load(f)
            tickers = config.get('GEX_TICKERS', ['SPY', 'SPX', 'QQQ'])
        else:
            tickers = ['SPY', 'SPX', 'QQQ', 'IWM', 'NVDA', 'TSLA', 'AAPL', 'MSFT']

        return jsonify({
            "tickers": tickers,
            "default": "SPY"
        })

    except Exception as e:
        return jsonify({"error": f"Failed to get tickers: {e}"}), 500


@app.route('/gex/summary')
def get_gex_summary():
    """
    Get batch GEX summary for all configured tickers.
    Returns quick overview with flip line and regime for each ticker.
    """
    try:
        import sys
        import json
        from pathlib import Path

        # Add QuantEngine to path
        quant_engine_path = Path(__file__).parent.parent / 'QuantEngine'
        if str(quant_engine_path) not in sys.path:
            sys.path.insert(0, str(quant_engine_path))

        from gex_calculator import (
            calculate_gex as calc_gex,
            get_option_chain_snapshot,
            get_spot_price
        )

        # Load config
        config_path = Path(__file__).parent.parent / 'data' / 'config.json'
        config = {}
        if config_path.exists():
            with open(config_path, 'r') as f:
                config = json.load(f)

        massive_api_key = config.get('MASSIVE_API_KEY', '')
        alphavantage_api_key = config.get('ALPHAVANTAGE_API_KEY', '')
        tickers = config.get('GEX_TICKERS', ['SPY', 'SPX', 'QQQ'])

        if not massive_api_key:
            return jsonify({"error": "Massive API key not configured"}), 400

        # Get optional ticker filter from query params
        ticker_filter = request.args.get('tickers')
        if ticker_filter:
            tickers = [t.strip().upper() for t in ticker_filter.split(',')]

        summaries = []
        errors = []

        for ticker in tickers:
            try:
                # Get options snapshot
                snap = get_option_chain_snapshot(massive_api_key, ticker, limit=250)

                if not snap.get("results"):
                    errors.append({"ticker": ticker, "error": "No options data"})
                    continue

                # Get spot price
                spot_price = get_spot_price(ticker, massive_api_key, alphavantage_api_key)
                if not spot_price:
                    errors.append({"ticker": ticker, "error": "Could not fetch spot price"})
                    continue

                # Calculate GEX
                df, agg, metrics, skipped = calc_gex(
                    snap, spot_price, massive_api_key, alphavantage_api_key
                )

                if df.empty:
                    errors.append({"ticker": ticker, "error": "No computable contracts"})
                    continue

                summaries.append({
                    "ticker": ticker,
                    "spot": spot_price,
                    "flip_line": metrics.get("flip_line"),
                    "put_wall": metrics.get("put_wall"),
                    "call_wall": metrics.get("call_wall"),
                    "total_gex": metrics.get("total_gex"),
                    "regime": metrics.get("regime", "unknown")
                })

            except Exception as e:
                errors.append({"ticker": ticker, "error": str(e)})

        return jsonify({
            "tickers": summaries,
            "errors": errors if errors else None,
            "timestamp": time.time()
        })

    except Exception as e:
        import traceback
        return jsonify({
            "error": f"Failed to get GEX summary: {str(e)}",
            "traceback": traceback.format_exc()
        }), 500


@app.route('/gex/price-comparison')
def get_gex_price_comparison():
    """
    Compare spot prices from different data sources for debugging discrepancies.
    Useful for understanding why GEX calculations might differ from other providers.
    """
    try:
        import sys
        import json
        from pathlib import Path

        # Add QuantEngine to path
        quant_engine_path = Path(__file__).parent.parent / 'QuantEngine'
        if str(quant_engine_path) not in sys.path:
            sys.path.insert(0, str(quant_engine_path))

        from gex_calculator import get_spot_price_comparison

        # Get ticker from query params
        ticker = request.args.get('ticker', 'SPY').upper()

        # Load config
        config_path = Path(__file__).parent.parent / 'data' / 'config.json'
        config = {}
        if config_path.exists():
            with open(config_path, 'r') as f:
                config = json.load(f)

        massive_api_key = config.get('MASSIVE_API_KEY', '')
        alphavantage_api_key = config.get('ALPHAVANTAGE_API_KEY', '')

        # Get prices from all sources
        prices = get_spot_price_comparison(ticker, massive_api_key, alphavantage_api_key)

        # Calculate max discrepancy
        valid_prices = [p for p in prices.values() if p is not None]
        if len(valid_prices) >= 2:
            max_price = max(valid_prices)
            min_price = min(valid_prices)
            max_discrepancy_percent = ((max_price - min_price) / min_price) * 100 if min_price > 0 else 0
        else:
            max_discrepancy_percent = 0

        return jsonify({
            "ticker": ticker,
            "prices": prices,
            "max_discrepancy_percent": round(max_discrepancy_percent, 4),
            "timestamp": time.time()
        })

    except Exception as e:
        import traceback
        return jsonify({
            "error": f"Failed to get price comparison: {str(e)}",
            "traceback": traceback.format_exc()
        }), 500


# ===============================
# Intrinsic Value / Valuation Endpoints
# ===============================

@app.route('/valuation/calculate')
def valuation_calculate():
    """
    Calculate intrinsic value using all 6 methods.
    
    Query params:
        ticker: Stock ticker symbol (required)
        store: Whether to store the result (default: true)
    
    Returns:
        JSON with all valuations, composite value, and divergence percentage
    """
    ticker = request.args.get('ticker')
    if not ticker:
        return jsonify({"error": "ticker parameter is required"}), 400
    
    store = request.args.get('store', 'true').lower() == 'true'
    
    try:
        result = calculate_intrinsic_value(ticker.upper())
        
        if 'error' in result and not result.get('valuations'):
            return jsonify(result), 500
        
        # Store in database if requested
        if store and result.get('valuations'):
            store_valuation(result)
            # Check for alerts
            alerts = check_alerts(result)
            if alerts:
                result['alerts'] = alerts
        
        return jsonify(result)
        
    except Exception as e:
        return jsonify({"error": f"Valuation calculation failed: {str(e)}"}), 500


@app.route('/valuation/history')
def valuation_history():
    """
    Get historical intrinsic value data for a ticker.
    
    Query params:
        ticker: Stock ticker symbol (required)
        days: Number of days of history (default: 365)
    
    Returns:
        JSON array of historical valuation records
    """
    ticker = request.args.get('ticker')
    if not ticker:
        return jsonify({"error": "ticker parameter is required"}), 400
    
    days = request.args.get('days', 365, type=int)
    
    try:
        history = get_history(ticker.upper(), days)
        return jsonify({
            "ticker": ticker.upper(),
            "days": days,
            "records": len(history),
            "history": history
        })
        
    except Exception as e:
        return jsonify({"error": f"Failed to get history: {str(e)}"}), 500


@app.route('/valuation/divergence')
def valuation_divergence():
    """
    Get current divergence from intrinsic value with detailed breakdown.
    
    Query params:
        ticker: Stock ticker symbol (required)
    
    Returns:
        JSON with current price, each valuation method, and divergence
    """
    ticker = request.args.get('ticker')
    if not ticker:
        return jsonify({"error": "ticker parameter is required"}), 400
    
    try:
        # Calculate fresh values
        result = calculate_intrinsic_value(ticker.upper())
        
        if 'error' in result and not result.get('valuations'):
            return jsonify(result), 500
        
        # Format response focused on divergence
        valuations = result.get('valuations', {})
        composite = result.get('composite', {})
        
        divergence_breakdown = {}
        for method, data in valuations.items():
            if data.get('applicable') and data.get('value'):
                method_divergence = ((result['current_price'] - data['value']) / data['value']) * 100
                divergence_breakdown[method] = {
                    'intrinsic_value': data['value'],
                    'divergence_pct': round(method_divergence, 2),
                    'confidence': data.get('confidence', 0)
                }
        
        return jsonify({
            "ticker": ticker.upper(),
            "company_name": result.get('company_name'),
            "current_price": result.get('current_price'),
            "composite_intrinsic_value": composite.get('value'),
            "composite_divergence_pct": composite.get('divergence_pct'),
            "verdict": composite.get('verdict'),
            "breakdown": divergence_breakdown,
            "timestamp": result.get('timestamp')
        })
        
    except Exception as e:
        return jsonify({"error": f"Failed to calculate divergence: {str(e)}"}), 500


@app.route('/valuation/scan')
def valuation_scan():
    """
    Batch scan multiple tickers for intrinsic value.
    
    Query params:
        tickers: Comma-separated list of ticker symbols (required)
        threshold: Divergence threshold percentage (default: 20)
    
    Returns:
        JSON array of valuation results sorted by divergence
    """
    tickers_param = request.args.get('tickers')
    if not tickers_param:
        return jsonify({"error": "tickers parameter is required"}), 400
    
    tickers = [t.strip().upper() for t in tickers_param.split(',') if t.strip()]
    threshold = request.args.get('threshold', 20, type=float)
    
    if not tickers:
        return jsonify({"error": "No valid tickers provided"}), 400
    
    try:
        results = batch_calculate(tickers)
        
        # Store results and check alerts
        for result in results:
            if result.get('valuations'):
                store_valuation(result)
                check_alerts(result)
        
        # Filter by threshold and sort by divergence
        filtered = []
        for r in results:
            if r.get('composite', {}).get('divergence_pct') is not None:
                div = r['composite']['divergence_pct']
                if abs(div) >= threshold:
                    filtered.append({
                        'ticker': r['ticker'],
                        'company_name': r.get('company_name'),
                        'current_price': r.get('current_price'),
                        'composite_value': r['composite'].get('value'),
                        'divergence_pct': div,
                        'verdict': r['composite'].get('verdict')
                    })
        
        # Sort by divergence (most undervalued first)
        filtered.sort(key=lambda x: x['divergence_pct'])
        
        return jsonify({
            "tickers_requested": len(tickers),
            "results_with_data": len([r for r in results if r.get('valuations')]),
            "exceeding_threshold": len(filtered),
            "threshold_pct": threshold,
            "results": filtered
        })
        
    except Exception as e:
        return jsonify({"error": f"Batch scan failed: {str(e)}"}), 500


@app.route('/valuation/watchlist', methods=['GET', 'POST', 'DELETE'])
def valuation_watchlist():
    """
    Manage the valuation watchlist.
    
    GET: Get all tickers in watchlist with latest valuations
    POST: Add ticker to watchlist (requires 'ticker' in JSON body)
    DELETE: Remove ticker from watchlist (requires 'ticker' in JSON body)
    """
    if request.method == 'GET':
        try:
            watchlist = get_watchlist()
            return jsonify({
                "count": len(watchlist),
                "watchlist": watchlist
            })
        except Exception as e:
            return jsonify({"error": f"Failed to get watchlist: {str(e)}"}), 500
    
    elif request.method == 'POST':
        data = request.get_json() or {}
        ticker = data.get('ticker')
        if not ticker:
            return jsonify({"error": "ticker is required"}), 400
        
        try:
            # Optionally calculate fresh values
            result = calculate_intrinsic_value(ticker.upper())
            company_name = result.get('company_name')
            sector = result.get('sector')
            
            # Add to watchlist
            alert_threshold = data.get('alert_threshold', 20.0)
            success = add_to_watchlist(ticker.upper(), company_name, sector, alert_threshold)
            
            if success:
                # Store the valuation
                if result.get('valuations'):
                    store_valuation(result)
                
                return jsonify({
                    "success": True,
                    "ticker": ticker.upper(),
                    "company_name": company_name,
                    "message": f"Added {ticker.upper()} to watchlist"
                })
            else:
                return jsonify({"error": "Failed to add to watchlist"}), 500
                
        except Exception as e:
            return jsonify({"error": f"Failed to add to watchlist: {str(e)}"}), 500
    
    elif request.method == 'DELETE':
        data = request.get_json() or {}
        ticker = data.get('ticker')
        if not ticker:
            return jsonify({"error": "ticker is required"}), 400
        
        try:
            success = remove_from_watchlist(ticker.upper())
            if success:
                return jsonify({
                    "success": True,
                    "message": f"Removed {ticker.upper()} from watchlist"
                })
            else:
                return jsonify({"error": "Failed to remove from watchlist"}), 500
                
        except Exception as e:
            return jsonify({"error": f"Failed to remove from watchlist: {str(e)}"}), 500


@app.route('/valuation/alerts')
def valuation_alerts():
    """
    Get recent valuation alerts.
    
    Query params:
        limit: Maximum number of alerts to return (default: 50)
    
    Returns:
        JSON array of recent alerts
    """
    limit = request.args.get('limit', 50, type=int)
    
    try:
        alerts = get_recent_alerts(limit)
        return jsonify({
            "count": len(alerts),
            "alerts": alerts
        })
        
    except Exception as e:
        return jsonify({"error": f"Failed to get alerts: {str(e)}"}), 500


@app.route('/valuation/alerts/<int:alert_id>/acknowledge', methods=['POST'])
def acknowledge_valuation_alert(alert_id):
    """Acknowledge a valuation alert"""
    try:
        success = acknowledge_alert(alert_id)
        if success:
            return jsonify({"success": True, "message": f"Alert {alert_id} acknowledged"})
        else:
            return jsonify({"error": "Failed to acknowledge alert"}), 500
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/valuation/undervalued')
def valuation_undervalued():
    """
    Get stocks that are significantly undervalued.
    
    Query params:
        threshold: Divergence threshold percentage (default: 20)
    
    Returns:
        JSON array of undervalued stocks from database
    """
    threshold = request.args.get('threshold', 20, type=float)
    
    try:
        results = get_divergence_scan(threshold)
        # Filter for undervalued only (negative divergence)
        undervalued = [r for r in results if r.get('divergence_pct', 0) < 0]
        
        return jsonify({
            "threshold_pct": threshold,
            "count": len(undervalued),
            "stocks": undervalued
        })
        
    except Exception as e:
        return jsonify({"error": f"Failed to scan: {str(e)}"}), 500


# ===============================
# Decision Cockpit Endpoints
# ===============================

@app.route('/cockpit/state')
def cockpit_state():
    """
    Get the complete Decision Cockpit state.
    
    This is a single-screen trading state view with:
    - REGIME: GEX state (positive/negative gamma) + transition flag
    - VOLATILITY: IV direction and state
    - STRUCTURE: Multi-lens walls (Today/Tactical/Regime) and no-trade zones
    - ACTION FILTER: Allowed/forbidden actions based on regime + transition
    
    Enhanced with:
    - OPEX-aware expiry weighting
    - Dealer-centric GEX calculation
    - Multi-lens analysis (0-2, 0-14, 0-60 DTE)
    - Transition detection (near flip, flip moving)
    
    Query params:
        ticker: Stock ticker (SPY, QQQ, IWM) - default: SPY
    
    Returns:
        JSON with regime, volatility, structure, action_filter, and net_series
    """
    ticker = request.args.get('ticker', 'SPY').upper()
    
    # Validate ticker
    supported_tickers = ['SPY', 'QQQ', 'IWM']
    if ticker not in supported_tickers:
        return jsonify({
            "error": f"Unsupported ticker. Use one of: {', '.join(supported_tickers)}"
        }), 400
    
    try:
        # Import GEX calculator with new functions
        from QuantEngine.gex_calculator import (
            get_option_chain_snapshot,
            get_spot_price,
            compute_cockpit_state as compute_gex_state
        )
        import json
        
        # Load config for API keys
        config_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data', 'config.json')
        config = {}
        if os.path.exists(config_path):
            with open(config_path, 'r') as f:
                config = json.load(f)
        
        massive_key = config.get('MASSIVE_API_KEY', '')
        alphavantage_key = config.get('ALPHAVANTAGE_API_KEY', '')
        
        # Get GEX state using new multi-lens computation
        # Strategy: Fetch wide range from API, then filter locally to ±20 strikes
        gex_state = {}
        snap = None  # Initialize snap for de-pin risk computation
        spot = None  # Initialize spot for de-pin risk computation
        try:
            # === STEP 1: Get spot price first (Yahoo/Alpha Vantage) ===
            spot = get_spot_price(ticker, massive_key, alphavantage_key)
            if spot:
                # === STEP 2: Fetch WIDE range from Massive API ===
                # Fetch ±60 points to ensure we have enough data
                # Then we'll filter locally to exactly ±20 strikes
                api_fetch_range = 60  # Points to request from API
                num_strikes_each_side = 20  # Strikes to keep each side of ATM
                strike_increment = 1.0  # $1 strikes for SPY
                
                snap = get_option_chain_snapshot(
                    massive_key, 
                    ticker, 
                    days_out=65,
                    spot_price=spot,
                    strike_range=api_fetch_range
                )
                
                if snap and snap.get('results'):
                    # === STEP 3: Process with symmetric ±20 strike filter ===
                    # compute_cockpit_state will use process_chain_symmetric
                    # which filters to ONLY the ±20 strikes around ATM
                    gex_state = compute_gex_state(
                        snap, spot, 
                        ticker=ticker,
                        num_strikes_each_side=num_strikes_each_side,
                        strike_increment=strike_increment
                    )
                    
                    # Add metadata about the query
                    gex_state['strike_range'] = {
                        'api_fetch_range': api_fetch_range,
                        'processed_strikes_each_side': num_strikes_each_side,
                        'total_target_strikes': num_strikes_each_side * 2 + 1,
                        'center': spot
                    }
        except Exception as e:
            print(f"[cockpit] GEX calculation failed: {e}")
            import traceback
            traceback.print_exc()
        
        # Get volatility data
        volatility_data = {}
        try:
            volatility_data = analyze_volatility(ticker)
        except Exception as e:
            print(f"[cockpit] Volatility analysis failed: {e}")
        
        # Build cockpit state with new multi-lens data
        cockpit = DecisionCockpit(gex_state, volatility_data)
        state = cockpit.get_state(ticker)
        
        # Compute de-pin risk and add to state
        depin_risk_data = None
        print(f"[depin] Starting de-pin risk computation for {ticker}, spot={spot}, snap={snap is not None}")
        try:
            from QuantEngine.depin_risk import (
                fetch_5m_bars,
                convert_options_to_contracts,
                compute_depin_risk
            )
            from QuantEngine.depin_risk_database import (
                load_state,
                save_state,
                save_risk_result,
                get_risk_30m_ago
            )
            
            # Fetch latest 5m bar
            bars = fetch_5m_bars(ticker, period='5d')
            print(f"[depin] Fetched {len(bars) if bars else 0} bars, spot={spot}")
            if bars and spot:
                latest_bar = bars[-1]
                print(f"[depin] Latest bar: {latest_bar.ts}, close: {latest_bar.close}")
                
                # Fetch options snapshot (DTE <= 2) - reuse the same snapshot if available
                if snap and snap.get('results'):
                    print(f"[depin] Options snapshot has {len(snap['results'])} results")
                    options = convert_options_to_contracts(snap['results'], spot)
                    print(f"[depin] Converted to {len(options)} contracts")
                    options = [opt for opt in options if opt.dte <= 2]
                    print(f"[depin] After DTE filter: {len(options)} contracts")
                    
                    if options:
                        # Load rolling state
                        depin_state = load_state(ticker)
                        
                        # Compute de-pin risk
                        result = compute_depin_risk(
                            symbol=ticker,
                            bar=latest_bar,
                            options_snapshot=options,
                            state=depin_state,
                            bucket_dte_max=2,
                            strike_window_pct=0.01,
                            strike_window_floor=5.0,
                            vol_ref_30m=None
                        )
                        
                        # Save updated state and result
                        save_state(ticker, depin_state)
                        save_risk_result(result)
                        
                        # Get previous risk for delta calculation
                        prev_risk = get_risk_30m_ago(ticker)
                        delta_30m = None
                        delta_direction = None
                        if prev_risk is not None:
                            delta_30m = result.de_pin_risk - prev_risk
                            delta_direction = "up" if delta_30m > 0 else "down" if delta_30m < 0 else "stable"
                        
                        # Extract top 3 drivers (sorted by absolute contribution to x_raw)
                        # The drivers are: gex_collapse, trend_strength, move30, (1-pin_persist), liq_fade, wall_drift
                        drivers = [
                            {"name": "GEX Collapse", "contribution": result.gex_collapse * 1.40},
                            {"name": "Trend Strength", "contribution": result.trend_strength * 0.90},
                            {"name": "Move30", "contribution": result.move30 * 0.70},
                            {"name": "Pin Persist (inv)", "contribution": (1.0 - result.pin_persist) * 0.80},
                            {"name": "Liquidity Fade", "contribution": result.liq_fade * 0.50},
                            {"name": "Wall Drift", "contribution": result.wall_drift * 0.50},
                        ]
                        # Sort by absolute contribution and take top 3
                        drivers.sort(key=lambda x: abs(x["contribution"]), reverse=True)
                        top_3_drivers = drivers[:3]
                        
                        depin_risk_data = {
                            "score": result.de_pin_risk,
                            "band": result.band,
                            "delta_30m": delta_30m,
                            "delta_direction": delta_direction,
                            "guidance": result.guidance,
                            "drivers": top_3_drivers
                        }
        except Exception as e:
            print(f"[cockpit] De-pin risk calculation failed: {e}")
            import traceback
            traceback.print_exc()
            # Add error to response for debugging
            state["depin_risk_error"] = str(e)
        
        # Add de-pin risk to state if computed
        if depin_risk_data:
            state["de_pin_risk"] = depin_risk_data
        
        return jsonify(state)
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({
            "error": f"Failed to get cockpit state: {str(e)}",
            "ticker": ticker
        }), 500


@app.route('/cockpit/tickers')
def cockpit_tickers():
    """Get list of supported tickers for the cockpit"""
    return jsonify({
        "tickers": ['SPY', 'QQQ', 'IWM'],
        "default": "SPY"
    })


@app.route('/trade-ideas/allowed')
def trade_ideas_allowed():
    """
    Get allowed trade ideas based on current regime.
    
    Only returns ideas that are permitted by the current market regime,
    ranked by risk-adjusted ROI using GEX walls and options chain.
    
    Query params:
        ticker: Stock ticker (SPY, QQQ, IWM) - default: SPY
        max_ideas: Maximum number of ideas to return per timeframe (default: 3)
        timeframe: Timeframe selection - 'thisWeek', 'thisMonth', 'thisYear', or 'all' (default: 'all')
        min_dte: Optional override for minimum DTE (overrides timeframe)
        max_dte: Optional override for maximum DTE (overrides timeframe)
    
    Returns:
        JSON object with ideas grouped by timeframe (or single array if timeframe specified)
    """
    ticker = request.args.get('ticker', 'SPY').upper()
    max_ideas = int(request.args.get('max_ideas', 3))
    timeframe = request.args.get('timeframe', 'all')  # 'all', 'thisWeek', 'thisMonth', 'thisYear'
    min_dte_override = request.args.get('min_dte', type=int)
    max_dte_override = request.args.get('max_dte', type=int)
    
    # Validate ticker
    supported_tickers = ['SPY', 'QQQ', 'IWM']
    if ticker not in supported_tickers:
        return jsonify({
            "error": f"Unsupported ticker. Use one of: {', '.join(supported_tickers)}"
        }), 400
    
    try:
        # Import required modules
        from QuantEngine.trade_ideas_engine import (
            MarketContext,
            enhance_contract_with_pricing,
            generate_trade_ideas,
            load_trade_ideas_config
        )
        from QuantEngine.gex_calculator import (
            get_option_chain_snapshot,
            get_spot_price,
            parse_option_chain
        )
        from QuantEngine.decision_cockpit import DecisionCockpit
        from QuantEngine.volatility_analyzer import analyze_volatility
        import json
        
        # Load config for API keys
        config_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data', 'config.json')
        config = {}
        if os.path.exists(config_path):
            with open(config_path, 'r') as f:
                config = json.load(f)
        
        massive_key = config.get('MASSIVE_API_KEY', '')
        alphavantage_key = config.get('ALPHAVANTAGE_API_KEY', '')
        
        # Get spot price
        spot = get_spot_price(ticker, massive_key, alphavantage_key)
        if not spot:
            return jsonify({
                "error": "Failed to get spot price"
            }), 500
        
        # Fetch options chain (wide range for trade ideas)
        snap = get_option_chain_snapshot(
            massive_key,
            ticker,
            days_out=65,
            spot_price=spot,
            strike_range=60  # Wide range for trade ideas
        )
        
        if not snap or not snap.get('results'):
            return jsonify({
                "error": "Failed to fetch options chain"
            }), 500
        
        # Parse contracts
        contracts = parse_option_chain(snap, spot)
        
        # Enhance contracts with pricing
        quotes = []
        api_results = snap.get('results', [])
        api_index = {}  # Build index by (expiry, strike, type)
        
        for item in api_results:
            details = item.get('details', {})
            exp_str = details.get('expiration_date')
            strike = details.get('strike_price')
            contract_type = details.get('contract_type')
            if exp_str and strike and contract_type:
                key = (exp_str, float(strike), contract_type)
                api_index[key] = item
        
        for contract in contracts:
            key = (contract['expiry_str'], contract['strike'], contract['type'])
            api_item = api_index.get(key)
            if api_item:
                quote = enhance_contract_with_pricing(contract, spot, api_item)
                if quote:
                    quotes.append(quote)
        
        if not quotes:
            return jsonify({
                "error": "No valid option quotes found"
            }), 500
        
        # Get cockpit state for context (reuse the /cockpit/state endpoint logic to get de-pin risk)
        from QuantEngine.gex_calculator import compute_cockpit_state as compute_gex_state
        gex_state = compute_gex_state(snap, spot, ticker=ticker, num_strikes_each_side=20, strike_increment=1.0)
        volatility_data = analyze_volatility(ticker)
        cockpit = DecisionCockpit(gex_state, volatility_data)
        cockpit_state = cockpit.get_state(ticker)
        
        # Compute de-pin risk (same logic as /cockpit/state endpoint)
        depin_risk_data = None
        try:
            from QuantEngine.depin_risk import (
                fetch_5m_bars,
                convert_options_to_contracts,
                compute_depin_risk
            )
            from QuantEngine.depin_risk_database import (
                load_state,
                save_state,
                save_risk_result,
                get_risk_30m_ago
            )
            
            # Fetch latest 5m bar
            bars = fetch_5m_bars(ticker, period='5d')
            if bars and spot and snap and snap.get('results'):
                latest_bar = bars[-1]
                
                # Convert options snapshot (DTE <= 2)
                options = convert_options_to_contracts(snap['results'], spot)
                options = [opt for opt in options if opt.dte <= 2]
                
                if options:
                    # Load rolling state
                    depin_state = load_state(ticker)
                    
                    # Compute de-pin risk
                    result = compute_depin_risk(
                        symbol=ticker,
                        bar=latest_bar,
                        options_snapshot=options,
                        state=depin_state,
                        bucket_dte_max=2,
                        strike_window_pct=0.01,
                        strike_window_floor=5.0,
                        vol_ref_30m=None
                    )
                    
                    # Save updated state and result
                    save_state(ticker, depin_state)
                    save_risk_result(result)
                    
                    # Get previous risk for delta calculation
                    prev_risk = get_risk_30m_ago(ticker)
                    delta_30m = None
                    delta_direction = None
                    if prev_risk is not None:
                        delta_30m = result.de_pin_risk - prev_risk
                        delta_direction = "up" if delta_30m > 0 else "down" if delta_30m < 0 else "stable"
                    
                    depin_risk_data = {
                        "score": result.de_pin_risk,
                        "band": result.band,
                        "delta_30m": delta_30m,
                        "delta_direction": delta_direction,
                        "guidance": result.guidance,
                    }
        except Exception as e:
            print(f"[trade-ideas] De-pin risk calculation failed: {e}")
            # Don't fail the whole request, just log the error
        
        # Add de-pin risk to cockpit state if computed
        if depin_risk_data:
            cockpit_state["de_pin_risk"] = depin_risk_data
        
        # Extract events for earnings/macro detection
        events_data = {}
        try:
            from datetime import datetime, timedelta
            today = datetime.now().date()
            end_date = today + timedelta(days=1)
            
            # Check for earnings (reuse cockpit events logic)
            spy_top_holdings = ['AAPL', 'MSFT', 'NVDA', 'AMZN', 'GOOGL', 'GOOG', 'META', 'TSLA']
            earnings_count = 0
            mega_cap_earnings = False
            
            # Try yfinance for earnings
            try:
                import yfinance as yf
                for holding in spy_top_holdings[:10]:
                    try:
                        stock = yf.Ticker(holding)
                        calendar = stock.calendar
                        if calendar and isinstance(calendar, dict) and 'Earnings Date' in calendar:
                            earnings_dates = calendar['Earnings Date']
                            if isinstance(earnings_dates, list) and len(earnings_dates) > 0:
                                next_earnings = earnings_dates[0]
                                if isinstance(next_earnings, datetime):
                                    report_date = next_earnings.date()
                                elif isinstance(next_earnings, date):
                                    report_date = next_earnings
                                elif isinstance(next_earnings, str):
                                    report_date = datetime.strptime(next_earnings.split()[0], '%Y-%m-%d').date()
                                else:
                                    continue
                                
                                if today <= report_date <= end_date:
                                    earnings_count += 1
                                    if holding in ['AAPL', 'MSFT', 'GOOGL', 'META', 'AMZN']:
                                        mega_cap_earnings = True
                    except:
                        continue
            except:
                pass
            
            events_data = {
                'earningsClusterScore': earnings_count,
                'megaCapEarningsNext24h': mega_cap_earnings,
                'macroEventNext24h': False  # Could enhance with economic calendar
            }
        except Exception as e:
            print(f"[trade-ideas] Error fetching events: {e}")
            events_data = {
                'earningsClusterScore': 0,
                'megaCapEarningsNext24h': False,
                'macroEventNext24h': False
            }
        
        # Build MarketContext from cockpit state
        regime_data = cockpit_state.get('regime', {})
        structure_data = cockpit_state.get('structure', {})
        volatility_data_state = cockpit_state.get('volatility', {})
        action_filter_data = cockpit_state.get('action_filter', {})
        depin_risk_data = cockpit_state.get('de_pin_risk')
        
        # Map volatility state to regime
        vol_state = volatility_data_state.get('state', 'NORMAL')
        volatility_regime_map = {
            'COMPRESSED': 'LOW',
            'NORMAL': 'NORMAL',
            'ELEVATED': 'ELEVATED',
            'EXPANDING': 'EXTREME'
        }
        volatility_regime = volatility_regime_map.get(vol_state, 'NORMAL')
        
        # Map trend regime from spot vs flip
        # Flip line is in regime section, not structure
        flip_line = regime_data.get('flip_line')
        if flip_line and spot:
            if spot > flip_line * 1.02:
                trend_regime = 'UP'
            elif spot < flip_line * 0.98:
                trend_regime = 'DOWN'
            else:
                trend_regime = 'RANGE'
        else:
            trend_regime = 'RANGE'
        
        # Get pin confidence from depin risk
        pin_confidence = 'LOW'
        if depin_risk_data:
            depin_band = depin_risk_data.get('band', 'MID')
            if depin_band == 'LOW':
                pin_confidence = 'HIGH'
            elif depin_band == 'MID':
                pin_confidence = 'MEDIUM'
        
        # Get action filter
        action_filter = action_filter_data.get('status', 'REAL_OK')
        if action_filter == 'WAIT':
            action_filter = 'WAIT_FOR_CLARITY'
        elif action_filter == 'BLOCKED':
            action_filter = 'BLOCKED'
        elif action_filter == 'PAPER_ONLY':
            action_filter = 'PAPER_ONLY'
        else:
            action_filter = 'REAL_OK'
        
        # Map GEX state from regime data
        # Use is_positive if available, otherwise fall back to label
        is_positive = regime_data.get('is_positive')
        if is_positive is not None:
            gex_state = 'POSITIVE' if is_positive else 'NEGATIVE'
        else:
            # Fall back to label parsing
            regime_label = regime_data.get('label', 'UNKNOWN')
            if 'POSITIVE' in regime_label.upper():
                gex_state = 'POSITIVE'
            elif 'NEGATIVE' in regime_label.upper():
                gex_state = 'NEGATIVE'
            else:
                gex_state = 'NEUTRAL'
        
        context = MarketContext(
            symbol=ticker,
            asOf=datetime.now().isoformat(),
            spot=spot,
            volatilityRegime=volatility_regime,
            trendRegime=trend_regime,
            liquidityState='NORMAL',
            putWall=structure_data.get('put_wall'),
            callWall=structure_data.get('call_wall'),
            flipLine=flip_line,
            rangePts=structure_data.get('range_pts'),
            depinRisk=depin_risk_data.get('score') if depin_risk_data else None,
            pinConfidence=pin_confidence,
            gexState=gex_state,
            earningsClusterScore=events_data.get('earningsClusterScore', 0),
            macroEventNext24h=events_data.get('macroEventNext24h', False),
            megaCapEarningsNext24h=events_data.get('megaCapEarningsNext24h', False),
            actionFilter=action_filter
        )
        
        # Load trade ideas config
        trade_config = load_trade_ideas_config(ticker)
        trade_config['maxIdeasToShow'] = max_ideas
        
        # Handle timeframe selection and DTE overrides
        if min_dte_override is not None or max_dte_override is not None:
            # User-specified DTE range overrides everything
            trade_config['expiryCandidates'] = [{
                "minDTE": min_dte_override if min_dte_override is not None else 0,
                "maxDTE": max_dte_override if max_dte_override is not None else 365
            }]
            trade_config['earningsWeekExpiryCandidates'] = trade_config['expiryCandidates']
            timeframes_to_generate = ['custom']
        elif timeframe == 'all':
            # Generate for all timeframes
            timeframes_to_generate = ['thisWeek', 'thisMonth', 'thisYear']
        else:
            # Single timeframe
            timeframes_to_generate = [timeframe]
        
        # Check for blocking reasons BEFORE generating ideas (use base config)
        from QuantEngine.trade_ideas_engine import check_global_blocks, determine_mode, filter_expiry_candidates
        base_config = load_trade_ideas_config(ticker)  # Use base config for blocking check
        is_blocked, block_reason = check_global_blocks(context, base_config)
        
        # Import helper functions for diagnostics
        from QuantEngine.trade_ideas_engine import (
            is_spot_above_flip_line,
            is_depin_risk_confirmed,
            is_depin_risk_elevated,
            get_allowed_strategies
        )
        
        # Calculate flip line position and DePin status
        spot_above_flip = is_spot_above_flip_line(context, base_config)
        depin_elevated = is_depin_risk_elevated(context, base_config)
        depin_confirmed = is_depin_risk_confirmed(context, base_config)
        allowed_strategies_list = get_allowed_strategies(context, base_config)
        
        # Determine strategy filter reason if strategies were filtered
        strategy_filter_reason = None
        if not allowed_strategies_list:
            if spot_above_flip is False:
                strategy_filter_reason = "Spot below flip line - only hedged structures allowed (not yet implemented)"
            elif context.gexState == "NEGATIVE":
                strategy_filter_reason = "GEX negative - time-decay strategies blocked"
            elif depin_confirmed:
                strategy_filter_reason = f"DePin risk confirmed ({context.depinRisk}) - all strategies blocked"
            elif spot_above_flip is None:
                strategy_filter_reason = "Flip line unavailable - conservative blocking"
        
        # Collect diagnostic information
        diagnostics = {
            'context': {
                'symbol': context.symbol,
                'spot': context.spot,
                'gexState': context.gexState,
                'actionFilter': context.actionFilter,
                'depinRisk': context.depinRisk,
                'pinConfidence': context.pinConfidence,
                'putWall': context.putWall,
                'callWall': context.callWall,
                'flipLine': context.flipLine,
                'volatilityRegime': context.volatilityRegime,
                'trendRegime': context.trendRegime,
                'earningsClusterScore': context.earningsClusterScore,
                'macroEventNext24h': context.macroEventNext24h,
                'megaCapEarningsNext24h': context.megaCapEarningsNext24h,
            },
            'flipLineAnalysis': {
                'spotAboveFlip': spot_above_flip,
                'flipLine': context.flipLine,
                'spot': context.spot,
            },
            'depinRiskAnalysis': {
                'depinRisk': context.depinRisk,
                'depinRiskElevated': depin_elevated,
                'depinRiskConfirmed': depin_confirmed,
            },
            'strategyFiltering': {
                'allowedStrategies': allowed_strategies_list,
                'strategyFilterReason': strategy_filter_reason,
            },
            'blocked': is_blocked,
            'blockReason': block_reason,
            'quotesCount': len(quotes),
            'config': {
                'strategies': trade_config.get('baseStrategySet', []),
                'minCredit': trade_config.get('riskEnvelope', {}).get('minCredit', 0.20),
                'maxLoss': trade_config.get('riskEnvelope', {}).get('maxLossPerIdea', 1000.0),
                'minOI': trade_config.get('minOI', 100),
                'minVolume': trade_config.get('minVolume', 10),
            }
        }
        
        # Generate trade ideas for each timeframe
        all_ideas_by_timeframe = {}
        
        if is_blocked:
            # If blocked, return empty for all timeframes
            for tf in timeframes_to_generate:
                all_ideas_by_timeframe[tf] = []
        else:
            # Generate ideas for each timeframe
            original_config = trade_config.copy()
            
            for tf in timeframes_to_generate:
                # Set expiry candidates for this timeframe
                if tf == 'custom':
                    # Already set above with DTE overrides
                    pass
                elif tf in ['thisWeek', 'thisMonth', 'thisYear']:
                    tf_config = original_config.get('timeframes', {}).get(tf, {})
                    if tf_config:
                        trade_config['expiryCandidates'] = tf_config.get('expiryCandidates', [{"minDTE": 7, "maxDTE": 10}])
                        trade_config['earningsWeekExpiryCandidates'] = tf_config.get('earningsWeekExpiryCandidates', trade_config['expiryCandidates'])
                    else:
                        # Fallback to default if timeframe config not found
                        trade_config['expiryCandidates'] = [{"minDTE": 7, "maxDTE": 10}]
                        trade_config['earningsWeekExpiryCandidates'] = [{"minDTE": 5, "maxDTE": 7}]
                
                # Generate ideas for this timeframe
                ideas = generate_trade_ideas(context, quotes, trade_config)
                all_ideas_by_timeframe[tf] = ideas
                
                # Restore original config for next iteration
                trade_config = original_config.copy()
        
        # Collect diagnostic info (use first timeframe's diagnostics)
        ideas_for_diagnostics = all_ideas_by_timeframe.get(timeframes_to_generate[0] if timeframes_to_generate else 'thisMonth', [])
        
        # If blocked or no ideas, add diagnostic info
        if is_blocked or len(ideas_for_diagnostics) == 0:
            # Check additional reasons why no ideas might be generated
            mode = determine_mode(context, base_config)
            # Use first timeframe's config for diagnostics
            diag_config = trade_config.copy()
            if timeframes_to_generate and timeframes_to_generate[0] in ['thisWeek', 'thisMonth', 'thisYear']:
                tf_config = diag_config.get('timeframes', {}).get(timeframes_to_generate[0], {})
                if tf_config:
                    diag_config['expiryCandidates'] = tf_config.get('expiryCandidates', [{"minDTE": 7, "maxDTE": 10}])
            filtered_quotes = filter_expiry_candidates(quotes, context, diag_config)
            
            diagnostics['mode'] = mode
            diagnostics['filteredQuotesCount'] = len(filtered_quotes)
            diagnostics['totalQuotesCount'] = len(quotes)
            
            # Get available DTE values for diagnostics
            if quotes:
                available_dtes = sorted(set(q.dte for q in quotes if q.dte >= 0))
                diagnostics['availableDTEs'] = available_dtes[:10]  # First 10 for display
            
            # Check if we have valid quotes after filtering
            if len(filtered_quotes) == 0:
                from datetime import datetime
                day_of_week = datetime.now().weekday()
                day_name = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'][day_of_week]
                
                reasons = [
                    f'No options quotes match expiry criteria (Today: {day_name})',
                ]
                
                if day_of_week >= 4:
                    if day_of_week == 4:
                        reasons.append('Friday: Looking for 0DTE or next week (1-5 DTE)')
                    else:
                        reasons.append('Weekend: Looking for next week options (1-5 DTE)')
                else:
                    base_ranges = diag_config.get("expiryCandidates", [{"minDTE": 7, "maxDTE": 10}])
                    reasons.append(f'Required expiries: {base_ranges}')
                
                if quotes:
                    reasons.append(f'Available DTE values: {available_dtes[:10]}')
                
                diagnostics['additionalReasons'] = reasons
            elif not is_blocked:
                # Not blocked but no ideas - check candidate generation
                diagnostics['additionalReasons'] = [
                    'No valid candidates found after applying filters',
                    f'Check: delta targets, wall buffers, liquidity requirements',
                ]
        
        # Convert to JSON-serializable format
        from QuantEngine.trade_ideas_engine import TradeIdea
        def idea_to_dict(idea: TradeIdea) -> dict:
            return {
                'id': idea.id,
                'symbol': idea.symbol,
                'asOf': idea.asOf,
                'strategy': idea.strategy,
                'label': idea.label,
                'allowed': idea.allowed,
                'mode': idea.mode,
                'confidence': idea.confidence,
                'reasons': idea.reasons,
                'preMarket': idea.preMarket,
                'marketStatusNote': idea.marketStatusNote,
                'executions': [
                    {
                        'expiry': exec.expiry,
                        'dte': exec.dte,
                        'legs': [
                            {
                                'action': leg.action,
                                'type': leg.type,
                                'strike': leg.strike,
                                'priceMid': leg.priceMid,
                                'bid': leg.bid,
                                'ask': leg.ask,
                                'delta': leg.delta,
                                'oi': leg.oi,
                                'volume': leg.volume
                            }
                            for leg in exec.legs
                        ],
                        'metrics': {
                            'netCredit': exec.netCredit,
                            'width': exec.width,
                            'maxProfit': exec.maxProfit,
                            'maxLoss': exec.maxLoss,
                            'roc': exec.roc,
                            'breakeven': exec.breakeven,
                            'pop': exec.pop,
                            'ev': exec.ev,
                            'wallBufferPct': exec.wallBufferPct,
                            'liquidityScore': exec.liquidityScore,
                            'score': exec.score
                        }
                    }
                    for exec in idea.executions
                ],
                'best': {
                    'expiry': idea.best.expiry,
                    'dte': idea.best.dte,
                    'netCredit': idea.best.netCredit,
                    'maxProfit': idea.best.maxProfit,
                    'maxLoss': idea.best.maxLoss,
                    'roc': idea.best.roc,
                    'pop': idea.best.pop,
                    'ev': idea.best.ev,
                    'wallBufferPct': idea.best.wallBufferPct,
                    'liquidityGrade': idea.best.liquidityGrade,
                    'score': idea.best.score
                } if idea.best else None,
                'contextSnapshot': {
                    'spot': idea.contextSnapshot.spot,
                    'putWall': idea.contextSnapshot.putWall,
                    'callWall': idea.contextSnapshot.callWall,
                    'flipLine': idea.contextSnapshot.flipLine,
                    'depinRisk': idea.contextSnapshot.depinRisk,
                    'gexState': idea.contextSnapshot.gexState,
                    'pinConfidence': idea.contextSnapshot.pinConfidence,
                    'earningsClusterScore': idea.contextSnapshot.earningsClusterScore,
                    'macroEventNext24h': idea.contextSnapshot.macroEventNext24h
                } if idea.contextSnapshot else None
            }
        
        # Return ideas with diagnostics
        if timeframe == 'all':
            # Return grouped by timeframe
            result = {
                'ideasByTimeframe': {
                    tf: [idea_to_dict(idea) for idea in ideas_list]
                    for tf, ideas_list in all_ideas_by_timeframe.items()
                },
                'diagnostics': diagnostics,
            }
        else:
            # Return single array for specific timeframe
            tf_ideas = all_ideas_by_timeframe.get(timeframe, [])
            result = {
                'ideas': [idea_to_dict(idea) for idea in tf_ideas],
                'diagnostics': diagnostics,
            }
        
        return jsonify(result)
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({
            "error": f"Failed to generate trade ideas: {str(e)}",
            "ideas": [],
            "diagnostics": {
                "blocked": True,
                "blockReason": f"Error: {str(e)}"
            }
        }), 500


@app.route('/cockpit/events')
def cockpit_events():
    """
    Get upcoming high-impact market events for cockpit display.
    
    Uses Alpha Vantage for earnings calendar and generates OPEX dates.
    
    Query params:
        days: Number of days ahead to look (default: 7)
    
    Returns:
        JSON with badges (compact for header) and full_calendar (expandable section)
    """
    from datetime import datetime, timedelta, date
    import requests
    
    days_ahead = int(request.args.get('days', 7))
    
    try:
        events = []
        today = datetime.now().date()
        end_date = today + timedelta(days=days_ahead)
        
        # SPY top holdings for earnings filter
        spy_top_holdings = ['AAPL', 'MSFT', 'NVDA', 'AMZN', 'GOOGL', 'GOOG', 'META', 'TSLA', 'BRK.B', 'JPM', 
                           'V', 'MA', 'UNH', 'HD', 'PG', 'JNJ', 'XOM', 'CVX', 'ABBV', 'MRK', 'PFE', 'KO', 'PEP',
                           'COST', 'WMT', 'BAC', 'WFC', 'NFLX', 'ADBE', 'CRM', 'ORCL', 'INTC', 'AMD', 'QCOM']
        
        # 1. Load config
        config_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data', 'config.json')
        config = {}
        if os.path.exists(config_path):
            import json
            with open(config_path, 'r') as f:
                config = json.load(f)
        
        # Try FMP first (more reliable for earnings)
        fmp_key = config.get('FMP_API_KEY', '')
        if fmp_key:
            try:
                earnings_url = f"https://financialmodelingprep.com/api/v3/earning_calendar?from={today}&to={end_date}&apikey={fmp_key}"
                resp = requests.get(earnings_url, timeout=10)
                if resp.status_code == 200:
                    earnings_data = resp.json()
                    for earning in earnings_data:
                        ticker = earning.get('symbol', '')
                        if ticker in spy_top_holdings:
                            event_date = earning.get('date', '')
                            event_time = earning.get('time', '')
                            time_label = 'BMO' if event_time == 'bmo' else ('AH' if event_time == 'amc' else '')
                            events.append({
                                'type': 'earnings',
                                'title': f'{ticker} Earnings',
                                'date': event_date,
                                'time': time_label,
                                'impact': 'high',
                                'ticker': ticker,
                                'source': 'fmp'
                            })
                    print(f"[cockpit/events] Loaded {len([e for e in events if e['type']=='earnings'])} earnings from FMP")
            except Exception as e:
                print(f"[cockpit/events] FMP earnings error: {e}")
        
        # Fallback to yfinance for earnings if no FMP key or no earnings found
        if not [e for e in events if e['type'] == 'earnings']:
            try:
                import yfinance as yf
                print(f"[cockpit/events] Trying yfinance for earnings...")
                earnings_count = 0
                for ticker in spy_top_holdings[:15]:  # Limit to top 15 to avoid rate limits
                    try:
                        stock = yf.Ticker(ticker)
                        calendar = stock.calendar
                        if calendar and isinstance(calendar, dict) and 'Earnings Date' in calendar:
                            earnings_dates = calendar['Earnings Date']
                            if isinstance(earnings_dates, list) and len(earnings_dates) > 0:
                                # Get next earnings date (first in list)
                                next_earnings = earnings_dates[0]
                                if isinstance(next_earnings, datetime):
                                    report_date = next_earnings.date()
                                elif isinstance(next_earnings, date):
                                    report_date = next_earnings
                                elif isinstance(next_earnings, str):
                                    report_date = datetime.strptime(next_earnings.split()[0], '%Y-%m-%d').date()
                                else:
                                    continue
                                
                                if today <= report_date <= end_date:
                                    # Check if we already have this earnings event
                                    if not any(e.get('ticker') == ticker and e.get('date') == report_date.isoformat() for e in events):
                                        events.append({
                                            'type': 'earnings',
                                            'title': f'{ticker} Earnings',
                                            'date': report_date.isoformat(),
                                            'time': '',
                                            'impact': 'high',
                                            'ticker': ticker,
                                            'source': 'yfinance'
                                        })
                                        earnings_count += 1
                    except Exception as ticker_error:
                        # Skip this ticker if there's an error
                        continue
                print(f"[cockpit/events] Loaded {earnings_count} earnings from yfinance (total earnings events: {len([e for e in events if e['type']=='earnings'])})")
            except Exception as e:
                print(f"[cockpit/events] yfinance earnings error: {e}")
                import traceback
                traceback.print_exc()
        
        # 2. Add OPEX dates (3rd Friday of each month)
        def get_third_friday(year, month):
            """Get the third Friday of a given month"""
            from calendar import monthcalendar
            cal = monthcalendar(year, month)
            fridays = [week[4] for week in cal if week[4] != 0]
            return fridays[2] if len(fridays) >= 3 else fridays[-1]
        
        # Check current and next month for OPEX
        for month_offset in range(2):
            check_date = today + timedelta(days=30 * month_offset)
            opex_day = get_third_friday(check_date.year, check_date.month)
            opex_date = datetime(check_date.year, check_date.month, opex_day).date()
            
            if today <= opex_date <= end_date:
                events.append({
                    'type': 'opex',
                    'title': 'Monthly OPEX',
                    'date': opex_date.isoformat(),
                    'time': 'Close',
                    'impact': 'high',
                    'ticker': None,
                    'source': 'calculated'
                })
        
        # 3. Add known recurring high-impact events (FOMC dates for 2026)
        fomc_dates_2026 = [
            '2026-01-28', '2026-01-29',  # Jan meeting
            '2026-03-17', '2026-03-18',  # Mar meeting
            '2026-05-05', '2026-05-06',  # May meeting
            '2026-06-16', '2026-06-17',  # Jun meeting
            '2026-07-28', '2026-07-29',  # Jul meeting
            '2026-09-15', '2026-09-16',  # Sep meeting
            '2026-11-03', '2026-11-04',  # Nov meeting
            '2026-12-15', '2026-12-16',  # Dec meeting
        ]
        
        for fomc_date_str in fomc_dates_2026:
            try:
                fomc_date = datetime.strptime(fomc_date_str, '%Y-%m-%d').date()
                if today <= fomc_date <= end_date:
                    # Only add the second day (decision day)
                    if fomc_date_str.endswith(('29', '18', '06', '17', '16', '04')):
                        events.append({
                            'type': 'fomc',
                            'title': 'FOMC Decision',
                            'date': fomc_date.isoformat(),
                            'time': '2:00 PM',
                            'impact': 'high',
                            'ticker': None,
                            'source': 'scheduled'
                        })
            except (ValueError, TypeError):
                pass
        
        # Sort events by date
        events.sort(key=lambda x: x['date'])
        
        # Create compact badges for header (top 3 upcoming)
        badges = []
        for event in events[:3]:
            event_date = datetime.strptime(event['date'], '%Y-%m-%d').date()
            
            # Format relative date
            days_until = (event_date - today).days
            if days_until == 0:
                date_label = 'Today'
            elif days_until == 1:
                date_label = 'Tomorrow'
            else:
                date_label = event_date.strftime('%a')  # Mon, Tue, etc.
            
            # Create compact badge text
            if event['type'] == 'earnings':
                badge_text = f"{event['ticker']}"
            elif event['type'] == 'opex':
                badge_text = 'OPEX'
            elif event['type'] == 'fomc':
                badge_text = 'FOMC'
            else:
                title = event['title']
                short_names = ['CPI', 'PPI', 'NFP', 'GDP']
                badge_text = next((name for name in short_names if name in title), title[:10])
            
            badges.append({
                'text': badge_text,
                'type': event['type'],
                'date_label': date_label,
                'full_title': event['title'],
                'impact': event['impact']
            })
        
        return jsonify({
            'badges': badges,
            'events': events,
            'days_ahead': days_ahead,
            'count': len(events)
        })
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({
            "error": f"Failed to get cockpit events: {str(e)}",
            "badges": [],
            "events": []
        }), 500


@app.route('/depin/risk')
def depin_risk():
    """
    Compute de-pin risk for a ticker.
    
    This endpoint:
    1. Fetches latest 5-minute bar from yfinance
    2. Fetches options snapshot (DTE <= 2) from Massive API
    3. Loads rolling state from database
    4. Computes de-pin risk score (0-100)
    5. Saves updated state and result
    
    Query params:
        ticker: Stock/ETF symbol (default: SPY)
        bucket_dte_max: Max DTE for options bucket (default: 2)
        strike_window_pct: Strike window as % of spot (default: 0.01)
        strike_window_floor: Minimum strike window in points (default: 5.0)
    
    Returns:
        JSON with de-pin risk result including:
        - de_pin_risk: Score 0-100
        - band: LOW/MID/HIGH
        - guidance: Actionable text
        - All component metrics (atr5m, pin_persist, trend_strength, etc.)
    """
    ticker = request.args.get('ticker', 'SPY').upper()
    bucket_dte_max = int(request.args.get('bucket_dte_max', 2))
    strike_window_pct = float(request.args.get('strike_window_pct', 0.01))
    strike_window_floor = float(request.args.get('strike_window_floor', 5.0))
    
    try:
        from QuantEngine.depin_risk import (
            fetch_5m_bars,
            convert_options_to_contracts,
            compute_depin_risk
        )
        from QuantEngine.depin_risk_database import (
            load_state,
            save_state,
            save_risk_result
        )
        from QuantEngine.gex_calculator import (
            get_option_chain_snapshot,
            get_spot_price
        )
        import json
        
        # Load config for API keys
        config_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data', 'config.json')
        config = {}
        if os.path.exists(config_path):
            with open(config_path, 'r') as f:
                config = json.load(f)
        
        massive_key = config.get('MASSIVE_API_KEY', '')
        if not massive_key:
            return jsonify({"error": "MASSIVE_API_KEY not configured"}), 500
        
        # Step 1: Fetch latest 5m bar
        bars = fetch_5m_bars(ticker, period='5d')
        if not bars:
            return jsonify({"error": f"Failed to fetch 5m bars for {ticker}"}), 500
        
        latest_bar = bars[-1]  # Most recent bar
        
        # Step 2: Get spot price
        alphavantage_key = config.get('ALPHAVANTAGE_API_KEY', '')
        spot = get_spot_price(ticker, massive_key, alphavantage_key)
        if not spot:
            spot = latest_bar.close  # Fallback to bar close
        
        # Step 3: Fetch options snapshot (DTE <= 2)
        strike_range = max(strike_window_pct * spot, strike_window_floor) * 2  # ±range
        snap = get_option_chain_snapshot(
            massive_key,
            ticker,
            days_out=5,  # Only need near-term expiries
            spot_price=spot,
            strike_range=strike_range
        )
        
        if not snap or not snap.get('results'):
            return jsonify({"error": f"Failed to fetch options data for {ticker}"}), 500
        
        # Step 4: Convert options to contracts
        options = convert_options_to_contracts(snap['results'], spot)
        
        # Filter to DTE <= bucket_dte_max
        options = [opt for opt in options if opt.dte <= bucket_dte_max]
        
        if not options:
            return jsonify({
                "error": f"No options found with DTE <= {bucket_dte_max}",
                "ticker": ticker,
                "spot": spot
            }), 400
        
        # Step 5: Load rolling state
        state = load_state(ticker)
        
        # Step 6: Compute de-pin risk
        result = compute_depin_risk(
            symbol=ticker,
            bar=latest_bar,
            options_snapshot=options,
            state=state,
            bucket_dte_max=bucket_dte_max,
            strike_window_pct=strike_window_pct,
            strike_window_floor=strike_window_floor,
            vol_ref_30m=None  # Can add time-of-day median later
        )
        
        # Step 7: Save updated state and result
        save_state(ticker, state)
        save_risk_result(result)
        
        # Step 8: Return result as JSON
        return jsonify({
            "symbol": result.symbol,
            "timestamp": result.ts,
            "spot": result.spot,
            "bucket": result.bucket,
            "de_pin_risk": result.de_pin_risk,
            "band": result.band,
            "guidance": result.guidance,
            "dominant_strike": result.dominant_strike,
            "put_wall": result.put_wall,
            "call_wall": result.call_wall,
            "atr5m": result.atr5m,
            "pin_persist": result.pin_persist,
            "trend_strength": result.trend_strength,
            "move30": result.move30,
            "vol_ratio": result.vol_ratio,
            "net_gex": result.net_gex,
            "gex_change_30m": result.gex_change_30m,
            "gex_collapse": result.gex_collapse,
            "liq_fade": result.liq_fade,
            "wall_drift": result.wall_drift,
            "x_raw": result.x_raw,
            "options_count": len(options)
        })
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({
            "error": f"Failed to compute de-pin risk: {str(e)}",
            "ticker": ticker
        }), 500


@app.route('/depin/state')
def depin_state():
    """
    Get current rolling state for a ticker.
    
    Query params:
        ticker: Stock/ETF symbol (default: SPY)
    
    Returns:
        JSON with current rolling state (EMA, ATR, history arrays)
    """
    ticker = request.args.get('ticker', 'SPY').upper()
    
    try:
        from QuantEngine.depin_risk_database import load_state
        
        state = load_state(ticker)
        
        return jsonify({
            "symbol": ticker,
            "prev_close": state.prev_close,
            "ema8": state.ema8,
            "ema21": state.ema21,
            "atr14": state.atr14,
            "tr_history": state.tr_history,
            "closes_30m": state.closes_30m,
            "netgex_30m": state.netgex_30m,
            "pin_scores_15m": state.pin_scores_15m,
            "vol_30m": state.vol_30m
        })
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({
            "error": f"Failed to get state: {str(e)}",
            "ticker": ticker
        }), 500


@app.route('/depin/history')
def depin_history():
    """
    Get recent de-pin risk history for a ticker.
    
    Query params:
        ticker: Stock/ETF symbol (default: SPY)
        limit: Maximum number of records (default: 100)
    
    Returns:
        JSON array of historical risk results
    """
    ticker = request.args.get('ticker', 'SPY').upper()
    limit = int(request.args.get('limit', 100))
    
    try:
        from QuantEngine.depin_risk_database import get_risk_history
        
        history = get_risk_history(ticker, limit=limit)
        
        return jsonify({
            "symbol": ticker,
            "count": len(history),
            "history": history
        })
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({
            "error": f"Failed to get history: {str(e)}",
            "ticker": ticker
        }), 500


@app.route('/')
def index():
    """API index with endpoint documentation"""
    endpoints = {
        "status": "Market News API is running",
        "endpoints": {
            "report": "/report.json",
            "news": {
                "json": "/news.json",
                "intelligence": "/news/intelligence",
                "status": "/news/status"
            },
            "quantbot": {
                "opportunities": "/quantbot/opportunities",
                "market_analysis": "/quantbot/market-analysis",
                "news": "/quantbot/news",
                "signals": "/quantbot/signals",
                "status": "/quantbot/status"
            },
            "economic_calendar": {
                "all": "/economic-calendar",
                "upcoming": "/economic-calendar/upcoming",
                "high_impact": "/economic-calendar/high-impact"
            },
            "gex": {
                "calculate": "/gex/calculate?ticker=SPY",
                "summary": "/gex/summary",
                "tickers": "/gex/tickers",
                "price_comparison": "/gex/price-comparison?ticker=SPY"
            },
            "valuation": {
                "calculate": "/valuation/calculate?ticker=AAPL",
                "history": "/valuation/history?ticker=AAPL&days=365",
                "divergence": "/valuation/divergence?ticker=AAPL",
                "scan": "/valuation/scan?tickers=AAPL,MSFT,GOOGL&threshold=20",
                "watchlist": "/valuation/watchlist",
                "alerts": "/valuation/alerts",
                "undervalued": "/valuation/undervalued?threshold=20"
            },
            "cockpit": {
                "state": "/cockpit/state?ticker=SPY",
                "tickers": "/cockpit/tickers",
                "events": "/cockpit/events?days=3"
            },
            "depin": {
                "risk": "/depin/risk?ticker=SPY",
                "state": "/depin/state?ticker=SPY",
                "history": "/depin/history?ticker=SPY&limit=100"
            }
        }
    }
    return jsonify(endpoints)

if __name__ == '__main__':
    # Ensure data/report.json exists for the API to serve
    if not os.path.exists('data/report.json'):
        print("Warning: report.json not found in the current directory. Please run generate_report.py first.")
    app.run(host='0.0.0.0', port=5000, debug=True)