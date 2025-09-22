from flask import Flask, send_from_directory, jsonify, request
import os
from flask_cors import CORS
import subprocess
import time

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
        if is_report_stale():
            # Regenerate the report
            subprocess.run(['python3', 'generate_report.py'], check=True)
        return send_from_directory(os.getcwd(), REPORT_PATH)
    except FileNotFoundError:
        return jsonify({"error": "report.json not found"}), 404
    except subprocess.CalledProcessError as e:
        return jsonify({"error": f"Failed to regenerate report: {e}"}), 500

@app.route('/news.json')
def get_news_json():
    try:
        return send_from_directory(os.getcwd(), "data/news.json")
    except FileNotFoundError:
        return jsonify({"error": "data/news.json not found"}), 404

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

@app.route('/')
def index():
    return "Market News API is running. Access /report.json for data. QuantBot endpoints: /quantbot/*, Economic Calendar: /economic-calendar/*"

if __name__ == '__main__':
    # Ensure data/report.json exists for the API to serve
    if not os.path.exists('data/report.json'):
        print("Warning: report.json not found in the current directory. Please run generate_report.py first.")
    app.run(host='0.0.0.0', port=5000, debug=True)