from MarketDashboard import get_market_dashboard
from Scanner1 import get_trade_ideas
from GammaScalpingAnalyzer import GammaScalpingAnalyzer
from datetime import datetime, timedelta
import json
import yfinance as yf
import pandas as pd
import fmp_api

# Load configuration
with open('config.json', 'r') as f:
    config = json.load(f)

SORT_TRADES_BY = config['SORT_TRADES_BY']

def get_vix_data_from_yfinance():
    today = datetime.now()
    # Fetch data for the last 30 days to ensure we get enough data for trends
    start_date = today - timedelta(days=30)
    
    try:
        vix_data = yf.download("^VIX", start=start_date, end=today)
        if not vix_data.empty:
            # Convert DataFrame to a list of dictionaries for JSON serialization
            # Only include 'Date' and 'Close' price
            vix_list = []
            for index, row in vix_data.iterrows():
                vix_list.append({
                    "date": index.strftime('%Y-%m-%d'),
                    "close": float(round(row['Close'], 2))
                })
            return vix_list
        else:
            print("No VIX data found from yfinance for the specified period.")
            return []
    except Exception as e:
        print(f"Error fetching VIX data from yfinance: {e}")
        return []


def generate_html_report(dashboard_data, trade_ideas, gamma_analysis=None):
    """
    Generates an HTML report from the dashboard, trade ideas, and gamma analysis.
    """

    # Sort trade ideas
    if SORT_TRADES_BY == "Prob. of Success":
        trade_ideas.sort(key=lambda x: float(x['metric_value'].replace('%', '')) if x.get('metric_name') == "Prob. of Success" else -1, reverse=True)
    elif SORT_TRADES_BY == "Credit":
        trade_ideas.sort(key=lambda x: x['cost'], reverse=True)
    elif SORT_TRADES_BY == "Implied Move":
        trade_ideas.sort(key=lambda x: float(x['metric_value'].replace('+/-', '').replace('%', '')) if x.get('metric_name') == "Implied Move" else float('inf'))
    else:
        trade_ideas.sort(key=lambda x: x['ticker'])

    html = f"""
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Market Report</title>
        <style>
            body {{ font-family: sans-serif; margin: 2em; background-color: #f4f4f9; color: #333; }}
            .container {{ max-width: 800px; margin: auto; background: white; padding: 20px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }}
            h1, h2 {{ color: #2c3e50; border-bottom: 2px solid #3498db; padding-bottom: 10px; }}
            .report-header {{ text-align: center; margin-bottom: 20px; }}
            .sentiment {{ font-size: 1.2em; font-weight: bold; text-align: center; padding: 10px; border-radius: 5px; margin-bottom: 20px; }}
            .bullish {{ background-color: #e8f5e9; color: #2e7d32; }}
            .bearish {{ background-color: #ffebee; color: #c62828; }}
            .neutral {{ background-color: #fffde7; color: #f57f17; }}
            .gamma-scalping {{ background-color: #e3f2fd; color: #1976d2; }}
            .premium-selling {{ background-color: #fff3e0; color: #f57c00; }}
            .mixed {{ background-color: #f3e5f5; color: #7b1fa2; }}
            table {{ width: 100%; border-collapse: collapse; margin-bottom: 20px; }}
            th, td {{ padding: 12px; border: 1px solid #ddd; text-align: left; }}
            th {{ background-color: #3498db; color: white; }}
            .trade-idea {{ border-left-width: 5px; border-left-style: solid; padding: 15px; margin-bottom: 15px; border-radius: 5px; }}
            .bullish-trade {{ border-left-color: #2ecc71; background-color: #e8f5e9; }}
            .bearish-trade {{ border-left-color: #e74c3c; background-color: #ffebee; }}
            .volatility-trade {{ border-left-color: #f1c40f; background-color: #fffde7; }}
            .gamma-trade {{ border-left-color: #2196f3; background-color: #e3f2fd; }}
            .premium-trade {{ border-left-color: #ff9800; background-color: #fff3e0; }}
            .gemini-analysis {{ background-color: #f9f9f9; border-left: 5px solid #9b59b6; padding: 15px; margin-top: 10px; border-radius: 5px; }}
            .footer {{ text-align: center; margin-top: 20px; font-size: 0.8em; color: #777; }}
        </style>
    </head>
    <body>
        <div class="container">
            <div class="report-header">
                <h1>Daily Market Report</h1>
                <p>Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
            </div>

            <h2>Market Sentiment Analysis</h2>
    """

    sentiment_class = "neutral"
    if "BULLISH" in dashboard_data["sentiment"]:
        sentiment_class = "bullish"
    elif "BEARISH" in dashboard_data["sentiment"]:
        sentiment_class = "bearish"
    html += f'<div class="sentiment {sentiment_class}">{dashboard_data["sentiment"]}</div>'

    html += """
            <table>
                <tr><th>Indicator</th><th>Ticker</th><th>Last Price</th><th>Direction</th></tr>
    """
    for item in dashboard_data["indicators"]:
        html += f"<tr><td>{item['name']}</td><td>{item['ticker']}</td><td>{item['price']}</td><td>{item['direction']}</td></tr>"
    html += "</table>"

    # Add Gamma Scalping Analysis section
    if gamma_analysis:
        html += "<h2>Gamma Scalping vs Premium Selling Analysis</h2>"
        
        # Market-wide recommendation
        gamma_class = "neutral"
        if gamma_analysis['market_recommendation'] == "GAMMA_SCALPING_FAVORED":
            gamma_class = "gamma-scalping"
        elif gamma_analysis['market_recommendation'] == "PREMIUM_SELLING_FAVORED":
            gamma_class = "premium-selling"
        else:
            gamma_class = "mixed"
        
        html += f'<div class="sentiment {gamma_class}">'
        html += f"{gamma_analysis['market_recommendation'].replace('_', ' ').title()}"
        html += f"<br>Market Gamma Score: {gamma_analysis['avg_gamma_score']:.1f}/100"
        html += f"<br>Gamma Scalping Tickers: {gamma_analysis['gamma_scalping_count']} | Premium Selling Tickers: {gamma_analysis['premium_selling_count']}"
        html += '</div>'

        # Individual ticker analysis
        html += """
                <table>
                    <tr><th>Ticker</th><th>Recommendation</th><th>Score</th><th>Primary Reason</th></tr>
        """
        for ticker_analysis in gamma_analysis['individual_analysis']:
            html += f"<tr><td>{ticker_analysis['ticker']}</td><td>{ticker_analysis['recommendation'].replace('_', ' ')}</td><td>{ticker_analysis['gamma_score']:.0f}</td><td>{ticker_analysis['reasons'][0] if ticker_analysis['reasons'] else 'N/A'}</td></tr>"
        html += "</table>"

    html += "<h2>Potential Trade Ideas</h2>"
    for trade in trade_ideas:
        if "Error" in trade.get("strategy", ""):
            continue

        trade_class = ""
        analysis = ""
        cost_type = "Credit" if trade['cost'] > 0 else "Debit"

        if "Bull" in trade["strategy"]:
            trade_class = "bullish-trade"
            analysis = "This is a bullish strategy. "
            if "BULLISH" in dashboard_data["sentiment"]:
                analysis += "This aligns with the current bullish market sentiment."
            elif "BEARISH" in dashboard_data["sentiment"]:
                analysis += "This contradicts the current bearish market sentiment."

        elif "Bear" in trade["strategy"]:
            trade_class = "bearish-trade"
            analysis = "This is a bearish strategy. "
            if "BEARISH" in dashboard_data["sentiment"]:
                analysis += "This aligns with the current bearish market sentiment."
            elif "BULLISH" in dashboard_data["sentiment"]:
                analysis += "This contradicts the current bullish market sentiment."

        elif "Straddle" in trade["strategy"] or "Strangle" in trade["strategy"]:
            trade_class = "volatility-trade"
            analysis = "This is a non-directional volatility strategy. It profits if the stock makes a large price move, up or down. It is suitable for uncertain market conditions or when a big news event is expected."

        html += f"""
        <div class="trade-idea {trade_class}">
            <strong>{trade['ticker']} - {trade['strategy']} ({trade['expiry']})</strong><br>
            - {trade['details']}<br>
            - {cost_type}: ${abs(trade['cost']):.2f}<br>
            - <strong>{trade['metric_name']}: {trade['metric_value']}</strong>
            {f"<br>Max Profit: ${trade['max_profit']:.2f} | Max Loss: ${trade['max_loss']:.2f} | Risk/Reward: {trade['risk_reward_ratio']:.2f}:1" if trade.get('max_profit') is not None else ""}
        </div>
        <div class="gemini-analysis">
            <strong>Analysis:</strong> {analysis}
            {f"<br>This trade has an estimated {trade['metric_name']} of {trade['metric_value']} based on current market conditions." if "Prob. of Success" in trade['metric_name'] else ""}
        </div>
        """

    html += """
            <div class="footer">
                <p>This report is for informational purposes only and does not constitute financial advice.</p>
            </div>
        </div>
    </body>
    </html>
    """
    return html

if __name__ == "__main__":
    print("Generating report...")
    dashboard_info = get_market_dashboard()
    trades = get_trade_ideas()
    vix_data = get_vix_data_from_yfinance()
    
    # Add gamma scalping analysis
    print("Analyzing gamma scalping opportunities...")
    gamma_analyzer = GammaScalpingAnalyzer()
    gamma_tickers = config.get('GAMMA_SCALPING_TICKERS', ['SPY', 'QQQ', 'IWM', 'AAPL', 'TSLA', 'NVDA', 'MSFT', 'GOOGL'])
    gamma_analysis = gamma_analyzer.get_market_gamma_scalping_analysis(gamma_tickers)

    print(f"Type of dashboard_info: {type(dashboard_info)}")
    print(f"Type of trades: {type(trades)}")
    print(f"Type of vix_data: {type(vix_data)}")
    print(f"Type of gamma_analysis: {type(gamma_analysis)}")

    # Check types of elements within dashboard_info
    if isinstance(dashboard_info, dict) and "indicators" in dashboard_info:
        for i, item in enumerate(dashboard_info["indicators"]):
            for key, value in item.items():
                print(f"  dashboard_info[\"indicators\"][{i}][\"{key}\"] type: {type(value)}")

    # Check types of elements within trades
    if isinstance(trades, list):
        for i, trade in enumerate(trades):
            if isinstance(trade, dict):
                for key, value in trade.items():
                    print(f"  trades[{i}][\"{key}\"] type: {type(value)}")

    # Check types of elements within vix_data
    if isinstance(vix_data, list):
        for i, item in enumerate(vix_data):
            if isinstance(item, dict):
                for key, value in item.items():
                    print(f"  vix_data[{i}][\"{key}\"] type: {type(value)}")

    # Collect skipped tickers from Scanner1.py (if any)
    skipped_tickers = []
    for trade in trades:
        if "Error" in trade.get("strategy", ""):
            skipped_tickers.append(trade)

    # Filter out error trades for the main report
    trades = [trade for trade in trades if "Error" not in trade.get("strategy", "")]

    # Generate HTML report with gamma analysis
    html_content = generate_html_report(dashboard_info, trades, gamma_analysis)
    with open("report.html", "w") as f:
        f.write(html_content)
    print("Report saved to report.html")

    # --- Build top_strategies dynamically from trade ideas and sentiment ---
    def classify_strategy(trade):
        name = trade['strategy'].lower()
        if 'bull' in name or 'covered call' in name or 'long' in name:
            return 'bullish'
        elif 'bear' in name or 'protective put' in name or 'short' in name:
            return 'bearish'
        elif 'straddle' in name or 'strangle' in name or 'gamma' in name or 'iron condor' in name:
            return 'neutral'
        else:
            return 'other'

    sentiment = dashboard_info.get("sentiment", "").upper()
    if "BULLISH" in sentiment:
        sentiment_type = 'bullish'
    elif "BEARISH" in sentiment:
        sentiment_type = 'bearish'
    else:
        sentiment_type = 'neutral'

    from collections import defaultdict
    strategy_groups = defaultdict(list)
    for trade in trades:
        strat_type = classify_strategy(trade)
        strategy_groups[strat_type].append(trade)

    # Build top_strategies list
    top_strategies = []
    for strat in strategy_groups[sentiment_type]:
        strat_name = strat['strategy']
        existing = next((s for s in top_strategies if s['name'] == strat_name), None)
        if not existing:
            top_strategies.append({
                "name": strat_name,
                "score": strat.get('max_profit', 0) or strat.get('cost', 0),
                "description": strat.get('details', ''),
                "top_tickers": []
            })
            existing = top_strategies[-1]
        existing['top_tickers'].append({
            "ticker": strat['ticker'],
            "score": strat.get('max_profit', 0) or strat.get('cost', 0),
            "setup": {
                "expiry": strat.get('expiry'),
                "details": strat.get('details')
            },
            "reason": strat.get('metric_name', '') + ': ' + strat.get('metric_value', '')
        })
    # Sort and take top 3
    top_strategies = sorted(top_strategies, key=lambda s: s['score'], reverse=True)[:3]

    # --- Fetch FMP Insights ---
    print("Fetching FMP insights from FMP API...")
    fmp_insights = fmp_api.get_fmp_insights()

    # Generate JSON report with gamma analysis and FMP insights
    report_data = {
        "timestamp": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        "market_sentiment": dashboard_info,
        "trade_ideas": trades,
        "skipped_tickers": skipped_tickers,
        "vix_data": vix_data,
        "gamma_analysis": gamma_analysis,
        "top_strategies": top_strategies,
        "economic_calendar": fmp_insights.get("economic_calendar", []),
        "earnings_calendar": fmp_insights.get("earnings_calendar", []),
        "top_gainers": fmp_insights.get("top_gainers", []),
        "top_losers": fmp_insights.get("top_losers", []),
        "indices": fmp_insights.get("indices", []),
    }
    with open("report.json", "w") as f:
        json.dump(report_data, f, indent=4)
    print("Report saved to report.json")
    
    # Display gamma scalping summary
    if gamma_analysis:
        print(f"\n--- GAMMA SCALPING ANALYSIS ---")
        print(f"Market Recommendation: {gamma_analysis['market_recommendation']}")
        print(f"Average Gamma Score: {gamma_analysis['avg_gamma_score']:.1f}/100")
        print(f"Gamma Scalping Opportunities: {gamma_analysis['gamma_scalping_count']}")
        print(f"Premium Selling Opportunities: {gamma_analysis['premium_selling_count']}")