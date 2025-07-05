from MarketDashboard import get_market_dashboard
from Scanner1 import get_trade_ideas
from datetime import datetime
import json

# Load configuration
with open('config.json', 'r') as f:
    config = json.load(f)

SORT_TRADES_BY = config['SORT_TRADES_BY']


def generate_html_report(dashboard_data, trade_ideas):
    """
    Generates an HTML report from the dashboard and trade ideas.
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
            table {{ width: 100%; border-collapse: collapse; margin-bottom: 20px; }}
            th, td {{ padding: 12px; border: 1px solid #ddd; text-align: left; }}
            th {{ background-color: #3498db; color: white; }}
            .trade-idea {{ border-left-width: 5px; border-left-style: solid; padding: 15px; margin-bottom: 15px; border-radius: 5px; }}
            .bullish-trade {{ border-left-color: #2ecc71; background-color: #e8f5e9; }}
            .bearish-trade {{ border-left-color: #e74c3c; background-color: #ffebee; }}
            .volatility-trade {{ border-left-color: #f1c40f; background-color: #fffde7; }}
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
            <strong>Gemini Analysis:</strong> {analysis}
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
    
    # Collect skipped tickers from Scanner1.py (if any)
    skipped_tickers = []
    for trade in trades:
        if "Error" in trade.get("strategy", ""):
            skipped_tickers.append(trade)

    # Filter out error trades for the main report
    trades = [trade for trade in trades if "Error" not in trade.get("strategy", "")]

    # Generate HTML report
    html_content = generate_html_report(dashboard_info, trades)
    with open("report.html", "w") as f:
        f.write(html_content)
    print("Report saved to report.html")

    # Generate JSON report
    report_data = {
        "timestamp": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        "market_sentiment": dashboard_info,
        "trade_ideas": trades,
        "skipped_tickers": skipped_tickers
    }
    with open("report.json", "w") as f:
        json.dump(report_data, f, indent=4)
    print("Report saved to report.json")
