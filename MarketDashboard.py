import yfinance as yf
from datetime import datetime

def get_market_dashboard():
    """
    Fetches, displays, and analyzes key market indicators to gauge sentiment.
    """
    tickers = {
        "ES=F": {"name": "S&P 500 Futures", "sentiment": "bullish_if_up"},
        "NQ=F": {"name": "Nasdaq 100 Futures", "sentiment": "bullish_if_up"},
        "^VIX": {"name": "VIX (Fear Index)", "sentiment": "bullish_if_down"},
        "^TNX": {"name": "10-Year Treasury Yield", "sentiment": "bullish_if_down"},
        "DX-Y.NYB": {"name": "US Dollar Index", "sentiment": "bullish_if_down"}
    }

    dashboard_data = []
    sentiment_score = 0

    for ticker, info in tickers.items():
        name = info["name"]
        sentiment_direction = info["sentiment"]
        try:
            data = yf.Ticker(ticker).history(period='2d')
            if len(data) > 1:
                price = data['Close'][-1]
                prev_price = data['Close'][-2]
                change = price - prev_price
                
                if change > 0:
                    direction = "UP"
                    if sentiment_direction == "bullish_if_up":
                        sentiment_score += 1
                    else:
                        sentiment_score -= 1
                else:
                    direction = "DOWN"
                    if sentiment_direction == "bullish_if_down":
                        sentiment_score += 1
                    else:
                        sentiment_score -= 1

                formatted_price = f"{price:.2f}%" if ticker == "^TNX" else f"{price:,.2f}"
                dashboard_data.append({"name": name, "ticker": ticker, "price": formatted_price, "direction": direction})

        except Exception as e:
            dashboard_data.append({"name": name, "ticker": ticker, "price": "N/A", "direction": f"Error: {e}"})

    if sentiment_score > 1:
        overall_sentiment = "Leaning BULLISH 🐂"
    elif sentiment_score < -1:
        overall_sentiment = "Leaning BEARISH 🐻"
    else:
        overall_sentiment = "NEUTRAL / MIXED SIGNALS 😐"
        
    return {"indicators": dashboard_data, "sentiment": overall_sentiment}

if __name__ == "__main__":
    report = get_market_dashboard()
    print("--- Market Sentiment Dashboard ---")
    print(f"As of: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    for item in report["indicators"]:
        print(f"{item['name']} ({item['ticker']}): {item['price']} ({item['direction']})")
    print(f"\nOverall Sentiment: {report['sentiment']}")
