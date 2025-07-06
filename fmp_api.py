import os
import requests
from datetime import datetime, timedelta
from dotenv import load_dotenv

load_dotenv(dotenv_path=".env")
FMP_API_KEY = os.environ.get("FMP_API_KEY")
if not FMP_API_KEY:
    raise RuntimeError("FMP_API_KEY not found in .env file. Please add it as FMP_API_KEY=your_key_here")

def get_fmp_insights():
    today = datetime.now().date()
    week_later = today + timedelta(days=7)
    def fetch(url):
        resp = requests.get(url)
        if resp.status_code == 200:
            return resp.json()
        return []
    calendar_url = f"https://financialmodelingprep.com/api/v3/economic_calendar?from={today}&to={week_later}&apikey={FMP_API_KEY}"
    earnings_url = f"https://financialmodelingprep.com/api/v3/earning_calendar?from={today}&to={week_later}&apikey={FMP_API_KEY}"
    gainers_url = f"https://financialmodelingprep.com/api/v3/stock_market/gainers?apikey={FMP_API_KEY}"
    losers_url = f"https://financialmodelingprep.com/api/v3/stock_market/losers?apikey={FMP_API_KEY}"
    indices_url = f"https://financialmodelingprep.com/api/v3/quotes/index?apikey={FMP_API_KEY}"
    return {
        "economic_calendar": fetch(calendar_url),
        "earnings_calendar": fetch(earnings_url),
        "top_gainers": fetch(gainers_url),
        "top_losers": fetch(losers_url),
        "indices": fetch(indices_url),
    } 