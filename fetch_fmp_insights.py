import os
import requests
from datetime import datetime, timedelta
from dotenv import load_dotenv
import json

load_dotenv(dotenv_path=".env")
FMP_API_KEY = os.environ.get("FMP_API_KEY")
if not FMP_API_KEY:
    raise RuntimeError("FMP_API_KEY not found in .env file. Please add it as FMP_API_KEY=your_key_here")

today = datetime.now().date()
week_later = today + timedelta(days=7)

def fetch_and_print(name, url):
    print(f"\nFetching {name}...")
    resp = requests.get(url)
    if resp.status_code == 200:
        data = resp.json()
        print(json.dumps(data[:3] if isinstance(data, list) else data, indent=2))
    else:
        print(f"Failed to fetch {name}: {resp.status_code}")

# Economic Calendar
calendar_url = f"https://financialmodelingprep.com/api/v3/economic_calendar?from={today}&to={week_later}&apikey={FMP_API_KEY}"
fetch_and_print("Economic Calendar", calendar_url)

# Earnings Calendar
earnings_url = f"https://financialmodelingprep.com/api/v3/earning_calendar?from={today}&to={week_later}&apikey={FMP_API_KEY}"
fetch_and_print("Earnings Calendar", earnings_url)

# Market Movers: Gainers
movers_gainers_url = f"https://financialmodelingprep.com/api/v3/stock_market/gainers?apikey={FMP_API_KEY}"
fetch_and_print("Top Gainers", movers_gainers_url)

# Market Movers: Losers
movers_losers_url = f"https://financialmodelingprep.com/api/v3/stock_market/losers?apikey={FMP_API_KEY}"
fetch_and_print("Top Losers", movers_losers_url)

# Indices/Volatility
indices_url = f"https://financialmodelingprep.com/api/v3/quotes/index?apikey={FMP_API_KEY}"
fetch_and_print("Indices", indices_url) 