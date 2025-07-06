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
    # Economic Calendar (US only)
    calendar_url = f"https://financialmodelingprep.com/api/v3/economic_calendar?from={today}&to={week_later}&apikey={FMP_API_KEY}"
    economic_calendar = [e for e in fetch(calendar_url) if e.get('country') == 'US']
    # Earnings Calendar (NYSE/NASDAQ only)
    earnings_url = f"https://financialmodelingprep.com/api/v3/earning_calendar?from={today}&to={week_later}&apikey={FMP_API_KEY}"
    earnings_calendar = [e for e in fetch(earnings_url) if e.get('exchange') in ('NYSE', 'NASDAQ')]
    # Top Gainers (NYSE/NASDAQ only)
    gainers_url = f"https://financialmodelingprep.com/api/v3/gainers?apikey={FMP_API_KEY}"
    top_gainers = [g for g in fetch(gainers_url) if g.get('exchangeShortName') in ('NYSE', 'NASDAQ')]
    # Top Losers (NYSE/NASDAQ only)
    losers_url = f"https://financialmodelingprep.com/api/v3/losers?apikey={FMP_API_KEY}"
    top_losers = [l for l in fetch(losers_url) if l.get('exchangeShortName') in ('NYSE', 'NASDAQ')]
    # Major US Indices only
    indices_url = f"https://financialmodelingprep.com/api/v3/quotes/index?apikey={FMP_API_KEY}"
    us_indices = {'^GSPC', '^DJI', '^IXIC', '^VIX', 'SPY', 'QQQ', 'DIA'}
    indices = [i for i in fetch(indices_url) if i.get('symbol') in us_indices]
    return {
        "economic_calendar": economic_calendar,
        "earnings_calendar": earnings_calendar,
        "top_gainers": top_gainers,
        "top_losers": top_losers,
        "indices": indices,
    } 