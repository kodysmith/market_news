import os
import requests
from datetime import datetime, timedelta
import numpy as np
from dotenv import load_dotenv

load_dotenv(dotenv_path=".env")
FMP_API_KEY = os.environ.get("FMP_API_KEY")
if not FMP_API_KEY:
    raise RuntimeError("FMP_API_KEY not found in .env file. Please add it as FMP_API_KEY=your_key_here")

def get_historical_volatility(symbol, days=252):
    """
    Calculate historical volatility for a stock over the specified period.
    Returns annualized volatility percentage.
    """
    try:
        # Get historical price data for volatility calculation
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days + 30)  # Extra buffer for weekends/holidays
        
        url = f"https://financialmodelingprep.com/api/v3/historical-price-full/{symbol}"
        params = {
            'from': start_date.strftime('%Y-%m-%d'),
            'to': end_date.strftime('%Y-%m-%d'),
            'apikey': FMP_API_KEY
        }
        
        response = requests.get(url, params=params)
        if response.status_code == 200:
            data = response.json()
            if 'historical' in data and len(data['historical']) > 1:
                # Calculate daily returns
                prices = [float(day['close']) for day in reversed(data['historical'])]
                if len(prices) < 20:  # Need minimum data points
                    return None
                
                # Calculate daily returns
                returns = []
                for i in range(1, len(prices)):
                    daily_return = (prices[i] - prices[i-1]) / prices[i-1]
                    returns.append(daily_return)
                
                # Calculate annualized volatility
                if len(returns) > 0:
                    volatility = np.std(returns) * np.sqrt(252)  # Annualized
                    return volatility * 100  # Convert to percentage
                    
        return None
    except Exception as e:
        print(f"Error calculating historical volatility for {symbol}: {e}")
        return None

def calculate_iv_rank(symbol, current_iv=None, lookback_days=252):
    """
    Calculate IV Rank for a stock. Since FMP doesn't provide IV directly,
    we'll use historical volatility as a proxy and calculate its percentile rank.
    """
    try:
        # Get historical volatility data points over time
        end_date = datetime.now()
        volatility_data = []
        
        # Calculate volatility for different periods to build a distribution
        for i in range(0, lookback_days, 30):  # Sample every 30 days
            sample_date = end_date - timedelta(days=i)
            start_date = sample_date - timedelta(days=30)
            
            url = f"https://financialmodelingprep.com/api/v3/historical-price-full/{symbol}"
            params = {
                'from': start_date.strftime('%Y-%m-%d'),
                'to': sample_date.strftime('%Y-%m-%d'),
                'apikey': FMP_API_KEY
            }
            
            response = requests.get(url, params=params)
            if response.status_code == 200:
                data = response.json()
                if 'historical' in data and len(data['historical']) > 10:
                    prices = [float(day['close']) for day in reversed(data['historical'])]
                    if len(prices) > 1:
                        returns = []
                        for j in range(1, len(prices)):
                            daily_return = (prices[j] - prices[j-1]) / prices[j-1]
                            returns.append(daily_return)
                        
                        if len(returns) > 0:
                            vol = np.std(returns) * np.sqrt(252) * 100
                            volatility_data.append(vol)
        
        # Calculate current volatility
        current_vol = current_iv if current_iv else get_historical_volatility(symbol, 30)
        
        if current_vol and len(volatility_data) > 10:
            # Calculate percentile rank
            volatility_data.append(current_vol)
            volatility_data.sort()
            
            rank = volatility_data.index(current_vol) / len(volatility_data) * 100
            return {
                'iv_rank': round(rank, 1),
                'current_iv': round(current_vol, 2),
                'iv_high_52w': round(max(volatility_data), 2),
                'iv_low_52w': round(min(volatility_data), 2)
            }
            
        return None
    except Exception as e:
        print(f"Error calculating IV rank for {symbol}: {e}")
        return None

def get_options_data_for_symbol(symbol):
    """
    Get options-related data for a symbol including IV metrics.
    This is a simplified version since FMP doesn't have dedicated options endpoints.
    """
    try:
        # Get current quote
        quote_url = f"https://financialmodelingprep.com/api/v3/quote/{symbol}?apikey={FMP_API_KEY}"
        quote_response = requests.get(quote_url)
        
        if quote_response.status_code == 200:
            quote_data = quote_response.json()
            if quote_data:
                current_price = quote_data[0]['price']
                
                # Calculate IV metrics
                iv_data = calculate_iv_rank(symbol)
                historical_vol = get_historical_volatility(symbol)
                
                return {
                    'symbol': symbol,
                    'current_price': current_price,
                    'iv_rank': iv_data['iv_rank'] if iv_data else None,
                    'current_iv': iv_data['current_iv'] if iv_data else historical_vol,
                    'iv_high_52w': iv_data['iv_high_52w'] if iv_data else None,
                    'iv_low_52w': iv_data['iv_low_52w'] if iv_data else None,
                    'historical_volatility': historical_vol
                }
        
        return None
    except Exception as e:
        print(f"Error getting options data for {symbol}: {e}")
        return None

def get_fmp_insights():
    today = datetime.now().date()
    week_later = today + timedelta(days=7)
    def fetch(url):
        resp = requests.get(url)
        if resp.status_code == 200:
            return resp.json()
        return []
    
    # Economic Calendar (US only, Medium/High impact)
    calendar_url = f"https://financialmodelingprep.com/api/v3/economic_calendar?from={today}&to={week_later}&apikey={FMP_API_KEY}"
    economic_calendar = [
        e for e in fetch(calendar_url)
        if e.get('country') == 'US' and e.get('impact') in ('Medium', 'High')
    ]
    
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
    
    # Add IV data for top gainers/losers (sample a few to avoid API limits)
    enhanced_gainers = []
    for gainer in top_gainers[:5]:  # Limit to top 5 to avoid API limits
        symbol = gainer.get('symbol')
        if symbol:
            iv_data = get_options_data_for_symbol(symbol)
            if iv_data:
                gainer['iv_rank'] = iv_data['iv_rank']
                gainer['current_iv'] = iv_data['current_iv']
        enhanced_gainers.append(gainer)
    
    enhanced_losers = []
    for loser in top_losers[:5]:  # Limit to top 5 to avoid API limits
        symbol = loser.get('symbol')
        if symbol:
            iv_data = get_options_data_for_symbol(symbol)
            if iv_data:
                loser['iv_rank'] = iv_data['iv_rank']
                loser['current_iv'] = iv_data['current_iv']
        enhanced_losers.append(loser)
    
    return {
        "economic_calendar": economic_calendar,
        "earnings_calendar": earnings_calendar,
        "top_gainers": enhanced_gainers,
        "top_losers": enhanced_losers,
        "indices": indices,
    } 