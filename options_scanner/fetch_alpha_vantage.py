"""
Alpha Vantage API integration for fetching options data
"""

import requests
import json
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from datetime import datetime, timedelta
import time

@dataclass
class OptionQuote:
    """Represents a single option quote"""
    contract_id: str
    symbol: str
    expiry: str
    strike: float
    right: str  # 'call' or 'put'
    bid: float
    ask: float
    mid: float
    last: float
    volume: int
    open_interest: int
    implied_volatility: Optional[float] = None
    delta: Optional[float] = None
    gamma: Optional[float] = None
    theta: Optional[float] = None
    vega: Optional[float] = None
    rho: Optional[float] = None

class AlphaVantageClient:
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.base_url = "https://www.alphavantage.co/query"
        self.rate_limit_delay = 12  # Alpha Vantage free tier: 5 calls per minute
    
    def _make_request(self, params: Dict[str, str]) -> Dict[str, Any]:
        """Make API request with rate limiting"""
        time.sleep(self.rate_limit_delay)  # Rate limiting
        
        try:
            response = requests.get(self.base_url, params=params, timeout=30)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            print(f"API request failed: {e}")
            return {}
    
    def get_spot(self, ticker: str) -> Optional[float]:
        """Get current spot price for a ticker"""
        params = {
            'function': 'GLOBAL_QUOTE',
            'symbol': ticker,
            'apikey': self.api_key
        }
        
        data = self._make_request(params)
        
        if 'Global Quote' in data and '05. price' in data['Global Quote']:
            try:
                return float(data['Global Quote']['05. price'])
            except (ValueError, TypeError):
                pass
        
        print(f"Failed to get spot price for {ticker}")
        return None
    
    def get_option_chain(self, ticker: str) -> List[OptionQuote]:
        """Get options chain for a ticker"""
        params = {
            'function': 'HISTORICAL_OPTIONS',
            'symbol': ticker,
            'apikey': self.api_key
        }
        
        data = self._make_request(params)
        
        if 'data' not in data:
            print(f"No options data found for {ticker}")
            return []
        
        options = []
        for option_data in data['data']:
            try:
                option = self._parse_option_quote(option_data)
                if option:
                    options.append(option)
            except Exception as e:
                print(f"Error parsing option: {e}")
                continue
        
        return options
    
    def _parse_option_quote(self, data: Dict[str, Any]) -> Optional[OptionQuote]:
        """Parse individual option quote from API response"""
        try:
            # Calculate mid price
            bid = float(data.get('bid', 0))
            ask = float(data.get('ask', 0))
            mid = (bid + ask) / 2 if bid > 0 and ask > 0 else float(data.get('mark', 0))
            
            # Parse implied volatility
            iv = None
            if 'implied_volatility' in data:
                try:
                    iv = float(data['implied_volatility'])
                except (ValueError, TypeError):
                    pass
            
            # Parse Greeks
            delta = None
            if 'delta' in data:
                try:
                    delta = float(data['delta'])
                except (ValueError, TypeError):
                    pass
            
            gamma = None
            if 'gamma' in data:
                try:
                    gamma = float(data['gamma'])
                except (ValueError, TypeError):
                    pass
            
            theta = None
            if 'theta' in data:
                try:
                    theta = float(data['theta'])
                except (ValueError, TypeError):
                    pass
            
            vega = None
            if 'vega' in data:
                try:
                    vega = float(data['vega'])
                except (ValueError, TypeError):
                    pass
            
            rho = None
            if 'rho' in data:
                try:
                    rho = float(data['rho'])
                except (ValueError, TypeError):
                    pass
            
            return OptionQuote(
                contract_id=data.get('contractID', ''),
                symbol=data.get('symbol', ''),
                expiry=data.get('expiration', ''),
                strike=float(data.get('strike', 0)),
                right=data.get('type', '').lower(),
                bid=bid,
                ask=ask,
                mid=mid,
                last=float(data.get('last', 0)),
                volume=int(data.get('volume', 0)),
                open_interest=int(data.get('open_interest', 0)),
                implied_volatility=iv,
                delta=delta,
                gamma=gamma,
                theta=theta,
                vega=vega,
                rho=rho
            )
        except Exception as e:
            print(f"Error parsing option quote: {e}")
            return None
    
    def get_put_options(self, ticker: str) -> List[OptionQuote]:
        """Get only put options for a ticker"""
        all_options = self.get_option_chain(ticker)
        return [opt for opt in all_options if opt.right == 'put']
    
    def filter_by_dte(self, options: List[OptionQuote], min_dte: int, max_dte: int) -> List[OptionQuote]:
        """Filter options by days to expiration"""
        filtered = []
        today = datetime.now().date()
        
        for option in options:
            try:
                expiry_date = datetime.strptime(option.expiry, '%Y-%m-%d').date()
                dte = (expiry_date - today).days
                
                if min_dte <= dte <= max_dte:
                    filtered.append(option)
            except ValueError:
                # Skip options with invalid expiry dates
                continue
        
        return filtered
    
    def get_otm_puts(self, ticker: str, spot_price: float, min_dte: int = 20, max_dte: int = 45) -> List[OptionQuote]:
        """Get out-of-the-money put options within DTE range"""
        all_puts = self.get_put_options(ticker)
        dte_filtered = self.filter_by_dte(all_puts, min_dte, max_dte)
        
        # Filter for OTM puts (strike < spot)
        otm_puts = [put for put in dte_filtered if put.strike < spot_price]
        
        # Sort by strike price (descending for bull put spreads)
        otm_puts.sort(key=lambda x: x.strike, reverse=True)
        
        return otm_puts

