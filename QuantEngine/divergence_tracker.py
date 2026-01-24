"""
SPY/SPX Divergence Tracker
Tracks price movements between SPY and SPX to identify arbitrage opportunities
and understand retail vs institutional trading behavior.
"""

import time
from datetime import datetime, timedelta
from typing import Optional, Dict, List, Tuple
import requests
import json
from pathlib import Path
import yfinance as yf


class DivergenceTracker:
    """Track SPY/SPX price divergence and identify first mover"""
    
    def __init__(self, alphavantage_api_key: str, massive_api_key: Optional[str] = None, config: Optional[Dict] = None):
        self.alphavantage_api_key = alphavantage_api_key
        self.massive_api_key = massive_api_key  # Keep for fallback, but prefer yfinance for SPX
        self.config = config or {}
        
        # Divergence settings
        self.spx_spy_ratio = 10.0  # SPX is typically ~10x SPY
        self.divergence_threshold = self.config.get("divergence_threshold_percent", 0.1)
        self.convergence_threshold = self.config.get("convergence_threshold_percent", 0.05)
        
        # Price history (in-memory for now)
        self.price_history: List[Dict] = []
        self.last_update_time: Optional[float] = None
        self.min_update_interval = 12.0  # Alpha Vantage rate limit: ~5 calls/minute
        
        # Current state
        self.current_spy_price: Optional[float] = None
        self.current_spx_price: Optional[float] = None
        self.previous_spy_price: Optional[float] = None
        self.previous_spx_price: Optional[float] = None
        
        # Divergence tracking
        self.divergence_start_time: Optional[float] = None
        self.last_first_mover: Optional[str] = None
        self.last_first_mover_time: Optional[float] = None
    
    def fetch_price_alphavantage(self, symbol: str) -> Optional[float]:
        """Fetch current price from Alpha Vantage (works for stocks/ETFs like SPY)"""
        try:
            url = "https://www.alphavantage.co/query"
            params = {
                "function": "GLOBAL_QUOTE",
                "symbol": symbol,
                "apikey": self.alphavantage_api_key,
            }
            r = requests.get(url, params=params, timeout=10)
            r.raise_for_status()
            data = r.json()
            
            quote = data.get("Global Quote", {})
            price_str = quote.get("05. price")
            
            if price_str:
                return float(price_str)
        except Exception as e:
            print(f"[divergence] Alpha Vantage failed for {symbol}: {e}")
        return None
    
    def fetch_price_yfinance(self, symbol: str) -> Optional[float]:
        """Fetch current price from Yahoo Finance via yfinance (works for indices like SPX)"""
        try:
            # For indices like SPX, use ^ prefix
            yf_symbol = f"^{symbol}" if symbol == "SPX" else symbol
            ticker = yf.Ticker(yf_symbol)
            info = ticker.info
            
            # Try to get current price from info
            current_price = info.get('regularMarketPrice') or info.get('previousClose')
            if current_price:
                return float(current_price)
            
            # Fallback: get latest price from history
            hist = ticker.history(period="1d", interval="1m")
            if not hist.empty:
                return float(hist['Close'].iloc[-1])
        except Exception as e:
            print(f"[divergence] yfinance failed for {symbol}: {e}")
        return None
    
    def fetch_price_massive(self, symbol: str) -> Optional[float]:
        """Fetch current price from Massive API v2 (fallback for indices)"""
        if not self.massive_api_key:
            return None
        
        try:
            url = f"https://api.massive.com/v2/snapshot/locale/us/markets/stocks/tickers/{symbol}"
            params = {"apiKey": self.massive_api_key}
            r = requests.get(url, params=params, timeout=10)
            r.raise_for_status()
            data = r.json()
            
            tkr = data.get("ticker", {})
            if not isinstance(tkr, dict):
                return None
            
            # Priority order: lastTrade -> lastQuote midpoint -> min.c -> day.c -> prevDay.c
            last_trade = tkr.get("lastTrade", {})
            p = last_trade.get("p")
            if isinstance(p, (int, float)) and p > 0:
                return float(p)
            
            last_quote = tkr.get("lastQuote", {})
            bid = last_quote.get("p")
            ask = last_quote.get("P")
            if isinstance(bid, (int, float)) and isinstance(ask, (int, float)) and bid > 0 and ask > 0:
                return float((bid + ask) / 2.0)
            
            minute = tkr.get("min", {})
            mc = minute.get("c")
            if isinstance(mc, (int, float)) and mc > 0:
                return float(mc)
            
            day = tkr.get("day", {})
            dc = day.get("c")
            if isinstance(dc, (int, float)) and dc > 0:
                return float(dc)
            
            prev = tkr.get("prevDay", {})
            pc = prev.get("c")
            if isinstance(pc, (int, float)) and pc > 0:
                return float(pc)
        except Exception as e:
            print(f"[divergence] Massive API failed for {symbol}: {e}")
        return None
    
    def fetch_price(self, symbol: str) -> Optional[float]:
        """
        Fetch current price with smart fallback:
        - For SPY: Use Alpha Vantage (or yfinance as fallback)
        - For SPX: Use yfinance (Yahoo Finance supports indices with ^ prefix)
        """
        # SPX is an index - use yfinance (Yahoo Finance supports ^SPX)
        if symbol == "SPX":
            price = self.fetch_price_yfinance(symbol)
            if price:
                return price
            # Fallback to Massive API if yfinance fails
            if self.massive_api_key:
                return self.fetch_price_massive(symbol)
            return None
        
        # For stocks/ETFs like SPY, try Alpha Vantage first
        price = self.fetch_price_alphavantage(symbol)
        if price:
            return price
        
        # Fallback to yfinance if Alpha Vantage fails
        price = self.fetch_price_yfinance(symbol)
        if price:
            return price
        
        # Final fallback to Massive API if available
        if self.massive_api_key:
            return self.fetch_price_massive(symbol)
        
        return None
    
    def fetch_current_prices(self) -> Tuple[Optional[float], Optional[float]]:
        """
        Fetch current prices for SPY and SPX
        Returns: (spy_price, spx_price)
        
        Note: SPY uses Alpha Vantage (ETF), SPX uses Massive API (index - not supported by Alpha Vantage)
        """
        # Check rate limiting
        now = time.time()
        if self.last_update_time and (now - self.last_update_time) < self.min_update_interval:
            # Too soon, return cached prices
            return self.current_spy_price, self.current_spx_price
        
        # Fetch SPY (ETF - use Alpha Vantage)
        spy_price = self.fetch_price("SPY")
        if spy_price:
            time.sleep(0.5)  # Small delay between calls
        
        # Fetch SPX (Index - use Massive API, Alpha Vantage doesn't support indices)
        spx_price = self.fetch_price("SPX")
        
        # Update state only if we got both prices
        if spy_price and spx_price:
            self.previous_spy_price = self.current_spy_price
            self.previous_spx_price = self.current_spx_price
            self.current_spy_price = spy_price
            self.current_spx_price = spx_price
            self.last_update_time = now
            
            # Store in history
            self.store_price_point(spy_price, spx_price)
        elif spy_price:
            # Got SPY but not SPX - update SPY only (partial update)
            self.current_spy_price = spy_price
        elif spx_price:
            # Got SPX but not SPY - update SPX only (partial update)
            self.current_spx_price = spx_price
        
        return spy_price, spx_price
    
    def calculate_divergence(self, spy_price: float, spx_price: float) -> Dict:
        """
        Calculate divergence between SPY and SPX
        
        Returns:
            {
                'absolute_delta': float,
                'percentage_divergence': float,
                'normalized_spx': float,
                'expected_spx': float,
                'direction': str,  # 'spy_high', 'spx_high', 'converged'
                'is_diverging': bool
            }
        """
        # Normalize SPX to SPY scale
        normalized_spx = spx_price / self.spx_spy_ratio
        expected_spx = spy_price * self.spx_spy_ratio
        
        # Calculate absolute delta
        absolute_delta = abs(normalized_spx - spy_price)
        
        # Calculate percentage divergence
        if spy_price > 0:
            percentage_divergence = ((normalized_spx - spy_price) / spy_price) * 100
        else:
            percentage_divergence = 0.0
        
        # Determine direction
        if abs(percentage_divergence) < self.convergence_threshold:
            direction = "converged"
            is_diverging = False
        elif normalized_spx > spy_price:
            direction = "spx_high"
            is_diverging = abs(percentage_divergence) > self.divergence_threshold
        else:
            direction = "spy_high"
            is_diverging = abs(percentage_divergence) > self.divergence_threshold
        
        return {
            'absolute_delta': absolute_delta,
            'percentage_divergence': percentage_divergence,
            'normalized_spx': normalized_spx,
            'expected_spx': expected_spx,
            'direction': direction,
            'is_diverging': is_diverging
        }
    
    def detect_first_mover(self) -> Optional[Dict]:
        """
        Detect which asset moved first based on price changes
        
        Returns:
            {
                'asset': 'SPY' or 'SPX',
                'spy_change': float,
                'spx_change': float,
                'spy_change_pct': float,
                'spx_change_pct': float,
                'timestamp': float
            } or None if insufficient data
        """
        if not all([self.current_spy_price, self.current_spx_price, 
                   self.previous_spy_price, self.previous_spx_price]):
            return None
        
        # Calculate price changes
        spy_change = self.current_spy_price - self.previous_spy_price
        spx_change = self.current_spx_price - self.previous_spx_price
        
        # Calculate percentage changes
        spy_change_pct = (spy_change / self.previous_spy_price) * 100 if self.previous_spy_price > 0 else 0
        spx_change_pct = (spx_change / self.previous_spx_price) * 100 if self.previous_spx_price > 0 else 0
        
        # Normalize SPX change to SPY scale for comparison
        spx_change_normalized = spx_change / self.spx_spy_ratio
        spx_change_pct_normalized = spx_change_pct  # Percentage is scale-independent
        
        # Determine first mover based on magnitude of change
        # If both moved, the one with larger absolute change (in SPY terms) is the first mover
        if abs(spy_change) > abs(spx_change_normalized):
            first_mover = 'SPY'
        elif abs(spx_change_normalized) > abs(spy_change):
            first_mover = 'SPX'
        else:
            # Equal magnitude, use percentage change
            if abs(spy_change_pct) > abs(spx_change_pct_normalized):
                first_mover = 'SPY'
            else:
                first_mover = 'SPX'
        
        result = {
            'asset': first_mover,
            'spy_change': spy_change,
            'spx_change': spx_change,
            'spy_change_pct': spy_change_pct,
            'spx_change_pct': spx_change_pct,
            'timestamp': time.time()
        }
        
        # Update tracking
        self.last_first_mover = first_mover
        self.last_first_mover_time = time.time()
        
        return result
    
    def store_price_point(self, spy_price: float, spx_price: float):
        """Store price point in history"""
        now = time.time()
        
        # Calculate divergence
        divergence = self.calculate_divergence(spy_price, spx_price)
        
        # Detect first mover
        first_mover = self.detect_first_mover()
        
        # Track divergence duration
        if divergence['is_diverging']:
            if self.divergence_start_time is None:
                self.divergence_start_time = now
        else:
            self.divergence_start_time = None
        
        # Calculate divergence duration
        divergence_duration = None
        if self.divergence_start_time:
            divergence_duration = now - self.divergence_start_time
        
        # Store data point
        data_point = {
            'timestamp': now,
            'datetime': datetime.fromtimestamp(now).isoformat(),
            'spy_price': spy_price,
            'spx_price': spx_price,
            'normalized_spx': divergence['normalized_spx'],
            'divergence': divergence,
            'first_mover': first_mover,
            'divergence_duration': divergence_duration
        }
        
        self.price_history.append(data_point)
        
        # Limit history size
        max_history = self.config.get("max_history_points", 1000)
        if len(self.price_history) > max_history:
            self.price_history = self.price_history[-max_history:]
    
    def get_divergence_history(self, limit: Optional[int] = None) -> List[Dict]:
        """Get divergence history"""
        if limit:
            return self.price_history[-limit:]
        return self.price_history
    
    def get_current_state(self) -> Dict:
        """Get current divergence state"""
        if not self.current_spy_price or not self.current_spx_price:
            return {
                'has_data': False,
                'error': 'No price data available'
            }
        
        divergence = self.calculate_divergence(self.current_spy_price, self.current_spx_price)
        first_mover = self.detect_first_mover()
        
        divergence_duration = None
        if self.divergence_start_time:
            divergence_duration = time.time() - self.divergence_start_time
        
        return {
            'has_data': True,
            'spy_price': self.current_spy_price,
            'spx_price': self.current_spx_price,
            'normalized_spx': divergence['normalized_spx'],
            'divergence': divergence,
            'first_mover': first_mover,
            'divergence_duration': divergence_duration,
            'last_update': self.last_update_time,
            'history_count': len(self.price_history)
        }
