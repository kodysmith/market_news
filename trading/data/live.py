"""
Live Data Loader

Loads real-time market data from APIs (yfinance, broker API, etc.).
"""

import yfinance as yf
from typing import Dict, Any, Optional
from datetime import datetime, timedelta
import logging
from .interface import DataLoaderInterface

logger = logging.getLogger(__name__)


class LiveDataLoader(DataLoaderInterface):
    """Loads real-time market data from APIs"""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.cache_duration = config.get('cache_duration', 60)  # Cache for 60 seconds
        self.price_cache = {}
        self.cache_timestamps = {}

    def load_market_data(self, date: Optional[datetime] = None) -> Dict[str, Any]:
        """Load current market data"""
        if date is None:
            date = datetime.now()

        market_data = {}

        # Load SPY data
        spy_data = self._load_ticker_data('SPY', date)
        if spy_data:
            market_data['spy'] = spy_data

        # Load VIX
        vix_data = self._load_vix_data()
        market_data['vix'] = vix_data

        return market_data

    def get_current_prices(self, tickers: list) -> Dict[str, float]:
        """Get current prices for tickers"""
        prices = {}

        for ticker in tickers:
            price = self._get_cached_price(ticker)
            if price is None:
                price = self._fetch_price(ticker)
                self._cache_price(ticker, price)
            prices[ticker] = price

        return prices

    def _load_ticker_data(self, ticker: str, date: datetime) -> Optional[Dict[str, Any]]:
        """Load historical data for a ticker"""
        try:
            stock = yf.Ticker(ticker)
            hist = stock.history(period='1mo')  # Last month

            if len(hist) == 0:
                return None

            # Get last 50 days
            recent = hist.tail(50)

            return {
                'prices': recent['Close'].tolist(),
                'volumes': recent['Volume'].tolist(),
                'current_price': recent['Close'].iloc[-1]
            }

        except Exception as e:
            logger.error(f"Error loading data for {ticker}: {e}")
            return None

    def _load_vix_data(self) -> Dict[str, Any]:
        """Load VIX data"""
        try:
            vix = yf.Ticker('^VIX')
            hist = vix.history(period='5d')

            if len(hist) > 0:
                current_level = hist['Close'].iloc[-1]
                prev_level = hist['Close'].iloc[-2] if len(hist) > 1 else current_level
                change = current_level - prev_level

                return {
                    'level': float(current_level),
                    'change': float(change)
                }
            else:
                return {'level': 20, 'change': 0}

        except Exception as e:
            logger.warning(f"Error loading VIX: {e}")
            return {'level': 20, 'change': 0}

    def _get_cached_price(self, ticker: str) -> Optional[float]:
        """Get cached price if still valid"""
        if ticker in self.price_cache:
            cache_time = self.cache_timestamps.get(ticker)
            if cache_time:
                age = (datetime.now() - cache_time).total_seconds()
                if age < self.cache_duration:
                    return self.price_cache[ticker]
        return None

    def _fetch_price(self, ticker: str) -> float:
        """Fetch current price from API"""
        try:
            stock = yf.Ticker(ticker)
            data = stock.history(period='1d')
            if len(data) > 0:
                return float(data['Close'].iloc[-1])
            else:
                logger.warning(f"No price data for {ticker}")
                return 100.0  # Fallback
        except Exception as e:
            logger.error(f"Error fetching price for {ticker}: {e}")
            return 100.0  # Fallback

    def _cache_price(self, ticker: str, price: float):
        """Cache price with timestamp"""
        self.price_cache[ticker] = price
        self.cache_timestamps[ticker] = datetime.now()
