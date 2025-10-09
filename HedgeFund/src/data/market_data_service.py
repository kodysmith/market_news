"""
Module: Market Data Service
Purpose: Fetch and cache market data from multiple sources
Dependencies: Redis (cache), Alpaca API, yfinance (backup)
Exports: MarketDataService

Provides real-time and historical market data with caching for performance.
Automatically fails over to backup data source if primary unavailable.
"""

import yfinance as yf
import redis
from typing import Optional, Dict, List
from datetime import datetime, date, timedelta
from decimal import Decimal
import logging
import json
import numpy as np

logger = logging.getLogger(__name__)


class MarketDataService:
    """
    Market data service with caching and failover
    
    Primary source: Alpaca (when API key available)
    Fallback: yfinance (free, reliable)
    Cache: Redis (5-second TTL for prices)
    """
    
    def __init__(self, config, redis_client: Optional[redis.Redis] = None):
        """
        Initialize market data service
        
        Args:
            config: BrokerConfig from configuration
            redis_client: Redis connection (optional, will create if None)
        """
        self.config = config
        self.redis = redis_client
        self.cache_ttl = 5  # seconds
        
        # Try to initialize Alpaca (if API keys available)
        self.alpaca_available = False
        if config.alpaca_api_key:
            try:
                # Use the correct client for live quotes
                from alpaca.data.live import StockDataStream
                from alpaca.data.requests import StockLatestQuoteRequest
                from alpaca.data import StockHistoricalDataClient
                
                self.alpaca_data = StockHistoricalDataClient(
                    config.alpaca_api_key,
                    config.alpaca_secret_key
                )
                self.alpaca_available = True
                logger.info("Alpaca market data initialized")
            except Exception as e:
                logger.warning(f"Alpaca not available: {e}, using yfinance")
        else:
            logger.info("No Alpaca API keys, using yfinance only")
    
    def get_price(self, symbol: str) -> Decimal:
        """
        Get current price for symbol
        
        Args:
            symbol: Asset symbol (SPY, QQQ, etc.)
        
        Returns:
            Current price as Decimal
        
        Example:
            >>> data_service = MarketDataService(config)
            >>> price = data_service.get_price('QQQ')
            >>> print(f"QQQ: ${price}")
        """
        # Check cache first
        if self.redis:
            try:
                cached = self.redis.get(f"price:{symbol}")
                if cached:
                    return Decimal(cached)
            except:
                pass
        
        # Fetch from primary source
        try:
            if self.alpaca_available:
                price = self._get_price_alpaca(symbol)
            else:
                price = self._get_price_yfinance(symbol)
            
            # Cache the result
            if self.redis and price:
                try:
                    self.redis.setex(f"price:{symbol}", self.cache_ttl, str(price))
                except:
                    pass
            
            return price
            
        except Exception as e:
            logger.error(f"Failed to get price for {symbol}: {e}")
            # Try fallback
            try:
                return self._get_price_yfinance(symbol)
            except:
                raise ValueError(f"Could not get price for {symbol}")
    
    def _get_price_alpaca(self, symbol: str) -> Decimal:
        """Get price from Alpaca"""
        try:
            from alpaca.data.requests import StockLatestQuoteRequest
            
            # Get latest quote using the correct method
            request = StockLatestQuoteRequest(symbol_or_symbols=symbol)
            quotes = self.alpaca_data.get_stock_latest_quote(request)
            
            if symbol in quotes:
                quote = quotes[symbol]
                # Use mid price
                mid_price = (quote.bid_price + quote.ask_price) / 2.0
                return Decimal(str(round(mid_price, 2)))
            else:
                raise ValueError(f"No quote data for {symbol}")
        except Exception as e:
            logger.warning(f"Alpaca quote failed for {symbol}: {e}, falling back to yfinance")
            # Fall back to yfinance
            return self._get_price_yfinance(symbol)
    
    def _get_price_yfinance(self, symbol: str) -> Decimal:
        """Get price from yfinance"""
        ticker = yf.Ticker(symbol)
        
        # Try different methods to get current price
        try:
            # Method 1: info dict
            price = ticker.info.get('regularMarketPrice') or ticker.info.get('currentPrice')
            if price:
                return Decimal(str(round(price, 2)))
        except:
            pass
        
        try:
            # Method 2: fast_info (newer API)
            price = ticker.fast_info.last_price
            if price:
                return Decimal(str(round(price, 2)))
        except:
            pass
        
        try:
            # Method 3: Latest bar
            hist = ticker.history(period='1d')
            if not hist.empty:
                return Decimal(str(round(hist['Close'].iloc[-1], 2)))
        except:
            pass
        
        raise ValueError(f"Could not fetch price for {symbol}")
    
    def get_iv(self, symbol: str) -> Decimal:
        """
        Get implied volatility estimate
        
        Uses realized volatility * 1.1 as proxy for IV
        
        Args:
            symbol: Asset symbol
        
        Returns:
            Implied volatility (annualized)
        """
        # Check cache
        if self.redis:
            try:
                cached = self.redis.get(f"iv:{symbol}")
                if cached:
                    return Decimal(cached)
            except:
                pass
        
        # Calculate from historical volatility
        try:
            ticker = yf.Ticker(symbol)
            hist = ticker.history(period='1mo')
            
            if len(hist) < 20:
                # Not enough data, use default
                iv = 0.25
            else:
                returns = hist['Close'].pct_change().dropna()
                realized_vol = returns.std() * np.sqrt(252)  # Annualized
                iv = realized_vol * 1.1  # Add typical IV skew
            
            # Reasonable bounds
            iv = max(0.10, min(iv, 1.00))
            iv_decimal = Decimal(str(round(iv, 4)))
            
            # Cache
            if self.redis:
                try:
                    self.redis.setex(f"iv:{symbol}", 300, str(iv_decimal))  # 5 min TTL for IV
                except:
                    pass
            
            return iv_decimal
            
        except Exception as e:
            logger.warning(f"Could not calculate IV for {symbol}: {e}, using default")
            return Decimal('0.25')
    
    def get_option_chain(self, symbol: str, expiration: date) -> List[Dict]:
        """
        Get options chain for symbol and expiration
        
        Args:
            symbol: Underlying symbol
            expiration: Expiration date
        
        Returns:
            List of option contracts with prices
        """
        # This would use Alpaca options API in production
        # For now, return empty list (will build when Alpaca account ready)
        logger.info(f"Option chain request: {symbol} {expiration}")
        return []
    
    def get_historical_prices(self, symbol: str, start_date: date, end_date: date) -> Dict[date, Decimal]:
        """
        Get historical prices for backtesting / analysis
        
        Args:
            symbol: Asset symbol
            start_date: Start date
            end_date: End date
        
        Returns:
            Dict mapping date to price
        """
        ticker = yf.Ticker(symbol)
        hist = ticker.history(start=start_date, end=end_date)
        
        return {
            d.date(): Decimal(str(round(price, 2)))
            for d, price in hist['Close'].items()
        }
    
    def health_check(self) -> bool:
        """Check if service is healthy"""
        try:
            # Try to get a price
            price = self.get_price('SPY')
            return price > 0
        except:
            return False


if __name__ == "__main__":
    """Test market data service"""
    import sys
    sys.path.insert(0, '/mnt/4tb/stock_scanner/market_news/HedgeFund')
    
    from src.common.config import load_config
    
    print("Testing Market Data Service...")
    
    config = load_config()
    
    # Initialize (without Redis for now - will work with fallback)
    data_service = MarketDataService(config.broker, redis_client=None)
    
    print(f"\n✓ Service initialized")
    print(f"  Alpaca available: {data_service.alpaca_available}")
    
    # Test price fetching
    print(f"\n📈 Fetching current prices...")
    for symbol in ['SPY', 'QQQ', 'DIA', 'IWM']:
        try:
            price = data_service.get_price(symbol)
            iv = data_service.get_iv(symbol)
            print(f"  {symbol}: ${price:>7} (IV: {iv:.1%})")
        except Exception as e:
            print(f"  {symbol}: Error - {e}")
    
    # Test historical data
    print(f"\n📊 Testing historical data...")
    try:
        end_date = date.today()
        start_date = end_date - timedelta(days=5)
        hist = data_service.get_historical_prices('SPY', start_date, end_date)
        print(f"  Got {len(hist)} days of SPY prices")
        if hist:
            last_date = max(hist.keys())
            print(f"  Latest: {last_date} = ${hist[last_date]}")
    except Exception as e:
        print(f"  Error: {e}")
    
    # Health check
    print(f"\n🏥 Health check: {data_service.health_check()}")
    
    print("\n✅ Market data service working!")

