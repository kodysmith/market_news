"""
Live Data Manager for QuantBot

Integrates real-time data from multiple sources:
- Yahoo Finance (real-time prices, options)
- Alpha Vantage (intraday data, options)
- FMP (Financial Modeling Prep) News API
- Economic data feeds

Provides unified interface for live market data streaming.
"""

import asyncio
import aiohttp
import requests
import json
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta
import logging
import os
from pathlib import Path
import time
from dataclasses import dataclass
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

logger = logging.getLogger(__name__)

@dataclass
class LiveQuote:
    """Real-time quote data structure"""
    symbol: str
    price: float
    volume: int
    timestamp: datetime
    bid: Optional[float] = None
    ask: Optional[float] = None
    change: Optional[float] = None
    change_percent: Optional[float] = None

@dataclass
class NewsItem:
    """News item with sentiment"""
    title: str
    content: str
    url: str
    source: str
    timestamp: datetime
    sentiment_score: Optional[float] = None
    relevance_score: Optional[float] = None
    tickers: List[str] = None

class LiveDataManager:
    """
    Real-time data manager integrating multiple live data sources
    """

    def __init__(self, config: Dict[str, Any]):
        self.config = config

        # API Keys from environment
        self.fmp_api_key = os.getenv('FMP_API_KEY')
        self.alpha_vantage_key = os.getenv('ALPHA_VANTAGE_API_KEY')

        # Rate limiting
        self.yf_last_request = 0
        self.yf_rate_limit = 2.0  # 2 seconds between Yahoo requests
        self.av_last_request = 0
        self.av_rate_limit = 12.0  # Alpha Vantage: ~5 calls/minute
        self.fmp_last_request = 0
        self.fmp_rate_limit = 1.0  # FMP: 1000 calls/day

        # Data cache
        self.price_cache = {}
        self.news_cache = []
        self.options_cache = {}

        # Session management
        self.http_session = None

        logger.info("🔴 LiveDataManager initialized")

    async def initialize(self):
        """Initialize async HTTP session"""
        if self.http_session is None:
            self.http_session = aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=30)
            )
        logger.info("✅ Live data session initialized")

    async def close(self):
        """Close HTTP session"""
        if self.http_session:
            await self.http_session.close()
            self.http_session = None

    async def get_live_quotes(self, symbols: List[str]) -> Dict[str, LiveQuote]:
        """
        Get real-time quotes from Yahoo Finance

        Args:
            symbols: List of stock symbols

        Returns:
            Dictionary of symbol -> LiveQuote
        """
        if not self.http_session:
            await self.initialize()

        quotes = {}

        try:
            # Rate limiting for Yahoo Finance
            current_time = time.time()
            if current_time - self.yf_last_request < self.yf_rate_limit:
                await asyncio.sleep(self.yf_rate_limit - (current_time - self.yf_last_request))

            # Import and use their existing Yahoo Finance integration
            import sys
            sys.path.append('../..')  # Add parent directory to path

            try:
                # Try to import from options_scanner directory
                import sys
                from pathlib import Path

                options_scanner_path = Path(__file__).parent.parent.parent.parent / 'options_scanner'
                if str(options_scanner_path) not in sys.path:
                    sys.path.insert(0, str(options_scanner_path))

                from fetch_yahoo_finance import YahooFinanceFetcher
                yf_fetcher = YahooFinanceFetcher()

                for symbol in symbols:
                    try:
                        # Get real-time quote data
                        quote_data = await asyncio.get_event_loop().run_in_executor(
                            None, yf_fetcher.fetch_quote_data, symbol
                        )

                        if quote_data:
                            quote = LiveQuote(
                                symbol=symbol,
                                price=quote_data.get('regularMarketPrice', 0),
                                volume=quote_data.get('regularMarketVolume', 0),
                                timestamp=datetime.now(),
                                bid=quote_data.get('bid'),
                                ask=quote_data.get('ask'),
                                change=quote_data.get('regularMarketChange'),
                                change_percent=quote_data.get('regularMarketChangePercent')
                            )
                            quotes[symbol] = quote
                            self.price_cache[symbol] = quote

                    except Exception as e:
                        logger.warning(f"Failed to fetch {symbol} from Yahoo: {e}")
                        continue

            except (ImportError, ModuleNotFoundError):
                logger.warning("Yahoo Finance fetcher not available, using fallback")
                # Fallback to yfinance library
                import yfinance as yf

                for symbol in symbols:
                    try:
                        ticker = yf.Ticker(symbol)
                        info = ticker.info

                        quote = LiveQuote(
                            symbol=symbol,
                            price=info.get('currentPrice', info.get('regularMarketPrice', 0)),
                            volume=info.get('volume', info.get('regularMarketVolume', 0)),
                            timestamp=datetime.now(),
                            bid=info.get('bid'),
                            ask=info.get('ask')
                        )
                        quotes[symbol] = quote
                        self.price_cache[symbol] = quote

                    except Exception as e:
                        logger.warning(f"Failed to fetch {symbol} from yfinance: {e}")

            self.yf_last_request = time.time()

        except Exception as e:
            logger.error(f"Error fetching live quotes: {e}")

        return quotes

    async def get_intraday_data(self, symbol: str, interval: str = '5m',
                               bars: int = 100) -> pd.DataFrame:
        """
        Get intraday data from Alpha Vantage

        Args:
            symbol: Stock symbol
            interval: Time interval ('1min', '5min', '15min', '30min', '60min')
            bars: Number of bars to retrieve

        Returns:
            DataFrame with OHLCV data
        """
        if not self.alpha_vantage_key:
            logger.warning("Alpha Vantage API key not found")
            return pd.DataFrame()

        try:
            # Rate limiting
            current_time = time.time()
            if current_time - self.av_last_request < self.av_rate_limit:
                await asyncio.sleep(self.av_rate_limit - (current_time - self.av_last_request))

            # Alpha Vantage intraday API
            url = "https://www.alphavantage.co/query"
            params = {
                "function": "TIME_SERIES_INTRADAY",
                "symbol": symbol,
                "interval": interval,
                "apikey": self.alpha_vantage_key,
                "outputsize": "compact"
            }

            async with self.http_session.get(url, params=params) as response:
                if response.status == 200:
                    data = await response.json()

                    # Parse Alpha Vantage response
                    time_series_key = f"Time Series ({interval})"
                    if time_series_key in data:
                        time_series = data[time_series_key]

                        # Convert to DataFrame
                        df_data = []
                        for timestamp, values in time_series.items():
                            df_data.append({
                                'timestamp': pd.to_datetime(timestamp),
                                'open': float(values['1. open']),
                                'high': float(values['2. high']),
                                'low': float(values['3. low']),
                                'close': float(values['4. close']),
                                'volume': int(values['5. volume'])
                            })

                        df = pd.DataFrame(df_data)
                        df = df.sort_values('timestamp').tail(bars)  # Get most recent bars
                        df.set_index('timestamp', inplace=True)

                        self.av_last_request = time.time()
                        return df

        except Exception as e:
            logger.error(f"Error fetching intraday data for {symbol}: {e}")

        return pd.DataFrame()

    async def get_live_news(self, limit: int = 50) -> List[NewsItem]:
        """
        Get live news from FMP API with sentiment analysis

        Args:
            limit: Maximum number of news items to retrieve

        Returns:
            List of NewsItem objects with sentiment scores
        """
        if not self.fmp_api_key:
            logger.warning("FMP API key not found")
            return []

        try:
            # Rate limiting
            current_time = time.time()
            if current_time - self.fmp_last_request < self.fmp_rate_limit:
                await asyncio.sleep(self.fmp_rate_limit - (current_time - self.fmp_last_request))

            # FMP News API
            url = f"https://financialmodelingprep.com/api/v4/general_news?page=0&limit={limit}&apikey={self.fmp_api_key}"

            async with self.http_session.get(url) as response:
                if response.status == 200:
                    news_data = await response.json()

                    news_items = []
                    for item in news_data:
                        # Extract tickers mentioned in content
                        tickers = self.extract_tickers_from_text(
                            item.get('title', '') + ' ' + item.get('text', '')
                        )

                        # Calculate sentiment score
                        sentiment_score = self.calculate_news_sentiment(
                            item.get('title', '') + ' ' + item.get('text', '')
                        )

                        news_item = NewsItem(
                            title=item.get('title', ''),
                            content=item.get('text', ''),
                            url=item.get('url', ''),
                            source=item.get('site', ''),
                            timestamp=pd.to_datetime(item.get('publishedDate', datetime.now())),
                            sentiment_score=sentiment_score,
                            tickers=tickers
                        )
                        news_items.append(news_item)

                    # Cache news items
                    self.news_cache = news_items[:100]  # Keep last 100 items

                    self.fmp_last_request = time.time()
                    return news_items

        except Exception as e:
            logger.error(f"Error fetching live news: {e}")

        return []

    async def get_options_data(self, underlying: str) -> Dict[str, Any]:
        """
        Get live options data from Yahoo Finance

        Args:
            underlying: Underlying stock symbol

        Returns:
            Options chain data
        """
        try:
            # Use existing Yahoo Finance options fetcher
            import sys
            sys.path.append('../..')

            from options_scanner.fetch_yahoo_finance import YahooFinanceFetcher
            fetcher = YahooFinanceFetcher()

            # Get options data
            options_data = await asyncio.get_event_loop().run_in_executor(
                None, fetcher.fetch_options_data, underlying
            )

            if options_data:
                self.options_cache[underlying] = {
                    'data': options_data,
                    'timestamp': datetime.now()
                }

            return options_data or {}

        except Exception as e:
            logger.error(f"Error fetching options data for {underlying}: {e}")
            return {}

    def extract_tickers_from_text(self, text: str) -> List[str]:
        """Extract stock tickers from text using regex patterns"""
        import re

        # Common ticker patterns
        ticker_pattern = r'\b[A-Z]{1,5}\b'

        # Find potential tickers
        potential_tickers = re.findall(ticker_pattern, text.upper())

        # Filter out common words that might match
        exclude_words = {'THE', 'AND', 'FOR', 'ARE', 'BUT', 'NOT', 'YOU', 'ALL', 'CAN', 'HER', 'WAS', 'ONE', 'OUR', 'HAD', 'BY', 'HOT', 'HOW', 'ITS', 'WHO', 'DID', 'NEW', 'NOW', 'OLD', 'SEE', 'WAY', 'WEB', 'WHO', 'WOW', 'YES', 'YET'}

        tickers = [t for t in potential_tickers if t not in exclude_words and len(t) >= 2]

        return tickers

    def calculate_news_sentiment(self, text: str) -> float:
        """
        Calculate sentiment score for news text

        Returns:
            Float between -1 (very negative) and 1 (very positive)
        """
        try:
            # Simple sentiment analysis based on word lists
            positive_words = {
                'bull', 'bullish', 'buy', 'strong', 'gains', 'rally', 'surge', 'boost',
                'profit', 'growth', 'rise', 'higher', 'beat', 'upgrade', 'positive',
                'win', 'success', 'breakthrough', 'record', 'soar', 'climb'
            }

            negative_words = {
                'bear', 'bearish', 'sell', 'weak', 'losses', 'drop', 'fall', 'decline',
                'loss', 'crash', 'plunge', 'slump', 'downgrade', 'negative', 'concern',
                'risk', 'fail', 'disaster', 'bankruptcy', 'lawsuit', 'scandal'
            }

            text_lower = text.lower()
            words = text_lower.split()

            positive_count = sum(1 for word in words if word in positive_words)
            negative_count = sum(1 for word in words if word in negative_words)

            total_sentiment_words = positive_count + negative_count

            if total_sentiment_words == 0:
                return 0.0

            # Normalize to [-1, 1]
            sentiment_score = (positive_count - negative_count) / total_sentiment_words

            return sentiment_score

        except Exception as e:
            logger.warning(f"Error calculating news sentiment: {e}")
            return 0.0

    async def get_economic_data(self) -> Dict[str, Any]:
        """
        Get economic indicators (GDP, unemployment, inflation, etc.)

        Returns:
            Dictionary of economic indicators with calendar event data
        """
        economic_data = {}

        try:
            # FRED API for economic data (if available)
            fred_api_key = os.getenv('FRED_API_KEY')

            if fred_api_key:
                indicators = [
                    ('GDP', 'GDP'),
                    ('UNRATE', 'Unemployment Rate'),
                    ('CPIAUCSL', 'Consumer Price Index'),
                    ('FEDFUNDS', 'Federal Funds Rate'),
                    ('PAYEMS', 'Non-Farm Payrolls'),
                    ('INDPRO', 'Industrial Production'),
                    ('HOUST', 'Housing Starts'),
                    ('DGS10', '10-Year Treasury Rate')
                ]

                for series_id, name in indicators:
                    try:
                        url = f"https://api.stlouisfed.org/fred/series/observations"
                        params = {
                            'series_id': series_id,
                            'api_key': fred_api_key,
                            'file_type': 'json',
                            'limit': 5,  # Get last 5 observations for trend
                            'sort_order': 'desc'
                        }

                        async with self.http_session.get(url, params=params) as response:
                            if response.status == 200:
                                data = await response.json()
                                if data.get('observations') and len(data['observations']) > 0:
                                    latest = data['observations'][0]  # Most recent first
                                    prev = data['observations'][1] if len(data['observations']) > 1 else None

                                    value = float(latest.get('value', 0))
                                    date = latest.get('date', '')

                                    # Calculate change if previous value available
                                    change = None
                                    if prev and prev.get('value'):
                                        prev_value = float(prev.get('value', 0))
                                        change = value - prev_value

                                    economic_data[series_id.lower()] = {
                                        'name': name,
                                        'value': value,
                                        'date': date,
                                        'change': change,
                                        'series_id': series_id
                                    }

                    except Exception as e:
                        logger.warning(f"Failed to fetch {series_id}: {e}")

            # Enhanced fallback data with calendar-style events
            if not economic_data:
                from datetime import datetime, timedelta
                base_date = datetime.now()

                economic_data = {
                    'gdp': {
                        'name': 'GDP',
                        'value': 28953.5,
                        'date': (base_date - timedelta(days=90)).strftime('%Y-%m-%d'),
                        'change': 1.2,
                        'series_id': 'GDP'
                    },
                    'unrate': {
                        'name': 'Unemployment Rate',
                        'value': 4.2,
                        'date': (base_date - timedelta(days=30)).strftime('%Y-%m-%d'),
                        'change': -0.1,
                        'series_id': 'UNRATE'
                    },
                    'cpiaucsl': {
                        'name': 'Consumer Price Index',
                        'value': 314.5,
                        'date': (base_date - timedelta(days=30)).strftime('%Y-%m-%d'),
                        'change': 0.3,
                        'series_id': 'CPIAUCSL'
                    },
                    'fedfunds': {
                        'name': 'Federal Funds Rate',
                        'value': 5.25,
                        'date': (base_date - timedelta(days=30)).strftime('%Y-%m-%d'),
                        'change': 0.0,
                        'series_id': 'FEDFUNDS'
                    },
                    'payems': {
                        'name': 'Non-Farm Payrolls',
                        'value': 275000,
                        'date': (base_date - timedelta(days=30)).strftime('%Y-%m-%d'),
                        'change': 25000,
                        'series_id': 'PAYEMS'
                    }
                }

        except Exception as e:
            logger.error(f"Error fetching economic data: {e}")

        return economic_data

    async def get_economic_calendar_async(self) -> List[Dict[str, Any]]:
        """
        Get economic calendar events for display (async version)

        Returns:
            List of calendar events with dates, titles, and impacts
        """
        calendar_events = []

        try:
            # Get the economic data we have
            economic_data = await self.get_economic_data()

            # Convert to calendar format
            for key, data in economic_data.items():
                if isinstance(data, dict):
                    event = {
                        'id': f"econ_{data.get('series_id', key)}_{data.get('date', 'unknown')}",
                        'title': f"{data.get('name', key)}: {data.get('value', 'N/A')}",
                        'date': data.get('date', datetime.now().strftime('%Y-%m-%d')),
                        'time': '08:30',  # Default economic data release time
                        'impact': self._determine_economic_impact(data),
                        'actual': data.get('value'),
                        'previous': data.get('value') - data.get('change', 0) if data.get('change') is not None else None,
                        'forecast': None,  # Would need additional API for forecasts
                        'currency': 'USD',
                        'source': 'FRED' if os.getenv('FRED_API_KEY') else 'Sample Data'
                    }
                    calendar_events.append(event)

            # Sort by date (most recent first)
            calendar_events.sort(key=lambda x: x['date'], reverse=True)

        except Exception as e:
            logger.error(f"Error creating economic calendar: {e}")

        return calendar_events

    def get_economic_calendar(self) -> List[Dict[str, Any]]:
        """
        Get economic calendar events for display (synchronous wrapper)

        Returns:
            List of calendar events with dates, titles, and impacts
        """
        try:
            # Try to get the current event loop
            loop = asyncio.get_event_loop()
            if loop.is_running():
                # If loop is already running, we can't use asyncio.run()
                # Fall back to sample data
                return self._get_sample_calendar_events()
            else:
                # Use asyncio.run() if no loop is running
                return asyncio.run(self.get_economic_calendar_async())
        except RuntimeError:
            # No event loop, use asyncio.run()
            return asyncio.run(self.get_economic_calendar_async())

    def _get_sample_calendar_events(self) -> List[Dict[str, Any]]:
        """Get sample calendar events when async is not available"""
        from datetime import datetime, timedelta

        base_date = datetime.now()
        return [
            {
                'id': 'sample_gdp',
                'title': 'GDP: 28953.5',
                'date': (base_date - timedelta(days=90)).strftime('%Y-%m-%d'),
                'time': '08:30',
                'impact': 'high',
                'actual': 28953.5,
                'previous': 28750.2,
                'forecast': None,
                'currency': 'USD',
                'source': 'Sample Data'
            },
            {
                'id': 'sample_unrate',
                'title': 'Unemployment Rate: 4.2',
                'date': (base_date - timedelta(days=30)).strftime('%Y-%m-%d'),
                'time': '08:30',
                'impact': 'medium',
                'actual': 4.2,
                'previous': 4.3,
                'forecast': None,
                'currency': 'USD',
                'source': 'Sample Data'
            },
            {
                'id': 'sample_payems',
                'title': 'Non-Farm Payrolls: 275000',
                'date': (base_date - timedelta(days=30)).strftime('%Y-%m-%d'),
                'time': '08:30',
                'impact': 'high',
                'actual': 275000,
                'previous': 250000,
                'forecast': None,
                'currency': 'USD',
                'source': 'Sample Data'
            }
        ]

    def _determine_economic_impact(self, data: Dict[str, Any]) -> str:
        """Determine the market impact level of an economic indicator"""
        name = data.get('name', '').lower()
        change = data.get('change')

        # High impact indicators
        high_impact = ['gdp', 'non-farm payrolls', 'federal funds rate', 'consumer price index']
        if any(term in name.lower() for term in high_impact):
            return 'high'

        # Medium impact indicators
        medium_impact = ['unemployment', 'industrial production', 'housing starts']
        if any(term in name.lower() for term in medium_impact):
            return 'medium'

        # Check for significant changes
        if change is not None:
            if abs(change) > 0.5:  # Significant change
                return 'medium'

        return 'low'

    def get_cached_quote(self, symbol: str) -> Optional[LiveQuote]:
        """Get cached quote if available and not stale"""
        if symbol in self.price_cache:
            cached_quote = self.price_cache[symbol]
            cache_age = datetime.now() - cached_quote.timestamp

            # Cache valid for 5 minutes
            if cache_age.seconds < 300:
                return cached_quote

        return None

    def get_recent_news_sentiment(self, symbol: str = None, hours: int = 24) -> float:
        """
        Get aggregated news sentiment for a symbol or market overall

        Args:
            symbol: Specific symbol, or None for market sentiment
            hours: Lookback period in hours

        Returns:
            Average sentiment score
        """
        cutoff_time = datetime.now().replace(tzinfo=None) - timedelta(hours=hours)

        relevant_news = [
            item for item in self.news_cache
            if item.timestamp.replace(tzinfo=None) >= cutoff_time and
            (symbol is None or symbol.upper() in [t.upper() for t in (item.tickers or [])])
        ]

        if not relevant_news:
            return 0.0

        sentiment_scores = [item.sentiment_score for item in relevant_news if item.sentiment_score is not None]

        if not sentiment_scores:
            return 0.0

        return np.mean(sentiment_scores)

    async def update_real_time_data(self, symbols: List[str]) -> bool:
        """Update real-time data cache for given symbols"""
        try:
            # Refresh quotes for the universe
            quotes = await self.get_live_quotes(symbols)

            # Update cache with fresh data
            for symbol, quote in quotes.items():
                if quote:
                    self.quote_cache[symbol] = {
                        'quote': quote,
                        'timestamp': datetime.now()
                    }

            logger.info(f"✅ Updated real-time data for {len(quotes)} symbols")
            return True

        except Exception as e:
            logger.error(f"Failed to update real-time data: {e}")
            return False

    async def get_market_overview(self) -> Dict[str, Any]:
        """
        Get comprehensive market overview with live data

        Returns:
            Complete market state snapshot
        """
        try:
            # Major indices
            indices = ['SPY', 'QQQ', 'IWM', 'VXX']
            quotes = await self.get_live_quotes(indices)

            # Economic data
            economic_data = await self.get_economic_data()

            # Recent news sentiment
            market_sentiment = self.get_recent_news_sentiment(hours=24)

            # VIX for volatility
            vix_quote = quotes.get('VXX', {})
            volatility_level = vix_quote.price if hasattr(vix_quote, 'price') else 20.0

            market_overview = {
                'timestamp': datetime.now().isoformat(),
                'indices': {
                    symbol: {
                        'price': quote.price if hasattr(quote, 'price') else 0,
                        'change': quote.change if hasattr(quote, 'change') else 0,
                        'change_percent': quote.change_percent if hasattr(quote, 'change_percent') else 0
                    } for symbol, quote in quotes.items()
                },
                'economic_indicators': economic_data,
                'market_sentiment': market_sentiment,
                'volatility_level': volatility_level,
                'data_quality': 'live' if quotes else 'fallback'
            }

            return market_overview

        except Exception as e:
            logger.error(f"Error getting market overview: {e}")
            return {
                'timestamp': datetime.now().isoformat(),
                'error': str(e),
                'data_quality': 'error'
            }
