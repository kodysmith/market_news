"""
Real-Time Price Aggregator
Collects and normalizes prices from all exchanges
"""

import asyncio
from typing import Dict, List, Optional
from decimal import Decimal
from datetime import datetime, timedelta
from collections import defaultdict
import logging

from ..common.models import PriceQuote
from ..exchanges.exchange_manager import ExchangeManager

logger = logging.getLogger(__name__)


class PriceAggregator:
    """
    Real-time price aggregation from multiple exchanges
    
    Maintains in-memory cache of latest prices from all exchanges
    for fast arbitrage detection.
    """
    
    def __init__(self, exchange_manager: ExchangeManager, max_staleness_seconds: int = 5):
        """
        Initialize price aggregator
        
        Args:
            exchange_manager: Manages all exchange connections
            max_staleness_seconds: Maximum age of price data before considered stale
        """
        self.exchange_manager = exchange_manager
        self.max_staleness = timedelta(seconds=max_staleness_seconds)
        
        # Price cache: {pair: {exchange: PriceQuote}}
        self.price_cache: Dict[str, Dict[str, PriceQuote]] = defaultdict(dict)
        
        # Statistics
        self.updates_received = 0
        self.stale_prices_detected = 0
        
        logger.info("PriceAggregator initialized")
    
    async def start_monitoring(self, pairs: List[str]):
        """
        Start monitoring prices for all pairs
        
        Args:
            pairs: List of trading pairs to monitor
        """
        logger.info(f"Starting price monitoring for {len(pairs)} pairs...")
        
        # Subscribe to websocket feeds on all exchanges
        await self.exchange_manager.subscribe_all_tickers(pairs)
        
        # Start background task to periodically refresh prices
        asyncio.create_task(self._refresh_prices_loop(pairs))
        
        logger.info("✅ Price monitoring started")
    
    async def _refresh_prices_loop(self, pairs: List[str]):
        """Background task to refresh prices every few seconds"""
        while True:
            try:
                for pair in pairs:
                    # Get prices from all exchanges
                    prices = await self.exchange_manager.get_all_prices(pair)
                    
                    # Update cache
                    for exchange_name, quote in prices.items():
                        if quote:
                            self.price_cache[pair][exchange_name] = quote
                            self.updates_received += 1
                
                # Wait before next refresh
                await asyncio.sleep(1)  # Refresh every second
                
            except Exception as e:
                logger.error(f"Error in price refresh loop: {e}")
                await asyncio.sleep(5)
    
    def get_prices(self, pair: str) -> Dict[str, Optional[PriceQuote]]:
        """
        Get current prices for a pair from all exchanges
        
        Args:
            pair: Trading pair
        
        Returns:
            Dict mapping exchange name to PriceQuote (or None if stale/unavailable)
        """
        if pair not in self.price_cache:
            return {}
        
        prices = {}
        now = datetime.now()
        
        for exchange_name, quote in self.price_cache[pair].items():
            # Check if price is stale
            age = now - quote.timestamp
            
            if age > self.max_staleness:
                self.stale_prices_detected += 1
                logger.debug(f"Stale price for {pair} on {exchange_name}: {age.total_seconds():.1f}s old")
                prices[exchange_name] = None
            else:
                prices[exchange_name] = quote
        
        return prices
    
    def get_all_pairs_prices(self) -> Dict[str, Dict[str, PriceQuote]]:
        """
        Get all prices for all monitored pairs
        
        Returns:
            Dict mapping pair to dict of exchange prices
        """
        all_prices = {}
        
        for pair in self.price_cache.keys():
            prices = self.get_prices(pair)
            # Only include if we have at least 2 valid prices
            valid_prices = {k: v for k, v in prices.items() if v is not None}
            if len(valid_prices) >= 2:
                all_prices[pair] = valid_prices
        
        return all_prices
    
    def get_statistics(self) -> Dict:
        """Get aggregator statistics"""
        total_pairs = len(self.price_cache)
        total_quotes = sum(len(exchanges) for exchanges in self.price_cache.values())
        
        # Count fresh vs stale
        fresh_count = 0
        stale_count = 0
        now = datetime.now()
        
        for pair_prices in self.price_cache.values():
            for quote in pair_prices.values():
                if (now - quote.timestamp) <= self.max_staleness:
                    fresh_count += 1
                else:
                    stale_count += 1
        
        return {
            'monitored_pairs': total_pairs,
            'total_quotes_cached': total_quotes,
            'fresh_quotes': fresh_count,
            'stale_quotes': stale_count,
            'updates_received': self.updates_received,
            'freshness_rate': fresh_count / total_quotes if total_quotes > 0 else 0
        }


