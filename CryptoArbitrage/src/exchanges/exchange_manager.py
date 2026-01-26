"""
Exchange Manager
Coordinates connections to all configured exchanges
"""

import asyncio
from typing import Dict, List, Optional
from decimal import Decimal
import logging

from ..common.config import AppConfig
from ..common.models import PriceQuote, ExchangeBalance
from .binance_connector import BinanceConnector
from .coinbase_connector import CoinbaseConnector
from .kraken_connector import KrakenConnector
from .exchange_interface import ExchangeInterface

logger = logging.getLogger(__name__)


class ExchangeManager:
    """
    Manages connections to all exchanges
    Provides unified interface for querying prices and executing trades
    """
    
    def __init__(self, config: AppConfig, simulation_mode: bool = False):
        """
        Initialize exchange manager
        
        Args:
            config: Application configuration
            simulation_mode: If True, use mock data (no API keys needed)
        """
        self.config = config
        self.simulation_mode = simulation_mode
        self.exchanges: Dict[str, ExchangeInterface] = {}
        
        logger.info(f"ExchangeManager initializing (simulation={'ON' if simulation_mode else 'OFF'})")
        
        # Initialize exchanges based on config
        self._initialize_exchanges()
    
    def _initialize_exchanges(self):
        """Initialize all configured exchanges"""
        for exchange_name, exchange_config in self.config.exchanges.items():
            if not exchange_config.enabled:
                logger.info(f"Skipping disabled exchange: {exchange_name}")
                continue
            
            try:
                if exchange_name == 'binance':
                    self.exchanges['binance'] = BinanceConnector(
                        exchange_config, 
                        self.simulation_mode
                    )
                elif exchange_name == 'coinbase':
                    self.exchanges['coinbase'] = CoinbaseConnector(
                        exchange_config,
                        self.simulation_mode
                    )
                elif exchange_name == 'kraken':
                    self.exchanges['kraken'] = KrakenConnector(
                        exchange_config,
                        self.simulation_mode
                    )
                else:
                    logger.warning(f"Unknown exchange: {exchange_name}")
                    
            except Exception as e:
                logger.error(f"Failed to initialize {exchange_name}: {e}")
        
        logger.info(f"Initialized {len(self.exchanges)} exchanges: {list(self.exchanges.keys())}")
    
    async def connect_all(self) -> Dict[str, bool]:
        """
        Connect to all exchanges
        
        Returns:
            Dict mapping exchange name to success status
        """
        logger.info("Connecting to all exchanges...")
        
        # Connect to all exchanges in parallel
        tasks = {
            name: exchange.connect()
            for name, exchange in self.exchanges.items()
        }
        
        results = await asyncio.gather(*tasks.values(), return_exceptions=True)
        
        connection_status = {}
        for (name, _), result in zip(tasks.items(), results):
            if isinstance(result, Exception):
                logger.error(f"❌ {name}: Connection failed - {result}")
                connection_status[name] = False
            else:
                status = "✅" if result else "❌"
                logger.info(f"{status} {name}: {'Connected' if result else 'Failed'}")
                connection_status[name] = result
        
        connected_count = sum(1 for v in connection_status.values() if v)
        logger.info(f"Connected to {connected_count}/{len(self.exchanges)} exchanges")
        
        return connection_status
    
    async def disconnect_all(self):
        """Disconnect from all exchanges"""
        logger.info("Disconnecting from all exchanges...")
        
        tasks = [exchange.disconnect() for exchange in self.exchanges.values()]
        await asyncio.gather(*tasks, return_exceptions=True)
        
        logger.info("All exchanges disconnected")
    
    async def subscribe_all_tickers(self, pairs: List[str]):
        """
        Subscribe to ticker updates for all pairs on all exchanges
        
        Args:
            pairs: List of trading pairs to monitor
        """
        logger.info(f"Subscribing to {len(pairs)} pairs on all exchanges...")
        
        tasks = [
            exchange.subscribe_ticker(pairs)
            for exchange in self.exchanges.values()
        ]
        
        await asyncio.gather(*tasks, return_exceptions=True)
        
        logger.info(f"✅ Subscribed to tickers on {len(self.exchanges)} exchanges")
    
    async def get_all_prices(self, pair: str) -> Dict[str, Optional[PriceQuote]]:
        """
        Get current price for a pair from all exchanges
        
        Args:
            pair: Trading pair (e.g., 'BTC/USDT')
        
        Returns:
            Dict mapping exchange name to PriceQuote
        """
        tasks = {
            name: exchange.get_ticker(pair)
            for name, exchange in self.exchanges.items()
        }
        
        results = await asyncio.gather(*tasks.values(), return_exceptions=True)
        
        prices = {}
        for (name, _), result in zip(tasks.items(), results):
            if isinstance(result, Exception):
                logger.error(f"Error getting price for {pair} on {name}: {result}")
                prices[name] = None
            else:
                prices[name] = result
        
        return prices
    
    async def get_all_balances(self) -> Dict[str, ExchangeBalance]:
        """
        Get balances from all exchanges
        
        Returns:
            Dict mapping exchange name to ExchangeBalance
        """
        tasks = {
            name: exchange.get_balance()
            for name, exchange in self.exchanges.items()
        }
        
        results = await asyncio.gather(*tasks.values(), return_exceptions=True)
        
        balances = {}
        for (name, _), result in zip(tasks.items(), results):
            if isinstance(result, Exception):
                logger.error(f"Error getting balance from {name}: {result}")
            else:
                balances[name] = result
        
        return balances
    
    def get_exchange(self, name: str) -> Optional[ExchangeInterface]:
        """Get specific exchange by name"""
        return self.exchanges.get(name)
    
    def get_active_exchanges(self) -> List[str]:
        """Get list of connected exchanges"""
        return [
            name for name, exchange in self.exchanges.items()
            if exchange.is_connected
        ]
    
    async def health_check_all(self) -> Dict[str, Dict]:
        """
        Health check all exchanges
        
        Returns:
            Dict mapping exchange name to health status
        """
        tasks = {
            name: exchange.health_check()
            for name, exchange in self.exchanges.items()
        }
        
        results = await asyncio.gather(*tasks.values(), return_exceptions=True)
        
        health_status = {}
        for (name, _), result in zip(tasks.items(), results):
            if isinstance(result, Exception):
                health_status[name] = {
                    'healthy': False,
                    'error': str(result)
                }
            else:
                health_status[name] = result
        
        return health_status


