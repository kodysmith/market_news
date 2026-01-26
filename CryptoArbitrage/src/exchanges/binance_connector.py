"""
Binance Exchange Connector
CCXT-based implementation with websocket support
"""

import asyncio
from typing import Dict, List, Optional
from decimal import Decimal
from datetime import datetime
import logging
import uuid

from ..common.models import CryptoOrder, PriceQuote, ExchangeBalance, OrderStatus, OrderSide, OrderType
from ..common.config import ExchangeConfig
from .exchange_interface import (
    ExchangeInterface, ExchangeError, InsufficientBalanceError, OrderRejectedError
)

logger = logging.getLogger(__name__)

# Try to import CCXT
try:
    import ccxt.async_support as ccxt
    import ccxt.pro as ccxtpro
    CCXT_AVAILABLE = True
except ImportError:
    CCXT_AVAILABLE = False
    logger.warning("CCXT not installed. Install with: pip install ccxt")


class BinanceConnector(ExchangeInterface):
    """
    Binance exchange connector using CCXT
    
    Supports:
    - REST API for trading
    - Websocket for real-time prices
    - Simulation mode (no API keys needed)
    """
    
    def __init__(self, config: ExchangeConfig, simulation_mode: bool = False):
        """
        Initialize Binance connector
        
        Args:
            config: Exchange configuration
            simulation_mode: If True, use mock data instead of real API
        """
        self.config = config
        self.simulation_mode = simulation_mode
        self._name = 'binance'
        self._is_connected = False
        
        # CCXT clients
        self.exchange: Optional[ccxt.binance] = None
        self.exchange_ws: Optional[ccxtpro.binance] = None
        
        # Price cache from websocket
        self.price_cache: Dict[str, PriceQuote] = {}
        
        # Simulation data
        self.simulated_balances = {
            'USDT': Decimal('5000'),
            'BTC': Decimal('0'),
            'ETH': Decimal('0')
        }
        
        logger.info(f"Binance connector initialized (simulation={'ON' if simulation_mode else 'OFF'})")
    
    async def connect(self) -> bool:
        """Establish connection to Binance"""
        try:
            if not CCXT_AVAILABLE:
                if self.simulation_mode:
                    logger.info("Binance: Simulation mode (CCXT not available)")
                    self._is_connected = True
                    return True
                else:
                    logger.error("CCXT not installed")
                    return False
            
            if self.simulation_mode:
                # Simulation mode - no API keys needed
                self.exchange = ccxt.binance({
                    'enableRateLimit': True,
                })
                self._is_connected = True
                logger.info("✅ Binance connected (simulation mode)")
                return True
            
            # Real mode - need API keys
            if not self.config.api_key:
                logger.warning("Binance API keys not configured - using simulation")
                self.simulation_mode = True
                self._is_connected = True
                return True
            
            self.exchange = ccxt.binance({
                'apiKey': self.config.api_key,
                'secret': self.config.api_secret,
                'enableRateLimit': True,
                'options': {
                    'defaultType': 'spot',
                }
            })
            
            # Test connection
            await self.exchange.load_markets()
            
            # Initialize websocket client for real-time prices
            self.exchange_ws = ccxtpro.binance({
                'apiKey': self.config.api_key,
                'secret': self.config.api_secret,
                'enableRateLimit': True,
            })
            
            self._is_connected = True
            logger.info("✅ Binance connected (live mode)")
            return True
            
        except Exception as e:
            logger.error(f"Failed to connect to Binance: {e}")
            self._is_connected = False
            return False
    
    async def disconnect(self):
        """Close connections"""
        if self.exchange:
            await self.exchange.close()
        if self.exchange_ws:
            await self.exchange_ws.close()
        self._is_connected = False
        logger.info("Binance disconnected")
    
    async def subscribe_ticker(self, pairs: List[str]):
        """Subscribe to real-time ticker updates"""
        if self.simulation_mode or not self.exchange_ws:
            logger.info(f"Binance: Simulating ticker subscription for {len(pairs)} pairs")
            # In simulation, we'll generate mock prices
            return
        
        try:
            # Subscribe to ticker stream for each pair
            for pair in pairs:
                # Start background task to watch ticker
                asyncio.create_task(self._watch_ticker(pair))
            
            logger.info(f"✅ Binance: Subscribed to {len(pairs)} ticker feeds")
            
        except Exception as e:
            logger.error(f"Binance ticker subscription failed: {e}")
    
    async def _watch_ticker(self, pair: str):
        """Background task to watch ticker updates"""
        while self._is_connected and self.exchange_ws:
            try:
                ticker = await self.exchange_ws.watch_ticker(pair)
                
                # Update cache
                self.price_cache[pair] = PriceQuote(
                    exchange=self._name,
                    pair=pair,
                    bid=Decimal(str(ticker['bid'])) if ticker['bid'] else Decimal('0'),
                    ask=Decimal(str(ticker['ask'])) if ticker['ask'] else Decimal('0'),
                    mid=Decimal(str(ticker['last'])) if ticker['last'] else Decimal('0'),
                    timestamp=datetime.fromtimestamp(ticker['timestamp'] / 1000),
                    latency_ms=0.0,  # Calculate if needed
                    volume_24h=Decimal(str(ticker.get('quoteVolume', 0)))
                )
                
            except Exception as e:
                logger.error(f"Error watching {pair} on Binance: {e}")
                await asyncio.sleep(1)
                break
    
    async def get_ticker(self, pair: str) -> Optional[PriceQuote]:
        """Get current ticker"""
        # Check cache first (from websocket)
        if pair in self.price_cache:
            quote = self.price_cache[pair]
            # Check if stale (>5 seconds old)
            age = (datetime.now() - quote.timestamp).total_seconds()
            if age < 5:
                return quote
        
        # Fall back to REST API
        if self.simulation_mode or not self.exchange:
            # Return mock price
            return self._generate_mock_price(pair)
        
        try:
            ticker = await self.exchange.fetch_ticker(pair)
            
            quote = PriceQuote(
                exchange=self._name,
                pair=pair,
                bid=Decimal(str(ticker['bid'])) if ticker['bid'] else Decimal('0'),
                ask=Decimal(str(ticker['ask'])) if ticker['ask'] else Decimal('0'),
                mid=Decimal(str(ticker['last'])) if ticker['last'] else Decimal('0'),
                timestamp=datetime.fromtimestamp(ticker['timestamp'] / 1000),
                latency_ms=0.0,
                volume_24h=Decimal(str(ticker.get('quoteVolume', 0)))
            )
            
            # Cache it
            self.price_cache[pair] = quote
            
            return quote
            
        except Exception as e:
            logger.error(f"Failed to get ticker for {pair} on Binance: {e}")
            return None
    
    async def get_balance(self, currency: Optional[str] = None) -> ExchangeBalance:
        """Get account balance"""
        if self.simulation_mode or not self.exchange:
            # Return simulated balance
            return ExchangeBalance(
                exchange=self._name,
                timestamp=datetime.now(),
                balances=self.simulated_balances.copy(),
                total_usd=sum(self.simulated_balances.values()),
                available=self.simulated_balances.copy(),
                locked={}
            )
        
        try:
            balance = await self.exchange.fetch_balance()
            
            balances_dict = {}
            available_dict = {}
            locked_dict = {}
            
            for currency, amount in balance['total'].items():
                if amount > 0:
                    balances_dict[currency] = Decimal(str(amount))
                    available_dict[currency] = Decimal(str(balance['free'].get(currency, 0)))
                    locked_dict[currency] = Decimal(str(balance['used'].get(currency, 0)))
            
            # Calculate USD total (simplified - would need price conversion)
            total_usd = balances_dict.get('USDT', Decimal('0'))
            
            return ExchangeBalance(
                exchange=self._name,
                timestamp=datetime.now(),
                balances=balances_dict,
                total_usd=total_usd,
                available=available_dict,
                locked=locked_dict
            )
            
        except Exception as e:
            logger.error(f"Failed to get balance from Binance: {e}")
            raise ExchangeError(f"Balance query failed: {e}")
    
    async def submit_market_order(self,
                                  pair: str,
                                  side: str,
                                  quantity: Decimal) -> CryptoOrder:
        """Submit market order"""
        order_id = f"BIN-{uuid.uuid4().hex[:12]}"
        
        if self.simulation_mode or not self.exchange:
            logger.info(f"📝 Binance SIMULATION: {side.upper()} {quantity} {pair} @ MARKET")
            
            # Simulate order
            order = CryptoOrder(
                id=order_id,
                exchange=self._name,
                pair=pair,
                side=OrderSide.BUY if side.lower() == 'buy' else OrderSide.SELL,
                order_type=OrderType.MARKET,
                quantity=quantity,
                price=None,
                created_at=datetime.now(),
                status=OrderStatus.FILLED,  # Instant fill in simulation
                exchange_order_id=f"SIM-{order_id}",
                submitted_at=datetime.now(),
                filled_at=datetime.now(),
                filled_quantity=quantity,
                filled_price=Decimal('43000'),  # Mock price
                fee_paid=quantity * Decimal('0.001'),  # 0.1% fee
                fee_currency='USDT'
            )
            
            # Update simulated balance
            self._update_simulated_balance(pair, side, quantity)
            
            return order
        
        try:
            # Submit real order
            result = await self.exchange.create_market_order(
                symbol=pair,
                side=side.lower(),
                amount=float(quantity)
            )
            
            order = CryptoOrder(
                id=order_id,
                exchange=self._name,
                pair=pair,
                side=OrderSide.BUY if side.lower() == 'buy' else OrderSide.SELL,
                order_type=OrderType.MARKET,
                quantity=quantity,
                price=None,
                created_at=datetime.now(),
                status=OrderStatus.SUBMITTED,
                exchange_order_id=result['id'],
                submitted_at=datetime.now()
            )
            
            logger.info(f"✅ Binance order submitted: {order_id}")
            
            return order
            
        except ccxt.InsufficientFunds as e:
            logger.error(f"Insufficient funds on Binance: {e}")
            raise InsufficientBalanceError(str(e))
        except Exception as e:
            logger.error(f"Binance order failed: {e}")
            raise OrderRejectedError(str(e))
    
    async def cancel_order(self, order_id: str, pair: str) -> bool:
        """Cancel order"""
        if self.simulation_mode:
            logger.info(f"Binance SIMULATION: Cancelled order {order_id}")
            return True
        
        try:
            await self.exchange.cancel_order(order_id, pair)
            logger.info(f"✅ Binance order cancelled: {order_id}")
            return True
        except Exception as e:
            logger.error(f"Failed to cancel Binance order: {e}")
            return False
    
    async def get_order_status(self, order_id: str, pair: str) -> CryptoOrder:
        """Get order status"""
        if self.simulation_mode:
            # Return mock filled order
            return CryptoOrder(
                id=order_id,
                exchange=self._name,
                pair=pair,
                side=OrderSide.BUY,
                order_type=OrderType.MARKET,
                quantity=Decimal('1'),
                price=None,
                created_at=datetime.now(),
                status=OrderStatus.FILLED
            )
        
        try:
            order_data = await self.exchange.fetch_order(order_id, pair)
            
            # Parse status
            status_map = {
                'open': OrderStatus.PENDING,
                'closed': OrderStatus.FILLED,
                'canceled': OrderStatus.CANCELLED,
                'rejected': OrderStatus.REJECTED
            }
            
            return CryptoOrder(
                id=order_id,
                exchange=self._name,
                pair=pair,
                side=OrderSide.BUY if order_data['side'] == 'buy' else OrderSide.SELL,
                order_type=OrderType.MARKET if order_data['type'] == 'market' else OrderType.LIMIT,
                quantity=Decimal(str(order_data['amount'])),
                price=Decimal(str(order_data['price'])) if order_data['price'] else None,
                created_at=datetime.fromtimestamp(order_data['timestamp'] / 1000),
                status=status_map.get(order_data['status'], OrderStatus.PENDING),
                filled_quantity=Decimal(str(order_data.get('filled', 0))),
                filled_price=Decimal(str(order_data.get('average', 0))) if order_data.get('average') else None
            )
            
        except Exception as e:
            logger.error(f"Failed to get order status from Binance: {e}")
            raise ExchangeError(f"Order status query failed: {e}")
    
    def get_fee_structure(self) -> Dict[str, Decimal]:
        """Get fee structure"""
        return {
            'maker_fee': self.config.maker_fee_pct,
            'taker_fee': self.config.taker_fee_pct
        }
    
    async def health_check(self) -> Dict[str, any]:
        """Health check"""
        try:
            if self.simulation_mode:
                return {
                    'healthy': True,
                    'connected': True,
                    'mode': 'simulation',
                    'cached_pairs': len(self.price_cache)
                }
            
            if not self.exchange:
                return {
                    'healthy': False,
                    'connected': False,
                    'error': 'Not initialized'
                }
            
            # Ping exchange
            status = await self.exchange.fetch_status()
            
            return {
                'healthy': status['status'] == 'ok',
                'connected': self._is_connected,
                'mode': 'live',
                'exchange_status': status['status'],
                'cached_pairs': len(self.price_cache)
            }
            
        except Exception as e:
            logger.error(f"Binance health check failed: {e}")
            return {
                'healthy': False,
                'connected': False,
                'error': str(e)
            }
    
    @property
    def name(self) -> str:
        """Exchange name"""
        return self._name
    
    @property
    def is_connected(self) -> bool:
        """Is connected"""
        return self._is_connected
    
    def _generate_mock_price(self, pair: str) -> PriceQuote:
        """Generate mock price for simulation"""
        # Simple mock prices
        base_prices = {
            'BTC/USDT': 43000,
            'ETH/USDT': 2300,
            'SOL/USDT': 100,
            'BNB/USDT': 310,
        }
        
        base_price = Decimal(str(base_prices.get(pair, 100)))
        
        # Add small spread
        spread = base_price * Decimal('0.0002')  # 0.02% spread
        
        return PriceQuote(
            exchange=self._name,
            pair=pair,
            bid=base_price - spread,
            ask=base_price + spread,
            mid=base_price,
            timestamp=datetime.now(),
            latency_ms=10.0,
            volume_24h=Decimal('10000000')
        )
    
    def _update_simulated_balance(self, pair: str, side: str, quantity: Decimal):
        """Update simulated balances after trade"""
        base, quote = pair.split('/')
        
        if side.lower() == 'buy':
            # Buying base, spending quote
            self.simulated_balances[base] = self.simulated_balances.get(base, Decimal('0')) + quantity
            cost = quantity * Decimal('43000')  # Mock price
            self.simulated_balances[quote] = self.simulated_balances.get(quote, Decimal('0')) - cost
        else:
            # Selling base, receiving quote
            self.simulated_balances[base] = self.simulated_balances.get(base, Decimal('0')) - quantity
            proceeds = quantity * Decimal('43000')  # Mock price
            self.simulated_balances[quote] = self.simulated_balances.get(quote, Decimal('0')) + proceeds


