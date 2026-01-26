"""
Coinbase Pro Exchange Connector
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

try:
    import ccxt.async_support as ccxt
    import ccxt.pro as ccxtpro
    CCXT_AVAILABLE = True
except ImportError:
    CCXT_AVAILABLE = False
    logger.warning("CCXT not installed")


class CoinbaseConnector(ExchangeInterface):
    """Coinbase Pro exchange connector"""
    
    def __init__(self, config: ExchangeConfig, simulation_mode: bool = False):
        self.config = config
        self.simulation_mode = simulation_mode
        self._name = 'coinbase'
        self._is_connected = False
        
        self.exchange: Optional[ccxt.coinbasepro] = None
        self.exchange_ws: Optional[ccxtpro.coinbasepro] = None
        self.price_cache: Dict[str, PriceQuote] = {}
        
        self.simulated_balances = {
            'USD': Decimal('5000'),
            'BTC': Decimal('0'),
            'ETH': Decimal('0')
        }
        
        logger.info(f"Coinbase connector initialized (simulation={'ON' if simulation_mode else 'OFF'})")
    
    async def connect(self) -> bool:
        """Connect to Coinbase"""
        try:
            if not CCXT_AVAILABLE:
                if self.simulation_mode:
                    self._is_connected = True
                    return True
                return False
            
            if self.simulation_mode or not self.config.api_key:
                self.simulation_mode = True
                self.exchange = ccxt.coinbasepro({'enableRateLimit': True})
                self._is_connected = True
                logger.info("✅ Coinbase connected (simulation mode)")
                return True
            
            self.exchange = ccxt.coinbasepro({
                'apiKey': self.config.api_key,
                'secret': self.config.api_secret,
                'password': self.config.api_passphrase,
                'enableRateLimit': True,
            })
            
            await self.exchange.load_markets()
            
            self.exchange_ws = ccxtpro.coinbasepro({
                'apiKey': self.config.api_key,
                'secret': self.config.api_secret,
                'password': self.config.api_passphrase,
            })
            
            self._is_connected = True
            logger.info("✅ Coinbase connected (live mode)")
            return True
            
        except Exception as e:
            logger.error(f"Failed to connect to Coinbase: {e}")
            return False
    
    async def disconnect(self):
        if self.exchange:
            await self.exchange.close()
        if self.exchange_ws:
            await self.exchange_ws.close()
        self._is_connected = False
    
    async def subscribe_ticker(self, pairs: List[str]):
        if self.simulation_mode:
            return
        
        for pair in pairs:
            asyncio.create_task(self._watch_ticker(pair))
        logger.info(f"✅ Coinbase: Subscribed to {len(pairs)} tickers")
    
    async def _watch_ticker(self, pair: str):
        while self._is_connected and self.exchange_ws:
            try:
                ticker = await self.exchange_ws.watch_ticker(pair)
                self.price_cache[pair] = PriceQuote(
                    exchange=self._name,
                    pair=pair,
                    bid=Decimal(str(ticker['bid'])) if ticker['bid'] else Decimal('0'),
                    ask=Decimal(str(ticker['ask'])) if ticker['ask'] else Decimal('0'),
                    mid=Decimal(str(ticker['last'])) if ticker['last'] else Decimal('0'),
                    timestamp=datetime.fromtimestamp(ticker['timestamp'] / 1000),
                    latency_ms=0.0,
                    volume_24h=Decimal(str(ticker.get('quoteVolume', 0)))
                )
            except Exception as e:
                logger.error(f"Error watching {pair} on Coinbase: {e}")
                await asyncio.sleep(1)
                break
    
    async def get_ticker(self, pair: str) -> Optional[PriceQuote]:
        if pair in self.price_cache:
            quote = self.price_cache[pair]
            age = (datetime.now() - quote.timestamp).total_seconds()
            if age < 5:
                return quote
        
        if self.simulation_mode:
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
            self.price_cache[pair] = quote
            return quote
        except Exception as e:
            logger.error(f"Failed to get ticker for {pair} on Coinbase: {e}")
            return None
    
    async def get_balance(self, currency: Optional[str] = None) -> ExchangeBalance:
        if self.simulation_mode:
            return ExchangeBalance(
                exchange=self._name,
                timestamp=datetime.now(),
                balances=self.simulated_balances.copy(),
                total_usd=self.simulated_balances.get('USD', Decimal('0')),
                available=self.simulated_balances.copy(),
                locked={}
            )
        
        try:
            balance = await self.exchange.fetch_balance()
            balances_dict = {k: Decimal(str(v)) for k, v in balance['total'].items() if v > 0}
            available_dict = {k: Decimal(str(balance['free'].get(k, 0))) for k in balances_dict}
            locked_dict = {k: Decimal(str(balance['used'].get(k, 0))) for k in balances_dict}
            
            return ExchangeBalance(
                exchange=self._name,
                timestamp=datetime.now(),
                balances=balances_dict,
                total_usd=balances_dict.get('USD', Decimal('0')),
                available=available_dict,
                locked=locked_dict
            )
        except Exception as e:
            logger.error(f"Failed to get balance from Coinbase: {e}")
            raise ExchangeError(f"Balance query failed: {e}")
    
    async def submit_market_order(self, pair: str, side: str, quantity: Decimal) -> CryptoOrder:
        order_id = f"CB-{uuid.uuid4().hex[:12]}"
        
        if self.simulation_mode:
            logger.info(f"📝 Coinbase SIMULATION: {side.upper()} {quantity} {pair}")
            return CryptoOrder(
                id=order_id,
                exchange=self._name,
                pair=pair,
                side=OrderSide.BUY if side.lower() == 'buy' else OrderSide.SELL,
                order_type=OrderType.MARKET,
                quantity=quantity,
                price=None,
                created_at=datetime.now(),
                status=OrderStatus.FILLED,
                exchange_order_id=f"SIM-{order_id}",
                filled_quantity=quantity,
                filled_price=Decimal('43200'),  # Mock slightly different price
                fee_paid=quantity * Decimal('0.006'),  # 0.6% taker fee
                fee_currency='USD'
            )
        
        try:
            result = await self.exchange.create_market_order(pair, side.lower(), float(quantity))
            return CryptoOrder(
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
        except Exception as e:
            logger.error(f"Coinbase order failed: {e}")
            raise OrderRejectedError(str(e))
    
    async def cancel_order(self, order_id: str, pair: str) -> bool:
        if self.simulation_mode:
            return True
        try:
            await self.exchange.cancel_order(order_id, pair)
            return True
        except:
            return False
    
    async def get_order_status(self, order_id: str, pair: str) -> CryptoOrder:
        if self.simulation_mode:
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
        
        order_data = await self.exchange.fetch_order(order_id, pair)
        status_map = {
            'open': OrderStatus.PENDING,
            'closed': OrderStatus.FILLED,
            'canceled': OrderStatus.CANCELLED
        }
        
        return CryptoOrder(
            id=order_id,
            exchange=self._name,
            pair=pair,
            side=OrderSide.BUY if order_data['side'] == 'buy' else OrderSide.SELL,
            order_type=OrderType.MARKET,
            quantity=Decimal(str(order_data['amount'])),
            price=None,
            created_at=datetime.fromtimestamp(order_data['timestamp'] / 1000),
            status=status_map.get(order_data['status'], OrderStatus.PENDING)
        )
    
    def get_fee_structure(self) -> Dict[str, Decimal]:
        return {
            'maker_fee': self.config.maker_fee_pct,
            'taker_fee': self.config.taker_fee_pct
        }
    
    async def health_check(self) -> Dict[str, any]:
        if self.simulation_mode:
            return {'healthy': True, 'connected': True, 'mode': 'simulation'}
        
        try:
            if self.exchange:
                await self.exchange.fetch_status()
                return {'healthy': True, 'connected': self._is_connected}
            return {'healthy': False, 'connected': False}
        except:
            return {'healthy': False, 'connected': False}
    
    @property
    def name(self) -> str:
        return self._name
    
    @property
    def is_connected(self) -> bool:
        return self._is_connected
    
    def _generate_mock_price(self, pair: str) -> PriceQuote:
        """Generate mock price (slightly different from Binance for arbitrage)"""
        base_prices = {
            'BTC/USDT': 43200,  # $200 more than Binance
            'ETH/USDT': 2305,
            'SOL/USDT': 101,
        }
        base_price = Decimal(str(base_prices.get(pair, 100)))
        spread = base_price * Decimal('0.0003')
        
        return PriceQuote(
            exchange=self._name,
            pair=pair,
            bid=base_price - spread,
            ask=base_price + spread,
            mid=base_price,
            timestamp=datetime.now(),
            latency_ms=15.0,
            volume_24h=Decimal('8000000')
        )


