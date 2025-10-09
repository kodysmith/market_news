"""
Module: Alpaca Client
Purpose: Alpaca broker API implementation
Dependencies: alpaca-py, BrokerInterface
Exports: AlpacaClient

Implements BrokerInterface for Alpaca Markets API.
Handles options and stock trading via Alpaca.
"""

from typing import List, Optional, Dict
from decimal import Decimal
from datetime import datetime, date
import logging
import uuid

from ..common.models import Order, Fill, Position, PositionType, OrderStatus, OrderAction
from ..common.config import BrokerConfig
from .broker_interface import BrokerInterface, BrokerError, OrderRejectedError, InsufficientFundsError

logger = logging.getLogger(__name__)

# Try to import Alpaca SDK
try:
    from alpaca.trading.client import TradingClient
    from alpaca.trading.requests import LimitOrderRequest, MarketOrderRequest
    from alpaca.trading.enums import OrderSide, TimeInForce, OrderClass
    ALPACA_AVAILABLE = True
except ImportError:
    ALPACA_AVAILABLE = False
    # Create dummy classes for type hints when SDK not available
    OrderSide = type('OrderSide', (), {'BUY': 'buy', 'SELL': 'sell'})
    logger.warning("alpaca-py not installed. Install with: pip install alpaca-py")


class AlpacaClient(BrokerInterface):
    """
    Alpaca broker client implementation
    
    Connects to Alpaca Markets API for trade execution.
    Supports both paper and live trading.
    """
    
    def __init__(self, config: BrokerConfig):
        """
        Initialize Alpaca client
        
        Args:
            config: Broker configuration with API keys
        """
        self.config = config
        self.client = None
        
        if not ALPACA_AVAILABLE:
            logger.error("Alpaca SDK not available")
            return
        
        if not config.alpaca_api_key:
            logger.warning("No Alpaca API keys configured")
            return
        
        try:
            self.client = TradingClient(
                api_key=config.alpaca_api_key,
                secret_key=config.alpaca_secret_key,
                paper=config.alpaca_paper_mode
            )
            logger.info(f"Alpaca client initialized ({'paper' if config.alpaca_paper_mode else 'live'})")
        except Exception as e:
            logger.error(f"Failed to initialize Alpaca client: {e}")
    
    def submit_order(self, order: Order) -> Fill:
        """
        Submit order to Alpaca
        
        Args:
            order: Order to submit
        
        Returns:
            Fill object
        """
        if not self.client:
            raise BrokerError("Alpaca client not initialized")
        
        try:
            # Build Alpaca order request
            side = self._get_order_side(order.action)
            
            # Create order request based on type
            if order.order_type == 'limit':
                alpaca_order = LimitOrderRequest(
                    symbol=order.option_symbol or order.asset,
                    qty=order.quantity,
                    side=side,
                    time_in_force=TimeInForce.DAY,
                    limit_price=float(order.limit_price) if order.limit_price else None
                )
            else:  # market
                alpaca_order = MarketOrderRequest(
                    symbol=order.option_symbol or order.asset,
                    qty=order.quantity,
                    side=side,
                    time_in_force=TimeInForce.DAY
                )
            
            # Submit to Alpaca
            alpaca_response = self.client.submit_order(alpaca_order)
            
            # Wait for fill (simplified - in production would poll)
            # For now, assume filled at limit price
            fill_price = order.limit_price if order.limit_price else order.expected_premium
            
            # Create Fill object
            fill = Fill(
                order_id=order.id,
                fill_id=f"FILL-{uuid.uuid4().hex[:8]}",
                timestamp=datetime.now(),
                asset=order.asset,
                symbol=order.option_symbol or order.asset,
                quantity=order.quantity,
                fill_price=fill_price or Decimal('0.0'),
                commission=Decimal(str(self.config.option_commission * order.quantity)),
                broker="alpaca"
            )
            
            logger.info(f"Order filled: {order.id} → {fill.fill_id}")
            return fill
        
        except Exception as e:
            logger.error(f"Order submission failed: {e}")
            raise BrokerError(f"Alpaca order failed: {e}")
    
    def _get_order_side(self, action: OrderAction) -> OrderSide:
        """Convert OrderAction to Alpaca OrderSide"""
        if action in [OrderAction.SELL_PUT, OrderAction.SELL_CALL, OrderAction.SELL_SHARES]:
            return OrderSide.SELL
        else:
            return OrderSide.BUY
    
    def cancel_order(self, order_id: str) -> bool:
        """Cancel order"""
        if not self.client:
            return False
        
        try:
            self.client.cancel_order_by_id(order_id)
            logger.info(f"Order cancelled: {order_id}")
            return True
        except Exception as e:
            logger.error(f"Cancel failed: {e}")
            return False
    
    def get_order_status(self, order_id: str) -> str:
        """Get order status"""
        if not self.client:
            return "error"
        
        try:
            order = self.client.get_order_by_id(order_id)
            return str(order.status)
        except Exception as e:
            logger.error(f"Failed to get order status: {e}")
            return "error"
    
    def get_positions(self) -> List[Position]:
        """Get positions from Alpaca"""
        if not self.client:
            return []
        
        try:
            alpaca_positions = self.client.get_all_positions()
            positions = []
            
            for pos in alpaca_positions:
                # Convert to our Position model
                position = Position(
                    id=f"ALP-{pos.symbol}",
                    asset=pos.symbol[:6].strip() if len(pos.symbol) > 10 else pos.symbol,
                    position_type=PositionType.SHARES,  # Simplified
                    quantity=int(pos.qty),
                    entry_price=Decimal(str(pos.avg_entry_price)),
                    entry_date=datetime.now(),  # Alpaca doesn't provide this
                    current_price=Decimal(str(pos.current_price)),
                    market_value=Decimal(str(pos.market_value)),
                    unrealized_pnl=Decimal(str(pos.unrealized_pl))
                )
                positions.append(position)
            
            return positions
        
        except Exception as e:
            logger.error(f"Failed to get positions: {e}")
            return []
    
    def get_account(self) -> Dict:
        """Get account info"""
        if not self.client:
            return {}
        
        try:
            account = self.client.get_account()
            
            return {
                'cash': Decimal(str(account.cash)),
                'buying_power': Decimal(str(account.buying_power)),
                'equity': Decimal(str(account.equity)),
                'portfolio_value': Decimal(str(account.portfolio_value)),
                'pattern_day_trader': account.pattern_day_trader,
                'daytrade_count': account.daytrade_count
            }
        
        except Exception as e:
            logger.error(f"Failed to get account: {e}")
            return {}
    
    def health_check(self) -> bool:
        """Health check"""
        if not self.client:
            return False
        
        try:
            account = self.client.get_account()
            return account.status == 'ACTIVE'
        except:
            return False


if __name__ == "__main__":
    """Test Alpaca client"""
    import sys
    sys.path.insert(0, '/mnt/4tb/stock_scanner/market_news/HedgeFund')
    
    from src.common.config import load_config
    
    print("Testing Alpaca Client...")
    
    # Load config
    config = load_config()
    
    # Initialize client
    client = AlpacaClient(config.broker)
    
    print(f"\n✓ Alpaca client initialized")
    print(f"  SDK available: {ALPACA_AVAILABLE}")
    print(f"  Paper mode: {config.broker.alpaca_paper_mode}")
    print(f"  Has API keys: {bool(config.broker.alpaca_api_key)}")
    
    if client.client:
        print(f"\n📊 Testing account access...")
        
        # Get account
        account = client.get_account()
        if account:
            print(f"  Cash: ${account.get('cash', 0):,.2f}")
            print(f"  Buying power: ${account.get('buying_power', 0):,.2f}")
            print(f"  Equity: ${account.get('equity', 0):,.2f}")
        else:
            print(f"  Could not retrieve account (API keys needed)")
        
        # Health check
        health = client.health_check()
        print(f"\n🏥 Health check: {health}")
    else:
        print(f"\n⚠️  Client not connected (API keys needed)")
        print(f"  Add to .env:")
        print(f"    ALPACA_API_KEY=your_key_here")
        print(f"    ALPACA_SECRET_KEY=your_secret_here")
    
    print("\n✅ Alpaca client ready (add API keys to activate)!")

