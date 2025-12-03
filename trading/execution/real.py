"""
Real Trading Executor

Executes real trades via broker API (Alpaca, IBKR, etc.).
"""

from typing import Dict, Any, Optional
import logging
from .interface import ExecutorInterface

logger = logging.getLogger(__name__)


class RealExecutor(ExecutorInterface):
    """Real trading executor - executes via broker API"""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.broker = config.get('broker', 'alpaca')
        self.api_key = config.get('api_key')
        self.api_secret = config.get('api_secret')
        self.base_url = config.get('base_url')

        # Initialize broker client
        self.client = self._init_broker_client()

    def _init_broker_client(self):
        """Initialize broker API client"""
        if self.broker == 'alpaca':
            try:
                import alpaca_trade_api as tradeapi
                return tradeapi.REST(self.api_key, self.api_secret, self.base_url, api_version='v2')
            except ImportError:
                logger.error("alpaca_trade_api not installed")
                return None
        else:
            logger.warning(f"Broker {self.broker} not yet implemented")
            return None

    def execute_trade(self, ticker: str, quantity: float, order_type: str = 'market') -> Dict[str, Any]:
        """Execute a real trade via broker API"""
        if not self.client:
            raise ValueError("Broker client not initialized")

        try:
            # Submit order
            if quantity > 0:
                side = 'buy'
            else:
                side = 'sell'
                quantity = abs(quantity)

            order = self.client.submit_order(
                symbol=ticker,
                qty=quantity,
                side=side,
                type=order_type,
                time_in_force='day'
            )

            result = {
                'ticker': ticker,
                'quantity': quantity if side == 'buy' else -quantity,
                'order_id': order.id,
                'status': order.status,
                'fill_price': None  # Will be updated when filled
            }

            logger.info(f"Real trade submitted: {ticker} {side} {quantity} (order_id: {order.id})")
            return result

        except Exception as e:
            logger.error(f"Error executing real trade: {e}")
            raise

    def get_positions(self) -> Dict[str, float]:
        """Get current positions from broker"""
        if not self.client:
            return {}

        try:
            positions = self.client.list_positions()
            return {pos.symbol: float(pos.market_value) for pos in positions}
        except Exception as e:
            logger.error(f"Error getting positions: {e}")
            return {}

    def get_portfolio_value(self) -> float:
        """Get total portfolio value from broker"""
        if not self.client:
            return 0.0

        try:
            account = self.client.get_account()
            return float(account.portfolio_value)
        except Exception as e:
            logger.error(f"Error getting portfolio value: {e}")
            return 0.0
