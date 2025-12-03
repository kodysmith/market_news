"""
Paper Trading Executor

Simulates trades and stores positions in Supabase (no real money).
"""

from typing import Dict, Any, Optional
import logging
from .interface import ExecutorInterface
from ..state.supabase import SupabaseStateStore

logger = logging.getLogger(__name__)


class PaperExecutor(ExecutorInterface):
    """Paper trading executor - simulates trades, stores in Supabase"""

    def __init__(self, config: Dict[str, Any], state_store: SupabaseStateStore):
        self.config = config
        self.state_store = state_store
        self.run_id = None
        self.commission_rate = config.get('commission_rate', 0.001)
        self.slippage_rate = config.get('slippage_rate', 0.0005)

    def set_run_id(self, run_id: str):
        """Set the current run ID"""
        self.run_id = run_id

    def execute_trade(self, ticker: str, quantity: float, order_type: str = 'market',
                     price: Optional[float] = None) -> Dict[str, Any]:
        """Execute a simulated trade"""
        if price is None:
            raise ValueError("Price required for paper execution")

        # Apply slippage
        if quantity > 0:
            fill_price = price * (1 + self.slippage_rate)
        else:
            fill_price = price * (1 - self.slippage_rate)

        trade_value = abs(quantity) * fill_price
        commission = trade_value * self.commission_rate

        # Store order in database
        order_data = {
            'run_id': self.run_id,
            'ticker': ticker,
            'quantity': quantity,
            'order_type': order_type,
            'fill_price': fill_price,
            'commission': commission,
            'status': 'filled'
        }

        order_id = self.state_store.create_order(order_data)

        # Update positions
        self.state_store.update_position(
            run_id=self.run_id,
            ticker=ticker,
            quantity=quantity,
            price=fill_price
        )

        result = {
            'ticker': ticker,
            'quantity': quantity,
            'fill_price': fill_price,
            'commission': commission,
            'order_id': order_id,
            'status': 'filled'
        }

        logger.info(f"Paper trade executed: {ticker} {quantity} @ ${fill_price:.2f}")
        return result

    def get_positions(self) -> Dict[str, float]:
        """Get current positions from database"""
        if not self.run_id:
            return {}

        positions = self.state_store.get_positions(self.run_id)
        return {p['ticker']: p['value'] for p in positions}

    def get_portfolio_value(self) -> float:
        """Get total portfolio value from database"""
        if not self.run_id:
            return 0.0

        run = self.state_store.get_run(self.run_id)
        return run.get('portfolio_value', 0.0) if run else 0.0
