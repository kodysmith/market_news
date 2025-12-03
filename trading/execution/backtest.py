"""
Backtest Executor

In-memory execution for backtesting with simulated fills and transaction costs.
"""

from typing import Dict, Any, List, Optional
import logging
from .interface import ExecutorInterface

logger = logging.getLogger(__name__)


class BacktestExecutor(ExecutorInterface):
    """In-memory executor for backtesting"""

    def __init__(self, config: Dict[str, Any], initial_capital: float = 100000):
        self.config = config
        self.initial_capital = initial_capital
        self.cash = initial_capital
        self.positions = {}  # ticker -> shares
        self.position_values = {}  # ticker -> value
        self.commission_rate = config.get('commission_rate', 0.001)  # 0.1% commission
        self.slippage_rate = config.get('slippage_rate', 0.0005)  # 0.05% slippage
        self.trade_history = []

    def execute_trade(self, ticker: str, quantity: float, order_type: str = 'market',
                     price: Optional[float] = None) -> Dict[str, Any]:
        """Execute a trade with simulated fills"""
        if price is None:
            raise ValueError("Price required for backtest execution")

        # Apply slippage
        if quantity > 0:  # Buy
            fill_price = price * (1 + self.slippage_rate)
        else:  # Sell
            fill_price = price * (1 - self.slippage_rate)

        # Calculate trade value
        trade_value = abs(quantity) * fill_price
        commission = trade_value * self.commission_rate
        total_cost = trade_value + commission

        # Check if we have enough cash (for buys) or shares (for sells)
        if quantity > 0:  # Buy
            if total_cost > self.cash:
                logger.warning(f"Insufficient cash for {ticker} trade: need ${total_cost:.2f}, have ${self.cash:.2f}")
                quantity = (self.cash - commission) / fill_price
                trade_value = quantity * fill_price
                total_cost = trade_value + commission

            self.cash -= total_cost
            self.positions[ticker] = self.positions.get(ticker, 0) + quantity
        else:  # Sell
            current_shares = self.positions.get(ticker, 0)
            if abs(quantity) > current_shares:
                logger.warning(f"Insufficient shares for {ticker} trade: need {abs(quantity)}, have {current_shares}")
                quantity = -current_shares
                trade_value = abs(quantity) * fill_price
                total_cost = trade_value + commission

            self.cash += trade_value - commission
            self.positions[ticker] = current_shares + quantity  # quantity is negative

        # Update position value
        if self.positions.get(ticker, 0) == 0:
            self.position_values.pop(ticker, None)
        else:
            self.position_values[ticker] = self.positions[ticker] * fill_price

        result = {
            'ticker': ticker,
            'quantity': quantity,
            'fill_price': fill_price,
            'commission': commission,
            'total_cost': total_cost if quantity > 0 else -total_cost,
            'timestamp': None  # Will be set by caller
        }

        self.trade_history.append(result)
        return result

    def get_positions(self) -> Dict[str, float]:
        """Get current position values"""
        return self.position_values.copy()

    def get_portfolio_value(self) -> float:
        """Get total portfolio value"""
        total_value = self.cash + sum(self.position_values.values())
        return total_value

    def update_prices(self, prices: Dict[str, float]):
        """Update position values with new prices"""
        for ticker, shares in self.positions.items():
            if shares != 0 and ticker in prices:
                self.position_values[ticker] = shares * prices[ticker]

    def get_trade_history(self) -> List[Dict[str, Any]]:
        """Get trade history"""
        return self.trade_history.copy()
