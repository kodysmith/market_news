"""
Executor Interface

Abstract interface for trade execution in different modes.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional


class ExecutorInterface(ABC):
    """Abstract interface for executors"""

    @abstractmethod
    def execute_trade(self, ticker: str, quantity: float, order_type: str = 'market') -> Dict[str, Any]:
        """
        Execute a trade

        Args:
            ticker: Ticker symbol
            quantity: Number of shares (positive = buy, negative = sell)
            order_type: Order type ('market', 'limit', etc.)

        Returns:
            Execution result with fill price, quantity, etc.
        """
        pass

    @abstractmethod
    def get_positions(self) -> Dict[str, float]:
        """
        Get current positions

        Returns:
            Dictionary of ticker -> position value
        """
        pass

    @abstractmethod
    def get_portfolio_value(self) -> float:
        """
        Get total portfolio value

        Returns:
            Total portfolio value
        """
        pass
