"""
Data Loader Interface

Abstract interface for data loading in different modes.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
from datetime import datetime


class DataLoaderInterface(ABC):
    """Abstract interface for data loaders"""

    @abstractmethod
    def load_market_data(self, date: Optional[datetime] = None) -> Dict[str, Any]:
        """
        Load market data for a specific date (or current date if None)

        Args:
            date: Date to load data for (None = current/latest)

        Returns:
            Dictionary with market data (prices, VIX, etc.)
        """
        pass

    @abstractmethod
    def get_current_prices(self, tickers: list) -> Dict[str, float]:
        """
        Get current prices for tickers

        Args:
            tickers: List of ticker symbols

        Returns:
            Dictionary of ticker -> price
        """
        pass
