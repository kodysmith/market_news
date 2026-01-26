"""
Abstract Exchange Interface
Defines the contract all exchange connectors must implement
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional
from decimal import Decimal
from datetime import datetime

from ..common.models import CryptoOrder, PriceQuote, ExchangeBalance


class ExchangeError(Exception):
    """Base exception for exchange errors"""
    pass


class InsufficientBalanceError(ExchangeError):
    """Raised when insufficient balance for trade"""
    pass


class OrderRejectedError(ExchangeError):
    """Raised when exchange rejects order"""
    pass


class ExchangeInterface(ABC):
    """
    Abstract base class for exchange connectors
    
    All exchange implementations must provide these methods
    for unified arbitrage execution across exchanges.
    """
    
    @abstractmethod
    async def connect(self) -> bool:
        """
        Establish connection to exchange (REST + Websocket)
        
        Returns:
            True if successful
        """
        pass
    
    @abstractmethod
    async def disconnect(self):
        """Close all connections"""
        pass
    
    @abstractmethod
    async def subscribe_ticker(self, pairs: List[str]):
        """
        Subscribe to real-time price updates via websocket
        
        Args:
            pairs: List of trading pairs (e.g., ['BTC/USDT', 'ETH/USDT'])
        """
        pass
    
    @abstractmethod
    async def get_ticker(self, pair: str) -> Optional[PriceQuote]:
        """
        Get current ticker/price for a pair
        
        Args:
            pair: Trading pair (e.g., 'BTC/USDT')
        
        Returns:
            PriceQuote with bid/ask/mid or None if not available
        """
        pass
    
    @abstractmethod
    async def get_balance(self, currency: Optional[str] = None) -> ExchangeBalance:
        """
        Get account balance
        
        Args:
            currency: Specific currency or None for all
        
        Returns:
            ExchangeBalance with all holdings
        """
        pass
    
    @abstractmethod
    async def submit_market_order(self, 
                                  pair: str,
                                  side: str,  # 'buy' or 'sell'
                                  quantity: Decimal) -> CryptoOrder:
        """
        Submit market order for immediate execution
        
        Args:
            pair: Trading pair
            side: 'buy' or 'sell'
            quantity: Amount of base currency
        
        Returns:
            CryptoOrder with exchange order ID
        
        Raises:
            InsufficientBalanceError: Not enough funds
            OrderRejectedError: Exchange rejected order
            ExchangeError: Other exchange errors
        """
        pass
    
    @abstractmethod
    async def cancel_order(self, order_id: str, pair: str) -> bool:
        """
        Cancel an open order
        
        Args:
            order_id: Exchange order ID
            pair: Trading pair
        
        Returns:
            True if cancelled successfully
        """
        pass
    
    @abstractmethod
    async def get_order_status(self, order_id: str, pair: str) -> CryptoOrder:
        """
        Get status of an order
        
        Args:
            order_id: Exchange order ID
            pair: Trading pair
        
        Returns:
            Updated CryptoOrder with current status
        """
        pass
    
    @abstractmethod
    def get_fee_structure(self) -> Dict[str, Decimal]:
        """
        Get exchange fee structure
        
        Returns:
            Dict with 'maker_fee' and 'taker_fee' percentages
        """
        pass
    
    @abstractmethod
    async def health_check(self) -> Dict[str, any]:
        """
        Check exchange connectivity and health
        
        Returns:
            Dict with health status metrics
        """
        pass
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Exchange name"""
        pass
    
    @property
    @abstractmethod
    def is_connected(self) -> bool:
        """Is exchange connected"""
        pass


