"""
Module: Broker Interface (Abstract Base)
Purpose: Define unified interface for all broker clients
Dependencies: Models
Exports: BrokerInterface

Provides abstract base class that all broker implementations must follow.
Ensures consistent interface regardless of broker.
"""

from abc import ABC, abstractmethod
from typing import List, Optional, Dict
from decimal import Decimal

from ..common.models import Order, Fill, Position


class BrokerInterface(ABC):
    """
    Abstract base class for broker clients
    
    All broker implementations (Alpaca, Interactive Brokers, etc.)
    must implement this interface for consistency.
    """
    
    @abstractmethod
    def submit_order(self, order: Order) -> Fill:
        """
        Submit order to broker
        
        Args:
            order: Order to submit
        
        Returns:
            Fill object with execution details
        
        Raises:
            BrokerError: If order fails
        """
        pass
    
    @abstractmethod
    def cancel_order(self, order_id: str) -> bool:
        """
        Cancel pending order
        
        Args:
            order_id: Order ID to cancel
        
        Returns:
            True if cancelled successfully
        """
        pass
    
    @abstractmethod
    def get_order_status(self, order_id: str) -> str:
        """
        Get order status
        
        Args:
            order_id: Order ID to check
        
        Returns:
            Status string ('pending', 'filled', 'cancelled', etc.)
        """
        pass
    
    @abstractmethod
    def get_positions(self) -> List[Position]:
        """
        Get current positions from broker
        
        Returns:
            List of Position objects
        """
        pass
    
    @abstractmethod
    def get_account(self) -> Dict:
        """
        Get account information
        
        Returns:
            Dict with account details (cash, buying_power, etc.)
        """
        pass
    
    @abstractmethod
    def health_check(self) -> bool:
        """
        Check if broker connection is healthy
        
        Returns:
            True if connection working
        """
        pass


class BrokerError(Exception):
    """Base exception for broker errors"""
    pass


class OrderRejectedError(BrokerError):
    """Order was rejected by broker"""
    pass


class InsufficientFundsError(BrokerError):
    """Insufficient funds for order"""
    pass


class ConnectionError(BrokerError):
    """Connection to broker failed"""
    pass

