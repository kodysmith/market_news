"""
Core Data Models for Crypto Arbitrage
Pure Python dataclasses for type safety and clarity
"""

from dataclasses import dataclass, field
from datetime import datetime
from decimal import Decimal
from typing import Optional, List, Dict
from enum import Enum


class OrderStatus(Enum):
    """Order status enumeration"""
    PENDING = "pending"
    SUBMITTED = "submitted"
    FILLED = "filled"
    PARTIALLY_FILLED = "partially_filled"
    CANCELLED = "cancelled"
    REJECTED = "rejected"
    FAILED = "failed"


class OrderSide(Enum):
    """Order side"""
    BUY = "buy"
    SELL = "sell"


class OrderType(Enum):
    """Order type"""
    MARKET = "market"
    LIMIT = "limit"


@dataclass
class ExchangeInfo:
    """Exchange metadata and fee structure"""
    name: str
    display_name: str
    maker_fee: Decimal  # Fee for limit orders
    taker_fee: Decimal  # Fee for market orders
    withdrawal_fee: Dict[str, Decimal]  # Withdrawal fees by coin
    min_trade_size: Dict[str, Decimal]  # Minimum order size by pair
    is_available: bool = True
    supports_websocket: bool = True


@dataclass
class PriceQuote:
    """Price quote from an exchange"""
    exchange: str
    pair: str
    bid: Decimal
    ask: Decimal
    mid: Decimal
    timestamp: datetime
    latency_ms: float  # Time from exchange to us
    volume_24h: Optional[Decimal] = None


@dataclass
class ArbitrageOpportunity:
    """Detected arbitrage opportunity"""
    id: str
    detected_at: datetime
    
    # Trading pair
    pair: str  # e.g., "BTC/USDT"
    base_asset: str  # e.g., "BTC"
    quote_asset: str  # e.g., "USDT"
    
    # Buy side (cheaper exchange)
    buy_exchange: str
    buy_price: Decimal
    buy_fee_pct: Decimal
    
    # Sell side (expensive exchange)
    sell_exchange: str
    sell_price: Decimal
    sell_fee_pct: Decimal
    
    # Spread calculations
    gross_spread_pct: Decimal  # Price difference
    fees_total_pct: Decimal  # Combined fees
    net_spread_pct: Decimal  # Profit after fees
    
    # Execution parameters
    max_quantity: Decimal  # Max we can trade
    estimated_profit_usd: Decimal
    
    # Risk metrics
    price_staleness_ms: float  # Age of price data
    estimated_slippage_pct: Decimal
    confidence_score: float  # 0-1, higher = better
    
    # Lifecycle tracking
    first_seen_at: Optional[datetime] = None
    last_seen_at: Optional[datetime] = None
    expired_at: Optional[datetime] = None
    opportunity_duration_ms: Optional[float] = None
    times_seen: int = 1
    
    # Status
    is_executed: bool = False
    execution_id: Optional[str] = None


@dataclass
class CryptoOrder:
    """Order to submit to exchange"""
    id: str
    exchange: str
    pair: str
    side: OrderSide
    order_type: OrderType
    quantity: Decimal
    price: Optional[Decimal]  # For limit orders
    
    # Metadata
    created_at: datetime
    arbitrage_id: Optional[str] = None  # Link to arbitrage opportunity
    
    # Status tracking
    status: OrderStatus = OrderStatus.PENDING
    exchange_order_id: Optional[str] = None
    submitted_at: Optional[datetime] = None
    filled_at: Optional[datetime] = None
    
    # Fill details
    filled_quantity: Decimal = Decimal('0')
    filled_price: Optional[Decimal] = None
    fee_paid: Decimal = Decimal('0')
    fee_currency: Optional[str] = None


@dataclass
class ArbitrageExecution:
    """Complete arbitrage execution (both legs)"""
    id: str
    opportunity_id: str
    started_at: datetime
    
    # Trade details
    pair: str
    quantity: Decimal
    
    # Buy leg
    buy_exchange: str
    buy_order_id: str
    buy_status: OrderStatus
    
    # Sell leg
    sell_exchange: str
    sell_order_id: str
    sell_status: OrderStatus
    
    # Optional fields (with defaults)
    completed_at: Optional[datetime] = None
    buy_filled_price: Optional[Decimal] = None
    buy_filled_quantity: Decimal = Decimal('0')
    buy_fee: Decimal = Decimal('0')
    sell_filled_price: Optional[Decimal] = None
    sell_filled_quantity: Decimal = Decimal('0')
    sell_fee: Decimal = Decimal('0')
    
    # Results
    is_successful: bool = False
    realized_profit_usd: Optional[Decimal] = None
    realized_spread_pct: Optional[Decimal] = None
    execution_time_ms: Optional[float] = None
    
    # Risk tracking
    had_failures: bool = False
    failure_reason: Optional[str] = None
    required_reversal: bool = False


@dataclass
class ExchangeBalance:
    """Balance on an exchange"""
    exchange: str
    timestamp: datetime
    
    # Balances by currency
    balances: Dict[str, Decimal]  # {currency: amount}
    
    # USD equivalent
    total_usd: Decimal
    
    # Available vs locked
    available: Dict[str, Decimal]
    locked: Dict[str, Decimal]


@dataclass
class PerformanceMetrics:
    """System performance metrics"""
    timestamp: datetime
    period_start: datetime
    period_end: datetime
    
    # Opportunity metrics
    opportunities_detected: int
    opportunities_executed: int
    execution_rate: float
    
    # Financial metrics
    total_profit_usd: Decimal
    total_fees_usd: Decimal
    net_profit_usd: Decimal
    roi_pct: Decimal
    
    # Execution quality
    avg_execution_time_ms: float
    avg_slippage_pct: Decimal
    success_rate: float
    
    # By exchange
    profits_by_exchange: Dict[str, Decimal]
    trades_by_exchange: Dict[str, int]
    
    # By pair
    profits_by_pair: Dict[str, Decimal]
    trades_by_pair: Dict[str, int]

