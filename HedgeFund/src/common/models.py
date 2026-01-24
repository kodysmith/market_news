"""
Module: Core Data Models
Purpose: Shared data structures used across all components
Dependencies: None (pure Python dataclasses)
Exports: Position, Order, Fill, Signal, PortfolioState

This module provides the foundational data models for the entire trading system.
All components use these standardized structures for type safety and consistency.
"""

from dataclasses import dataclass, field
from datetime import datetime, date
from typing import Optional, Dict, List, Literal
from enum import Enum
from decimal import Decimal


class PositionType(Enum):
    """Type of position held"""
    SHORT_PUT = "short_put"
    COVERED_CALL = "covered_call"
    LONG_PUT = "long_put"
    LONG_CALL = "long_call"
    SHARES = "shares"
    HEDGE_PUT = "hedge_put"
    HEDGE_CALL = "hedge_call"


class OrderStatus(Enum):
    """Order lifecycle status"""
    PENDING = "pending"
    SUBMITTED = "submitted"
    FILLED = "filled"
    PARTIAL = "partial"
    CANCELLED = "cancelled"
    REJECTED = "rejected"
    EXPIRED = "expired"


class OrderAction(Enum):
    """Order action type"""
    SELL_PUT = "sell_put"
    BUY_PUT = "buy_put"
    SELL_CALL = "sell_call"
    BUY_CALL = "buy_call"
    BUY_SHARES = "buy_shares"
    SELL_SHARES = "sell_shares"


@dataclass
class Position:
    """
    Represents a single position in the portfolio
    
    Can be stock shares or options (long/short)
    Includes current market value and Greeks
    """
    id: str
    asset: str
    position_type: PositionType
    quantity: int
    entry_price: Decimal
    entry_date: datetime
    
    # Optional for options
    strike: Optional[Decimal] = None
    expiration: Optional[date] = None
    option_type: Optional[Literal['C', 'P']] = None
    
    # Market values (updated real-time)
    current_price: Decimal = Decimal('0.0')
    market_value: Decimal = Decimal('0.0')
    unrealized_pnl: Decimal = Decimal('0.0')
    
    # Greeks (for options)
    delta: Decimal = Decimal('0.0')
    gamma: Decimal = Decimal('0.0')
    theta: Decimal = Decimal('0.0')
    vega: Decimal = Decimal('0.0')
    
    # Metadata
    status: str = "open"
    close_date: Optional[datetime] = None
    close_price: Optional[Decimal] = None
    realized_pnl: Optional[Decimal] = None
    notes: str = ""
    
    @property
    def is_option(self) -> bool:
        """Check if position is an option"""
        return self.strike is not None
    
    @property
    def is_short(self) -> bool:
        """Check if position is short (sold)"""
        return self.position_type in [
            PositionType.SHORT_PUT,
            PositionType.COVERED_CALL
        ]
    
    @property
    def is_long(self) -> bool:
        """Check if position is long (bought)"""
        return not self.is_short
    
    @property
    def notional_value(self) -> Decimal:
        """Calculate notional value"""
        if self.is_option:
            return Decimal(str(self.strike)) * Decimal(str(self.quantity)) * Decimal('100')
        else:
            return self.current_price * Decimal(str(self.quantity))
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for serialization"""
        return {
            'id': self.id,
            'asset': self.asset,
            'position_type': self.position_type.value,
            'quantity': self.quantity,
            'strike': float(self.strike) if self.strike else None,
            'expiration': self.expiration.isoformat() if self.expiration else None,
            'entry_price': float(self.entry_price),
            'current_price': float(self.current_price),
            'market_value': float(self.market_value),
            'unrealized_pnl': float(self.unrealized_pnl),
            'delta': float(self.delta),
            'status': self.status
        }


@dataclass
class Order:
    """
    Represents a trading order (before execution)
    
    Used to request trades from the execution engine
    """
    id: str
    timestamp: datetime
    asset: str
    action: OrderAction
    quantity: int
    
    # For options
    strike: Optional[Decimal] = None
    expiration: Optional[date] = None
    option_type: Optional[Literal['C', 'P']] = None
    
    # Order details
    order_type: Literal['market', 'limit'] = 'limit'
    limit_price: Optional[Decimal] = None
    time_in_force: Literal['day', 'gtc', 'ioc', 'fok'] = 'day'
    
    # Status tracking
    status: OrderStatus = OrderStatus.PENDING
    broker: str = ""
    
    # Execution details (filled after execution)
    fill_price: Optional[Decimal] = None
    fill_timestamp: Optional[datetime] = None
    fill_quantity: int = 0
    commission: Decimal = Decimal('0.0')
    
    # Metadata
    strategy_component: str = ""
    reasoning: str = ""
    expected_premium: Optional[Decimal] = None
    
    @property
    def option_symbol(self) -> Optional[str]:
        """Generate OCC-format option symbol"""
        if not self.strike or not self.expiration:
            return None
        
        # OCC format: SYMBOL  YYMMDDCPPPPPPPP
        padded_symbol = self.asset.ljust(6)
        exp_str = self.expiration.strftime('%y%m%d')
        strike_str = f"{int(self.strike * 1000):08d}"
        
        return f"{padded_symbol}{exp_str}{self.option_type}{strike_str}"
    
    @property
    def is_filled(self) -> bool:
        """Check if order is completely filled"""
        return self.status == OrderStatus.FILLED
    
    def to_dict(self) -> Dict:
        """Convert to dictionary"""
        return {
            'id': self.id,
            'timestamp': self.timestamp.isoformat(),
            'asset': self.asset,
            'action': self.action.value,
            'quantity': self.quantity,
            'strike': float(self.strike) if self.strike else None,
            'expiration': self.expiration.isoformat() if self.expiration else None,
            'option_symbol': self.option_symbol,
            'order_type': self.order_type,
            'limit_price': float(self.limit_price) if self.limit_price else None,
            'status': self.status.value,
            'fill_price': float(self.fill_price) if self.fill_price else None
        }


@dataclass
class Fill:
    """
    Represents a filled order (execution confirmation)
    
    Created after broker confirms execution
    """
    order_id: str
    fill_id: str
    timestamp: datetime
    asset: str
    symbol: str
    quantity: int
    fill_price: Decimal
    commission: Decimal
    broker: str
    
    # Derived values
    total_value: Decimal = field(init=False)
    net_proceeds: Decimal = field(init=False)
    
    def __post_init__(self):
        """Calculate derived values"""
        # For options, multiply by 100
        multiplier = 100 if len(self.symbol) > 10 else 1  # Options have long symbols
        self.total_value = self.fill_price * Decimal(str(self.quantity)) * Decimal(str(multiplier))
        self.net_proceeds = self.total_value - self.commission
    
    @property
    def slippage_pct(self) -> Optional[Decimal]:
        """Calculate slippage if expected price was recorded"""
        # Would need expected_price from Order
        return None
    
    def to_dict(self) -> Dict:
        """Convert to dictionary"""
        return {
            'order_id': self.order_id,
            'fill_id': self.fill_id,
            'timestamp': self.timestamp.isoformat(),
            'asset': self.asset,
            'symbol': self.symbol,
            'quantity': self.quantity,
            'fill_price': float(self.fill_price),
            'commission': float(self.commission),
            'broker': self.broker,
            'total_value': float(self.total_value),
            'net_proceeds': float(self.net_proceeds)
        }


@dataclass
class Signal:
    """
    Trading signal generated by strategy
    
    Represents what the strategy wants to do
    """
    id: str
    timestamp: datetime
    asset: str
    action: OrderAction
    quantity: int
    
    # For options
    strike: Optional[Decimal] = None
    expiration: Optional[date] = None
    option_type: Optional[Literal['C', 'P']] = None
    dte: int = 0
    
    # Pricing guidance
    expected_price: Optional[Decimal] = None
    max_price: Optional[Decimal] = None
    min_price: Optional[Decimal] = None
    
    # Context
    reasoning: str = ""
    strategy_component: str = ""
    priority: int = 5  # 1=highest, 10=lowest
    
    # Risk metrics
    expected_delta: Decimal = Decimal('0.0')
    capital_required: Decimal = Decimal('0.0')
    
    def to_order(self, limit_price: Optional[Decimal] = None) -> Order:
        """Convert signal to executable order"""
        return Order(
            id=f"ORD-{self.id}",
            timestamp=datetime.now(),
            asset=self.asset,
            action=self.action,
            quantity=self.quantity,
            strike=self.strike,
            expiration=self.expiration,
            option_type=self.option_type,
            order_type='limit' if limit_price else 'market',
            limit_price=limit_price or self.expected_price,
            strategy_component=self.strategy_component,
            reasoning=self.reasoning
        )
    
    def to_dict(self) -> Dict:
        """Convert to dictionary"""
        return {
            'id': self.id,
            'timestamp': self.timestamp.isoformat(),
            'asset': self.asset,
            'action': self.action.value,
            'quantity': self.quantity,
            'strike': float(self.strike) if self.strike else None,
            'expiration': self.expiration.isoformat() if self.expiration else None,
            'expected_price': float(self.expected_price) if self.expected_price else None,
            'reasoning': self.reasoning,
            'strategy_component': self.strategy_component
        }


@dataclass
class PortfolioState:
    """
    Current state of the entire portfolio
    
    Snapshot of all positions, cash, exposure at a point in time
    """
    timestamp: datetime
    cash: Decimal
    nav: Decimal
    
    # Positions by asset
    positions: Dict[str, List[Position]] = field(default_factory=dict)
    
    # Aggregate metrics
    total_delta: Decimal = Decimal('0.0')
    total_equity_value: Decimal = Decimal('0.0')
    total_options_value: Decimal = Decimal('0.0')
    margin_used: Decimal = Decimal('0.0')
    
    # Risk metrics
    current_drawdown: Decimal = Decimal('0.0')
    peak_nav: Decimal = Decimal('0.0')
    
    @property
    def position_count(self) -> int:
        """Total number of positions"""
        return sum(len(pos_list) for pos_list in self.positions.values())
    
    @property
    def capital_deployed_pct(self) -> Decimal:
        """Percentage of capital deployed"""
        if self.nav == 0:
            return Decimal('0.0')
        return (self.total_equity_value + self.total_options_value) / self.nav
    
    def get_net_delta(self, asset: Optional[str] = None) -> Decimal:
        """Get net delta for asset or total portfolio"""
        if asset:
            if asset not in self.positions:
                return Decimal('0.0')
            return sum(pos.delta for pos in self.positions[asset])
        else:
            return self.total_delta
    
    def get_positions_by_type(self, position_type: PositionType) -> List[Position]:
        """Get all positions of a specific type"""
        result = []
        for asset_positions in self.positions.values():
            result.extend([p for p in asset_positions if p.position_type == position_type])
        return result
    
    def to_dict(self) -> Dict:
        """Convert to dictionary"""
        return {
            'timestamp': self.timestamp.isoformat(),
            'cash': float(self.cash),
            'nav': float(self.nav),
            'total_delta': float(self.total_delta),
            'position_count': self.position_count,
            'capital_deployed_pct': float(self.capital_deployed_pct),
            'current_drawdown': float(self.current_drawdown),
            'positions': {
                asset: [p.to_dict() for p in positions]
                for asset, positions in self.positions.items()
            }
        }


@dataclass
class RiskMetrics:
    """
    Risk metrics for portfolio
    
    Calculated by Risk Manager service
    """
    timestamp: datetime
    nav: Decimal
    
    # Volatility metrics
    daily_volatility: Decimal = Decimal('0.0')
    annual_volatility: Decimal = Decimal('0.0')
    
    # Return metrics
    sharpe_ratio: Decimal = Decimal('0.0')
    sortino_ratio: Decimal = Decimal('0.0')
    
    # Risk metrics
    value_at_risk_95: Decimal = Decimal('0.0')
    max_drawdown: Decimal = Decimal('0.0')
    current_drawdown: Decimal = Decimal('0.0')
    
    # Exposure metrics
    total_delta: Decimal = Decimal('0.0')
    gross_exposure: Decimal = Decimal('0.0')
    net_exposure: Decimal = Decimal('0.0')
    
    # Position metrics
    position_count: int = 0
    concentration_top5: Decimal = Decimal('0.0')
    
    def to_dict(self) -> Dict:
        """Convert to dictionary"""
        return {
            'timestamp': self.timestamp.isoformat(),
            'nav': float(self.nav),
            'sharpe_ratio': float(self.sharpe_ratio),
            'max_drawdown': float(self.max_drawdown),
            'current_drawdown': float(self.current_drawdown),
            'total_delta': float(self.total_delta),
            'var_95': float(self.value_at_risk_95)
        }


@dataclass
class TradeExecution:
    """
    Complete record of a trade execution
    
    Combines Order + Fill + context for audit trail
    """
    trade_id: str
    order: Order
    fill: Fill
    timestamp: datetime
    
    # P&L tracking
    pnl: Decimal = Decimal('0.0')
    pnl_pct: Decimal = Decimal('0.0')
    
    # Portfolio impact
    nav_before: Decimal = Decimal('0.0')
    nav_after: Decimal = Decimal('0.0')
    delta_before: Decimal = Decimal('0.0')
    delta_after: Decimal = Decimal('0.0')
    
    # Risk tracking
    risk_approved: bool = True
    risk_reason: str = ""
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for logging"""
        return {
            'trade_id': self.trade_id,
            'timestamp': self.timestamp.isoformat(),
            'asset': self.order.asset,
            'action': self.order.action.value,
            'quantity': self.order.quantity,
            'fill_price': float(self.fill.fill_price),
            'commission': float(self.fill.commission),
            'pnl': float(self.pnl),
            'nav_after': float(self.nav_after),
            'delta_after': float(self.delta_after),
            'broker': self.fill.broker,
            'order': self.order.to_dict(),
            'fill': self.fill.to_dict()
        }


# Type aliases for clarity
AssetSymbol = str
OptionSymbol = str
Timestamp = datetime
Price = Decimal
Quantity = int
Delta = Decimal


if __name__ == "__main__":
    # Quick tests of data models
    from datetime import timedelta
    
    print("Testing Core Data Models...")
    
    # Test Position
    position = Position(
        id="POS-001",
        asset="QQQ",
        position_type=PositionType.SHORT_PUT,
        quantity=5,
        entry_price=Decimal('8.50'),
        entry_date=datetime.now(),
        strike=Decimal('400.0'),
        expiration=date.today() + timedelta(days=14),
        option_type='P',
        current_price=Decimal('7.25'),
        delta=Decimal('-150.0')
    )
    
    print(f"✓ Position created: {position.asset} {position.position_type.value}")
    print(f"  Is option: {position.is_option}")
    print(f"  Is short: {position.is_short}")
    print(f"  Notional: ${position.notional_value:,.0f}")
    
    # Test Order
    order = Order(
        id="ORD-001",
        timestamp=datetime.now(),
        asset="SPY",
        action=OrderAction.SELL_PUT,
        quantity=3,
        strike=Decimal('450.0'),
        expiration=date.today() + timedelta(days=14),
        option_type='P',
        limit_price=Decimal('9.50')
    )
    
    print(f"\n✓ Order created: {order.action.value} {order.quantity} contracts")
    print(f"  Option symbol: {order.option_symbol}")
    
    # Test Signal
    signal = Signal(
        id="SIG-001",
        timestamp=datetime.now(),
        asset="DIA",
        action=OrderAction.SELL_PUT,
        quantity=2,
        strike=Decimal('350.0'),
        expiration=date.today() + timedelta(days=14),
        option_type='P',
        expected_price=Decimal('7.80'),
        reasoning="Weekly put sale at 30% delta",
        strategy_component="income_generation"
    )
    
    print(f"\n✓ Signal created: {signal.strategy_component}")
    print(f"  Expected premium: ${signal.expected_price}")
    
    # Test PortfolioState
    portfolio = PortfolioState(
        timestamp=datetime.now(),
        cash=Decimal('50000.00'),
        nav=Decimal('175000.00'),
        positions={'QQQ': [position]},
        total_delta=Decimal('-150.0')
    )
    
    print(f"\n✓ Portfolio state created:")
    print(f"  NAV: ${portfolio.nav:,.0f}")
    print(f"  Cash: ${portfolio.cash:,.0f}")
    print(f"  Positions: {portfolio.position_count}")
    print(f"  Net Delta: {portfolio.get_net_delta()}")
    print(f"  Capital Deployed: {portfolio.capital_deployed_pct:.1%}")
    
    print("\n✅ All data models working correctly!")



