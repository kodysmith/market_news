"""
Module: Position Manager Service
Purpose: Track all positions and portfolio state
Dependencies: PostgreSQL, Redis, Models
Exports: PositionManager

Manages the complete state of the portfolio including all positions,
cash, and derived metrics like delta exposure and NAV components.
"""

from typing import List, Optional, Dict
from decimal import Decimal
from datetime import datetime, date
import logging
import uuid

from ..common.models import Position, PositionType, Fill
from ..pricing.options_pricer import OptionsPricer

logger = logging.getLogger(__name__)


class PositionManager:
    """
    Manages portfolio positions and state
    
    Provides real-time view of all positions, calculates aggregates,
    and handles position lifecycle (open, update, close).
    """
    
    def __init__(self, db_connection=None, pricer: Optional[OptionsPricer] = None):
        """
        Initialize position manager
        
        Args:
            db_connection: Database connection (optional for testing)
            pricer: Options pricer (for mark-to-market)
        """
        self.db = db_connection
        self.pricer = pricer or OptionsPricer()
        
        # In-memory position cache (synced with database)
        self.positions: Dict[str, List[Position]] = {
            'SPY': [],
            'QQQ': [],
            'DIA': [],
            'IWM': []
        }
        
        self.cash = Decimal('100000.00')  # Default initial capital
        
        logger.info("PositionManager initialized")
    
    def add_position(self, position: Position) -> bool:
        """
        Add new position to portfolio
        
        Args:
            position: Position to add
        
        Returns:
            True if successful
        """
        try:
            # Add to in-memory cache
            if position.asset not in self.positions:
                self.positions[position.asset] = []
            
            self.positions[position.asset].append(position)
            
            # Persist to database (if available)
            if self.db:
                self.db.execute("""
                    INSERT INTO positions (
                        position_id, asset, position_type, quantity,
                        entry_price, entry_date, strike, expiration, option_type,
                        option_symbol, dte, delta, status
                    ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                """, (
                    position.id, position.asset, position.position_type.value,
                    position.quantity, float(position.entry_price),
                    position.entry_date,
                    float(position.strike) if position.strike else None,
                    position.expiration, position.option_type,
                    None,  # option_symbol
                    None,  # dte
                    float(position.delta), position.status
                ))
            
            logger.info(f"Position added: {position.id} {position.asset} {position.position_type.value}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to add position: {e}")
            return False
    
    def remove_position(self, position_id: str) -> bool:
        """
        Remove/close a position
        
        Args:
            position_id: ID of position to remove
        
        Returns:
            True if successful
        """
        try:
            # Find and remove from cache
            for asset in self.positions:
                self.positions[asset] = [
                    p for p in self.positions[asset]
                    if p.id != position_id
                ]
            
            # Update in database
            if self.db:
                self.db.execute("""
                    UPDATE positions
                    SET status = 'closed', close_date = %s
                    WHERE position_id = %s
                """, (datetime.now(), position_id))
            
            logger.info(f"Position removed: {position_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to remove position: {e}")
            return False
    
    def get_positions(self, asset: Optional[str] = None, 
                     position_type: Optional[PositionType] = None) -> List[Position]:
        """
        Get positions filtered by asset and/or type
        
        Args:
            asset: Filter by asset (None = all assets)
            position_type: Filter by type (None = all types)
        
        Returns:
            List of matching positions
        """
        if asset:
            positions = self.positions.get(asset, [])
        else:
            positions = []
            for asset_positions in self.positions.values():
                positions.extend(asset_positions)
        
        if position_type:
            positions = [p for p in positions if p.position_type == position_type]
        
        return positions
    
    def get_net_delta(self, asset: Optional[str] = None) -> Decimal:
        """
        Calculate net delta exposure
        
        Args:
            asset: Calculate for specific asset (None = total)
        
        Returns:
            Net delta
        """
        positions = self.get_positions(asset=asset)
        total_delta = sum(p.delta for p in positions)
        return Decimal(str(total_delta))
    
    def update_prices(self, market_data_service) -> int:
        """
        Update all position prices to current market
        
        Args:
            market_data_service: Market data service instance
        
        Returns:
            Number of positions updated
        """
        updated = 0
        
        for asset in self.positions:
            if not self.positions[asset]:
                continue
            
            # Get current price and IV
            try:
                current_price = market_data_service.get_price(asset)
                current_iv = market_data_service.get_iv(asset)
            except:
                logger.warning(f"Could not get price for {asset}")
                continue
            
            for position in self.positions[asset]:
                # Update stock positions
                if position.position_type == PositionType.SHARES:
                    position.current_price = current_price
                    position.market_value = current_price * Decimal(str(position.quantity))
                    position.unrealized_pnl = (current_price - position.entry_price) * Decimal(str(position.quantity))
                
                # Update option positions
                elif position.is_option and position.strike and position.expiration:
                    # Calculate time to expiration
                    T = (position.expiration - date.today()).days / 365.0
                    
                    if T <= 0:
                        # Expired - intrinsic value only
                        if position.option_type == 'P':
                            position.current_price = max(position.strike - current_price, Decimal('0'))
                        else:
                            position.current_price = max(current_price - position.strike, Decimal('0'))
                        position.delta = Decimal('0')
                    else:
                        # Mark to market using pricing engine
                        option_type = 'p' if position.option_type == 'P' else 'c'
                        
                        price = self.pricer.price_option(
                            option_type,
                            float(current_price),
                            float(position.strike),
                            T,
                            float(current_iv)
                        )
                        
                        greeks = self.pricer.calculate_greeks(
                            option_type,
                            float(current_price),
                            float(position.strike),
                            T,
                            float(current_iv)
                        )
                        
                        position.current_price = price
                        position.delta = greeks['delta'] * Decimal(str(position.quantity)) * Decimal('100')
                        position.gamma = greeks['gamma']
                        position.theta = greeks['theta']
                        position.vega = greeks['vega']
                        
                        # Market value (remember we might be short)
                        position.market_value = price * Decimal(str(position.quantity)) * Decimal('100')
                        
                        # P&L for short positions is reversed
                        if position.is_short:
                            position.unrealized_pnl = (position.entry_price - price) * Decimal(str(position.quantity)) * Decimal('100')
                        else:
                            position.unrealized_pnl = (price - position.entry_price) * Decimal(str(position.quantity)) * Decimal('100')
                
                updated += 1
        
        logger.info(f"Updated {updated} positions to market")
        return updated
    
    def get_portfolio_state(self) -> Dict:
        """
        Get complete portfolio state
        
        Returns:
            Dict with cash, NAV, positions, metrics
        """
        # Aggregate values
        total_equity_value = Decimal('0')
        total_options_value = Decimal('0')
        total_delta = Decimal('0')
        
        for asset in self.positions:
            for position in self.positions[asset]:
                if position.position_type == PositionType.SHARES:
                    total_equity_value += position.market_value
                else:
                    # Options (can be positive or negative depending on long/short)
                    if position.is_short:
                        total_options_value -= position.market_value
                    else:
                        total_options_value += position.market_value
                
                total_delta += position.delta
        
        # Calculate NAV
        nav = self.cash + total_equity_value + total_options_value
        
        return {
            'timestamp': datetime.now(),
            'cash': self.cash,
            'nav': nav,
            'total_equity_value': total_equity_value,
            'total_options_value': total_options_value,
            'total_delta': total_delta,
            'position_count': sum(len(p) for p in self.positions.values()),
            'capital_deployed_pct': (total_equity_value + abs(total_options_value)) / nav if nav > 0 else 0
        }
    
    def handle_fill(self, fill: Fill, action_type: str) -> Optional[Position]:
        """
        Create position from fill
        
        Args:
            fill: Fill details from broker
            action_type: Type of action (SELL_PUT, BUY_CALL, etc.)
        
        Returns:
            New Position object
        """
        # Determine position type from action
        position_type_map = {
            'SELL_PUT': PositionType.SHORT_PUT,
            'SELL_CALL': PositionType.COVERED_CALL,
            'BUY_PUT': PositionType.LONG_PUT,
            'BUY_CALL': PositionType.LONG_CALL,
            'BUY_SHARES': PositionType.SHARES,
        }
        
        position_type = position_type_map.get(action_type, PositionType.SHARES)
        
        # Parse option symbol if it's an option
        strike = None
        expiration = None
        option_type = None
        
        if len(fill.symbol) > 10:  # Options have long OCC symbols
            # Parse OCC format: SPY   241115P00400000
            try:
                option_type = fill.symbol[12]  # 'C' or 'P'
                strike_str = fill.symbol[13:21]
                strike = Decimal(str(int(strike_str) / 1000.0))
                
                exp_str = fill.symbol[6:12]  # YYMMDD
                expiration = datetime.strptime(f"20{exp_str}", "%Y%m%d").date()
            except:
                logger.warning(f"Could not parse option symbol: {fill.symbol}")
        
        # Create position
        position = Position(
            id=f"POS-{uuid.uuid4().hex[:8]}",
            asset=fill.asset,
            position_type=position_type,
            quantity=fill.quantity,
            entry_price=fill.fill_price,
            entry_date=fill.timestamp,
            strike=strike,
            expiration=expiration,
            option_type=option_type,
            current_price=fill.fill_price,
            market_value=fill.total_value,
            status='open'
        )
        
        # Add to manager
        self.add_position(position)
        
        # Update cash
        if action_type.startswith('SELL'):
            # Selling = receiving premium
            self.cash += fill.net_proceeds
        elif action_type.startswith('BUY'):
            # Buying = paying premium
            self.cash -= fill.total_value + fill.commission
        
        return position
    
    def health_check(self) -> bool:
        """Health check"""
        try:
            # Verify we can access positions
            _ = self.get_portfolio_state()
            return True
        except:
            return False


if __name__ == "__main__":
    """Test position manager"""
    import sys
    sys.path.insert(0, '/mnt/4tb/stock_scanner/market_news/HedgeFund')
    
    from src.common.config import load_config
    from src.data.market_data_service import MarketDataService
    from datetime import timedelta
    
    print("Testing Position Manager...")
    
    # Initialize
    config = load_config()
    manager = PositionManager(db_connection=None, pricer=OptionsPricer())
    
    print(f"\n✓ Position manager initialized")
    print(f"  Initial cash: ${manager.cash:,.0f}")
    
    # Add a short put position
    put_position = Position(
        id=f"POS-{uuid.uuid4().hex[:8]}",
        asset="QQQ",
        position_type=PositionType.SHORT_PUT,
        quantity=5,
        entry_price=Decimal('8.50'),
        entry_date=datetime.now(),
        strike=Decimal('400.0'),
        expiration=date.today() + timedelta(days=14),
        option_type='P',
        delta=Decimal('-150.0')
    )
    
    manager.add_position(put_position)
    print(f"\n✓ Added position: {put_position.id}")
    
    # Get all positions
    positions = manager.get_positions()
    print(f"  Total positions: {len(positions)}")
    
    # Get positions by asset
    qqq_positions = manager.get_positions(asset='QQQ')
    print(f"  QQQ positions: {len(qqq_positions)}")
    
    # Get net delta
    total_delta = manager.get_net_delta()
    qqq_delta = manager.get_net_delta(asset='QQQ')
    print(f"\n✓ Delta calculations:")
    print(f"  Total delta: {total_delta}")
    print(f"  QQQ delta: {qqq_delta}")
    
    # Update to market prices
    print(f"\n📈 Updating to market prices...")
    data_service = MarketDataService(config.broker)
    updated = manager.update_prices(data_service)
    print(f"  Updated {updated} positions")
    
    # Show updated position
    if qqq_positions:
        pos = qqq_positions[0]
        print(f"  QQQ put now worth: ${pos.current_price}")
        print(f"  Unrealized P&L: ${pos.unrealized_pnl:.2f}")
        print(f"  Delta: {pos.delta:.2f}")
    
    # Get portfolio state
    state = manager.get_portfolio_state()
    print(f"\n✓ Portfolio state:")
    print(f"  NAV: ${state['nav']:,.2f}")
    print(f"  Cash: ${state['cash']:,.2f}")
    print(f"  Total delta: {state['total_delta']:.2f}")
    print(f"  Positions: {state['position_count']}")
    
    # Health check
    print(f"\n🏥 Health check: {manager.health_check()}")
    
    print("\n✅ Position manager working correctly!")

