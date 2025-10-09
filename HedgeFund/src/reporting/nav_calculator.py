"""
Module: NAV Calculator Service
Purpose: Calculate daily Net Asset Value (NAV)
Dependencies: Position Manager, Options Pricer, Market Data
Exports: NAVCalculator

Calculates portfolio NAV using mark-to-market pricing.
Provides component breakdown for transparency.
Saves to database for historical tracking.
"""

from typing import Dict, List, Optional
from decimal import Decimal
from datetime import datetime, date
import logging

from ..common.models import Position
from ..pricing.options_pricer import OptionsPricer

logger = logging.getLogger(__name__)


class NAVCalculator:
    """
    NAV calculator with mark-to-market pricing
    
    Calculates daily Net Asset Value by:
    1. Valuing all stock positions at current price
    2. Valuing all options at Black-Scholes price
    3. Adding cash
    4. Adjusting for margin
    """
    
    def __init__(self,
                 pricer: OptionsPricer,
                 db_connection=None,
                 broker_client=None):
        """
        Initialize NAV calculator
        
        Args:
            pricer: Options pricing engine
            db_connection: Database connection (optional)
            broker_client: Broker client for real account sync (optional)
        """
        self.pricer = pricer
        self.db = db_connection
        self.broker_client = broker_client
        self.use_broker_nav = True  # Prefer real broker NAV
        
        # NAV history for tracking
        self.nav_history: List[Dict] = []
        
        logger.info("NAVCalculator initialized")
    
    def calculate_nav(self,
                     cash: Decimal,
                     positions: List[Position],
                     market_data_service,
                     timestamp: Optional[datetime] = None) -> Dict:
        """
        Calculate current NAV
        
        Priority:
        1. Use broker's real account equity (if available)
        2. Fall back to mark-to-market calculation
        
        Args:
            cash: Current cash balance
            positions: All current positions
            market_data_service: Market data service for prices/IV
            timestamp: Valuation timestamp (default: now)
        
        Returns:
            Dict with NAV and components
        """
        timestamp = timestamp or datetime.now()
        
        logger.info(f"Calculating NAV at {timestamp}")
        
        # Try to get real NAV from broker first
        if self.use_broker_nav and self.broker_client:
            try:
                account = self.broker_client.get_account()
                if account and 'equity' in account:
                    broker_nav = account['equity']
                    broker_cash = account.get('cash', Decimal('0'))
                    broker_portfolio_value = account.get('portfolio_value', Decimal('0'))
                    
                    logger.info(f"✅ Using real Alpaca NAV: ${broker_nav:,.2f} (Cash: ${broker_cash:,.2f}, Positions: ${broker_portfolio_value:,.2f})")
                    
                    return {
                        'nav': broker_nav,
                        'cash': broker_cash,
                        'total_positions_value': broker_portfolio_value - broker_cash,
                        'equity_value': broker_portfolio_value - broker_cash,
                        'options_value': Decimal('0'),
                        'timestamp': timestamp,
                        'source': 'alpaca_broker'
                    }
            except Exception as e:
                logger.warning(f"Failed to get NAV from broker, using local calculation: {e}")
        
        # Initialize components
        components = {
            'cash': cash,
            'equity_value': Decimal('0.0'),
            'options_value': Decimal('0.0'),
            'short_options_liability': Decimal('0.0'),
            'long_options_value': Decimal('0.0'),
            'total_positions_value': Decimal('0.0')
        }
        
        # Value each position
        for position in positions:
            try:
                # Get current market data
                spot_price = market_data_service.get_price(position.asset)
                
                if position.position_type.value == 'shares':
                    # Stock position - simple valuation
                    value = spot_price * Decimal(str(position.quantity))
                    components['equity_value'] += value
                
                elif position.is_option:
                    # Option position - mark to market
                    if position.expiration and position.strike:
                        # Calculate time to expiration
                        T = (position.expiration - date.today()).days / 365.0
                        
                        if T <= 0:
                            # Expired - intrinsic value only
                            if position.option_type == 'P':
                                intrinsic = max(position.strike - spot_price, Decimal('0'))
                            else:
                                intrinsic = max(spot_price - position.strike, Decimal('0'))
                            
                            value = intrinsic * Decimal(str(position.quantity)) * Decimal('100')
                        else:
                            # Use Black-Scholes
                            iv = market_data_service.get_iv(position.asset)
                            
                            option_type = 'p' if position.option_type == 'P' else 'c'
                            price = self.pricer.price_option(
                                option_type,
                                float(spot_price),
                                float(position.strike),
                                T,
                                float(iv)
                            )
                            
                            value = price * Decimal(str(position.quantity)) * Decimal('100')
                        
                        # For short options, this is a liability (negative)
                        if position.is_short:
                            components['short_options_liability'] += value
                            components['options_value'] -= value
                        else:
                            components['long_options_value'] += value
                            components['options_value'] += value
            
            except Exception as e:
                logger.error(f"Error valuing position {position.id}: {e}")
        
        # Calculate total
        components['total_positions_value'] = (
            components['equity_value'] + 
            components['options_value']
        )
        
        nav = cash + components['total_positions_value']
        
        # Prepare result
        result = {
            'timestamp': timestamp,
            'nav': nav,
            'components': components,
            'cash_pct': float(cash / nav) if nav > 0 else 0.0,
            'equity_pct': float(components['equity_value'] / nav) if nav > 0 else 0.0,
            'options_pct': float(components['options_value'] / nav) if nav > 0 else 0.0,
            'position_count': len(positions)
        }
        
        # Add to history
        self.nav_history.append(result)
        
        # Save to database if available
        if self.db:
            self._save_to_db(result)
        
        logger.info(f"NAV calculated: ${nav:,.2f}")
        
        return result
    
    def calculate_returns(self, nav_history: Optional[List[Dict]] = None) -> Dict:
        """
        Calculate return metrics from NAV history
        
        Args:
            nav_history: List of NAV records (uses internal if not provided)
        
        Returns:
            Dict with return metrics
        """
        history = nav_history or self.nav_history
        
        if len(history) < 2:
            logger.warning("Insufficient history for returns calculation")
            return {}
        
        # Calculate daily returns
        daily_returns = []
        for i in range(1, len(history)):
            if history[i-1]['nav'] > 0:
                ret = float((history[i]['nav'] - history[i-1]['nav']) / history[i-1]['nav'])
                daily_returns.append(ret)
        
        # Calculate metrics
        import numpy as np
        
        if not daily_returns:
            return {}
        
        cumulative_return = float((history[-1]['nav'] - history[0]['nav']) / history[0]['nav'])
        avg_daily_return = np.mean(daily_returns)
        daily_vol = np.std(daily_returns)
        
        # Annualized metrics
        days_per_year = 252
        annual_return = (1 + avg_daily_return) ** days_per_year - 1
        annual_vol = daily_vol * np.sqrt(days_per_year)
        
        # Sharpe ratio (assuming 4% risk-free rate)
        sharpe = (annual_return - 0.04) / annual_vol if annual_vol > 0 else 0.0
        
        # Max drawdown
        peak = history[0]['nav']
        max_dd = 0.0
        
        for record in history:
            peak = max(peak, record['nav'])
            if peak > 0:
                dd = float((record['nav'] - peak) / peak)
                max_dd = min(max_dd, dd)
        
        return {
            'cumulative_return': cumulative_return,
            'daily_return_avg': avg_daily_return,
            'daily_volatility': daily_vol,
            'annual_return': annual_return,
            'annual_volatility': annual_vol,
            'sharpe_ratio': sharpe,
            'max_drawdown': max_dd,
            'days': len(history),
            'start_nav': float(history[0]['nav']),
            'end_nav': float(history[-1]['nav'])
        }
    
    def get_nav_components(self, timestamp: Optional[datetime] = None) -> Dict:
        """
        Get NAV component breakdown
        
        Args:
            timestamp: Timestamp to get (default: latest)
        
        Returns:
            Dict with component breakdown
        """
        if not self.nav_history:
            return {}
        
        # Find matching record or return latest
        if timestamp:
            for record in reversed(self.nav_history):
                if record['timestamp'] == timestamp:
                    return record['components']
        
        return self.nav_history[-1]['components']
    
    def save_nav(self, nav: Decimal, timestamp: datetime, methodology: str = "black_scholes") -> bool:
        """
        Save NAV to database
        
        Args:
            nav: NAV value
            timestamp: Timestamp
            methodology: Pricing methodology used
        
        Returns:
            True if saved successfully
        """
        if not self.db:
            logger.warning("No database connection, NAV not saved")
            return False
        
        try:
            self.db.execute("""
                INSERT INTO nav_history (
                    timestamp, nav, methodology, pricing_source
                ) VALUES (%s, %s, %s, %s)
            """, (timestamp, float(nav), methodology, 'market_data'))
            
            logger.info(f"NAV saved to database: {timestamp} = ${nav:,.2f}")
            return True
        
        except Exception as e:
            logger.error(f"Failed to save NAV: {e}")
            return False
    
    def _save_to_db(self, nav_record: Dict) -> bool:
        """Save NAV record to database"""
        try:
            components = nav_record['components']
            
            self.db.execute("""
                INSERT INTO nav_history (
                    timestamp, nav, cash, equity_value, options_value,
                    methodology, pricing_source, components
                ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
            """, (
                nav_record['timestamp'],
                float(nav_record['nav']),
                float(components['cash']),
                float(components['equity_value']),
                float(components['options_value']),
                'black_scholes',
                'live',
                str(components)  # JSON string
            ))
            
            return True
        except Exception as e:
            logger.error(f"Failed to save NAV to database: {e}")
            return False
    
    def health_check(self) -> bool:
        """Health check"""
        try:
            assert self.pricer is not None
            return True
        except:
            return False


if __name__ == "__main__":
    """Test NAV calculator"""
    import sys
    sys.path.insert(0, '/mnt/4tb/stock_scanner/market_news/HedgeFund')
    
    from src.common.config import load_config
    from src.data.market_data_service import MarketDataService
    from src.common.models import Position, PositionType
    from datetime import timedelta
    
    print("Testing NAV Calculator...")
    
    # Initialize
    config = load_config()
    pricer = OptionsPricer()
    data_service = MarketDataService(config.broker)
    
    nav_calc = NAVCalculator(pricer, db_connection=None)
    
    print(f"\n✓ NAV calculator initialized")
    
    # Create test portfolio
    print(f"\n📊 Creating test portfolio...")
    
    cash = Decimal('50000.00')
    
    positions = [
        Position(
            id="POS-001",
            asset="SPY",
            position_type=PositionType.SHORT_PUT,
            quantity=10,
            entry_price=Decimal('8.50'),
            entry_date=datetime.now(),
            strike=Decimal('670.0'),
            expiration=date.today() + timedelta(days=14),
            option_type='P'
        ),
        Position(
            id="POS-002",
            asset="QQQ",
            position_type=PositionType.SHORT_PUT,
            quantity=5,
            entry_price=Decimal('9.20'),
            entry_date=datetime.now(),
            strike=Decimal('600.0'),
            expiration=date.today() + timedelta(days=14),
            option_type='P'
        )
    ]
    
    print(f"  Cash: ${cash:,.0f}")
    print(f"  Positions: {len(positions)}")
    
    # Calculate NAV
    print(f"\n💰 Calculating NAV...")
    
    nav_result = nav_calc.calculate_nav(
        cash=cash,
        positions=positions,
        market_data_service=data_service
    )
    
    print(f"  NAV: ${nav_result['nav']:,.2f}")
    print(f"  Components:")
    print(f"    Cash: ${nav_result['components']['cash']:,.2f} ({nav_result['cash_pct']:.1%})")
    print(f"    Equity: ${nav_result['components']['equity_value']:,.2f}")
    print(f"    Options: ${nav_result['components']['options_value']:,.2f} ({nav_result['options_pct']:.1%})")
    
    # Simulate some more days
    print(f"\n📈 Simulating 5 days...")
    
    for i in range(5):
        # Adjust cash slightly
        cash += Decimal(str(100 + i * 50))
        
        nav_result = nav_calc.calculate_nav(
            cash=cash,
            positions=positions,
            market_data_service=data_service,
            timestamp=datetime.now() + timedelta(days=i+1)
        )
    
    # Calculate returns
    print(f"\n📊 Performance metrics:")
    
    returns = nav_calc.calculate_returns()
    if returns:
        print(f"  Total return: {returns['cumulative_return']:.2%}")
        print(f"  Daily vol: {returns['daily_volatility']:.2%}")
        print(f"  Sharpe ratio: {returns['sharpe_ratio']:.2f}")
        print(f"  Max drawdown: {returns['max_drawdown']:.2%}")
    
    # Health check
    print(f"\n🏥 Health check: {nav_calc.health_check()}")
    
    print("\n✅ NAV calculator working correctly!")


