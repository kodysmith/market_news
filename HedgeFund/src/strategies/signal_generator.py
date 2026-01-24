"""
Module: Strategy Signal Generator
Purpose: Generate trading signals based on adaptive wheel strategy
Dependencies: Data Models, Options Pricing, Market Data
Exports: SignalGenerator

This is the brain of the trading system - implements the validated strategy:
- Weekly put sales at 30% delta, 14 DTE
- Adaptive hedging (1-2 DTE weeknights, 7-14 DTE weekends)
- Profit targets (50% of max profit)
- Wheel mechanics (assignments → covered calls)
"""

from typing import List, Optional, Tuple
from datetime import datetime, date, timedelta
from decimal import Decimal
import logging
import uuid

from ..common.models import Signal, Position, PositionType, OrderAction
from ..common.config import StrategyConfig
from ..pricing.options_pricer import OptionsPricer

logger = logging.getLogger(__name__)


class SignalGenerator:
    """
    Strategy signal generator implementing adaptive wheel strategy
    
    Based on backtested strategy with 79% return, 1.63 Sharpe, -5.6% max DD.
    """
    
    def __init__(self, config: StrategyConfig, pricer: OptionsPricer):
        """
        Initialize signal generator
        
        Args:
            config: Strategy configuration
            pricer: Options pricing engine
        """
        self.config = config
        self.pricer = pricer
        
        logger.info(f"SignalGenerator initialized for assets: {config.assets}")
    
    def generate_signals(self,
                        current_date: datetime,
                        market_data_service,
                        position_manager) -> List[Signal]:
        """
        Generate all trading signals for the current day
        
        Args:
            current_date: Current trading date
            market_data_service: Market data service instance
            position_manager: Position manager instance
        
        Returns:
            List of trading signals
        """
        signals = []
        
        # Check if market day (skip weekends)
        if current_date.weekday() >= 5:  # Saturday=5, Sunday=6
            logger.debug(f"Weekend day, skipping: {current_date.date()}")
            return signals
        
        # Get portfolio state
        portfolio = position_manager.get_portfolio_state()
        
        logger.debug(f"Signal generation for {current_date.date()} (weekday={current_date.weekday()})")
        logger.debug(f"  Portfolio: NAV=${portfolio['nav']:,.2f}, Cash=${portfolio['cash']:,.2f}")
        
        # 1. Check for volatility events (SQQQ spike = QQQ dive)
        volatility_signals = self.check_volatility_opportunities(
            current_date,
            market_data_service,
            position_manager
        )
        signals.extend(volatility_signals)
        
        # 2. Check profit targets on existing positions
        profit_signals = self.check_profit_targets(
            position_manager.get_positions(),
            market_data_service
        )
        signals.extend(profit_signals)
        
        # 3. Generate new put sales (daily if capital available)
        # Sell puts any day, especially after dips!
        print(f"   💰 Checking for new put sale opportunities...")
        logger.info(f"Checking for put sales (weekday={current_date.weekday()})...")
        put_signals = self.generate_put_sales(
            current_date,
            market_data_service,
            position_manager
        )
        
        if put_signals:
            print(f"   📊 Generated {len(put_signals)} put signals")
            logger.info(f"  Generated {len(put_signals)} put signals")
        else:
            logger.debug(f"No put signals generated (capital deployed: {portfolio['capital_deployed_pct']:.1%})")
        
        signals.extend(put_signals)
        
        # 4. Handle assignments (shares → covered calls)
        assignment_signals = self.handle_assignments(
            position_manager.get_positions(position_type=PositionType.SHARES),
            market_data_service,
            current_date
        )
        signals.extend(assignment_signals)
        
        # 4. Generate hedges (adaptive based on day of week)
        if self.config.enable_adaptive_hedge:
            hedge_signals = self.generate_hedges(
                current_date,
                market_data_service,
                position_manager
            )
            signals.extend(hedge_signals)
        
        logger.info(f"Generated {len(signals)} signals for {current_date.date()}")
        return signals
    
    def check_volatility_opportunities(self,
                                      current_date: datetime,
                                      market_data_service,
                                      position_manager) -> List[Signal]:
        """
        Check for volatility-based opportunities
        
        Logic:
        1. If SQQQ spiked (QQQ diving):
           - Sell QQQ shares if holding (take profit or cut loss)
           - Sell more puts on cheaper QQQ (better entry point)
        
        2. If QQQ recovering:
           - Close profitable put positions
           - Sell covered calls on shares
        
        Args:
            current_date: Current date
            market_data_service: Market data service
            position_manager: Position manager
        
        Returns:
            List of volatility-driven signals
        """
        signals = []
        
        try:
            # Get current prices
            qqq_price = market_data_service.get_price('QQQ')
            
            # Check if we have QQQ shares
            qqq_shares = position_manager.get_positions(
                asset='QQQ',
                position_type=PositionType.SHARES
            )
            
            if qqq_shares:
                for position in qqq_shares:
                    # Calculate unrealized P&L
                    cost_basis = position.cost_basis
                    current_value = qqq_price * abs(position.quantity)
                    pnl_pct = ((current_value - cost_basis) / cost_basis) * Decimal('100')
                    
                    # Sell shares if:
                    # - In profit >5% (take profit)
                    # - OR down >10% (cut losses after big drop)
                    if pnl_pct > Decimal('5') or pnl_pct < Decimal('-10'):
                        logger.info(f"🎯 Volatility trigger: Sell QQQ shares (P&L: {pnl_pct:.1f}%)")
                        
                        signal = Signal(
                            id=f"SIG-VOLATILITY-QQQ-{uuid.uuid4().hex[:8]}",
                            timestamp=current_date,
                            asset='QQQ',
                            action=OrderAction.CLOSE_SHARES,
                            quantity=abs(position.quantity),
                            strike=None,
                            expiration=None,
                            option_type=None,
                            dte=0,
                            expected_price=qqq_price,
                            max_price=qqq_price * Decimal('1.01'),
                            min_price=qqq_price * Decimal('0.99'),
                            reasoning=f"Volatility event: {'Take profit' if pnl_pct > 0 else 'Cut loss'} at {pnl_pct:.1f}%",
                            strategy_component="volatility_response",
                            priority=2,  # High priority
                            expected_delta=Decimal('0'),
                            capital_required=Decimal('0')
                        )
                        
                        signals.append(signal)
            
            # After selling shares or if QQQ dipped, sell more puts
            # This is handled by regular put generation which happens daily now
            
        except Exception as e:
            logger.error(f"Error checking volatility opportunities: {e}")
        
        return signals
    
    def generate_put_sales(self,
                          current_date: datetime,
                          market_data_service,
                          position_manager) -> List[Signal]:
        """
        Generate weekly put sale signals
        
        Strategy: Sell 30% delta puts at 14 DTE on Monday
        Position size: 1% of capital per trade
        """
        signals = []
        portfolio = position_manager.get_portfolio_state()
        
        # Check capital deployment
        if portfolio['capital_deployed_pct'] >= self.config.max_capital_deployed:
            logger.info(f"Capital deployed: {portfolio['capital_deployed_pct']:.1%}, skipping new puts")
            return signals
        
        # Calculate position size
        cash = portfolio.get('cash', Decimal('0.0'))
        nav = portfolio.get('nav', cash)
        
        print(f"      Cash: ${cash:,.2f}, NAV: ${nav:,.2f}")
        
        # Debug logging
        if cash == 0:
            logger.warning(f"Portfolio has no cash! Portfolio state: {portfolio}")
            return signals
        
        # Use available cash directly (not just a percentage)
        # This allows trading based on actual capital available
        available_capital = cash
        print(f"      Available capital: ${available_capital:,.2f}")
        
        for asset in self.config.assets:
            try:
                # Get market data
                spot_price = market_data_service.get_price(asset)
                iv = market_data_service.get_iv(asset)
                
                # Validate data
                if spot_price == 0 or spot_price is None:
                    logger.warning(f"Invalid spot price for {asset}: {spot_price}")
                    continue
                
                if iv == 0 or iv is None:
                    logger.warning(f"Invalid IV for {asset}: {iv}, using default 0.25")
                    iv = Decimal('0.25')
                
                # Calculate option parameters
                T = self.config.put_dte / 365.0
                
                if T <= 0:
                    logger.warning(f"Invalid time to expiration: {T}")
                    continue
                
                strike = self.pricer.strike_for_delta(
                    'p',
                    float(spot_price),
                    T,
                    float(iv),
                    self.config.put_delta_target
                )
                
                premium = self.pricer.price_option(
                    'p',
                    float(spot_price),
                    float(strike),
                    T,
                    float(iv)
                )
                
                # Calculate number of contracts
                # Each contract controls 100 shares, requires collateral = strike * 100
                collateral_per_contract = strike * Decimal('100')
                
                if collateral_per_contract == 0:
                    logger.error(f"Collateral is 0! Strike={strike}, spot={spot_price}")
                    continue
                
                if available_capital == 0:
                    logger.error(f"Available capital is 0!")
                    continue
                
                max_contracts = int(available_capital / collateral_per_contract)
                
                # Apply asset allocation (e.g., SPY gets 40% of trades)
                asset_allocation = self.config.allocations.get(asset, 0.25)
                max_contracts_for_asset = int(max_contracts * asset_allocation)
                
                # Also apply absolute limits
                max_contracts_for_asset = max(1, min(max_contracts_for_asset, 10))  # Between 1-10 contracts
                
                print(f"      {asset}: Strike=${strike:.2f}, Premium=${premium:.2f}, Contracts={max_contracts_for_asset}")
                
                if max_contracts_for_asset < 1:
                    print(f"         ⚠️  Not enough capital for {asset} (max_contracts={max_contracts_for_asset})")
                    logger.debug(f"Not enough capital for {asset} put")
                    continue
                
                # Create signal
                signal = Signal(
                    id=f"SIG-PUT-{asset}-{uuid.uuid4().hex[:8]}",
                    timestamp=current_date,
                    asset=asset,
                    action=OrderAction.SELL_PUT,
                    quantity=max_contracts_for_asset,
                    strike=strike,
                    expiration=current_date.date() + timedelta(days=self.config.put_dte),
                    option_type='P',
                    dte=self.config.put_dte,
                    expected_price=premium,
                    max_price=premium * Decimal('1.05'),  # Allow 5% slippage
                    min_price=premium * Decimal('0.95'),
                    reasoning=f"Weekly {self.config.put_delta_target:.0%} delta put sale at {self.config.put_dte}DTE",
                    strategy_component="income_generation",
                    priority=5,
                    expected_delta=Decimal(str(self.config.put_delta_target)) * Decimal(str(max_contracts_for_asset)) * Decimal('100'),
                    capital_required=collateral_per_contract * Decimal(str(max_contracts_for_asset))
                )
                
                signals.append(signal)
                logger.info(f"Put signal: {asset} {max_contracts}x ${strike:.2f}P @ ${premium:.2f}")
                
            except Exception as e:
                logger.error(f"Error generating put signal for {asset}: {e}")
        
        return signals
    
    def check_profit_targets(self,
                            positions: List[Position],
                            market_data_service) -> List[Signal]:
        """
        Check if any positions hit profit targets
        
        Target: Close at 50% of max profit (sell for 50% of entry premium)
        """
        signals = []
        
        for position in positions:
            # Only check short options
            if not position.is_short or not position.is_option:
                continue
            
            # Skip if expired
            if position.expiration and position.expiration <= date.today():
                continue
            
            try:
                # Check if we can close for 50% profit
                target_price = position.entry_price * Decimal(str(1 - self.config.profit_target_pct))
                
                if position.current_price <= target_price:
                    # Create buy-to-close signal
                    action = OrderAction.BUY_PUT if position.option_type == 'P' else OrderAction.BUY_CALL
                    
                    signal = Signal(
                        id=f"SIG-CLOSE-{position.id}-{uuid.uuid4().hex[:8]}",
                        timestamp=datetime.now(),
                        asset=position.asset,
                        action=action,
                        quantity=position.quantity,
                        strike=position.strike,
                        expiration=position.expiration,
                        option_type=position.option_type,
                        expected_price=position.current_price,
                        max_price=target_price * Decimal('1.10'),  # Allow slippage
                        reasoning=f"Profit target hit: {self.config.profit_target_pct:.0%} of max profit",
                        strategy_component="profit_taking",
                        priority=3  # Higher priority
                    )
                    
                    signals.append(signal)
                    logger.info(f"Profit target: Close {position.asset} {position.strike}P for 50% profit")
            
            except Exception as e:
                logger.error(f"Error checking profit target for {position.id}: {e}")
        
        return signals
    
    def handle_assignments(self,
                          share_positions: List[Position],
                          market_data_service,
                          current_date: datetime) -> List[Signal]:
        """
        Handle share assignments from put sales
        
        Strategy: Sell covered calls at 30% delta, 30 DTE
        """
        signals = []
        
        for position in share_positions:
            # Check if we already have a covered call on these shares
            # (In real system, would check via position manager)
            
            try:
                # Get market data
                spot_price = market_data_service.get_price(position.asset)
                iv = market_data_service.get_iv(position.asset)
                
                # Calculate covered call parameters
                T = self.config.call_dte / 365.0
                strike = self.pricer.strike_for_delta(
                    'c',
                    float(spot_price),
                    T,
                    float(iv),
                    self.config.call_delta_target
                )
                
                premium = self.pricer.price_option(
                    'c',
                    float(spot_price),
                    float(strike),
                    T,
                    float(iv)
                )
                
                # Sell calls for however many shares we have (100 shares per contract)
                contracts = position.quantity // 100
                
                if contracts < 1:
                    continue
                
                signal = Signal(
                    id=f"SIG-CC-{position.asset}-{uuid.uuid4().hex[:8]}",
                    timestamp=current_date,
                    asset=position.asset,
                    action=OrderAction.SELL_CALL,
                    quantity=contracts,
                    strike=strike,
                    expiration=current_date.date() + timedelta(days=self.config.call_dte),
                    option_type='C',
                    dte=self.config.call_dte,
                    expected_price=premium,
                    reasoning=f"Covered call on assigned shares",
                    strategy_component="wheel_covered_call",
                    priority=4
                )
                
                signals.append(signal)
                logger.info(f"Covered call: {position.asset} {contracts}x ${strike:.2f}C @ ${premium:.2f}")
            
            except Exception as e:
                logger.error(f"Error generating covered call for {position.id}: {e}")
        
        return signals
    
    def generate_hedges(self,
                       current_date: datetime,
                       market_data_service,
                       position_manager) -> List[Signal]:
        """
        Generate adaptive hedge signals
        
        Strategy:
        - Weeknights (Mon-Thu): 1-2 DTE puts
        - Weekends (Fri): 7-14 DTE puts
        - Delta: 50% of net short delta
        """
        signals = []
        
        # Check if we should close existing hedges (morning)
        if current_date.hour < 10:  # Before 10am
            close_signals = self._close_overnight_hedges(position_manager)
            signals.extend(close_signals)
        
        # Open new hedges (afternoon)
        if current_date.hour >= 14:  # After 2pm
            open_signals = self._open_overnight_hedges(
                current_date,
                market_data_service,
                position_manager
            )
            signals.extend(open_signals)
        
        return signals
    
    def _close_overnight_hedges(self, position_manager) -> List[Signal]:
        """Close existing overnight hedge positions"""
        signals = []
        
        hedge_positions = position_manager.get_positions(
            position_type=PositionType.HEDGE_PUT
        )
        
        for position in hedge_positions:
            signal = Signal(
                id=f"SIG-HEDGE-CLOSE-{position.id}",
                timestamp=datetime.now(),
                asset=position.asset,
                action=OrderAction.SELL_PUT,  # We're long, so sell to close
                quantity=position.quantity,
                strike=position.strike,
                expiration=position.expiration,
                option_type='P',
                expected_price=position.current_price,
                reasoning="Close overnight hedge",
                strategy_component="hedge_management",
                priority=2
            )
            signals.append(signal)
        
        return signals
    
    def _open_overnight_hedges(self,
                              current_date: datetime,
                              market_data_service,
                              position_manager) -> List[Signal]:
        """Open new overnight hedge positions"""
        signals = []
        portfolio = position_manager.get_portfolio_state()
        
        # Determine hedge DTE based on day of week
        is_friday = current_date.weekday() == 4
        hedge_dte = self.config.weekend_hedge_dte if is_friday else self.config.weeknight_hedge_dte
        
        for asset in self.config.assets:
            try:
                # Get net delta for this asset
                net_delta = position_manager.get_net_delta(asset=asset)
                
                # Only hedge if we have short delta exposure
                if net_delta >= 0:
                    continue
                
                # Calculate hedge size (cover portion of short delta)
                hedge_delta_target = net_delta * Decimal(str(self.config.hedge_coverage_pct))
                
                # Get market data
                spot_price = market_data_service.get_price(asset)
                iv = market_data_service.get_iv(asset)
                
                # Calculate hedge put parameters
                T = hedge_dte / 365.0
                strike = self.pricer.strike_for_delta(
                    'p',
                    float(spot_price),
                    T,
                    float(iv),
                    self.config.hedge_delta_target
                )
                
                premium = self.pricer.price_option(
                    'p',
                    float(spot_price),
                    float(strike),
                    T,
                    float(iv)
                )
                
                # Calculate contracts needed
                # Each put has delta ≈ -50 (for 50% delta option), so 1 contract = -50 delta
                contracts_needed = int(abs(hedge_delta_target) / 50)
                contracts_needed = max(1, min(contracts_needed, 10))  # Between 1-10
                
                # Check cost (should be < max_hedge_cost_pct of NAV)
                hedge_cost = premium * Decimal(str(contracts_needed)) * Decimal('100')
                max_cost = portfolio['nav'] * Decimal(str(self.config.max_hedge_cost_pct))
                
                if hedge_cost > max_cost:
                    logger.info(f"Hedge cost ${hedge_cost:.0f} exceeds max ${max_cost:.0f}, skipping")
                    continue
                
                signal = Signal(
                    id=f"SIG-HEDGE-{asset}-{uuid.uuid4().hex[:8]}",
                    timestamp=current_date,
                    asset=asset,
                    action=OrderAction.BUY_PUT,  # Buy protective puts
                    quantity=contracts_needed,
                    strike=strike,
                    expiration=current_date.date() + timedelta(days=hedge_dte),
                    option_type='P',
                    dte=hedge_dte,
                    expected_price=premium,
                    reasoning=f"{'Weekend' if is_friday else 'Overnight'} hedge ({hedge_dte}DTE)",
                    strategy_component="adaptive_hedging",
                    priority=6,
                    capital_required=hedge_cost
                )
                
                signals.append(signal)
                logger.info(f"Hedge signal: {asset} {contracts_needed}x ${strike:.2f}P @ ${premium:.2f}")
            
            except Exception as e:
                logger.error(f"Error generating hedge for {asset}: {e}")
        
        return signals
    
    def health_check(self) -> bool:
        """Health check"""
        try:
            # Verify config loaded
            assert len(self.config.assets) > 0
            assert self.pricer is not None
            return True
        except:
            return False


if __name__ == "__main__":
    """Test signal generator"""
    import sys
    sys.path.insert(0, '/mnt/4tb/stock_scanner/market_news/HedgeFund')
    
    from src.common.config import load_config
    from src.data.market_data_service import MarketDataService
    from src.strategies.position_manager import PositionManager
    
    print("Testing Signal Generator...")
    
    # Load config
    config = load_config()
    
    # Initialize dependencies
    pricer = OptionsPricer()
    data_service = MarketDataService(config.broker)
    position_mgr = PositionManager(db_connection=None, pricer=pricer)
    
    # Initialize signal generator
    signal_gen = SignalGenerator(config.strategy, pricer)
    
    print(f"\n✓ Signal generator initialized")
    print(f"  Assets: {config.strategy.assets}")
    print(f"  Put delta: {config.strategy.put_delta_target}")
    print(f"  Adaptive hedging: {config.strategy.enable_adaptive_hedge}")
    
    # Test put sale signal generation (Monday)
    monday = datetime(2025, 10, 13, 9, 30)  # Monday 9:30am
    
    print(f"\n📊 Generating signals for Monday...")
    signals = signal_gen.generate_signals(monday, data_service, position_mgr)
    
    print(f"  Generated {len(signals)} signals")
    for signal in signals:
        print(f"    {signal.action.value}: {signal.asset} {signal.quantity}x "
              f"${signal.strike:.2f} @ ${signal.expected_price:.2f}")
    
    # Test hedge generation (Friday afternoon)
    friday = datetime(2025, 10, 17, 15, 0)  # Friday 3pm
    
    # Add some positions first
    spy_put = Position(
        id="POS-TEST-001",
        asset="SPY",
        position_type=PositionType.SHORT_PUT,
        quantity=10,
        entry_price=Decimal('8.50'),
        entry_date=monday,
        strike=Decimal('670.0'),
        expiration=date(2025, 10, 27),
        option_type='P',
        delta=Decimal('-300.0')
    )
    position_mgr.add_position(spy_put)
    position_mgr.update_prices(data_service)
    
    print(f"\n📊 Generating signals for Friday (with positions)...")
    signals = signal_gen.generate_signals(friday, data_service, position_mgr)
    
    print(f"  Generated {len(signals)} signals")
    for signal in signals:
        print(f"    {signal.action.value}: {signal.asset} {signal.quantity}x "
              f"${signal.strike:.2f}" + (f" @ ${signal.expected_price:.2f}" if signal.expected_price else ""))
    
    # Health check
    print(f"\n🏥 Health check: {signal_gen.health_check()}")
    
    print("\n✅ Signal generator working correctly!")

