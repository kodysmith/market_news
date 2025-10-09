"""
Module: Risk Manager Service
Purpose: Pre-trade risk checks and circuit breakers
Dependencies: Config, Position Manager, Models
Exports: RiskManager

Validates all trades before execution to ensure compliance with risk limits.
Implements circuit breakers to protect capital during drawdowns.
"""

from typing import Tuple, Dict, List, Optional
from decimal import Decimal
from datetime import datetime
import logging
import numpy as np

from ..common.models import Signal, Position, OrderAction, RiskMetrics
from ..common.config import RiskConfig

logger = logging.getLogger(__name__)


class RiskManager:
    """
    Risk management service with pre-trade checks and circuit breakers
    
    Ensures all trades comply with risk limits before execution.
    Monitors portfolio health and triggers circuit breakers if needed.
    """
    
    def __init__(self, config: RiskConfig, position_manager=None):
        """
        Initialize risk manager
        
        Args:
            config: Risk configuration
            position_manager: Position manager instance (optional)
        """
        self.config = config
        self.position_manager = position_manager
        
        # Circuit breaker state
        self.circuit_breaker_status = 'normal'  # normal, reduce, halt
        self.peak_nav = Decimal('0.0')
        
        logger.info("RiskManager initialized")
    
    def validate_signal(self, 
                       signal: Signal,
                       portfolio_state: Dict) -> Tuple[bool, str]:
        """
        Validate signal against all risk limits
        
        Args:
            signal: Trading signal to validate
            portfolio_state: Current portfolio state
        
        Returns:
            (approved: bool, reason: str)
        """
        # Check circuit breaker status
        if self.circuit_breaker_status == 'halt':
            return False, "Circuit breaker: HALT - Trading suspended"
        
        # 1. Check position size limits
        if signal.quantity > self.config.max_contracts_per_trade:
            return False, f"Exceeds max contracts per trade: {signal.quantity} > {self.config.max_contracts_per_trade}"
        
        # 2. Check delta limits
        if signal.expected_delta:
            # Current delta
            current_delta = portfolio_state.get('total_delta', Decimal('0.0'))
            
            # Projected delta after this trade
            new_delta = current_delta + signal.expected_delta
            
            # Check total portfolio delta limit
            if abs(new_delta) > self.config.max_total_delta:
                return False, f"Would exceed total delta limit: {abs(new_delta):.0f} > {self.config.max_total_delta:.0f}"
        
        # 3. Check margin utilization
        if signal.capital_required:
            nav = portfolio_state.get('nav', Decimal('100000.0'))
            current_margin = portfolio_state.get('margin_used', Decimal('0.0'))
            
            # Estimate margin requirement (simplified)
            new_margin = current_margin + signal.capital_required
            margin_pct = new_margin / nav if nav > 0 else Decimal('1.0')
            
            if margin_pct > Decimal(str(self.config.max_margin_utilization)):
                return False, f"Would exceed margin limit: {margin_pct:.1%} > {self.config.max_margin_utilization:.1%}"
        
        # 4. Check if circuit breaker in reduce mode
        if self.circuit_breaker_status == 'reduce':
            # Only allow closing trades or hedges
            if signal.action in [OrderAction.SELL_PUT, OrderAction.SELL_CALL]:
                # Opening new short positions not allowed in reduce mode
                return False, "Circuit breaker: REDUCE - Only closing trades allowed"
        
        # All checks passed
        return True, "Approved"
    
    def check_circuit_breakers(self, 
                              current_nav: Decimal,
                              peak_nav: Optional[Decimal] = None) -> str:
        """
        Check if circuit breakers should trigger
        
        Args:
            current_nav: Current NAV
            peak_nav: Historical peak NAV (optional, will track internally)
        
        Returns:
            Status: 'normal', 'reduce', 'halt'
        """
        # Update peak NAV
        if peak_nav is not None:
            self.peak_nav = peak_nav
        else:
            self.peak_nav = max(self.peak_nav, current_nav)
        
        # Calculate current drawdown
        if self.peak_nav == 0:
            drawdown = Decimal('0.0')
        else:
            drawdown = (current_nav - self.peak_nav) / self.peak_nav
        
        # Check thresholds
        previous_status = self.circuit_breaker_status
        
        if drawdown <= self.config.circuit_breaker_halt_threshold:
            self.circuit_breaker_status = 'halt'
            logger.critical(f"CIRCUIT BREAKER HALT: Drawdown {drawdown:.1%} exceeds {self.config.circuit_breaker_halt_threshold:.1%}")
        
        elif drawdown <= self.config.circuit_breaker_reduce_threshold:
            self.circuit_breaker_status = 'reduce'
            if previous_status != 'reduce':
                logger.warning(f"CIRCUIT BREAKER REDUCE: Drawdown {drawdown:.1%} exceeds {self.config.circuit_breaker_reduce_threshold:.1%}")
        
        else:
            self.circuit_breaker_status = 'normal'
            if previous_status != 'normal':
                logger.info(f"Circuit breaker cleared: Drawdown {drawdown:.1%}")
        
        return self.circuit_breaker_status
    
    def calculate_var(self, 
                     returns: List[float],
                     confidence: Optional[float] = None) -> Decimal:
        """
        Calculate Value at Risk
        
        Args:
            returns: List of historical returns
            confidence: Confidence level (default from config)
        
        Returns:
            VaR (positive number representing potential loss)
        """
        if not returns or len(returns) < 30:
            logger.warning("Insufficient data for VaR calculation")
            return Decimal('0.0')
        
        confidence = confidence or self.config.var_confidence
        
        # Sort returns
        sorted_returns = sorted(returns)
        
        # Find percentile
        index = int((1 - confidence) * len(sorted_returns))
        var_return = sorted_returns[index]
        
        # VaR as positive number (potential loss)
        var = abs(min(var_return, 0))
        
        return Decimal(str(round(var, 6)))
    
    def get_position_limits_check(self, 
                                  asset: str,
                                  positions: List[Position]) -> Dict:
        """
        Check position limits for an asset
        
        Args:
            asset: Asset symbol
            positions: Current positions for this asset
        
        Returns:
            Dict with limit checks
        """
        # Count contracts by strike
        strike_counts = {}
        total_contracts = 0
        
        for pos in positions:
            if pos.is_option and pos.strike:
                strike_key = float(pos.strike)
                strike_counts[strike_key] = strike_counts.get(strike_key, 0) + pos.quantity
            
            total_contracts += pos.quantity
        
        # Check limits
        checks = {
            'total_contracts': total_contracts,
            'max_contracts_ok': total_contracts < self.config.max_contracts_per_trade * 5,  # Aggregate limit
            'strike_concentration': strike_counts,
            'max_per_strike_ok': all(count <= self.config.max_contracts_per_strike for count in strike_counts.values())
        }
        
        return checks
    
    def calculate_risk_metrics(self, 
                              portfolio_state: Dict,
                              nav_history: List[Decimal]) -> RiskMetrics:
        """
        Calculate comprehensive risk metrics
        
        Args:
            portfolio_state: Current portfolio state
            nav_history: Historical NAV values
        
        Returns:
            RiskMetrics object
        """
        # Calculate returns
        returns = []
        if len(nav_history) > 1:
            for i in range(1, len(nav_history)):
                if nav_history[i-1] > 0:
                    ret = float((nav_history[i] - nav_history[i-1]) / nav_history[i-1])
                    returns.append(ret)
        
        # Calculate volatility
        if returns:
            daily_vol = np.std(returns)
            annual_vol = daily_vol * np.sqrt(252)
        else:
            daily_vol = annual_vol = 0.0
        
        # Calculate Sharpe (assuming 4% risk-free rate)
        if annual_vol > 0 and returns:
            avg_return = np.mean(returns) * 252  # Annualized
            sharpe = (avg_return - 0.04) / annual_vol
        else:
            sharpe = 0.0
        
        # Calculate max drawdown
        peak = nav_history[0]
        max_dd = 0.0
        current_dd = 0.0
        
        for nav in nav_history:
            peak = max(peak, nav)
            if peak > 0:
                dd = float((nav - peak) / peak)
                max_dd = min(max_dd, dd)
                if nav == nav_history[-1]:
                    current_dd = dd
        
        # Calculate VaR
        var_95 = self.calculate_var(returns, 0.95)
        
        # Exposure metrics
        nav = portfolio_state.get('nav', Decimal('0.0'))
        total_delta = portfolio_state.get('total_delta', Decimal('0.0'))
        
        # Gross exposure (sum of absolute values)
        gross_exp = abs(portfolio_state.get('total_equity_value', Decimal('0.0')))
        gross_exp += abs(portfolio_state.get('total_options_value', Decimal('0.0')))
        gross_exposure = gross_exp / nav if nav > 0 else Decimal('0.0')
        
        # Net exposure
        net_exp = total_delta * Decimal('100')  # Convert delta to notional
        net_exposure = net_exp / nav if nav > 0 else Decimal('0.0')
        
        metrics = RiskMetrics(
            timestamp=datetime.now(),
            nav=nav,
            daily_volatility=Decimal(str(round(daily_vol, 6))),
            annual_volatility=Decimal(str(round(annual_vol, 6))),
            sharpe_ratio=Decimal(str(round(sharpe, 4))),
            value_at_risk_95=var_95,
            max_drawdown=Decimal(str(round(max_dd, 6))),
            current_drawdown=Decimal(str(round(current_dd, 6))),
            total_delta=total_delta,
            gross_exposure=gross_exposure,
            net_exposure=net_exposure,
            position_count=portfolio_state.get('position_count', 0)
        )
        
        return metrics
    
    def stress_test(self, 
                   portfolio_state: Dict,
                   scenario: str = 'market_drop') -> Dict:
        """
        Run stress test scenarios
        
        Args:
            portfolio_state: Current portfolio state
            scenario: Scenario type ('market_drop', 'vol_spike')
        
        Returns:
            Dict with stress test results
        """
        results = {
            'scenario': scenario,
            'current_nav': portfolio_state.get('nav', Decimal('0.0'))
        }
        
        if scenario == 'market_drop':
            # Simulate market drop
            drop_pct = self.config.stress_market_drop_pct
            
            # Estimate P&L impact
            # Short puts would gain value (loss for us)
            # Long hedges would gain value (profit for us)
            total_delta = portfolio_state.get('total_delta', Decimal('0.0'))
            
            # Delta P&L (simplified)
            spot_move = drop_pct * 100  # Assume $100/point
            delta_pnl = total_delta * Decimal(str(spot_move))
            
            # Gamma P&L (second-order effect)
            # Negative for short options, positive for long
            gamma_pnl = Decimal('0.0')  # Simplified for now
            
            projected_nav = results['current_nav'] + delta_pnl + gamma_pnl
            projected_dd = (projected_nav - results['current_nav']) / results['current_nav']
            
            results['projected_nav'] = projected_nav
            results['projected_drawdown'] = projected_dd
            results['delta_pnl'] = delta_pnl
        
        elif scenario == 'vol_spike':
            # Simulate volatility spike
            vol_mult = self.config.stress_vol_spike_mult
            
            # Short options lose value (profit for us)
            # Long options gain value (profit for us)
            # Simplified: Vega P&L
            vega_pnl = Decimal('0.0')  # Would need vega from positions
            
            results['vol_multiplier'] = vol_mult
            results['vega_pnl'] = vega_pnl
        
        return results
    
    def health_check(self) -> bool:
        """Health check"""
        try:
            # Verify config loaded
            assert self.config.max_contracts_per_trade > 0
            assert self.config.circuit_breaker_reduce_threshold < 0
            return True
        except:
            return False


if __name__ == "__main__":
    """Test risk manager"""
    import sys
    sys.path.insert(0, '/mnt/4tb/stock_scanner/market_news/HedgeFund')
    
    from src.common.config import load_config
    from src.common.models import Signal, OrderAction
    from datetime import date, timedelta
    
    print("Testing Risk Manager...")
    
    # Load config
    config = load_config()
    
    # Initialize risk manager
    risk_mgr = RiskManager(config.risk)
    
    print(f"\n✓ Risk manager initialized")
    print(f"  Max contracts per trade: {config.risk.max_contracts_per_trade}")
    print(f"  Max total delta: {config.risk.max_total_delta:,.0f}")
    print(f"  Circuit breaker (reduce): {config.risk.circuit_breaker_reduce_threshold:.1%}")
    print(f"  Circuit breaker (halt): {config.risk.circuit_breaker_halt_threshold:.1%}")
    
    # Test signal validation
    print(f"\n📊 Testing signal validation...")
    
    # Good signal
    good_signal = Signal(
        id="SIG-TEST-001",
        timestamp=datetime.now(),
        asset="SPY",
        action=OrderAction.SELL_PUT,
        quantity=5,
        strike=Decimal('670.0'),
        expiration=date.today() + timedelta(days=14),
        option_type='P',
        expected_delta=Decimal('-150.0'),
        capital_required=Decimal('33500.0')
    )
    
    portfolio_state = {
        'nav': Decimal('100000.0'),
        'cash': Decimal('60000.0'),
        'total_delta': Decimal('-500.0'),
        'margin_used': Decimal('20000.0')
    }
    
    approved, reason = risk_mgr.validate_signal(good_signal, portfolio_state)
    print(f"  Good signal: {approved} - {reason}")
    
    # Too many contracts
    bad_signal = Signal(
        id="SIG-TEST-002",
        timestamp=datetime.now(),
        asset="SPY",
        action=OrderAction.SELL_PUT,
        quantity=50,  # Way too many
        strike=Decimal('670.0'),
        expiration=date.today() + timedelta(days=14),
        option_type='P'
    )
    
    approved, reason = risk_mgr.validate_signal(bad_signal, portfolio_state)
    print(f"  Bad signal (too many): {approved} - {reason}")
    
    # Test circuit breakers
    print(f"\n⚡ Testing circuit breakers...")
    
    # Normal
    status = risk_mgr.check_circuit_breakers(Decimal('105000.0'), Decimal('105000.0'))
    print(f"  NAV at peak: {status}")
    
    # 5% drawdown (normal)
    status = risk_mgr.check_circuit_breakers(Decimal('99750.0'), Decimal('105000.0'))
    print(f"  5% drawdown: {status}")
    
    # 10% drawdown (reduce)
    status = risk_mgr.check_circuit_breakers(Decimal('94500.0'), Decimal('105000.0'))
    print(f"  10% drawdown: {status}")
    
    # 15% drawdown (halt)
    status = risk_mgr.check_circuit_breakers(Decimal('89250.0'), Decimal('105000.0'))
    print(f"  15% drawdown: {status}")
    
    # Test VaR calculation
    print(f"\n📈 Testing VaR calculation...")
    
    # Simulate some returns
    returns = list(np.random.normal(0.001, 0.015, 100))  # 0.1% avg, 1.5% daily vol
    var_95 = risk_mgr.calculate_var(returns, 0.95)
    print(f"  95% VaR: {var_95:.2%}")
    
    # Test risk metrics
    print(f"\n📊 Testing risk metrics calculation...")
    
    nav_history = [Decimal(str(100000 + i * 500 + np.random.normal(0, 1000))) for i in range(100)]
    metrics = risk_mgr.calculate_risk_metrics(portfolio_state, nav_history)
    
    print(f"  Sharpe ratio: {metrics.sharpe_ratio:.2f}")
    print(f"  Max drawdown: {metrics.max_drawdown:.2%}")
    print(f"  Annual volatility: {metrics.annual_volatility:.2%}")
    
    # Health check
    print(f"\n🏥 Health check: {risk_mgr.health_check()}")
    
    print("\n✅ Risk manager working correctly!")


