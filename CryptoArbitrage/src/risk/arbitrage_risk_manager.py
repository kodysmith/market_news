"""
Arbitrage Risk Manager
Pre-trade validation and risk controls
"""

from typing import Tuple, Dict
from decimal import Decimal
from datetime import datetime, timedelta
import logging

from ..common.models import ArbitrageOpportunity, ExchangeBalance
from ..common.config import RiskConfig

logger = logging.getLogger(__name__)


class ArbitrageRiskManager:
    """
    Risk management for arbitrage execution
    
    Validates opportunities before execution to ensure:
    - Sufficient balances
    - Spread still exists
    - Risk limits not breached
    - Rate limits respected
    """
    
    def __init__(self, config: RiskConfig):
        """
        Initialize risk manager
        
        Args:
            config: Risk configuration
        """
        self.config = config
        
        # Circuit breaker tracking
        self.consecutive_failures = 0
        self.circuit_breaker_active = False
        self.circuit_breaker_until: Optional[datetime] = None
        
        # Rate limiting
        self.trades_last_hour = []
        self.trades_today = 0
        self.current_date = datetime.now().date()
        
        # Loss tracking
        self.daily_pnl = Decimal('0')
        self.last_reset_date = datetime.now().date()
        
        logger.info("ArbitrageRiskManager initialized")
        logger.info(f"Max daily loss: ${config.max_daily_loss_usd}")
        logger.info(f"Max trades/hour: {config.max_trades_per_hour}")
    
    def validate_opportunity(self,
                            opportunity: ArbitrageOpportunity,
                            buy_balance: ExchangeBalance,
                            sell_balance: ExchangeBalance) -> Tuple[bool, str]:
        """
        Validate arbitrage opportunity before execution
        
        Args:
            opportunity: Opportunity to validate
            buy_balance: Balance on buy exchange
            sell_balance: Balance on sell exchange
        
        Returns:
            Tuple of (approved, reason)
        """
        # Reset daily counters if new day
        self._check_daily_reset()
        
        # Check circuit breaker
        if self.circuit_breaker_active:
            if datetime.now() < self.circuit_breaker_until:
                return False, "Circuit breaker active"
            else:
                self._reset_circuit_breaker()
        
        # Check daily loss limit
        if self.daily_pnl <= -self.config.max_daily_loss_usd:
            logger.warning(f"Daily loss limit reached: ${abs(self.daily_pnl):.2f}")
            self._activate_circuit_breaker(timedelta(hours=24))
            return False, f"Daily loss limit exceeded: ${abs(self.daily_pnl):.2f}"
        
        # Check rate limits
        if len(self.trades_last_hour) >= self.config.max_trades_per_hour:
            return False, f"Hourly trade limit reached ({self.config.max_trades_per_hour})"
        
        if self.trades_today >= self.config.max_trades_per_day:
            return False, f"Daily trade limit reached ({self.config.max_trades_per_day})"
        
        # Check balances
        quote_needed = opportunity.buy_price * opportunity.max_quantity
        base_needed = opportunity.max_quantity
        
        buy_quote_balance = buy_balance.available.get(opportunity.quote_asset, Decimal('0'))
        sell_base_balance = sell_balance.available.get(opportunity.base_asset, Decimal('0'))
        
        if buy_quote_balance < quote_needed:
            return False, f"Insufficient {opportunity.quote_asset} on {opportunity.buy_exchange}"
        
        if sell_base_balance < base_needed:
            return False, f"Insufficient {opportunity.base_asset} on {opportunity.sell_exchange}"
        
        # Check minimum balances
        if buy_quote_balance < self.config.min_balance_per_exchange_usd:
            return False, f"Balance too low on {opportunity.buy_exchange}"
        
        # Check spread still profitable
        if opportunity.net_spread_pct < Decimal('0.5'):
            return False, "Spread too small (< 0.5%)"
        
        # Check price freshness
        if opportunity.price_staleness_ms > 5000:  # 5 seconds
            return False, "Price data too stale"
        
        # All checks passed
        return True, "Validated"
    
    def record_execution(self, execution: any, success: bool, profit: Decimal):
        """
        Record execution result for risk tracking
        
        Args:
            execution: Execution object
            success: Whether execution succeeded
            profit: Realized profit (can be negative)
        """
        now = datetime.now()
        
        # Update daily P&L
        self.daily_pnl += profit
        
        # Update rate limiting
        self.trades_last_hour.append(now)
        self.trades_today += 1
        
        # Clean old trades from last hour
        hour_ago = now - timedelta(hours=1)
        self.trades_last_hour = [t for t in self.trades_last_hour if t > hour_ago]
        
        # Update consecutive failures
        if success:
            self.consecutive_failures = 0
        else:
            self.consecutive_failures += 1
            
            # Check if circuit breaker should activate
            if self.consecutive_failures >= self.config.max_consecutive_failures:
                logger.error(f"Circuit breaker triggered: {self.consecutive_failures} consecutive failures")
                self._activate_circuit_breaker(timedelta(minutes=30))
    
    def _check_daily_reset(self):
        """Reset daily counters if new day"""
        today = datetime.now().date()
        
        if today != self.current_date:
            logger.info(f"New day - resetting daily counters. Yesterday P&L: ${self.daily_pnl:.2f}")
            self.daily_pnl = Decimal('0')
            self.trades_today = 0
            self.current_date = today
            self.last_reset_date = today
    
    def _activate_circuit_breaker(self, duration: timedelta):
        """Activate circuit breaker"""
        self.circuit_breaker_active = True
        self.circuit_breaker_until = datetime.now() + duration
        logger.error(f"🚨 CIRCUIT BREAKER ACTIVATED until {self.circuit_breaker_until}")
    
    def _reset_circuit_breaker(self):
        """Reset circuit breaker"""
        self.circuit_breaker_active = False
        self.circuit_breaker_until = None
        self.consecutive_failures = 0
        logger.info("✅ Circuit breaker reset")
    
    def get_risk_status(self) -> Dict:
        """Get current risk status"""
        return {
            'circuit_breaker_active': self.circuit_breaker_active,
            'circuit_breaker_until': self.circuit_breaker_until.isoformat() if self.circuit_breaker_until else None,
            'consecutive_failures': self.consecutive_failures,
            'daily_pnl': float(self.daily_pnl),
            'trades_last_hour': len(self.trades_last_hour),
            'trades_today': self.trades_today,
            'can_trade': not self.circuit_breaker_active and self.daily_pnl > -self.config.max_daily_loss_usd
        }


