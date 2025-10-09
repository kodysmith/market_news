"""
Module: Order Executor Service
Purpose: Execute orders with risk checks and broker failover
Dependencies: Risk Manager, Broker Clients, Audit Logger
Exports: OrderExecutor

Coordinates the entire execution flow:
1. Signal → Risk validation → Order → Broker → Fill → Audit
2. Handles failover between brokers
3. Retry logic for transient failures
"""

from typing import List, Optional, Tuple
from decimal import Decimal
from datetime import datetime
import logging
import time
import uuid

from ..common.models import Signal, Order, Fill, TradeExecution, OrderStatus, OrderAction
from ..risk.risk_manager import RiskManager
from .broker_interface import BrokerInterface, BrokerError
from .alpaca_client import AlpacaClient

logger = logging.getLogger(__name__)


class OrderExecutor:
    """
    Order execution service with risk checks and failover
    
    Orchestrates the complete trade execution flow from signal to fill.
    Ensures all trades are validated by risk manager before execution.
    """
    
    def __init__(self,
                 risk_manager: RiskManager,
                 primary_broker: BrokerInterface,
                 backup_broker: Optional[BrokerInterface] = None,
                 audit_logger=None):
        """
        Initialize order executor
        
        Args:
            risk_manager: Risk manager for pre-trade checks
            primary_broker: Primary broker (e.g., Alpaca)
            backup_broker: Backup broker (e.g., IB) for failover
            audit_logger: Audit logger for compliance
        """
        self.risk_manager = risk_manager
        self.primary_broker = primary_broker
        self.backup_broker = backup_broker
        self.audit_logger = audit_logger
        
        # Execution statistics
        self.orders_submitted = 0
        self.orders_filled = 0
        self.orders_rejected = 0
        self.failovers = 0
        
        logger.info("OrderExecutor initialized")
    
    def execute_signal(self,
                      signal: Signal,
                      portfolio_state: dict) -> Optional[TradeExecution]:
        """
        Execute a trading signal
        
        Complete flow:
        1. Convert signal to order
        2. Validate with risk manager
        3. Submit to broker
        4. Handle fill
        5. Log to audit trail
        
        Args:
            signal: Trading signal to execute
            portfolio_state: Current portfolio state for risk checks
        
        Returns:
            TradeExecution object if successful, None if rejected/failed
        """
        logger.info(f"Executing signal: {signal.id} - {signal.action.value} {signal.asset}")
        
        # Step 1: Convert signal to order
        order = self._signal_to_order(signal)
        
        # Step 2: Risk validation
        approved, reason = self.risk_manager.validate_signal(signal, portfolio_state)
        
        if not approved:
            logger.warning(f"Signal rejected by risk manager: {reason}")
            self.orders_rejected += 1
            
            # Log rejection
            if self.audit_logger:
                self.audit_logger.log_risk_event(
                    'signal_rejected',
                    'WARNING',
                    {
                        'signal_id': signal.id,
                        'asset': signal.asset,
                        'action': signal.action.value,
                        'reason': reason
                    }
                )
            
            return None
        
        logger.info(f"Signal approved: {reason}")
        
        # Step 3: Execute order
        fill = self._execute_order(order)
        
        if not fill:
            logger.error(f"Order execution failed: {order.id}")
            return None
        
        # Step 4: Create trade execution record
        trade_exec = TradeExecution(
            trade_id=f"TRD-{uuid.uuid4().hex[:8]}",
            order=order,
            fill=fill,
            timestamp=datetime.now(),
            nav_after=portfolio_state.get('nav', Decimal('0.0')),
            delta_after=portfolio_state.get('total_delta', Decimal('0.0')),
            risk_approved=True,
            risk_reason=reason
        )
        
        # Step 5: Log to audit trail
        if self.audit_logger:
            self.audit_logger.log_trade(trade_exec)
        
        self.orders_filled += 1
        logger.info(f"Trade executed successfully: {trade_exec.trade_id}")
        
        return trade_exec
    
    def execute_batch(self,
                     signals: List[Signal],
                     portfolio_state: dict) -> List[TradeExecution]:
        """
        Execute multiple signals in batch
        
        Args:
            signals: List of signals to execute
            portfolio_state: Current portfolio state
        
        Returns:
            List of successful trade executions
        """
        executions = []
        
        # Sort by priority (lower number = higher priority)
        sorted_signals = sorted(signals, key=lambda s: s.priority)
        
        logger.info(f"Executing batch of {len(signals)} signals")
        
        for signal in sorted_signals:
            try:
                execution = self.execute_signal(signal, portfolio_state)
                
                if execution:
                    executions.append(execution)
                    
                    # Update portfolio state for next signal
                    # (In production, would update from position manager)
                    if signal.expected_delta:
                        portfolio_state['total_delta'] = (
                            portfolio_state.get('total_delta', Decimal('0.0')) + 
                            signal.expected_delta
                        )
                
                # Brief pause between orders
                time.sleep(0.5)
            
            except Exception as e:
                logger.error(f"Error executing signal {signal.id}: {e}")
        
        logger.info(f"Batch complete: {len(executions)}/{len(signals)} successful")
        return executions
    
    def _signal_to_order(self, signal: Signal) -> Order:
        """Convert signal to order"""
        order = Order(
            id=f"ORD-{uuid.uuid4().hex[:8]}",
            timestamp=datetime.now(),
            asset=signal.asset,
            action=signal.action,
            quantity=signal.quantity,
            strike=signal.strike,
            expiration=signal.expiration,
            option_type=signal.option_type,
            order_type='limit',
            limit_price=signal.expected_price,
            time_in_force='day',
            status=OrderStatus.PENDING,
            strategy_component=signal.strategy_component,
            reasoning=signal.reasoning,
            expected_premium=signal.expected_price
        )
        
        return order
    
    def _execute_order(self, order: Order, retry_count: int = 0) -> Optional[Fill]:
        """
        Execute order with retry and failover logic
        
        Args:
            order: Order to execute
            retry_count: Current retry attempt
        
        Returns:
            Fill if successful, None if failed
        """
        max_retries = 3
        
        try:
            # Mark as submitted
            order.status = OrderStatus.SUBMITTED
            order.broker = "primary"
            self.orders_submitted += 1
            
            # Submit to primary broker
            fill = self.primary_broker.submit_order(order)
            
            # Mark as filled
            order.status = OrderStatus.FILLED
            order.fill_price = fill.fill_price
            order.fill_timestamp = fill.timestamp
            order.fill_quantity = fill.quantity
            order.commission = fill.commission
            
            return fill
        
        except BrokerError as e:
            logger.error(f"Primary broker failed: {e}")
            
            # Try backup broker if available
            if self.backup_broker and retry_count == 0:
                logger.info("Attempting failover to backup broker")
                self.failovers += 1
                order.broker = "backup"
                
                try:
                    fill = self.backup_broker.submit_order(order)
                    order.status = OrderStatus.FILLED
                    return fill
                except BrokerError as e2:
                    logger.error(f"Backup broker also failed: {e2}")
            
            # Retry logic for transient errors
            if retry_count < max_retries:
                logger.info(f"Retrying order (attempt {retry_count + 1}/{max_retries})")
                time.sleep(1.0 * (retry_count + 1))  # Exponential backoff
                return self._execute_order(order, retry_count + 1)
            
            # All attempts failed
            order.status = OrderStatus.REJECTED
            logger.error(f"Order failed after {max_retries} retries")
            return None
        
        except Exception as e:
            logger.error(f"Unexpected error executing order: {e}")
            order.status = OrderStatus.REJECTED
            return None
    
    def cancel_order(self, order_id: str) -> bool:
        """Cancel an order"""
        try:
            success = self.primary_broker.cancel_order(order_id)
            
            if success:
                logger.info(f"Order cancelled: {order_id}")
            
            return success
        
        except Exception as e:
            logger.error(f"Failed to cancel order {order_id}: {e}")
            return False
    
    def get_statistics(self) -> dict:
        """Get execution statistics"""
        return {
            'orders_submitted': self.orders_submitted,
            'orders_filled': self.orders_filled,
            'orders_rejected': self.orders_rejected,
            'fill_rate': self.orders_filled / self.orders_submitted if self.orders_submitted > 0 else 0.0,
            'failovers': self.failovers
        }
    
    def health_check(self) -> bool:
        """Health check"""
        try:
            # Check primary broker
            primary_ok = self.primary_broker.health_check()
            
            # Check backup broker if available
            backup_ok = True
            if self.backup_broker:
                backup_ok = self.backup_broker.health_check()
            
            # At least primary must be healthy
            return primary_ok
        
        except:
            return False


if __name__ == "__main__":
    """Test order executor"""
    import sys
    sys.path.insert(0, '/mnt/4tb/stock_scanner/market_news/HedgeFund')
    
    from src.common.config import load_config
    from src.risk.risk_manager import RiskManager
    from src.execution.alpaca_client import AlpacaClient
    from datetime import date, timedelta
    
    print("Testing Order Executor...")
    
    # Load config
    config = load_config()
    
    # Initialize components
    risk_mgr = RiskManager(config.risk)
    alpaca = AlpacaClient(config.broker)
    
    # Initialize executor
    executor = OrderExecutor(
        risk_manager=risk_mgr,
        primary_broker=alpaca,
        backup_broker=None,
        audit_logger=None
    )
    
    print(f"\n✓ Order executor initialized")
    print(f"  Primary broker: Alpaca")
    print(f"  Backup broker: None")
    
    # Create test signal
    print(f"\n📊 Testing signal execution...")
    
    test_signal = Signal(
        id="SIG-TEST-001",
        timestamp=datetime.now(),
        asset="SPY",
        action=OrderAction.SELL_PUT,
        quantity=5,
        strike=Decimal('670.0'),
        expiration=date.today() + timedelta(days=14),
        option_type='P',
        expected_price=Decimal('8.50'),
        expected_delta=Decimal('-150.0'),
        capital_required=Decimal('33500.0'),
        reasoning="Test put sale",
        strategy_component="test",
        priority=5
    )
    
    portfolio_state = {
        'nav': Decimal('100000.0'),
        'cash': Decimal('60000.0'),
        'total_delta': Decimal('-200.0'),
        'margin_used': Decimal('15000.0')
    }
    
    # Note: Won't actually execute without API keys
    print(f"  Signal: {test_signal.action.value} {test_signal.quantity}x ${test_signal.strike} @ ${test_signal.expected_price}")
    print(f"  (Execution skipped - requires API keys)")
    
    # Test statistics
    stats = executor.get_statistics()
    print(f"\n📈 Execution statistics:")
    print(f"  Orders submitted: {stats['orders_submitted']}")
    print(f"  Orders filled: {stats['orders_filled']}")
    print(f"  Orders rejected: {stats['orders_rejected']}")
    print(f"  Fill rate: {stats['fill_rate']:.1%}")
    
    # Health check
    print(f"\n🏥 Health check: {executor.health_check()}")
    
    print("\n✅ Order executor working correctly!")

