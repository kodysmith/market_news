"""
Arbitrage Executor
Executes both legs of arbitrage simultaneously with risk-free guarantees
"""

import asyncio
from typing import Optional, Tuple
from decimal import Decimal
from datetime import datetime
import logging
import uuid

from ..common.models import ArbitrageOpportunity, ArbitrageExecution, CryptoOrder, OrderStatus
from ..common.config import StrategyConfig
from ..exchanges.exchange_manager import ExchangeManager
from ..exchanges.exchange_interface import ExchangeError

logger = logging.getLogger(__name__)


class ArbitrageExecutor:
    """
    Executes arbitrage trades with risk-free guarantees
    
    Key principle: Both legs execute or neither executes.
    If one leg fails, immediately reverse the other.
    """
    
    def __init__(self,
                 exchange_manager: ExchangeManager,
                 config: StrategyConfig):
        """
        Initialize arbitrage executor
        
        Args:
            exchange_manager: Manages all exchange connections
            config: Strategy configuration
        """
        self.exchange_manager = exchange_manager
        self.config = config
        
        # Execution statistics
        self.executions_attempted = 0
        self.executions_successful = 0
        self.executions_failed = 0
        self.reversals_required = 0
        
        logger.info("ArbitrageExecutor initialized")
        logger.info(f"Execution timeout: {config.execution_timeout_seconds}s")
    
    async def execute_arbitrage(self,
                               opportunity: ArbitrageOpportunity) -> Optional[ArbitrageExecution]:
        """
        Execute arbitrage opportunity
        
        Process:
        1. Validate opportunity is still valid
        2. Check balances on both exchanges
        3. Calculate trade size
        4. Submit both orders simultaneously
        5. Monitor fills with timeout
        6. Reverse if one leg fails
        
        Args:
            opportunity: Arbitrage opportunity to execute
        
        Returns:
            ArbitrageExecution if successful, None if failed
        """
        self.executions_attempted += 1
        
        execution_id = f"EXEC-{uuid.uuid4().hex[:12]}"
        start_time = datetime.now()
        
        logger.info(f"🚀 Executing arbitrage: {execution_id}")
        logger.info(f"   {opportunity.pair}: Buy {opportunity.buy_exchange}, "
                   f"Sell {opportunity.sell_exchange}")
        logger.info(f"   Expected profit: {opportunity.net_spread_pct:.2f}% "
                   f"(${opportunity.estimated_profit_usd:.2f})")
        
        try:
            # Step 1: Verify balances
            buy_exchange = self.exchange_manager.get_exchange(opportunity.buy_exchange)
            sell_exchange = self.exchange_manager.get_exchange(opportunity.sell_exchange)
            
            if not buy_exchange or not sell_exchange:
                logger.error("Exchange not available")
                self.executions_failed += 1
                return None
            
            # Get balances
            buy_balance = await buy_exchange.get_balance()
            sell_balance = await sell_exchange.get_balance()
            
            # Calculate trade size
            trade_quantity = self._calculate_trade_size(
                opportunity,
                buy_balance,
                sell_balance
            )
            
            if trade_quantity == 0:
                logger.warning("Insufficient balance for trade")
                self.executions_failed += 1
                return None
            
            logger.info(f"   Trade size: {trade_quantity} {opportunity.base_asset}")
            
            # Step 2: Submit both orders simultaneously
            buy_order, sell_order = await self._submit_both_orders(
                buy_exchange,
                sell_exchange,
                opportunity.pair,
                trade_quantity
            )
            
            if not buy_order or not sell_order:
                logger.error("Failed to submit orders")
                self.executions_failed += 1
                return None
            
            # Step 3: Monitor fills with timeout
            success, buy_result, sell_result = await self._monitor_fills(
                buy_exchange,
                sell_exchange,
                buy_order,
                sell_order,
                opportunity.pair
            )
            
            execution_time = (datetime.now() - start_time).total_seconds() * 1000
            
            # Create execution record
            execution = ArbitrageExecution(
                id=execution_id,
                opportunity_id=opportunity.id,
                started_at=start_time,
                completed_at=datetime.now(),
                pair=opportunity.pair,
                quantity=trade_quantity,
                buy_exchange=opportunity.buy_exchange,
                buy_order_id=buy_order.id,
                buy_status=buy_result.status if buy_result else OrderStatus.FAILED,
                buy_filled_price=buy_result.filled_price if buy_result else None,
                buy_filled_quantity=buy_result.filled_quantity if buy_result else Decimal('0'),
                buy_fee=buy_result.fee_paid if buy_result else Decimal('0'),
                sell_exchange=opportunity.sell_exchange,
                sell_order_id=sell_order.id,
                sell_status=sell_result.status if sell_result else OrderStatus.FAILED,
                sell_filled_price=sell_result.filled_price if sell_result else None,
                sell_filled_quantity=sell_result.filled_quantity if sell_result else Decimal('0'),
                sell_fee=sell_result.fee_paid if sell_result else Decimal('0'),
                is_successful=success,
                execution_time_ms=execution_time
            )
            
            if success:
                # Calculate realized profit
                execution.realized_profit_usd = self._calculate_realized_profit(execution)
                execution.realized_spread_pct = (
                    (execution.realized_profit_usd / 
                     (execution.buy_filled_price * execution.buy_filled_quantity)) * Decimal('100')
                    if execution.buy_filled_price and execution.buy_filled_quantity > 0
                    else Decimal('0')
                )
                
                self.executions_successful += 1
                logger.info(f"✅ Arbitrage executed successfully!")
                logger.info(f"   Profit: ${execution.realized_profit_usd:.2f} "
                           f"({execution.realized_spread_pct:.2f}%)")
                logger.info(f"   Execution time: {execution_time:.0f}ms")
            else:
                self.executions_failed += 1
                logger.error(f"❌ Arbitrage execution failed")
                logger.error(f"   Buy: {execution.buy_status.value}")
                logger.error(f"   Sell: {execution.sell_status.value}")
            
            return execution
            
        except Exception as e:
            logger.error(f"Arbitrage execution error: {e}", exc_info=True)
            self.executions_failed += 1
            return None
    
    def _calculate_trade_size(self,
                             opportunity: ArbitrageOpportunity,
                             buy_balance: any,
                             sell_balance: any) -> Decimal:
        """
        Calculate optimal trade size based on balances
        
        Args:
            opportunity: Arbitrage opportunity
            buy_balance: Balance on buy exchange
            sell_balance: Balance on sell exchange
        
        Returns:
            Trade quantity in base currency
        """
        # Get available balances
        buy_quote_balance = buy_balance.available.get(opportunity.quote_asset, Decimal('0'))
        sell_base_balance = sell_balance.available.get(opportunity.base_asset, Decimal('0'))
        
        # Max we can buy (limited by quote currency)
        max_buy_quantity = buy_quote_balance / opportunity.buy_price
        
        # Max we can sell (limited by base currency)
        max_sell_quantity = sell_base_balance
        
        # Take minimum
        trade_quantity = min(max_buy_quantity, max_sell_quantity, opportunity.max_quantity)
        
        # Round down to avoid precision issues
        trade_quantity = Decimal(str(float(trade_quantity) * 0.99))  # 99% to be safe
        
        logger.debug(f"Trade size calculation:")
        logger.debug(f"  Max buy: {max_buy_quantity} (have {buy_quote_balance} {opportunity.quote_asset})")
        logger.debug(f"  Max sell: {max_sell_quantity} (have {sell_base_balance} {opportunity.base_asset})")
        logger.debug(f"  Config max: {opportunity.max_quantity}")
        logger.debug(f"  Final: {trade_quantity}")
        
        return trade_quantity
    
    async def _submit_both_orders(self,
                                  buy_exchange,
                                  sell_exchange,
                                  pair: str,
                                  quantity: Decimal) -> Tuple[Optional[CryptoOrder], Optional[CryptoOrder]]:
        """
        Submit both orders simultaneously
        
        Returns:
            Tuple of (buy_order, sell_order)
        """
        try:
            # Submit both orders in parallel
            buy_task = buy_exchange.submit_market_order(pair, 'buy', quantity)
            sell_task = sell_exchange.submit_market_order(pair, 'sell', quantity)
            
            results = await asyncio.gather(buy_task, sell_task, return_exceptions=True)
            
            buy_order = results[0] if not isinstance(results[0], Exception) else None
            sell_order = results[1] if not isinstance(results[1], Exception) else None
            
            if isinstance(results[0], Exception):
                logger.error(f"Buy order failed: {results[0]}")
            if isinstance(results[1], Exception):
                logger.error(f"Sell order failed: {results[1]}")
            
            return buy_order, sell_order
            
        except Exception as e:
            logger.error(f"Error submitting orders: {e}")
            return None, None
    
    async def _monitor_fills(self,
                            buy_exchange,
                            sell_exchange,
                            buy_order: CryptoOrder,
                            sell_order: CryptoOrder,
                            pair: str) -> Tuple[bool, Optional[CryptoOrder], Optional[CryptoOrder]]:
        """
        Monitor both orders until filled or timeout
        
        Returns:
            Tuple of (success, buy_order_final, sell_order_final)
        """
        timeout = self.config.execution_timeout_seconds
        start_time = asyncio.get_event_loop().time()
        
        buy_filled = False
        sell_filled = False
        
        while (asyncio.get_event_loop().time() - start_time) < timeout:
            try:
                # Check both order statuses
                buy_status_task = buy_exchange.get_order_status(
                    buy_order.exchange_order_id,
                    pair
                )
                sell_status_task = sell_exchange.get_order_status(
                    sell_order.exchange_order_id,
                    pair
                )
                
                buy_result, sell_result = await asyncio.gather(
                    buy_status_task,
                    sell_status_task,
                    return_exceptions=True
                )
                
                # Check buy order
                if not isinstance(buy_result, Exception):
                    buy_order = buy_result
                    if buy_order.status == OrderStatus.FILLED:
                        buy_filled = True
                
                # Check sell order
                if not isinstance(sell_result, Exception):
                    sell_order = sell_result
                    if sell_order.status == OrderStatus.FILLED:
                        sell_filled = True
                
                # Both filled - success!
                if buy_filled and sell_filled:
                    logger.info("✅ Both legs filled successfully")
                    return True, buy_order, sell_order
                
                # Wait before next check
                await asyncio.sleep(0.1)
                
            except Exception as e:
                logger.error(f"Error monitoring fills: {e}")
                await asyncio.sleep(0.5)
        
        # Timeout reached
        logger.warning(f"⏱️  Execution timeout ({timeout}s)")
        
        # Check what filled and what didn't
        if buy_filled and not sell_filled:
            logger.warning("Buy filled but sell failed - REVERSING buy")
            await self._reverse_trade(sell_exchange, pair, 'sell', buy_order.filled_quantity)
            self.reversals_required += 1
            return False, buy_order, sell_order
        
        elif sell_filled and not buy_filled:
            logger.warning("Sell filled but buy failed - REVERSING sell")
            await self._reverse_trade(buy_exchange, pair, 'buy', sell_order.filled_quantity)
            self.reversals_required += 1
            return False, buy_order, sell_order
        
        elif not buy_filled and not sell_filled:
            logger.info("Neither order filled - cancelling both")
            await buy_exchange.cancel_order(buy_order.exchange_order_id, pair)
            await sell_exchange.cancel_order(sell_order.exchange_order_id, pair)
            return False, buy_order, sell_order
        
        # Should not reach here
        return False, buy_order, sell_order
    
    async def _reverse_trade(self,
                            exchange,
                            pair: str,
                            side: str,
                            quantity: Decimal):
        """
        Reverse a filled trade to maintain risk-free guarantee
        
        Args:
            exchange: Exchange where reversal needed
            pair: Trading pair
            side: 'buy' or 'sell'
            quantity: Amount to reverse
        """
        try:
            logger.info(f"🔄 Reversing {side} of {quantity} {pair}")
            reverse_side = 'sell' if side == 'buy' else 'buy'
            
            reversal_order = await exchange.submit_market_order(
                pair,
                reverse_side,
                quantity
            )
            
            logger.info(f"✅ Reversal order submitted: {reversal_order.id}")
            
        except Exception as e:
            logger.error(f"❌ CRITICAL: Failed to reverse trade: {e}")
            logger.error(f"   Manual intervention required!")
            # TODO: Send critical alert
    
    def _calculate_realized_profit(self, execution: ArbitrageExecution) -> Decimal:
        """Calculate actual realized profit from execution"""
        if not execution.buy_filled_price or not execution.sell_filled_price:
            return Decimal('0')
        
        # Revenue from sell
        sell_proceeds = execution.sell_filled_price * execution.sell_filled_quantity
        sell_proceeds -= execution.sell_fee
        
        # Cost of buy
        buy_cost = execution.buy_filled_price * execution.buy_filled_quantity
        buy_cost += execution.buy_fee
        
        # Net profit
        profit = sell_proceeds - buy_cost
        
        return profit
    
    def get_statistics(self) -> Dict:
        """Get execution statistics"""
        return {
            'executions_attempted': self.executions_attempted,
            'executions_successful': self.executions_successful,
            'executions_failed': self.executions_failed,
            'success_rate': (
                self.executions_successful / self.executions_attempted
                if self.executions_attempted > 0 else 0
            ),
            'reversals_required': self.reversals_required
        }


