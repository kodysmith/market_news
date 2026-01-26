#!/usr/bin/env python3
"""
Paper Trading Mode for Crypto Arbitrage

Runs the arbitrage system in simulation mode:
- Connects to real exchanges (read-only)
- Detects real opportunities
- Simulates trade execution without placing real orders
- Tracks performance as if trades were executed
- Logs all opportunities and their lifecycles
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import asyncio
import signal
from datetime import datetime, timedelta
from decimal import Decimal
import logging
from pathlib import Path
from typing import Optional

from src.common.config import load_config
from src.exchanges.exchange_manager import ExchangeManager
from src.data.price_aggregator import PriceAggregator
from src.detection.arbitrage_detector import ArbitrageDetector
from src.risk.arbitrage_risk_manager import ArbitrageRiskManager
from src.database.repository import ArbitrageRepository
from src.monitoring.opportunity_lifecycle_tracker import OpportunityLifecycleTracker
from src.common.models import ArbitrageExecution, OrderStatus, ArbitrageOpportunity

# Setup logging
log_dir = Path(__file__).parent.parent / 'logs'
log_dir.mkdir(exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_dir / 'paper_trading.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class PaperTradingSimulator:
    """
    Paper trading simulator
    
    Simulates trade execution with realistic assumptions:
    - Execution latency
    - Slippage
    - Partial fills
    - Market impact
    """
    
    def __init__(self, config):
        self.config = config
        
        # Simulation parameters
        self.execution_latency_ms = 150  # Average execution time
        self.slippage_pct = Decimal('0.05')  # 0.05% average slippage
        self.fill_success_rate = 0.85  # 85% of trades execute successfully
        
    async def simulate_execution(self, opportunity: ArbitrageOpportunity) -> ArbitrageExecution:
        """
        Simulate executing an arbitrage opportunity
        
        Args:
            opportunity: The opportunity to execute
            
        Returns:
            Simulated execution result
        """
        start_time = datetime.now()
        execution_id = f"SIM-{start_time.strftime('%Y%m%d-%H%M%S')}-{opportunity.id[:8]}"
        
        # Simulate network latency
        await asyncio.sleep(self.execution_latency_ms / 1000)
        
        # Determine if execution is successful (85% success rate)
        import random
        is_successful = random.random() < self.fill_success_rate
        
        if not is_successful:
            # Failed execution
            logger.warning(f"❌ Simulated execution FAILED: {opportunity.pair} - "
                          f"Market moved or liquidity issues")
            
            return ArbitrageExecution(
                id=execution_id,
                opportunity_id=opportunity.id,
                started_at=start_time,
                completed_at=datetime.now(),
                pair=opportunity.pair,
                quantity=Decimal('0'),
                buy_exchange=opportunity.buy_exchange,
                buy_order_id=f"SIM-BUY-{opportunity.id[:8]}",
                buy_status=OrderStatus.FAILED,
                sell_exchange=opportunity.sell_exchange,
                sell_order_id=f"SIM-SELL-{opportunity.id[:8]}",
                sell_status=OrderStatus.FAILED,
                is_successful=False,
                execution_time_ms=(datetime.now() - start_time).total_seconds() * 1000,
                had_failures=True,
                failure_reason="Simulated market movement"
            )
        
        # Successful execution with slippage
        quantity = opportunity.max_quantity * Decimal('0.9')  # Use 90% of max
        
        # Apply slippage to prices
        buy_price_with_slippage = opportunity.buy_price * (1 + self.slippage_pct / 100)
        sell_price_with_slippage = opportunity.sell_price * (1 - self.slippage_pct / 100)
        
        # Calculate fees
        buy_fee = buy_price_with_slippage * quantity * opportunity.buy_fee_pct / 100
        sell_fee = sell_price_with_slippage * quantity * opportunity.sell_fee_pct / 100
        
        # Calculate profit
        buy_cost = buy_price_with_slippage * quantity + buy_fee
        sell_proceeds = sell_price_with_slippage * quantity - sell_fee
        realized_profit = sell_proceeds - buy_cost
        
        # Calculate realized spread
        realized_spread_pct = (realized_profit / buy_cost) * 100
        
        end_time = datetime.now()
        execution_time_ms = (end_time - start_time).total_seconds() * 1000
        
        logger.info(f"✅ Simulated execution SUCCESS: {opportunity.pair}")
        logger.info(f"   Buy {quantity:.8f} @ ${buy_price_with_slippage:.2f} on {opportunity.buy_exchange}")
        logger.info(f"   Sell {quantity:.8f} @ ${sell_price_with_slippage:.2f} on {opportunity.sell_exchange}")
        logger.info(f"   Profit: ${realized_profit:.2f} ({realized_spread_pct:.2f}%)")
        logger.info(f"   Execution time: {execution_time_ms:.0f}ms")
        
        return ArbitrageExecution(
            id=execution_id,
            opportunity_id=opportunity.id,
            started_at=start_time,
            completed_at=end_time,
            pair=opportunity.pair,
            quantity=quantity,
            buy_exchange=opportunity.buy_exchange,
            buy_order_id=f"SIM-BUY-{opportunity.id[:8]}",
            buy_status=OrderStatus.FILLED,
            buy_filled_price=buy_price_with_slippage,
            buy_filled_quantity=quantity,
            buy_fee=buy_fee,
            sell_exchange=opportunity.sell_exchange,
            sell_order_id=f"SIM-SELL-{opportunity.id[:8]}",
            sell_status=OrderStatus.FILLED,
            sell_filled_price=sell_price_with_slippage,
            sell_filled_quantity=quantity,
            sell_fee=sell_fee,
            is_successful=True,
            realized_profit_usd=realized_profit,
            realized_spread_pct=realized_spread_pct,
            execution_time_ms=execution_time_ms,
            had_failures=False,
            required_reversal=False
        )


class PaperTradingDaemon:
    """Paper trading daemon with opportunity lifecycle tracking"""
    
    def __init__(self):
        self.running = False
        self.config = None
        
        # Components
        self.exchange_manager: Optional[ExchangeManager] = None
        self.price_aggregator: Optional[PriceAggregator] = None
        self.detector: Optional[ArbitrageDetector] = None
        self.risk_manager: Optional[ArbitrageRiskManager] = None
        self.repository: Optional[ArbitrageRepository] = None
        self.lifecycle_tracker: Optional[OpportunityLifecycleTracker] = None
        self.simulator: Optional[PaperTradingSimulator] = None
        
        # Statistics
        self.opportunities_detected = 0
        self.opportunities_executed = 0
        self.total_simulated_profit = Decimal('0')
        
        # Tracking
        self.session_start = datetime.now()
    
    async def initialize(self) -> bool:
        """Initialize all components"""
        try:
            logger.info("=" * 70)
            logger.info("📝 PAPER TRADING MODE - NO REAL TRADES")
            logger.info("=" * 70)
            logger.info("")
            
            # Load configuration
            self.config = load_config()
            logger.info(f"✅ Config loaded: {self.config.environment}")
            
            # Initialize database
            self.repository = ArbitrageRepository(self.config.database.sqlite_path)
            logger.info("✅ Database initialized")
            
            # Initialize lifecycle tracker
            self.lifecycle_tracker = OpportunityLifecycleTracker(self.repository)
            logger.info("✅ Lifecycle tracker initialized")
            
            # Initialize exchange manager (read-only mode)
            self.exchange_manager = ExchangeManager(self.config, simulation_mode=True)
            connection_status = await self.exchange_manager.connect_all()
            
            connected_count = sum(1 for v in connection_status.values() if v)
            if connected_count < 2:
                logger.error(f"Only {connected_count} exchanges connected. Need at least 2.")
                return False
            
            logger.info(f"✅ Connected to {connected_count} exchanges (read-only)")
            
            # Get fee structures
            exchange_fees = {}
            for exchange_name, exchange in self.exchange_manager.exchanges.items():
                exchange_fees[exchange_name] = exchange.get_fee_structure()
            
            # Initialize price aggregator
            self.price_aggregator = PriceAggregator(
                self.exchange_manager,
                self.config.strategy.max_price_staleness_seconds
            )
            logger.info("✅ Price aggregator initialized")
            
            # Initialize arbitrage detector
            self.detector = ArbitrageDetector(
                self.price_aggregator,
                self.config.strategy,
                exchange_fees
            )
            logger.info("✅ Arbitrage detector initialized")
            
            # Initialize risk manager
            self.risk_manager = ArbitrageRiskManager(self.config.risk)
            logger.info("✅ Risk manager initialized")
            
            # Initialize paper trading simulator
            self.simulator = PaperTradingSimulator(self.config)
            logger.info("✅ Paper trading simulator initialized")
            
            logger.info("")
            logger.info("=" * 70)
            logger.info("✅ ALL SYSTEMS READY FOR PAPER TRADING")
            logger.info("=" * 70)
            logger.info(f"Monitoring {len(self.config.strategy.trading_pairs)} pairs")
            logger.info(f"Min spread: {self.config.strategy.min_spread_pct}%")
            logger.info(f"Session started: {self.session_start}")
            logger.info("=" * 70)
            logger.info("")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Initialization failed: {e}", exc_info=True)
            return False
    
    async def start(self, duration_minutes: Optional[int] = None):
        """
        Start paper trading
        
        Args:
            duration_minutes: Run for this many minutes (None = run until stopped)
        """
        self.running = True
        
        # Setup signal handlers
        signal.signal(signal.SIGTERM, self._signal_handler)
        signal.signal(signal.SIGINT, self._signal_handler)
        
        # Initialize
        if not await self.initialize():
            logger.error("Initialization failed. Exiting.")
            return
        
        # Calculate end time if duration specified
        end_time = None
        if duration_minutes:
            end_time = datetime.now() + timedelta(minutes=duration_minutes)
            logger.info(f"⏰ Will run for {duration_minutes} minutes (until {end_time.strftime('%H:%M:%S')})")
        
        # Start price monitoring
        await self.price_aggregator.start_monitoring(self.config.strategy.trading_pairs)
        
        # Give websockets time to connect
        logger.info("Waiting for price feeds to stabilize...")
        await asyncio.sleep(5)
        
        # Main loop
        logger.info("🔍 Starting opportunity detection...")
        logger.info("")
        
        last_stats_time = datetime.now()
        stats_interval = 60  # Print stats every 60 seconds
        
        while self.running:
            try:
                # Check if we should stop (time limit reached)
                if end_time and datetime.now() >= end_time:
                    logger.info(f"⏰ Time limit reached ({duration_minutes} minutes)")
                    break
                
                await self._detection_cycle()
                
                # Print statistics periodically
                now = datetime.now()
                if (now - last_stats_time).total_seconds() >= stats_interval:
                    self._print_session_stats()
                    last_stats_time = now
                
                # Check every 100ms for opportunities
                await asyncio.sleep(0.1)
                
            except asyncio.CancelledError:
                logger.info("Detection loop cancelled")
                break
            except Exception as e:
                logger.error(f"Error in main loop: {e}", exc_info=True)
                await asyncio.sleep(1)
        
        # Shutdown
        await self.shutdown()
    
    async def _detection_cycle(self):
        """Single iteration of detection and execution"""
        # Detect opportunities
        opportunities = self.detector.detect_opportunities()
        
        if opportunities:
            # Track all opportunities in lifecycle tracker
            for opp in opportunities:
                tracked_opp = self.lifecycle_tracker.track_opportunity(opp)
                self.opportunities_detected += 1
            
            logger.info(f"🎯 Found {len(opportunities)} opportunities")
            
            # Sort by profitability
            opportunities.sort(key=lambda x: x.net_spread_pct, reverse=True)
            
            # Simulate execution of best opportunity
            best_opp = opportunities[0]
            
            # Validate with risk manager
            # For paper trading, we'll use mock balances
            mock_balance = {'total_usd': Decimal('10000'), 'USDT': Decimal('10000')}
            approved, reason = self.risk_manager.validate_opportunity(best_opp, mock_balance, mock_balance)
            
            if not approved:
                logger.info(f"⚠️  Opportunity rejected: {reason}")
            else:
                # Simulate execution
                execution = await self.simulator.simulate_execution(best_opp)
                
                # Save execution
                self.repository.save_execution(execution)
                self.opportunities_executed += 1
                
                if execution.is_successful and execution.realized_profit_usd:
                    self.total_simulated_profit += execution.realized_profit_usd
        
        # Check for expired opportunities
        self.lifecycle_tracker.check_and_expire_stale_opportunities(opportunities, staleness_threshold_seconds=1.0)
    
    def _signal_handler(self, signum, frame):
        """Handle shutdown signals"""
        logger.info(f"📡 Received signal {signum}. Shutting down...")
        self.running = False
    
    def _print_session_stats(self):
        """Print current session statistics"""
        runtime_seconds = (datetime.now() - self.session_start).total_seconds()
        runtime_minutes = runtime_seconds / 60
        
        lifecycle_stats = self.lifecycle_tracker.get_statistics()
        
        logger.info("")
        logger.info("=" * 70)
        logger.info("📊 SESSION STATISTICS")
        logger.info("=" * 70)
        logger.info(f"Runtime: {runtime_minutes:.1f} minutes")
        logger.info(f"Opportunities detected: {self.opportunities_detected}")
        logger.info(f"Unique opportunities tracked: {lifecycle_stats['total_tracked']}")
        logger.info(f"Opportunities expired: {lifecycle_stats['total_expired']}")
        logger.info(f"Active opportunities: {lifecycle_stats['active_now']}")
        logger.info(f"Opportunities executed (simulated): {self.opportunities_executed}")
        logger.info(f"Total simulated profit: ${self.total_simulated_profit:.2f}")
        logger.info("")
        logger.info("OPPORTUNITY LIFECYCLE METRICS:")
        logger.info(f"  Average duration: {lifecycle_stats['avg_duration_seconds']:.2f} seconds")
        logger.info(f"  Median duration (P50): {lifecycle_stats['p50_duration_ms']:.0f} ms")
        logger.info(f"  P90 duration: {lifecycle_stats['p90_duration_ms']:.0f} ms")
        logger.info(f"  P95 duration: {lifecycle_stats['p95_duration_ms']:.0f} ms")
        logger.info(f"  Min duration: {lifecycle_stats['min_duration_ms']:.0f} ms")
        logger.info(f"  Max duration: {lifecycle_stats['max_duration_ms']:.0f} ms")
        logger.info("=" * 70)
        logger.info("")
    
    async def shutdown(self):
        """Graceful shutdown"""
        logger.info("🛑 Shutting down paper trading daemon...")
        
        # Print final statistics
        self._print_session_stats()
        
        # Disconnect exchanges
        if self.exchange_manager:
            await self.exchange_manager.disconnect_all()
        
        logger.info("✅ Shutdown complete")


async def main():
    """Main entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Paper Trading for Crypto Arbitrage')
    parser.add_argument('--duration', type=int, default=None,
                       help='Run for N minutes (default: run until stopped)')
    args = parser.parse_args()
    
    daemon = PaperTradingDaemon()
    
    try:
        await daemon.start(duration_minutes=args.duration)
    except KeyboardInterrupt:
        logger.info("⌨️  Keyboard interrupt")
    except Exception as e:
        logger.error(f"Fatal error: {e}", exc_info=True)
        return 1
    
    return 0


if __name__ == '__main__':
    try:
        exit_code = asyncio.run(main())
        sys.exit(exit_code)
    except KeyboardInterrupt:
        logger.info("🛑 Forced exit")
        sys.exit(0)


