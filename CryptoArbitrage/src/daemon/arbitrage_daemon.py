#!/usr/bin/env python3
"""
24/7 Crypto Arbitrage Daemon
Continuously monitors exchanges and executes arbitrage opportunities
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

import asyncio
import signal
from datetime import datetime
from typing import Optional
import logging
from pathlib import Path

from ..common.config import load_config
from ..exchanges.exchange_manager import ExchangeManager
from ..data.price_aggregator import PriceAggregator
from ..detection.arbitrage_detector import ArbitrageDetector
from ..execution.arbitrage_executor import ArbitrageExecutor
from ..risk.arbitrage_risk_manager import ArbitrageRiskManager
from ..database.repository import ArbitrageRepository
from ..monitoring.performance_tracker import PerformanceTracker

# Setup logging
log_dir = Path(__file__).parent.parent.parent / 'logs'
log_dir.mkdir(exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_dir / 'arbitrage.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class ArbitrageDaemon:
    """
    24/7 crypto arbitrage daemon
    
    Runs continuously:
    - Monitors price feeds from all exchanges
    - Detects arbitrage opportunities
    - Executes profitable trades
    - Tracks performance
    """
    
    def __init__(self, simulation_mode: bool = False):
        """
        Initialize daemon
        
        Args:
            simulation_mode: If True, uses mock data and doesn't execute real trades
        """
        self.simulation_mode = simulation_mode
        self.running = False
        self.config = None
        
        # Components (initialized in setup)
        self.exchange_manager: Optional[ExchangeManager] = None
        self.price_aggregator: Optional[PriceAggregator] = None
        self.detector: Optional[ArbitrageDetector] = None
        self.executor: Optional[ArbitrageExecutor] = None
        self.risk_manager: Optional[ArbitrageRiskManager] = None
        self.repository: Optional[ArbitrageRepository] = None
        self.performance_tracker: Optional[PerformanceTracker] = None
        
        # Statistics
        self.opportunities_detected = 0
        self.opportunities_executed = 0
        
    async def initialize(self) -> bool:
        """Initialize all components"""
        try:
            logger.info("=" * 70)
            logger.info("🚀 CRYPTO ARBITRAGE DAEMON INITIALIZING")
            logger.info("=" * 70)
            logger.info(f"Mode: {'SIMULATION' if self.simulation_mode else 'LIVE'}")
            logger.info("")
            
            # Load configuration
            self.config = load_config()
            logger.info(f"✅ Config loaded: {self.config.environment}")
            
            # Initialize database
            self.repository = ArbitrageRepository(self.config.database.sqlite_path)
            logger.info("✅ Database initialized")
            
            # Initialize exchange manager
            self.exchange_manager = ExchangeManager(self.config, self.simulation_mode)
            connection_status = await self.exchange_manager.connect_all()
            
            connected_count = sum(1 for v in connection_status.values() if v)
            if connected_count < 2:
                logger.error(f"Only {connected_count} exchanges connected. Need at least 2.")
                return False
            
            logger.info(f"✅ Connected to {connected_count} exchanges")
            
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
            
            # Initialize executor
            self.executor = ArbitrageExecutor(
                self.exchange_manager,
                self.config.strategy
            )
            logger.info("✅ Arbitrage executor initialized")
            
            # Initialize risk manager
            self.risk_manager = ArbitrageRiskManager(self.config.risk)
            logger.info("✅ Risk manager initialized")
            
            # Initialize performance tracker
            self.performance_tracker = PerformanceTracker(self.repository)
            logger.info("✅ Performance tracker initialized")
            
            logger.info("")
            logger.info("=" * 70)
            logger.info("✅ ALL SYSTEMS INITIALIZED")
            logger.info("=" * 70)
            logger.info(f"Monitoring {len(self.config.strategy.trading_pairs)} pairs")
            logger.info(f"Min spread: {self.config.strategy.min_spread_pct}%")
            logger.info(f"Max position: ${self.config.strategy.max_position_size_usd}")
            logger.info("=" * 70)
            logger.info("")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Initialization failed: {e}", exc_info=True)
            return False
    
    async def start(self):
        """Start the daemon"""
        self.running = True
        
        # Setup signal handlers
        signal.signal(signal.SIGTERM, self._signal_handler)
        signal.signal(signal.SIGINT, self._signal_handler)
        
        # Initialize
        if not await self.initialize():
            logger.error("Initialization failed. Exiting.")
            return
        
        # Start price monitoring
        await self.price_aggregator.start_monitoring(self.config.strategy.trading_pairs)
        
        # Give websockets time to connect
        logger.info("Waiting for price feeds to stabilize...")
        await asyncio.sleep(5)
        
        # Main loop
        logger.info("🔍 Starting arbitrage detection loop...")
        logger.info("")
        
        while self.running:
            try:
                await self._detection_cycle()
                
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
            logger.info(f"🎯 Found {len(opportunities)} opportunities")
            
            # Sort by profitability
            opportunities.sort(key=lambda x: x.net_spread_pct, reverse=True)
            
            # Execute best opportunity
            for opp in opportunities[:3]:  # Try top 3
                # Save opportunity to database
                self.repository.save_opportunity(opp)
                self.opportunities_detected += 1
                
                # Get balances for validation
                buy_balance = await self.exchange_manager.get_exchange(opp.buy_exchange).get_balance()
                sell_balance = await self.exchange_manager.get_exchange(opp.sell_exchange).get_balance()
                
                # Validate with risk manager
                approved, reason = self.risk_manager.validate_opportunity(opp, buy_balance, sell_balance)
                
                if not approved:
                    logger.info(f"⚠️  Opportunity rejected: {reason}")
                    continue
                
                # Execute arbitrage
                execution = await self.executor.execute_arbitrage(opp)
                
                if execution:
                    # Save execution
                    self.repository.save_execution(execution)
                    self.opportunities_executed += 1
                    
                    # Record for risk tracking
                    profit = execution.realized_profit_usd or Decimal('0')
                    self.risk_manager.record_execution(execution, execution.is_successful, profit)
                    
                    # Only execute one at a time
                    break
    
    def _signal_handler(self, signum, frame):
        """Handle shutdown signals"""
        logger.info(f"📡 Received signal {signum}. Shutting down...")
        self.running = False
        
        # Cancel all async tasks
        for task in asyncio.all_tasks():
            task.cancel()
    
    async def shutdown(self):
        """Graceful shutdown"""
        logger.info("🛑 Shutting down arbitrage daemon...")
        
        # Print final statistics
        logger.info("")
        logger.info("=" * 70)
        logger.info("📊 SESSION STATISTICS")
        logger.info("=" * 70)
        logger.info(f"Opportunities detected: {self.opportunities_detected}")
        logger.info(f"Opportunities executed: {self.opportunities_executed}")
        logger.info(f"Execution rate: {self.opportunities_executed / max(1, self.opportunities_detected):.1%}")
        
        if self.performance_tracker:
            perf = self.performance_tracker.calculate_current_performance()
            logger.info(f"Total profit: ${perf.get('total_profit_usd', 0):.2f}")
            logger.info(f"Success rate: {perf.get('success_rate', 0):.1%}")
        
        logger.info("=" * 70)
        logger.info("")
        
        # Disconnect exchanges
        if self.exchange_manager:
            await self.exchange_manager.disconnect_all()
        
        logger.info("✅ Shutdown complete")


async def main():
    """Main entry point"""
    # Check if simulation mode
    simulation = os.getenv('CRYPTO_ENV', 'development') in ['development', 'simulation']
    
    daemon = ArbitrageDaemon(simulation_mode=simulation)
    
    try:
        await daemon.start()
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


