#!/usr/bin/env python3
"""
24/7 Paper Trading Daemon
Runs continuously, executes daily trading cycles, auto-restarts on failure
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

import asyncio
import signal
import time
from datetime import datetime, time as dt_time
from typing import Optional
import logging
import pytz
from pathlib import Path

# Import our trading engine
from src.main.trading_engine import TradingEngine
from src.common.config import load_config

# Setup logging
log_dir = Path(__file__).parent.parent.parent / 'logs'
log_dir.mkdir(exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_dir / 'daemon.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class PaperTradingDaemon:
    """24/7 daemon for paper trading"""
    
    def __init__(self):
        self.running = False
        self.engine: Optional[TradingEngine] = None
        self.timezone = pytz.timezone('US/Eastern')
        self.config = None
        self.last_trading_day = None
        self.consecutive_failures = 0
        self.max_consecutive_failures = 5
        
    async def initialize(self):
        """Initialize the trading engine"""
        try:
            logger.info("🚀 Initializing Paper Trading Daemon...")
            
            # Load configuration
            self.config = load_config(environment='paper')
            logger.info(f"Config loaded: {self.config.environment}")
            
            # Initialize trading engine (happens in __init__)
            self.engine = TradingEngine(self.config)
            
            # Verify engine is healthy
            health = self.engine.health_check()
            logger.info(f"Engine health: {health}")
            
            logger.info("✅ Trading engine initialized successfully")
            self.consecutive_failures = 0
            return True
            
        except Exception as e:
            logger.error(f"❌ Initialization failed: {e}", exc_info=True)
            self.consecutive_failures += 1
            return False
    
    async def start(self):
        """Start the daemon"""
        self.running = True
        logger.info("=" * 70)
        logger.info("24/7 PAPER TRADING DAEMON STARTING")
        logger.info("=" * 70)
        
        # Setup signal handlers for graceful shutdown
        signal.signal(signal.SIGTERM, self._signal_handler)
        signal.signal(signal.SIGINT, self._signal_handler)
        
        # Initialize
        if not await self.initialize():
            logger.error("Failed to initialize. Retrying in 60 seconds...")
            await asyncio.sleep(60)
            return await self.start()  # Retry
        
        # Main loop
        while self.running:
            try:
                await self._main_loop_iteration()
                
            except Exception as e:
                logger.error(f"💥 Main loop error: {e}", exc_info=True)
                self.consecutive_failures += 1
                
                if self.consecutive_failures >= self.max_consecutive_failures:
                    logger.critical(f"❌ {self.max_consecutive_failures} consecutive failures. Reinitializing...")
                    await self.shutdown()
                    await asyncio.sleep(30)
                    await self.initialize()
                else:
                    logger.warning(f"Failure {self.consecutive_failures}/{self.max_consecutive_failures}. Continuing...")
                    await asyncio.sleep(60)
            
            # Check every 60 seconds
            try:
                await asyncio.sleep(60)
            except asyncio.CancelledError:
                logger.info("Main loop cancelled")
                break
        
        # Graceful shutdown
        await self.shutdown()
        logger.info("✅ Daemon stopped gracefully")
    
    async def _main_loop_iteration(self):
        """Single iteration of the main loop"""
        now = datetime.now(self.timezone)
        current_date = now.date()
        
        # Check if it's a trading day
        if not self._is_trading_day(now):
            if now.hour == 0 and now.minute == 0:  # Log once per day
                logger.info(f"📅 {current_date} is not a trading day (weekend/holiday)")
            return
        
        # Check if we should execute today's trading cycle
        should_trade, reason = self._should_execute_cycle(now)
        
        if should_trade:
            logger.info(f"🔔 Executing trading cycle: {reason}")
            await self._execute_trading_cycle(now)
            self.last_trading_day = current_date
            self.consecutive_failures = 0  # Reset on success
        
        # Health check
        await self._health_check()
    
    def _is_trading_day(self, dt: datetime) -> bool:
        """Check if it's a trading day"""
        # Skip weekends
        if dt.weekday() >= 5:  # Saturday=5, Sunday=6
            return False
        
        # TODO: Add holiday calendar
        # For now, assume all weekdays are trading days
        
        return True
    
    def _should_execute_cycle(self, dt: datetime) -> tuple[bool, str]:
        """Determine if we should execute a trading cycle"""
        current_date = dt.date()
        current_time = dt.time()
        
        # Already traded today?
        if self.last_trading_day == current_date:
            return False, "Already traded today"
        
        # Market open time (9:30 AM ET)
        market_open = dt_time(9, 30)
        market_close = dt_time(16, 0)
        
        # Execute between 9:30 AM and 10:00 AM
        execution_window_start = dt_time(9, 30)
        execution_window_end = dt_time(10, 0)
        
        if execution_window_start <= current_time <= execution_window_end:
            return True, f"Market open execution window ({current_time.strftime('%H:%M')})"
        
        # If it's after 10 AM and we haven't traded yet, execute anyway
        if current_time > execution_window_end and self.last_trading_day != current_date:
            return True, f"Late execution (missed window, executing now at {current_time.strftime('%H:%M')})"
        
        return False, "Outside execution window"
    
    async def _execute_trading_cycle(self, timestamp: datetime):
        """Execute a full trading cycle"""
        try:
            logger.info("=" * 70)
            logger.info(f"📊 TRADING CYCLE: {timestamp.strftime('%Y-%m-%d %H:%M:%S %Z')}")
            logger.info("=" * 70)
            
            # Run the daily cycle (synchronous method)
            result = self.engine.run_daily_cycle(timestamp)
            
            # Log results
            logger.info("📈 Trading Cycle Results:")
            logger.info(f"   NAV: ${result.get('nav', 0):,.2f}")
            logger.info(f"   Signals Generated: {result.get('signals_generated', 0)}")
            logger.info(f"   Trades Executed: {result.get('trades_executed', 0)}")
            logger.info(f"   Status: {result.get('status', 'unknown')}")
            
            errors = result.get('errors', [])
            if errors:
                logger.warning(f"   Errors: {len(errors)}")
                for error in errors[:5]:  # Show first 5
                    logger.warning(f"     - {error}")
            
            logger.info("=" * 70)
            
        except Exception as e:
            logger.error(f"❌ Trading cycle failed: {e}", exc_info=True)
            raise
    
    async def _health_check(self):
        """Perform health checks"""
        try:
            if self.engine:
                health = self.engine.health_check()
                
                # Handle both boolean and dict status values
                unhealthy_services = []
                for name, status in health.items():
                    if name == 'overall':
                        continue  # Skip overall status
                    
                    # Status can be bool or dict with 'healthy' key
                    if isinstance(status, bool):
                        if not status:
                            unhealthy_services.append(name)
                    elif isinstance(status, dict):
                        if not status.get('healthy', False):
                            unhealthy_services.append(name)
                
                if unhealthy_services:
                    logger.warning(f"⚠️  Unhealthy services: {', '.join(unhealthy_services)}")
                
        except Exception as e:
            logger.error(f"Health check failed: {e}")
    
    def _signal_handler(self, signum, frame):
        """Handle shutdown signals"""
        if not self.running:
            # Already shutting down, force exit
            logger.warning("Force exit!")
            import sys
            sys.exit(0)
        
        logger.info(f"📡 Received signal {signum}. Shutting down gracefully...")
        self.running = False
        
        # Cancel all running tasks
        for task in asyncio.all_tasks():
            task.cancel()
    
    async def shutdown(self):
        """Graceful shutdown"""
        logger.info("🛑 Shutting down daemon...")
        
        if self.engine:
            try:
                self.engine.stop()
                logger.info("✅ Engine shutdown complete")
            except Exception as e:
                logger.error(f"Error during shutdown: {e}")


async def main():
    """Main entry point"""
    daemon = PaperTradingDaemon()
    
    try:
        await daemon.start()
    except KeyboardInterrupt:
        logger.info("⌨️  Keyboard interrupt received")
        await daemon.shutdown()
    except asyncio.CancelledError:
        logger.info("🛑 Tasks cancelled, shutting down...")
        await daemon.shutdown()
    except Exception as e:
        logger.error(f"💥 Fatal error: {e}", exc_info=True)
        await daemon.shutdown()
        return 1
    
    return 0


if __name__ == '__main__':
    try:
        exit_code = asyncio.run(main())
        sys.exit(exit_code)
    except KeyboardInterrupt:
        logger.info("🛑 Forced exit")
        sys.exit(0)

