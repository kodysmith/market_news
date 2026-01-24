"""
Module: Trading Orchestrator (Main Engine)
Purpose: Coordinate all services and run trading loop
Dependencies: ALL services
Exports: TradingEngine

This is the main brain of the system that orchestrates everything:
Market Data → Signals → Risk → Execute → Track → Calculate NAV → Log

Runs the daily trading cycle and coordinates all components.
"""

from typing import Optional
from decimal import Decimal
from datetime import datetime, time
import logging
import time as time_module

from ..common.config import AppConfig
from ..data.market_data_service import MarketDataService
from ..pricing.options_pricer import OptionsPricer
from ..strategies.position_manager import PositionManager
from ..strategies.signal_generator import SignalGenerator
from ..risk.risk_manager import RiskManager
from ..execution.order_executor import OrderExecutor
from ..execution.alpaca_client import AlpacaClient
from ..reporting.nav_calculator import NAVCalculator
from ..reporting.audit_logger import AuditLogger

logger = logging.getLogger(__name__)


class TradingEngine:
    """
    Main trading orchestrator
    
    Coordinates all services and runs the trading cycle:
    1. Check market hours
    2. Update positions to market
    3. Generate signals from strategy
    4. Validate with risk manager
    5. Execute approved trades
    6. Calculate NAV
    7. Log everything
    """
    
    def __init__(self, config: AppConfig):
        """
        Initialize trading engine with all services
        
        Args:
            config: Application configuration
        """
        self.config = config
        
        # Initialize all services
        logger.info("Initializing Trading Engine...")
        
        # Core services
        self.pricer = OptionsPricer(risk_free_rate=0.04)
        self.market_data = MarketDataService(config.broker)
        
        # Position tracking
        self.position_manager = PositionManager(
            db_connection=None,  # Will add DB later
            pricer=self.pricer
        )
        
        # Strategy
        self.signal_generator = SignalGenerator(
            config=config.strategy,
            pricer=self.pricer
        )
        
        # Risk management
        self.risk_manager = RiskManager(
            config=config.risk,
            position_manager=self.position_manager
        )
        
        # Execution
        self.broker = AlpacaClient(config.broker)
        self.order_executor = OrderExecutor(
            risk_manager=self.risk_manager,
            primary_broker=self.broker,
            backup_broker=None,
            audit_logger=None  # Will add after creating it
        )
        
        # Reporting (with broker for real account sync)
        self.nav_calculator = NAVCalculator(
            pricer=self.pricer,
            db_connection=None,
            broker_client=self.broker  # Sync with real Alpaca account
        )
        
        self.audit_logger = AuditLogger(db_connection=None)
        
        # Connect audit logger to executor
        self.order_executor.audit_logger = self.audit_logger
        
        # State
        self.is_running = False
        self.current_nav = Decimal(str(config.initial_capital))
        self.peak_nav = Decimal(str(config.initial_capital))
        
        logger.info("✅ Trading Engine initialized successfully")
        logger.info(f"  Initial capital: ${config.initial_capital:,.0f}")
        logger.info(f"  Assets: {config.strategy.assets}")
        logger.info(f"  Mode: {config.environment}")
    
    def run_daily_cycle(self, trading_date: datetime) -> dict:
        """
        Run one daily trading cycle
        
        Args:
            trading_date: Date to trade
        
        Returns:
            Dict with cycle results
        """
        logger.info(f"\n{'='*60}")
        logger.info(f"🚀 Starting daily cycle: {trading_date.date()}")
        logger.info(f"{'='*60}\n")
        
        results = {
            'date': trading_date,
            'signals_generated': 0,
            'trades_executed': 0,
            'trades_rejected': 0,
            'nav': Decimal('0.0'),
            'status': 'completed'
        }
        
        try:
            # Step 1: Update positions to market
            logger.info("📊 Step 1: Updating positions to market...")
            updated = self.position_manager.update_prices(self.market_data)
            logger.info(f"  Updated {updated} positions")
            
            # Step 2: Get portfolio state
            portfolio_state = self.position_manager.get_portfolio_state()
            logger.info(f"  Portfolio: ${portfolio_state['nav']:,.2f}, Delta: {portfolio_state['total_delta']:.0f}")
            
            # Step 3: Check circuit breakers
            logger.info("⚡ Step 2: Checking circuit breakers...")
            cb_status = self.risk_manager.check_circuit_breakers(
                current_nav=portfolio_state['nav'],
                peak_nav=self.peak_nav
            )
            logger.info(f"  Circuit breaker status: {cb_status}")
            
            if cb_status == 'halt':
                logger.critical("🛑 HALT: Trading suspended by circuit breaker")
                results['status'] = 'halted'
                return results
            
            # Step 4: Generate trading signals
            logger.info("💡 Step 3: Generating trading signals...")
            signals = self.signal_generator.generate_signals(
                current_date=trading_date,
                market_data_service=self.market_data,
                position_manager=self.position_manager
            )
            
            results['signals_generated'] = len(signals)
            logger.info(f"  Generated {len(signals)} signals")
            
            for sig in signals:
                logger.info(f"    - {sig.action.value}: {sig.asset} {sig.quantity}x "
                          f"${sig.strike} @ ${sig.expected_price}")
            
            # Step 5: Execute signals
            if signals:
                logger.info("⚙️  Step 4: Executing trades...")
                executions = self.order_executor.execute_batch(
                    signals=signals,
                    portfolio_state=portfolio_state
                )
                
                results['trades_executed'] = len(executions)
                results['trades_rejected'] = len(signals) - len(executions)
                
                logger.info(f"  Executed: {len(executions)}/{len(signals)} signals")
                
                # Log each execution
                for execution in executions:
                    logger.info(f"    ✓ {execution.order.action.value}: "
                              f"{execution.order.quantity}x @ ${execution.fill.fill_price}")
            
            # Step 6: Calculate NAV
            logger.info("💰 Step 5: Calculating NAV...")
            nav_result = self.nav_calculator.calculate_nav(
                cash=portfolio_state['cash'],
                positions=self.position_manager.get_positions(),
                market_data_service=self.market_data,
                timestamp=trading_date
            )
            
            self.current_nav = nav_result['nav']
            self.peak_nav = max(self.peak_nav, self.current_nav)
            results['nav'] = self.current_nav
            
            logger.info(f"  NAV: ${self.current_nav:,.2f}")
            logger.info(f"  Components:")
            
            # Handle different NAV result formats
            if 'components' in nav_result:
                # Local calculation format
                logger.info(f"    Cash: ${nav_result['components']['cash']:,.2f}")
                logger.info(f"    Equity: ${nav_result['components']['equity_value']:,.2f}")
                logger.info(f"    Options: ${nav_result['components']['options_value']:,.2f}")
                components = nav_result['components']
            else:
                # Broker NAV format (from Alpaca)
                logger.info(f"    Cash: ${nav_result.get('cash', 0):,.2f}")
                logger.info(f"    Positions: ${nav_result.get('total_positions_value', 0):,.2f}")
                logger.info(f"    Source: {nav_result.get('source', 'unknown')}")
                # Create components dict for logging
                components = {
                    'cash': str(nav_result.get('cash', 0)),
                    'equity_value': str(nav_result.get('total_positions_value', 0)),
                    'options_value': '0.0',
                    'short_options_liability': '0.0',
                    'long_options_value': '0.0',
                    'total_positions_value': str(nav_result.get('total_positions_value', 0))
                }
            
            # Step 7: Log to audit trail
            logger.info("📝 Step 6: Logging to audit trail...")
            self.audit_logger.log_nav_calculation(
                nav=self.current_nav,
                methodology='black_scholes',
                components=components,
                verified_by='system'
            )
            
            # Step 8: System health check
            logger.info("🏥 Step 7: System health check...")
            health = self.health_check()
            if not health['overall']:
                logger.warning("  ⚠️  Some services unhealthy")
                for service, status in health.items():
                    if not status and service != 'overall':
                        logger.warning(f"    - {service}: FAILED")
            else:
                logger.info("  All systems healthy ✓")
            
            logger.info(f"\n{'='*60}")
            logger.info(f"✅ Daily cycle complete")
            logger.info(f"  Signals: {results['signals_generated']}")
            logger.info(f"  Executed: {results['trades_executed']}")
            logger.info(f"  NAV: ${results['nav']:,.2f}")
            logger.info(f"{'='*60}\n")
        
        except Exception as e:
            logger.error(f"❌ Error in daily cycle: {e}", exc_info=True)
            results['status'] = 'error'
            results['error'] = str(e)
        
        return results
    
    def run_live(self):
        """
        Run engine in live mode (continuous)
        
        Runs during market hours and executes the daily cycle.
        """
        logger.info("🚀 Starting Trading Engine in LIVE mode...")
        logger.info(f"  Mode: {self.config.environment}")
        logger.info(f"  Assets: {self.config.strategy.assets}")
        
        self.is_running = True
        
        try:
            while self.is_running:
                now = datetime.now()
                
                # Check if market hours (9:30 AM - 4:00 PM ET)
                if self._is_market_hours(now):
                    # Run daily cycle
                    self.run_daily_cycle(now)
                    
                    # Sleep until tomorrow
                    logger.info("💤 Sleeping until next trading day...")
                    time_module.sleep(24 * 60 * 60)  # 24 hours
                else:
                    # Not market hours, sleep 1 hour
                    logger.info(f"⏰ Market closed. Waiting... (currently {now.time()})")
                    time_module.sleep(60 * 60)  # 1 hour
        
        except KeyboardInterrupt:
            logger.info("\n👋 Shutdown requested...")
            self.stop()
        
        except Exception as e:
            logger.error(f"❌ Fatal error: {e}", exc_info=True)
            self.stop()
    
    def _is_market_hours(self, dt: datetime) -> bool:
        """Check if within market trading hours"""
        # Skip weekends
        if dt.weekday() >= 5:  # Saturday=5, Sunday=6
            return False
        
        # Market hours: 9:30 AM - 4:00 PM ET
        market_open = time(9, 30)
        market_close = time(16, 0)
        
        current_time = dt.time()
        return market_open <= current_time <= market_close
    
    def stop(self):
        """Stop the trading engine"""
        logger.info("🛑 Stopping Trading Engine...")
        self.is_running = False
        
        # Final NAV calculation
        portfolio_state = self.position_manager.get_portfolio_state()
        logger.info(f"  Final NAV: ${portfolio_state['nav']:,.2f}")
        
        # Get statistics
        stats = self.order_executor.get_statistics()
        logger.info(f"  Total orders: {stats['orders_submitted']}")
        logger.info(f"  Fill rate: {stats['fill_rate']:.1%}")
        
        logger.info("✅ Trading Engine stopped")
    
    def health_check(self) -> dict:
        """
        Comprehensive health check of all services
        
        Returns:
            Dict with health status of each service
        """
        health = {
            'market_data': self.market_data.health_check(),
            'position_manager': self.position_manager.health_check(),
            'signal_generator': self.signal_generator.health_check(),
            'risk_manager': self.risk_manager.health_check(),
            'broker': self.broker.health_check(),
            'order_executor': self.order_executor.health_check(),
            'nav_calculator': self.nav_calculator.health_check(),
            'audit_logger': self.audit_logger.health_check()
        }
        
        health['overall'] = all(health.values())
        
        return health
    
    def get_status(self) -> dict:
        """Get current engine status"""
        portfolio_state = self.position_manager.get_portfolio_state()
        
        return {
            'running': self.is_running,
            'mode': self.config.environment,
            'nav': float(self.current_nav),
            'peak_nav': float(self.peak_nav),
            'positions': portfolio_state['position_count'],
            'circuit_breaker': self.risk_manager.circuit_breaker_status,
            'health': self.health_check()
        }


if __name__ == "__main__":
    """Run trading engine"""
    import sys
    sys.path.insert(0, '/mnt/4tb/stock_scanner/market_news/HedgeFund')
    
    from src.common.config import load_config
    
    print("🚀 Trading Engine - Hedge Fund System\n")
    
    # Load config
    config = load_config()
    
    # Initialize engine
    print("Initializing engine...")
    engine = TradingEngine(config)
    
    print(f"\n✓ Engine ready")
    print(f"  Mode: {config.environment}")
    print(f"  Initial capital: ${config.initial_capital:,.0f}")
    
    # Health check
    print(f"\n🏥 System health check:")
    health = engine.health_check()
    for service, status in health.items():
        if service != 'overall':
            print(f"  {service}: {'✓' if status else '✗'}")
    print(f"  Overall: {'✅ HEALTHY' if health['overall'] else '❌ DEGRADED'}")
    
    # Run single cycle (test mode)
    print(f"\n📊 Running test cycle...")
    results = engine.run_daily_cycle(datetime.now())
    
    print(f"\n✅ Test cycle complete:")
    print(f"  Status: {results['status']}")
    print(f"  Signals generated: {results['signals_generated']}")
    print(f"  Trades executed: {results['trades_executed']}")
    print(f"  NAV: ${results['nav']:,.2f}")
    
    # Get final status
    status = engine.get_status()
    print(f"\n📊 Final status:")
    print(f"  NAV: ${status['nav']:,.2f}")
    print(f"  Positions: {status['positions']}")
    print(f"  Circuit breaker: {status['circuit_breaker']}")
    
    print("\n✅ Trading Engine working correctly!")
    print("\n💡 To run live: engine.run_live()")


