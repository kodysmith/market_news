"""
Unified Strategy Runner

Runs the trading strategy in any mode (backtest, paper, live).
Same core logic, different adapters.
"""

from typing import Dict, Any, Optional
from datetime import datetime
import logging
from .mode_factory import ModeFactory
from .config import ConfigLoader
from .core.regime_detector import RegimeDetector
from .core.portfolio_engine import PortfolioEngine
from .core.hedging import HedgingModule
from .core.risk_manager import RiskManager

logger = logging.getLogger(__name__)


class StrategyRunner:
    """Unified runner for all trading modes"""

    def __init__(self, mode: str, config_path: Optional[str] = None):
        self.mode = mode
        self.config_loader = ConfigLoader(config_path)
        self.config = self.config_loader.load(mode)

        # Create mode-specific adapters
        self.data_loader = ModeFactory.create_data_loader(mode, self.config)
        self.state_store = ModeFactory.create_state_store(mode, self.config)
        self.executor = ModeFactory.create_executor(
            mode, self.config, self.state_store,
            initial_capital=self.config.get('initial_capital', 100000)
        )

        # Create core modules (shared across all modes)
        self.regime_detector = RegimeDetector(self.config)
        self.portfolio_engine = PortfolioEngine(self.config)
        self.hedging_module = HedgingModule(self.config)
        self.risk_manager = RiskManager(self.config)

        # Set run ID for paper/live modes
        if mode in ['paper', 'live']:
            run_id = self._create_run()
            if hasattr(self.executor, 'set_run_id'):
                self.executor.set_run_id(run_id)

    def _create_run(self) -> str:
        """Create a new run record"""
        run_data = {
            'mode': self.mode,
            'status': 'running',
            'started_at': datetime.now().isoformat(),
            'portfolio_value': self.config.get('initial_capital', 100000),
            'config': self.config
        }
        return self.state_store.save_run(run_data)

    def run(self, date: Optional[datetime] = None) -> Dict[str, Any]:
        """
        Run the trading strategy for one day

        Args:
            date: Date to run for (None = current date)

        Returns:
            Execution results
        """
        try:
            # Set date for backtest mode
            if self.mode == 'backtest' and date:
                self.data_loader.set_current_date(date)

            # 1. Load market data
            market_data = self.data_loader.load_market_data(date)
            if not market_data:
                raise ValueError("Failed to load market data")

            # 2. Detect regime
            regime = self.regime_detector.detect_regime(market_data)
            logger.info(f"Detected regime: {regime['regime']} (confidence: {regime['confidence']:.1%})")

            # 3. Calculate target weights
            current_positions = self.executor.get_positions()
            target_weights = self.portfolio_engine.calculate_weights(regime, current_positions)

            # 4. Validate weights
            validation = self.risk_manager.validate_weights(target_weights)
            if not validation['is_valid']:
                logger.warning(f"Weight validation failed: {validation['violations']}")
                # Use conservative weights if validation fails
                target_weights = self.config.get('base_weights', {'SPY': 0.33, 'QQQ': 0.33, 'TQQQ': 0.34})

            # 5. Get current prices
            tickers = list(target_weights.keys())
            prices = self.data_loader.get_current_prices(tickers)

            # 6. Calculate rebalance trades
            portfolio_value = self.executor.get_portfolio_value()
            trades = self.portfolio_engine.calculate_rebalance_trades(
                current_positions, target_weights, portfolio_value
            )

            # 7. Validate trades
            trade_validation = self.risk_manager.validate_trades(
                trades, current_positions, portfolio_value
            )
            if not trade_validation['is_valid']:
                logger.warning(f"Trade validation failed: {trade_validation['violations']}")

            # 8. Execute trades
            executed_trades = []
            for ticker, trade_amount in trades.items():
                if trade_amount == 0:
                    continue

                if ticker not in prices:
                    logger.warning(f"No price available for {ticker}, skipping trade")
                    continue

                price = prices[ticker]
                quantity = trade_amount / price

                try:
                    result = self.executor.execute_trade(ticker, quantity, 'market', price)
                    executed_trades.append(result)
                except Exception as e:
                    logger.error(f"Error executing trade for {ticker}: {e}")

            # 9. Update prices for backtest mode
            if self.mode == 'backtest':
                self.executor.update_prices(prices)

            # 10. Check hedging
            if self.hedging_module.should_hedge(regime, portfolio_value):
                hedge_size = self.hedging_module.calculate_hedge_size(portfolio_value)
                logger.info(f"Hedging recommended: {hedge_size}")
                # Hedging execution would go here (future enhancement)

            # 11. Save state
            final_positions = self.executor.get_positions()
            final_value = self.executor.get_portfolio_value()

            if self.mode == 'backtest':
                self.state_store.save_positions(final_positions, self._create_run())
            else:
                self.state_store.save_positions(final_positions, self.executor.run_id)
                self.state_store.save_regime_history(self.executor.run_id, regime)
                self.state_store.update_run(self.executor.run_id, {
                    'portfolio_value': final_value,
                    'regime': regime,
                    'status': 'completed',
                    'completed_at': datetime.now().isoformat()
                })

            # 12. Return results
            return {
                'mode': self.mode,
                'date': date.isoformat() if date else datetime.now().isoformat(),
                'regime': regime,
                'target_weights': target_weights,
                'executed_trades': executed_trades,
                'final_positions': final_positions,
                'portfolio_value': final_value,
                'validation': validation,
                'trade_validation': trade_validation
            }

        except Exception as e:
            logger.error(f"Error running strategy: {e}", exc_info=True)
            raise


def run_strategy(mode: str, config_path: Optional[str] = None,
                start_date: Optional[str] = None,
                end_date: Optional[str] = None) -> Dict[str, Any]:
    """
    Run strategy in specified mode

    Args:
        mode: 'backtest', 'paper', or 'live'
        config_path: Path to config directory
        start_date: Start date for backtest (YYYY-MM-DD)
        end_date: End date for backtest (YYYY-MM-DD)

    Returns:
        Results dictionary
    """
    runner = StrategyRunner(mode, config_path)

    if mode == 'backtest' and start_date and end_date:
        # Run backtest over date range
        from datetime import datetime, timedelta
        start = datetime.strptime(start_date, '%Y-%m-%d')
        end = datetime.strptime(end_date, '%Y-%m-%d')

        results = []
        current = start
        while current <= end:
            try:
                result = runner.run(current)
                results.append(result)
                current += timedelta(days=1)
            except Exception as e:
                logger.error(f"Error on {current}: {e}")
                current += timedelta(days=1)

        return {
            'mode': 'backtest',
            'start_date': start_date,
            'end_date': end_date,
            'results': results,
            'final_value': results[-1]['portfolio_value'] if results else 0
        }
    else:
        # Single run (paper or live)
        return runner.run()
