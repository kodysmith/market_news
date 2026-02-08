"""
Local Runner Script

Simple script to run the trading system locally in any mode.
Useful for testing and development without needing Flask.
"""

import sys
import os
import argparse
import logging
from datetime import datetime

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from trading.runner import run_strategy, StrategyRunner

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def run_backtest(start_date: str, end_date: str, config_path: str = 'config', enable_options_hedging: bool = False):
    """Run backtest mode"""
    logger.info(f"Starting backtest from {start_date} to {end_date}")
    
    if enable_options_hedging:
        logger.info("⚠️  Options hedging enabled (positions calculated but NOT executed)")
        # Set environment variable to override config
        import os
        os.environ['ENABLE_OPTIONS_HEDGING'] = 'true'
    
    try:
        results = run_strategy(
            mode='backtest',
            config_path=config_path,
            start_date=start_date,
            end_date=end_date
        )
        
        print("\n" + "="*70)
        print("BACKTEST RESULTS")
        print("="*70)
        print(f"Start Date: {start_date}")
        print(f"End Date: {end_date}")
        
        if results.get('results'):
            initial_value = results['results'][0].get('portfolio_value', 0)
            final_value = results['final_value']
            total_return = (final_value - initial_value) / initial_value if initial_value > 0 else 0
            
            # Count total trades
            total_trades = sum(len(r.get('executed_trades', [])) for r in results['results'])
            
            print(f"Initial Value: ${initial_value:,.2f}")
            print(f"Final Value: ${final_value:,.2f}")
            print(f"Total Return: {total_return:.2%}")
            print(f"Number of Trading Days: {len(results['results'])}")
            print(f"Total Trades Executed: {total_trades}")
        else:
            print(f"Final Value: ${results.get('final_value', 0):,.2f}")
            print("⚠️  No results - check data files and logs")
        
        print("="*70)
        print(f"\nResults saved to: data/results/")
        
        return results
        
    except Exception as e:
        logger.error(f"Backtest failed: {e}", exc_info=True)
        raise


def run_paper_or_live(mode: str, config_path: str = 'config'):
    """Run paper or live trading mode"""
    if mode not in ['paper', 'live']:
        raise ValueError(f"Mode must be 'paper' or 'live', got: {mode}")
    
    logger.info(f"Starting {mode} trading mode")
    
    # Check environment variables
    if mode == 'paper':
        if not os.environ.get('SUPABASE_URL') or not os.environ.get('SUPABASE_KEY'):
            logger.warning("SUPABASE_URL and SUPABASE_KEY not set. Paper mode requires Supabase.")
            logger.info("Set them with: export SUPABASE_URL=... export SUPABASE_KEY=...")
            raise ValueError("Supabase credentials required for paper mode")
    
    if mode == 'live':
        if not os.environ.get('ALPACA_API_KEY') or not os.environ.get('ALPACA_API_SECRET'):
            logger.warning("ALPACA_API_KEY and ALPACA_API_SECRET not set. Live mode requires broker API keys.")
            raise ValueError("Broker API keys required for live mode")
    
    try:
        runner = StrategyRunner(mode, config_path)
        result = runner.run()
        
        print("\n" + "="*70)
        print(f"{mode.upper()} TRADING RESULTS")
        print("="*70)
        print(f"Date: {result.get('date', 'N/A')}")
        print(f"Regime: {result.get('regime', {}).get('regime', 'N/A')}")
        print(f"Confidence: {result.get('regime', {}).get('confidence', 0):.1%}")
        print(f"Portfolio Value: ${result.get('portfolio_value', 0):,.2f}")
        
        positions = result.get('final_positions', {})
        if positions:
            print("\nPositions:")
            for ticker, qty in positions.items():
                if qty != 0:
                    print(f"  {ticker}: {qty:.4f} shares")
        
        trades = result.get('executed_trades', [])
        if trades:
            print(f"\nExecuted Trades: {len(trades)}")
            for trade in trades:
                print(f"  {trade.get('ticker', 'N/A')}: {trade.get('quantity', 0):.4f} @ ${trade.get('price', 0):.2f}")
        
        print("="*70)
        
        return result
        
    except Exception as e:
        logger.error(f"{mode.capitalize()} trading failed: {e}", exc_info=True)
        raise


def main():
    parser = argparse.ArgumentParser(
        description='Run trading system locally in any mode',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run backtest
  python run_local.py --mode backtest --start-date 2024-01-01 --end-date 2024-01-31
  
  # Run backtest with options hedging enabled
  python run_local.py --mode backtest --start-date 2024-01-01 --end-date 2024-01-31 --enable-options-hedging
  
  # Run paper trading
  export SUPABASE_URL="https://your-project.supabase.co"
  export SUPABASE_KEY="your-anon-key"
  python run_local.py --mode paper
  
  # Run live trading (⚠️ WARNING: uses real money)
  export ALPACA_API_KEY="your-key"
  export ALPACA_API_SECRET="your-secret"
  python run_local.py --mode live
        """
    )
    
    parser.add_argument(
        '--mode',
        type=str,
        required=True,
        choices=['backtest', 'paper', 'live'],
        help='Trading mode: backtest, paper, or live'
    )
    
    parser.add_argument(
        '--start-date',
        type=str,
        help='Start date for backtest (YYYY-MM-DD). Required for backtest mode.'
    )
    
    parser.add_argument(
        '--end-date',
        type=str,
        help='End date for backtest (YYYY-MM-DD). Required for backtest mode.'
    )
    
    parser.add_argument(
        '--config-path',
        type=str,
        default='config',
        help='Path to config directory (default: config)'
    )
    
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Enable verbose logging'
    )
    
    parser.add_argument(
        '--enable-options-hedging',
        action='store_true',
        default=False,
        help='Enable options hedging (default: False). Requires enable_hedging=true in config.'
    )
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Validate arguments
    if args.mode == 'backtest':
        if not args.start_date or not args.end_date:
            parser.error("--start-date and --end-date are required for backtest mode")
        run_backtest(args.start_date, args.end_date, args.config_path, args.enable_options_hedging)
    else:
        if args.start_date or args.end_date:
            logger.warning(f"start-date and end-date are ignored in {args.mode} mode")
        run_paper_or_live(args.mode, args.config_path)


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nInterrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Fatal error: {e}", exc_info=True)
        sys.exit(1)

