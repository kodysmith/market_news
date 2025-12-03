"""
Backtest Runner

Runs backtest using the modular trading system.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from trading.runner import run_strategy
import argparse
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description='Run trading strategy backtest')
    parser.add_argument('--start-date', type=str, required=True,
                       help='Start date (YYYY-MM-DD)')
    parser.add_argument('--end-date', type=str, required=True,
                       help='End date (YYYY-MM-DD)')
    parser.add_argument('--config-path', type=str, default='config',
                       help='Path to config directory')
    parser.add_argument('--output-dir', type=str, default='data/results',
                       help='Output directory for results')

    args = parser.parse_args()

    logger.info(f"Starting backtest from {args.start_date} to {args.end_date}")

    try:
        results = run_strategy(
            mode='backtest',
            config_path=args.config_path,
            start_date=args.start_date,
            end_date=args.end_date
        )

        logger.info(f"Backtest completed")
        logger.info(f"Final portfolio value: ${results['final_value']:,.2f}")

        # Print summary
        if results.get('results'):
            initial_value = results['results'][0].get('portfolio_value', 0)
            final_value = results['final_value']
            total_return = (final_value - initial_value) / initial_value if initial_value > 0 else 0

            print("\n" + "="*70)
            print("BACKTEST SUMMARY")
            print("="*70)
            print(f"Start Date: {args.start_date}")
            print(f"End Date: {args.end_date}")
            print(f"Initial Value: ${initial_value:,.2f}")
            print(f"Final Value: ${final_value:,.2f}")
            print(f"Total Return: {total_return:.2%}")
            print("="*70)

        return results

    except Exception as e:
        logger.error(f"Backtest failed: {e}", exc_info=True)
        raise


if __name__ == '__main__':
    main()
