#!/usr/bin/env python3
"""
Simple test for optimizer
"""

from backtest_options_strategy import OptionsProtectionStrategy

def test_single_backtest():
    # Create strategy
    strategy = OptionsProtectionStrategy(
        ticker='AMD',
        initial_capital=1000000,
        start_date='2020-01-01',
        end_date='2021-01-01'
    )

    # Run backtest
    results = strategy.run_backtest()

    print(f"Final Value: ${results['final_value']:,.0f}")
    print(f"Total Return: {results['total_return']:.1%}")

if __name__ == "__main__":
    test_single_backtest()
