#!/usr/bin/env python3
"""
Test the Aggressive Growth Strategy
"""

from aggressive_growth_strategy import AggressiveGrowthStrategy

def test_aggressive_growth():
    """Test aggressive growth strategy"""
    print("🧪 Testing Aggressive Growth Strategy")
    print("=" * 50)

    # Create strategy
    strategy = AggressiveGrowthStrategy(
        initial_capital=100000,  # $100k account
        core_allocation=0.4,      # 40% in TQQQ
        income_allocation=0.3,    # 30% in income (bull put spreads)
        momentum_allocation=0.2,  # 20% in momentum (bull call spreads)
        moonshot_allocation=0.1,  # 10% in moonshots
        trade_frequency_days=7,   # Weekly trading
        max_loss_per_trade=0.05,  # 5% max loss per trade
        target_win_rate=0.5       # Target 50% win rate (more realistic)
    )

    # Run backtest
    results = strategy.run_backtest()

    # Print results
    strategy.print_summary(results)

    print("\n📊 Strategy Summary:")
    print(f"Initial Capital: ${results['initial_capital']:,.0f}")
    print(f"Final Value: ${results['final_value']:,.0f}")
    print(f"Total Return: {results['total_return']:.1%}")
    print(f"Benchmark (TQQQ): {results['benchmark_return']:.1%}")
    print(f"Excess Return: {results['excess_return']:.1%}")

    return results

if __name__ == "__main__":
    results = test_aggressive_growth()
