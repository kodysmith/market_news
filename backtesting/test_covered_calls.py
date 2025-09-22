#!/usr/bin/env python3
"""
Test the Covered Calls Strategy
"""

from covered_calls_strategy import CoveredCallsStrategy

def test_covered_calls(ticker='TQQQ'):
    """Test covered calls strategy"""
    print(f"🧪 Testing Covered Calls Strategy on {ticker}")
    print("=" * 50)

    # Create strategy
    strategy = CoveredCallsStrategy(
        ticker=ticker,
        initial_capital=250000,  # $250k for shares
        call_dte=30,
        min_ev_threshold=0.02,  # 2% EV minimum
        roll_dte_threshold=10,
        trade_frequency_days=7
    )

    # Run backtest
    results = strategy.run_backtest()

    # Print results
    strategy.print_summary(results)

    print("\n📊 Strategy Summary:")
    print(f"Initial Shares: {results.get('initial_shares', 'N/A')}")
    print(f"Final Shares: {results['shares_owned']}")
    print(f"Total Trades: {results['total_trades']}")
    print(f"Premium Collected: ${sum(t.get('proceeds', 0) for t in results['trades'] if t['action'] == 'SELL_COVERED_CALL'):,.0f}")

    return results

if __name__ == "__main__":
    import sys
    ticker = sys.argv[1] if len(sys.argv) > 1 else 'TQQQ'
    results = test_covered_calls(ticker)
