#!/usr/bin/env python3
"""
Debug Aggressive Growth Strategy
"""

from aggressive_growth_strategy import AggressiveGrowthStrategy
import pandas as pd

def debug_strategy():
    # Create strategy
    strategy = AggressiveGrowthStrategy(
        initial_capital=100000,
        core_allocation=0.4,
        income_allocation=0.3,
        momentum_allocation=0.2,
        moonshot_allocation=0.1,
        trade_frequency_days=7,
        max_loss_per_trade=0.05,
        target_win_rate=0.4  # Lower threshold for testing
    )

    # Buy core position first - use later date for volatility
    start_date = pd.to_datetime('2020-02-15')
    print(f"Start date: {start_date}")

    # Check if date exists in data
    tqqq_data = strategy._get_asset_data('TQQQ')
    print(f"TQQQ data available from {tqqq_data.index[0]} to {tqqq_data.index[-1]}")
    print(f"Date in TQQQ data: {start_date in tqqq_data.index}")

    if start_date in tqqq_data.index:
        print("Buying core position...")
        strategy._buy_core_position(start_date)
        print(f'Core positions: {len(strategy.core_positions)}')
        print(f'Cash after core: ${strategy.cash:,.0f}')
    else:
        # Find nearest valid date
        valid_dates = tqqq_data.index[tqqq_data.index >= start_date]
        if len(valid_dates) > 0:
            nearest_date = valid_dates[0]
            print(f"Using nearest valid date: {nearest_date}")
            strategy._buy_core_position(nearest_date)
            print(f'Core positions: {len(strategy.core_positions)}')
            print(f'Cash after core: ${strategy.cash:,.0f}')
            start_date = nearest_date

    # Check QQQ data
    qqq_data = strategy._get_asset_data('QQQ')
    if start_date in qqq_data.index:
        price = float(qqq_data.loc[start_date, 'Close'])
        vol = float(qqq_data.loc[start_date, '20d_Vol'])
        print(f'QQQ price: {price:.2f}, vol: {vol:.3f}')

        # Test strike calculation
        short_strike = strategy._strike_for_put_delta(price, vol, 35, -0.15)
        long_strike = strategy._strike_for_put_delta(price, vol, 35, -0.05)
        print(f'Short strike: {short_strike:.2f}, Long strike: {long_strike:.2f}')

    # Now try income trade
    print("Trying income trade...")

    # Check capital allocation
    print(f'Income capital allocated: ${strategy.income_capital:,.0f}')
    print(f'Available cash: ${strategy.cash:,.0f}')

    # Manually check the trade logic
    qqq_data = strategy._get_asset_data('QQQ')
    current_price = float(qqq_data.loc[start_date, 'Close'])
    vol = float(qqq_data.loc[start_date, '20d_Vol'])

    short_strike = strategy._strike_for_put_delta(current_price, vol, 35, -0.15)
    long_strike = strategy._strike_for_put_delta(current_price, vol, 35, -0.05)

    short_premium = strategy._estimate_put_premium(current_price, short_strike, vol, 35)
    long_premium = strategy._estimate_put_premium(current_price, long_strike, vol, 35)
    net_premium = short_premium - long_premium

    max_loss = (short_strike - long_strike) - net_premium
    risk_amount = max_loss * 100

    prob_win = strategy._calculate_spread_win_prob(current_price, short_strike, long_strike, vol, 35)

    print(f'Net premium: {net_premium:.4f}')
    print(f'Max loss per contract: ${risk_amount:.2f}')
    print(f'Win probability: {prob_win:.3f}')
    print(f'Risk threshold: ${strategy.income_capital * strategy.max_loss_per_trade:.2f}')

    result = strategy._trade_income_layer(start_date)
    print(f'Income trade result: {result}')
    print(f'Income positions: {len(strategy.income_positions)}')

if __name__ == "__main__":
    debug_strategy()
