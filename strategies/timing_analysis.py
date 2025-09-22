#!/usr/bin/env python3
"""
Analyze timing patterns of SPY options opportunities
"""

import sqlite3
import pandas as pd
from datetime import datetime, timedelta
import numpy as np

def analyze_timing_patterns():
    """Analyze when high-yield options opportunities appear"""
    conn = sqlite3.connect('spy_options_data/spy_options.db')
    
    # Get all options data with timestamps
    query = '''
        SELECT strike, bid, ask, mark, volume, open_interest, 
               implied_volatility, delta, expiration, date, created_at
        FROM options_data 
        WHERE type = 'put'
        ORDER BY created_at, strike
    '''
    
    df = pd.read_sql_query(query, conn)
    conn.close()
    
    print("="*60)
    print("SPY OPTIONS TIMING ANALYSIS")
    print("="*60)
    print(f"Total data points: {len(df):,}")
    
    # Convert timestamps
    df['created_at'] = pd.to_datetime(df['created_at'])
    df['date'] = pd.to_datetime(df['date'])
    
    # Add time features
    df['day_of_week'] = df['created_at'].dt.day_name()
    df['hour'] = df['created_at'].dt.hour
    df['minute'] = df['created_at'].dt.minute
    df['time_of_day'] = df['created_at'].dt.strftime('%H:%M')
    
    # Group by strike and get the best option for each strike at each time
    df_grouped = df.groupby(['created_at', 'strike']).agg({
        'bid': 'max',
        'ask': 'min', 
        'mark': 'mean',
        'volume': 'sum',
        'open_interest': 'sum',
        'implied_volatility': 'mean',
        'delta': 'mean',
        'expiration': 'first',
        'date': 'first',
        'day_of_week': 'first',
        'hour': 'first',
        'time_of_day': 'first'
    }).reset_index()
    
    print(f"Unique time-strike combinations: {len(df_grouped):,}")
    
    # Calculate spreads for each time point
    all_spreads = []
    
    for timestamp in df_grouped['created_at'].unique():
        time_data = df_grouped[df_grouped['created_at'] == timestamp].sort_values('strike')
        
        if len(time_data) < 2:
            continue
            
        # Calculate spreads for this timestamp
        spreads = []
        for i in range(len(time_data) - 1):
            short_put = time_data.iloc[i + 1]  # Higher strike (short)
            long_put = time_data.iloc[i]       # Lower strike (long)
            
            short_strike = short_put['strike']
            long_strike = long_put['strike']
            width = short_strike - long_strike
            
            if width <= 0 or width > 10:
                continue
                
            short_bid = short_put['bid']
            long_ask = long_put['ask']
            credit = short_bid - long_ask
            
            if credit <= 0:
                continue
                
            yield_pct = credit / width
            short_iv = short_put['implied_volatility']
            short_delta = abs(short_put['delta'])
            
            # Calculate expected value
            win_rate = 1 - short_delta
            max_profit = credit * 100
            max_loss = (width - credit) * 100
            expected_value = (win_rate * max_profit) - ((1 - win_rate) * max_loss)
            
            spread = {
                'timestamp': timestamp,
                'day_of_week': short_put['day_of_week'],
                'hour': short_put['hour'],
                'time_of_day': short_put['time_of_day'],
                'short_strike': short_strike,
                'long_strike': long_strike,
                'width': width,
                'credit': credit,
                'yield': yield_pct,
                'short_iv': short_iv,
                'short_delta': short_delta,
                'expected_value': expected_value,
                'volume': short_put['volume'],
                'open_interest': short_put['open_interest']
            }
            spreads.append(spread)
        
        all_spreads.extend(spreads)
    
    if not all_spreads:
        print("No spreads found for timing analysis")
        return
    
    spreads_df = pd.DataFrame(all_spreads)
    
    print(f"\nFound {len(spreads_df)} spread opportunities across time")
    
    # 1. Day of Week Analysis
    print(f"\n" + "="*60)
    print("DAY OF WEEK ANALYSIS")
    print("="*60)
    
    day_stats = spreads_df.groupby('day_of_week').agg({
        'yield': ['count', 'mean', 'median', 'std'],
        'expected_value': ['mean', 'median'],
        'short_iv': 'mean'
    }).round(2)
    
    print("Opportunities by day of week:")
    print("-" * 50)
    for day in ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']:
        if day in day_stats.index:
            count = day_stats.loc[day, ('yield', 'count')]
            mean_yield = day_stats.loc[day, ('yield', 'mean')]
            mean_ev = day_stats.loc[day, ('expected_value', 'mean')]
            mean_iv = day_stats.loc[day, ('short_iv', 'mean')]
            print(f"{day:<10}: {count:>4} opportunities, {mean_yield:>6.1%} avg yield, ${mean_ev:>7.0f} avg EV, {mean_iv:>5.1%} avg IV")
    
    # 2. Hour of Day Analysis
    print(f"\n" + "="*60)
    print("HOUR OF DAY ANALYSIS")
    print("="*60)
    
    hour_stats = spreads_df.groupby('hour').agg({
        'yield': ['count', 'mean', 'median'],
        'expected_value': ['mean', 'median'],
        'short_iv': 'mean'
    }).round(2)
    
    print("Opportunities by hour of day:")
    print("-" * 50)
    for hour in range(24):
        if hour in hour_stats.index:
            count = hour_stats.loc[hour, ('yield', 'count')]
            mean_yield = hour_stats.loc[hour, ('yield', 'mean')]
            mean_ev = hour_stats.loc[hour, ('expected_value', 'mean')]
            mean_iv = hour_stats.loc[hour, ('short_iv', 'mean')]
            print(f"{hour:02d}:00     : {count:>4} opportunities, {mean_yield:>6.1%} avg yield, ${mean_ev:>7.0f} avg EV, {mean_iv:>5.1%} avg IV")
    
    # 3. High-Yield Opportunities Timing
    print(f"\n" + "="*60)
    print("HIGH-YIELD OPPORTUNITIES TIMING (>1000% yield)")
    print("="*60)
    
    high_yield = spreads_df[spreads_df['yield'] > 10.0]  # >1000%
    
    if len(high_yield) > 0:
        print(f"Found {len(high_yield)} high-yield opportunities")
        
        # Day of week for high yield
        high_yield_days = high_yield['day_of_week'].value_counts()
        print(f"\nHigh-yield opportunities by day:")
        for day, count in high_yield_days.items():
            print(f"  {day}: {count} opportunities")
        
        # Hour of day for high yield
        high_yield_hours = high_yield['hour'].value_counts().sort_index()
        print(f"\nHigh-yield opportunities by hour:")
        for hour, count in high_yield_hours.items():
            print(f"  {hour:02d}:00: {count} opportunities")
        
        # Best times for high yield
        print(f"\nBest times for high-yield opportunities:")
        best_times = high_yield.groupby(['day_of_week', 'hour']).size().sort_values(ascending=False)
        for (day, hour), count in best_times.head(10).items():
            print(f"  {day} {hour:02d}:00: {count} opportunities")
    
    # 4. Positive EV Opportunities Timing
    print(f"\n" + "="*60)
    print("POSITIVE EV OPPORTUNITIES TIMING")
    print("="*60)
    
    positive_ev = spreads_df[spreads_df['expected_value'] > 0]
    
    if len(positive_ev) > 0:
        print(f"Found {len(positive_ev)} positive EV opportunities")
        
        # Day of week for positive EV
        ev_days = positive_ev['day_of_week'].value_counts()
        print(f"\nPositive EV opportunities by day:")
        for day, count in ev_days.items():
            print(f"  {day}: {count} opportunities")
        
        # Hour of day for positive EV
        ev_hours = positive_ev['hour'].value_counts().sort_index()
        print(f"\nPositive EV opportunities by hour:")
        for hour, count in ev_hours.items():
            print(f"  {hour:02d}:00: {count} opportunities")
    
    # 5. Market Hours vs After Hours
    print(f"\n" + "="*60)
    print("MARKET HOURS vs AFTER HOURS ANALYSIS")
    print("="*60)
    
    # Define market hours (9:30 AM - 4:00 PM ET)
    spreads_df['is_market_hours'] = (
        (spreads_df['hour'] >= 9) & 
        (spreads_df['hour'] < 16) &
        (spreads_df['day_of_week'].isin(['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday']))
    )
    
    market_hours = spreads_df[spreads_df['is_market_hours']]
    after_hours = spreads_df[~spreads_df['is_market_hours']]
    
    print(f"Market Hours (9:30 AM - 4:00 PM ET, Mon-Fri):")
    print(f"  Total opportunities: {len(market_hours):,}")
    print(f"  Average yield: {market_hours['yield'].mean():.1%}")
    print(f"  Average EV: ${market_hours['expected_value'].mean():.0f}")
    print(f"  High-yield opportunities: {len(market_hours[market_hours['yield'] > 10.0]):,}")
    
    print(f"\nAfter Hours:")
    print(f"  Total opportunities: {len(after_hours):,}")
    print(f"  Average yield: {after_hours['yield'].mean():.1%}")
    print(f"  Average EV: ${after_hours['expected_value'].mean():.0f}")
    print(f"  High-yield opportunities: {len(after_hours[after_hours['yield'] > 10.0]):,}")
    
    # 6. Frequency Analysis
    print(f"\n" + "="*60)
    print("FREQUENCY ANALYSIS")
    print("="*60)
    
    # Count unique timestamps
    unique_times = spreads_df['timestamp'].nunique()
    total_hours = (spreads_df['timestamp'].max() - spreads_df['timestamp'].min()).total_seconds() / 3600
    
    print(f"Data spans {total_hours:.1f} hours")
    print(f"Unique time points: {unique_times}")
    print(f"Average opportunities per hour: {len(spreads_df) / total_hours:.1f}")
    print(f"Average opportunities per time point: {len(spreads_df) / unique_times:.1f}")
    
    # 7. Best Trading Times Summary
    print(f"\n" + "="*60)
    print("BEST TRADING TIMES SUMMARY")
    print("="*60)
    
    # Find the best combinations
    best_combinations = spreads_df.groupby(['day_of_week', 'hour']).agg({
        'yield': ['count', 'mean'],
        'expected_value': 'mean'
    }).round(2)
    
    # Filter for combinations with at least 5 opportunities
    best_combinations = best_combinations[best_combinations[('yield', 'count')] >= 5]
    
    if len(best_combinations) > 0:
        # Sort by average yield
        best_combinations = best_combinations.sort_values(('yield', 'mean'), ascending=False)
        
        print("Top 10 best times for options opportunities:")
        print("-" * 60)
        print(f"{'Day':<10} {'Hour':<6} {'Count':<6} {'Avg Yield':<10} {'Avg EV':<10}")
        print("-" * 60)
        
        for (day, hour), row in best_combinations.head(10).iterrows():
            count = row[('yield', 'count')]
            avg_yield = row[('yield', 'mean')]
            avg_ev = row[('expected_value', 'mean')]
            print(f"{day:<10} {hour:02d}:00 {count:<6} {avg_yield:<9.1%} ${avg_ev:<9.0f}")
    
    print(f"\n" + "="*60)
    print("CONCLUSION")
    print("="*60)
    print("✅ Successfully analyzed timing patterns of options opportunities")
    print("📊 Key insights about when opportunities appear")
    print("⏰ Best times for high-yield and positive EV spreads identified")
    print("📈 Market hours vs after-hours performance compared")

if __name__ == "__main__":
    analyze_timing_patterns()
