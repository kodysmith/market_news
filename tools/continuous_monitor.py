#!/usr/bin/env python3
"""
Continuous SPY Options Monitor
Monitors for high-yield opportunities throughout the trading day
"""

import requests
import json
import time
from datetime import datetime
import os
from typing import List, Dict, Any

class SPYOptionsMonitor:
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.base_url = "https://www.alphavantage.co/query"
        self.last_alert_time = {}
        
    def fetch_current_options(self) -> List[Dict[str, Any]]:
        """Fetch current SPY options data"""
        try:
            response = requests.get(
                self.base_url,
                params={
                    'function': 'HISTORICAL_OPTIONS',
                    'symbol': 'SPY',
                    'apikey': self.api_key
                }
            )
            
            if response.status_code == 200:
                data = response.json()
                if 'data' in data:
                    return data['data']
            return []
        except Exception as e:
            print(f"Error fetching options data: {e}")
            return []
    
    def find_bull_put_spreads(self, options_data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Find potential bull put spreads"""
        # Filter for put options
        puts = [opt for opt in options_data if opt.get('type') == 'put']
        
        if len(puts) < 2:
            return []
        
        # Sort by strike price
        puts.sort(key=lambda x: float(x.get('strike', 0)))
        
        spreads = []
        for i in range(len(puts) - 1):
            short_put = puts[i + 1]  # Higher strike (short)
            long_put = puts[i]       # Lower strike (long)
            
            try:
                short_strike = float(short_put.get('strike', 0))
                long_strike = float(long_put.get('strike', 0))
                width = short_strike - long_strike
                
                if width <= 0 or width > 10:
                    continue
                
                short_bid = float(short_put.get('bid', 0))
                long_ask = float(long_put.get('ask', 0))
                credit = short_bid - long_ask
                
                if credit <= 0:
                    continue
                
                yield_pct = credit / width
                short_iv = float(short_put.get('implied_volatility', 0))
                short_delta = abs(float(short_put.get('delta', 0)))
                
                # Calculate expected value
                win_rate = 1 - short_delta
                max_profit = credit * 100
                max_loss = (width - credit) * 100
                expected_value = (win_rate * max_profit) - ((1 - win_rate) * max_loss)
                
                spread = {
                    'short_strike': short_strike,
                    'long_strike': long_strike,
                    'width': width,
                    'credit': credit,
                    'yield': yield_pct,
                    'short_iv': short_iv,
                    'short_delta': short_delta,
                    'expected_value': expected_value,
                    'volume': int(short_put.get('volume', 0)),
                    'open_interest': int(short_put.get('open_interest', 0)),
                    'expiration': short_put.get('expiration', ''),
                    'timestamp': datetime.now()
                }
                spreads.append(spread)
                
            except (ValueError, TypeError) as e:
                continue
        
        return spreads
    
    def filter_opportunities(self, spreads: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Filter spreads based on criteria"""
        opportunities = []
        
        for spread in spreads:
            # High-yield criteria
            if (spread['yield'] > 5.0 and  # >500% yield
                spread['expected_value'] > 100 and  # >$100 EV
                spread['volume'] > 0 and  # Has volume
                spread['open_interest'] > 100 and  # Good liquidity
                spread['short_iv'] > 0.15):  # Reasonable IV
                
                opportunities.append(spread)
        
        return opportunities
    
    def format_alert(self, spread: Dict[str, Any]) -> str:
        """Format alert message for a spread"""
        return (
            f"🚨 HIGH-YIELD OPPORTUNITY DETECTED!\n"
            f"📊 Spread: ${spread['short_strike']:.0f}/${spread['long_strike']:.0f} Put\n"
            f"💰 Credit: ${spread['credit']:.2f}\n"
            f"📈 Yield: {spread['yield']:.0%}\n"
            f"💵 Expected Value: ${spread['expected_value']:.0f}\n"
            f"📊 IV: {spread['short_iv']:.1%}\n"
            f"📅 Expiration: {spread['expiration']}\n"
            f"📊 Volume: {spread['volume']:,}\n"
            f"⏰ Time: {spread['timestamp'].strftime('%H:%M:%S')}"
        )
    
    def should_alert(self, spread: Dict[str, Any]) -> bool:
        """Check if we should send an alert for this spread"""
        spread_id = f"{spread['short_strike']:.0f}/{spread['long_strike']:.0f}"
        current_time = spread['timestamp']
        
        # Don't alert more than once per hour for the same spread
        if spread_id in self.last_alert_time:
            time_diff = current_time - self.last_alert_time[spread_id]
            if time_diff.total_seconds() < 3600:  # 1 hour
                return False
        
        self.last_alert_time[spread_id] = current_time
        return True
    
    def monitor_continuously(self, check_interval: int = 300):
        """Continuously monitor for opportunities"""
        print("🔍 Starting SPY Options Monitor...")
        print(f"⏰ Checking every {check_interval} seconds")
        print("🎯 Looking for high-yield opportunities...")
        print("-" * 60)
        
        while True:
            try:
                current_time = datetime.now()
                print(f"\n[{current_time.strftime('%H:%M:%S')}] Checking for opportunities...")
                
                # Fetch current options data
                options_data = self.fetch_current_options()
                
                if not options_data:
                    print("❌ No options data available")
                    time.sleep(check_interval)
                    continue
                
                # Find spreads
                spreads = self.find_bull_put_spreads(options_data)
                
                if not spreads:
                    print("❌ No valid spreads found")
                    time.sleep(check_interval)
                    continue
                
                # Filter opportunities
                opportunities = self.filter_opportunities(spreads)
                
                if opportunities:
                    print(f"✅ Found {len(opportunities)} high-yield opportunities!")
                    
                    for spread in opportunities:
                        if self.should_alert(spread):
                            print("\n" + "="*60)
                            print(self.format_alert(spread))
                            print("="*60)
                else:
                    print(f"ℹ️  Found {len(spreads)} spreads, but none meet high-yield criteria")
                
                # Show summary
                if spreads:
                    best_spread = max(spreads, key=lambda x: x['yield'])
                    print(f"📊 Best spread: ${best_spread['short_strike']:.0f}/${best_spread['long_strike']:.0f} "
                          f"({best_spread['yield']:.0%} yield, ${best_spread['expected_value']:.0f} EV)")
                
                time.sleep(check_interval)
                
            except KeyboardInterrupt:
                print("\n🛑 Monitor stopped by user")
                break
            except Exception as e:
                print(f"❌ Error in monitoring loop: {e}")
                time.sleep(check_interval)

def main():
    # Get API key
    api_key = os.getenv('ALPHAVANTAGE_API_KEY')
    if not api_key:
        api_key = input("Enter your Alpha Vantage API key: ")
    
    # Create monitor
    monitor = SPYOptionsMonitor(api_key)
    
    # Start monitoring
    try:
        monitor.monitor_continuously(check_interval=300)  # Check every 5 minutes
    except KeyboardInterrupt:
        print("\n👋 Goodbye!")

if __name__ == "__main__":
    main()

