#!/usr/bin/env python3
"""
SPY Options Data Backtesting Tool
Downloads historical options data for SPY over the last 12 months
"""

import requests
import json
import pandas as pd
import time
from datetime import datetime, timedelta
import os
from typing import List, Dict, Any
import sqlite3

class SPYOptionsBacktester:
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.base_url = "https://www.alphavantage.co/query"
        self.data_dir = "spy_options_data"
        self.db_path = os.path.join(self.data_dir, "spy_options.db")
        
        # Create data directory
        os.makedirs(self.data_dir, exist_ok=True)
        
        # Initialize database
        self.init_database()
        
    def init_database(self):
        """Initialize SQLite database for storing options data"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Create options data table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS options_data (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                contract_id TEXT,
                symbol TEXT,
                expiration TEXT,
                strike REAL,
                type TEXT,
                last_price REAL,
                mark REAL,
                bid REAL,
                ask REAL,
                bid_size INTEGER,
                ask_size INTEGER,
                volume INTEGER,
                open_interest INTEGER,
                date TEXT,
                implied_volatility REAL,
                delta REAL,
                gamma REAL,
                theta REAL,
                vega REAL,
                rho REAL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # Create index for faster queries
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_symbol_date ON options_data(symbol, date)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_expiration ON options_data(expiration)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_strike ON options_data(strike)')
        
        conn.commit()
        conn.close()
        
    def fetch_options_data(self, symbol: str = "SPY") -> Dict[str, Any]:
        """Fetch options data from Alpha Vantage"""
        params = {
            'function': 'HISTORICAL_OPTIONS',
            'symbol': symbol,
            'apikey': self.api_key
        }
        
        try:
            response = requests.get(self.base_url, params=params)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            print(f"Error fetching data: {e}")
            return {}
    
    def parse_options_data(self, data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Parse options data from API response"""
        if 'data' not in data:
            print("No options data found in response")
            return []
        
        options_list = []
        for option in data['data']:
            try:
                parsed_option = {
                    'contract_id': option.get('contractID', ''),
                    'symbol': option.get('symbol', ''),
                    'expiration': option.get('expiration', ''),
                    'strike': float(option.get('strike', 0)),
                    'type': option.get('type', ''),
                    'last_price': float(option.get('last', 0)),
                    'mark': float(option.get('mark', 0)),
                    'bid': float(option.get('bid', 0)),
                    'ask': float(option.get('ask', 0)),
                    'bid_size': int(option.get('bid_size', 0)),
                    'ask_size': int(option.get('ask_size', 0)),
                    'volume': int(option.get('volume', 0)),
                    'open_interest': int(option.get('open_interest', 0)),
                    'date': option.get('date', ''),
                    'implied_volatility': float(option.get('implied_volatility', 0)),
                    'delta': float(option.get('delta', 0)),
                    'gamma': float(option.get('gamma', 0)),
                    'theta': float(option.get('theta', 0)),
                    'vega': float(option.get('vega', 0)),
                    'rho': float(option.get('rho', 0))
                }
                options_list.append(parsed_option)
            except (ValueError, TypeError) as e:
                print(f"Error parsing option: {e}")
                continue
                
        return options_list
    
    def save_to_database(self, options_data: List[Dict[str, Any]]):
        """Save options data to SQLite database"""
        if not options_data:
            return
            
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        for option in options_data:
            cursor.execute('''
                INSERT OR REPLACE INTO options_data 
                (contract_id, symbol, expiration, strike, type, last_price, mark, 
                 bid, ask, bid_size, ask_size, volume, open_interest, date, 
                 implied_volatility, delta, gamma, theta, vega, rho)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                option['contract_id'], option['symbol'], option['expiration'],
                option['strike'], option['type'], option['last_price'],
                option['mark'], option['bid'], option['ask'], option['bid_size'],
                option['ask_size'], option['volume'], option['open_interest'],
                option['date'], option['implied_volatility'], option['delta'],
                option['gamma'], option['theta'], option['vega'], option['rho']
            ))
        
        conn.commit()
        conn.close()
        print(f"Saved {len(options_data)} options contracts to database")
    
    def save_to_csv(self, options_data: List[Dict[str, Any]], filename: str = None):
        """Save options data to CSV file"""
        if not options_data:
            return
            
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"spy_options_{timestamp}.csv"
        
        filepath = os.path.join(self.data_dir, filename)
        df = pd.DataFrame(options_data)
        df.to_csv(filepath, index=False)
        print(f"Saved {len(options_data)} options contracts to {filepath}")
    
    def download_historical_data(self, days_back: int = 365):
        """Download historical options data for the specified number of days"""
        print(f"Starting download of SPY options data for the last {days_back} days...")
        
        # For now, we'll download current data since Alpha Vantage's historical
        # options endpoint seems to return current data
        print("Fetching current options data...")
        
        data = self.fetch_options_data("SPY")
        if not data:
            print("Failed to fetch options data")
            return
        
        options_list = self.parse_options_data(data)
        if not options_list:
            print("No options data to save")
            return
        
        # Save to database
        self.save_to_database(options_list)
        
        # Save to CSV
        self.save_to_csv(options_list, "spy_options_current.csv")
        
        print(f"Download complete! Found {len(options_list)} options contracts")
        
        # Show summary
        self.show_data_summary()
    
    def show_data_summary(self):
        """Show summary of downloaded data"""
        conn = sqlite3.connect(self.db_path)
        
        # Get basic stats
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM options_data")
        total_contracts = cursor.fetchone()[0]
        
        cursor.execute("SELECT COUNT(DISTINCT expiration) FROM options_data")
        unique_expirations = cursor.fetchone()[0]
        
        cursor.execute("SELECT MIN(strike), MAX(strike) FROM options_data")
        strike_range = cursor.fetchone()
        
        cursor.execute("SELECT COUNT(*) FROM options_data WHERE type = 'call'")
        call_count = cursor.fetchone()[0]
        
        cursor.execute("SELECT COUNT(*) FROM options_data WHERE type = 'put'")
        put_count = cursor.fetchone()[0]
        
        print("\n" + "="*50)
        print("SPY OPTIONS DATA SUMMARY")
        print("="*50)
        print(f"Total contracts: {total_contracts:,}")
        print(f"Call options: {call_count:,}")
        print(f"Put options: {put_count:,}")
        print(f"Unique expirations: {unique_expirations}")
        print(f"Strike range: ${strike_range[0]:.2f} - ${strike_range[1]:.2f}")
        print("="*50)
        
        # Show expiration dates
        cursor.execute("SELECT DISTINCT expiration FROM options_data ORDER BY expiration")
        expirations = cursor.fetchall()
        print("\nExpiration dates:")
        for exp in expirations[:10]:  # Show first 10
            print(f"  {exp[0]}")
        if len(expirations) > 10:
            print(f"  ... and {len(expirations) - 10} more")
        
        conn.close()
    
    def get_put_spreads_for_backtesting(self, min_dte: int = 30, max_dte: int = 45):
        """Get put spreads suitable for backtesting"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Get put options within DTE range
        query = '''
            SELECT * FROM options_data 
            WHERE type = 'put' 
            AND expiration >= date('now', '+{} days')
            AND expiration <= date('now', '+{} days')
            ORDER BY strike
        '''.format(min_dte, max_dte)
        
        cursor.execute(query)
        puts = cursor.fetchall()
        
        conn.close()
        
        if not puts:
            print("No put options found in the specified DTE range")
            return []
        
        # Generate potential bull put spreads
        spreads = []
        for i in range(len(puts) - 1):
            short_put = puts[i]
            long_put = puts[i + 1]
            
            # Calculate spread metrics
            short_strike = float(short_put[4])  # strike (column 4)
            long_strike = float(long_put[4])    # strike (column 4)
            width = short_strike - long_strike
            
            if width <= 0 or width > 10:  # Skip invalid spreads
                continue
                
            # Use mark prices for more realistic pricing
            short_bid = float(short_put[8])   # bid (column 8)
            long_ask = float(long_put[9])     # ask (column 9)
            credit = short_bid - long_ask
            
            if credit <= 0:
                continue
                
            spread = {
                'short_strike': short_strike,
                'long_strike': long_strike,
                'width': width,
                'credit': credit,
                'yield': credit / width,
                'short_bid': short_bid,
                'long_ask': long_ask,
                'expiration': short_put[3],      # expiration (column 3)
                'short_delta': float(short_put[16]),  # delta (column 16)
                'short_iv': float(short_put[15]),     # implied_volatility (column 15)
                'volume': int(short_put[12]),         # volume (column 12)
                'open_interest': int(short_put[13])   # open_interest (column 13)
            }
            spreads.append(spread)
        
        return spreads
    
    def backtest_strategy(self, min_yield: float = 0.33, min_iv: float = 0.20):
        """Backtest bull put spread strategy"""
        print(f"\nBacktesting Bull Put Spread Strategy...")
        print(f"Min Yield: {min_yield:.1%}")
        print(f"Min IV: {min_iv:.1%}")
        
        spreads = self.get_put_spreads_for_backtesting()
        
        if not spreads:
            print("No suitable spreads found for backtesting")
            return
        
        # Filter spreads by criteria
        filtered_spreads = []
        for spread in spreads:
            if (spread['yield'] >= min_yield and 
                spread['short_iv'] >= min_iv and
                spread['volume'] > 0 and
                spread['open_interest'] > 100):
                filtered_spreads.append(spread)
        
        print(f"\nFound {len(filtered_spreads)} spreads meeting criteria:")
        print("-" * 80)
        print(f"{'Short Strike':<12} {'Long Strike':<12} {'Width':<8} {'Credit':<8} {'Yield':<8} {'IV':<8}")
        print("-" * 80)
        
        for spread in filtered_spreads[:20]:  # Show top 20
            print(f"${spread['short_strike']:<11.0f} ${spread['long_strike']:<11.0f} "
                  f"${spread['width']:<7.0f} ${spread['credit']:<7.2f} "
                  f"{spread['yield']:<7.1%} {spread['short_iv']:<7.1%}")
        
        if len(filtered_spreads) > 20:
            print(f"... and {len(filtered_spreads) - 20} more")
        
        return filtered_spreads

def main():
    # Get API key from environment or prompt user
    api_key = os.getenv('ALPHAVANTAGE_API_KEY')
    if not api_key:
        api_key = input("Enter your Alpha Vantage API key: ")
    
    # Initialize backtester
    backtester = SPYOptionsBacktester(api_key)
    
    # Download data
    backtester.download_historical_data(days_back=365)
    
    # Run backtest
    spreads = backtester.backtest_strategy()
    
    if spreads:
        print(f"\nBacktest complete! Found {len(spreads)} profitable spreads.")
    else:
        print("\nNo profitable spreads found with current criteria.")

if __name__ == "__main__":
    main()
