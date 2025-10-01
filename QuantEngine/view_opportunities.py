#!/usr/bin/env python3
"""
View Current Overbought/Oversold Opportunities

Quick dashboard to view the current list of opportunities.
"""

import json
import sys
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any

# Add QuantEngine to path
quant_engine_dir = Path(__file__).parent
sys.path.insert(0, str(quant_engine_dir))

def load_scan_data() -> Dict[str, Any]:
    """Load the latest scan data"""
    data_file = Path("overbought_oversold_data.json")
    
    if not data_file.exists():
        print("❌ No scan data found. Run the scanner first:")
        print("   python3 overbought_oversold_scanner.py --scan")
        return {}
    
    try:
        with open(data_file, 'r') as f:
            return json.load(f)
    except Exception as e:
        print(f"❌ Error loading data: {e}")
        return {}

def format_timestamp(timestamp_str: str) -> str:
    """Format timestamp for display"""
    try:
        dt = datetime.fromisoformat(timestamp_str)
        return dt.strftime('%Y-%m-%d %H:%M:%S')
    except:
        return timestamp_str

def display_opportunities():
    """Display current opportunities"""
    data = load_scan_data()
    
    if not data:
        return
    
    stocks = data.get('stocks', {})
    last_scan = data.get('last_scan', 'Unknown')
    
    print("="*80)
    print("📊 OVERBOUGHT/OVERSOLD OPPORTUNITIES DASHBOARD")
    print("="*80)
    print(f"📅 Last Scan: {format_timestamp(last_scan)}")
    print(f"📈 Total Stocks Monitored: {len(stocks)}")
    print()
    
    # Categorize stocks
    categories = {
        'extreme_overbought': [],
        'overbought': [],
        'oversold': [],
        'extreme_oversold': [],
        'neutral': []
    }
    
    for ticker, stock_data in stocks.items():
        condition = stock_data.get('condition', 'neutral')
        if condition in categories:
            categories[condition].append((ticker, stock_data))
    
    # Display by category
    for category, stocks_list in categories.items():
        if not stocks_list:
            continue
            
        category_name = category.replace('_', ' ').title()
        emoji = {
            'extreme_overbought': '🔥',
            'overbought': '📈',
            'oversold': '📉',
            'extreme_oversold': '❄️',
            'neutral': '⚪'
        }.get(category, '📊')
        
        print(f"{emoji} {category_name.upper()} ({len(stocks_list)} stocks)")
        print("-" * 60)
        
        # Sort by RSI (descending for overbought, ascending for oversold)
        if 'overbought' in category:
            stocks_list.sort(key=lambda x: x[1]['rsi'], reverse=True)
        else:
            stocks_list.sort(key=lambda x: x[1]['rsi'])
        
        for ticker, stock_data in stocks_list[:10]:  # Show top 10
            rsi = stock_data.get('rsi', 0)
            price = stock_data.get('current_price', 0)
            change = stock_data.get('price_change', 0)
            signal = stock_data.get('signal', 'HOLD')
            confidence = stock_data.get('confidence', 0)
            
            print(f"  {ticker:6} | RSI: {rsi:5.1f} | ${price:8.2f} ({change:+5.1f}%) | {signal:4} | {confidence:5.1%}")
        
        if len(stocks_list) > 10:
            print(f"  ... and {len(stocks_list) - 10} more")
        print()
    
    # Show top opportunities by confidence
    all_opportunities = []
    for ticker, stock_data in stocks.items():
        if stock_data.get('condition') != 'neutral':
            all_opportunities.append((ticker, stock_data))
    
    if all_opportunities:
        print("🎯 TOP OPPORTUNITIES BY CONFIDENCE")
        print("-" * 60)
        
        # Sort by confidence
        all_opportunities.sort(key=lambda x: x[1]['confidence'], reverse=True)
        
        for i, (ticker, stock_data) in enumerate(all_opportunities[:15], 1):
            rsi = stock_data.get('rsi', 0)
            price = stock_data.get('current_price', 0)
            change = stock_data.get('price_change', 0)
            signal = stock_data.get('signal', 'HOLD')
            confidence = stock_data.get('confidence', 0)
            condition = stock_data.get('condition', 'neutral')
            target = stock_data.get('target_price', 0)
            stop_loss = stock_data.get('stop_loss', 0)
            
            print(f"  {i:2}. {ticker:6} | {signal:4} | RSI: {rsi:5.1f} | ${price:8.2f} ({change:+5.1f}%) | {confidence:5.1%} | {condition.replace('_', ' ').title()}")
            print(f"      Target: ${target:.2f} | Stop: ${stop_loss:.2f}")
        
        print()
    
    # Show scan history
    scan_history = data.get('scan_history', [])
    if scan_history:
        print("📈 RECENT SCAN HISTORY")
        print("-" * 60)
        
        for scan in scan_history[-5:]:  # Last 5 scans
            timestamp = format_timestamp(scan.get('timestamp', ''))
            overbought = scan.get('overbought_count', 0)
            oversold = scan.get('oversold_count', 0)
            extreme_overbought = scan.get('extreme_overbought_count', 0)
            extreme_oversold = scan.get('extreme_oversold_count', 0)
            
            print(f"  {timestamp} | Overbought: {overbought} | Oversold: {oversold} | Extreme OB: {extreme_overbought} | Extreme OS: {extreme_oversold}")
        
        print()
    
    print("="*80)
    print("💡 TIP: Run 'python3 overbought_oversold_scanner.py --scan' to update the data")
    print("🔄 Run 'python3 schedule_scanner.py' to start continuous monitoring")
    print("="*80)

def main():
    """Main entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(description='View Overbought/Oversold Opportunities')
    parser.add_argument('--category', choices=['overbought', 'oversold', 'extreme_overbought', 'extreme_oversold'], 
                       help='Filter by category')
    parser.add_argument('--min-confidence', type=float, default=0.0, 
                       help='Minimum confidence level (0.0-1.0)')
    
    args = parser.parse_args()
    
    display_opportunities()

if __name__ == "__main__":
    main()

