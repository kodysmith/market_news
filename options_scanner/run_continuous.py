#!/usr/bin/env python3
"""
Run the options scanner continuously for testing
"""

import os
import sys
import time
from datetime import datetime

def setup_test_environment():
    """Set up test environment"""
    os.environ['ALPHAVANTAGE_API_KEY'] = 'test_key'
    os.environ['FIREBASE_SERVICE_ACCOUNT_JSON_BASE64'] = 'dGVzdA=='
    os.environ['PUBLIC_BUCKET_URL'] = 'https://api-hvi4gdtdka-uc.a.run.app'
    os.environ['REPORT_OBJECT_PATH'] = '/report.json'
    os.environ['TICKERS'] = 'SPY'
    os.environ['SCAN_INTERVAL_MINUTES'] = '1'  # Run every minute for testing
    os.environ['MARKET_TZ'] = 'America/New_York'

def run_continuous_demo():
    """Run continuous scanning demo"""
    print("🔄 Starting Continuous Options Scanner Demo")
    print("Press Ctrl+C to stop")
    print("=" * 50)
    
    setup_test_environment()
    
    try:
        from main_worker import OptionsScanner
        scanner = OptionsScanner()
        
        run_count = 0
        while True:
            run_count += 1
            print(f"\n🔄 Run #{run_count} at {datetime.now().strftime('%H:%M:%S')}")
            
            result = scanner.run_once()
            
            if result['status'] == 'success':
                print(f"✅ Found {result['total_ideas']} opportunities, sent {result['alerts_sent']} alerts")
            elif result['status'] == 'skipped':
                print(f"⏰ {result['reason']}")
            else:
                print(f"❌ Error: {result.get('reason', 'Unknown error')}")
            
            print("⏳ Waiting 60 seconds until next scan...")
            time.sleep(60)  # Wait 1 minute between scans
            
    except KeyboardInterrupt:
        print("\n🛑 Continuous scanning stopped by user")
    except Exception as e:
        print(f"\n❌ Error in continuous scanning: {e}")

if __name__ == "__main__":
    run_continuous_demo()

