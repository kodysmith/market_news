#!/usr/bin/env python3
"""
Start the options scanner with live updates
"""

import os
import sys
import time
import threading
from datetime import datetime

def setup_environment():
    """Set up environment for scanning"""
    os.environ['ALPHAVANTAGE_API_KEY'] = 'test_key'
    os.environ['FIREBASE_SERVICE_ACCOUNT_JSON_BASE64'] = 'dGVzdA=='
    os.environ['PUBLIC_BUCKET_URL'] = 'https://api-hvi4gdtdka-uc.a.run.app'
    os.environ['REPORT_OBJECT_PATH'] = '/report.json'
    os.environ['TICKERS'] = 'SPY'
    os.environ['SCAN_INTERVAL_MINUTES'] = '1'
    os.environ['MARKET_TZ'] = 'America/New_York'

def run_scanner_cycle():
    """Run one scanner cycle"""
    try:
        from test_local import test_with_mock_data
        print(f"\n🔄 Scanner cycle at {datetime.now().strftime('%H:%M:%S')}")
        test_with_mock_data()
        print("✅ Report updated successfully")
    except Exception as e:
        print(f"❌ Scanner error: {e}")

def main():
    """Main scanner loop"""
    print("🚀 Starting Options Scanner (Live Mode)")
    print("=" * 50)
    print("📊 Scanner will update report.json every 60 seconds")
    print("🌐 Report available at: http://localhost:8081/report.json")
    print("⏹️  Press Ctrl+C to stop")
    print("=" * 50)
    
    setup_environment()
    
    # Run initial scan
    run_scanner_cycle()
    
    try:
        while True:
            time.sleep(60)  # Wait 60 seconds
            run_scanner_cycle()
    except KeyboardInterrupt:
        print("\n🛑 Scanner stopped by user")
        print("👋 Goodbye!")

if __name__ == "__main__":
    main()

