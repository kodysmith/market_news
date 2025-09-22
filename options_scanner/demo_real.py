#!/usr/bin/env python3
"""
Demo script to run the options scanner with real API data
"""

import os
import sys
from datetime import datetime

def setup_environment():
    """Set up environment variables for real API testing"""
    print("🔧 Setting up environment for real API testing...")
    
    # Check if API key is provided
    api_key = input("Enter your Alpha Vantage API key (or press Enter to use mock data): ").strip()
    
    if not api_key:
        print("⚠️  No API key provided, will use mock data")
        return False
    
    # Set environment variables
    os.environ['ALPHAVANTAGE_API_KEY'] = api_key
    os.environ['FIREBASE_SERVICE_ACCOUNT_JSON_BASE64'] = 'dGVzdA=='  # Mock Firebase for demo
    os.environ['PUBLIC_BUCKET_URL'] = 'https://api-hvi4gdtdka-uc.a.run.app'
    os.environ['REPORT_OBJECT_PATH'] = '/report.json'
    os.environ['TICKERS'] = 'SPY'
    os.environ['SCAN_INTERVAL_MINUTES'] = '5'
    os.environ['MARKET_TZ'] = 'America/New_York'
    
    print("✅ Environment configured")
    return True

def run_scanner_demo():
    """Run the scanner with real or mock data"""
    print("\n🚀 Starting Options Scanner Demo")
    print("=" * 50)
    
    # Set up environment
    use_real_api = setup_environment()
    
    if use_real_api:
        print("\n📡 Using real Alpha Vantage API...")
        try:
            from main_worker import OptionsScanner
            scanner = OptionsScanner()
            result = scanner.run_once()
            
            print(f"\n📊 Scan Result: {result}")
            
            if result['status'] == 'success':
                print(f"✅ Found {result['total_ideas']} opportunities")
                print(f"📤 Sent {result['alerts_sent']} alerts")
                print(f"💰 Average EV: ${result['avg_ev']:.2f}")
                print(f"🎯 Average POP: {result['avg_pop']:.1%}")
            elif result['status'] == 'skipped':
                print(f"⏰ {result['reason']}")
            else:
                print(f"❌ Error: {result.get('reason', 'Unknown error')}")
                
        except Exception as e:
            print(f"❌ Error running scanner: {e}")
            print("💡 This might be due to API rate limits or invalid credentials")
    else:
        print("\n🧪 Using mock data for demo...")
        from test_local import test_with_mock_data
        test_with_mock_data()
    
    print("\n📄 Generated report.json:")
    try:
        with open('report.json', 'r') as f:
            print(f.read())
    except FileNotFoundError:
        print("❌ No report.json found")

def main():
    """Main demo function"""
    print("🎯 Options Scanner Demo")
    print("This demo will test the options scanner with real or mock data")
    print()
    
    try:
        run_scanner_demo()
    except KeyboardInterrupt:
        print("\n🛑 Demo stopped by user")
    except Exception as e:
        print(f"\n❌ Demo failed: {e}")
        sys.exit(1)
    
    print("\n🎉 Demo completed!")
    print("\nNext steps:")
    print("1. Configure your Alpha Vantage API key in .env file")
    print("2. Set up Firebase credentials for push notifications")
    print("3. Deploy to Cloud Run or set up cron job")
    print("4. Monitor the generated report.json file")

if __name__ == "__main__":
    main()

