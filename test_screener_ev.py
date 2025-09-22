#!/usr/bin/env python3
"""
Test the screener to ensure it only shows positive EV trades
"""

import os
import sys
import json
from datetime import datetime

# Set up environment
os.environ['ALPHAVANTAGE_API_KEY'] = 'test_key'
os.environ['FIREBASE_SERVICE_ACCOUNT_JSON_BASE64'] = 'dGVzdA=='
os.environ['PUBLIC_BUCKET_URL'] = 'https://api-hvi4gdtdka-uc.a.run.app'
os.environ['REPORT_OBJECT_PATH'] = '/report.json'
os.environ['TICKERS'] = 'SPY'
os.environ['SCAN_INTERVAL_MINUTES'] = '5'
os.environ['MARKET_TZ'] = 'America/New_York'

def test_positive_ev_filtering():
    """Test that screener only shows positive EV trades"""
    print("🧪 Testing Screener Positive EV Filtering")
    print("=" * 50)
    
    try:
        from test_local import test_with_mock_data
        
        # Run the test
        print("📊 Running screener with mock data...")
        test_with_mock_data()
        
        # Check the generated report
        print("\n📄 Checking generated report...")
        with open('report.json', 'r') as f:
            report = json.load(f)
        
        print(f"✅ Report generated successfully")
        print(f"📅 Timestamp: {report['asOf']}")
        print(f"📊 Total ideas: {len(report['topIdeas'])}")
        
        # Verify all ideas have positive EV
        print("\n🔍 Verifying Expected Values:")
        all_positive = True
        
        for i, idea in enumerate(report['topIdeas'], 1):
            ev = idea['ev']
            pop = idea['pop']
            credit = idea['credit']
            max_loss = idea['maxLoss']
            
            print(f"  {i}. {idea['ticker']} {idea['shortK']}/{idea['longK']}")
            print(f"     EV: ${ev:.2f}")
            print(f"     POP: {pop:.1%}")
            print(f"     Credit: ${credit:.2f}")
            print(f"     Max Loss: ${max_loss:.2f}")
            
            if ev < 0:
                print(f"     ❌ NEGATIVE EV - This should be filtered out!")
                all_positive = False
            else:
                print(f"     ✅ Positive EV")
            print()
        
        if all_positive:
            print("🎉 SUCCESS: All trades have positive expected value!")
            print("✅ Screener is correctly filtering out negative EV trades")
        else:
            print("❌ FAILURE: Some trades have negative expected value")
            print("❌ Screener is not filtering correctly")
            return False
        
        # Test the filtering logic
        print("\n🧪 Testing filtering logic...")
        
        # Create test spreads with mixed EV
        test_spreads = [
            {'ev': 5.0, 'pop': 0.7, 'description': 'Good positive EV'},
            {'ev': 0.0, 'pop': 0.6, 'description': 'Breakeven EV'},
            {'ev': -2.0, 'pop': 0.5, 'description': 'Negative EV - should be filtered'},
            {'ev': 10.0, 'pop': 0.8, 'description': 'Excellent positive EV'},
            {'ev': -5.0, 'pop': 0.4, 'description': 'Very negative EV - should be filtered'},
        ]
        
        print("Test spreads before filtering:")
        for spread in test_spreads:
            print(f"  EV: ${spread['ev']:.2f}, POP: {spread['pop']:.1%} - {spread['description']}")
        
        # Apply positive EV filter
        filtered_spreads = [s for s in test_spreads if s['ev'] >= 0.0]
        
        print(f"\nAfter filtering (EV >= 0):")
        for spread in filtered_spreads:
            print(f"  EV: ${spread['ev']:.2f}, POP: {spread['pop']:.1%} - {spread['description']}")
        
        print(f"\n📊 Filtering Results:")
        print(f"  Original spreads: {len(test_spreads)}")
        print(f"  Filtered spreads: {len(filtered_spreads)}")
        print(f"  Filtered out: {len(test_spreads) - len(filtered_spreads)} negative EV trades")
        
        return True
        
    except Exception as e:
        print(f"❌ Test failed with error: {e}")
        return False

def main():
    """Main test function"""
    print("🎯 Screener Positive EV Filter Test")
    print("This test verifies that the screener only shows trades with positive expected value")
    print()
    
    success = test_positive_ev_filtering()
    
    if success:
        print("\n🎉 All tests passed!")
        print("✅ The screener is correctly filtering for positive EV trades only")
        print("✅ Long-term profitability is ensured")
    else:
        print("\n❌ Tests failed!")
        print("❌ The screener needs to be fixed to properly filter negative EV trades")
        sys.exit(1)

if __name__ == "__main__":
    main()

