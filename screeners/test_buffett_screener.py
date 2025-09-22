#!/usr/bin/env python3
"""
Test script for the Buffett screener
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from buffett_screener_api import BuffettScreener

def test_screener():
    """Test the Buffett screener with a small sample"""
    print("🧪 Testing Buffett Screener...")
    
    screener = BuffettScreener()
    
    # Test with a small sample of companies
    test_symbols = ['AAPL', 'MSFT', 'JNJ', 'KO', 'PG']
    
    print(f"Testing with symbols: {test_symbols}")
    
    try:
        results = screener.run_screening(test_symbols)
        
        print(f"\n📊 Results Summary:")
        print(f"  Total companies screened: {results['total_companies_screened']}")
        print(f"  Companies passing screen: {results['companies_passing_screen']}")
        print(f"  Companies with MOS ≥30%: {results['companies_with_mos_30_plus']}")
        print(f"  Screening duration: {results['screening_duration_seconds']:.1f} seconds")
        print(f"  Average moat score: {results['avg_moat_score']:.1f}/10")
        print(f"  Average management score: {results['avg_management_score']:.1f}/10")
        
        if results['top_candidates']:
            print(f"\n🎯 Top Candidates:")
            for company in results['top_candidates']:
                dcf = company.get('dcf', {})
                print(f"  {company['symbol']}: MOS {dcf.get('margin_of_safety', 0):.1%}, Moat {company['moat_score']}/10")
        
        # Save results
        import json
        with open('test_buffett_results.json', 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        print(f"\n✅ Test completed successfully!")
        print(f"Results saved to test_buffett_results.json")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_screener()

