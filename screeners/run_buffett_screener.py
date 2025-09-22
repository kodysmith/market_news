#!/usr/bin/env python3
"""
Run the Buffett screener with real API calls
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from buffett_screener_api import BuffettScreener
import json

def main():
    """Run the Buffett screener and display results"""
    print("🏰 Starting Buffett Screener with Real API Data...")
    print("=" * 60)
    
    screener = BuffettScreener()
    
    # Test with a small sample first
    test_symbols = ['AAPL', 'MSFT', 'JNJ', 'KO', 'PG', 'WMT', 'BRK-B']
    
    print(f"📊 Screening {len(test_symbols)} companies: {', '.join(test_symbols)}")
    print()
    
    try:
        results = screener.run_screening(test_symbols)
        
        print("📈 SCREENING RESULTS")
        print("=" * 60)
        print(f"Total companies screened: {results['total_companies_screened']}")
        print(f"Companies passing screen: {results['companies_passing_screen']}")
        print(f"Companies with MOS ≥30%: {results['companies_with_mos_30_plus']}")
        print(f"Screening duration: {results['screening_duration_seconds']:.1f} seconds")
        print(f"Average moat score: {results['avg_moat_score']:.1f}/10")
        print(f"Average management score: {results['avg_management_score']:.1f}/10")
        print()
        
        if results['companies_passing_screen'] > 0:
            print("✅ COMPANIES PASSING SCREEN")
            print("=" * 60)
            for company in results['all_companies']:
                if company['passes']:
                    dcf = company.get('dcf', {})
                    print(f"🎯 {company['symbol']} - {company['name']}")
                    print(f"   Sector: {company['sector']}")
                    print(f"   Market Cap: ${company['market_cap']/1e9:.1f}B")
                    print(f"   Score: {company['score']}/100")
                    print(f"   Moat: {company['moat_score']}/10, Mgmt: {company['management_score']}/10")
                    if dcf:
                        print(f"   Intrinsic Value: ${dcf.get('intrinsic_value_per_share', 0):.2f}")
                        print(f"   Current Price: ${company['current_price']:.2f}")
                        print(f"   Margin of Safety: {dcf.get('margin_of_safety', 0)*100:.1f}%")
                    print()
        else:
            print("❌ NO COMPANIES PASSED THE SCREEN")
            print("=" * 60)
            print("This is common in current market conditions due to:")
            print("• High valuations (P/E ratios > 30)")
            print("• Low growth rates")
            print("• High debt levels")
            print("• Missing financial data")
            print()
            
            print("📊 TOP CANDIDATES (by score)")
            print("=" * 60)
            # Sort by score and show top 3
            sorted_companies = sorted(results['all_companies'], key=lambda x: x['score'], reverse=True)
            for i, company in enumerate(sorted_companies[:3]):
                print(f"{i+1}. {company['symbol']} - Score: {company['score']}/100")
                print(f"   Reasons failed: {', '.join(company['reasons'][:3])}")
                print()
        
        # Save detailed results
        with open('buffett_screener_detailed.json', 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        print("💾 Detailed results saved to buffett_screener_detailed.json")
        
        # Show API status
        print()
        print("🔌 API STATUS")
        print("=" * 60)
        print(f"Alpha Vantage: {'✅ Available' if screener.alpha_vantage_key else '❌ Not configured'}")
        print(f"FMP API: {'✅ Available' if screener.fmp_key else '❌ Not configured'}")
        print("Yahoo Finance: ✅ Available (primary data source)")
        
        if not screener.alpha_vantage_key and not screener.fmp_key:
            print()
            print("💡 TIP: Add API keys to .env file for enhanced data:")
            print("   ALPHAVANTAGE_API_KEY=your_key_here")
            print("   FMP_API_KEY=your_key_here")
        
    except Exception as e:
        print(f"❌ Error running screener: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()

