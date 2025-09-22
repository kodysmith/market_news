#!/usr/bin/env python3
"""
Test Live Data Integration for QuantBot

Verifies that all live data sources are working correctly:
- Yahoo Finance real-time quotes
- Alpha Vantage intraday data
- FMP news API with sentiment
- Economic data feeds

Run this before starting QuantBot to ensure all APIs are configured.
"""

import asyncio
import sys
from pathlib import Path
import json
from datetime import datetime

# Add QuantEngine to path
sys.path.insert(0, str(Path(__file__).parent))

from engine.data_ingestion.live_data_manager import LiveDataManager

class LiveDataTester:
    """Test live data integrations"""

    def __init__(self):
        self.config = {
            "universe": {
                "equities": ["SPY", "QQQ", "AAPL", "MSFT"],
                "sectors": ["XLE", "XLF"],
                "leveraged": ["TQQQ"]
            }
        }
        self.data_manager = None

    async def initialize(self):
        """Initialize the data manager"""
        print("🔧 Initializing Live Data Manager...")
        self.data_manager = LiveDataManager(self.config)
        await self.data_manager.initialize()
        print("✅ Live Data Manager initialized")

    async def test_yahoo_finance(self):
        """Test Yahoo Finance integration"""
        print("\n📊 Testing Yahoo Finance Live Quotes...")

        symbols = ["SPY", "QQQ", "AAPL"]
        quotes = await self.data_manager.get_live_quotes(symbols)

        if quotes:
            print(f"✅ Successfully fetched {len(quotes)} quotes:")
            for symbol, quote in quotes.items():
                print(f"   {symbol}: ${quote.price:.2f} (Vol: {quote.volume:,})")
            return True
        else:
            print("❌ Failed to fetch Yahoo Finance quotes")
            return False

    async def test_alpha_vantage(self):
        """Test Alpha Vantage integration"""
        print("\n📈 Testing Alpha Vantage Intraday Data...")

        try:
            intraday_data = await self.data_manager.get_intraday_data("SPY", interval="5min", bars=10)

            if not intraday_data.empty:
                print(f"✅ Successfully fetched {len(intraday_data)} intraday bars")
                print("   Sample data:")
                print(f"   {intraday_data.head(3)}")
                return True
            else:
                print("❌ No intraday data received (check API key)")
                return False

        except Exception as e:
            print(f"❌ Alpha Vantage test failed: {e}")
            return False

    async def test_fmp_news(self):
        """Test FMP News API integration"""
        print("\n📰 Testing FMP News API...")

        try:
            news_items = await self.data_manager.get_live_news(limit=5)

            if news_items:
                print(f"✅ Successfully fetched {len(news_items)} news items")
                for i, item in enumerate(news_items[:3]):
                    print(f"   {i+1}. {item.title[:60]}...")
                    print(f"       Sentiment: {item.sentiment_score:.2f}, Source: {item.source}")
                return True
            else:
                print("❌ No news data received (check API key)")
                return False

        except Exception as e:
            print(f"❌ FMP News test failed: {e}")
            return False

    async def test_economic_data(self):
        """Test economic data integration"""
        print("\n💰 Testing Economic Data...")

        try:
            economic_data = await self.data_manager.get_economic_data()

            if economic_data:
                print(f"✅ Successfully fetched {len(economic_data)} economic indicators:")
                for indicator, value in economic_data.items():
                    print(f"   {indicator}: {value}")
                return True
            else:
                print("❌ No economic data received")
                return False

        except Exception as e:
            print(f"❌ Economic data test failed: {e}")
            return False

    async def test_market_overview(self):
        """Test complete market overview"""
        print("\n🌍 Testing Complete Market Overview...")

        try:
            overview = await self.data_manager.get_market_overview()

            if overview and 'error' not in overview:
                print("✅ Successfully fetched market overview:")
                print(f"   Data Quality: {overview.get('data_quality', 'unknown')}")
                print(f"   Market Sentiment: {overview.get('market_sentiment', 0):.2f}")

                indices = overview.get('indices', {})
                if indices:
                    print(f"   Major Indices: {len(indices)} fetched")

                economic = overview.get('economic_indicators', {})
                if economic:
                    print(f"   Economic Indicators: {len(economic)} fetched")

                return True
            else:
                print("❌ Failed to fetch market overview")
                if 'error' in overview:
                    print(f"   Error: {overview['error']}")
                return False

        except Exception as e:
            print(f"❌ Market overview test failed: {e}")
            return False

    def check_api_keys(self):
        """Check if API keys are configured"""
        print("\n🔑 Checking API Key Configuration...")

        import os
        from dotenv import load_dotenv

        load_dotenv()

        keys_status = {
            'FMP_API_KEY': bool(os.getenv('FMP_API_KEY')),
            'ALPHA_VANTAGE_API_KEY': bool(os.getenv('ALPHA_VANTAGE_API_KEY')),
            'FRED_API_KEY': bool(os.getenv('FRED_API_KEY'))
        }

        print("API Keys Status:")
        for key_name, configured in keys_status.items():
            status = "✅ Configured" if configured else "❌ Missing"
            print(f"   {key_name}: {status}")

        return keys_status

    async def run_all_tests(self):
        """Run all live data integration tests"""

        print("🧪 QuantBot Live Data Integration Test Suite")
        print("=" * 50)
        print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print()

        # Check API keys first
        api_keys = self.check_api_keys()

        # Initialize data manager
        await self.initialize()

        # Run tests
        test_results = {}

        test_results['yahoo_finance'] = await self.test_yahoo_finance()
        test_results['alpha_vantage'] = await self.test_alpha_vantage()
        test_results['fmp_news'] = await self.test_fmp_news()
        test_results['economic_data'] = await self.test_economic_data()
        test_results['market_overview'] = await self.test_market_overview()

        # Cleanup
        await self.data_manager.close()

        # Report results
        print("\n" + "=" * 50)
        print("📊 Test Results Summary")
        print("=" * 50)

        passed = sum(test_results.values())
        total = len(test_results)

        print(f"Tests Passed: {passed}/{total}")

        if passed == total:
            print("🎉 ALL TESTS PASSED - Live data integration is ready!")
            print("\n🚀 QuantBot can now run with real market data!")
        elif passed >= total * 0.7:
            print("⚠️ MOST TESTS PASSED - QuantBot will work with some fallback data")
        else:
            print("❌ MANY TESTS FAILED - Check API keys and network connectivity")

        # Detailed results
        print("\nDetailed Results:")
        for test_name, passed in test_results.items():
            status = "✅ PASSED" if passed else "❌ FAILED"
            print(f"   {test_name.replace('_', ' ').title()}: {status}")

        # Recommendations
        print("\n💡 Recommendations:")
        if not api_keys['FMP_API_KEY']:
            print("   - Set FMP_API_KEY in .env for news data")
        if not api_keys['ALPHA_VANTAGE_API_KEY']:
            print("   - Set ALPHA_VANTAGE_API_KEY in .env for intraday data")
        if not api_keys['FRED_API_KEY']:
            print("   - Set FRED_API_KEY in .env for economic data")

        print(f"\nFinished: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

        return passed == total

async def main():
    """Main test function"""
    tester = LiveDataTester()

    try:
        success = await tester.run_all_tests()
        return success
    except Exception as e:
        print(f"❌ Test suite failed with error: {e}")
        return False

if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)
