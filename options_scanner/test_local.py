#!/usr/bin/env python3
"""
Test the options scanner locally with mock data
"""

import os
import sys
from datetime import datetime, timezone

# Set up test environment
os.environ['ALPHAVANTAGE_API_KEY'] = 'test_key'
os.environ['FIREBASE_SERVICE_ACCOUNT_JSON_BASE64'] = 'dGVzdA=='  # "test" in base64
os.environ['PUBLIC_BUCKET_URL'] = 'https://api-hvi4gdtdka-uc.a.run.app'
os.environ['REPORT_OBJECT_PATH'] = '/report.json'
os.environ['TICKERS'] = 'SPY'
os.environ['SCAN_INTERVAL_MINUTES'] = '5'
os.environ['MARKET_TZ'] = 'America/New_York'

# Import our modules
from database import DatabaseManager
from spreads import SpreadAnalyzer, BullPutSpread
from alerts import AlertManager
from report import ReportGenerator

def test_with_mock_data():
    """Test the scanner with mock data"""
    print("🧪 Testing Options Scanner with Mock Data")
    print("=" * 50)
    
    # Initialize components
    db_manager = DatabaseManager("test_scanner.db")
    spread_analyzer = SpreadAnalyzer()
    alert_manager = AlertManager(db_manager)
    report_generator = ReportGenerator("https://api-hvi4gdtdka-uc.a.run.app", "/report.json")
    
    # Create mock bull put spreads
    mock_spreads = [
        BullPutSpread(
            ticker="SPY",
            expiry="2025-10-18",
            short_strike=510.0,
            long_strike=505.0,
            width=5.0,
            credit=0.95,
            max_loss=4.05,
            pop=0.64,
            ev=50.0,  # High EV to pass the per-100 test
            dte=33,
            short_iv=0.22,
            long_iv=0.20,
            bid_ask_width=0.06,
            oi_short=1423,
            oi_long=1312,
            vol_short=289,
            vol_long=214,
            fill_score=0.82,
            spread_id="test_spread_1",
            timestamp=datetime.now().isoformat()
        ),
        BullPutSpread(
            ticker="SPY",
            expiry="2025-10-18",
            short_strike=505.0,
            long_strike=500.0,
            width=5.0,
            credit=0.75,
            max_loss=4.25,
            pop=0.58,
            ev=30.0,  # High EV to pass the per-100 test
            dte=33,
            short_iv=0.20,
            long_iv=0.18,
            bid_ask_width=0.08,
            oi_short=1200,
            oi_long=1100,
            vol_short=200,
            vol_long=150,
            fill_score=0.75,
            spread_id="test_spread_2",
            timestamp=datetime.now().isoformat()
        ),
        BullPutSpread(
            ticker="SPY",
            expiry="2025-10-18",
            short_strike=500.0,
            long_strike=495.0,
            width=5.0,
            credit=0.55,
            max_loss=4.45,
            pop=0.52,
            ev=0.05,  # Low EV - should be filtered out
            dte=33,
            short_iv=0.18,
            long_iv=0.16,
            bid_ask_width=0.10,
            oi_short=800,
            oi_long=700,
            vol_short=100,
            vol_long=80,
            fill_score=0.60,
            spread_id="test_spread_3",
            timestamp=datetime.now().isoformat()
        )
    ]
    
    print(f"📊 Created {len(mock_spreads)} mock spreads")
    
    # Test filtering
    print("\n🔍 Testing spread filtering...")
    filtered_spreads = spread_analyzer.apply_filters(mock_spreads)
    print(f"✅ {len(filtered_spreads)} spreads passed filters")
    
    for i, spread in enumerate(filtered_spreads, 1):
        print(f"  {i}. {spread.ticker} {spread.short_strike}/{spread.long_strike} - "
              f"EV: ${spread.ev:.2f}, POP: {spread.pop:.1%}, Credit: ${spread.credit:.2f}")
    
    # Test alert decisions
    print("\n🔔 Testing alert decisions...")
    alerts_to_send = alert_manager.decide_alerts(filtered_spreads)
    print(f"📤 {len(alerts_to_send)} alerts would be sent")
    
    for i, spread in enumerate(alerts_to_send, 1):
        alert_msg = alert_manager.format_alert_message(spread)
        print(f"  {i}. {alert_msg['title']}")
        print(f"     {alert_msg['body']}")
    
    # Test report generation
    print("\n📄 Testing report generation...")
    report = report_generator.generate_report(
        top_ideas=filtered_spreads,
        universe=["SPY"],
        dte_window=[20, 45],
        thresholds={'minPOP': 0.50, 'minEVPer100': 0.10},
        alerts_sent=[spread.spread_id for spread in alerts_to_send]
    )
    
    # Validate report
    if report_generator.validate_report(report):
        print("✅ Report structure is valid")
    else:
        print("❌ Report validation failed")
        return False
    
    # Save report
    if report_generator.publish_report(report):
        print("✅ Report saved to report.json")
    else:
        print("❌ Failed to save report")
        return False
    
    # Get summary
    summary = report_generator.get_report_summary(report)
    print(f"\n📊 Report Summary:")
    print(f"  Total ideas: {summary['total_ideas']}")
    print(f"  Average EV: ${summary['avg_ev']:.2f}")
    print(f"  Average POP: {summary['avg_pop']:.1%}")
    print(f"  Best EV: ${summary['best_ev']:.2f}")
    print(f"  Best POP: {summary['best_pop']:.1%}")
    print(f"  Alerts sent: {summary.get('alerts_sent', 0)}")
    
    # Test database operations
    print("\n💾 Testing database operations...")
    for spread in filtered_spreads:
        snapshot_data = {
            'timestamp': spread.timestamp,
            'ticker': spread.ticker,
            'expiry': spread.expiry,
            'short_k': spread.short_strike,
            'long_k': spread.long_strike,
            'width': spread.width,
            'credit': spread.credit,
            'max_loss': spread.max_loss,
            'pop': spread.pop,
            'ev': spread.ev,
            'dte': spread.dte,
            'bid_ask_w': spread.bid_ask_width,
            'oi_short': spread.oi_short,
            'oi_long': spread.oi_long,
            'vol_short': spread.vol_short,
            'vol_long': spread.vol_long,
            'iv_short': spread.short_iv,
            'iv_long': spread.long_iv
        }
        db_manager.persist_spread_snapshot(snapshot_data)
    
    print("✅ Database operations completed")
    
    # Clean up test database
    import os
    if os.path.exists("test_scanner.db"):
        os.remove("test_scanner.db")
        print("🧹 Cleaned up test database")
    
    print("\n🎉 All tests completed successfully!")
    return True

if __name__ == "__main__":
    success = test_with_mock_data()
    sys.exit(0 if success else 1)
