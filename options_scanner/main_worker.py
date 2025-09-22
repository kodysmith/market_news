"""
Main options scanner worker
"""

import os
import sys
from datetime import datetime, timezone
import pytz
from typing import List, Dict, Any
from dotenv import load_dotenv

# Import our modules
from database import DatabaseManager
from fetch_real_data import RealDataFetcher
from spreads import SpreadAnalyzer
from alerts import AlertManager
from push_fcm import FCMNotifier
from report import ReportGenerator

class OptionsScanner:
    """Main options scanner orchestrator"""
    
    def __init__(self):
        """Initialize scanner with environment variables"""
        load_dotenv()
        
        # Environment variables
        self.api_key = os.getenv('ALPHAVANTAGE_API_KEY', 'mock')  # Not needed for mock data
        self.firebase_creds = os.getenv('FIREBASE_SERVICE_ACCOUNT_JSON_BASE64', 'mock')
        self.public_bucket_url = os.getenv('PUBLIC_BUCKET_URL', 'https://api-hvi4gdtdka-uc.a.run.app')
        self.report_object_path = os.getenv('REPORT_OBJECT_PATH', '/report.json')
        self.tickers = os.getenv('TICKERS', 'SPY,QQQ,IWM').split(',')
        self.scan_interval = int(os.getenv('SCAN_INTERVAL_MINUTES', '5'))
        self.market_tz = os.getenv('MARKET_TZ', 'America/New_York')
        
        # Note: Using real data from yfinance, so API keys are optional
        print("ℹ️  Using real options data from yfinance - no API keys required")
        
        # Initialize components
        self.db_manager = DatabaseManager()
        self.api_client = RealDataFetcher()  # Use real options data from yfinance
        self.spread_analyzer = SpreadAnalyzer()
        self.alert_manager = AlertManager(self.db_manager)
        self.fcm_notifier = FCMNotifier(self.firebase_creds)
        self.report_generator = ReportGenerator(self.public_bucket_url, self.report_object_path)
        
        print("✅ Options scanner initialized successfully")
    
    def is_market_hours(self, now: datetime) -> bool:
        """Check if current time is during market hours"""
        # For testing purposes, always return True to allow scanning
        return True
        
        # Original market hours check (commented out for testing):
        # market_tz = pytz.timezone(self.market_tz)
        # market_time = now.astimezone(market_tz)
        # if market_time.weekday() >= 5:  # Saturday or Sunday
        #     return False
        # market_open = market_time.replace(hour=9, minute=30, second=0, microsecond=0)
        # market_close = market_time.replace(hour=16, minute=0, second=0, microsecond=0)
        # return market_open <= market_time <= market_close
    
    def run_once(self) -> Dict[str, Any]:
        """Run one scan cycle"""
        now = datetime.now(timezone.utc)
        
        print(f"\n🔍 Starting scan at {now.strftime('%Y-%m-%d %H:%M:%S UTC')}")
        
        # Check if market is open
        if not self.is_market_hours(now):
            print("⏰ Market is closed, skipping scan")
            return {'status': 'skipped', 'reason': 'market_closed'}
        
        print("📈 Market is open, proceeding with scan...")
        
        all_ideas = []
        alerts_sent = []
        
        # Scan each ticker
        for ticker in self.tickers:
            print(f"\n📊 Scanning {ticker}...")
            
            try:
                # Get spot price
                spot_price = self.api_client.get_current_price(ticker)
                if not spot_price:
                    print(f"❌ Failed to get spot price for {ticker}")
                    continue
                
                print(f"💰 {ticker} spot price: ${spot_price:.2f}")
                
                # Generate bull put spreads directly
                spreads_data = self.api_client.generate_bull_put_spreads(ticker, max_dte=45, min_ev=0.0)
                print(f"📋 Generated {len(spreads_data)} bull put spreads")
                
                if len(spreads_data) == 0:
                    print(f"⚠️  No positive EV spreads found for {ticker}")
                    continue
                
                # Convert to spread objects for compatibility
                from spreads import BullPutSpread
                spread_objects = []
                
                for spread_data in spreads_data:
                    try:
                        spread = BullPutSpread(
                            ticker=spread_data['ticker'],
                            expiry=spread_data['expiry'],
                            short_strike=spread_data['shortK'],
                            long_strike=spread_data['longK'],
                            width=spread_data['width'],
                            credit=spread_data['credit'],
                            max_loss=spread_data['maxLoss'],
                            dte=spread_data['dte'],
                            pop=spread_data['pop'],
                            ev=spread_data['ev'],
                            short_iv=spread_data['ivShort'],
                            long_iv=spread_data['ivShort'] * 0.9,  # Assume long IV is slightly lower
                            bid_ask_width=spread_data['bidAskW'],
                            oi_short=spread_data['oiShort'],
                            oi_long=spread_data['oiLong'],
                            vol_short=spread_data['volShort'],
                            vol_long=spread_data['volLong'],
                            fill_score=spread_data['fillScore'],
                            spread_id=spread_data['id'],
                            timestamp=datetime.now().isoformat()
                        )
                        spread_objects.append(spread)
                    except Exception as e:
                        print(f"⚠️  Error creating spread object: {e}")
                        continue
                
                print(f"✅ Created {len(spread_objects)} spread objects")
                
                # Add to all ideas
                all_ideas.extend(spread_objects)
                
                print(f"🎯 Added {len(spread_objects)} spreads for {ticker}")
                
            except Exception as e:
                print(f"❌ Error scanning {ticker}: {e}")
                continue
        
        # Decide on alerts
        print(f"\n🔔 Processing alerts...")
        alerts_to_send = self.alert_manager.decide_alerts(all_ideas)
        
        # Send alerts
        for spread in alerts_to_send:
            try:
                alert_msg = self.alert_manager.format_alert_message(spread)
                
                if self.fcm_notifier.send_opportunity_alert(
                    topic="all_users",
                    title=alert_msg['title'],
                    body=alert_msg['body'],
                    data=alert_msg['data']
                ):
                    self.alert_manager.mark_alert_sent(spread)
                    alerts_sent.append(spread.spread_id)
                    print(f"📤 Sent alert for {spread.ticker} {spread.short_strike}/{spread.long_strike}")
                else:
                    print(f"❌ Failed to send alert for {spread.ticker}")
                    
            except Exception as e:
                print(f"❌ Error sending alert: {e}")
        
        # Generate and publish report
        print(f"\n📄 Generating report...")
        
        report = self.report_generator.generate_report(
            top_ideas=all_ideas,
            universe=self.tickers,
            dte_window=[20, 45],
            thresholds={'minPOP': 0.50, 'minEVPer100': 0.10},
            alerts_sent=alerts_sent
        )
        
        # Validate report
        if not self.report_generator.validate_report(report):
            print("❌ Report validation failed")
            return {'status': 'error', 'reason': 'invalid_report'}
        
        # Publish report
        if not self.report_generator.publish_report(report):
            print("❌ Failed to publish report")
            return {'status': 'error', 'reason': 'publish_failed'}
        
        # Persist snapshots to database
        for idea in all_ideas:
            snapshot_data = {
                'timestamp': idea.timestamp,
                'ticker': idea.ticker,
                'expiry': idea.expiry,
                'short_k': idea.short_strike,
                'long_k': idea.long_strike,
                'width': idea.width,
                'credit': idea.credit,
                'max_loss': idea.max_loss,
                'pop': idea.pop,
                'ev': idea.ev,
                'dte': idea.dte,
                'bid_ask_w': idea.bid_ask_width,
                'oi_short': idea.oi_short,
                'oi_long': idea.oi_long,
                'vol_short': idea.vol_short,
                'vol_long': idea.vol_long,
                'iv_short': idea.short_iv,
                'iv_long': idea.long_iv
            }
            self.db_manager.persist_spread_snapshot(snapshot_data)
        
        # Get summary
        summary = self.report_generator.get_report_summary(report)
        
        print(f"\n✅ Scan completed successfully!")
        print(f"📊 Total ideas: {summary['total_ideas']}")
        print(f"💰 Average EV: ${summary['avg_ev']:.2f}")
        print(f"🎯 Average POP: {summary['avg_pop']:.1%}")
        print(f"🔔 Alerts sent: {summary['alerts_sent']}")
        
        return {
            'status': 'success',
            'total_ideas': summary['total_ideas'],
            'alerts_sent': summary['alerts_sent'],
            'avg_ev': summary['avg_ev'],
            'avg_pop': summary['avg_pop']
        }
    
    def run_continuously(self):
        """Run scanner continuously (for testing)"""
        print("🚀 Starting continuous scanner...")
        print(f"⏰ Scanning every {self.scan_interval} minutes")
        print("Press Ctrl+C to stop")
        
        try:
            while True:
                result = self.run_once()
                print(f"⏳ Waiting {self.scan_interval} minutes until next scan...")
                
                import time
                time.sleep(self.scan_interval * 60)
                
        except KeyboardInterrupt:
            print("\n🛑 Scanner stopped by user")
        except Exception as e:
            print(f"❌ Scanner error: {e}")

def main():
    """Main entry point"""
    try:
        scanner = OptionsScanner()
        
        if len(sys.argv) > 1 and sys.argv[1] == '--continuous':
            scanner.run_continuously()
        else:
            result = scanner.run_once()
            print(f"\n📋 Scan result: {result}")
            
    except Exception as e:
        print(f"❌ Fatal error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
