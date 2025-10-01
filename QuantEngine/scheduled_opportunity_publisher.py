#!/usr/bin/env python3
"""
Scheduled Opportunity Publisher

Runs opportunity scans on schedule and publishes results to Firebase
for the mobile app. Integrates with the opportunity database scanner.
"""

import schedule
import time
import logging
import json
import requests
import pandas as pd
from datetime import datetime, timedelta
import sys
import os
from typing import Dict, List, Any

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from opportunity_database import OpportunityDatabase

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('opportunity_publisher.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class OpportunityPublisher:
    """Publishes opportunities to Firebase for mobile app"""
    
    def __init__(self, firebase_url: str = "https://us-central1-kardova-capital.cloudfunctions.net"):
        self.firebase_url = firebase_url
        self.db = OpportunityDatabase()
        
    def publish_opportunities(self, opportunities: List[Dict[str, Any]]) -> bool:
        """Publish opportunities to Firebase"""
        try:
            # Format opportunities for mobile app
            mobile_opportunities = []
            
            for opp in opportunities:
                mobile_opp = {
                    'type': 'trading_opportunity',
                    'ticker': opp['ticker'],
                    'sector': opp['sector'],
                    'opportunity_type': opp['opportunity_type'],
                    'overall_score': opp['overall_score'],
                    'current_price': opp['current_price'],
                    'target_price': opp['target_price'],
                    'stop_loss': opp['stop_loss'],
                    'risk_reward_ratio': opp['risk_reward_ratio'],
                    'technical_score': opp['technical_score'],
                    'fundamental_score': opp['fundamental_score'],
                    'rsi': opp['rsi'],
                    'trend_direction': opp['trend_direction'],
                    'pe_ratio': opp['pe_ratio'],
                    'revenue_growth': opp['revenue_growth'],
                    'analyst_rating': opp['analyst_rating'],
                    'scan_date': opp['scan_date'],
                    'timestamp': datetime.now().isoformat(),
                    'published': True
                }
                mobile_opportunities.append(mobile_opp)
            
            # Publish to Firebase
            response = requests.post(
                f"{self.firebase_url}/publish-opportunities",
                headers={'Content-Type': 'application/json'},
                json={
                    'opportunities': mobile_opportunities,
                    'scan_timestamp': datetime.now().isoformat(),
                    'total_count': len(mobile_opportunities)
                },
                timeout=30
            )
            
            if response.status_code == 200:
                logger.info(f"✅ Published {len(mobile_opportunities)} opportunities to mobile app")
                return True
            else:
                logger.error(f"❌ Failed to publish opportunities: {response.status_code} - {response.text}")
                return False
                
        except Exception as e:
            logger.error(f"❌ Error publishing opportunities: {e}")
            return False
    
    def publish_sector_summary(self, sector_summary: pd.DataFrame) -> bool:
        """Publish sector summary to Firebase"""
        try:
            # Convert DataFrame to mobile format
            mobile_sectors = []
            for _, row in sector_summary.iterrows():
                mobile_sector = {
                    'type': 'sector_summary',
                    'sector': row['sector'],
                    'total_opportunities': int(row['total_opportunities']),
                    'avg_score': float(row['avg_score']),
                    'max_score': float(row['max_score']),
                    'min_score': float(row['min_score']),
                    'strong_buy_count': int(row['strong_buy']),
                    'buy_count': int(row['buy']),
                    'hold_count': int(row['hold']),
                    'sell_count': int(row['sell']),
                    'strong_sell_count': int(row['strong_sell']),
                    'timestamp': datetime.now().isoformat(),
                    'published': True
                }
                mobile_sectors.append(mobile_sector)
            
            # Publish to Firebase
            response = requests.post(
                f"{self.firebase_url}/publish-sector-summary",
                headers={'Content-Type': 'application/json'},
                json={
                    'sectors': mobile_sectors,
                    'summary_timestamp': datetime.now().isoformat()
                },
                timeout=30
            )
            
            if response.status_code == 200:
                logger.info(f"✅ Published sector summary to mobile app")
                return True
            else:
                logger.error(f"❌ Failed to publish sector summary: {response.status_code} - {response.text}")
                return False
                
        except Exception as e:
            logger.error(f"❌ Error publishing sector summary: {e}")
            return False
    
    def run_scan_and_publish(self):
        """Run full scan and publish results"""
        logger.info("🚀 Starting scheduled opportunity scan and publish...")
        
        try:
            # Run opportunity scan
            results = self.db.scan_all_sectors()
            
            if results['total_opportunities'] == 0:
                logger.warning("⚠️ No opportunities found in scan")
                return
            
            # Get top opportunities for mobile app
            top_opportunities = self.db.query_opportunities(min_score=60, limit=50)
            
            if not top_opportunities.empty:
                # Convert to list of dicts
                opportunities_list = top_opportunities.to_dict('records')
                
                # Publish opportunities
                opp_success = self.publish_opportunities(opportunities_list)
                
                if opp_success:
                    logger.info(f"📱 Published {len(opportunities_list)} opportunities to mobile app")
                else:
                    logger.error("❌ Failed to publish opportunities")
            
            # Get sector summary
            sector_summary = self.db.get_sector_summary()
            
            if not sector_summary.empty:
                # Publish sector summary
                sector_success = self.publish_sector_summary(sector_summary)
                
                if sector_success:
                    logger.info("📊 Published sector summary to mobile app")
                else:
                    logger.error("❌ Failed to publish sector summary")
            
            # Log results
            logger.info(f"✅ Scan and publish completed:")
            logger.info(f"  Total opportunities: {results['total_opportunities']}")
            logger.info(f"  Published to mobile: {len(opportunities_list) if not top_opportunities.empty else 0}")
            
        except Exception as e:
            logger.error(f"❌ Scan and publish failed: {e}")
            raise
        finally:
            self.db.close()
    
    def close(self):
        """Close database connection"""
        if hasattr(self, 'db'):
            self.db.close()


def main():
    """Main scheduler function"""
    logger.info("🕐 Starting Opportunity Publisher Scheduler")
    
    publisher = OpportunityPublisher()
    
    try:
        # Schedule scans
        # Market hours scans
        schedule.every().day.at("09:30").do(publisher.run_scan_and_publish)  # Market open
        schedule.every().day.at("12:00").do(publisher.run_scan_and_publish)  # Midday
        schedule.every().day.at("15:30").do(publisher.run_scan_and_publish)  # Market close
        
        # Weekend scan
        schedule.every().saturday.at("10:00").do(publisher.run_scan_and_publish)
        
        logger.info("📅 Scheduler configured:")
        logger.info("  - 9:30 AM EST: Market open scan")
        logger.info("  - 12:00 PM EST: Midday scan") 
        logger.info("  - 3:30 PM EST: Market close scan")
        logger.info("  - Saturday 10:00 AM EST: Weekend scan")
        
        # Run initial scan
        logger.info("🔄 Running initial scan and publish...")
        publisher.run_scan_and_publish()
        
        # Keep running
        logger.info("⏰ Scheduler running... Press Ctrl+C to stop")
        
        while True:
            schedule.run_pending()
            time.sleep(60)  # Check every minute
            
    except KeyboardInterrupt:
        logger.info("🛑 Scheduler stopped by user")
    except Exception as e:
        logger.error(f"❌ Scheduler error: {e}")
        raise
    finally:
        publisher.close()


if __name__ == '__main__':
    main()
