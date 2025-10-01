#!/usr/bin/env python3
"""
Opportunity Scanner Scheduler

Runs the opportunity database scanner on a schedule.
Can be run as a cron job or daemon.
"""

import schedule
import time
import logging
from datetime import datetime
import sys
import os

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from opportunity_database import OpportunityDatabase

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('opportunity_scanner.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


def run_opportunity_scan():
    """Run the opportunity scan and log results"""
    logger.info("🚀 Starting scheduled opportunity scan...")
    
    try:
        db = OpportunityDatabase()
        results = db.scan_all_sectors()
        
        logger.info(f"✅ Scan completed successfully")
        logger.info(f"📊 Total opportunities found: {results['total_opportunities']}")
        
        # Log sector breakdown
        for sector, count in results['sector_breakdown'].items():
            logger.info(f"  {sector}: {count} opportunities")
        
        # Log top opportunities
        top_opps = db.get_top_opportunities(5)
        if not top_opps.empty:
            logger.info("🏆 Top 5 opportunities:")
            for _, opp in top_opps.iterrows():
                logger.info(f"  {opp['ticker']} ({opp['sector']}): {opp['opportunity_type']} - Score: {opp['overall_score']:.1f}")
        
        db.close()
        
    except Exception as e:
        logger.error(f"❌ Scan failed: {e}")
        raise


def main():
    """Main scheduler function"""
    logger.info("🕐 Starting Opportunity Scanner Scheduler")
    
    # Schedule scans
    # Every 4 hours during market hours (9 AM - 4 PM EST)
    schedule.every().day.at("09:00").do(run_opportunity_scan)
    schedule.every().day.at("13:00").do(run_opportunity_scan)
    schedule.every().day.at("16:00").do(run_opportunity_scan)
    
    # Weekend scan (Saturday morning)
    schedule.every().saturday.at("10:00").do(run_opportunity_scan)
    
    logger.info("📅 Scheduler configured:")
    logger.info("  - 9:00 AM EST: Morning scan")
    logger.info("  - 1:00 PM EST: Midday scan") 
    logger.info("  - 4:00 PM EST: Evening scan")
    logger.info("  - Saturday 10:00 AM EST: Weekend scan")
    
    # Run initial scan
    logger.info("🔄 Running initial scan...")
    run_opportunity_scan()
    
    # Keep running
    logger.info("⏰ Scheduler running... Press Ctrl+C to stop")
    
    try:
        while True:
            schedule.run_pending()
            time.sleep(60)  # Check every minute
    except KeyboardInterrupt:
        logger.info("🛑 Scheduler stopped by user")
    except Exception as e:
        logger.error(f"❌ Scheduler error: {e}")
        raise


if __name__ == '__main__':
    main()
