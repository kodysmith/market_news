#!/usr/bin/env python3
"""
Scheduler for Overbought/Oversold Scanner

Runs the scanner every few hours and maintains an ongoing list of opportunities.
"""

import os
import sys
import time
import schedule
import logging
from pathlib import Path
from datetime import datetime

# Add QuantEngine to path
quant_engine_dir = Path(__file__).parent
sys.path.insert(0, str(quant_engine_dir))

from overbought_oversold_scanner import OverboughtOversoldScanner

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('scanner_scheduler.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

class ScannerScheduler:
    """Scheduler for running the overbought/oversold scanner"""
    
    def __init__(self):
        self.scanner = OverboughtOversoldScanner()
        logger.info("✅ Scanner Scheduler initialized")
    
    def run_scheduled_scan(self):
        """Run a scheduled scan"""
        logger.info("🕐 Running scheduled scan...")
        try:
            report = self.scanner.run_scan(save_report=True)
            
            # Log summary
            with open(self.scanner.data_file, 'r') as f:
                data = json.load(f)
            
            overbought = len([s for s in data['stocks'].values() if s['condition'] in ['overbought', 'extreme_overbought']])
            oversold = len([s for s in data['stocks'].values() if s['condition'] in ['oversold', 'extreme_oversold']])
            
            logger.info(f"📊 Scan completed: {overbought} overbought, {oversold} oversold stocks")
            
        except Exception as e:
            logger.error(f"❌ Scheduled scan failed: {e}")
    
    def start_scheduler(self, interval_hours: int = 4):
        """Start the scheduler"""
        logger.info(f"🚀 Starting scanner scheduler (every {interval_hours} hours)")
        
        # Schedule the job
        schedule.every(interval_hours).hours.do(self.run_scheduled_scan)
        
        # Run initial scan
        logger.info("🔄 Running initial scan...")
        self.run_scheduled_scan()
        
        # Keep running
        while True:
            try:
                schedule.run_pending()
                time.sleep(60)  # Check every minute
            except KeyboardInterrupt:
                logger.info("🛑 Scheduler stopped by user")
                break
            except Exception as e:
                logger.error(f"❌ Scheduler error: {e}")
                time.sleep(300)  # Wait 5 minutes before retry

def main():
    """Main entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Scanner Scheduler')
    parser.add_argument('--interval', type=int, default=4, help='Scan interval in hours (default: 4)')
    parser.add_argument('--once', action='store_true', help='Run once and exit')
    
    args = parser.parse_args()
    
    scheduler = ScannerScheduler()
    
    if args.once:
        logger.info("🔄 Running single scan...")
        scheduler.run_scheduled_scan()
    else:
        scheduler.start_scheduler(args.interval)

if __name__ == "__main__":
    main()

