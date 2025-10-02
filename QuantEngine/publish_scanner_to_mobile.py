#!/usr/bin/env python3
"""
Publish Production Scanner Data to Mobile App
Sends improved scanner results to Firebase for mobile app display
"""

import requests
import json
import logging
from datetime import datetime
import sys
import os

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from production_scanner import ProductionScanner

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ScannerPublisher:
    def __init__(self, firebase_api_url: str):
        self.firebase_api_url = firebase_api_url
        self.scanner = ProductionScanner()
        
    def publish_to_mobile(self, tickers: list = None) -> dict:
        """Run scanner and publish results to mobile app"""
        try:
            # Default tickers if none provided
            if tickers is None:
                tickers = [
                    'AAPL', 'MSFT', 'GOOGL', 'TSLA', 'NVDA', 'AMZN', 'META', 'NFLX', 'AMD', 'INTC',
                    'JPM', 'BAC', 'WFC', 'GS', 'MS', 'XOM', 'CVX', 'COP', 'EOG', 'SLB',
                    'JNJ', 'PFE', 'UNH', 'ABBV', 'MRK', 'TMO', 'ABT', 'DHR', 'BMY', 'LLY',
                    'HD', 'LOW', 'MCD', 'SBUX', 'NKE', 'WMT', 'TGT', 'COST', 'PG', 'KO',
                    'SPY', 'QQQ', 'IWM', 'XLF', 'XLK', 'XLE', 'XLV', 'XLI', 'XLY', 'XLP'
                ]
            
            logger.info(f"🚀 Running production scanner for {len(tickers)} tickers...")
            
            # Run the production scanner
            results = self.scanner.scan_production(tickers)
            
            if results['successful_scans'] == 0:
                logger.error("❌ No successful scans - cannot publish data")
                return {'success': False, 'error': 'No successful scans'}
            
            # Prepare data for Firebase
            firebase_data = {
                'opportunities': results['opportunities'],
                'scan_timestamp': results['scan_timestamp'],
                'total_tickers': results['total_tickers'],
                'successful_scans': results['successful_scans'],
                'failed_scans': results['failed_scans'],
                'summary': results['summary']
            }
            
            # Publish to Firebase
            response = requests.post(
                f"{self.firebase_api_url}/publish-scanner-data",
                json=firebase_data,
                headers={'Content-Type': 'application/json'},
                timeout=30
            )
            
            if response.status_code == 200:
                response_data = response.json()
                logger.info(f"✅ Successfully published {results['successful_scans']} opportunities to mobile app")
                logger.info(f"📱 Mobile app now has access to {results['summary']['total_opportunities']} opportunities")
                logger.info(f"🎯 High confidence signals: {results['summary']['high_confidence_signals']}")
                
                return {
                    'success': True,
                    'published_opportunities': results['successful_scans'],
                    'summary': results['summary'],
                    'firebase_response': response_data
                }
            else:
                logger.error(f"❌ Firebase publish failed: {response.status_code} - {response.text}")
                return {
                    'success': False,
                    'error': f"Firebase error: {response.status_code}",
                    'response': response.text
                }
                
        except Exception as e:
            logger.error(f"❌ Publishing failed: {e}")
            return {'success': False, 'error': str(e)}
    
    def publish_top_opportunities(self, limit: int = 10) -> dict:
        """Publish only the top opportunities for mobile home screen"""
        try:
            # Run scanner with popular tickers
            popular_tickers = [
                'AAPL', 'MSFT', 'GOOGL', 'TSLA', 'NVDA', 'AMZN', 'META', 'NFLX', 'AMD', 'INTC',
                'JPM', 'BAC', 'WFC', 'SPY', 'QQQ', 'XLK', 'XLF', 'XLE', 'XLV', 'XLI'
            ]
            
            logger.info(f"🎯 Publishing top {limit} opportunities to mobile home screen...")
            
            results = self.scanner.scan_production(popular_tickers)
            
            if results['successful_scans'] == 0:
                return {'success': False, 'error': 'No successful scans'}
            
            # Sort by confidence and take top opportunities
            opportunities = sorted(results['opportunities'], key=lambda x: x['confidence'], reverse=True)
            top_opportunities = opportunities[:limit]
            
            # Prepare data for Firebase
            firebase_data = {
                'opportunities': top_opportunities,
                'scan_timestamp': results['scan_timestamp'],
                'total_tickers': results['total_tickers'],
                'successful_scans': len(top_opportunities),
                'failed_scans': results['failed_scans'],
                'summary': {
                    **results['summary'],
                    'top_opportunities_only': True,
                    'limit': limit
                }
            }
            
            # Publish to Firebase
            response = requests.post(
                f"{self.firebase_api_url}/publish-scanner-data",
                json=firebase_data,
                headers={'Content-Type': 'application/json'},
                timeout=30
            )
            
            if response.status_code == 200:
                response_data = response.json()
                logger.info(f"✅ Published top {len(top_opportunities)} opportunities to mobile home screen")
                
                # Log top opportunities
                for i, opp in enumerate(top_opportunities[:5], 1):
                    target = opp['targets_stops']['target'] if opp['targets_stops']['target'] else 0
                    stop = opp['targets_stops']['stop_loss'] if opp['targets_stops']['stop_loss'] else 0
                    logger.info(f"  {i}. {opp['ticker']} | {opp['signal']} | RSI: {opp['technical_indicators']['rsi']:.1f} | "
                              f"Conf: {opp['confidence']}% | Target: ${target:.2f} | Stop: ${stop:.2f}")
                
                return {
                    'success': True,
                    'published_opportunities': len(top_opportunities),
                    'top_opportunities': top_opportunities,
                    'firebase_response': response_data
                }
            else:
                logger.error(f"❌ Firebase publish failed: {response.status_code} - {response.text}")
                return {'success': False, 'error': f"Firebase error: {response.status_code}"}
                
        except Exception as e:
            logger.error(f"❌ Publishing top opportunities failed: {e}")
            return {'success': False, 'error': str(e)}

def main():
    """Test the scanner publisher"""
    # Firebase API URL
    firebase_api_url = "https://us-central1-kardova-capital.cloudfunctions.net/api"
    
    publisher = ScannerPublisher(firebase_api_url)
    
    print("🚀 Testing Scanner Publisher...")
    print("=" * 50)
    
    # Test publishing top opportunities
    result = publisher.publish_top_opportunities(limit=5)
    
    if result['success']:
        print(f"\n✅ SUCCESS!")
        print(f"Published: {result['published_opportunities']} opportunities")
        print(f"High confidence signals available on mobile home screen!")
    else:
        print(f"\n❌ FAILED: {result['error']}")

if __name__ == "__main__":
    main()
