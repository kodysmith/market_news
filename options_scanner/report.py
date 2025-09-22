"""
Report generation and publishing
"""

import json
import requests
from typing import List, Dict, Any
from datetime import datetime, timezone
from spreads import BullPutSpread

class ReportGenerator:
    """Generate and publish options scanner reports"""
    
    def __init__(self, public_bucket_url: str, report_object_path: str = "/report.json"):
        """
        Initialize report generator
        
        Args:
            public_bucket_url: Base URL for report publishing
            report_object_path: Path to report JSON file
        """
        self.public_bucket_url = public_bucket_url.rstrip('/')
        self.report_object_path = report_object_path.lstrip('/')
        self.full_url = f"{self.public_bucket_url}/{self.report_object_path}"
    
    def generate_report(self, top_ideas: List[BullPutSpread], 
                       universe: List[str], 
                       dte_window: List[int],
                       thresholds: Dict[str, float],
                       alerts_sent: List[str]) -> Dict[str, Any]:
        """
        Generate report JSON structure
        
        Args:
            top_ideas: List of top spread ideas
            universe: List of tickers scanned
            dte_window: [min_dte, max_dte] range
            thresholds: Alert thresholds used
            alerts_sent: List of alert IDs sent this run
        
        Returns:
            Report dictionary
        """
        # Convert spreads to dictionaries
        ideas_data = []
        for spread in top_ideas:
            ideas_data.append({
                'ticker': spread.ticker,
                'strategy': 'BULL_PUT',
                'expiry': spread.expiry,
                'shortK': spread.short_strike,
                'longK': spread.long_strike,
                'width': spread.width,
                'credit': spread.credit,
                'maxLoss': spread.max_loss,
                'dte': spread.dte,
                'pop': spread.pop,
                'ev': spread.ev,
                'ivShort': spread.short_iv,
                'ivLong': spread.long_iv,
                'bidAskW': spread.bid_ask_width,
                'oiShort': spread.oi_short,
                'oiLong': spread.oi_long,
                'volShort': spread.vol_short,
                'volLong': spread.vol_long,
                'fillScore': spread.fill_score,
                'id': spread.spread_id
            })
        
        report = {
            'asOf': datetime.now(timezone.utc).isoformat(),
            'scanner': {
                'universe': universe,
                'dteWindow': dte_window,
                'thresholds': thresholds
            },
            'topIdeas': ideas_data,
            'alertsSentThisRun': alerts_sent
        }
        
        return report
    
    def publish_report(self, report: Dict[str, Any]) -> bool:
        """
        Publish report to public URL
        
        Args:
            report: Report dictionary to publish
        
        Returns:
            True if published successfully, False otherwise
        """
        try:
            # Convert to JSON
            json_data = json.dumps(report, indent=2)
            
            # For now, we'll just save to local file
            # In production, this would upload to Cloud Storage or similar
            with open('report.json', 'w') as f:
                f.write(json_data)
            
            print(f"✅ Report saved to report.json")
            print(f"📊 Report contains {len(report['topIdeas'])} ideas")
            print(f"🔔 {len(report['alertsSentThisRun'])} alerts sent this run")
            
            return True
            
        except Exception as e:
            print(f"❌ Failed to publish report: {e}")
            return False
    
    def upload_to_cloud_storage(self, report: Dict[str, Any]) -> bool:
        """
        Upload report to cloud storage (placeholder for production)
        
        Args:
            report: Report dictionary to upload
        
        Returns:
            True if uploaded successfully, False otherwise
        """
        try:
            # This would be implemented with actual cloud storage SDK
            # For now, just simulate success
            print(f"📤 Would upload report to {self.full_url}")
            return True
            
        except Exception as e:
            print(f"❌ Failed to upload to cloud storage: {e}")
            return False
    
    def get_report_summary(self, report: Dict[str, Any]) -> Dict[str, Any]:
        """Get summary statistics from report"""
        ideas = report.get('topIdeas', [])
        
        if not ideas:
            return {
                'total_ideas': 0,
                'avg_ev': 0.0,
                'avg_pop': 0.0,
                'best_ev': 0.0,
                'best_pop': 0.0,
                'tickers': []
            }
        
        evs = [idea['ev'] for idea in ideas]
        pops = [idea['pop'] for idea in ideas]
        tickers = list(set(idea['ticker'] for idea in ideas))
        
        return {
            'total_ideas': len(ideas),
            'avg_ev': sum(evs) / len(evs),
            'avg_pop': sum(pops) / len(pops),
            'best_ev': max(evs),
            'best_pop': max(pops),
            'tickers': tickers,
            'alerts_sent': len(report.get('alertsSentThisRun', []))
        }
    
    def validate_report(self, report: Dict[str, Any]) -> bool:
        """Validate report structure"""
        required_fields = ['asOf', 'scanner', 'topIdeas', 'alertsSentThisRun']
        
        for field in required_fields:
            if field not in report:
                print(f"❌ Missing required field: {field}")
                return False
        
        # Validate scanner section
        scanner = report['scanner']
        if not all(field in scanner for field in ['universe', 'dteWindow', 'thresholds']):
            print("❌ Invalid scanner section")
            return False
        
        # Validate ideas structure
        for idea in report['topIdeas']:
            required_idea_fields = [
                'ticker', 'strategy', 'expiry', 'shortK', 'longK', 
                'width', 'credit', 'maxLoss', 'dte', 'pop', 'ev', 'id'
            ]
            
            if not all(field in idea for field in required_idea_fields):
                print(f"❌ Invalid idea structure: {idea}")
                return False
        
        print("✅ Report structure is valid")
        return True

