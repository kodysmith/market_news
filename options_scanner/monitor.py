#!/usr/bin/env python3
"""
Monitor the options scanner status
"""

import requests
import json
import time
from datetime import datetime

def check_scanner_status():
    """Check if scanner is running and serving data"""
    try:
        response = requests.get('http://localhost:8081/report.json', timeout=5)
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Scanner is running!")
            print(f"📅 Last updated: {data['asOf']}")
            print(f"📊 Total ideas: {len(data['topIdeas'])}")
            print(f"🔔 Alerts sent: {len(data['alertsSentThisRun'])}")
            
            if data['topIdeas']:
                idea = data['topIdeas'][0]
                print(f"🎯 Best opportunity: {idea['ticker']} {idea['shortK']}/{idea['longK']} - EV: ${idea['ev']:.2f}")
            
            return True
        else:
            print(f"❌ Scanner returned status {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Scanner not accessible: {e}")
        return False

def main():
    """Monitor scanner status"""
    print("🔍 Options Scanner Monitor")
    print("=" * 30)
    
    while True:
        print(f"\n⏰ {datetime.now().strftime('%H:%M:%S')} - Checking scanner...")
        check_scanner_status()
        time.sleep(30)  # Check every 30 seconds

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n👋 Monitor stopped")

