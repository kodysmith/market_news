#!/usr/bin/env python3
"""
Test configuration for local development
"""

import os

# Set test environment variables
os.environ['ALPHAVANTAGE_API_KEY'] = 'your_alpha_vantage_api_key_here'
os.environ['FIREBASE_SERVICE_ACCOUNT_JSON_BASE64'] = 'dGVzdA=='  # "test" in base64
os.environ['PUBLIC_BUCKET_URL'] = 'https://api-hvi4gdtdka-uc.a.run.app'
os.environ['REPORT_OBJECT_PATH'] = '/report.json'
os.environ['TICKERS'] = 'SPY'
os.environ['SCAN_INTERVAL_MINUTES'] = '5'
os.environ['MARKET_TZ'] = 'America/New_York'

print("✅ Test environment configured")

