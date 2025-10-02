#!/usr/bin/env python3
"""
Test script to publish sample scanner data to mobile app
"""

import requests
import json
from datetime import datetime

def test_publish_scanner_data():
    """Test publishing sample scanner data to Firebase"""
    
    # Sample scanner data
    sample_opportunities = [
        {
            "ticker": "AAPL",
            "timestamp": datetime.now().isoformat(),
            "current_price": 254.63,
            "signal": "WEAK_SELL",
            "confidence": 90,
            "technical_indicators": {
                "rsi": 68.8,
                "macd": -1.2,
                "sma_20": 250.0,
                "sma_50": 245.0
            },
            "targets_stops": {
                "target": 246.08,
                "stop_loss": 262.48,
                "risk_reward": 1.33
            },
            "fundamentals": {
                "fundamental_score": 45,
                "pe_ratio": 28.5,
                "sector": "Technology"
            },
            "data_quality": 1.0
        },
        {
            "ticker": "TSLA",
            "timestamp": datetime.now().isoformat(),
            "current_price": 444.72,
            "signal": "SELL",
            "confidence": 80,
            "technical_indicators": {
                "rsi": 73.7,
                "macd": -2.1,
                "sma_20": 430.0,
                "sma_50": 420.0
            },
            "targets_stops": {
                "target": 427.16,
                "stop_loss": 483.68,
                "risk_reward": 1.33
            },
            "fundamentals": {
                "fundamental_score": 15,
                "pe_ratio": 45.2,
                "sector": "Consumer Discretionary"
            },
            "data_quality": 1.0
        },
        {
            "ticker": "GOOGL",
            "timestamp": datetime.now().isoformat(),
            "current_price": 233.57,
            "signal": "WEAK_SELL",
            "confidence": 74,
            "technical_indicators": {
                "rsi": 64.0,
                "macd": -0.8,
                "sma_20": 240.0,
                "sma_50": 235.0
            },
            "targets_stops": {
                "target": 233.57,
                "stop_loss": 253.40,
                "risk_reward": 1.33
            },
            "fundamentals": {
                "fundamental_score": 70,
                "pe_ratio": 22.1,
                "sector": "Communication Services"
            },
            "data_quality": 1.0
        }
    ]
    
    # Prepare Firebase data for existing endpoint
    firebase_data = {
        "opportunities": sample_opportunities,
        "scan_timestamp": datetime.now().isoformat(),
        "total_count": 3
    }
    
    # Use existing opportunities endpoint
    firebase_url = "https://us-central1-kardova-capital.cloudfunctions.net/api/publish-opportunities"
    
    try:
        print("🚀 Publishing sample scanner data to mobile app...")
        
        response = requests.post(
            firebase_url,
            json=firebase_data,
            headers={'Content-Type': 'application/json'},
            timeout=30
        )
        
        if response.status_code == 200:
            result = response.json()
            print("✅ Successfully published scanner data!")
            print(f"📱 Published {len(sample_opportunities)} opportunities")
            print(f"🎯 High confidence signals: {firebase_data['summary']['high_confidence_signals']}")
            print(f"📊 Average confidence: {firebase_data['summary']['avg_confidence']}%")
            return True
        else:
            print(f"❌ Failed to publish: {response.status_code}")
            print(f"Response: {response.text}")
            return False
            
    except Exception as e:
        print(f"❌ Error publishing: {e}")
        return False

def test_fetch_scanner_data():
    """Test fetching scanner data from Firebase"""
    
    firebase_url = "https://us-central1-kardova-capital.cloudfunctions.net/api/scanner-opportunities"
    
    try:
        print("\n📱 Testing fetch scanner data from mobile app...")
        
        response = requests.get(
            firebase_url,
            headers={'Content-Type': 'application/json'},
            timeout=30
        )
        
        if response.status_code == 200:
            data = response.json()
            opportunities = data.get('opportunities', [])
            summary = data.get('summary', {})
            
            print("✅ Successfully fetched scanner data!")
            print(f"📊 Found {len(opportunities)} opportunities")
            
            if summary:
                print(f"🎯 High confidence signals: {summary.get('summary', {}).get('high_confidence_signals', 0)}")
                print(f"📈 Average confidence: {summary.get('summary', {}).get('avg_confidence', 0)}%")
            
            # Show top opportunities
            for i, opp in enumerate(opportunities[:3], 1):
                ticker = opp.get('ticker', 'N/A')
                signal = opp.get('signal', 'HOLD')
                confidence = opp.get('confidence', 0)
                rsi = opp.get('technical_indicators', {}).get('rsi', 50)
                price = opp.get('current_price', 0)
                
                print(f"  {i}. {ticker} | {signal} | RSI: {rsi:.1f} | Conf: {confidence}% | Price: ${price:.2f}")
            
            return True
        else:
            print(f"❌ Failed to fetch: {response.status_code}")
            print(f"Response: {response.text}")
            return False
            
    except Exception as e:
        print(f"❌ Error fetching: {e}")
        return False

if __name__ == "__main__":
    print("🧪 Testing Mobile App Scanner Integration")
    print("=" * 50)
    
    # Test publishing
    publish_success = test_publish_scanner_data()
    
    if publish_success:
        # Test fetching
        fetch_success = test_fetch_scanner_data()
        
        if fetch_success:
            print("\n🎉 SUCCESS! Mobile app integration is working!")
            print("📱 Scanner opportunities should now appear on the home screen")
        else:
            print("\n⚠️ Publishing worked but fetching failed")
    else:
        print("\n❌ Publishing failed - check Firebase functions")
