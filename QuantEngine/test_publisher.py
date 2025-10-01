#!/usr/bin/env python3
"""
Test the opportunity publisher
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from scheduled_opportunity_publisher import OpportunityPublisher

def test_publisher():
    """Test the opportunity publisher"""
    print("🧪 Testing Opportunity Publisher...")
    
    publisher = OpportunityPublisher()
    
    try:
        # Run a test scan and publish
        publisher.run_scan_and_publish()
        print("✅ Test completed successfully!")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
    finally:
        publisher.close()

if __name__ == '__main__':
    test_publisher()
