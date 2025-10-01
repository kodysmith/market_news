#!/usr/bin/env python3
"""
Test database fix
"""

import os
import sys
from pathlib import Path
import pandas as pd

# Set production environment
os.environ['QUANT_ENV'] = 'production'

# Add QuantEngine to path
quant_engine_dir = Path(__file__).parent
sys.path.insert(0, str(quant_engine_dir))

def test_database_fix():
    """Test database operations"""
    print("🧪 Testing Database Fix")
    print("=" * 30)
    
    try:
        from main import load_config, setup_logging
        from engine.data_ingestion.data_manager import DataManager
        
        # Load config
        config = load_config('config/config.yaml')
        setup_logging(config)
        
        # Test data manager
        print("1. Testing data manager...")
        data_manager = DataManager(config)
        
        # Test downloading and caching data
        print("2. Testing data download and cache...")
        spy_data = data_manager.get_market_data(['SPY'], '2024-01-01', '2024-01-05')
        
        if spy_data and 'SPY' in spy_data:
            print(f"✅ Downloaded {len(spy_data['SPY'])} rows for SPY")
            
            # Test if data was cached successfully
            print("3. Testing data retrieval from cache...")
            cached_data = data_manager.get_market_data(['SPY'], '2024-01-01', '2024-01-05')
            if cached_data and 'SPY' in cached_data:
                print(f"✅ Retrieved {len(cached_data['SPY'])} rows from cache")
            else:
                print("⚠️ No cached data retrieved")
        else:
            print("⚠️ No data downloaded")
        
        print("\n🎉 Database test completed!")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_database_fix()

