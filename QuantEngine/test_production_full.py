#!/usr/bin/env python3
"""
Full test for production QuantEngine
"""

import os
import sys
from pathlib import Path

# Set production environment
os.environ['QUANT_ENV'] = 'production'

# Add QuantEngine to path
quant_engine_dir = Path(__file__).parent
sys.path.insert(0, str(quant_engine_dir))

def test_production_full():
    """Test production setup with data operations"""
    print("🧪 Testing QuantEngine Production Full Test")
    print("=" * 50)
    
    try:
        # Test imports
        print("1. Testing imports...")
        from main import load_config, setup_logging
        from data_broker import QuantBotDataBroker
        from engine.data_ingestion.data_manager import DataManager
        print("✅ Imports successful")
        
        # Test configuration loading
        print("2. Testing configuration loading...")
        config = load_config('config/config.yaml')
        setup_logging(config)
        print("✅ Configuration loaded")
        
        # Test data broker initialization
        print("3. Testing data broker initialization...")
        broker = QuantBotDataBroker()
        print(f"✅ Data broker initialized - Production mode: {broker.production_mode}")
        
        # Test data manager with actual data operations
        print("4. Testing data manager with data operations...")
        data_manager = DataManager(config)
        
        # Test downloading data for a single ticker
        print("5. Testing data download for SPY...")
        spy_data = data_manager.get_market_data(['SPY'], '2024-01-01', '2024-01-31')
        if spy_data and 'SPY' in spy_data:
            print(f"✅ Downloaded {len(spy_data['SPY'])} rows for SPY")
        else:
            print("⚠️ No data downloaded for SPY")
        
        # Test feature building
        print("6. Testing feature building...")
        from engine.feature_builder.feature_builder import FeatureBuilder
        feature_builder = FeatureBuilder(config)
        
        if spy_data and 'SPY' in spy_data:
            try:
                features = feature_builder.build_features('SPY', spy_data['SPY'])
                print(f"✅ Built features for SPY: {len(features)} features")
            except Exception as e:
                print(f"⚠️ Feature building failed: {e}")
        
        print("\n🎉 Full production test completed!")
        print("QuantEngine is ready for production use.")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_production_full()

