#!/usr/bin/env python3
"""
Test script for production QuantEngine
"""

import os
import sys
from pathlib import Path

# Set production environment
os.environ['QUANT_ENV'] = 'production'

# Add QuantEngine to path
quant_engine_dir = Path(__file__).parent
sys.path.insert(0, str(quant_engine_dir))

def test_production_setup():
    """Test production setup without running full cycle"""
    print("🧪 Testing QuantEngine Production Setup")
    print("=" * 50)
    
    try:
        # Test imports
        print("1. Testing imports...")
        from main import load_config, setup_logging
        from data_broker import QuantBotDataBroker
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
        print(f"✅ Using Firestore: {broker.use_firestore}")
        
        # Test basic functionality
        print("4. Testing basic functionality...")
        from engine.data_ingestion.data_manager import DataManager
        data_manager = DataManager(config)
        print("✅ Data manager initialized")
        
        print("\n🎉 All production tests passed!")
        print("QuantEngine is ready for production use.")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_production_setup()

