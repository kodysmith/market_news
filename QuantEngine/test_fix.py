#!/usr/bin/env python3
"""Test script to verify the LiveDataManager fix"""

import sys
from pathlib import Path

# Add QuantEngine root to path
quant_engine_root = Path(__file__).parent
if str(quant_engine_root) not in sys.path:
    sys.path.insert(0, str(quant_engine_root))

def test_import():
    """Test that LiveDataManager can be imported and has the method"""
    try:
        from engine.data_ingestion.live_data_manager import LiveDataManager
        print("✅ LiveDataManager imported successfully")

        # Create instance
        config = {}  # Minimal config for testing
        manager = LiveDataManager(config)
        print("✅ LiveDataManager instance created")

        # Check method exists
        if hasattr(manager, 'update_real_time_data'):
            print("✅ update_real_time_data method exists!")
            return True
        else:
            print("❌ update_real_time_data method missing!")
            return False

    except Exception as e:
        print(f"❌ Import failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_quantbot_import():
    """Test that QuantBot can import everything"""
    try:
        from quant_bot import QuantBot
        print("✅ QuantBot imported successfully")

        # Pass config file path instead of loaded config
        bot = QuantBot('config/config.yaml')
        print("✅ QuantBot instance created")

        # Initialize components to create data_manager
        import asyncio
        asyncio.run(bot.initialize_components())
        print("✅ QuantBot components initialized")

        # Check if data_manager has the method
        if hasattr(bot.data_manager, 'update_real_time_data'):
            print("✅ QuantBot data_manager has update_real_time_data method!")
            return True
        else:
            print("❌ QuantBot data_manager missing update_real_time_data method!")
            return False

    except Exception as e:
        print(f"❌ QuantBot import failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("🧪 Testing LiveDataManager fix...")
    print()

    print("1. Testing LiveDataManager import...")
    test1_passed = test_import()
    print()

    print("2. Testing QuantBot import...")
    test2_passed = test_quantbot_import()
    print()

    if test1_passed and test2_passed:
        print("🎉 All tests passed! The fix should work.")
    else:
        print("❌ Some tests failed. Need to investigate further.")
