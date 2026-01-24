#!/usr/bin/env python3
"""
Test script to verify chat functionality fixes
"""

import asyncio
import sys
from pathlib import Path

# Add QuantEngine to path
quant_engine_root = Path(__file__).parent / 'QuantEngine'
if str(quant_engine_root) not in sys.path:
    sys.path.insert(0, str(quant_engine_root))

async def test_sector_chat():
    """Test sector research chat"""
    print("🧪 Testing Sector Research Chat...")
    
    try:
        from sector_research_chat import SectorResearchChat
        
        chat = SectorResearchChat()
        
        # Test simple question
        result = await chat.ask_question("hi")
        print(f"✅ Simple question result: {result.get('response', 'No response')}")
        
        # Test sector question
        result = await chat.ask_question("research technology sector")
        print(f"✅ Sector question result: {result.get('response', 'No response')}")
        
        # Test error handling
        result = await chat.ask_question("analyze xyz sector")
        print(f"✅ Error handling result: {result.get('response', 'No response')}")
        
        return True
        
    except Exception as e:
        print(f"❌ Sector chat test failed: {e}")
        return False

async def test_group_chat():
    """Test group analysis chat"""
    print("\n🧪 Testing Group Analysis Chat...")
    
    try:
        from group_analysis_chat import GroupAnalysisChat
        
        chat = GroupAnalysisChat()
        
        # Test simple question
        result = await chat.ask_question("hi")
        print(f"✅ Simple question result: {result.get('response', 'No response')}")
        
        # Test group question
        result = await chat.ask_question("analyze mega cap tech")
        print(f"✅ Group question result: {result.get('response', 'No response')}")
        
        # Test error handling
        result = await chat.ask_question("analyze xyz group")
        print(f"✅ Error handling result: {result.get('response', 'No response')}")
        
        return True
        
    except Exception as e:
        print(f"❌ Group chat test failed: {e}")
        return False

async def main():
    """Run all tests"""
    print("🚀 Testing Chat Functionality Fixes")
    print("=" * 50)
    
    sector_success = await test_sector_chat()
    group_success = await test_group_chat()
    
    print("\n" + "=" * 50)
    print("📊 Test Results:")
    print(f"  Sector Chat: {'✅ PASS' if sector_success else '❌ FAIL'}")
    print(f"  Group Chat: {'✅ PASS' if group_success else '❌ FAIL'}")
    
    if sector_success and group_success:
        print("\n🎉 All chat tests passed! The UI should now work properly.")
    else:
        print("\n⚠️ Some tests failed. Check the errors above.")

if __name__ == "__main__":
    asyncio.run(main())



