#!/usr/bin/env python3
"""
Test script for Enhanced Dashboard with AI Chat
Verifies that all components are working correctly
"""

import sys
import os
import asyncio
from pathlib import Path
import yfinance as yf
import json

# Add QuantEngine to path
quant_engine_root = Path(__file__).parent / 'QuantEngine'
if str(quant_engine_root) not in sys.path:
    sys.path.insert(0, str(quant_engine_root))

def test_imports():
    """Test that all required modules can be imported"""
    print("🔍 Testing imports...")
    
    try:
        import streamlit as st
        print("✅ Streamlit imported successfully")
    except ImportError as e:
        print(f"❌ Streamlit import failed: {e}")
        return False
    
    try:
        import plotly.graph_objects as go
        import plotly.express as px
        print("✅ Plotly imported successfully")
    except ImportError as e:
        print(f"❌ Plotly import failed: {e}")
        return False
    
    try:
        import pandas as pd
        import numpy as np
        print("✅ Pandas/NumPy imported successfully")
    except ImportError as e:
        print(f"❌ Pandas/NumPy import failed: {e}")
        return False
    
    try:
        import yfinance as yf
        print("✅ YFinance imported successfully")
    except ImportError as e:
        print(f"❌ YFinance import failed: {e}")
        return False
    
    try:
        from sector_research_chat import SectorResearchChat
        from group_analysis_chat import GroupAnalysisChat
        print("✅ Chat interfaces imported successfully")
    except ImportError as e:
        print(f"⚠️ Chat interfaces import failed: {e}")
        print("   This is expected if QuantEngine dependencies are not installed")
    
    return True

def test_symbol_data():
    """Test fetching data for expanded symbols"""
    print("\n📊 Testing symbol data fetching...")
    
    expanded_symbols = [
        'NVDA', 'NFLX', 'GOOG', 'MSFT', 'FDX', 'JNJ', 'DV', 
        'SPY', 'TQQQ', 'QQQ', 'TSLA', 'HD', 'KO'
    ]
    
    successful_symbols = []
    failed_symbols = []
    
    for symbol in expanded_symbols:
        try:
            ticker = yf.Ticker(symbol)
            hist = ticker.history(period="1d")
            
            if not hist.empty:
                current_price = hist['Close'].iloc[-1]
                successful_symbols.append((symbol, current_price))
                print(f"✅ {symbol}: ${current_price:.2f}")
            else:
                failed_symbols.append(symbol)
                print(f"❌ {symbol}: No data available")
                
        except Exception as e:
            failed_symbols.append(symbol)
            print(f"❌ {symbol}: {e}")
    
    print(f"\n📈 Summary: {len(successful_symbols)}/{len(expanded_symbols)} symbols successful")
    
    if failed_symbols:
        print(f"⚠️ Failed symbols: {', '.join(failed_symbols)}")
    
    return len(successful_symbols) > 0

def test_chat_interfaces():
    """Test chat interface initialization"""
    print("\n🤖 Testing chat interfaces...")
    
    try:
        from sector_research_chat import SectorResearchChat
        from group_analysis_chat import GroupAnalysisChat
        
        # Test sector chat
        sector_chat = SectorResearchChat()
        print("✅ Sector Research Chat initialized")
        
        # Test group chat
        group_chat = GroupAnalysisChat()
        print("✅ Group Analysis Chat initialized")
        
        return True
        
    except Exception as e:
        print(f"❌ Chat interface initialization failed: {e}")
        return False

def test_config_file():
    """Test configuration file"""
    print("\n⚙️ Testing configuration file...")
    
    config_path = Path("data/config.json")
    
    if not config_path.exists():
        print(f"❌ Config file not found: {config_path}")
        return False
    
    try:
        with open(config_path, 'r') as f:
            config = json.load(f)
        
        if 'GAMMA_SCALPING_TICKERS' in config:
            tickers = config['GAMMA_SCALPING_TICKERS']
            print(f"✅ Config loaded: {len(tickers)} tickers configured")
            
            # Check if expanded symbols are included
            expanded_symbols = ['NVDA', 'NFLX', 'GOOG', 'MSFT', 'FDX', 'JNJ', 'DV', 'SPY', 'TQQQ', 'QQQ', 'TSLA', 'HD', 'KO']
            included_symbols = [s for s in expanded_symbols if s in tickers]
            
            print(f"✅ Expanded symbols included: {len(included_symbols)}/{len(expanded_symbols)}")
            print(f"   Included: {', '.join(included_symbols)}")
            
            missing = [s for s in expanded_symbols if s not in tickers]
            if missing:
                print(f"⚠️ Missing: {', '.join(missing)}")
            
            return True
        else:
            print("❌ GAMMA_SCALPING_TICKERS not found in config")
            return False
            
    except Exception as e:
        print(f"❌ Config file error: {e}")
        return False

def test_dashboard_file():
    """Test dashboard file exists and is valid"""
    print("\n📊 Testing dashboard file...")
    
    dashboard_path = Path("QuantEngine/enhanced_dashboard_with_chat.py")
    
    if not dashboard_path.exists():
        print(f"❌ Dashboard file not found: {dashboard_path}")
        return False
    
    try:
        with open(dashboard_path, 'r') as f:
            content = f.read()
        
        # Check for key components
        if 'class EnhancedDashboardWithChat' in content:
            print("✅ Dashboard class found")
        else:
            print("❌ Dashboard class not found")
            return False
        
        if 'create_ai_chat_interface' in content:
            print("✅ AI chat interface found")
        else:
            print("❌ AI chat interface not found")
            return False
        
        if 'expanded_symbols' in content:
            print("✅ Expanded symbols configuration found")
        else:
            print("❌ Expanded symbols configuration not found")
            return False
        
        return True
        
    except Exception as e:
        print(f"❌ Dashboard file error: {e}")
        return False

def main():
    """Run all tests"""
    print("🧪 Enhanced Dashboard Test Suite")
    print("=" * 50)
    
    tests = [
        ("Import Test", test_imports),
        ("Symbol Data Test", test_symbol_data),
        ("Chat Interfaces Test", test_chat_interfaces),
        ("Config File Test", test_config_file),
        ("Dashboard File Test", test_dashboard_file)
    ]
    
    results = []
    
    for test_name, test_func in tests:
        print(f"\n{'='*20} {test_name} {'='*20}")
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"❌ {test_name} failed with exception: {e}")
            results.append((test_name, False))
    
    # Summary
    print("\n" + "="*50)
    print("📋 TEST SUMMARY")
    print("="*50)
    
    passed = 0
    total = len(results)
    
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status} {test_name}")
        if result:
            passed += 1
    
    print(f"\n📊 Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! Dashboard is ready to use.")
        print("\n🚀 To launch the dashboard:")
        print("   python run_enhanced_dashboard.py")
        print("   or")
        print("   streamlit run QuantEngine/enhanced_dashboard_with_chat.py")
    else:
        print("⚠️ Some tests failed. Please check the errors above.")
        print("   The dashboard may still work with limited functionality.")

if __name__ == "__main__":
    main()



