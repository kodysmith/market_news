#!/usr/bin/env python3
"""
Test GPU and Ollama Setup for Enhanced Market Scanner
"""

import sys
import os
import json
from pathlib import Path

def test_python_environment():
    """Test Python environment"""
    print("🐍 Testing Python Environment...")
    print(f"Python Version: {sys.version}")
    print(f"Python Path: {sys.executable}")
    
    # Check required packages
    required_packages = [
        'torch', 'pandas', 'numpy', 'yfinance', 'requests', 
        'ollama', 'transformers', 'accelerate'
    ]
    
    missing_packages = []
    for package in required_packages:
        try:
            __import__(package)
            print(f"✅ {package}")
        except ImportError:
            print(f"❌ {package}")
            missing_packages.append(package)
    
    if missing_packages:
        print(f"\n❌ Missing packages: {', '.join(missing_packages)}")
        print("Run: pip install -r requirements_scanner.txt")
        return False
    
    print("✅ All Python packages available")
    return True

def test_gpu_setup():
    """Test GPU setup"""
    print("\n🎮 Testing GPU Setup...")
    
    try:
        import torch
        print(f"PyTorch Version: {torch.__version__}")
        
        if torch.cuda.is_available():
            print(f"✅ CUDA Available: {torch.version.cuda}")
            print(f"✅ GPU Count: {torch.cuda.device_count()}")
            
            for i in range(torch.cuda.device_count()):
                props = torch.cuda.get_device_properties(i)
                print(f"  GPU {i}: {props.name}")
                print(f"    Memory: {props.total_memory / 1e9:.1f} GB")
                print(f"    Compute Capability: {props.major}.{props.minor}")
            
            # Test GPU computation
            device = torch.device("cuda:0")
            x = torch.randn(1000, 1000, device=device)
            y = torch.randn(1000, 1000, device=device)
            z = torch.mm(x, y)
            print(f"✅ GPU computation test passed: {z.shape}")
            
            return True
        else:
            print("❌ CUDA not available")
            return False
            
    except ImportError:
        print("❌ PyTorch not installed")
        return False
    except Exception as e:
        print(f"❌ GPU test failed: {e}")
        return False

def test_ollama_setup():
    """Test Ollama setup"""
    print("\n🤖 Testing Ollama Setup...")
    
    try:
        import ollama
        print("✅ Ollama Python client available")
        
        # Test connection
        try:
            models = ollama.list()
            print(f"✅ Ollama connection successful")
            print(f"Available models: {[m['name'] for m in models['models']]}")
            
            # Test model generation
            test_prompt = "What is 2+2? Answer briefly."
            response = ollama.generate(
                model='llama3.2:latest',
                prompt=test_prompt,
                options={'temperature': 0.1, 'num_predict': 50}
            )
            
            print(f"✅ Model generation test passed")
            print(f"Response: {response['response'][:100]}...")
            
            return True
            
        except Exception as e:
            print(f"❌ Ollama connection failed: {e}")
            print("Make sure Ollama is running: ollama serve")
            return False
            
    except ImportError:
        print("❌ Ollama Python client not installed")
        return False

def test_market_data():
    """Test market data fetching"""
    print("\n📊 Testing Market Data Fetching...")
    
    try:
        import yfinance as yf
        import pandas as pd
        
        # Test fetching AAPL data
        ticker = yf.Ticker("AAPL")
        info = ticker.info
        hist = ticker.history(period="5d")
        
        if not hist.empty:
            print(f"✅ Yahoo Finance data fetching successful")
            print(f"  AAPL Price: ${info.get('regularMarketPrice', 'N/A')}")
            print(f"  Historical data: {len(hist)} days")
            print(f"  Latest close: ${hist['Close'].iloc[-1]:.2f}")
            return True
        else:
            print("❌ No historical data received")
            return False
            
    except Exception as e:
        print(f"❌ Market data test failed: {e}")
        return False

def test_enhanced_scanner():
    """Test enhanced scanner"""
    print("\n🚀 Testing Enhanced Scanner...")
    
    try:
        # Add QuantEngine to path
        sys.path.append('QuantEngine')
        
        from enhanced_scanner_gpu import EnhancedScannerGPU
        
        # Create scanner
        scanner = EnhancedScannerGPU()
        
        print(f"✅ Enhanced Scanner initialized")
        print(f"  GPU Enabled: {scanner.config['gpu']['enabled']}")
        print(f"  LLM Enabled: {scanner.config['llm']['enabled']}")
        print(f"  Device: {scanner.device}")
        
        return True
        
    except Exception as e:
        print(f"❌ Enhanced Scanner test failed: {e}")
        return False

def create_test_report():
    """Create test report"""
    print("\n📋 Creating Test Report...")
    
    report = {
        "timestamp": str(pd.Timestamp.now()),
        "python_version": sys.version,
        "python_path": sys.executable,
        "tests": {
            "python_environment": test_python_environment(),
            "gpu_setup": test_gpu_setup(),
            "ollama_setup": test_ollama_setup(),
            "market_data": test_market_data(),
            "enhanced_scanner": test_enhanced_scanner()
        }
    }
    
    # Save report
    with open("test_report.json", "w") as f:
        json.dump(report, f, indent=2)
    
    print(f"✅ Test report saved to test_report.json")
    
    # Print summary
    print("\n" + "="*50)
    print("🎯 TEST SUMMARY")
    print("="*50)
    
    all_passed = True
    for test_name, result in report["tests"].items():
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{test_name.replace('_', ' ').title()}: {status}")
        if not result:
            all_passed = False
    
    print("="*50)
    if all_passed:
        print("🎉 ALL TESTS PASSED! System ready for enhanced scanning.")
    else:
        print("⚠️ Some tests failed. Please check the errors above.")
    
    return all_passed

def main():
    """Main test function"""
    print("🧪 Enhanced Market Scanner - GPU/LLM Setup Test")
    print("=" * 60)
    
    # Create necessary directories
    Path("logs").mkdir(exist_ok=True)
    Path("reports").mkdir(exist_ok=True)
    Path("data").mkdir(exist_ok=True)
    
    # Run all tests
    all_passed = create_test_report()
    
    if all_passed:
        print("\n🚀 Next Steps:")
        print("1. Start enhanced scanner: ./start_scanner_gpu.sh")
        print("2. Start LLM analysis: ./start_llm_analysis.sh")
        print("3. Start web dashboard: ./start_dashboard.sh")
        print("4. Check logs: tail -f logs/enhanced_scanner_gpu.log")
    else:
        print("\n🔧 Fix Issues:")
        print("1. Install missing packages: pip install -r requirements_scanner.txt")
        print("2. Start Ollama: ollama serve")
        print("3. Check GPU drivers: nvidia-smi")
        print("4. Re-run test: python test_gpu_setup.py")

if __name__ == "__main__":
    main()
