#!/usr/bin/env python3
"""
Trading Engine Launcher
Run this from the HedgeFund directory
"""

import sys
import os

# Add the project root to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Now import and run
from src.main.trading_engine import TradingEngine
from src.common.config import load_config
from datetime import datetime

if __name__ == "__main__":
    print("🚀 Starting Hedge Fund Trading System...\n")
    
    # Load configuration
    config = load_config()
    
    print(f"Environment: {config.environment}")
    print(f"Initial Capital: ${config.initial_capital:,.0f}")
    print(f"Assets: {config.strategy.assets}\n")
    
    # Initialize trading engine
    engine = TradingEngine(config)
    
    # Quick health check
    print("Running health check...")
    health = engine.health_check()
    
    if not health['overall']:
        print("⚠️  Warning: Some systems not healthy:")
        for service, status in health.items():
            if not status and service != 'overall':
                print(f"  - {service}: ❌")
        print()
    else:
        print("✅ All systems healthy!\n")
    
    # Run a test cycle
    print("="*60)
    print("Running test trading cycle...")
    print("="*60)
    print()
    
    results = engine.run_daily_cycle(datetime.now())
    
    # Print results
    print()
    print("="*60)
    print("📊 CYCLE RESULTS")
    print("="*60)
    print(f"Status: {results['status']}")
    print(f"Signals Generated: {results['signals_generated']}")
    print(f"Trades Executed: {results['trades_executed']}")
    print(f"NAV: ${results['nav']:,.2f}")
    print("="*60)
    
    # Show final status
    status = engine.get_status()
    print()
    print("📈 SYSTEM STATUS")
    print(f"  Mode: {status['mode']}")
    print(f"  NAV: ${status['nav']:,.2f}")
    print(f"  Positions: {status['positions']}")
    print(f"  Circuit Breaker: {status['circuit_breaker']}")
    print()
    
    print("✅ Test cycle complete!")
    print()
    print("💡 To run continuously during market hours:")
    print("   engine.run_live()")
    print()



