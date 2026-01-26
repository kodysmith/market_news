#!/usr/bin/env python3
"""
Quick system test
Verifies all components can initialize and work together
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

import asyncio
from decimal import Decimal

print("=" * 70)
print("🧪 CRYPTO ARBITRAGE SYSTEM TEST")
print("=" * 70)
print()

# Test 1: Config loading
print("Test 1: Config Loading")
try:
    from src.common.config import load_config
    config = load_config()
    print(f"  ✅ Config loaded: {config.environment}")
    print(f"     Exchanges: {list(config.exchanges.keys())}")
    print(f"     Pairs: {len(config.strategy.trading_pairs)}")
except Exception as e:
    print(f"  ❌ Failed: {e}")
    sys.exit(1)

print()

# Test 2: Database initialization
print("Test 2: Database")
try:
    from src.database.repository import ArbitrageRepository
    repo = ArbitrageRepository(config.database.sqlite_path)
    print(f"  ✅ Database initialized: {config.database.sqlite_path}")
except Exception as e:
    print(f"  ❌ Failed: {e}")
    sys.exit(1)

print()

# Test 3: Exchange Manager
print("Test 3: Exchange Manager (Simulation Mode)")
try:
    from src.exchanges.exchange_manager import ExchangeManager
    
    async def test_exchanges():
        manager = ExchangeManager(config, simulation_mode=True)
        status = await manager.connect_all()
        print(f"  ✅ Exchange manager initialized")
        print(f"     Connected: {sum(1 for v in status.values() if v)}/{len(status)}")
        
        # Test getting prices
        prices = await manager.get_all_prices('BTC/USDT')
        print(f"     Prices available: {len(prices)}")
        for exchange, quote in prices.items():
            if quote:
                print(f"       {exchange}: ${quote.mid}")
        
        await manager.disconnect_all()
        return True
    
    asyncio.run(test_exchanges())
    
except Exception as e:
    print(f"  ❌ Failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print()

# Test 4: Arbitrage Detector
print("Test 4: Arbitrage Detector")
try:
    from src.detection.arbitrage_detector import ArbitrageDetector
    from src.data.price_aggregator import PriceAggregator
    
    async def test_detector():
        manager = ExchangeManager(config, simulation_mode=True)
        await manager.connect_all()
        
        aggregator = PriceAggregator(manager)
        
        exchange_fees = {
            'binance': {'maker_fee': Decimal('0.1'), 'taker_fee': Decimal('0.1')},
            'coinbase': {'maker_fee': Decimal('0.4'), 'taker_fee': Decimal('0.6')},
            'kraken': {'maker_fee': Decimal('0.16'), 'taker_fee': Decimal('0.26')}
        }
        
        detector = ArbitrageDetector(aggregator, config.strategy, exchange_fees)
        print(f"  ✅ Detector initialized")
        
        await manager.disconnect_all()
    
    asyncio.run(test_detector())
    
except Exception as e:
    print(f"  ❌ Failed: {e}")
    sys.exit(1)

print()

# Test 5: Risk Manager
print("Test 5: Risk Manager")
try:
    from src.risk.arbitrage_risk_manager import ArbitrageRiskManager
    
    risk_mgr = ArbitrageRiskManager(config.risk)
    status = risk_mgr.get_risk_status()
    print(f"  ✅ Risk manager initialized")
    print(f"     Can trade: {status['can_trade']}")
    print(f"     Circuit breaker: {'ACTIVE' if status['circuit_breaker_active'] else 'OFF'}")
    
except Exception as e:
    print(f"  ❌ Failed: {e}")
    sys.exit(1)

print()

print("=" * 70)
print("✅ ALL TESTS PASSED!")
print("=" * 70)
print()
print("System is ready to run!")
print()
print("Next steps:")
print("  1. Start daemon:    ./start_arbitrage.sh")
print("  2. Start dashboard: ./start_dashboard.sh")
print("  3. Open browser:    http://localhost:5001")
print()
print("=" * 70)


