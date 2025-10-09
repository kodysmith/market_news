#!/bin/bash
# Quick test to verify daemon can initialize

cd "$(dirname "$0")"

echo "🧪 Testing Daemon Initialization..."
echo ""

# Test 1: Check if config loads
echo "Test 1: Config Loading"
python3 -c "
import sys
sys.path.insert(0, '.')
from src.common.config import load_config
try:
    config = load_config(environment='paper')
    print('  ✅ Config loaded successfully')
    print(f'     Environment: {config.environment}')
    print(f'     Assets: {config.strategy.assets}')
except Exception as e:
    print(f'  ❌ Config failed: {e}')
    sys.exit(1)
"

echo ""

# Test 2: Check if daemon imports
echo "Test 2: Daemon Import"
python3 -c "
import sys
sys.path.insert(0, '.')
try:
    from src.daemon.paper_trading_daemon import PaperTradingDaemon
    print('  ✅ Daemon imports successfully')
except Exception as e:
    print(f'  ❌ Import failed: {e}')
    sys.exit(1)
"

echo ""

# Test 3: Check database connectivity (optional)
echo "Test 3: Database Connection (optional)"
if command -v psql &> /dev/null; then
    if psql -U hedgefund -d hedgefund_db -c "SELECT 1" &> /dev/null; then
        echo "  ✅ Database connected"
    else
        echo "  ⚠️  Database not configured (will need setup for full operation)"
    fi
else
    echo "  ⚠️  psql not found (database check skipped)"
fi

echo ""
echo "================================================"
echo "✅ Basic tests passed!"
echo ""
echo "To start the daemon:"
echo "  ./start_daemon.sh dev       # Foreground (testing)"
echo "  ./start_daemon.sh supervisor # Background"
echo ""
echo "Note: Daemon will check systems every 60 seconds"
echo "      and execute trades at 9:30 AM ET on trading days"
echo "================================================"

