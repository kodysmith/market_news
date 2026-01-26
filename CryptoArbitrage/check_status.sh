#!/bin/bash
# Check crypto arbitrage system status

cd "$(dirname "$0")"

echo "================================================"
echo "📊 Crypto Arbitrage System Status"
echo "================================================"
echo ""

# Check if daemon is running
if pgrep -f "arbitrage_daemon.py" > /dev/null; then
    echo "✅ Daemon: RUNNING"
    echo "   PID: $(pgrep -f arbitrage_daemon.py)"
else
    echo "❌ Daemon: STOPPED"
fi

echo ""

# Check if dashboard is running
if pgrep -f "dashboard_server.py" > /dev/null; then
    echo "✅ Dashboard: RUNNING"
    echo "   URL: http://localhost:5001"
else
    echo "❌ Dashboard: STOPPED"
fi

echo ""

# Check database
if [ -f "data/arbitrage.db" ]; then
    echo "✅ Database: EXISTS"
    
    # Get stats
    OPPORTUNITIES=$(sqlite3 data/arbitrage.db "SELECT COUNT(*) FROM opportunities;" 2>/dev/null || echo "0")
    EXECUTIONS=$(sqlite3 data/arbitrage.db "SELECT COUNT(*) FROM executions WHERE is_successful = 1;" 2>/dev/null || echo "0")
    PROFIT=$(sqlite3 data/arbitrage.db "SELECT COALESCE(SUM(realized_profit_usd), 0) FROM executions WHERE is_successful = 1;" 2>/dev/null || echo "0")
    
    echo "   Opportunities detected: $OPPORTUNITIES"
    echo "   Successful executions: $EXECUTIONS"
    echo "   Total profit: \$$PROFIT"
else
    echo "⚠️  Database: NOT CREATED"
    echo "   Run: python test_system.py (will create it)"
fi

echo ""

# Check logs
if [ -f "logs/arbitrage.log" ]; then
    echo "📝 Recent logs (last 5 lines):"
    tail -5 logs/arbitrage.log | sed 's/^/   /'
else
    echo "📝 Logs: No log file yet"
fi

echo ""
echo "================================================"
EOF
cat check_status.sh


