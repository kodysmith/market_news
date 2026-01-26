#!/bin/bash
# Health Check Script
# Monitors crypto arbitrage service health

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo "========================================"
echo "Crypto Arbitrage Service Health Check"
echo "========================================"
echo ""

cd "$(dirname "$0")"

# Check service status
echo "1. Service Status:"
if sudo systemctl is-active crypto-arbitrage-paper.service > /dev/null 2>&1; then
    echo -e "   ${GREEN}✅ Service is running${NC}"
    UPTIME=$(sudo systemctl show crypto-arbitrage-paper.service --property=ActiveEnterTimestamp --value)
    echo "   Started: $UPTIME"
else
    echo -e "   ${RED}❌ Service is not running!${NC}"
    echo "   Last status:"
    sudo systemctl status crypto-arbitrage-paper.service --no-pager | tail -5
fi
echo ""

# Check if opportunities are being detected
echo "2. Recent Opportunities:"
if [ -f "data/arbitrage.db" ]; then
    COUNT_1H=$(sqlite3 data/arbitrage.db "SELECT COUNT(*) FROM opportunities WHERE detected_at > datetime('now', '-1 hour')" 2>/dev/null)
    COUNT_5M=$(sqlite3 data/arbitrage.db "SELECT COUNT(*) FROM opportunities WHERE detected_at > datetime('now', '-5 minutes')" 2>/dev/null)
    COUNT_TODAY=$(sqlite3 data/arbitrage.db "SELECT COUNT(*) FROM opportunities WHERE DATE(detected_at) = DATE('now')" 2>/dev/null)
    
    echo "   Last 5 minutes: $COUNT_5M opportunities"
    echo "   Last hour: $COUNT_1H opportunities"
    echo "   Today: $COUNT_TODAY opportunities"
    
    if [ "$COUNT_5M" -gt 0 ]; then
        echo -e "   ${GREEN}✅ System is actively detecting opportunities${NC}"
    elif [ "$COUNT_1H" -gt 0 ]; then
        echo -e "   ${YELLOW}⚠️  No opportunities in last 5 minutes (but some in last hour)${NC}"
    else
        echo -e "   ${RED}❌ No opportunities detected recently!${NC}"
    fi
else
    echo -e "   ${RED}⚠️  Database not found${NC}"
fi
echo ""

# Check log file
echo "3. Log Files:"
if [ -f "logs/paper_trading.log" ]; then
    LOG_SIZE=$(du -h logs/paper_trading.log | cut -f1)
    LOG_LINES=$(wc -l < logs/paper_trading.log)
    echo "   Size: $LOG_SIZE ($LOG_LINES lines)"
    
    # Check for recent activity
    LAST_LOG=$(tail -1 logs/paper_trading.log 2>/dev/null | cut -d' ' -f1-2)
    echo "   Last entry: $LAST_LOG"
else
    echo -e "   ${YELLOW}⚠️  Log file not found${NC}"
fi
echo ""

# Check database
echo "4. Database:"
if [ -f "data/arbitrage.db" ]; then
    DB_SIZE=$(du -h data/arbitrage.db | cut -f1)
    echo "   Size: $DB_SIZE"
    
    TOTAL_OPPS=$(sqlite3 data/arbitrage.db "SELECT COUNT(*) FROM opportunities" 2>/dev/null)
    TOTAL_EXECS=$(sqlite3 data/arbitrage.db "SELECT COUNT(*) FROM executions" 2>/dev/null)
    echo "   Total opportunities: $TOTAL_OPPS"
    echo "   Total executions: $TOTAL_EXECS"
else
    echo -e "   ${RED}⚠️  Database not found${NC}"
fi
echo ""

# Check for errors
echo "5. Recent Errors:"
if [ -f "logs/paper_trading.log" ]; then
    ERROR_COUNT=$(grep -c "ERROR\|Exception\|Failed" logs/paper_trading.log 2>/dev/null | tail -1 || echo 0)
    RECENT_ERRORS=$(grep "ERROR\|Exception\|Failed" logs/paper_trading.log 2>/dev/null | tail -5)
    
    if [ -z "$RECENT_ERRORS" ]; then
        echo -e "   ${GREEN}✅ No recent errors${NC}"
    else
        echo -e "   ${YELLOW}⚠️  Found errors (last 5):${NC}"
        echo "$RECENT_ERRORS" | while IFS= read -r line; do
            echo "   $line"
        done
    fi
else
    echo -e "   ${YELLOW}⚠️  Cannot check errors - log not found${NC}"
fi
echo ""

# Resource usage
echo "6. Resource Usage:"
if pgrep -f "paper_trading.py" > /dev/null 2>&1; then
    PID=$(pgrep -f "paper_trading.py")
    echo "   Process ID: $PID"
    
    # Get memory and CPU usage
    MEM=$(ps -p $PID -o rss= 2>/dev/null)
    CPU=$(ps -p $PID -o %cpu= 2>/dev/null)
    
    if [ -n "$MEM" ]; then
        MEM_MB=$((MEM / 1024))
        echo "   Memory: ${MEM_MB} MB"
        echo "   CPU: ${CPU}%"
        
        if [ "$MEM_MB" -gt 800 ]; then
            echo -e "   ${YELLOW}⚠️  High memory usage${NC}"
        fi
    fi
else
    echo -e "   ${YELLOW}⚠️  Process not found${NC}"
fi
echo ""

# Network connectivity
echo "7. Exchange Connectivity:"
for EXCHANGE in "api.binance.com" "api.pro.coinbase.com" "api.kraken.com"; do
    if ping -c 1 -W 2 "$EXCHANGE" > /dev/null 2>&1; then
        echo -e "   ${GREEN}✅${NC} $EXCHANGE reachable"
    else
        echo -e "   ${RED}❌${NC} $EXCHANGE unreachable!"
    fi
done
echo ""

# Overall health score
echo "========================================"
echo "Overall Health Assessment:"
echo "========================================"

HEALTH_SCORE=0
HEALTH_ISSUES=()

# Check service running
if sudo systemctl is-active crypto-arbitrage-paper.service > /dev/null 2>&1; then
    ((HEALTH_SCORE += 30))
else
    HEALTH_ISSUES+=("Service not running")
fi

# Check opportunities
if [ "$COUNT_5M" -gt 0 ] 2>/dev/null; then
    ((HEALTH_SCORE += 30))
elif [ "$COUNT_1H" -gt 0 ] 2>/dev/null; then
    ((HEALTH_SCORE += 15))
    HEALTH_ISSUES+=("No recent opportunities")
else
    HEALTH_ISSUES+=("No opportunities detected")
fi

# Check database
if [ -f "data/arbitrage.db" ] && [ -s "data/arbitrage.db" ]; then
    ((HEALTH_SCORE += 20))
else
    HEALTH_ISSUES+=("Database issue")
fi

# Check logs
if [ -f "logs/paper_trading.log" ] && [ -s "logs/paper_trading.log" ]; then
    ((HEALTH_SCORE += 10))
else
    HEALTH_ISSUES+=("Log file issue")
fi

# Check for errors
if [ -z "$RECENT_ERRORS" ] 2>/dev/null; then
    ((HEALTH_SCORE += 10))
else
    HEALTH_ISSUES+=("Recent errors found")
fi

# Display score
if [ $HEALTH_SCORE -ge 80 ]; then
    echo -e "${GREEN}✅ HEALTHY${NC} (Score: $HEALTH_SCORE/100)"
    echo "System is operating normally."
elif [ $HEALTH_SCORE -ge 50 ]; then
    echo -e "${YELLOW}⚠️  WARNING${NC} (Score: $HEALTH_SCORE/100)"
    echo "System has some issues:"
    for issue in "${HEALTH_ISSUES[@]}"; do
        echo "  - $issue"
    done
else
    echo -e "${RED}❌ CRITICAL${NC} (Score: $HEALTH_SCORE/100)"
    echo "System has critical issues:"
    for issue in "${HEALTH_ISSUES[@]}"; do
        echo "  - $issue"
    done
    echo ""
    echo "Recommended actions:"
    echo "  1. Check service logs: sudo journalctl -u crypto-arbitrage-paper.service -n 50"
    echo "  2. Restart service: sudo systemctl restart crypto-arbitrage-paper.service"
    echo "  3. Check API keys in .env"
    echo "  4. Verify network connectivity"
fi

echo ""
echo "========================================"
echo "Health check complete at $(date)"
echo "========================================"


