#!/bin/bash
# Network Monitor Script
# Monitors connectivity to exchanges and restarts service if needed

cd "$(dirname "$0")"

EXCHANGES=("api.binance.com" "api.pro.coinbase.com" "api.kraken.com")
FAILED=0
LOG_FILE="logs/network_monitor.log"

# Create log directory if needed
mkdir -p logs

for EXCHANGE in "${EXCHANGES[@]}"; do
    # Ping with 2 second timeout
    ping -c 1 -W 2 $EXCHANGE > /dev/null 2>&1
    
    if [ $? -ne 0 ]; then
        echo "$(date '+%Y-%m-%d %H:%M:%S'): Cannot reach $EXCHANGE" >> $LOG_FILE
        FAILED=$((FAILED + 1))
    fi
done

# If 2 or more exchanges are unreachable, restart service
if [ $FAILED -ge 2 ]; then
    echo "$(date '+%Y-%m-%d %H:%M:%S'): ALERT - $FAILED exchanges unreachable. Restarting service..." >> $LOG_FILE
    
    # Check if service is running
    if sudo systemctl is-active crypto-arbitrage-paper.service > /dev/null 2>&1; then
        sudo systemctl restart crypto-arbitrage-paper.service
        echo "$(date '+%Y-%m-%d %H:%M:%S'): Service restarted" >> $LOG_FILE
    else
        echo "$(date '+%Y-%m-%d %H:%M:%S'): Service was not running, starting..." >> $LOG_FILE
        sudo systemctl start crypto-arbitrage-paper.service
    fi
fi

# Clean old log entries (keep last 1000 lines)
if [ -f "$LOG_FILE" ]; then
    tail -1000 "$LOG_FILE" > "$LOG_FILE.tmp"
    mv "$LOG_FILE.tmp" "$LOG_FILE"
fi


