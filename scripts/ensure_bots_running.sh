#!/bin/bash
# Ensure IBKR TWS and trading bots are running.
# Intended to be called by cron every 5 minutes during market hours.

LOG="/tmp/ensure_bots.log"
BOT_DIR="/Users/kody/base/MarketNews"
VENV="$BOT_DIR/venv/bin/activate"

log() {
    echo "$(date '+%Y-%m-%d %H:%M:%S')  $1" >> "$LOG"
}

# --- 0. Prevent Mac from sleeping during market hours ---
if ! pgrep -f "caffeinate" > /dev/null 2>&1; then
    caffeinate -d -t 28800 &  # Keep awake for 8 hours
    log "Started caffeinate (prevent sleep for 8h)"
fi

# --- 1. Ensure IBKR TWS is running ---
if ! pgrep -f "jts.jar\|ibgateway\|tws.jar\|Trader Workstation" > /dev/null 2>&1; then
    log "TWS not running — launching..."
    open "$HOME/Applications/Trader Workstation/Trader Workstation.app" &
    log "TWS launch initiated. Waiting 60s for startup..."
    sleep 60
else
    log "TWS is running."
fi

# --- 2. Ensure main IC bot is running ---
if ! pgrep -f "bot/main.py.*--mode paper" > /dev/null 2>&1; then
    log "Main IC bot not running — starting..."
    cd "$BOT_DIR"
    source "$VENV"
    nohup python bot/main.py --mode paper >> /tmp/bot_paper.log 2>&1 &
    log "Main IC bot started (PID: $!)"
else
    log "Main IC bot is running (PID: $(pgrep -f 'bot/main.py.*--mode paper'))"
fi

# --- 3. Ensure daily cashflow bot is running ---
if ! pgrep -f "bot/daily_cashflow.py.*--mode paper" > /dev/null 2>&1; then
    log "Daily cashflow bot not running — starting..."
    cd "$BOT_DIR"
    source "$VENV"
    nohup python bot/daily_cashflow.py --mode paper >> /tmp/bot_daily_cashflow.log 2>&1 &
    log "Daily cashflow bot started (PID: $!)"
else
    log "Daily cashflow bot is running (PID: $(pgrep -f 'bot/daily_cashflow.py.*--mode paper'))"
fi

# --- 4. Log summary ---
log "Check complete. Bots: main=$(pgrep -f 'bot/main.py' | head -1)  daily=$(pgrep -f 'bot/daily_cashflow.py' | head -1)  TWS=$(pgrep -f 'jts.jar\|Trader Workstation' | head -1)"
