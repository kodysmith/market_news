#!/bin/bash
# Ensure IBKR TWS and trading bots are running.
# Called by cron every 5 minutes during market hours.
# Crash-resilient: checks process health, not just existence.

LOG="/tmp/ensure_bots.log"
BOT_DIR="/Users/kody/base/MarketNews"
VENV="$BOT_DIR/venv/bin/activate"

log() {
    echo "$(date '+%Y-%m-%d %H:%M:%S')  $1" >> "$LOG"
}

is_alive() {
    # Check if a PID is alive and not a zombie
    local pid=$1
    [ -n "$pid" ] && kill -0 "$pid" 2>/dev/null
}

# --- 0. Prevent Mac from sleeping during market hours ---
if ! pgrep -f "caffeinate" > /dev/null 2>&1; then
    caffeinate -d -t 28800 &
    log "Started caffeinate"
fi

# --- 1. Ensure IBKR TWS is running ---
TWS_PID=$(pgrep -f "jts.jar\|ibgateway\|tws.jar\|Trader Workstation" | head -1)
if [ -z "$TWS_PID" ] || ! is_alive "$TWS_PID"; then
    log "TWS not running — launching..."
    open "$HOME/Applications/Trader Workstation/Trader Workstation.app" &
    log "TWS launch initiated. Waiting 45s..."
    sleep 45
else
    log "TWS running (PID: $TWS_PID)"
fi

# --- 2. Ensure main IC bot is running ---
MAIN_PID=$(pgrep -f "bot/main.py.*--mode paper" | head -1)
if [ -z "$MAIN_PID" ] || ! is_alive "$MAIN_PID"; then
    # Kill any zombie/orphaned processes
    pkill -f "bot/main.py.*--mode paper" 2>/dev/null
    sleep 1
    log "Main IC bot not running — starting..."
    cd "$BOT_DIR"
    source "$VENV"
    nohup python bot/main.py --mode paper >> /tmp/bot_paper.log 2>&1 &
    log "Main IC bot started (PID: $!)"
else
    log "Main IC bot running (PID: $MAIN_PID)"
fi

# --- 3. Ensure daily cashflow bot is running ---
DAILY_PID=$(pgrep -f "bot/daily_cashflow.py.*--mode paper" | head -1)
if [ -z "$DAILY_PID" ] || ! is_alive "$DAILY_PID"; then
    pkill -f "bot/daily_cashflow.py.*--mode paper" 2>/dev/null
    sleep 1
    log "Daily cashflow bot not running — starting..."
    cd "$BOT_DIR"
    source "$VENV"
    nohup python bot/daily_cashflow.py --mode paper >> /tmp/bot_daily_cashflow.log 2>&1 &
    log "Daily cashflow bot started (PID: $!)"
else
    log "Daily cashflow bot running (PID: $DAILY_PID)"
fi

# --- 4. Health check: verify bots responded recently ---
# If the daily cashflow log hasn't been updated in 30 min during market hours, restart
HOUR=$(date +%H)
if [ "$HOUR" -ge 6 ] && [ "$HOUR" -le 13 ]; then
    DAILY_LOG="/tmp/bot_daily_cashflow.log"
    if [ -f "$DAILY_LOG" ]; then
        LAST_MOD=$(stat -f %m "$DAILY_LOG" 2>/dev/null || echo 0)
        NOW=$(date +%s)
        AGE=$(( NOW - LAST_MOD ))
        if [ "$AGE" -gt 1800 ]; then
            log "WARNING: Daily cashflow log stale (${AGE}s). Restarting..."
            pkill -f "bot/daily_cashflow.py" 2>/dev/null
            sleep 2
            cd "$BOT_DIR"
            source "$VENV"
            nohup python bot/daily_cashflow.py --mode paper >> /tmp/bot_daily_cashflow.log 2>&1 &
            log "Daily cashflow bot force-restarted (PID: $!)"
        fi
    fi
fi

log "Check done. main=$(pgrep -f 'bot/main.py' | head -1) daily=$(pgrep -f 'bot/daily_cashflow.py' | head -1) TWS=$(pgrep -f 'jts\|Trader' | head -1)"
