#!/bin/bash
# Stop the running daemon

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "🛑 Stopping Hedge Fund Paper Trading Daemon..."
echo ""

# Check if running as systemd service
if systemctl is-active --quiet hedgefund-daemon 2>/dev/null; then
    echo "Found systemd service, stopping..."
    sudo systemctl stop hedgefund-daemon
    echo "✅ Systemd service stopped"
    exit 0
fi

# Check if running under supervisor
if [ -f logs/supervisor.sock ]; then
    echo "Found supervisor process, stopping..."
    ../venv/bin/supervisorctl -c supervisor.conf stop hedgefund-daemon
    echo "✅ Supervisor process stopped"
    exit 0
fi

# Check if running in background (via PID file)
if [ -f logs/daemon.pid ]; then
    PID=$(cat logs/daemon.pid)
    if kill -0 "$PID" 2>/dev/null; then
        echo "Found background process (PID: $PID), stopping..."
        kill "$PID"
        echo "✅ Background process stopped"
        rm logs/daemon.pid
        exit 0
    else
        echo "⚠️  PID file exists but process not running"
        rm logs/daemon.pid
    fi
fi

# Look for any running Python processes
PIDS=$(pgrep -f "paper_trading_daemon.py" || true)
if [ -n "$PIDS" ]; then
    echo "Found daemon process(es): $PIDS"
    echo "Stopping..."
    kill $PIDS
    sleep 2
    
    # Force kill if still running
    STILL_RUNNING=$(pgrep -f "paper_trading_daemon.py" || true)
    if [ -n "$STILL_RUNNING" ]; then
        echo "Force killing..."
        kill -9 $STILL_RUNNING
    fi
    
    echo "✅ Daemon processes stopped"
    exit 0
fi

echo "ℹ️  No running daemon found"


