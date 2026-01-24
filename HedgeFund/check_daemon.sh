#!/bin/bash
# Check daemon status and view logs

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "================================================"
echo "📊 Hedge Fund Daemon Status Check"
echo "================================================"
echo ""

# Check systemd service
if systemctl is-active --quiet hedgefund-daemon 2>/dev/null; then
    echo "✅ Systemd Service: RUNNING"
    systemctl status hedgefund-daemon --no-pager | head -20
    echo ""
    echo "Recent logs:"
    sudo journalctl -u hedgefund-daemon -n 20 --no-pager
    exit 0
fi

# Check supervisor
if [ -f logs/supervisor.sock ]; then
    echo "✅ Supervisor: RUNNING"
    ../venv/bin/supervisorctl -c supervisor.conf status
    echo ""
    echo "Recent logs:"
    tail -30 logs/daemon_stdout.log
    exit 0
fi

# Check background process
if [ -f logs/daemon.pid ]; then
    PID=$(cat logs/daemon.pid)
    if kill -0 "$PID" 2>/dev/null; then
        echo "✅ Background Process: RUNNING (PID: $PID)"
        echo ""
        echo "Recent logs:"
        tail -30 logs/daemon.out
        exit 0
    fi
fi

# Check for any Python process
PIDS=$(pgrep -f "paper_trading_daemon.py" || true)
if [ -n "$PIDS" ]; then
    echo "✅ Process Found: RUNNING (PID: $PIDS)"
    ps aux | grep paper_trading_daemon.py | grep -v grep
    exit 0
fi

echo "❌ Daemon: NOT RUNNING"
echo ""
echo "To start the daemon, run:"
echo "  ./start_daemon.sh dev        # Development (foreground)"
echo "  ./start_daemon.sh background # Background with nohup"
echo "  ./start_daemon.sh supervisor # With supervisor (recommended)"
echo "  sudo ./start_daemon.sh systemd # As system service"


