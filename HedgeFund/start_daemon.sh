#!/bin/bash
# Quick start script for the paper trading daemon
# Usage: ./start_daemon.sh [mode]
# Modes: dev (foreground), systemd, supervisor

set -e

MODE="${1:-dev}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "================================================"
echo "🚀 Starting Hedge Fund Paper Trading Daemon"
echo "================================================"
echo "Mode: $MODE"
echo "Directory: $SCRIPT_DIR"
echo ""

case "$MODE" in
    dev|foreground)
        echo "Starting in development mode (foreground)..."
        echo "Press Ctrl+C to stop"
        echo ""
        ../venv/bin/python src/daemon/paper_trading_daemon.py
        ;;
    
    background)
        echo "Starting in background mode..."
        nohup ../venv/bin/python src/daemon/paper_trading_daemon.py > logs/daemon.out 2>&1 &
        PID=$!
        echo "✅ Daemon started with PID: $PID"
        echo "   Logs: tail -f logs/daemon.out"
        echo "   Stop: kill $PID"
        echo "$PID" > logs/daemon.pid
        ;;
    
    systemd)
        echo "Installing and starting systemd service..."
        
        # Check if running as root
        if [ "$EUID" -ne 0 ]; then 
            echo "Systemd mode requires root privileges"
            echo "Run: sudo ./start_daemon.sh systemd"
            exit 1
        fi
        
        # Copy service file
        cp hedgefund-daemon.service /etc/systemd/system/
        
        # Reload systemd
        systemctl daemon-reload
        
        # Enable on boot
        systemctl enable hedgefund-daemon
        
        # Start service
        systemctl start hedgefund-daemon
        
        echo "✅ Systemd service installed and started"
        echo ""
        echo "Useful commands:"
        echo "  Status:  sudo systemctl status hedgefund-daemon"
        echo "  Logs:    sudo journalctl -u hedgefund-daemon -f"
        echo "  Stop:    sudo systemctl stop hedgefund-daemon"
        echo "  Restart: sudo systemctl restart hedgefund-daemon"
        echo "  Disable: sudo systemctl disable hedgefund-daemon"
        ;;
    
    supervisor)
        echo "Starting with Supervisor..."
        
        # Check if supervisor is installed
        if ! command -v supervisord &> /dev/null; then
            echo "Installing supervisor..."
            ../venv/bin/pip install supervisor
        fi
        
        # Start supervisord
        ../venv/bin/supervisord -c supervisor.conf
        
        sleep 2
        
        # Check status
        ../venv/bin/supervisorctl -c supervisor.conf status
        
        echo ""
        echo "✅ Supervisor started"
        echo ""
        echo "Useful commands:"
        echo "  Status:  supervisorctl -c supervisor.conf status"
        echo "  Logs:    tail -f logs/daemon_stdout.log"
        echo "  Stop:    supervisorctl -c supervisor.conf stop hedgefund-daemon"
        echo "  Restart: supervisorctl -c supervisor.conf restart hedgefund-daemon"
        echo "  Shutdown: supervisorctl -c supervisor.conf shutdown"
        ;;
    
    *)
        echo "❌ Unknown mode: $MODE"
        echo ""
        echo "Usage: ./start_daemon.sh [mode]"
        echo ""
        echo "Modes:"
        echo "  dev        - Run in foreground (default, for testing)"
        echo "  background - Run in background with nohup"
        echo "  systemd    - Install as systemd service (requires sudo)"
        echo "  supervisor - Run with supervisor (auto-restart)"
        exit 1
        ;;
esac

echo ""
echo "================================================"


