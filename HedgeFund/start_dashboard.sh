#!/bin/bash
# Start the dashboard server

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

PORT="${1:-5000}"

echo "================================================"
echo "🖥️  Starting Hedge Fund Dashboard"
echo "================================================"
echo "Port: $PORT"
echo "URL: http://localhost:$PORT"
echo ""
echo "Press Ctrl+C to stop"
echo "================================================"
echo ""

export DASHBOARD_PORT=$PORT

# Check if using SQLite or PostgreSQL
USE_SQLITE="${USE_SQLITE:-true}"

if [ "$USE_SQLITE" = "true" ]; then
    echo "Using SQLite database"
    # Activate virtual environment and run SQLite version
    ../venv/bin/python src/dashboard/dashboard_server_sqlite.py
else
    echo "Using PostgreSQL database"
    # Activate virtual environment and run PostgreSQL version
    ../venv/bin/python src/dashboard/dashboard_server.py
fi

