#!/bin/bash
# Start arbitrage dashboard

set -e

cd "$(dirname "$0")"

PORT="${1:-5001}"

echo "================================================"
echo "📊 Starting Crypto Arbitrage Dashboard"
echo "================================================"
echo "Port: $PORT"
echo "URL: http://localhost:$PORT"
echo ""
echo "Press Ctrl+C to stop"
echo "================================================"
echo ""

export DASHBOARD_PORT=$PORT

../venv/bin/python src/dashboard/dashboard_server.py


