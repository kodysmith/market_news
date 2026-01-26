#!/bin/bash
# Start crypto arbitrage daemon

set -e

cd "$(dirname "$0")"

echo "================================================"
echo "₿  Starting Crypto Arbitrage Daemon"
echo "================================================"
echo ""

# Check environment
ENV="${CRYPTO_ENV:-development}"
echo "Environment: $ENV"

if [ "$ENV" = "production" ]; then
    echo "⚠️  WARNING: Production mode - REAL TRADING!"
    read -p "Continue? (y/n): " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        echo "Cancelled"
        exit 1
    fi
fi

echo ""
echo "Starting daemon..."
echo "Press Ctrl+C to stop"
echo ""

# Run daemon
../venv/bin/python src/daemon/arbitrage_daemon.py


