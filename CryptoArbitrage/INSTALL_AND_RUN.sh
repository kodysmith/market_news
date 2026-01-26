#!/bin/bash
# One-command install and run for crypto arbitrage

set -e

echo "════════════════════════════════════════════════════════════"
echo "  🚀 CRYPTO ARBITRAGE - AUTO INSTALL & RUN"
echo "════════════════════════════════════════════════════════════"
echo ""

cd "$(dirname "$0")"

# Step 1: Install dependencies
echo "Step 1: Installing dependencies..."
if [ -d "../venv" ]; then
    echo "  Using existing virtual environment"
else
    echo "  Creating virtual environment..."
    cd .. && python3 -m venv venv && cd CryptoArbitrage
fi

source ../venv/bin/activate
echo "  Installing packages..."
pip install -q ccxt flask flask-cors pydantic pyyaml python-dotenv psutil aiosqlite 2>&1 | grep -v "Requirement already satisfied" || true
echo "  ✅ Dependencies installed"
echo ""

# Step 2: Test system  
echo "Step 2: Testing system..."
python test_system.py
echo ""

# Step 3: Instructions
echo "════════════════════════════════════════════════════════════"
echo "  ✅ SYSTEM READY!"
echo "════════════════════════════════════════════════════════════"
echo ""
echo "To start the system:"
echo ""
echo "  Terminal 1:"
echo "    cd /mnt/4tb/stock_scanner/market_news/CryptoArbitrage"
echo "    ./start_arbitrage.sh"
echo ""
echo "  Terminal 2:"
echo "    cd /mnt/4tb/stock_scanner/market_news/CryptoArbitrage"
echo "    ./start_dashboard.sh"
echo ""
echo "  Browser:"
echo "    http://localhost:5001"
echo ""
echo "════════════════════════════════════════════════════════════"
echo ""
echo "📖 Read MORNING_BRIEFING.md for complete guide!"
echo ""
