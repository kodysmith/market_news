#!/bin/bash
# Install all dependencies for crypto arbitrage system

set -e

cd "$(dirname "$0")"

echo "================================================"
echo "📦 Installing Crypto Arbitrage Dependencies"
echo "================================================"
echo ""

# Check if venv exists
if [ ! -d "../venv" ]; then
    echo "Creating virtual environment..."
    cd ..
    python3 -m venv venv
    cd CryptoArbitrage
    echo "✅ Virtual environment created"
else
    echo "✅ Virtual environment found"
fi

echo ""
echo "Activating virtual environment..."
source ../venv/bin/activate

echo ""
echo "Installing dependencies from requirements.txt..."
pip install -r requirements.txt --quiet

echo ""
echo "================================================"
echo "✅ INSTALLATION COMPLETE!"
echo "================================================"
echo ""
echo "Installed packages:"
echo "  - ccxt (exchange API)"
echo "  - ccxt.pro (websockets)"  
echo "  - flask (dashboard)"
echo "  - pydantic (config validation)"
echo "  - and more..."
echo ""
echo "Next steps:"
echo "  1. Test system: python test_system.py"
echo "  2. Start daemon: ./start_arbitrage.sh"
echo "  3. Start dashboard: ./start_dashboard.sh"
echo ""
echo "================================================"


