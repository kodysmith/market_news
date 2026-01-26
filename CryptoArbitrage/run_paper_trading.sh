#!/bin/bash
# Run Paper Trading Mode
# 
# This script runs the crypto arbitrage system in paper trading mode,
# which detects real opportunities but simulates execution without placing real trades.

set -e

cd "$(dirname "$0")"

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}  Crypto Arbitrage - Paper Trading${NC}"
echo -e "${GREEN}========================================${NC}"
echo ""

# Check if virtual environment exists
if [ -d "venv" ]; then
    echo -e "${GREEN}✓${NC} Activating virtual environment..."
    source venv/bin/activate
else
    echo -e "${YELLOW}⚠${NC} Virtual environment not found. Using system Python."
fi

# Check for duration argument
DURATION_ARG=""
if [ ! -z "$1" ]; then
    DURATION_ARG="--duration $1"
    echo -e "${GREEN}✓${NC} Running for $1 minutes"
else
    echo -e "${GREEN}✓${NC} Running until stopped (Ctrl+C to exit)"
fi

echo ""
echo -e "${GREEN}Starting paper trading...${NC}"
echo ""

# Set environment to simulation mode
export CRYPTO_ENV=simulation

# Run paper trading
python3 scripts/paper_trading.py $DURATION_ARG

echo ""
echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}Paper trading session complete!${NC}"
echo -e "${GREEN}========================================${NC}"
echo ""
echo "To analyze results, run:"
echo "  ./analyze_opportunities.sh"
echo ""


