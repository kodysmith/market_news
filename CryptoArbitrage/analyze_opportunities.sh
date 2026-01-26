#!/bin/bash
# Analyze Arbitrage Opportunities
# 
# This script analyzes logged opportunities to show:
# - How many opportunities per day
# - How long opportunities stay open
# - Which pairs/exchanges are most profitable

set -e

cd "$(dirname "$0")"

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}  Opportunity Analysis${NC}"
echo -e "${GREEN}========================================${NC}"
echo ""

# Check if virtual environment exists
if [ -d "venv" ]; then
    source venv/bin/activate
fi

# Check if database exists
if [ ! -f "data/arbitrage.db" ]; then
    echo -e "${YELLOW}⚠${NC} No database found. Run paper trading first:"
    echo "  ./run_paper_trading.sh 10"
    echo ""
    exit 1
fi

echo -e "${GREEN}✓${NC} Analyzing opportunities from database..."
echo ""

# Run analysis
python3 scripts/analyze_opportunities.py

echo ""


