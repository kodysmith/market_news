#!/usr/bin/env bash
# Launch the SPX IC Bot in simulation mode.
# Records all entry/exit decisions and simulated P&L to Supabase.
# Does NOT require IBKR — uses yfinance for market data and BS for pricing.
#
# Usage:
#   ./run_bot.sh              # Daemon mode (runs all day via scheduler)
#   ./run_bot.sh --once       # Single shot (entry + exit check, then exit)

set -e
cd "$(dirname "$0")"

# Load environment variables
set -a
source .env
set +a

# Prefer live TWS for real-time data; yfinance fallback handles the rest
export IBKR_PORT=${IBKR_PORT:-7496}

# Activate virtualenv
if [ -d "venv" ]; then
    source venv/bin/activate
fi

echo "========================================"
echo "  SPX Iron Condor Bot — Simulation Mode"
echo "  Market data: yfinance (IBKR if available)"
echo "  Fills: Black-Scholes model prices"
echo "  State: Supabase (bot_positions / bot_trades)"
echo "========================================"
echo ""

python -m bot.main --mode dry-run "$@"
