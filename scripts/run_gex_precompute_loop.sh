#!/usr/bin/env bash
# Run GEX/cockpit precompute every 5 minutes in a loop (keeps SPY, SPX, XSP, NDX fresh in compute_result_cache).
# Use on the server alongside run_compute_worker.py. Requires API running (API_BASE_URL) and .env (SUPABASE_URL + SUPABASE_SECRET_KEY).
#
# Usage:
#   ./scripts/run_gex_precompute_loop.sh
#   Or: bash scripts/run_gex_precompute_loop.sh
#
# Stop with Ctrl+C.

set -e
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

if [[ -f .env ]]; then
  set -a
  source .env
  set +a
fi

# Use repo venv so dependencies (supabase, requests, etc.) are available
if [[ -d "$ROOT/venv/bin" ]]; then
  source "$ROOT/venv/bin/activate"
fi

INTERVAL=300  # 5 minutes

echo "GEX/cockpit precompute loop: every ${INTERVAL}s (Ctrl+C to stop)"
while true; do
  python3 scripts/run_gex_cockpit_precompute.py
  sleep "$INTERVAL"
done
