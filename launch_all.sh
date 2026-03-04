#!/usr/bin/env bash
# launch_all.sh — Start all SPX bot, compute worker, GEX loop, and monitoring stack.
# Each process runs in its own named screen session so you can attach to any of them.
#
# Usage:
#   ./launch_all.sh          # start everything
#   ./launch_all.sh stop     # stop all screen sessions + docker stack
#
# Attach to a session:
#   screen -r bot            # SPX IC Bot
#   screen -r compute        # Compute worker daemon
#   screen -r gex            # GEX precompute loop
#
# Detach from a session:    Ctrl+A then D

set -e
ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT"

VENV_PYTHON="$ROOT/venv/bin/python3"
[ -x "$VENV_PYTHON" ] || VENV_PYTHON="python3"

# ── helpers ──────────────────────────────────────────────────────────────────

start_screen() {
    local name="$1"; shift
    if screen -list | grep -q "\.$name\b"; then
        echo "  [skip] screen '$name' already running"
    else
        echo "  [start] $name"
        screen -dmS "$name" bash -c "cd $ROOT; set -a; source .env; set +a; source venv/bin/activate 2>/dev/null; $*; exec bash"
    fi
}

stop_screen() {
    local name="$1"
    if screen -list | grep -q "\.$name\b"; then
        echo "  [stop] $name"
        screen -S "$name" -X quit
    fi
}

# ── stop mode ────────────────────────────────────────────────────────────────

if [[ "${1:-}" == "stop" ]]; then
    echo "=== Stopping all services ==="
    stop_screen bot
    stop_screen compute
    stop_screen gex
    echo "  [stop] docker monitoring stack"
    docker compose -f "$ROOT/monitoring/docker-compose.yml" down 2>/dev/null || true
    echo "Done."
    exit 0
fi

# ── start mode ───────────────────────────────────────────────────────────────

echo "========================================================"
echo "  Launching SPX Bot + Workers + Monitoring"
echo "========================================================"
echo ""

# 1. Monitoring stack (Prometheus + Grafana)
echo "[1/4] Monitoring stack (Prometheus + Grafana)..."
docker compose -f "$ROOT/monitoring/docker-compose.yml" up -d
echo "      Grafana  → http://localhost:3000  (admin / admin)"
echo "      Prometheus → http://localhost:9090"
echo ""

# 2. SPX Iron Condor Bot
echo "[2/4] SPX Iron Condor Bot (simulation/dry-run)..."
start_screen bot "$VENV_PYTHON -m bot.main --mode dry-run"
echo ""

# 3. Compute worker daemon
echo "[3/4] Compute job queue worker (daemon)..."
start_screen compute "$VENV_PYTHON scripts/run_compute_worker.py --daemon"
echo ""

# 4. GEX precompute loop (every 5 min)
echo "[4/4] GEX / cockpit precompute loop (every 5 min)..."
start_screen gex "bash scripts/run_gex_precompute_loop.sh"
echo ""

echo "========================================================"
echo "  All services launched!"
echo ""
echo "  Attach to a process:"
echo "    screen -r bot       # SPX Iron Condor Bot"
echo "    screen -r compute   # Compute worker"
echo "    screen -r gex       # GEX precompute loop"
echo ""
echo "  See all screens:      screen -ls"
echo "  Detach from screen:   Ctrl+A then D"
echo ""
echo "  Stop everything:      ./launch_all.sh stop"
echo "========================================================"
