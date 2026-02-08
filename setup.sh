#!/usr/bin/env bash
# One-time server setup for Fisher full-market scan: venv, deps, DB/queue, enqueue, cron.
#
# After cloning: create .env with DATABASE_URL, then run ./setup.sh from repo root.
# Cron will run the worker hourly and enqueue monthly so the scan runs continuously.
#
# Usage: ./setup.sh   (from repo root, with DATABASE_URL in .env or environment)

set -e

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO_ROOT"

echo "=========================================="
echo "Fisher full-market scan — server setup"
echo "=========================================="
echo "Repo: $REPO_ROOT"
echo ""

# 1. Load .env so DATABASE_URL is set for Python steps
if [ -f .env ]; then
  set -a
  . ./.env
  set +a
  echo "Loaded .env"
else
  echo "No .env found. Create .env with DATABASE_URL=postgresql://..."
  echo "Example: cp env_template.txt .env && edit .env"
fi

if [ -z "${DATABASE_URL}" ] && [ -z "${SUPABASE_DB_URL}" ]; then
  echo "ERROR: Set DATABASE_URL or SUPABASE_DB_URL in .env or environment."
  exit 1
fi

# 2. Python venv and deps
VENV="$REPO_ROOT/.venv"
if [ ! -d "$VENV" ]; then
  echo "Creating venv at .venv ..."
  python3 -m venv "$VENV"
fi
echo "Using Python: $VENV/bin/python3"
"$VENV/bin/pip" install -q -r requirements.txt
echo "Dependencies installed."
echo ""

# 3. Fisher full-scan setup: SEC universe, queue table, enqueue
echo "Running Fisher full-scan setup (SEC universe + queue table + enqueue)..."
"$VENV/bin/python3" scripts/setup_fisher_full_scan.py --reset
echo ""

# 4. Cron wrapper and log dir
mkdir -p logs
CRON_SCRIPT="$REPO_ROOT/scripts/run_fisher_cron.sh"
chmod +x "$CRON_SCRIPT"

# 5. Crontab entries
WORKER_CRON="0 * * * * $CRON_SCRIPT worker >> $REPO_ROOT/logs/fisher_worker.log 2>&1"
ENQUEUE_CRON="0 0 1 * * $CRON_SCRIPT enqueue >> $REPO_ROOT/logs/fisher_enqueue.log 2>&1"

if crontab -l 2>/dev/null | grep -q "run_fisher_cron.sh"; then
  echo "Cron entries for Fisher scan already present. Skipping crontab add."
  echo "Current crontab:"
  crontab -l 2>/dev/null | grep -E "run_fisher_cron|fisher" || true
else
  echo "Adding cron jobs..."
  (crontab -l 2>/dev/null; echo "$WORKER_CRON"; echo "$ENQUEUE_CRON") | crontab -
  echo "Cron installed:"
  echo "  Hourly: worker (one batch)"
  echo "  Monthly (1st 00:00): enqueue --reset"
fi

echo ""
echo "=========================================="
echo "Setup complete. Fisher full scan is scheduled."
echo "=========================================="
echo "  Worker:  hourly via cron (logs: logs/fisher_worker.log)"
echo "  Enqueue: 1st of month via cron (logs: logs/fisher_enqueue.log)"
echo ""
echo "  Manual run:  $VENV/bin/python3 scripts/run_fisher_scan_worker.py --batch 50"
echo "  View cron:   crontab -l"
echo "  Remove:     crontab -e  (delete lines with run_fisher_cron.sh)"
echo ""
