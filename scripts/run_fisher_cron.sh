#!/usr/bin/env bash
# Wrapper for cron: run Fisher scan worker (one batch) or monthly enqueue.
# Usage: run_fisher_cron.sh [worker|enqueue]
# Called by setup.sh-generated crontab; loads .env from repo root.

set -e
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
if [ -f .env ]; then
  set -a
  . ./.env
  set +a
fi
PYTHON="${ROOT}/.venv/bin/python3"
[ -x "$PYTHON" ] || PYTHON="python3"

case "${1:-worker}" in
  worker) exec "$PYTHON" scripts/run_fisher_scan_worker.py --once --batch 50 ;;
  enqueue) exec "$PYTHON" scripts/enqueue_fisher_scan.py --reset ;;
  *) echo "Usage: $0 worker|enqueue" >&2; exit 1 ;;
esac
