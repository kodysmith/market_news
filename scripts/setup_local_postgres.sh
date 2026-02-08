#!/usr/bin/env bash
# Start local Postgres in Docker and apply schemas for Fisher + trading.
# Usage: ./scripts/setup_local_postgres.sh   (run from repo root)

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$ROOT_DIR"

CONTAINER_NAME="marketnews-pg"
DB_NAME="marketnews"
DB_USER="postgres"
DB_PASSWORD="postgres"
# Use 5433 on host so we don't conflict with Homebrew/system Postgres on 5432
HOST_PORT="${HOST_PORT:-5433}"
DATABASE_URL="postgresql://${DB_USER}:${DB_PASSWORD}@localhost:${HOST_PORT}/${DB_NAME}"

echo "Using Postgres container: $CONTAINER_NAME, DB: $DB_NAME, host port: $HOST_PORT"

# Start container if not already running
if ! docker ps -q -f name="^${CONTAINER_NAME}$" 2>/dev/null | head -1 | grep -q .; then
  if docker ps -aq -f name="^${CONTAINER_NAME}$" 2>/dev/null | head -1 | grep -q .; then
    echo "Starting existing container..."
    docker start "$CONTAINER_NAME"
  else
    echo "Creating and starting container (host port $HOST_PORT to avoid conflict with existing Postgres on 5432)..."
    docker run --name "$CONTAINER_NAME" \
      -e POSTGRES_PASSWORD="$DB_PASSWORD" \
      -e POSTGRES_DB="$DB_NAME" \
      -p "${HOST_PORT}:5432" \
      -d postgres:16
  fi
  echo "Waiting for Postgres to be ready..."
  sleep 3
  for i in {1..30}; do
    if docker exec "$CONTAINER_NAME" pg_isready -U "$DB_USER" -d "$DB_NAME" 2>/dev/null; then
      break
    fi
    sleep 1
  done
fi

# Apply schemas (idempotent)
echo "Applying auth stub for local RLS..."
docker exec -i "$CONTAINER_NAME" psql -U "$DB_USER" -d "$DB_NAME" < "$SCRIPT_DIR/local_postgres_init.sql"

echo "Applying supabase_schema.sql..."
docker exec -i "$CONTAINER_NAME" psql -U "$DB_USER" -d "$DB_NAME" < "$ROOT_DIR/supabase_schema.sql"

echo "Applying supabase_schema_fisher.sql..."
docker exec -i "$CONTAINER_NAME" psql -U "$DB_USER" -d "$DB_NAME" < "$ROOT_DIR/supabase_schema_fisher.sql"

echo ""
echo "Local Postgres is ready."
echo ""
echo "Set in your shell (or add to .env):"
echo "  export DATABASE_URL=\"$DATABASE_URL\""
echo ""
echo "Then run Fisher pipeline (e.g. limit 5 companies):"
echo "  python3 -c \"from fisher.edgar_watcher import run_watcher; print(run_watcher(limit=5))\""
echo "  python3 -c \"from fisher.market_updater import run_market_updater; print(run_market_updater(days=90, update_peers=True))\""
echo "  python3 -c \"from fisher.scoring_engine import run_scoring_job; print(run_scoring_job())\""
echo ""
