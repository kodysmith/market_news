#!/usr/bin/env bash
# Deploy MarketNews APIs to Google Cloud Run.
# Usage: ./scripts/deploy_cloudrun.sh   (from repo root)
#        PROJECT_ID=my-project REGION=us-east1 ./scripts/deploy_cloudrun.sh

set -e

# Run from repo root (parent of scripts/)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$ROOT_DIR"

PROJECT_ID="${PROJECT_ID:-$(gcloud config get-value project 2>/dev/null)}"
REGION="${REGION:-us-central1}"
SERVICE_NAME="${SERVICE_NAME:-marketnews-api}"

if [ -z "$PROJECT_ID" ]; then
  echo "Error: PROJECT_ID not set. Run: gcloud config set project YOUR_PROJECT_ID"
  exit 1
fi

echo "Deploying $SERVICE_NAME to Cloud Run (project=$PROJECT_ID, region=$REGION)"
echo "Building from source and deploying..."
gcloud run deploy "$SERVICE_NAME" \
  --source . \
  --region "$REGION" \
  --allow-unauthenticated \
  --set-env-vars "PORT=8080" \
  --platform managed

echo ""
echo "Done. Service URL:"
gcloud run services describe "$SERVICE_NAME" --region "$REGION" --format 'value(status.url)'
echo ""
echo "Set API_BASE_URL in your Flutter app (or .env) to the URL above."
