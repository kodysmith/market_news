#!/bin/bash
# Deploy price monitor to Google Cloud Functions
#
# Prerequisites:
#   1. gcloud CLI installed: brew install google-cloud-sdk
#   2. Authenticated: gcloud auth login
#   3. Project set: gcloud config set project YOUR_PROJECT_ID
#   4. APIs enabled: gcloud services enable cloudfunctions.googleapis.com cloudscheduler.googleapis.com
#   5. Firebase service account has Cloud Messaging permissions

set -e

# ─── Config ──────────────────────────────────────────────────────────────────
FUNCTION_NAME="price-monitor"
REGION="us-east1"
RUNTIME="python312"
MEMORY="256MB"
TIMEOUT="30s"

# Load env vars from .env
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ENV_FILE="$SCRIPT_DIR/../../.env"

if [ -f "$ENV_FILE" ]; then
    SUPABASE_URL=$(grep '^SUPABASE_URL=' "$ENV_FILE" | cut -d= -f2)
    SUPABASE_KEY=$(grep '^SUPABASE_KEY=' "$ENV_FILE" | cut -d= -f2)
else
    echo "Error: .env file not found at $ENV_FILE"
    exit 1
fi

echo "Deploying $FUNCTION_NAME to $REGION..."

# Deploy the function
gcloud functions deploy "$FUNCTION_NAME" \
    --gen2 \
    --runtime "$RUNTIME" \
    --region "$REGION" \
    --source "$SCRIPT_DIR" \
    --entry-point price_monitor \
    --trigger-http \
    --allow-unauthenticated \
    --memory "$MEMORY" \
    --timeout "$TIMEOUT" \
    --set-env-vars "SUPABASE_URL=$SUPABASE_URL,SUPABASE_KEY=$SUPABASE_KEY,FCM_TOPIC=market_alerts"

# Get the function URL
FUNCTION_URL=$(gcloud functions describe "$FUNCTION_NAME" --region "$REGION" --format='value(serviceConfig.uri)' 2>/dev/null || echo "")
if [ -z "$FUNCTION_URL" ]; then
    FUNCTION_URL=$(gcloud functions describe "$FUNCTION_NAME" --region "$REGION" --format='value(httpsTrigger.url)' 2>/dev/null || echo "unknown")
fi

echo ""
echo "Function deployed: $FUNCTION_URL"

# Create Cloud Scheduler job (every 1 minute, M-F 9:25-16:20 ET)
echo ""
echo "Creating Cloud Scheduler job..."

# Delete existing job if it exists
gcloud scheduler jobs delete "${FUNCTION_NAME}-trigger" \
    --location "$REGION" --quiet 2>/dev/null || true

gcloud scheduler jobs create http "${FUNCTION_NAME}-trigger" \
    --location "$REGION" \
    --schedule="* 9-16 * * 1-5" \
    --time-zone="America/New_York" \
    --uri="$FUNCTION_URL" \
    --http-method=POST \
    --attempt-deadline=30s \
    --description="Trigger price monitor every minute during market hours"

echo ""
echo "=== DEPLOYMENT COMPLETE ==="
echo "Function: $FUNCTION_URL"
echo "Schedule: Every minute, M-F 9:00-16:59 ET"
echo ""
echo "Test: curl -X POST $FUNCTION_URL"
echo "Logs: gcloud functions logs read $FUNCTION_NAME --region $REGION --limit 20"
echo "Manage alerts: python scripts/manage_alerts.py list"
