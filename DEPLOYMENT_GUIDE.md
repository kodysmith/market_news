# Deployment Guide

Guide for deploying the trading system to GCP Cloud Run.

## Prerequisites

- ✅ **Local testing completed** (see `SETUP_LOCAL.md`)
- ✅ **Backtest validation** - Strategy tested on historical data
- ✅ **Paper trading validation** (optional but recommended) - Tested with real-time data locally
- GCP account with billing enabled
- gcloud CLI installed and configured
- Docker installed (for local testing)

**Important**: Before deploying to cloud, make sure you've:
1. ✅ Tested the system locally with backtest mode
2. ✅ Validated with paper trading locally (if using paper/live modes)
3. ✅ Verified all components work correctly

## Step 0: Local Testing (Do This First!)

Before deploying to cloud, test everything locally:

```bash
# 1. Test the system
python test_trading_system.py

# 2. Run a backtest
python run_local.py --mode backtest --start-date 2024-01-01 --end-date 2024-01-31

# 3. Test paper trading locally (requires Supabase)
export SUPABASE_URL="https://your-project.supabase.co"
export SUPABASE_KEY="your-anon-key"
python run_local.py --mode paper
```

See `SETUP_LOCAL.md` for complete local setup instructions.

## Step 1: Set Up Supabase

1. Create a Supabase project at https://supabase.com
2. Go to SQL Editor
3. Copy and run the contents of `supabase_schema.sql`
4. Note your Supabase URL and anon key

## Step 2: Set Up GCP Secret Manager

Store your secrets in GCP Secret Manager:

```bash
# Set your project
gcloud config set project YOUR_PROJECT_ID

# Create secrets
echo -n "https://your-project.supabase.co" | gcloud secrets create supabase-url --data-file=-
echo -n "your-anon-key" | gcloud secrets create supabase-key --data-file=-
echo -n "your-alpaca-api-key" | gcloud secrets create alpaca-api-key --data-file=-
echo -n "your-alpaca-api-secret" | gcloud secrets create alpaca-api-secret --data-file=-
```

## Step 3: Build and Deploy to Cloud Run

### Option A: Deploy from Source

```bash
gcloud run deploy trading-system \
  --source . \
  --platform managed \
  --region us-central1 \
  --allow-unauthenticated \
  --set-env-vars TRADING_MODE=paper \
  --set-secrets SUPABASE_URL=supabase-url:latest,SUPABASE_KEY=supabase-key:latest \
  --memory 512Mi \
  --cpu 1 \
  --timeout 300 \
  --max-instances 1
```

### Option B: Build Docker Image First

```bash
# Build image
docker build -t gcr.io/YOUR_PROJECT_ID/trading-system .

# Push to GCR
docker push gcr.io/YOUR_PROJECT_ID/trading-system

# Deploy
gcloud run deploy trading-system \
  --image gcr.io/YOUR_PROJECT_ID/trading-system \
  --platform managed \
  --region us-central1 \
  --allow-unauthenticated \
  --set-env-vars TRADING_MODE=paper \
  --set-secrets SUPABASE_URL=supabase-url:latest,SUPABASE_KEY=supabase-key:latest
```

## Step 4: Set Up Cloud Scheduler

Create a scheduled job to run daily at 3:50pm ET (20:50 UTC):

```bash
# Get your Cloud Run service URL
SERVICE_URL=$(gcloud run services describe trading-system --region us-central1 --format 'value(status.url)')

# Create scheduler job
gcloud scheduler jobs create http trading-daily \
  --location us-central1 \
  --schedule="50 20 * * 1-5" \
  --uri="${SERVICE_URL}/run" \
  --http-method=POST \
  --headers="Content-Type=application/json" \
  --message-body='{"mode":"paper"}' \
  --time-zone="America/New_York"
```

Note: The schedule `50 20 * * 1-5` means:
- 20:50 UTC (3:50pm ET)
- Weekdays only (Monday-Friday)

## Step 5: Grant Permissions

Grant Cloud Scheduler permission to invoke Cloud Run:

```bash
PROJECT_NUMBER=$(gcloud projects describe $(gcloud config get-value project) --format='value(projectNumber)')
SERVICE_ACCOUNT="${PROJECT_NUMBER}-compute@developer.gserviceaccount.com"

gcloud run services add-iam-policy-binding trading-system \
  --region us-central1 \
  --member="serviceAccount:${SERVICE_ACCOUNT}" \
  --role="roles/run.invoker"
```

## Step 6: Test Deployment

### Test Health Endpoint

```bash
SERVICE_URL=$(gcloud run services describe trading-system --region us-central1 --format 'value(status.url)')
curl ${SERVICE_URL}/health
```

### Test Run Endpoint

```bash
curl -X POST ${SERVICE_URL}/run \
  -H "Content-Type: application/json" \
  -d '{"mode": "paper"}'
```

### Test Scheduler

```bash
# Trigger the job manually
gcloud scheduler jobs run trading-daily --location us-central1
```

## Step 7: Monitor

### View Logs

```bash
gcloud run services logs read trading-system --region us-central1
```

### View Scheduler Jobs

```bash
gcloud scheduler jobs list --location us-central1
```

### View Scheduler Execution History

```bash
gcloud scheduler jobs describe trading-daily --location us-central1
```

## Switching to Live Mode

⚠️ **WARNING**: Only switch to live mode after thorough paper trading validation (2-4 weeks minimum).

1. Update the Cloud Run service:

```bash
gcloud run services update trading-system \
  --region us-central1 \
  --set-env-vars TRADING_MODE=live \
  --set-secrets ALPACA_API_KEY=alpaca-api-key:latest,ALPACA_API_SECRET=alpaca-api-secret:latest
```

2. Update the scheduler to use live mode (optional):

```bash
gcloud scheduler jobs update http trading-daily \
  --location us-central1 \
  --message-body='{"mode":"live"}'
```

## Troubleshooting

### Service Not Starting

- Check logs: `gcloud run services logs read trading-system --region us-central1`
- Verify secrets are accessible
- Check environment variables

### Scheduler Not Triggering

- Verify service account has `run.invoker` role
- Check scheduler job status
- Verify the schedule is correct (UTC time)

### Database Connection Issues

- Verify Supabase URL and key are correct
- Check Supabase RLS policies
- Ensure schema has been run

## Cost Estimation

- Cloud Run: ~$0.40/month (1 request/day, 512MB memory)
- Cloud Scheduler: Free (up to 3 jobs)
- Secret Manager: ~$0.06/month (4 secrets)
- Supabase: Free tier available

Total: ~$0.50/month for paper trading

## Security Best Practices

1. Use Secret Manager for all secrets (never hardcode)
2. Enable Cloud Run authentication for production
3. Use service accounts with minimal permissions
4. Enable audit logging
5. Regularly rotate API keys
6. Monitor for unusual activity

## Next Steps

### Before Deployment

1. ✅ **Test locally** (see `SETUP_LOCAL.md`):
   - Run backtests: `python run_local.py --mode backtest --start-date 2024-01-01 --end-date 2024-01-31`
   - Test paper trading: `python run_local.py --mode paper`
   - Verify all components work correctly

### Deployment Steps

2. ✅ Deploy to Cloud Run
3. ✅ Set up Cloud Scheduler
4. ✅ Monitor paper trading for 2-4 weeks
5. ✅ Compare paper results to backtest
6. ⚠️ Only then consider live trading

### Local vs Cloud

- **Local**: Use for development, testing, and debugging
- **Cloud**: Use for production automation and monitoring
- **Recommended**: Develop locally → Validate locally → Deploy to cloud → Monitor → Go live
