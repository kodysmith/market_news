# Deploy MarketNews APIs to Google Cloud Run

This guide deploys the Flask API server (report, news, valuation, cockpit, Fisher, GEX, etc.) as a container to **Google Cloud Run**. Cost for light use (hundreds of requests per week) is typically **$0/month** (free tier).

## Prerequisites

- **Google Cloud project** with billing enabled (free tier still applies).
- **gcloud CLI** installed and authenticated:  
  [Install gcloud](https://cloud.google.com/sdk/docs/install) then `gcloud auth login` and `gcloud config set project YOUR_PROJECT_ID`.

## Quick deploy (from repo root)

```bash
# Set your GCP project
export PROJECT_ID=your-gcp-project-id
gcloud config set project $PROJECT_ID

# Deploy from source (Cloud Build builds the Dockerfile, then deploys to Cloud Run)
gcloud run deploy marketnews-api \
  --source . \
  --region us-central1 \
  --allow-unauthenticated \
  --set-env-vars "PORT=8080"
```

After the first deploy, you'll get a URL like `https://marketnews-api-xxxxx-uc.a.run.app`. Use that as `API_BASE_URL` in your Flutter app (e.g. in `.env` or build config).

## Optional: set env vars and secrets

If your APIs need API keys or a database URL, set them in Cloud Run:

```bash
gcloud run deploy marketnews-api \
  --source . \
  --region us-central1 \
  --allow-unauthenticated \
  --set-env-vars "PORT=8080,ALPHAVANTAGE_API_KEY=your-key,FMP_API_KEY=your-key"
```

For Supabase/Fisher (Postgres), set the connection string (use Secret Manager in production):

```bash
gcloud run deploy marketnews-api \
  --source . \
  --region us-central1 \
  --allow-unauthenticated \
  --set-env-vars "PORT=8080" \
  --set-env-vars "DATABASE_URL=postgresql://user:pass@host:5432/postgres"
```

Or use **Secret Manager** and reference in Cloud Run:

```bash
# Create secret (one-time)
echo -n "postgresql://..." | gcloud secrets create db-url --data-file=-

# Deploy with secret
gcloud run deploy marketnews-api \
  --source . \
  --region us-central1 \
  --allow-unauthenticated \
  --set-secrets "DATABASE_URL=db-url:latest"
```

## Deploy script

From repo root:

```bash
./scripts/deploy_cloudrun.sh
```

Or with a custom project/region:

```bash
PROJECT_ID=my-project REGION=us-east1 ./scripts/deploy_cloudrun.sh
```

## What gets deployed

- **Dockerfile** builds a Python 3.11 image with `apis.api:app` (Flask app from `apis/app_factory.py`).
- **Gunicorn** runs the app on `0.0.0.0:$PORT` (Cloud Run sets `PORT`, usually 8080).
- **Included**: `apis/`, `QuantEngine/`, `fisher/`, `utils/`, `data/` (see `.dockerignore` for exclusions).

## Local test of the image

```bash
docker build -t marketnews-api .
docker run -p 8080:8080 -e PORT=8080 marketnews-api
# Then open http://localhost:8080/report.json (or /fisher/universe, etc.)
```

## Cost

- **Free tier**: 2M requests/month, 180k vCPU-seconds, 360k GiB-seconds.
- **Light use** (hundreds of calls per week): almost always **$0/month**.
- [Cloud Run pricing](https://cloud.google.com/run/pricing)
