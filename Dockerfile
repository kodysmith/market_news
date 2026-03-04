# Dockerfile for MarketNews APIs (Cloud Run)
# Serves apis/app_factory (report, news, valuation, cockpit, fisher, gex, etc.)

FROM python:3.11-slim

WORKDIR /app

# System deps for some Python packages
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Optional: extras used by some routes (uncomment if needed)
# RUN pip install --no-cache-dir pyyaml supabase

# Application code (apis, QuantEngine, fisher, utils, data are included; see .dockerignore)
COPY . .

ENV PYTHONUNBUFFERED=1
ENV PORT=8080
EXPOSE 8080

# Cloud Run sets PORT; use it. Single worker is enough for low traffic.
CMD ["sh", "-c", "gunicorn -b 0.0.0.0:${PORT:-8080} -w 1 --timeout 120 apis.api:app"]
