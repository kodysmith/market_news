# Dockerfile for Cloud Run

FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Install additional dependencies for trading system
RUN pip install --no-cache-dir \
    flask>=2.3.0 \
    pyyaml>=6.0 \
    yfinance>=0.2.0 \
    supabase>=2.0.0 \
    google-cloud-secret-manager>=2.16.0 \
    alpaca-trade-api>=3.0.0 \
    pandas>=2.0.0 \
    numpy>=1.24.0

# Copy application code
COPY . .

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV PORT=8080

# Expose port
EXPOSE 8080

# Run Flask app
CMD ["python", "main_live.py"]
