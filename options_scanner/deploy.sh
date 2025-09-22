#!/bin/bash

# Options Scanner Deployment Script

echo "🚀 Deploying Options Scanner..."

# Check if .env file exists
if [ ! -f .env ]; then
    echo "❌ .env file not found. Please copy env_example.txt to .env and configure it."
    exit 1
fi

# Install dependencies
echo "📦 Installing dependencies..."
pip install -r requirements.txt

# Run tests
echo "🧪 Running tests..."
python test_scanner.py

if [ $? -ne 0 ]; then
    echo "❌ Tests failed. Please fix issues before deploying."
    exit 1
fi

echo "✅ Tests passed!"

# Test single run
echo "🔍 Testing scanner with single run..."
python main_worker.py

if [ $? -eq 0 ]; then
    echo "✅ Scanner test successful!"
    echo "📊 Check report.json for results"
else
    echo "❌ Scanner test failed. Check logs above."
    exit 1
fi

echo "🎉 Deployment complete!"
echo ""
echo "Next steps:"
echo "1. Set up cron job or Cloud Scheduler"
echo "2. Monitor logs for any issues"
echo "3. Check Firebase notifications are working"
echo "4. Verify report.json is being generated correctly"

