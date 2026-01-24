# News Intelligence Bot Setup Guide

## Quick Start

The News Intelligence Bot is a 24/7 service that:
- Fetches news from multiple sources (FMP, NewsAPI, RSS feeds, Alpha Vantage)
- Classifies news by importance (1-5 scale)
- Generates LLM summaries with actionable intelligence
- Publishes to database and mobile app API

## API Keys Setup

### Required Keys

#### 1. Financial Modeling Prep (FMP)
You already have this key configured.

#### 2. NewsAPI.org (FREE - REQUIRED)
- **URL:** https://newsapi.org/register
- **Free Tier:** 100 requests/day (sufficient)
- **Time to sign up:** 2 minutes
- **Instructions:**
  1. Go to https://newsapi.org/register
  2. Enter email and create account
  3. Copy your API key
  4. Add to `.env` as `NEWSAPI_KEY=your_key`

### Optional Keys (Service works without these)

#### 3. Alpha Vantage (OPTIONAL)
- **URL:** https://www.alphavantage.co/support/#api-key
- **Free Tier:** 25 requests/day
- **Time to sign up:** 1 minute
- **Note:** Service has RSS fallback if not configured

#### 4. Anthropic Claude (OPTIONAL - RECOMMENDED)
- **URL:** https://console.anthropic.com/
- **Free Tier:** Available for testing
- **Time to sign up:** 3 minutes
- **Note:** Service uses rule-based fallback if not configured
- **Why recommended:** Much better summaries for trading decisions

## Installation

### Step 1: Install Dependencies

```bash
cd /mnt/4tb/stock_scanner/market_news
pip install -r requirements_news.txt
```

### Step 2: Configure Environment

```bash
# Copy template
cp .env.news_template .env

# Edit .env and add your API keys
nano .env
```

Minimum configuration (just add NewsAPI key):
```
FMP_API_KEY=your_existing_fmp_key
NEWSAPI_KEY=your_newsapi_key_here
```

Full configuration (recommended):
```
FMP_API_KEY=your_existing_fmp_key
NEWSAPI_KEY=your_newsapi_key_here
ALPHAVANTAGE_API_KEY=your_alphavantage_key
ANTHROPIC_API_KEY=your_anthropic_key
```

### Step 3: Initialize Database

```bash
python -m news_bot.database --init
```

### Step 4: Test Run

```bash
python -m news_bot.news_service --test
```

This will:
- Fetch news from all sources
- Classify by importance
- Generate summaries
- Export to `data/news.json`
- Create status file

Check the output and verify it's working.

### Step 5: Install as System Service

```bash
sudo bash setup_news_service.sh
```

This creates a systemd service that:
- Runs 24/7
- Auto-restarts on failure
- Logs to journalctl and file

### Step 6: Start the Service

```bash
# Start the service
sudo systemctl start market-news-bot

# Enable auto-start on boot
sudo systemctl enable market-news-bot

# Check status
sudo systemctl status market-news-bot

# View logs
sudo journalctl -u market-news-bot -f
```

## Service Behavior

### Timing
- **News fetch:** Every 10 minutes
- **LLM summaries:** Every 30 minutes (batch processing)
- **QQQ refresh:** Every 24 hours
- **Database cleanup:** Every 24 hours (removes articles >7 days old)

### Data Output
- **Database:** `/mnt/4tb/stock_scanner/market_news/data/market_news.db`
- **Mobile App JSON:** `/mnt/4tb/stock_scanner/market_news/data/news.json`
- **Status:** `/mnt/4tb/stock_scanner/market_news/data/news_bot_status.json`
- **Logs:** `/mnt/4tb/stock_scanner/market_news/data/logs/news_bot.log`

### API Endpoints

#### Get News (Mobile App)
```
GET http://localhost:5000/news.json
```

#### Get News Intelligence (Filtered)
```
GET http://localhost:5000/news/intelligence?importance=4&limit=50
```

#### Get Bot Status
```
GET http://localhost:5000/news/status
```

## Importance Levels

### 5 - Critical
- Fed decisions, FOMC announcements
- War, military conflicts
- Market crashes, trading halts
- Bank failures, systemic crises

### 4 - High
- Interest rate changes
- Inflation data (CPI, PPI)
- Major economic indicators (GDP, employment)
- Mega-cap earnings (AAPL, MSFT, NVDA, etc.)

### 3 - Medium
- QQQ constituent news
- Sector rotation
- Analyst upgrades/downgrades
- M&A activity

### 2 - Low
- Individual stock movements
- General business news

### 1 - Noise
- Promotional content
- Spam

## Monitoring

### Check Service Status
```bash
sudo systemctl status market-news-bot
```

### View Logs
```bash
# Live logs
sudo journalctl -u market-news-bot -f

# Last 100 lines
sudo journalctl -u market-news-bot -n 100

# File logs
tail -f /mnt/4tb/stock_scanner/market_news/data/logs/news_bot.log
```

### Check Database Stats
```bash
python -c "
from news_bot.database import NewsDatabase
with NewsDatabase() as db:
    print(db.get_stats())
"
```

### View Status File
```bash
cat /mnt/4tb/stock_scanner/market_news/data/news_bot_status.json | python -m json.tool
```

## Troubleshooting

### Service won't start
```bash
# Check logs for errors
sudo journalctl -u market-news-bot -n 50

# Test manually
python -m news_bot.news_service --test
```

### No articles being fetched
1. Check API keys in `.env`
2. Verify internet connection
3. Check rate limits on APIs
4. View logs for specific errors

### LLM summaries not generating
1. Check if Anthropic API key is configured
2. Service will use fallback if no key (rule-based summaries)
3. Check logs for API errors

### Mobile app not showing news
1. Verify `/mnt/4tb/stock_scanner/market_news/data/news.json` exists
2. Check file has content and is valid JSON
3. Restart Flask API: `python apis/api.py`

## Manual Operations

### Stop Service
```bash
sudo systemctl stop market-news-bot
```

### Restart Service
```bash
sudo systemctl restart market-news-bot
```

### Run One-Time Fetch
```bash
python -m news_bot.news_service --test
```

### Clear Database
```bash
rm /mnt/4tb/stock_scanner/market_news/data/market_news.db
python -m news_bot.database --init
```

## Cost Estimates

With free tier APIs:
- **FMP:** Already have paid plan
- **NewsAPI:** Free (100 req/day) - plenty for 6 req/hour
- **Alpha Vantage:** Free (25 req/day) - optional
- **RSS Feeds:** Free
- **Anthropic Claude:** ~$0.50-1.00/day for 50 summaries/day

**Total daily cost:** $0.50-1.00 (only if using Claude)

**Free tier option:** Works perfectly without Claude using rule-based summaries.

## Next Steps

After setup:
1. Let it run for 24 hours
2. Check the morning news feed in mobile app
3. Review importance classifications
4. Adjust keywords in `news_bot/config.py` if needed
5. Monitor logs for any issues

## Support

Check logs first:
```bash
sudo journalctl -u market-news-bot -n 100
```

Most issues are related to:
1. Missing API keys
2. API rate limits
3. Internet connectivity


