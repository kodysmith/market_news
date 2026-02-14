# API Keys Quick Setup Guide

Get your bot running in under 5 minutes!

## Required: NewsAPI.org (FREE)

**Time: 2 minutes**

1. Go to: https://newsapi.org/register
2. Enter your email
3. Create a password
4. Verify email (check spam folder)
5. Copy your API key from the dashboard
6. Add to `.env`: `NEWSAPI_KEY=your_key_here`

**Free Tier:** 100 requests/day (plenty for our use - we make ~6/hour)

---

## Optional: Anthropic Claude (RECOMMENDED)

**Time: 3 minutes**

**Why?** Much better news summaries with trading intelligence.

**Without it:** Service works fine with rule-based summaries.

**With it:** Get summaries like:
```
"Fed rate cut signals dovish pivot. Bullish for growth stocks (NVDA, TSLA). 
Pre-market watch for QQQ breakout above $485."
```

**Setup:**
1. Go to: https://console.anthropic.com/
2. Sign up with Google/email
3. Go to API Keys section
4. Create new key
5. Copy key
6. Add to `.env`: `ANTHROPIC_API_KEY=your_key_here`

**Cost:** 
- Free tier: $5 credit (lasts ~5-10 days)
- After: ~$0.50-1.00/day
- Worth it for hedge fund use!

---

## FMP (Financial Modeling Prep) — Quote/spot fallback

**Why?** When IBKR is down or unavailable, the API uses FMP for stock quotes and the dashboard price strip (reliable fallback).

**Setup:** Add `FMP_API_KEY` to `data/config.json` (same file as `MASSIVE_API_KEY`, `ALPHAVANTAGE_API_KEY`). The Flask API reads it from there. If you already use FMP in `.env` for news, copy that key into `config.json` for the quote fallback.

---

## Optional: Alpha Vantage (OPTIONAL)

**Time: 1 minute**

**Why?** Extra news source with sentiment data.

**Without it:** Service has RSS fallback (CNBC, Reuters, Bloomberg).

**Setup:**
1. Go to: https://www.alphavantage.co/support/#api-key
2. Enter email
3. Copy key immediately (they email it too)
4. Add to `.env`: `ALPHAVANTAGE_API_KEY=your_key_here`

**Free Tier:** 25 requests/day

---

## Minimal Setup (Works Today)

Just get NewsAPI key (2 minutes):

```bash
# Edit .env
nano .env

# Add these lines:
FMP_API_KEY=your_existing_fmp_key
NEWSAPI_KEY=paste_newsapi_key_here
```

Save and you're ready to run!

---

## Full Setup (Recommended)

Get all three keys (6 minutes total):

```bash
# Edit .env
nano .env

# Add all keys:
FMP_API_KEY=your_existing_fmp_key
NEWSAPI_KEY=paste_newsapi_key_here
ALPHAVANTAGE_API_KEY=paste_alphavantage_key_here
ANTHROPIC_API_KEY=paste_anthropic_key_here
```

---

## Testing Your Keys

After adding keys to `.env`:

```bash
cd /mnt/4tb/stock_scanner/market_news

# Install dependencies
pip install -r requirements_news.txt

# Initialize database
python -m news_bot.database --init

# Test run (fetches news with your keys)
python -m news_bot.news_service --test
```

You should see:
- News articles fetched from sources
- Articles classified by importance
- Summaries generated
- `data/news.json` created

If you see errors about API keys, double-check:
1. Keys are in `.env` (not `env_template.txt`)
2. No extra spaces in key values
3. No quotes around keys

---

## Key Priority

**Must have:**
- NewsAPI (free, 2 min signup)

**Should have:**
- Anthropic Claude (paid but cheap, much better summaries)

**Nice to have:**
- Alpha Vantage (free, extra news source)

**Already have:**
- FMP (you're set!)

---

## Free Tier Limits

All on free tier:
- **FMP:** Your existing plan
- **NewsAPI:** 100 req/day
- **Alpha Vantage:** 25 req/day
- **RSS Feeds:** Unlimited (CNBC, Reuters, Bloomberg)

Bot makes:
- ~144 fetches/day (every 10 min)
- Split across all sources
- Well within limits!

---

## Questions?

**"Do I need all keys to start?"**
No! Just NewsAPI is enough. Other sources add redundancy.

**"Will it work without Claude?"**
Yes! Uses rule-based summaries. Not as smart but functional.

**"What if I hit rate limits?"**
Service automatically falls back to other sources. No crashes.

**"Can I add keys later?"**
Yes! Just edit `.env` and restart: `sudo systemctl restart market-news-bot`


