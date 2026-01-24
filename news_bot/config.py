"""
Configuration for News Intelligence Bot
"""
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# API Keys
FMP_API_KEY = os.getenv("FMP_API_KEY", "")
NEWSAPI_KEY = os.getenv("NEWSAPI_KEY", "")
ALPHAVANTAGE_API_KEY = os.getenv("ALPHAVANTAGE_API_KEY", "")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")

# Database
DB_PATH = "/mnt/4tb/stock_scanner/market_news/data/market_news.db"
JSON_OUTPUT_PATH = "/mnt/4tb/stock_scanner/market_news/data/news.json"
LOG_PATH = "/mnt/4tb/stock_scanner/market_news/data/logs/news_bot.log"
STATUS_FILE = "/mnt/4tb/stock_scanner/market_news/data/news_bot_status.json"

# Service Configuration
FETCH_INTERVAL_SECONDS = 600  # 10 minutes
LLM_BATCH_INTERVAL_SECONDS = 1800  # 30 minutes
QQQ_REFRESH_INTERVAL_SECONDS = 86400  # 24 hours
CLEANUP_INTERVAL_SECONDS = 86400  # 24 hours
ARCHIVE_DAYS = 7

# News Sources
RSS_FEEDS = [
    "https://www.cnbc.com/id/100003114/device/rss/rss.html",  # CNBC Top News
    "https://feeds.reuters.com/reuters/businessNews",  # Reuters Business
    "https://feeds.bloomberg.com/markets/news.rss",  # Bloomberg Markets
]

# Classification Keywords
CRITICAL_KEYWORDS = [
    "federal reserve decision", "fed decision", "fomc announcement",
    "interest rate cut", "interest rate hike", "rate decision",
    "war", "military conflict", "invasion", "attack",
    "market crash", "circuit breaker", "trading halt",
    "emergency meeting", "bank failure", "systemic risk",
    "nuclear", "cyber attack on financial"
]

HIGH_KEYWORDS = [
    "interest rate", "inflation", "cpi", "ppi", "pce",
    "employment", "jobs report", "unemployment", "nonfarm payroll",
    "gdp", "recession", "economic growth",
    "earnings", "guidance", "revenue miss", "earnings beat",
    "fed", "powell", "yellen", "treasury",
    "china trade", "tariff", "trade war",
    "oil price", "crude", "opec"
]

MEDIUM_KEYWORDS = [
    "analyst upgrade", "analyst downgrade", "price target",
    "sector rotation", "institutional buying", "institutional selling",
    "merger", "acquisition", "buyout", "takeover",
    "fda approval", "regulatory approval",
    "dividend", "buyback", "stock split",
    "ipo", "direct listing"
]

MEGA_CAP_TICKERS = [
    "AAPL", "MSFT", "NVDA", "GOOGL", "GOOG", "AMZN", "META", "TSLA",
    "BRK.B", "AVGO", "JPM", "LLY", "V", "MA", "WMT", "XOM"
]

# LLM Configuration
LLM_MODEL = "claude-3-haiku-20240307"  # Fast and cost-effective
LLM_MAX_TOKENS = 300
LLM_TEMPERATURE = 0.3

# Mobile App Settings
MAX_NEWS_ITEMS_FOR_APP = 50

# Logging
LOG_LEVEL = "INFO"


