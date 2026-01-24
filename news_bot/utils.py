"""
Utility functions for News Intelligence Bot
"""
import hashlib
import re
from datetime import datetime
from typing import List, Set
import logging

logger = logging.getLogger(__name__)


def create_content_hash(headline: str, source: str, published_date: str) -> str:
    """Create unique hash for article deduplication"""
    content = f"{headline}|{source}|{published_date}"
    return hashlib.md5(content.encode()).hexdigest()


def extract_tickers(text: str) -> List[str]:
    """
    Extract potential stock tickers from text
    Returns uppercase tickers that match common patterns
    """
    if not text:
        return []
    
    # Pattern: $AAPL or AAPL mentioned in context
    pattern = r'\$?([A-Z]{1,5})(?:\s|,|\.|\)|$)'
    matches = re.findall(pattern, text)
    
    # Filter out common false positives
    exclude = {
        'A', 'I', 'CEO', 'CFO', 'USA', 'US', 'UK', 'EU', 'AI', 'IT', 'TV',
        'THE', 'AND', 'FOR', 'ARE', 'NOT', 'BUT', 'ALL', 'CAN', 'WHO',
        'HAS', 'HAD', 'WAS', 'ITS', 'HIS', 'HER', 'OUR', 'NEW', 'NOW',
        'GET', 'GOT', 'PUT', 'SET', 'RUN', 'CUT', 'ADD', 'TOP', 'BIG',
        'END', 'AGO', 'YET', 'OLD', 'OWN', 'SEE', 'SAY', 'TWO', 'MAY',
        'DAY', 'WAY', 'USE', 'HER', 'ANY', 'HOW', 'OUT', 'MAN', 'TRY'
    }
    
    tickers = [t for t in matches if t not in exclude and len(t) >= 2]
    
    # Remove duplicates while preserving order
    seen = set()
    unique_tickers = []
    for ticker in tickers:
        if ticker not in seen:
            seen.add(ticker)
            unique_tickers.append(ticker)
    
    return unique_tickers[:10]  # Limit to top 10


def validate_ticker(ticker: str, valid_tickers: Set[str]) -> bool:
    """Validate if ticker is in our watchlist"""
    return ticker.upper() in valid_tickers


def parse_date(date_str: str) -> str:
    """
    Parse various date formats to ISO format
    """
    if not date_str:
        return datetime.utcnow().isoformat()
    
    # Already ISO format
    if 'T' in date_str and ('Z' in date_str or '+' in date_str):
        return date_str
    
    # Try common formats
    formats = [
        "%Y-%m-%d %H:%M:%S",
        "%Y-%m-%d",
        "%a, %d %b %Y %H:%M:%S %Z",
        "%a, %d %b %Y %H:%M:%S %z",
    ]
    
    for fmt in formats:
        try:
            dt = datetime.strptime(date_str, fmt)
            return dt.isoformat()
        except ValueError:
            continue
    
    # Default to current time if parsing fails
    logger.warning(f"Could not parse date: {date_str}, using current time")
    return datetime.utcnow().isoformat()


def clean_text(text: str) -> str:
    """Clean text by removing extra whitespace and special characters"""
    if not text:
        return ""
    
    # Remove multiple spaces
    text = re.sub(r'\s+', ' ', text)
    
    # Remove special characters but keep basic punctuation
    text = re.sub(r'[^\w\s\.\,\!\?\;\:\-\(\)\$\%]', '', text)
    
    return text.strip()


def truncate_text(text: str, max_length: int = 500) -> str:
    """Truncate text to max length, ending at sentence boundary if possible"""
    if not text or len(text) <= max_length:
        return text
    
    # Try to truncate at sentence boundary
    truncated = text[:max_length]
    last_period = truncated.rfind('.')
    last_question = truncated.rfind('?')
    last_exclaim = truncated.rfind('!')
    
    last_sentence_end = max(last_period, last_question, last_exclaim)
    
    if last_sentence_end > max_length * 0.7:  # At least 70% of desired length
        return truncated[:last_sentence_end + 1]
    
    return truncated + "..."


def calculate_keyword_score(text: str, keywords: List[str]) -> float:
    """
    Calculate keyword relevance score (0-1)
    Higher score means more keywords matched
    """
    if not text or not keywords:
        return 0.0
    
    text_lower = text.lower()
    matches = sum(1 for keyword in keywords if keyword.lower() in text_lower)
    
    # Normalize by number of keywords, cap at 1.0
    score = min(matches / 5.0, 1.0)  # 5+ matches = perfect score
    return score


def is_spam_or_promotional(headline: str, text: str) -> bool:
    """
    Detect spam or promotional content
    """
    spam_indicators = [
        'free trial', 'click here', 'subscribe now', 'limited time',
        'act now', 'exclusive offer', 'sponsored', 'advertisement',
        'buy now', 'discount code', 'promo code', '% off'
    ]
    
    combined = (headline + ' ' + (text or '')).lower()
    
    return any(indicator in combined for indicator in spam_indicators)


def extract_source_domain(url: str) -> str:
    """Extract domain name from URL"""
    if not url:
        return "unknown"
    
    # Remove protocol
    domain = re.sub(r'https?://', '', url)
    
    # Remove www.
    domain = re.sub(r'^www\.', '', domain)
    
    # Get just the domain
    domain = domain.split('/')[0]
    
    return domain


def is_market_hours() -> bool:
    """
    Check if current time is during US market hours (9:30 AM - 4:00 PM ET)
    """
    from datetime import datetime
    import pytz
    
    try:
        et = pytz.timezone('America/New_York')
        now_et = datetime.now(et)
        
        # Market is closed on weekends
        if now_et.weekday() >= 5:  # Saturday = 5, Sunday = 6
            return False
        
        market_open = now_et.replace(hour=9, minute=30, second=0, microsecond=0)
        market_close = now_et.replace(hour=16, minute=0, second=0, microsecond=0)
        
        return market_open <= now_et <= market_close
    except Exception as e:
        logger.error(f"Error checking market hours: {e}")
        return False


