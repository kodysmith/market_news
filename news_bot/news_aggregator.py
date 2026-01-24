"""
Multi-source news aggregator
Fetches news from FMP, NewsAPI, RSS feeds, and Alpha Vantage
"""
import requests
import feedparser
import time
from datetime import datetime, timedelta
from typing import List, Dict, Optional
import logging
from . import config
from .utils import create_content_hash, extract_source_domain, parse_date, clean_text

logger = logging.getLogger(__name__)


class NewsAggregator:
    """Aggregate news from multiple sources"""
    
    def __init__(self):
        self.fmp_key = config.FMP_API_KEY
        self.newsapi_key = config.NEWSAPI_KEY
        self.alphavantage_key = config.ALPHAVANTAGE_API_KEY
        
        # Rate limiting
        self.last_request_time = {}
        self.min_request_interval = 1.0  # seconds between requests
        
    def _rate_limit(self, source: str):
        """Enforce rate limiting between requests"""
        now = time.time()
        last_time = self.last_request_time.get(source, 0)
        elapsed = now - last_time
        
        if elapsed < self.min_request_interval:
            time.sleep(self.min_request_interval - elapsed)
        
        self.last_request_time[source] = time.time()
    
    def fetch_all_news(self) -> List[Dict]:
        """Fetch news from all available sources"""
        all_news = []
        
        # FMP News
        try:
            fmp_news = self.fetch_fmp_news()
            all_news.extend(fmp_news)
            logger.info(f"Fetched {len(fmp_news)} articles from FMP")
        except Exception as e:
            logger.error(f"Error fetching FMP news: {e}")
        
        # NewsAPI
        try:
            newsapi_news = self.fetch_newsapi()
            all_news.extend(newsapi_news)
            logger.info(f"Fetched {len(newsapi_news)} articles from NewsAPI")
        except Exception as e:
            logger.error(f"Error fetching NewsAPI: {e}")
        
        # RSS Feeds
        try:
            rss_news = self.fetch_rss_feeds()
            all_news.extend(rss_news)
            logger.info(f"Fetched {len(rss_news)} articles from RSS feeds")
        except Exception as e:
            logger.error(f"Error fetching RSS feeds: {e}")
        
        # Alpha Vantage
        try:
            av_news = self.fetch_alphavantage_news()
            all_news.extend(av_news)
            logger.info(f"Fetched {len(av_news)} articles from Alpha Vantage")
        except Exception as e:
            logger.error(f"Error fetching Alpha Vantage news: {e}")
        
        logger.info(f"Total articles fetched: {len(all_news)}")
        return all_news
    
    def fetch_fmp_news(self) -> List[Dict]:
        """Fetch news from Financial Modeling Prep"""
        if not self.fmp_key:
            logger.warning("FMP API key not configured")
            return []
        
        news_items = []
        
        try:
            self._rate_limit('fmp')
            
            # General news
            url = f"https://financialmodelingprep.com/api/v3/fmp/articles"
            params = {
                'page': 0,
                'size': 50,
                'apikey': self.fmp_key
            }
            
            response = requests.get(url, params=params, timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                
                for article in data.get('content', []):
                    news_items.append({
                        'headline': clean_text(article.get('title', '')),
                        'source': 'fmp.com',
                        'url': article.get('link', ''),
                        'raw_summary': clean_text(article.get('content', '')[:500]),
                        'full_text': clean_text(article.get('content', '')),
                        'published_date': parse_date(article.get('date', '')),
                    })
            
            # Stock news (for major indices)
            self._rate_limit('fmp')
            tickers = ['QQQ', 'SPY', 'DIA']
            
            for ticker in tickers:
                url = f"https://financialmodelingprep.com/api/v3/stock_news"
                params = {
                    'tickers': ticker,
                    'limit': 20,
                    'apikey': self.fmp_key
                }
                
                response = requests.get(url, params=params, timeout=10)
                
                if response.status_code == 200:
                    data = response.json()
                    
                    for article in data:
                        news_items.append({
                            'headline': clean_text(article.get('title', '')),
                            'source': extract_source_domain(article.get('url', '')),
                            'url': article.get('url', ''),
                            'raw_summary': clean_text(article.get('text', '')[:500]),
                            'full_text': clean_text(article.get('text', '')),
                            'published_date': parse_date(article.get('publishedDate', '')),
                        })
                
                time.sleep(0.5)  # Be nice to the API
        
        except Exception as e:
            logger.error(f"FMP fetch error: {e}")
        
        return news_items
    
    def fetch_newsapi(self) -> List[Dict]:
        """Fetch news from NewsAPI.org"""
        if not self.newsapi_key:
            logger.warning("NewsAPI key not configured")
            return []
        
        news_items = []
        
        try:
            self._rate_limit('newsapi')
            
            # Business headlines
            url = "https://newsapi.org/v2/top-headlines"
            params = {
                'apiKey': self.newsapi_key,
                'category': 'business',
                'language': 'en',
                'country': 'us',
                'pageSize': 50
            }
            
            response = requests.get(url, params=params, timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                
                for article in data.get('articles', []):
                    news_items.append({
                        'headline': clean_text(article.get('title', '')),
                        'source': extract_source_domain(article.get('url', '')),
                        'url': article.get('url', ''),
                        'raw_summary': clean_text(article.get('description', '') or ''),
                        'full_text': clean_text(article.get('content', '') or ''),
                        'published_date': parse_date(article.get('publishedAt', '')),
                    })
            
            # Finance-specific search
            self._rate_limit('newsapi')
            
            url = "https://newsapi.org/v2/everything"
            params = {
                'apiKey': self.newsapi_key,
                'q': '(stocks OR nasdaq OR "federal reserve" OR inflation OR earnings)',
                'language': 'en',
                'sortBy': 'publishedAt',
                'pageSize': 50,
                'from': (datetime.utcnow() - timedelta(hours=24)).isoformat()
            }
            
            response = requests.get(url, params=params, timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                
                for article in data.get('articles', []):
                    news_items.append({
                        'headline': clean_text(article.get('title', '')),
                        'source': extract_source_domain(article.get('url', '')),
                        'url': article.get('url', ''),
                        'raw_summary': clean_text(article.get('description', '') or ''),
                        'full_text': clean_text(article.get('content', '') or ''),
                        'published_date': parse_date(article.get('publishedAt', '')),
                    })
        
        except Exception as e:
            logger.error(f"NewsAPI fetch error: {e}")
        
        return news_items
    
    def fetch_rss_feeds(self) -> List[Dict]:
        """Fetch news from RSS feeds"""
        news_items = []
        
        for feed_url in config.RSS_FEEDS:
            try:
                self._rate_limit('rss')
                
                feed = feedparser.parse(feed_url)
                source = extract_source_domain(feed_url)
                
                for entry in feed.entries[:30]:  # Limit per feed
                    # Extract published date
                    pub_date = None
                    if hasattr(entry, 'published_parsed') and entry.published_parsed:
                        pub_date = datetime(*entry.published_parsed[:6]).isoformat()
                    elif hasattr(entry, 'updated_parsed') and entry.updated_parsed:
                        pub_date = datetime(*entry.updated_parsed[:6]).isoformat()
                    else:
                        pub_date = datetime.utcnow().isoformat()
                    
                    # Extract summary
                    summary = ''
                    if hasattr(entry, 'summary'):
                        summary = entry.summary
                    elif hasattr(entry, 'description'):
                        summary = entry.description
                    
                    news_items.append({
                        'headline': clean_text(entry.title),
                        'source': source,
                        'url': entry.link,
                        'raw_summary': clean_text(summary)[:500],
                        'full_text': clean_text(summary),
                        'published_date': pub_date,
                    })
            
            except Exception as e:
                logger.error(f"RSS feed error for {feed_url}: {e}")
                continue
        
        return news_items
    
    def fetch_alphavantage_news(self) -> List[Dict]:
        """Fetch news from Alpha Vantage"""
        if not self.alphavantage_key:
            logger.warning("Alpha Vantage API key not configured")
            return []
        
        news_items = []
        
        try:
            self._rate_limit('alphavantage')
            
            # Fetch news sentiment
            url = "https://www.alphavantage.co/query"
            params = {
                'function': 'NEWS_SENTIMENT',
                'topics': 'technology,finance,economy',
                'limit': 50,
                'apikey': self.alphavantage_key
            }
            
            response = requests.get(url, params=params, timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                
                for article in data.get('feed', []):
                    news_items.append({
                        'headline': clean_text(article.get('title', '')),
                        'source': extract_source_domain(article.get('url', '')),
                        'url': article.get('url', ''),
                        'raw_summary': clean_text(article.get('summary', '')[:500]),
                        'full_text': clean_text(article.get('summary', '')),
                        'published_date': parse_date(article.get('time_published', '')),
                    })
        
        except Exception as e:
            logger.error(f"Alpha Vantage fetch error: {e}")
        
        return news_items
    
    def deduplicate_news(self, news_items: List[Dict]) -> List[Dict]:
        """Remove duplicate articles based on content hash"""
        seen_hashes = set()
        unique_items = []
        
        for item in news_items:
            content_hash = create_content_hash(
                item['headline'],
                item['source'],
                item['published_date']
            )
            
            if content_hash not in seen_hashes:
                seen_hashes.add(content_hash)
                item['content_hash'] = content_hash
                unique_items.append(item)
        
        logger.info(f"Deduplication: {len(news_items)} -> {len(unique_items)} articles")
        return unique_items


if __name__ == "__main__":
    # Test the aggregator
    logging.basicConfig(level=logging.INFO)
    
    aggregator = NewsAggregator()
    news = aggregator.fetch_all_news()
    unique_news = aggregator.deduplicate_news(news)
    
    print(f"\nFetched {len(unique_news)} unique articles")
    
    if unique_news:
        print("\nSample article:")
        print(f"Headline: {unique_news[0]['headline']}")
        print(f"Source: {unique_news[0]['source']}")
        print(f"Published: {unique_news[0]['published_date']}")
        print(f"Summary: {unique_news[0]['raw_summary'][:200]}...")


