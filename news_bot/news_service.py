"""
24/7 News Intelligence Service
Main daemon that orchestrates news fetching, classification, and summarization
"""
import time
import json
import sys
import signal
from datetime import datetime, timedelta
from typing import Dict
import logging
from pathlib import Path

from . import config
from .news_aggregator import NewsAggregator
from .classifier import NewsClassifier
from .llm_summarizer import LLMSummarizer
from .database import NewsDatabase

# Configure logging
log_dir = Path(config.LOG_PATH).parent
log_dir.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=getattr(logging, config.LOG_LEVEL),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(config.LOG_PATH),
        logging.StreamHandler(sys.stdout)
    ]
)

logger = logging.getLogger(__name__)


class NewsIntelligenceService:
    """Main service daemon"""
    
    def __init__(self):
        self.running = False
        self.aggregator = NewsAggregator()
        self.classifier = NewsClassifier()
        self.summarizer = LLMSummarizer()
        
        # Timing trackers
        self.last_fetch_time = 0
        self.last_llm_batch_time = 0
        self.last_qqq_refresh_time = 0
        self.last_cleanup_time = 0
        
        # Statistics
        self.stats = {
            'service_start_time': datetime.utcnow().isoformat(),
            'total_articles_fetched': 0,
            'total_articles_classified': 0,
            'total_articles_summarized': 0,
            'fetch_cycles': 0,
            'errors': 0,
            'last_successful_fetch': None,
            'last_error': None
        }
        
        logger.info("News Intelligence Service initialized")
    
    def start(self):
        """Start the service"""
        self.running = True
        logger.info("=" * 60)
        logger.info("News Intelligence Service Starting")
        logger.info("=" * 60)
        
        # Initialize database
        self._initialize_database()
        
        # Initial QQQ refresh
        self._refresh_qqq_constituents()
        
        # Register signal handlers
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
        
        # Main service loop
        logger.info("Entering main service loop...")
        
        try:
            while self.running:
                self._service_cycle()
                time.sleep(10)  # Check every 10 seconds
        
        except Exception as e:
            logger.error(f"Fatal error in service loop: {e}", exc_info=True)
            self.stats['last_error'] = str(e)
            self.stats['errors'] += 1
        
        finally:
            self._shutdown()
    
    def _service_cycle(self):
        """Execute one cycle of the service"""
        current_time = time.time()
        
        # Fetch news every 10 minutes
        if current_time - self.last_fetch_time >= config.FETCH_INTERVAL_SECONDS:
            self._fetch_and_classify_news()
            self.last_fetch_time = current_time
        
        # Run LLM batch every 30 minutes
        if current_time - self.last_llm_batch_time >= config.LLM_BATCH_INTERVAL_SECONDS:
            self._run_llm_batch()
            self.last_llm_batch_time = current_time
        
        # Refresh QQQ constituents every 24 hours
        if current_time - self.last_qqq_refresh_time >= config.QQQ_REFRESH_INTERVAL_SECONDS:
            self._refresh_qqq_constituents()
            self.last_qqq_refresh_time = current_time
        
        # Cleanup old articles every 24 hours
        if current_time - self.last_cleanup_time >= config.CLEANUP_INTERVAL_SECONDS:
            self._cleanup_old_articles()
            self.last_cleanup_time = current_time
        
        # Export to JSON for mobile app (after each fetch/summarization)
        self._export_to_json()
        
        # Update status file
        self._update_status()
    
    def _initialize_database(self):
        """Initialize database schema"""
        try:
            logger.info("Initializing database...")
            with NewsDatabase() as db:
                db.init_schema()
            logger.info("Database initialized successfully")
        except Exception as e:
            logger.error(f"Database initialization error: {e}")
            raise
    
    def _fetch_and_classify_news(self):
        """Fetch news from all sources and classify"""
        try:
            logger.info("Starting news fetch cycle...")
            
            # Fetch news
            news_items = self.aggregator.fetch_all_news()
            unique_news = self.aggregator.deduplicate_news(news_items)
            
            logger.info(f"Fetched {len(unique_news)} unique articles")
            
            if not unique_news:
                logger.warning("No news articles fetched")
                return
            
            # Classify articles
            classified_news = self.classifier.classify_batch(unique_news)
            
            # Insert into database
            inserted_count = 0
            with NewsDatabase() as db:
                for article in classified_news:
                    article_id = db.insert_article(
                        headline=article['headline'],
                        source=article['source'],
                        url=article['url'],
                        raw_summary=article['raw_summary'],
                        full_text=article.get('full_text', ''),
                        published_date=article['published_date'],
                        content_hash=article['content_hash']
                    )
                    
                    if article_id:
                        # Insert importance score
                        db.insert_importance_score(
                            article_id=article_id,
                            importance_level=article['importance_level'],
                            keyword_score=article['keyword_score'],
                            mentioned_tickers=article['mentioned_tickers'],
                            macro_events=article['macro_events'],
                            classification_reason=article['classification_reason']
                        )
                        inserted_count += 1
            
            logger.info(f"Inserted {inserted_count} new articles into database")
            
            # Update statistics
            self.stats['total_articles_fetched'] += len(unique_news)
            self.stats['total_articles_classified'] += inserted_count
            self.stats['fetch_cycles'] += 1
            self.stats['last_successful_fetch'] = datetime.utcnow().isoformat()
        
        except Exception as e:
            logger.error(f"Error in fetch/classify cycle: {e}", exc_info=True)
            self.stats['errors'] += 1
            self.stats['last_error'] = str(e)
    
    def _run_llm_batch(self):
        """Run LLM summarization batch"""
        try:
            logger.info("Starting LLM batch summarization...")
            
            with NewsDatabase() as db:
                # Get articles that need summarization
                articles_to_summarize = db.get_articles_for_llm_processing(limit=50)
                
                if not articles_to_summarize:
                    logger.info("No articles need summarization")
                    return
                
                logger.info(f"Processing {len(articles_to_summarize)} articles for summarization")
                
                # Summarize articles
                summaries = self.summarizer.summarize_batch(articles_to_summarize)
                
                # Insert summaries into database
                for summary in summaries:
                    db.insert_llm_summary(
                        article_id=summary['article_id'],
                        summary=summary['summary'],
                        affected_tickers=summary.get('affected_tickers', []),
                        time_sensitivity=summary.get('time_sensitivity', 'multi-day'),
                        actionable_insight=summary.get('actionable_insight', ''),
                        sentiment=summary.get('sentiment', 'neutral')
                    )
                
                logger.info(f"Completed {len(summaries)} LLM summaries")
                self.stats['total_articles_summarized'] += len(summaries)
        
        except Exception as e:
            logger.error(f"Error in LLM batch: {e}", exc_info=True)
            self.stats['errors'] += 1
    
    def _refresh_qqq_constituents(self):
        """Refresh QQQ constituents list"""
        try:
            logger.info("Refreshing QQQ constituents...")
            
            # Fetch QQQ holdings (using a simple approach)
            # In production, you'd fetch from a proper data source
            # For now, we'll use a hardcoded top-100 list
            
            constituents = self._fetch_qqq_holdings()
            
            with NewsDatabase() as db:
                db.update_qqq_constituents(constituents)
                ticker_list = db.get_qqq_constituents()
            
            # Update classifier with new constituents
            self.classifier.update_qqq_constituents(ticker_list)
            
            logger.info(f"Refreshed {len(ticker_list)} QQQ constituents")
        
        except Exception as e:
            logger.error(f"Error refreshing QQQ constituents: {e}")
    
    def _fetch_qqq_holdings(self) -> list:
        """Fetch QQQ holdings from API or use default list"""
        # Top NASDAQ-100 stocks (sample - in production, fetch from API)
        default_holdings = [
            {'ticker': 'AAPL', 'company_name': 'Apple Inc', 'weight': 12.0},
            {'ticker': 'MSFT', 'company_name': 'Microsoft Corp', 'weight': 10.5},
            {'ticker': 'NVDA', 'company_name': 'NVIDIA Corp', 'weight': 8.5},
            {'ticker': 'GOOGL', 'company_name': 'Alphabet Inc Class A', 'weight': 4.5},
            {'ticker': 'GOOG', 'company_name': 'Alphabet Inc Class C', 'weight': 4.0},
            {'ticker': 'AMZN', 'company_name': 'Amazon.com Inc', 'weight': 6.0},
            {'ticker': 'META', 'company_name': 'Meta Platforms Inc', 'weight': 5.0},
            {'ticker': 'TSLA', 'company_name': 'Tesla Inc', 'weight': 4.5},
            {'ticker': 'AVGO', 'company_name': 'Broadcom Inc', 'weight': 3.5},
            {'ticker': 'COST', 'company_name': 'Costco Wholesale Corp', 'weight': 3.0},
            {'ticker': 'NFLX', 'company_name': 'Netflix Inc', 'weight': 2.5},
            {'ticker': 'AMD', 'company_name': 'Advanced Micro Devices', 'weight': 2.0},
            {'ticker': 'ADBE', 'company_name': 'Adobe Inc', 'weight': 2.0},
            {'ticker': 'CSCO', 'company_name': 'Cisco Systems Inc', 'weight': 1.8},
            {'ticker': 'INTC', 'company_name': 'Intel Corp', 'weight': 1.7},
            {'ticker': 'CMCSA', 'company_name': 'Comcast Corp', 'weight': 1.6},
            {'ticker': 'PEP', 'company_name': 'PepsiCo Inc', 'weight': 1.5},
            {'ticker': 'QCOM', 'company_name': 'Qualcomm Inc', 'weight': 1.5},
            {'ticker': 'TXN', 'company_name': 'Texas Instruments', 'weight': 1.4},
            {'ticker': 'AMGN', 'company_name': 'Amgen Inc', 'weight': 1.3},
        ]
        
        return default_holdings
    
    def _cleanup_old_articles(self):
        """Remove old articles from database"""
        try:
            logger.info(f"Cleaning up articles older than {config.ARCHIVE_DAYS} days...")
            
            with NewsDatabase() as db:
                deleted = db.archive_old_articles(config.ARCHIVE_DAYS)
            
            logger.info(f"Cleaned up {deleted} old articles")
        
        except Exception as e:
            logger.error(f"Error cleaning up old articles: {e}")
    
    def _export_to_json(self):
        """Export news to JSON for mobile app"""
        try:
            with NewsDatabase() as db:
                articles = db.get_recent_articles_for_export(config.MAX_NEWS_ITEMS_FOR_APP)
            
            # Write to JSON file
            output_dir = Path(config.JSON_OUTPUT_PATH).parent
            output_dir.mkdir(parents=True, exist_ok=True)
            
            with open(config.JSON_OUTPUT_PATH, 'w') as f:
                json.dump(articles, f, indent=2)
            
            logger.debug(f"Exported {len(articles)} articles to {config.JSON_OUTPUT_PATH}")
        
        except Exception as e:
            logger.error(f"Error exporting to JSON: {e}")
    
    def _update_status(self):
        """Update service status file"""
        try:
            with NewsDatabase() as db:
                db_stats = db.get_stats()
            
            status = {
                **self.stats,
                'database_stats': db_stats,
                'status': 'running' if self.running else 'stopped',
                'last_update': datetime.utcnow().isoformat()
            }
            
            status_dir = Path(config.STATUS_FILE).parent
            status_dir.mkdir(parents=True, exist_ok=True)
            
            with open(config.STATUS_FILE, 'w') as f:
                json.dump(status, f, indent=2)
        
        except Exception as e:
            logger.error(f"Error updating status: {e}")
    
    def _signal_handler(self, signum, frame):
        """Handle shutdown signals"""
        logger.info(f"Received signal {signum}, shutting down gracefully...")
        self.running = False
    
    def _shutdown(self):
        """Clean shutdown"""
        logger.info("Shutting down News Intelligence Service...")
        self._update_status()
        logger.info("Service stopped")


def main():
    """Main entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(description='News Intelligence Service')
    parser.add_argument('--test', action='store_true', help='Run one cycle and exit')
    args = parser.parse_args()
    
    service = NewsIntelligenceService()
    
    if args.test:
        logger.info("Running in TEST mode - single cycle")
        service._initialize_database()
        service._refresh_qqq_constituents()
        service._fetch_and_classify_news()
        service._run_llm_batch()
        service._export_to_json()
        service._update_status()
        logger.info("Test cycle completed")
    else:
        service.start()


if __name__ == "__main__":
    main()


