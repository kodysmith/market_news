"""
Database management for News Intelligence Bot
"""
import sqlite3
import json
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional
import logging
from . import config

logger = logging.getLogger(__name__)


class NewsDatabase:
    """Manage news articles database"""
    
    def __init__(self, db_path: str = None):
        self.db_path = db_path or config.DB_PATH
        self.conn = None
        
    def connect(self):
        """Establish database connection"""
        self.conn = sqlite3.connect(self.db_path)
        self.conn.row_factory = sqlite3.Row
        return self.conn
    
    def close(self):
        """Close database connection"""
        if self.conn:
            self.conn.close()
            self.conn = None
    
    def __enter__(self):
        self.connect()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
    
    def init_schema(self):
        """Initialize database schema"""
        cursor = self.conn.cursor()
        
        # News articles table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS news_articles (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                headline TEXT NOT NULL,
                source TEXT NOT NULL,
                url TEXT UNIQUE NOT NULL,
                raw_summary TEXT,
                full_text TEXT,
                published_date TEXT NOT NULL,
                fetched_date TEXT NOT NULL,
                content_hash TEXT UNIQUE NOT NULL,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # Importance scores table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS importance_scores (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                article_id INTEGER NOT NULL,
                importance_level INTEGER NOT NULL,
                keyword_score REAL NOT NULL,
                mentioned_tickers TEXT,
                macro_events TEXT,
                classification_reason TEXT,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (article_id) REFERENCES news_articles(id)
            )
        """)
        
        # LLM summaries table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS llm_summaries (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                article_id INTEGER NOT NULL,
                summary TEXT NOT NULL,
                affected_tickers TEXT,
                time_sensitivity TEXT,
                actionable_insight TEXT,
                sentiment TEXT,
                processed_date TEXT NOT NULL,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (article_id) REFERENCES news_articles(id)
            )
        """)
        
        # QQQ constituents table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS qqq_constituents (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                ticker TEXT UNIQUE NOT NULL,
                company_name TEXT,
                weight REAL,
                last_updated TEXT NOT NULL,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # Create indexes for performance
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_article_published 
            ON news_articles(published_date DESC)
        """)
        
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_importance_level 
            ON importance_scores(importance_level DESC)
        """)
        
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_article_content_hash 
            ON news_articles(content_hash)
        """)
        
        self.conn.commit()
        logger.info("Database schema initialized successfully")
    
    def insert_article(self, headline: str, source: str, url: str, 
                      raw_summary: str, full_text: str, published_date: str,
                      content_hash: str) -> Optional[int]:
        """Insert a news article, return article_id or None if duplicate"""
        try:
            cursor = self.conn.cursor()
            cursor.execute("""
                INSERT INTO news_articles 
                (headline, source, url, raw_summary, full_text, published_date, 
                 fetched_date, content_hash)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (headline, source, url, raw_summary, full_text, published_date,
                  datetime.utcnow().isoformat(), content_hash))
            
            self.conn.commit()
            return cursor.lastrowid
        except sqlite3.IntegrityError:
            # Duplicate article
            return None
        except Exception as e:
            logger.error(f"Error inserting article: {e}")
            return None
    
    def insert_importance_score(self, article_id: int, importance_level: int,
                               keyword_score: float, mentioned_tickers: List[str],
                               macro_events: List[str], classification_reason: str):
        """Insert importance score for an article"""
        try:
            cursor = self.conn.cursor()
            cursor.execute("""
                INSERT INTO importance_scores
                (article_id, importance_level, keyword_score, mentioned_tickers,
                 macro_events, classification_reason)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (article_id, importance_level, keyword_score,
                  json.dumps(mentioned_tickers), json.dumps(macro_events),
                  classification_reason))
            
            self.conn.commit()
            return cursor.lastrowid
        except Exception as e:
            logger.error(f"Error inserting importance score: {e}")
            return None
    
    def insert_llm_summary(self, article_id: int, summary: str,
                          affected_tickers: List[str], time_sensitivity: str,
                          actionable_insight: str, sentiment: str):
        """Insert LLM summary for an article"""
        try:
            cursor = self.conn.cursor()
            cursor.execute("""
                INSERT INTO llm_summaries
                (article_id, summary, affected_tickers, time_sensitivity,
                 actionable_insight, sentiment, processed_date)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (article_id, summary, json.dumps(affected_tickers),
                  time_sensitivity, actionable_insight, sentiment,
                  datetime.utcnow().isoformat()))
            
            self.conn.commit()
            return cursor.lastrowid
        except Exception as e:
            logger.error(f"Error inserting LLM summary: {e}")
            return None
    
    def get_articles_for_llm_processing(self, limit: int = 50) -> List[Dict]:
        """Get articles that need LLM summarization (importance >= 3)"""
        cursor = self.conn.cursor()
        cursor.execute("""
            SELECT 
                a.id, a.headline, a.raw_summary, a.full_text,
                i.importance_level, i.mentioned_tickers
            FROM news_articles a
            JOIN importance_scores i ON a.id = i.article_id
            LEFT JOIN llm_summaries l ON a.id = l.article_id
            WHERE l.id IS NULL 
              AND i.importance_level >= 3
            ORDER BY i.importance_level DESC, a.published_date DESC
            LIMIT ?
        """, (limit,))
        
        articles = []
        for row in cursor.fetchall():
            articles.append({
                'id': row['id'],
                'headline': row['headline'],
                'raw_summary': row['raw_summary'],
                'full_text': row['full_text'],
                'importance_level': row['importance_level'],
                'mentioned_tickers': json.loads(row['mentioned_tickers'] or '[]')
            })
        
        return articles
    
    def get_recent_articles_for_export(self, limit: int = 50) -> List[Dict]:
        """Get recent articles for mobile app export"""
        cursor = self.conn.cursor()
        cursor.execute("""
            SELECT 
                a.id, a.headline, a.source, a.url, 
                COALESCE(l.summary, a.raw_summary) as summary,
                a.published_date,
                i.importance_level,
                COALESCE(l.affected_tickers, i.mentioned_tickers) as tickers,
                COALESCE(l.sentiment, 'neutral') as sentiment
            FROM news_articles a
            JOIN importance_scores i ON a.id = i.article_id
            LEFT JOIN llm_summaries l ON a.id = l.article_id
            WHERE a.published_date >= datetime('now', '-3 days')
            ORDER BY i.importance_level DESC, a.published_date DESC
            LIMIT ?
        """, (limit,))
        
        articles = []
        for row in cursor.fetchall():
            # Map importance level to impact string
            importance_map = {5: 'critical', 4: 'high', 3: 'medium', 2: 'low', 1: 'noise'}
            
            articles.append({
                'headline': row['headline'],
                'source': row['source'],
                'url': row['url'],
                'summary': row['summary'] or '',
                'sentiment': row['sentiment'],
                'tickers': json.loads(row['tickers'] or '[]'),
                'type': 'financial_news',
                'impact': importance_map.get(row['importance_level'], 'medium'),
                'published_date': row['published_date']
            })
        
        return articles
    
    def update_qqq_constituents(self, constituents: List[Dict]):
        """Update QQQ constituents list"""
        try:
            cursor = self.conn.cursor()
            
            # Clear old data
            cursor.execute("DELETE FROM qqq_constituents")
            
            # Insert new constituents
            for constituent in constituents:
                cursor.execute("""
                    INSERT INTO qqq_constituents (ticker, company_name, weight, last_updated)
                    VALUES (?, ?, ?, ?)
                """, (constituent['ticker'], constituent.get('company_name', ''),
                      constituent.get('weight', 0.0), datetime.utcnow().isoformat()))
            
            self.conn.commit()
            logger.info(f"Updated {len(constituents)} QQQ constituents")
        except Exception as e:
            logger.error(f"Error updating QQQ constituents: {e}")
    
    def get_qqq_constituents(self) -> List[str]:
        """Get list of QQQ constituent tickers"""
        cursor = self.conn.cursor()
        cursor.execute("SELECT ticker FROM qqq_constituents")
        return [row['ticker'] for row in cursor.fetchall()]
    
    def archive_old_articles(self, days: int = 7):
        """Delete articles older than specified days"""
        try:
            cursor = self.conn.cursor()
            cutoff_date = (datetime.utcnow() - timedelta(days=days)).isoformat()
            
            # Delete old summaries first (foreign key constraint)
            cursor.execute("""
                DELETE FROM llm_summaries
                WHERE article_id IN (
                    SELECT id FROM news_articles 
                    WHERE published_date < ?
                )
            """, (cutoff_date,))
            
            cursor.execute("""
                DELETE FROM importance_scores
                WHERE article_id IN (
                    SELECT id FROM news_articles 
                    WHERE published_date < ?
                )
            """, (cutoff_date,))
            
            cursor.execute("""
                DELETE FROM news_articles
                WHERE published_date < ?
            """, (cutoff_date,))
            
            deleted = cursor.rowcount
            self.conn.commit()
            logger.info(f"Archived {deleted} old articles")
            return deleted
        except Exception as e:
            logger.error(f"Error archiving old articles: {e}")
            return 0
    
    def get_stats(self) -> Dict:
        """Get database statistics"""
        cursor = self.conn.cursor()
        
        stats = {}
        
        cursor.execute("SELECT COUNT(*) as count FROM news_articles")
        stats['total_articles'] = cursor.fetchone()['count']
        
        cursor.execute("""
            SELECT importance_level, COUNT(*) as count 
            FROM importance_scores 
            GROUP BY importance_level
        """)
        stats['by_importance'] = {row['importance_level']: row['count'] 
                                  for row in cursor.fetchall()}
        
        cursor.execute("""
            SELECT COUNT(*) as count FROM llm_summaries
        """)
        stats['summarized_articles'] = cursor.fetchone()['count']
        
        cursor.execute("""
            SELECT COUNT(*) as count FROM qqq_constituents
        """)
        stats['qqq_constituents'] = cursor.fetchone()['count']
        
        return stats


if __name__ == "__main__":
    import sys
    
    # Initialize database
    if len(sys.argv) > 1 and sys.argv[1] == "--init":
        print(f"Initializing database at {config.DB_PATH}")
        with NewsDatabase() as db:
            db.init_schema()
            print("Database initialized successfully!")
            print(f"Stats: {db.get_stats()}")
    else:
        print("Usage: python database.py --init")


