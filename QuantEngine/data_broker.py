#!/usr/bin/env python3
"""
Data Broker for QuantBot and Market News App

Implements a producer-consumer pattern where:
- QuantBot (producer) continuously writes market data, opportunities, and analysis to database
- Market News App (consumer) reads from database for real-time data display

Database: SQLite (easily upgradeable to PostgreSQL/MySQL)
Tables: opportunities, market_analysis, news_feed, trading_signals
"""

import sqlite3
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
from pathlib import Path
import threading
import time
import os

# Firebase imports for production
try:
    import firebase_admin
    from firebase_admin import credentials, firestore
    FIREBASE_AVAILABLE = True
except ImportError:
    FIREBASE_AVAILABLE = False
    logging.warning("Firebase not available, using SQLite only")

logger = logging.getLogger(__name__)

class QuantBotDataBroker:
    """
    Data broker that manages QuantBot data storage and retrieval
    Supports both SQLite (development) and Firestore (production)
    """

    def __init__(self, db_path: str = None):
        # Detect production mode
        self.production_mode = os.getenv('QUANT_ENV', '').lower() in ['production', 'prod', 'live']

        if self.production_mode and FIREBASE_AVAILABLE:
            # Use Firebase Firestore in production
            self.use_firestore = True
            self._initialize_firebase()
            logger.info("QuantBotDataBroker: Using Firestore (Production Mode)")
        else:
            # Use SQLite in development
            self.use_firestore = False
            if db_path is None:
                # Use QuantEngine directory as base
                quant_engine_dir = Path(__file__).parent
                self.db_path = quant_engine_dir / "quantbot_data.db"
            else:
                self.db_path = Path(db_path)
            self.lock = threading.Lock()
            self._initialize_database()
            logger.info(f"QuantBotDataBroker: Using SQLite (Development Mode) - {self.db_path}")

    def _initialize_firebase(self):
        """Initialize Firebase for production use"""
        try:
            # Check if Firebase is already initialized
            if not firebase_admin._apps:
                # Try to initialize with default credentials first
                try:
                    firebase_admin.initialize_app()
                except Exception as cred_error:
                    logger.warning(f"Default credentials not found: {cred_error}")
                    # Try to use service account key file if available
                    service_account_path = os.getenv('GOOGLE_APPLICATION_CREDENTIALS')
                    if service_account_path and os.path.exists(service_account_path):
                        cred = credentials.Certificate(service_account_path)
                        firebase_admin.initialize_app(cred)
                        logger.info("Using service account credentials")
                    else:
                        raise cred_error
            self.db = firestore.client()
            logger.info("✅ Firebase initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize Firebase: {e}")
            logger.info("Falling back to SQLite database")
            # Fallback to SQLite
            self.use_firestore = False
            self.db_path = Path(__file__).parent / "quantbot_data_fallback.db"
            self.lock = threading.Lock()
            self._initialize_database()

    def _initialize_database(self):
        """Create database tables and indexes"""
        with self.lock:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            # Trading opportunities table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS opportunities (
                    id TEXT PRIMARY KEY,
                    symbol TEXT NOT NULL,
                    strategy TEXT NOT NULL,
                    direction TEXT NOT NULL,
                    entry_price REAL,
                    expected_return REAL,
                    expected_volatility REAL,
                    confidence REAL,
                    timeframe TEXT,
                    regime TEXT,
                    signal_strength REAL,
                    risk_reward_ratio REAL,
                    rank INTEGER,
                    timestamp TEXT NOT NULL,
                    created_at REAL,
                    updated_at REAL
                )
            ''')

            # Market analysis table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS market_analysis (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    regime TEXT NOT NULL,
                    confidence REAL,
                    volatility REAL,
                    sentiment REAL,
                    economic_score REAL,
                    active_strategies INTEGER,
                    market_conditions TEXT,  -- JSON
                    timestamp TEXT NOT NULL,
                    created_at REAL
                )
            ''')

            # Enhanced news feed table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS news_feed (
                    id TEXT PRIMARY KEY,
                    headline TEXT NOT NULL,
                    source TEXT NOT NULL,
                    url TEXT,
                    summary TEXT,
                    content TEXT,
                    sentiment TEXT,
                    sentiment_score REAL,
                    tickers TEXT,  -- JSON array
                    type TEXT,
                    impact TEXT,
                    published_date TEXT,
                    created_at REAL,
                    updated_at REAL
                )
            ''')

            # Trading signals table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS trading_signals (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    symbol TEXT NOT NULL,
                    signal TEXT NOT NULL,
                    strength REAL,
                    timeframe TEXT,
                    indicators TEXT,  -- JSON
                    timestamp TEXT NOT NULL,
                    created_at REAL
                )
            ''')

            # Economic calendar table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS economic_calendar (
                    id TEXT PRIMARY KEY,
                    title TEXT NOT NULL,
                    date TEXT NOT NULL,
                    time TEXT,
                    impact TEXT,
                    actual REAL,
                    previous REAL,
                    forecast REAL,
                    currency TEXT,
                    source TEXT,
                    created_at REAL,
                    updated_at REAL
                )
            ''')

            # Performance metrics table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS performance_metrics (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    total_return REAL,
                    sharpe_ratio REAL,
                    max_drawdown REAL,
                    win_rate REAL,
                    timestamp TEXT NOT NULL,
                    created_at REAL
                )
            ''')

            # Create indexes for performance
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_opportunities_symbol ON opportunities(symbol)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_opportunities_timestamp ON opportunities(timestamp)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_news_feed_published ON news_feed(published_date)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_news_feed_type ON news_feed(type)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_signals_symbol ON trading_signals(symbol)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_calendar_date ON economic_calendar(date)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_calendar_impact ON economic_calendar(impact)')

            conn.commit()
            conn.close()

        logger.info(f"✅ Database initialized at {self.db_path}")

    # OPPORTUNITIES METHODS
    def save_opportunities(self, opportunities: List[Dict[str, Any]]):
        """Save trading opportunities to database"""
        if not opportunities:
            return

        if self.use_firestore:
            # Save to Firestore
            batch = self.db.batch()
            opportunities_ref = self.db.collection('opportunities')

            for opp in opportunities:
                doc_id = opp.get('id', f"{opp.get('symbol')}_{opp.get('timestamp')}")
                doc_ref = opportunities_ref.document(doc_id)
                opp_data = {
                    **opp,
                    'created_at': firestore.SERVER_TIMESTAMP,
                    'updated_at': firestore.SERVER_TIMESTAMP
                }
                batch.set(doc_ref, opp_data)

            batch.commit()
            logger.info(f"💾 Saved {len(opportunities)} opportunities to Firestore")
        else:
            # Save to SQLite
            with self.lock:
                conn = sqlite3.connect(self.db_path)
                cursor = conn.cursor()

                for opp in opportunities:
                    cursor.execute('''
                        INSERT OR REPLACE INTO opportunities
                        (id, symbol, strategy, direction, entry_price, expected_return,
                         expected_volatility, confidence, timeframe, regime, signal_strength,
                         risk_reward_ratio, rank, timestamp, created_at, updated_at)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ''', (
                        opp.get('id'),
                        opp.get('symbol'),
                        opp.get('strategy'),
                        opp.get('direction'),
                        opp.get('entry_price'),
                        opp.get('expected_return'),
                        opp.get('expected_volatility'),
                        opp.get('confidence'),
                        opp.get('timeframe'),
                        opp.get('regime'),
                        opp.get('signal_strength'),
                        opp.get('risk_reward_ratio'),
                        opp.get('rank'),
                        opp.get('timestamp'),
                        time.time(),
                        time.time()
                    ))

                conn.commit()
                conn.close()

            logger.info(f"💾 Saved {len(opportunities)} opportunities to SQLite")

    def get_opportunities(self, limit: int = 50, symbol: str = None) -> List[Dict[str, Any]]:
        """Retrieve trading opportunities from database"""
        if self.use_firestore:
            # Retrieve from Firestore
            opportunities_ref = self.db.collection('opportunities')

            query = opportunities_ref.order_by('timestamp', direction=firestore.Query.DESCENDING).limit(limit)

            if symbol:
                query = query.where('symbol', '==', symbol)

            docs = query.stream()
            opportunities = []
            for doc in docs:
                opp_data = doc.to_dict()
                opp_data['id'] = doc.id
                opportunities.append(opp_data)

            return opportunities
        else:
            # Retrieve from SQLite
            with self.lock:
                conn = sqlite3.connect(self.db_path)
                cursor = conn.cursor()

                if symbol:
                    cursor.execute('''
                        SELECT * FROM opportunities
                        WHERE symbol = ?
                        ORDER BY timestamp DESC
                        LIMIT ?
                    ''', (symbol, limit))
                else:
                    cursor.execute('''
                        SELECT * FROM opportunities
                        ORDER BY timestamp DESC
                        LIMIT ?
                    ''', (limit,))

                rows = cursor.fetchall()
                conn.close()

            # Convert to dict format
            opportunities = []
            for row in rows:
                opportunities.append({
                    'id': row[0],
                    'symbol': row[1],
                    'strategy': row[2],
                'direction': row[3],
                'entry_price': row[4],
                'expected_return': row[5],
                'expected_volatility': row[6],
                'confidence': row[7],
                'timeframe': row[8],
                'regime': row[9],
                'signal_strength': row[10],
                'risk_reward_ratio': row[11],
                'rank': row[12],
                'timestamp': row[13]
            })

        return opportunities

    # MARKET ANALYSIS METHODS
    def save_market_analysis(self, analysis: Dict[str, Any]):
        """Save market analysis to database"""
        with self.lock:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            cursor.execute('''
                INSERT INTO market_analysis
                (regime, confidence, volatility, sentiment, economic_score,
                 active_strategies, market_conditions, timestamp, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                analysis.get('regime'),
                analysis.get('confidence'),
                analysis.get('volatility'),
                analysis.get('sentiment'),
                analysis.get('economic_score'),
                analysis.get('active_strategies'),
                json.dumps(analysis.get('market_conditions', {})),
                analysis.get('timestamp'),
                time.time()
            ))

            conn.commit()
            conn.close()

        logger.info("💾 Saved market analysis to database")

    def get_latest_market_analysis(self) -> Optional[Dict[str, Any]]:
        """Get latest market analysis"""
        with self.lock:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            cursor.execute('''
                SELECT * FROM market_analysis
                ORDER BY created_at DESC
                LIMIT 1
            ''')

            row = cursor.fetchone()
            conn.close()

        if row:
            return {
                'regime': row[1],
                'confidence': row[2],
                'volatility': row[3],
                'sentiment': row[4],
                'economic_score': row[5],
                'active_strategies': row[6],
                'market_conditions': json.loads(row[7]) if row[7] else {},
                'timestamp': row[8]
            }

        return None

    # NEWS FEED METHODS
    def save_news_items(self, news_items: List[Dict[str, Any]]):
        """Save news items to database"""
        if not news_items:
            return

        with self.lock:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            for item in news_items:
                cursor.execute('''
                    INSERT OR REPLACE INTO news_feed
                    (id, headline, source, url, summary, content, sentiment,
                     sentiment_score, tickers, type, impact, published_date, created_at, updated_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    item.get('id', f"news_{int(time.time() * 1000)}"),
                    item.get('headline'),
                    item.get('source'),
                    item.get('url'),
                    item.get('summary'),
                    item.get('content'),
                    item.get('sentiment'),
                    item.get('sentiment_score'),
                    json.dumps(item.get('tickers', [])),
                    item.get('type'),
                    item.get('impact'),
                    item.get('published_date'),
                    time.time(),
                    time.time()
                ))

            conn.commit()
            conn.close()

        logger.info(f"💾 Saved {len(news_items)} news items to database")

    def get_news_feed(self, limit: int = 50, news_type: str = None) -> List[Dict[str, Any]]:
        """Retrieve news feed from database"""
        with self.lock:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            if news_type:
                cursor.execute('''
                    SELECT * FROM news_feed
                    WHERE type = ?
                    ORDER BY published_date DESC
                    LIMIT ?
                ''', (news_type, limit))
            else:
                cursor.execute('''
                    SELECT * FROM news_feed
                    ORDER BY published_date DESC
                    LIMIT ?
                ''', (limit,))

            rows = cursor.fetchall()
            conn.close()

        # Convert to dict format
        news_items = []
        for row in rows:
            news_items.append({
                'id': row[0],
                'headline': row[1],
                'source': row[2],
                'url': row[3],
                'summary': row[4],
                'content': row[5],
                'sentiment': row[6],
                'sentiment_score': row[7],
                'tickers': json.loads(row[8]) if row[8] else [],
                'type': row[9],
                'impact': row[10],
                'published_date': row[11]
            })

        return news_items

    # TRADING SIGNALS METHODS
    def save_trading_signals(self, signals: List[Dict[str, Any]]):
        """Save trading signals to database"""
        if not signals:
            return

        with self.lock:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            for signal in signals:
                cursor.execute('''
                    INSERT INTO trading_signals
                    (symbol, signal, strength, timeframe, indicators, timestamp, created_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                ''', (
                    signal.get('symbol'),
                    signal.get('signal'),
                    signal.get('strength'),
                    signal.get('timeframe'),
                    json.dumps(signal.get('indicators', [])),
                    signal.get('timestamp'),
                    time.time()
                ))

            conn.commit()
            conn.close()

        logger.info(f"💾 Saved {len(signals)} trading signals to database")

    def get_trading_signals(self, symbol: str = None, limit: int = 20) -> List[Dict[str, Any]]:
        """Retrieve trading signals from database"""
        with self.lock:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            if symbol:
                cursor.execute('''
                    SELECT * FROM trading_signals
                    WHERE symbol = ?
                    ORDER BY created_at DESC
                    LIMIT ?
                ''', (symbol, limit))
            else:
                cursor.execute('''
                    SELECT * FROM trading_signals
                    ORDER BY created_at DESC
                    LIMIT ?
                ''', (limit,))

            rows = cursor.fetchall()
            conn.close()

        # Convert to dict format
        signals = []
        for row in rows:
            signals.append({
                'id': row[0],
                'symbol': row[1],
                'signal': row[2],
                'strength': row[3],
                'timeframe': row[4],
                'indicators': json.loads(row[5]) if row[5] else [],
                'timestamp': row[6]
            })

        return signals

    # PERFORMANCE METHODS
    def save_performance_metrics(self, metrics: Dict[str, Any]):
        """Save performance metrics to database"""
        with self.lock:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            cursor.execute('''
                INSERT INTO performance_metrics
                (total_return, sharpe_ratio, max_drawdown, win_rate, timestamp, created_at)
                VALUES (?, ?, ?, ?, ?, ?)
            ''', (
                metrics.get('total_return'),
                metrics.get('sharpe_ratio'),
                metrics.get('max_drawdown'),
                metrics.get('win_rate'),
                metrics.get('timestamp'),
                time.time()
            ))

            conn.commit()
            conn.close()

        logger.info("💾 Saved performance metrics to database")

    def get_performance_history(self, limit: int = 100) -> List[Dict[str, Any]]:
        """Get performance history"""
        with self.lock:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            cursor.execute('''
                SELECT * FROM performance_metrics
                ORDER BY created_at DESC
                LIMIT ?
            ''', (limit,))

            rows = cursor.fetchall()
            conn.close()

        # Convert to dict format
        metrics = []
        for row in rows:
            metrics.append({
                'id': row[0],
                'total_return': row[1],
                'sharpe_ratio': row[2],
                'max_drawdown': row[3],
                'win_rate': row[4],
                'timestamp': row[5]
            })

        return metrics

    # ECONOMIC CALENDAR METHODS
    def save_economic_calendar(self, calendar_events: List[Dict[str, Any]]):
        """Save economic calendar events to database"""
        if not calendar_events:
            return

        with self.lock:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            for event in calendar_events:
                cursor.execute('''
                    INSERT OR REPLACE INTO economic_calendar
                    (id, title, date, time, impact, actual, previous, forecast, currency, source, created_at, updated_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    event.get('id'),
                    event.get('title'),
                    event.get('date'),
                    event.get('time'),
                    event.get('impact'),
                    event.get('actual'),
                    event.get('previous'),
                    event.get('forecast'),
                    event.get('currency'),
                    event.get('source'),
                    time.time(),
                    time.time()
                ))

            conn.commit()
            conn.close()

        logger.info(f"💾 Saved {len(calendar_events)} economic calendar events to database")

    def get_economic_calendar(self, limit: int = 50, impact_filter: str = None) -> List[Dict[str, Any]]:
        """Retrieve economic calendar events from database"""
        with self.lock:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            if impact_filter:
                cursor.execute('''
                    SELECT * FROM economic_calendar
                    WHERE impact = ?
                    ORDER BY date DESC, time DESC
                    LIMIT ?
                ''', (impact_filter, limit))
            else:
                cursor.execute('''
                    SELECT * FROM economic_calendar
                    ORDER BY date DESC, time DESC
                    LIMIT ?
                ''', (limit,))

            rows = cursor.fetchall()
            conn.close()

        # Convert to dict format
        events = []
        for row in rows:
            events.append({
                'id': row[0],
                'title': row[1],
                'date': row[2],
                'time': row[3],
                'impact': row[4],
                'actual': row[5],
                'previous': row[6],
                'forecast': row[7],
                'currency': row[8],
                'source': row[9]
            })

        return events

    def get_upcoming_economic_events(self, days_ahead: int = 7) -> List[Dict[str, Any]]:
        """Get upcoming economic events within specified days"""
        from datetime import datetime, timedelta

        future_date = (datetime.now() + timedelta(days=days_ahead)).strftime('%Y-%m-%d')

        with self.lock:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            cursor.execute('''
                SELECT * FROM economic_calendar
                WHERE date >= date('now')
                AND date <= ?
                ORDER BY date ASC, time ASC
            ''', (future_date,))

            rows = cursor.fetchall()
            conn.close()

        # Convert to dict format
        events = []
        for row in rows:
            events.append({
                'id': row[0],
                'title': row[1],
                'date': row[2],
                'time': row[3],
                'impact': row[4],
                'actual': row[5],
                'previous': row[6],
                'forecast': row[7],
                'currency': row[8],
                'source': row[9]
            })

        return events

    # UTILITY METHODS
    def cleanup_old_data(self, days_to_keep: int = 30):
        """Clean up old data from database"""
        cutoff_time = time.time() - (days_to_keep * 24 * 60 * 60)

        with self.lock:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            # Clean old opportunities (keep last 100)
            cursor.execute('''
                DELETE FROM opportunities
                WHERE id NOT IN (
                    SELECT id FROM opportunities
                    ORDER BY created_at DESC
                    LIMIT 100
                )
            ''')

            # Clean old news (keep last 500)
            cursor.execute('''
                DELETE FROM news_feed
                WHERE id NOT IN (
                    SELECT id FROM news_feed
                    ORDER BY created_at DESC
                    LIMIT 500
                )
            ''')

            # Clean old signals (keep last 200)
            cursor.execute('''
                DELETE FROM trading_signals
                WHERE id NOT IN (
                    SELECT id FROM trading_signals
                    ORDER BY created_at DESC
                    LIMIT 200
                )
            ''')

            conn.commit()
            conn.close()

        logger.info(f"🧹 Cleaned up old data (keeping last {days_to_keep} days)")

    def get_database_stats(self) -> Dict[str, Any]:
        """Get database statistics"""
        with self.lock:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            stats = {}

            # Count records in each table
            tables = ['opportunities', 'market_analysis', 'news_feed', 'trading_signals', 'performance_metrics']
            for table in tables:
                cursor.execute(f'SELECT COUNT(*) FROM {table}')
                stats[f'{table}_count'] = cursor.fetchone()[0]

            # Database file size
            stats['db_size_mb'] = self.db_path.stat().st_size / (1024 * 1024)

            conn.close()

        return stats

    def export_data(self, format: str = 'json', output_dir: str = 'exports'):
        """Export database data to files"""
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)

        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

        if format == 'json':
            # Export each table as JSON
            tables = {
                'opportunities': self.get_opportunities(limit=1000),
                'market_analysis': [self.get_latest_market_analysis()] if self.get_latest_market_analysis() else [],
                'news_feed': self.get_news_feed(limit=500),
                'trading_signals': self.get_trading_signals(limit=200),
                'performance_metrics': self.get_performance_history(limit=500)
            }

            for table_name, data in tables.items():
                output_file = output_path / f'{table_name}_{timestamp}.json'
                with open(output_file, 'w') as f:
                    json.dump(data, f, indent=2, default=str)
                logger.info(f"📤 Exported {len(data)} {table_name} records to {output_file}")

        logger.info(f"✅ Data export complete to {output_path}")


# Global instance
data_broker = QuantBotDataBroker()

# Convenience functions for QuantBot
def save_opportunities_to_db(opportunities: List[Dict[str, Any]]):
    """Save opportunities to database"""
    data_broker.save_opportunities(opportunities)

def save_market_analysis_to_db(analysis: Dict[str, Any]):
    """Save market analysis to database"""
    data_broker.save_market_analysis(analysis)

def save_news_to_db(news_items: List[Dict[str, Any]]):
    """Save news items to database"""
    data_broker.save_news_items(news_items)

def save_signals_to_db(signals: List[Dict[str, Any]]):
    """Save trading signals to database"""
    data_broker.save_trading_signals(signals)

def save_performance_to_db(metrics: Dict[str, Any]):
    """Save performance metrics to database"""
    data_broker.save_performance_metrics(metrics)

def save_calendar_to_db(calendar_events: List[Dict[str, Any]]):
    """Save economic calendar events to database"""
    data_broker.save_economic_calendar(calendar_events)

if __name__ == "__main__":
    # Test the data broker
    print("🧪 Testing QuantBot Data Broker...")

    # Test database initialization
    broker = QuantBotDataBroker()

    # Test saving sample data
    sample_opportunities = [
        {
            'id': 'test_opp_1',
            'symbol': 'SPY',
            'strategy': 'breakout_trading',
            'direction': 'long',
            'entry_price': 450.0,
            'expected_return': 0.023,
            'confidence': 0.72,
            'timestamp': datetime.now().isoformat()
        }
    ]

    broker.save_opportunities(sample_opportunities)
    print("✅ Saved sample opportunities")

    # Test retrieval
    opportunities = broker.get_opportunities(limit=5)
    print(f"✅ Retrieved {len(opportunities)} opportunities")

    # Test stats
    stats = broker.get_database_stats()
    print(f"📊 Database stats: {stats}")

    print("✅ Data broker test complete!")
