"""
Intrinsic Value Historical Database
Stores and retrieves historical intrinsic value calculations for divergence tracking
"""

import sqlite3
import os
import logging
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional
from contextlib import contextmanager

logger = logging.getLogger(__name__)

# Database path
DB_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data', 'intrinsic_values.db')


def get_db_path() -> str:
    """Get the database path, creating directory if needed"""
    db_dir = os.path.dirname(DB_PATH)
    if not os.path.exists(db_dir):
        os.makedirs(db_dir)
    return DB_PATH


@contextmanager
def get_connection():
    """Context manager for database connections"""
    conn = sqlite3.connect(get_db_path())
    conn.row_factory = sqlite3.Row
    try:
        yield conn
        conn.commit()
    except Exception as e:
        conn.rollback()
        raise e
    finally:
        conn.close()


def init_database():
    """Initialize the database with required tables"""
    with get_connection() as conn:
        cursor = conn.cursor()
        
        # Main historical data table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS intrinsic_value_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                ticker TEXT NOT NULL,
                date DATE NOT NULL,
                price REAL NOT NULL,
                dcf_value REAL,
                dcf_confidence REAL,
                graham_value REAL,
                graham_confidence REAL,
                ddm_value REAL,
                ddm_confidence REAL,
                epv_value REAL,
                epv_confidence REAL,
                comps_value REAL,
                comps_confidence REAL,
                nav_value REAL,
                nav_confidence REAL,
                composite_value REAL,
                divergence_pct REAL,
                verdict TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(ticker, date)
            )
        ''')
        
        # Create index for faster queries
        cursor.execute('''
            CREATE INDEX IF NOT EXISTS idx_ticker_date 
            ON intrinsic_value_history(ticker, date DESC)
        ''')
        
        # Watchlist table for tracking user's stocks
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS watchlist (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                ticker TEXT UNIQUE NOT NULL,
                company_name TEXT,
                sector TEXT,
                added_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                last_calculated TIMESTAMP,
                alert_threshold_pct REAL DEFAULT 20.0
            )
        ''')
        
        # Alerts table for storing triggered alerts
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS valuation_alerts (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                ticker TEXT NOT NULL,
                alert_type TEXT NOT NULL,
                divergence_pct REAL NOT NULL,
                price REAL NOT NULL,
                composite_value REAL NOT NULL,
                triggered_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                acknowledged INTEGER DEFAULT 0
            )
        ''')
        
        logger.info("Database initialized successfully")


def store_valuation(result: Dict[str, Any]) -> bool:
    """
    Store a valuation result in the database.
    
    Args:
        result: Valuation result from IntrinsicValueCalculator.calculate_all()
        
    Returns:
        True if stored successfully, False otherwise
    """
    try:
        ticker = result.get('ticker')
        if not ticker:
            return False
        
        valuations = result.get('valuations', {})
        composite = result.get('composite', {})
        
        with get_connection() as conn:
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT OR REPLACE INTO intrinsic_value_history (
                    ticker, date, price,
                    dcf_value, dcf_confidence,
                    graham_value, graham_confidence,
                    ddm_value, ddm_confidence,
                    epv_value, epv_confidence,
                    comps_value, comps_confidence,
                    nav_value, nav_confidence,
                    composite_value, divergence_pct, verdict
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                ticker,
                datetime.now().date().isoformat(),
                result.get('current_price'),
                valuations.get('dcf', {}).get('value'),
                valuations.get('dcf', {}).get('confidence'),
                valuations.get('graham', {}).get('value'),
                valuations.get('graham', {}).get('confidence'),
                valuations.get('ddm', {}).get('value'),
                valuations.get('ddm', {}).get('confidence'),
                valuations.get('epv', {}).get('value'),
                valuations.get('epv', {}).get('confidence'),
                valuations.get('peer_comps', {}).get('value'),
                valuations.get('peer_comps', {}).get('confidence'),
                valuations.get('nav', {}).get('value'),
                valuations.get('nav', {}).get('confidence'),
                composite.get('value'),
                composite.get('divergence_pct'),
                composite.get('verdict')
            ))
            
            # Update watchlist last_calculated if ticker is in watchlist
            cursor.execute('''
                UPDATE watchlist SET last_calculated = CURRENT_TIMESTAMP
                WHERE ticker = ?
            ''', (ticker,))
            
        return True
        
    except Exception as e:
        logger.error(f"Failed to store valuation: {e}")
        return False


def get_history(ticker: str, days: int = 365) -> List[Dict[str, Any]]:
    """
    Get historical valuation data for a ticker.
    
    Args:
        ticker: Stock ticker symbol
        days: Number of days of history to retrieve
        
    Returns:
        List of historical valuation records
    """
    try:
        start_date = (datetime.now() - timedelta(days=days)).date().isoformat()
        
        with get_connection() as conn:
            cursor = conn.cursor()
            
            cursor.execute('''
                SELECT * FROM intrinsic_value_history
                WHERE ticker = ? AND date >= ?
                ORDER BY date ASC
            ''', (ticker.upper(), start_date))
            
            rows = cursor.fetchall()
            
            return [dict(row) for row in rows]
            
    except Exception as e:
        logger.error(f"Failed to get history: {e}")
        return []


def get_latest_valuation(ticker: str) -> Optional[Dict[str, Any]]:
    """
    Get the most recent valuation for a ticker.
    
    Args:
        ticker: Stock ticker symbol
        
    Returns:
        Latest valuation record or None
    """
    try:
        with get_connection() as conn:
            cursor = conn.cursor()
            
            cursor.execute('''
                SELECT * FROM intrinsic_value_history
                WHERE ticker = ?
                ORDER BY date DESC
                LIMIT 1
            ''', (ticker.upper(),))
            
            row = cursor.fetchone()
            return dict(row) if row else None
            
    except Exception as e:
        logger.error(f"Failed to get latest valuation: {e}")
        return None


def add_to_watchlist(ticker: str, company_name: str = None, sector: str = None,
                     alert_threshold: float = 20.0) -> bool:
    """
    Add a ticker to the watchlist.
    
    Args:
        ticker: Stock ticker symbol
        company_name: Company name (optional)
        sector: Sector (optional)
        alert_threshold: Divergence threshold for alerts (default 20%)
        
    Returns:
        True if added successfully
    """
    try:
        with get_connection() as conn:
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT OR REPLACE INTO watchlist (ticker, company_name, sector, alert_threshold_pct)
                VALUES (?, ?, ?, ?)
            ''', (ticker.upper(), company_name, sector, alert_threshold))
            
        return True
        
    except Exception as e:
        logger.error(f"Failed to add to watchlist: {e}")
        return False


def remove_from_watchlist(ticker: str) -> bool:
    """Remove a ticker from the watchlist"""
    try:
        with get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('DELETE FROM watchlist WHERE ticker = ?', (ticker.upper(),))
        return True
    except Exception as e:
        logger.error(f"Failed to remove from watchlist: {e}")
        return False


def get_watchlist() -> List[Dict[str, Any]]:
    """Get all tickers in the watchlist with their latest valuations"""
    try:
        with get_connection() as conn:
            cursor = conn.cursor()
            
            cursor.execute('''
                SELECT w.*, h.price, h.composite_value, h.divergence_pct, h.verdict, h.date
                FROM watchlist w
                LEFT JOIN (
                    SELECT ticker, price, composite_value, divergence_pct, verdict, date,
                           ROW_NUMBER() OVER (PARTITION BY ticker ORDER BY date DESC) as rn
                    FROM intrinsic_value_history
                ) h ON w.ticker = h.ticker AND h.rn = 1
                ORDER BY w.added_at DESC
            ''')
            
            rows = cursor.fetchall()
            return [dict(row) for row in rows]
            
    except Exception as e:
        logger.error(f"Failed to get watchlist: {e}")
        return []


def check_alerts(result: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Check if a valuation result triggers any alerts.
    
    Args:
        result: Valuation result
        
    Returns:
        List of triggered alerts
    """
    alerts = []
    ticker = result.get('ticker')
    composite = result.get('composite', {})
    divergence = composite.get('divergence_pct')
    
    if not ticker or divergence is None:
        return alerts
    
    try:
        with get_connection() as conn:
            cursor = conn.cursor()
            
            # Check if ticker is in watchlist and get threshold
            cursor.execute('''
                SELECT alert_threshold_pct FROM watchlist WHERE ticker = ?
            ''', (ticker.upper(),))
            
            row = cursor.fetchone()
            if not row:
                return alerts
            
            threshold = row['alert_threshold_pct']
            
            # Check for undervalued alert
            if divergence < -threshold:
                alert = {
                    'ticker': ticker,
                    'alert_type': 'undervalued',
                    'divergence_pct': divergence,
                    'price': result.get('current_price'),
                    'composite_value': composite.get('value'),
                    'message': f"{ticker} is {abs(divergence):.1f}% below intrinsic value"
                }
                alerts.append(alert)
                
                # Store alert
                cursor.execute('''
                    INSERT INTO valuation_alerts (ticker, alert_type, divergence_pct, price, composite_value)
                    VALUES (?, ?, ?, ?, ?)
                ''', (ticker, 'undervalued', divergence, result.get('current_price'), composite.get('value')))
            
            # Check for overvalued alert
            elif divergence > threshold:
                alert = {
                    'ticker': ticker,
                    'alert_type': 'overvalued',
                    'divergence_pct': divergence,
                    'price': result.get('current_price'),
                    'composite_value': composite.get('value'),
                    'message': f"{ticker} is {divergence:.1f}% above intrinsic value"
                }
                alerts.append(alert)
                
                # Store alert
                cursor.execute('''
                    INSERT INTO valuation_alerts (ticker, alert_type, divergence_pct, price, composite_value)
                    VALUES (?, ?, ?, ?, ?)
                ''', (ticker, 'overvalued', divergence, result.get('current_price'), composite.get('value')))
            
    except Exception as e:
        logger.error(f"Failed to check alerts: {e}")
    
    return alerts


def get_recent_alerts(limit: int = 50) -> List[Dict[str, Any]]:
    """Get recent valuation alerts"""
    try:
        with get_connection() as conn:
            cursor = conn.cursor()
            
            cursor.execute('''
                SELECT * FROM valuation_alerts
                ORDER BY triggered_at DESC
                LIMIT ?
            ''', (limit,))
            
            rows = cursor.fetchall()
            return [dict(row) for row in rows]
            
    except Exception as e:
        logger.error(f"Failed to get alerts: {e}")
        return []


def acknowledge_alert(alert_id: int) -> bool:
    """Mark an alert as acknowledged"""
    try:
        with get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('''
                UPDATE valuation_alerts SET acknowledged = 1 WHERE id = ?
            ''', (alert_id,))
        return True
    except Exception as e:
        logger.error(f"Failed to acknowledge alert: {e}")
        return False


def get_divergence_scan(threshold: float = 20.0) -> List[Dict[str, Any]]:
    """
    Get all tickers where divergence exceeds threshold.
    
    Args:
        threshold: Divergence threshold percentage
        
    Returns:
        List of tickers with significant divergence
    """
    try:
        with get_connection() as conn:
            cursor = conn.cursor()
            
            cursor.execute('''
                SELECT h.*, w.company_name, w.sector
                FROM intrinsic_value_history h
                JOIN (
                    SELECT ticker, MAX(date) as max_date
                    FROM intrinsic_value_history
                    GROUP BY ticker
                ) latest ON h.ticker = latest.ticker AND h.date = latest.max_date
                LEFT JOIN watchlist w ON h.ticker = w.ticker
                WHERE ABS(h.divergence_pct) >= ?
                ORDER BY h.divergence_pct ASC
            ''', (threshold,))
            
            rows = cursor.fetchall()
            return [dict(row) for row in rows]
            
    except Exception as e:
        logger.error(f"Failed to get divergence scan: {e}")
        return []


# Initialize database on module import
init_database()
