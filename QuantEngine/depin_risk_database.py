"""
De-Pin Risk State Database
Stores and retrieves rolling state for de-pin risk computation
"""

import sqlite3
import os
import json
import logging
from datetime import datetime
from typing import Optional, Dict, Any
from contextlib import contextmanager
try:
    from .depin_risk import RollingState
except ImportError:
    from depin_risk import RollingState

logger = logging.getLogger(__name__)

# Database path
DB_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data', 'depin_risk.db')


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
        
        # Rolling state table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS rolling_state (
                symbol TEXT PRIMARY KEY,
                prev_close REAL,
                ema8 REAL,
                ema21 REAL,
                atr14 REAL,
                tr_history TEXT,      -- JSON array
                closes_30m TEXT,      -- JSON array
                netgex_30m TEXT,      -- JSON array
                pin_scores_15m TEXT,  -- JSON array
                vol_30m TEXT,         -- JSON array
                updated_at TEXT
            )
        ''')
        
        # Risk history table (for tracking historical scores)
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS risk_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                symbol TEXT NOT NULL,
                timestamp TEXT NOT NULL,
                spot REAL NOT NULL,
                de_pin_risk INTEGER NOT NULL,
                band TEXT NOT NULL,
                dominant_strike REAL,
                put_wall REAL,
                call_wall REAL,
                atr5m REAL,
                pin_persist REAL,
                trend_strength REAL,
                move30 REAL,
                net_gex REAL,
                guidance TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # Create indexes for faster queries
        cursor.execute('''
            CREATE INDEX IF NOT EXISTS idx_risk_symbol_timestamp 
            ON risk_history(symbol, timestamp DESC)
        ''')
        
        logger.info("De-pin risk database initialized")


def load_state(symbol: str) -> RollingState:
    """
    Load rolling state for a symbol from database.
    
    Args:
        symbol: Stock/ETF symbol
    
    Returns:
        RollingState object (new if not found)
    """
    init_database()
    
    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute('SELECT * FROM rolling_state WHERE symbol = ?', (symbol.upper(),))
        row = cursor.fetchone()
        
        if row is None:
            # Return new state
            return RollingState()
        
        # Deserialize JSON arrays
        def parse_json_list(s: Optional[str]) -> list:
            if s is None or s == '':
                return []
            try:
                return json.loads(s)
            except (json.JSONDecodeError, TypeError):
                return []
        
        state = RollingState(
            prev_close=row['prev_close'],
            ema8=row['ema8'],
            ema21=row['ema21'],
            atr14=row['atr14'],
            tr_history=parse_json_list(row['tr_history']),
            closes_30m=parse_json_list(row['closes_30m']),
            netgex_30m=parse_json_list(row['netgex_30m']),
            pin_scores_15m=parse_json_list(row['pin_scores_15m']),
            vol_30m=parse_json_list(row['vol_30m'])
        )
        
        return state


def save_state(symbol: str, state: RollingState):
    """
    Save rolling state for a symbol to database.
    
    Args:
        symbol: Stock/ETF symbol
        state: RollingState object to save
    """
    init_database()
    
    def json_list(l: Optional[list]) -> str:
        if l is None:
            return '[]'
        return json.dumps(l)
    
    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute('''
            INSERT OR REPLACE INTO rolling_state 
            (symbol, prev_close, ema8, ema21, atr14, tr_history, closes_30m, 
             netgex_30m, pin_scores_15m, vol_30m, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            symbol.upper(),
            state.prev_close,
            state.ema8,
            state.ema21,
            state.atr14,
            json_list(state.tr_history),
            json_list(state.closes_30m),
            json_list(state.netgex_30m),
            json_list(state.pin_scores_15m),
            json_list(state.vol_30m),
            datetime.now().isoformat()
        ))


def save_risk_result(result):
    """
    Save a de-pin risk result to history.
    
    Args:
        result: DepinRiskResult object (from depin_risk module)
    """
    init_database()
    
    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute('''
            INSERT INTO risk_history 
            (symbol, timestamp, spot, de_pin_risk, band, dominant_strike, 
             put_wall, call_wall, atr5m, pin_persist, trend_strength, 
             move30, net_gex, guidance)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            result.symbol,
            result.ts,
            result.spot,
            result.de_pin_risk,
            result.band,
            result.dominant_strike,
            result.put_wall,
            result.call_wall,
            result.atr5m,
            result.pin_persist,
            result.trend_strength,
            result.move30,
            result.net_gex,
            result.guidance
        ))


def get_risk_history(symbol: str, limit: int = 100) -> list:
    """
    Get recent risk history for a symbol.
    
    Args:
        symbol: Stock/ETF symbol
        limit: Maximum number of records to return
    
    Returns:
        List of dicts with risk history
    """
    init_database()
    
    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute('''
            SELECT * FROM risk_history 
            WHERE symbol = ? 
            ORDER BY timestamp DESC 
            LIMIT ?
        ''', (symbol.upper(), limit))
        
        rows = cursor.fetchall()
        return [dict(row) for row in rows]


def get_latest_risk(symbol: str) -> Optional[Dict[str, Any]]:
    """
    Get the most recent risk result for a symbol.
    
    Args:
        symbol: Stock/ETF symbol
    
    Returns:
        Dict with latest risk data, or None if not found
    """
    history = get_risk_history(symbol, limit=1)
    if history:
        return history[0]
    return None


def get_risk_30m_ago(symbol: str) -> Optional[int]:
    """
    Get the de-pin risk score from approximately 30 minutes ago.
    
    This looks for a risk record that's closest to 30 minutes in the past,
    which corresponds to about 6 bars ago (5-minute bars).
    
    Args:
        symbol: Stock/ETF symbol
    
    Returns:
        Risk score (0-100) from ~30m ago, or None if not found
    """
    init_database()
    
    from datetime import datetime, timedelta
    
    # Look for records from 25-35 minutes ago (5-7 bars ago)
    with get_connection() as conn:
        cursor = conn.cursor()
        # Get records from the last 2 hours, sorted by timestamp
        cursor.execute('''
            SELECT de_pin_risk, timestamp 
            FROM risk_history 
            WHERE symbol = ? 
            ORDER BY timestamp DESC 
            LIMIT 20
        ''', (symbol.upper(),))
        
        rows = cursor.fetchall()
        if len(rows) < 7:
            # Not enough history
            return None
        
        # Return the 7th record (approximately 30-35 minutes ago for 5m bars)
        # This is a simple heuristic - in production you'd want to check actual timestamps
        if len(rows) >= 7:
            return rows[6]['de_pin_risk']
        
        return None
