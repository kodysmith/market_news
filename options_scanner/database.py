"""
Database schema and operations for options scanner
"""

import sqlite3
import json
from datetime import datetime
from typing import List, Dict, Any, Optional
import hashlib

class DatabaseManager:
    def __init__(self, db_path: str = "options_scanner.db"):
        self.db_path = db_path
        self.init_database()
    
    def init_database(self):
        """Initialize database with required tables"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Create pushed_alerts table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS pushed_alerts (
                id TEXT PRIMARY KEY,
                first_seen_ts TIMESTAMP NOT NULL,
                last_pushed_ts TIMESTAMP,
                last_ev REAL
            )
        ''')
        
        # Create spread_snapshots table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS spread_snapshots (
                ts TIMESTAMP NOT NULL,
                ticker TEXT NOT NULL,
                expiry DATE NOT NULL,
                short_k REAL NOT NULL,
                long_k REAL NOT NULL,
                width REAL NOT NULL,
                credit REAL NOT NULL,
                max_loss REAL NOT NULL,
                pop REAL NOT NULL,
                ev REAL NOT NULL,
                dte INTEGER NOT NULL,
                bid_ask_w REAL,
                oi_short INTEGER,
                oi_long INTEGER,
                vol_short INTEGER,
                vol_long INTEGER,
                iv_short REAL,
                iv_long REAL,
                PRIMARY KEY (ts, ticker, expiry, short_k, long_k)
            )
        ''')
        
        # Create indexes for performance
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_pushed_alerts_id ON pushed_alerts(id)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_spread_snapshots_ts ON spread_snapshots(ts)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_spread_snapshots_ticker ON spread_snapshots(ticker)')
        
        conn.commit()
        conn.close()
        
        # Store connection for in-memory databases
        if self.db_path == ":memory:":
            self._conn = sqlite3.connect(self.db_path)
            self._cursor = self._conn.cursor()
            
            # Recreate tables in the stored connection
            self._cursor.execute('''
                CREATE TABLE IF NOT EXISTS pushed_alerts (
                    id TEXT PRIMARY KEY,
                    first_seen_ts TIMESTAMP NOT NULL,
                    last_pushed_ts TIMESTAMP,
                    last_ev REAL
                )
            ''')
            
            self._cursor.execute('''
                CREATE TABLE IF NOT EXISTS spread_snapshots (
                    ts TIMESTAMP NOT NULL,
                    ticker TEXT NOT NULL,
                    expiry DATE NOT NULL,
                    short_k REAL NOT NULL,
                    long_k REAL NOT NULL,
                    width REAL NOT NULL,
                    credit REAL NOT NULL,
                    max_loss REAL NOT NULL,
                    pop REAL NOT NULL,
                    ev REAL NOT NULL,
                    dte INTEGER NOT NULL,
                    bid_ask_w REAL,
                    oi_short INTEGER,
                    oi_long INTEGER,
                    vol_short INTEGER,
                    vol_long INTEGER,
                    iv_short REAL,
                    iv_long REAL,
                    PRIMARY KEY (ts, ticker, expiry, short_k, long_k)
                )
            ''')
            
            self._conn.commit()
    
    def generate_spread_id(self, ticker: str, expiry: str, short_k: float, long_k: float, strategy: str = "BULL_PUT") -> str:
        """Generate unique ID for a spread"""
        id_string = f"{ticker}_{expiry}_{short_k}_{long_k}_{strategy}"
        return hashlib.md5(id_string.encode()).hexdigest()
    
    def get_pushed_alert(self, spread_id: str) -> Optional[Dict[str, Any]]:
        """Get pushed alert record for a spread"""
        if self.db_path == ":memory:" and hasattr(self, '_cursor'):
            cursor = self._cursor
        else:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
        
        cursor.execute('''
            SELECT id, first_seen_ts, last_pushed_ts, last_ev
            FROM pushed_alerts
            WHERE id = ?
        ''', (spread_id,))
        
        row = cursor.fetchone()
        
        if self.db_path != ":memory:":
            conn.close()
        
        if row:
            return {
                'id': row[0],
                'first_seen_ts': row[1],
                'last_pushed_ts': row[2],
                'last_ev': row[3]
            }
        return None
    
    def upsert_pushed_alert(self, spread_id: str, ev: float):
        """Insert or update pushed alert record"""
        if self.db_path == ":memory:" and hasattr(self, '_cursor'):
            cursor = self._cursor
            conn = self._conn
        else:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
        
        now = datetime.now().isoformat()
        
        # Check if exists
        existing = self.get_pushed_alert(spread_id)
        
        if existing:
            # Update existing record
            cursor.execute('''
                UPDATE pushed_alerts
                SET last_pushed_ts = ?, last_ev = ?
                WHERE id = ?
            ''', (now, ev, spread_id))
        else:
            # Insert new record
            cursor.execute('''
                INSERT INTO pushed_alerts (id, first_seen_ts, last_pushed_ts, last_ev)
                VALUES (?, ?, ?, ?)
            ''', (spread_id, now, now, ev))
        
        conn.commit()
        
        if self.db_path != ":memory:":
            conn.close()
    
    def should_send_alert(self, spread_id: str, current_ev: float) -> bool:
        """Determine if alert should be sent based on dedupe rules"""
        existing = self.get_pushed_alert(spread_id)
        
        if not existing:
            # New spread - always alert
            return True
        
        # Check if EV improved by >= 10%
        last_ev = existing['last_ev'] or 0
        ev_improvement = (current_ev - last_ev) / max(last_ev, 0.01)  # Avoid division by zero
        
        if ev_improvement >= 0.10:
            return True
        
        # Check if 60+ minutes since last push
        if existing['last_pushed_ts']:
            last_push = datetime.fromisoformat(existing['last_pushed_ts'])
            time_diff = datetime.now() - last_push
            if time_diff.total_seconds() >= 3600:  # 60 minutes
                return True
        
        return False
    
    def persist_spread_snapshot(self, spread_data: Dict[str, Any]):
        """Persist spread snapshot to database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT OR REPLACE INTO spread_snapshots
            (ts, ticker, expiry, short_k, long_k, width, credit, max_loss, pop, ev, dte,
             bid_ask_w, oi_short, oi_long, vol_short, vol_long, iv_short, iv_long)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            spread_data['timestamp'],
            spread_data['ticker'],
            spread_data['expiry'],
            spread_data['short_k'],
            spread_data['long_k'],
            spread_data['width'],
            spread_data['credit'],
            spread_data['max_loss'],
            spread_data['pop'],
            spread_data['ev'],
            spread_data['dte'],
            spread_data.get('bid_ask_w'),
            spread_data.get('oi_short'),
            spread_data.get('oi_long'),
            spread_data.get('vol_short'),
            spread_data.get('vol_long'),
            spread_data.get('iv_short'),
            spread_data.get('iv_long')
        ))
        
        conn.commit()
        conn.close()
    
    def get_recent_snapshots(self, hours: int = 24) -> List[Dict[str, Any]]:
        """Get recent spread snapshots for analysis"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cutoff_time = datetime.now().timestamp() - (hours * 3600)
        
        cursor.execute('''
            SELECT * FROM spread_snapshots
            WHERE ts > ?
            ORDER BY ts DESC
        ''', (cutoff_time,))
        
        columns = [description[0] for description in cursor.description]
        rows = cursor.fetchall()
        
        conn.close()
        
        return [dict(zip(columns, row)) for row in rows]
