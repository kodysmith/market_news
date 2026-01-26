"""
Database Repository
Data access layer for crypto arbitrage system
"""

import sqlite3
import json
from typing import List, Dict, Optional
from datetime import datetime, date
from decimal import Decimal
import logging
from pathlib import Path

from ..common.models import ArbitrageOpportunity, ArbitrageExecution

logger = logging.getLogger(__name__)


class ArbitrageRepository:
    """Data access layer for arbitrage system"""
    
    def __init__(self, db_path: str):
        """
        Initialize repository
        
        Args:
            db_path: Path to SQLite database
        """
        self.db_path = db_path
        self._ensure_database_exists()
        logger.info(f"ArbitrageRepository initialized: {db_path}")
    
    def _ensure_database_exists(self):
        """Create database and tables if they don't exist"""
        db_file = Path(self.db_path)
        db_file.parent.mkdir(parents=True, exist_ok=True)
        
        # Read schema
        schema_path = Path(__file__).parent / 'schema.sql'
        
        if not schema_path.exists():
            logger.warning("Schema file not found, skipping table creation")
            return
        
        with open(schema_path, 'r') as f:
            schema_sql = f.read()
        
        # Create tables
        conn = sqlite3.connect(self.db_path)
        conn.executescript(schema_sql)
        conn.commit()
        conn.close()
        
        logger.info("✅ Database initialized")
    
    def save_opportunity(self, opportunity: ArbitrageOpportunity):
        """Save arbitrage opportunity to database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT OR REPLACE INTO opportunities (
                id, detected_at, pair, base_asset, quote_asset,
                buy_exchange, buy_price, buy_fee_pct,
                sell_exchange, sell_price, sell_fee_pct,
                gross_spread_pct, fees_total_pct, net_spread_pct,
                max_quantity, estimated_profit_usd,
                price_staleness_ms, estimated_slippage_pct, confidence_score,
                first_seen_at, last_seen_at, expired_at, opportunity_duration_ms, times_seen,
                is_executed, execution_id
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            opportunity.id,
            opportunity.detected_at,
            opportunity.pair,
            opportunity.base_asset,
            opportunity.quote_asset,
            opportunity.buy_exchange,
            float(opportunity.buy_price),
            float(opportunity.buy_fee_pct),
            opportunity.sell_exchange,
            float(opportunity.sell_price),
            float(opportunity.sell_fee_pct),
            float(opportunity.gross_spread_pct),
            float(opportunity.fees_total_pct),
            float(opportunity.net_spread_pct),
            float(opportunity.max_quantity),
            float(opportunity.estimated_profit_usd),
            opportunity.price_staleness_ms,
            float(opportunity.estimated_slippage_pct),
            opportunity.confidence_score,
            opportunity.first_seen_at,
            opportunity.last_seen_at,
            opportunity.expired_at,
            opportunity.opportunity_duration_ms,
            opportunity.times_seen,
            opportunity.is_executed,
            opportunity.execution_id
        ))
        
        conn.commit()
        conn.close()
    
    def save_execution(self, execution: ArbitrageExecution):
        """Save arbitrage execution to database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO executions (
                id, opportunity_id, started_at, completed_at,
                pair, quantity,
                buy_exchange, buy_order_id, buy_status,
                buy_filled_price, buy_filled_quantity, buy_fee,
                sell_exchange, sell_order_id, sell_status,
                sell_filled_price, sell_filled_quantity, sell_fee,
                is_successful, realized_profit_usd, realized_spread_pct,
                execution_time_ms, had_failures, failure_reason, required_reversal
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            execution.id,
            execution.opportunity_id,
            execution.started_at,
            execution.completed_at,
            execution.pair,
            float(execution.quantity),
            execution.buy_exchange,
            execution.buy_order_id,
            execution.buy_status.value,
            float(execution.buy_filled_price) if execution.buy_filled_price else None,
            float(execution.buy_filled_quantity),
            float(execution.buy_fee),
            execution.sell_exchange,
            execution.sell_order_id,
            execution.sell_status.value,
            float(execution.sell_filled_price) if execution.sell_filled_price else None,
            float(execution.sell_filled_quantity),
            float(execution.sell_fee),
            execution.is_successful,
            float(execution.realized_profit_usd) if execution.realized_profit_usd else None,
            float(execution.realized_spread_pct) if execution.realized_spread_pct else None,
            execution.execution_time_ms,
            execution.had_failures,
            execution.failure_reason,
            execution.required_reversal
        ))
        
        conn.commit()
        conn.close()
        
        # Update opportunity as executed
        if execution.is_successful:
            self._mark_opportunity_executed(execution.opportunity_id, execution.id)
    
    def _mark_opportunity_executed(self, opportunity_id: str, execution_id: str):
        """Mark opportunity as executed"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            UPDATE opportunities
            SET is_executed = 1, execution_id = ?
            WHERE id = ?
        ''', (execution_id, opportunity_id))
        
        conn.commit()
        conn.close()
    
    def get_recent_opportunities(self, limit: int = 20) -> List[Dict]:
        """Get recent opportunities"""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT * FROM opportunities
            ORDER BY detected_at DESC
            LIMIT ?
        ''', (limit,))
        
        rows = cursor.fetchall()
        conn.close()
        
        return [dict(row) for row in rows]
    
    def get_recent_executions(self, limit: int = 20) -> List[Dict]:
        """Get recent executions"""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT * FROM executions
            ORDER BY started_at DESC
            LIMIT ?
        ''', (limit,))
        
        rows = cursor.fetchall()
        conn.close()
        
        return [dict(row) for row in rows]
    
    def get_daily_pnl(self, days: int = 30) -> List[Dict]:
        """Get daily P&L for last N days"""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT 
                DATE(started_at) as date,
                COUNT(*) as num_trades,
                SUM(CASE WHEN is_successful = 1 THEN 1 ELSE 0 END) as successful_trades,
                SUM(realized_profit_usd) as total_profit,
                AVG(realized_spread_pct) as avg_spread,
                AVG(execution_time_ms) as avg_execution_time
            FROM executions
            WHERE started_at >= DATE('now', '-' || ? || ' days')
            GROUP BY DATE(started_at)
            ORDER BY date DESC
        ''', (days,))
        
        rows = cursor.fetchall()
        conn.close()
        
        return [dict(row) for row in rows]
    
    def get_total_pnl(self) -> Decimal:
        """Get total realized profit"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT SUM(realized_profit_usd)
            FROM executions
            WHERE is_successful = 1
        ''')
        
        result = cursor.fetchone()[0]
        conn.close()
        
        return Decimal(str(result)) if result else Decimal('0')

