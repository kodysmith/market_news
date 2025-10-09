#!/usr/bin/env python3
"""
Web Dashboard for Hedge Fund Monitoring (SQLite Version)
Real-time system health, trading activity, and performance metrics
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from flask import Flask, render_template, jsonify, request
from flask_cors import CORS
import sqlite3
from datetime import datetime, timedelta
from typing import Dict, Any, List
import psutil
import json
from decimal import Decimal
import logging

from src.common.config import load_config

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__, template_folder='templates', static_folder='static')
CORS(app)

# Global config
config = None
db_path = None


class DecimalEncoder(json.JSONEncoder):
    """JSON encoder that handles Decimal types"""
    def default(self, obj):
        if isinstance(obj, Decimal):
            return float(obj)
        return super().default(obj)


app.json_encoder = DecimalEncoder


def get_db():
    """Get database connection"""
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    return conn


@app.route('/')
def index():
    """Main dashboard page"""
    return render_template('dashboard.html')


@app.route('/api/health')
def api_health():
    """Health check endpoint"""
    try:
        # Check if daemon is running
        daemon_running = is_daemon_running()
        
        # Check database
        db_healthy = check_database_health()
        
        # System resources
        cpu_percent = psutil.cpu_percent(interval=1)
        memory = psutil.virtual_memory()
        disk = psutil.disk_usage('/')
        
        return jsonify({
            'status': 'healthy' if daemon_running and db_healthy else 'unhealthy',
            'daemon_running': daemon_running,
            'database_connected': db_healthy,
            'system': {
                'cpu_percent': cpu_percent,
                'memory_percent': memory.percent,
                'memory_available_gb': memory.available / (1024**3),
                'disk_percent': disk.percent,
                'disk_free_gb': disk.free / (1024**3)
            },
            'timestamp': datetime.now().isoformat()
        })
    except Exception as e:
        logger.error(f"Health check error: {e}")
        return jsonify({
            'status': 'error',
            'error': str(e)
        }), 500


@app.route('/api/status')
def api_status():
    """Current trading status"""
    try:
        status = get_trading_status()
        return jsonify(status)
    except Exception as e:
        logger.error(f"Status error: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/performance')
def api_performance():
    """Performance metrics"""
    try:
        days = request.args.get('days', 30, type=int)
        perf = get_performance_metrics(days)
        return jsonify(perf)
    except Exception as e:
        logger.error(f"Performance error: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/trades/recent')
def api_recent_trades():
    """Recent trades"""
    try:
        limit = request.args.get('limit', 20, type=int)
        trades = get_recent_trades(limit)
        return jsonify(trades)
    except Exception as e:
        logger.error(f"Recent trades error: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/positions')
def api_positions():
    """Current positions"""
    try:
        positions = get_current_positions()
        return jsonify(positions)
    except Exception as e:
        logger.error(f"Positions error: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/nav/history')
def api_nav_history():
    """NAV history"""
    try:
        days = request.args.get('days', 30, type=int)
        nav = get_nav_history(days)
        return jsonify(nav)
    except Exception as e:
        logger.error(f"NAV history error: {e}")
        return jsonify({'error': str(e)}), 500


# Helper functions

def is_daemon_running() -> bool:
    """Check if daemon is running"""
    try:
        for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
            try:
                cmdline = proc.info.get('cmdline', [])
                if cmdline and 'paper_trading_daemon.py' in ' '.join(cmdline):
                    return True
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                pass
        return False
    except Exception as e:
        logger.error(f"Error checking daemon: {e}")
        return False


def check_database_health() -> bool:
    """Check database connectivity"""
    try:
        if not os.path.exists(db_path):
            return False
        conn = get_db()
        cursor = conn.cursor()
        cursor.execute('SELECT 1')
        conn.close()
        return True
    except Exception as e:
        logger.error(f"Database check failed: {e}")
        return False


def get_trading_status() -> Dict[str, Any]:
    """Get current trading status"""
    conn = get_db()
    cursor = conn.cursor()
    
    # Get latest NAV
    cursor.execute('''
        SELECT timestamp, nav, cash, positions_value, 
               daily_return, ytd_return, sharpe_ratio, max_drawdown,
               num_short_puts, num_covered_calls, num_shares, num_hedges,
               capital_deployed, capital_deployed_pct,
               portfolio_delta
        FROM nav_history
        ORDER BY timestamp DESC
        LIMIT 1
    ''')
    nav_row = cursor.fetchone()
    
    # Count today's trades
    cursor.execute('''
        SELECT COUNT(*)
        FROM trades
        WHERE DATE(timestamp) = DATE('now')
    ''')
    today_trades = cursor.fetchone()[0]
    
    # Count open positions
    cursor.execute('''
        SELECT COUNT(*)
        FROM positions
        WHERE status = 'open'
    ''')
    open_positions = cursor.fetchone()[0]
    
    # Last trade time
    cursor.execute('''
        SELECT timestamp, asset, action, status
        FROM trades
        ORDER BY timestamp DESC
        LIMIT 1
    ''')
    last_trade = cursor.fetchone()
    
    conn.close()
    
    result = {
        'nav': float(nav_row['nav']) if nav_row else 0,
        'cash': float(nav_row['cash']) if nav_row else 0,
        'positions_value': float(nav_row['positions_value'] or 0) if nav_row else 0,
        'daily_return': float(nav_row['daily_return'] or 0) if nav_row else 0,
        'ytd_return': float(nav_row['ytd_return'] or 0) if nav_row else 0,
        'sharpe_ratio': float(nav_row['sharpe_ratio'] or 0) if nav_row else 0,
        'max_drawdown': float(nav_row['max_drawdown'] or 0) if nav_row else 0,
        'num_short_puts': nav_row['num_short_puts'] if nav_row else 0,
        'num_covered_calls': nav_row['num_covered_calls'] if nav_row else 0,
        'num_shares': nav_row['num_shares'] if nav_row else 0,
        'num_hedges': nav_row['num_hedges'] if nav_row else 0,
        'capital_deployed': float(nav_row['capital_deployed'] or 0) if nav_row else 0,
        'capital_deployed_pct': float(nav_row['capital_deployed_pct'] or 0) if nav_row else 0,
        'portfolio_delta': float(nav_row['portfolio_delta'] or 0) if nav_row else 0,
        'today_trades': today_trades,
        'open_positions': open_positions,
        'last_trade': {
            'timestamp': last_trade['timestamp'] if last_trade else None,
            'asset': last_trade['asset'] if last_trade else None,
            'action': last_trade['action'] if last_trade else None,
            'status': last_trade['status'] if last_trade else None
        } if last_trade else None,
        'last_updated': nav_row['timestamp'] if nav_row else None
    }
    
    return result


def get_performance_metrics(days: int) -> Dict[str, Any]:
    """Get performance metrics for last N days"""
    conn = get_db()
    cursor = conn.cursor()
    
    cursor.execute('''
        SELECT 
            DATE(timestamp) as date,
            nav,
            daily_return,
            sharpe_ratio
        FROM nav_history
        WHERE timestamp >= datetime('now', '-' || ? || ' days')
        ORDER BY timestamp DESC
    ''', (days,))
    
    rows = cursor.fetchall()
    conn.close()
    
    if not rows:
        return {'error': 'No data'}
    
    # Calculate metrics
    returns = [float(row['daily_return'] or 0) for row in rows if row['daily_return']]
    
    total_return = ((float(rows[0]['nav']) / float(rows[-1]['nav'])) - 1) * 100 if len(rows) > 1 else 0
    avg_daily_return = sum(returns) / len(returns) if returns else 0
    win_rate = len([r for r in returns if r > 0]) / len(returns) if returns else 0
    
    return {
        'period_days': days,
        'total_return_pct': total_return,
        'avg_daily_return_pct': avg_daily_return * 100,
        'win_rate': win_rate * 100,
        'current_sharpe': float(rows[0]['sharpe_ratio'] or 0) if rows else 0,
        'data_points': len(rows)
    }


def get_recent_trades(limit: int) -> List[Dict[str, Any]]:
    """Get recent trades"""
    conn = get_db()
    cursor = conn.cursor()
    
    cursor.execute('''
        SELECT 
            timestamp,
            asset,
            action,
            quantity,
            strike,
            expiration,
            fill_price,
            status,
            pnl
        FROM trades
        ORDER BY timestamp DESC
        LIMIT ?
    ''', (limit,))
    
    rows = cursor.fetchall()
    conn.close()
    
    return [{
        'timestamp': row['timestamp'],
        'asset': row['asset'],
        'action': row['action'],
        'quantity': row['quantity'],
        'strike': float(row['strike'] or 0),
        'expiration': row['expiration'],
        'fill_price': float(row['fill_price'] or 0),
        'status': row['status'],
        'pnl': float(row['pnl'] or 0)
    } for row in rows]


def get_current_positions() -> List[Dict[str, Any]]:
    """Get current open positions"""
    conn = get_db()
    cursor = conn.cursor()
    
    cursor.execute('''
        SELECT 
            symbol,
            quantity,
            cost_basis,
            current_price,
            unrealized_pnl,
            position_type,
            date_opened,
            strike,
            expiration,
            delta,
            gamma,
            theta
        FROM positions
        WHERE status = 'open'
        ORDER BY date_opened DESC
    ''')
    
    rows = cursor.fetchall()
    conn.close()
    
    return [{
        'symbol': row['symbol'],
        'quantity': row['quantity'],
        'cost_basis': float(row['cost_basis'] or 0),
        'current_price': float(row['current_price'] or 0),
        'unrealized_pnl': float(row['unrealized_pnl'] or 0),
        'position_type': row['position_type'],
        'date_opened': row['date_opened'],
        'strike': float(row['strike'] or 0) if row['strike'] else None,
        'expiration': row['expiration'],
        'delta': float(row['delta'] or 0) if row['delta'] else None,
        'gamma': float(row['gamma'] or 0) if row['gamma'] else None,
        'theta': float(row['theta'] or 0) if row['theta'] else None
    } for row in rows]


def get_nav_history(days: int) -> Dict[str, Any]:
    """Get NAV history for charting"""
    conn = get_db()
    cursor = conn.cursor()
    
    cursor.execute('''
        SELECT 
            timestamp,
            nav,
            cash,
            positions_value,
            daily_return
        FROM nav_history
        WHERE timestamp >= datetime('now', '-' || ? || ' days')
        ORDER BY timestamp ASC
    ''', (days,))
    
    rows = cursor.fetchall()
    conn.close()
    
    return {
        'timestamps': [row['timestamp'] for row in rows],
        'nav': [float(row['nav']) for row in rows],
        'cash': [float(row['cash']) for row in rows],
        'positions_value': [float(row['positions_value'] or 0) for row in rows],
        'daily_returns': [float(row['daily_return'] or 0) * 100 for row in rows]
    }


if __name__ == '__main__':
    # Load config
    config = load_config(environment='paper')
    
    # Set database path
    db_path = os.getenv('SQLITE_DB_PATH', './data/hedgefund.db')
    db_path = os.path.abspath(db_path)
    
    if not os.path.exists(db_path):
        logger.error(f"Database not found: {db_path}")
        logger.error("Run: ./use_sqlite.sh to create the database")
        sys.exit(1)
    
    logger.info(f"Using SQLite database: {db_path}")
    
    # Run server
    port = int(os.environ.get('DASHBOARD_PORT', 5000))
    logger.info(f"🚀 Starting dashboard server on http://localhost:{port}")
    app.run(host='0.0.0.0', port=port, debug=False)

