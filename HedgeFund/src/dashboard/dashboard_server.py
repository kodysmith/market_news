#!/usr/bin/env python3
"""
Web Dashboard for Hedge Fund Monitoring
Real-time system health, trading activity, and performance metrics
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from flask import Flask, render_template, jsonify, request
from flask_cors import CORS
import asyncio
from datetime import datetime, timedelta
from typing import Dict, Any, List
import psutil
import json
from decimal import Decimal
import logging
import os
import sqlite3

from src.common.config import load_config

# Try to import asyncpg for PostgreSQL, fall back to aiosqlite
USE_SQLITE = os.getenv('USE_SQLITE', 'true').lower() == 'true'

if not USE_SQLITE:
    try:
        import asyncpg
        DB_TYPE = 'postgres'
    except ImportError:
        import aiosqlite
        DB_TYPE = 'sqlite'
        USE_SQLITE = True
else:
    try:
        import aiosqlite
        DB_TYPE = 'sqlite'
    except ImportError:
        DB_TYPE = 'sqlite3'  # Use synchronous sqlite3

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__, template_folder='templates', static_folder='static')
CORS(app)

# Global config
config = None
db_pool = None


class DecimalEncoder(json.JSONEncoder):
    """JSON encoder that handles Decimal types"""
    def default(self, obj):
        if isinstance(obj, Decimal):
            return float(obj)
        return super().default(obj)


app.json_encoder = DecimalEncoder


async def get_db_connection():
    """Get database connection"""
    global db_pool
    
    if USE_SQLITE or DB_TYPE == 'sqlite':
        # SQLite connection
        db_path = os.getenv('SQLITE_DB_PATH', './data/hedgefund.db')
        if DB_TYPE == 'sqlite3':
            # Synchronous sqlite3
            return sqlite3.connect(db_path)
        else:
            # Async aiosqlite
            if db_pool is None:
                import aiosqlite
                db_pool = await aiosqlite.connect(db_path)
            return db_pool
    else:
        # PostgreSQL connection
        if db_pool is None:
            db_pool = await asyncpg.create_pool(
                host=config.database.postgres_host,
                port=config.database.postgres_port,
                user=config.database.postgres_user,
                password=config.database.postgres_password,
                database=config.database.postgres_database,
                min_size=2,
                max_size=10
            )
        return db_pool


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
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        status = loop.run_until_complete(get_trading_status())
        loop.close()
        return jsonify(status)
    except Exception as e:
        logger.error(f"Status error: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/performance')
def api_performance():
    """Performance metrics"""
    try:
        days = request.args.get('days', 30, type=int)
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        perf = loop.run_until_complete(get_performance_metrics(days))
        loop.close()
        return jsonify(perf)
    except Exception as e:
        logger.error(f"Performance error: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/trades/recent')
def api_recent_trades():
    """Recent trades"""
    try:
        limit = request.args.get('limit', 20, type=int)
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        trades = loop.run_until_complete(get_recent_trades(limit))
        loop.close()
        return jsonify(trades)
    except Exception as e:
        logger.error(f"Recent trades error: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/positions')
def api_positions():
    """Current positions"""
    try:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        positions = loop.run_until_complete(get_current_positions())
        loop.close()
        return jsonify(positions)
    except Exception as e:
        logger.error(f"Positions error: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/logs')
def api_logs():
    """Recent logs"""
    try:
        limit = request.args.get('limit', 50, type=int)
        level = request.args.get('level', None)
        
        logs = get_recent_logs(limit, level)
        return jsonify(logs)
    except Exception as e:
        logger.error(f"Logs error: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/nav/history')
def api_nav_history():
    """NAV history"""
    try:
        days = request.args.get('days', 30, type=int)
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        nav = loop.run_until_complete(get_nav_history(days))
        loop.close()
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
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        result = loop.run_until_complete(check_db())
        loop.close()
        return result
    except:
        return False


async def check_db() -> bool:
    """Async database check"""
    try:
        if USE_SQLITE or DB_TYPE == 'sqlite':
            db_path = os.getenv('SQLITE_DB_PATH', './data/hedgefund.db')
            if os.path.exists(db_path):
                conn = sqlite3.connect(db_path)
                conn.execute('SELECT 1')
                conn.close()
                return True
            return False
        else:
            pool = await get_db_connection()
            async with pool.acquire() as conn:
                await conn.fetchval('SELECT 1')
            return True
    except Exception as e:
        logger.error(f"Database check failed: {e}")
        return False


async def get_trading_status() -> Dict[str, Any]:
    """Get current trading status"""
    if USE_SQLITE or DB_TYPE == 'sqlite':
        # SQLite version - use synchronous for simplicity
        db_path = os.getenv('SQLITE_DB_PATH', './data/hedgefund.db')
        conn = sqlite3.connect(db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
    else:
        pool = await get_db_connection()
        async with pool.acquire() as conn:
        # Get latest NAV
        nav_row = await conn.fetchrow('''
            SELECT timestamp, nav, cash, positions_value, 
                   daily_return, ytd_return, sharpe_ratio, max_drawdown,
                   num_short_puts, num_covered_calls, num_shares, num_hedges,
                   capital_deployed, capital_deployed_pct,
                   portfolio_delta
            FROM nav_history
            ORDER BY timestamp DESC
            LIMIT 1
        ''')
        
        # Count today's trades
        today_trades = await conn.fetchval('''
            SELECT COUNT(*)
            FROM trades
            WHERE DATE(timestamp) = CURRENT_DATE
        ''')
        
        # Count open positions
        open_positions = await conn.fetchval('''
            SELECT COUNT(*)
            FROM positions
            WHERE status = 'open'
        ''')
        
        # Last trade time
        last_trade = await conn.fetchrow('''
            SELECT timestamp, asset, action, status
            FROM trades
            ORDER BY timestamp DESC
            LIMIT 1
        ''')
        
        return {
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
                'timestamp': last_trade['timestamp'].isoformat() if last_trade else None,
                'asset': last_trade['asset'] if last_trade else None,
                'action': last_trade['action'] if last_trade else None,
                'status': last_trade['status'] if last_trade else None
            } if last_trade else None,
            'last_updated': nav_row['timestamp'].isoformat() if nav_row else None
        }


async def get_performance_metrics(days: int) -> Dict[str, Any]:
    """Get performance metrics for last N days"""
    pool = await get_db_connection()
    
    async with pool.acquire() as conn:
        rows = await conn.fetch('''
            SELECT 
                DATE(timestamp) as date,
                nav,
                daily_return,
                sharpe_ratio
            FROM nav_history
            WHERE timestamp >= NOW() - INTERVAL '%s days'
            ORDER BY timestamp DESC
        ''' % days)
        
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


async def get_recent_trades(limit: int) -> List[Dict[str, Any]]:
    """Get recent trades"""
    pool = await get_db_connection()
    
    async with pool.acquire() as conn:
        rows = await conn.fetch('''
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
            LIMIT $1
        ''', limit)
        
        return [{
            'timestamp': row['timestamp'].isoformat(),
            'asset': row['asset'],
            'action': row['action'],
            'quantity': row['quantity'],
            'strike': float(row['strike'] or 0),
            'expiration': row['expiration'].isoformat() if row['expiration'] else None,
            'fill_price': float(row['fill_price'] or 0),
            'status': row['status'],
            'pnl': float(row['pnl'] or 0)
        } for row in rows]


async def get_current_positions() -> List[Dict[str, Any]]:
    """Get current open positions"""
    pool = await get_db_connection()
    
    async with pool.acquire() as conn:
        rows = await conn.fetch('''
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
        
        return [{
            'symbol': row['symbol'],
            'quantity': row['quantity'],
            'cost_basis': float(row['cost_basis'] or 0),
            'current_price': float(row['current_price'] or 0),
            'unrealized_pnl': float(row['unrealized_pnl'] or 0),
            'position_type': row['position_type'],
            'date_opened': row['date_opened'].isoformat(),
            'strike': float(row['strike'] or 0) if row['strike'] else None,
            'expiration': row['expiration'].isoformat() if row['expiration'] else None,
            'delta': float(row['delta'] or 0) if row['delta'] else None,
            'gamma': float(row['gamma'] or 0) if row['gamma'] else None,
            'theta': float(row['theta'] or 0) if row['theta'] else None
        } for row in rows]


def get_recent_logs(limit: int, level: str = None) -> List[Dict[str, Any]]:
    """Get recent logs from file"""
    try:
        log_file = os.path.join(os.path.dirname(__file__), '../../logs/daemon.log')
        
        if not os.path.exists(log_file):
            return []
        
        logs = []
        with open(log_file, 'r') as f:
            lines = f.readlines()
            
        # Parse last N lines
        for line in lines[-limit:]:
            if not line.strip():
                continue
            
            # Simple parsing (adjust based on your log format)
            try:
                parts = line.split(' - ')
                if len(parts) >= 4:
                    timestamp = parts[0]
                    log_level = parts[2]
                    message = ' - '.join(parts[3:]).strip()
                    
                    if level and log_level != level:
                        continue
                    
                    logs.append({
                        'timestamp': timestamp,
                        'level': log_level,
                        'message': message
                    })
            except:
                pass
        
        return logs[-limit:]
    except Exception as e:
        logger.error(f"Error reading logs: {e}")
        return []


async def get_nav_history(days: int) -> Dict[str, Any]:
    """Get NAV history for charting"""
    pool = await get_db_connection()
    
    async with pool.acquire() as conn:
        rows = await conn.fetch('''
            SELECT 
                timestamp,
                nav,
                cash,
                positions_value,
                daily_return
            FROM nav_history
            WHERE timestamp >= NOW() - INTERVAL '%s days'
            ORDER BY timestamp ASC
        ''' % days)
        
        return {
            'timestamps': [row['timestamp'].isoformat() for row in rows],
            'nav': [float(row['nav']) for row in rows],
            'cash': [float(row['cash']) for row in rows],
            'positions_value': [float(row['positions_value'] or 0) for row in rows],
            'daily_returns': [float(row['daily_return'] or 0) * 100 for row in rows]
        }


if __name__ == '__main__':
    # Load config
    config = load_config(environment='paper')
    
    # Run server
    port = int(os.environ.get('DASHBOARD_PORT', 5000))
    logger.info(f"🚀 Starting dashboard server on http://localhost:{port}")
    app.run(host='0.0.0.0', port=port, debug=False)

