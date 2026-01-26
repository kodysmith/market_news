#!/usr/bin/env python3
"""
Crypto Arbitrage Dashboard
Real-time monitoring of arbitrage opportunities and executions
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from flask import Flask, render_template, jsonify
from flask_cors import CORS
import sqlite3
from datetime import datetime
import psutil
import logging

from ..common.config import load_config
from ..database.repository import ArbitrageRepository

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__, template_folder='templates')
CORS(app)

# Global state
config = None
repository = None


@app.route('/')
def index():
    """Main dashboard page"""
    return render_template('arbitrage_dashboard.html')


@app.route('/api/health')
def api_health():
    """System health"""
    try:
        daemon_running = is_daemon_running()
        
        cpu = psutil.cpu_percent(interval=1)
        memory = psutil.virtual_memory()
        
        return jsonify({
            'status': 'healthy' if daemon_running else 'degraded',
            'daemon_running': daemon_running,
            'system': {
                'cpu_percent': cpu,
                'memory_percent': memory.percent,
                'memory_available_gb': memory.available / (1024**3)
            },
            'timestamp': datetime.now().isoformat()
        })
    except Exception as e:
        return jsonify({'status': 'error', 'error': str(e)}), 500


@app.route('/api/opportunities/live')
def api_live_opportunities():
    """Get recent opportunities"""
    try:
        opportunities = repository.get_recent_opportunities(limit=20)
        return jsonify(opportunities)
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/executions/recent')
def api_recent_executions():
    """Get recent executions"""
    try:
        executions = repository.get_recent_executions(limit=20)
        return jsonify(executions)
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/performance/current')
def api_current_performance():
    """Get current performance metrics"""
    try:
        conn = sqlite3.connect(repository.db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        
        # Today's stats
        cursor.execute('''
            SELECT 
                COUNT(*) as total_executions,
                SUM(CASE WHEN is_successful = 1 THEN 1 ELSE 0 END) as successful,
                SUM(realized_profit_usd) as total_profit,
                AVG(execution_time_ms) as avg_execution_time,
                AVG(realized_spread_pct) as avg_spread
            FROM executions
            WHERE DATE(started_at) = DATE('now')
        ''')
        
        today = cursor.fetchone()
        
        # Total opportunities today
        cursor.execute('''
            SELECT COUNT(*) as count
            FROM opportunities
            WHERE DATE(detected_at) = DATE('now')
        ''')
        
        opp_count = cursor.fetchone()['count']
        
        # All-time profit
        cursor.execute('''
            SELECT SUM(realized_profit_usd) as total
            FROM executions
            WHERE is_successful = 1
        ''')
        
        all_time = cursor.fetchone()
        
        conn.close()
        
        return jsonify({
            'today': {
                'opportunities_detected': opp_count,
                'trades_executed': today['total_executions'] or 0,
                'successful_trades': today['successful'] or 0,
                'total_profit': float(today['total_profit'] or 0),
                'avg_execution_time_ms': float(today['avg_execution_time'] or 0),
                'avg_spread_pct': float(today['avg_spread'] or 0)
            },
            'all_time': {
                'total_profit': float(all_time['total'] or 0)
            }
        })
    except Exception as e:
        logger.error(f"Performance error: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/pnl/history')
def api_pnl_history():
    """Get P&L history"""
    try:
        days = int(request.args.get('days', 30))
        pnl_data = repository.get_daily_pnl(days)
        return jsonify(pnl_data)
    except Exception as e:
        return jsonify({'error': str(e)}), 500


def is_daemon_running() -> bool:
    """Check if daemon is running"""
    try:
        for proc in psutil.process_iter(['cmdline']):
            cmdline = proc.info.get('cmdline', [])
            if cmdline and 'arbitrage_daemon.py' in ' '.join(cmdline):
                return True
        return False
    except:
        return False


if __name__ == '__main__':
    config = load_config()
    repository = ArbitrageRepository(config.database.sqlite_path)
    
    port = config.monitoring.dashboard_port
    logger.info(f"🚀 Starting arbitrage dashboard on http://localhost:{port}")
    
    app.run(host='0.0.0.0', port=port, debug=False)


