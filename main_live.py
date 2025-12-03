"""
Flask HTTP Handler for Cloud Run

Provides /run endpoint for Cloud Scheduler and health check endpoint.
"""

from flask import Flask, request, jsonify
import os
import logging
from datetime import datetime
from trading.runner import run_strategy, StrategyRunner

app = Flask(__name__)
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.now().isoformat(),
        'mode': os.environ.get('TRADING_MODE', 'unknown')
    }), 200


@app.route('/run', methods=['POST'])
def run_trading_strategy():
    """
    Execute trading strategy

    Request body (optional):
    {
        "mode": "paper",  # Optional, defaults to TRADING_MODE env var
        "date": "2024-01-15"  # Optional, for backtest mode
    }
    """
    try:
        # Get mode from request or environment
        request_data = request.get_json() or {}
        mode = request_data.get('mode') or os.environ.get('TRADING_MODE', 'paper')

        if mode not in ['backtest', 'paper', 'live']:
            return jsonify({
                'error': f'Invalid mode: {mode}',
                'valid_modes': ['backtest', 'paper', 'live']
            }), 400

        logger.info(f"Starting trading run in {mode} mode")

        # Get config path
        config_path = os.environ.get('CONFIG_PATH', 'config')

        # Run strategy
        if mode == 'backtest' and 'start_date' in request_data and 'end_date' in request_data:
            results = run_strategy(
                mode=mode,
                config_path=config_path,
                start_date=request_data['start_date'],
                end_date=request_data['end_date']
            )
        else:
            # Single run
            runner = StrategyRunner(mode, config_path)
            results = runner.run()

        logger.info(f"Trading run completed successfully")

        return jsonify({
            'status': 'success',
            'mode': mode,
            'timestamp': datetime.now().isoformat(),
            'results': results
        }), 200

    except ValueError as e:
        logger.error(f"Validation error: {e}")
        return jsonify({
            'status': 'error',
            'error': str(e)
        }), 400

    except Exception as e:
        logger.error(f"Error running strategy: {e}", exc_info=True)
        return jsonify({
            'status': 'error',
            'error': str(e),
            'type': type(e).__name__
        }), 500


@app.route('/', methods=['GET'])
def index():
    """Root endpoint"""
    return jsonify({
        'service': 'Live Trading System',
        'version': '1.0.0',
        'endpoints': {
            '/health': 'GET - Health check',
            '/run': 'POST - Execute trading strategy'
        },
        'mode': os.environ.get('TRADING_MODE', 'unknown')
    }), 200


if __name__ == '__main__':
    # For local development
    port = int(os.environ.get('PORT', 8080))
    app.run(host='0.0.0.0', port=port, debug=True)
