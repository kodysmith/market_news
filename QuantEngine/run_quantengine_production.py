#!/usr/bin/env python3
"""
Production runner for QuantEngine

This script sets up the production environment and runs the QuantEngine
with production database connectivity.
"""

import os
import sys
import logging
from pathlib import Path

# Set production environment
os.environ['QUANT_ENV'] = 'production'

# Add QuantEngine to path
quant_engine_dir = Path(__file__).parent
sys.path.insert(0, str(quant_engine_dir))

# Configure logging for production
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('quant_engine_production.log'),
        logging.StreamHandler(sys.stdout)
    ]
)

logger = logging.getLogger(__name__)

def main():
    """Run QuantEngine in production mode"""
    logger.info("🚀 Starting QuantEngine in PRODUCTION mode")
    logger.info("📊 Data will be sent to Firestore production database")

    try:
        # Import and run the main QuantEngine with cycle command
        from main import main as quant_main
        import sys
        
        # Set up command line arguments for production
        original_argv = sys.argv
        sys.argv = ['run_quantengine_production.py', 'cycle']
        
        try:
            quant_main()
        finally:
            # Restore original argv
            sys.argv = original_argv

    except KeyboardInterrupt:
        logger.info("🛑 QuantEngine production run interrupted by user")
    except Exception as e:
        logger.error(f"❌ QuantEngine production run failed: {e}")
        raise
    finally:
        logger.info("🏁 QuantEngine production run completed")

if __name__ == "__main__":
    main()
