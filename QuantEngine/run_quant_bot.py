#!/usr/bin/env python3
"""
QuantBot Runner Script

Starts the autonomous quantitative trading bot that runs 24/7 to:
- Monitor market data continuously
- Detect market regimes in real-time
- Research and evolve trading strategies
- Discover profitable opportunities
- Adapt to changing market conditions

Usage:
    python3 run_quant_bot.py          # Run continuously
    python3 run_quant_bot.py --demo   # Run demo mode (limited time)
    python3 run_quant_bot.py --status # Show current bot status
"""

import sys
import asyncio
import argparse
import signal
import json
from pathlib import Path
from datetime import datetime
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - 🤖 QuantBot - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('quant_bot.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger('QuantBot')

# Add QuantEngine to path
sys.path.insert(0, str(Path(__file__).parent))

from quant_bot import QuantBot


class QuantBotRunner:
    """Manages QuantBot execution with proper signal handling"""

    def __init__(self):
        self.bot = None
        self.running = False

    async def start_bot(self, demo_mode: bool = False):
        """Start the QuantBot"""

        logger.info("🚀 Starting QuantEngine Autonomous Trading System")
        logger.info("=" * 60)

        try:
            # Initialize bot
            self.bot = QuantBot()
            self.running = True

            if demo_mode:
                logger.info("🎯 Running in DEMO MODE (limited functionality)")
                # Initialize components for demo
                await self.bot.initialize_components()
                # Initialize data broker for demo
                from data_broker import data_broker
                self.bot.data_broker = data_broker
                # Run for limited time in demo mode
                await asyncio.wait_for(self.run_demo(self.bot), timeout=300)  # 5 minutes
            else:
                logger.info("🔄 Starting 24/7 autonomous operation")
                await self.bot.start()

        except KeyboardInterrupt:
            logger.info("🛑 Shutdown requested by user")
        except asyncio.TimeoutError:
            logger.info("⏰ Demo mode completed")
        except Exception as e:
            logger.error(f"❌ Fatal error: {e}")
            raise
        finally:
            await self.shutdown()

    async def run_demo(self, bot):
        """Run bot in demo mode with simulated data"""

        logger.info("🎭 Running QuantBot demo simulation...")

        # Simulate bot operation for demo purposes
        demo_steps = [
            ("🔧 Initializing components", 2),
            ("📊 Connecting to data feeds", 3),
            ("🎭 Detecting market regime", 2),
            ("🧪 Researching strategies", 3),
            ("🔍 Scanning opportunities", 2),
            ("📈 Generating reports", 1),
        ]

        for step, duration in demo_steps:
            logger.info(f"   {step}...")
            await asyncio.sleep(duration)

            # Simulate some output and trigger actual functionality
            if "regime" in step.lower():
                bot.current_regime = "bull_market"
                logger.info("   ✅ Detected: Bull Market (confidence: 78%)")
            elif "strategies" in step.lower():
                # Create some demo strategies
                bot.strategy_portfolio = {
                    'demo_strategy_1': {'name': 'SMA_Crossover', 'performance': {'sharpe_ratio': 1.4}},
                    'demo_strategy_2': {'name': 'RSI_Reversal', 'performance': {'sharpe_ratio': 1.6}},
                    'demo_strategy_3': {'name': 'Volume_Breakout', 'performance': {'sharpe_ratio': 1.3}}
                }
                logger.info("   ✅ Generated 3 new strategies (avg Sharpe: 1.45)")
            elif "opportunities" in step.lower():
                # Actually scan for opportunities
                await bot.scan_opportunities()
                logger.info(f"   ✅ Found {len(bot.opportunity_cache)} trading opportunities (avg EV: 1.8%)")

        logger.info("🎉 Demo simulation complete!")
        logger.info("")
        logger.info("In full production mode, QuantBot would:")
        logger.info("• Run continuously 24/7")
        logger.info("• Process real market data and news")
        logger.info("• Adapt strategies to live market conditions")
        logger.info("• Execute trades through connected brokers")
        logger.info("• Learn from performance and improve over time")

    async def show_status(self):
        """Show current bot status"""

        logger.info("📊 QuantBot Status Report")
        logger.info("=" * 40)

        # Check if bot is running
        status_file = Path("bot_status.json")
        if status_file.exists():
            try:
                with open(status_file, 'r') as f:
                    status = json.load(f)

                logger.info(f"🤖 Bot Status: {'Running' if self.running else 'Stopped'}")
                logger.info(f"🎭 Current Regime: {status.get('regime', 'Unknown')}")
                logger.info(f"📈 Active Strategies: {status.get('active_strategies', 0)}")
                logger.info(f"🎯 Opportunities Found: {status.get('opportunities_found', 0)}")
                logger.info(f"📊 Performance: {status.get('total_performance', {}).get('total_return', 0):.1%}")
                logger.info(f"🔧 System Health: {status.get('system_health', 'Unknown')}")
                logger.info(f"🕒 Last Update: {status.get('timestamp', 'Never')}")

            except Exception as e:
                logger.error(f"Could not read status file: {e}")
        else:
            logger.info("🤖 Bot Status: Not running")
            logger.info("💡 Use 'python3 run_quant_bot.py' to start the bot")

        # Show recent activity
        log_file = Path("quant_bot.log")
        if log_file.exists():
            logger.info("\n📋 Recent Activity:")
            try:
                with open(log_file, 'r') as f:
                    lines = f.readlines()[-10:]  # Last 10 lines
                    for line in lines:
                        print(f"   {line.strip()}")
            except Exception as e:
                logger.error(f"Could not read log file: {e}")

    async def shutdown(self):
        """Graceful shutdown"""

        logger.info("🛑 Shutting down QuantBot Runner...")

        if self.bot:
            # Bot handles its own shutdown
            pass

        self.running = False

        logger.info("✅ QuantBot Runner shutdown complete")


def main():
    """Main entry point"""

    parser = argparse.ArgumentParser(description='QuantEngine Autonomous Trading Bot')
    parser.add_argument('--demo', action='store_true',
                       help='Run in demo mode (limited time)')
    parser.add_argument('--status', action='store_true',
                       help='Show current bot status')

    args = parser.parse_args()

    runner = QuantBotRunner()

    if args.status:
        # Show status and exit
        asyncio.run(runner.show_status())
        return

    # Setup signal handlers for graceful shutdown
    def signal_handler(signum, frame):
        logger.info(f"Received signal {signum}, initiating shutdown...")
        if runner.running:
            runner.running = False

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    try:
        # Start the bot
        asyncio.run(runner.start_bot(demo_mode=args.demo))

    except KeyboardInterrupt:
        logger.info("Shutdown requested")
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
