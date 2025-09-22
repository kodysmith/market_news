#!/usr/bin/env python3
"""
AI Quant Trading System - Main Entry Point

This is the central orchestrator for the AI Quant v1 system.
It coordinates data ingestion, strategy research, backtesting,
robustness testing, and reporting.
"""

import argparse
import logging
import sys
from pathlib import Path
from datetime import datetime
import json

# Add QuantEngine to path
sys.path.insert(0, str(Path(__file__).parent))

from utils.strategy_dsl import StrategyValidator, EXAMPLE_TQQQ_STRATEGY
from engine.data_ingestion.data_manager import DataManager
from engine.feature_builder.feature_builder import FeatureBuilder
from engine.backtest_engine.backtester import VectorizedBacktester
from engine.robustness_lab.robustness_tester import RobustnessTester
from engine.reporting_notes.report_generator import ReportGenerator
from research.research_agent import ResearchAgent


def setup_logging(config: dict):
    """Setup logging configuration"""
    log_level = getattr(logging, config.get('log_level', 'INFO').upper())
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('quant_engine.log'),
            logging.StreamHandler(sys.stdout)
        ]
    )


def load_config(config_path: str = "config/config.yaml") -> dict:
    """Load configuration from file"""
    try:
        import yaml
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        return config
    except ImportError:
        # Fallback to basic config if yaml not available
        return {
            'log_level': 'INFO',
            'data_path': 'data',
            'output_dir': 'reports',
            'start_date': '2010-01-01',
            'end_date': '2024-01-01',
            'initial_capital': 1000000,
            'commission_bps': 2.0,
            'slippage_bps': 1.0,
        }
    except FileNotFoundError:
        print(f"Config file not found: {config_path}")
        print("Using default configuration")
        return {}


def run_data_pipeline(config: dict):
    """Run the data ingestion pipeline"""
    print("🔄 Running data ingestion pipeline...")

    data_manager = DataManager(config)

    # Get market data for default universe
    universe = ['SPY', 'QQQ', 'IWM', 'VTI', 'TQQQ']
    market_data = data_manager.get_market_data(
        universe,
        config.get('start_date', '2010-01-01'),
        config.get('end_date', pd.Timestamp.now().strftime('%Y-%m-%d'))
    )

    print(f"✅ Downloaded data for {len(market_data)} tickers")

    # Build features
    feature_builder = FeatureBuilder(config)
    feature_data = feature_builder.build_features(market_data)

    print(f"✅ Built features for {len(feature_data)} tickers")

    return market_data, feature_data


def run_strategy_backtest(config: dict, strategy_spec: dict = None):
    """Run backtest for a specific strategy"""
    print("🔄 Running strategy backtest...")

    if strategy_spec is None:
        strategy_spec = EXAMPLE_TQQQ_STRATEGY

    # Validate strategy
    try:
        spec = StrategyValidator.validate_spec(strategy_spec)
        print(f"✅ Strategy '{spec.name}' validated")
    except Exception as e:
        print(f"❌ Strategy validation failed: {e}")
        return None

    # Get data
    data_manager = DataManager(config)
    market_data = data_manager.get_market_data(
        spec.universe,
        config.get('start_date', '2020-01-01'),
        config.get('end_date', pd.Timestamp.now().strftime('%Y-%m-%d'))
    )

    if not market_data:
        print("❌ No market data available")
        return None

    # Run backtest
    backtester = VectorizedBacktester(config)
    backtest_result = backtester.run_backtest(spec, market_data)

    print("✅ Backtest completed")
    print(".2%")

    # Run robustness tests
    robustness_tester = RobustnessTester(config)
    robustness_report = robustness_tester.run_full_robustness_suite([backtest_result], market_data)

    # Generate report
    report_generator = ReportGenerator(config)
    report_path = report_generator.generate_research_note(spec, backtest_result, robustness_report, market_data)

    print(f"✅ Research note generated: {report_path}")

    return {
        'strategy': spec,
        'backtest': backtest_result,
        'robustness': robustness_report,
        'report': report_path
    }


def run_research_cycle(config: dict, objectives: list = None):
    """Run a complete research cycle"""
    print("🔄 Running research cycle...")

    if objectives is None:
        objectives = ['trend_following', 'volatility_targeting']

    research_agent = ResearchAgent(config)
    results = research_agent.run_research_cycle(objectives)

    print(f"✅ Research cycle completed:")
    print(f"   - Hypotheses generated: {results['hypotheses_generated']}")
    print(f"   - Strategies tested: {results['strategies_tested']}")
    print(f"   - Strategies approved: {results['strategies_approved']}")
    print(f"   - Reports generated: {len(results['reports_generated'])}")

    return results


def run_tqqq_reference(config: dict):
    """Run the TQQQ reference implementation"""
    print("🔄 Running TQQQ reference strategy...")

    research_agent = ResearchAgent(config)
    results = research_agent.run_tqqq_reference_implementation()

    if 'error' in results:
        print(f"❌ Error: {results['error']}")
        return None

    print("✅ TQQQ reference strategy completed")
    print(".2%")
    print(f"   - Approved: {results['approved']}")
    print(f"   - Report: {results['report_path']}")

    return results


def show_system_status(config: dict):
    """Show system status and configuration"""
    print("🤖 AI Quant Trading System v1.0")
    print("=" * 50)

    print("\n📊 Configuration:")
    for key, value in config.items():
        if isinstance(value, (str, int, float, bool)):
            print(f"   {key}: {value}")

    print("\n📁 Directory Structure:")
    base_path = Path(".")
    dirs = [
        "config", "data", "engine", "research", "live", "reports", "tests", "utils"
    ]
    for dir_name in dirs:
        dir_path = base_path / dir_name
        status = "✅" if dir_path.exists() else "❌"
        print(f"   {status} {dir_name}/")

    print("\n🔧 Core Modules:")
    modules = [
        ("utils/strategy_dsl.py", "Strategy DSL"),
        ("engine/data_ingestion/data_manager.py", "Data Manager"),
        ("engine/backtest_engine/backtester.py", "Backtester"),
        ("engine/robustness_lab/robustness_tester.py", "Robustness Tester"),
        ("engine/reporting_notes/report_generator.py", "Report Generator"),
        ("research/research_agent.py", "Research Agent"),
    ]

    for module_path, description in modules:
        path = base_path / module_path
        status = "✅" if path.exists() else "❌"
        print(f"   {status} {description}")

    print("\n🚀 System Ready!")


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description='AI Quant Trading System v1')
    parser.add_argument('command', choices=[
        'status', 'data', 'backtest', 'research', 'tqqq', 'cycle'
    ], help='Command to run')
    parser.add_argument('--config', default='config/config.yaml',
                       help='Configuration file path')
    parser.add_argument('--objectives', nargs='+',
                       help='Research objectives (for research command)')

    args = parser.parse_args()

    # Load configuration
    config = load_config(args.config)
    setup_logging(config)

    print(f"🤖 AI Quant v1 - Command: {args.command}")

    if args.command == 'status':
        show_system_status(config)

    elif args.command == 'data':
        run_data_pipeline(config)

    elif args.command == 'backtest':
        run_strategy_backtest(config)

    elif args.command == 'research':
        objectives = args.objectives or ['trend_following', 'volatility_targeting']
        run_research_cycle(config, objectives)

    elif args.command == 'tqqq':
        run_tqqq_reference(config)

    elif args.command == 'cycle':
        # Run full cycle: data -> research -> backtest
        print("🔄 Running full system cycle...")

        # 1. Data pipeline
        market_data, feature_data = run_data_pipeline(config)

        # 2. Research cycle
        research_results = run_research_cycle(config, args.objectives)

        # 3. TQQQ reference
        tqqq_results = run_tqqq_reference(config)

        print("\n🎯 Cycle Summary:")
        print(f"   - Market data: {len(market_data)} tickers")
        print(f"   - Research results: {research_results['strategies_approved']} approved")
        print(f"   - TQQQ strategy: {'✅ Approved' if tqqq_results and tqqq_results['approved'] else '❌ Rejected'}")

    print("\n✨ Done!")


if __name__ == "__main__":
    main()

