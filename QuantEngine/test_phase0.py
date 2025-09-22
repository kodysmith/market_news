#!/usr/bin/env python3
"""
Phase 0 Test Script - Basic functionality testing

Tests data ingestion, backtesting, and reporting for Phase 0 requirements:
- Ingest daily bars (SPY, QQQ, TQQQ, sector ETFs)
- Backtester for long-only & simple hedged rules
- Basic reports with equity/DD/Sharpe + English summary
"""

import sys
import os
from pathlib import Path
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
import matplotlib.pyplot as plt

# Add QuantEngine to path
sys.path.insert(0, str(Path(__file__).parent))

def test_basic_imports():
    """Test basic imports work"""
    try:
        from utils.strategy_dsl import StrategyValidator, EXAMPLE_TQQQ_STRATEGY
        print("✅ Strategy DSL imports successful")
        return True
    except Exception as e:
        print(f"❌ Import failed: {e}")
        return False

def download_market_data(tickers, start_date, end_date):
    """Download market data for testing"""
    print(f"📥 Downloading data for {tickers}...")

    data = {}
    for ticker in tickers:
        try:
            print(f"  Downloading {ticker}...")
            df = yf.download(ticker, start=start_date, end=end_date, progress=False)

            if not df.empty:
                # Clean column names
                df.columns = df.columns.str.lower()
                if 'adj close' in df.columns:
                    df = df.rename(columns={'adj close': 'adj_close'})

                data[ticker] = df
                print(f"  ✅ {ticker}: {len(df)} rows")
            else:
                print(f"  ❌ {ticker}: No data")

        except Exception as e:
            print(f"  ❌ {ticker}: Download failed - {e}")

    return data

def create_simple_backtester():
    """Create a simple backtester for testing"""

    class SimpleBacktester:
        def __init__(self, config):
            self.config = config
            self.commission_bps = config.get('commission_bps', 2.0)
            self.slippage_bps = config.get('slippage_bps', 1.0)

        def run_backtest(self, strategy_spec, market_data):
            """Simple backtest implementation"""
            ticker = strategy_spec.get('universe', ['SPY'])[0]

            if ticker not in market_data:
                return None

            df = market_data[ticker].copy()

            # Generate signals based on strategy
            signals = self._generate_signals(strategy_spec, df)

            # Calculate positions
            positions = signals.fillna(0)

            # Calculate returns with costs
            returns = self._calculate_returns(df, positions)

            # Calculate metrics
            metrics = self._calculate_metrics(returns, positions)

            return {
                'returns': returns,
                'positions': positions,
                'metrics': metrics,
                'data': df
            }

        def _generate_signals(self, strategy_spec, df):
            """Generate trading signals"""
            signals = pd.Series(0, index=df.index)

            signals_df = strategy_spec.get('signals', [])
            entry_conditions = strategy_spec.get('entry', {})

            for signal in signals_df:
                if signal.get('type') == 'MA_cross':
                    fast_period = signal.get('params', {}).get('fast', 20)
                    slow_period = signal.get('params', {}).get('slow', 200)

                    fast_ma = df['close'].rolling(fast_period).mean()
                    slow_ma = df['close'].rolling(slow_period).mean()

                    ma_signal = (fast_ma > slow_ma).astype(int)
                    signals = signals | ma_signal

                elif signal.get('type') == 'IV_proxy':
                    # Simplified IV proxy
                    returns_vol = df['close'].pct_change().rolling(20).std() * np.sqrt(252)
                    threshold = signal.get('params', {}).get('low_thresh', 0.45)
                    vol_signal = (returns_vol < threshold).astype(int)
                    signals = signals | vol_signal

            return signals

        def _calculate_returns(self, df, positions):
            """Calculate strategy returns with costs"""
            asset_returns = df['close'].pct_change()

            # Position changes for transaction costs
            position_changes = positions.diff().abs()
            position_changes.iloc[0] = positions.iloc[0]

            # Costs
            commission_cost = position_changes * (self.commission_bps / 10000)
            slippage_cost = position_changes * (self.slippage_bps / 10000)
            total_costs = commission_cost + slippage_cost

            # Strategy returns
            strategy_returns = (positions.shift(1).fillna(0) * asset_returns) - total_costs

            return strategy_returns

        def _calculate_metrics(self, returns, positions):
            """Calculate performance metrics"""
            if returns.empty:
                return {}

            # Basic metrics
            cumulative_returns = (1 + returns).cumprod()
            total_return = cumulative_returns.iloc[-1] - 1

            # Annualized metrics
            days = len(returns)
            years = max(days / 252, 0.001)
            ann_return = (1 + total_return) ** (1 / years) - 1

            # Risk metrics
            ann_vol = returns.std() * np.sqrt(252)
            sharpe = ann_return / ann_vol if ann_vol > 0 else 0

            # Drawdown
            running_max = cumulative_returns.expanding().max()
            drawdown = (cumulative_returns - running_max) / running_max
            max_dd = drawdown.min()

            # Additional metrics
            win_rate = (returns > 0).mean()
            profit_factor = returns[returns > 0].sum() / abs(returns[returns < 0].sum()) if (returns < 0).any() else float('inf')

            return {
                'total_return': total_return,
                'ann_return': ann_return,
                'ann_vol': ann_vol,
                'sharpe': sharpe,
                'max_dd': max_dd,
                'win_rate': win_rate,
                'profit_factor': profit_factor,
                'num_trades': (returns != 0).sum(),
                'avg_trade': returns[returns != 0].mean(),
            }

    return SimpleBacktester

def generate_basic_report(result, strategy_spec, output_path):
    """Generate basic performance report"""

    if not result or 'metrics' not in result:
        print("❌ No results to report")
        return

    metrics = result['metrics']

    report = f"""# {strategy_spec.get('name', 'Strategy')} - Performance Report

**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## Performance Summary

| Metric | Value |
|--------|-------|
| Total Return | {metrics.get('total_return', 0):.2%} |
| Annualized Return | {metrics.get('ann_return', 0):.2%} |
| Annualized Volatility | {metrics.get('ann_vol', 0):.2%} |
| Sharpe Ratio | {metrics.get('sharpe', 0):.2f} |
| Max Drawdown | {metrics.get('max_dd', 0):.2%} |
| Win Rate | {metrics.get('win_rate', 0):.1%} |
| Profit Factor | {metrics.get('profit_factor', float('inf')):.2f} |
| Number of Trades | {metrics.get('num_trades', 0)} |
| Average Trade | {metrics.get('avg_trade', 0):.4%} |

## Analysis

"""

    # Sharpe analysis
    sharpe = metrics.get('sharpe', 0)
    if sharpe > 1.5:
        report += f"**Excellent risk-adjusted performance** with Sharpe ratio of {sharpe:.2f}.\n\n"
    elif sharpe > 1.0:
        report += f"**Strong risk-adjusted performance** with Sharpe ratio of {sharpe:.2f}.\n\n"
    elif sharpe > 0.5:
        report += f"**Acceptable risk-adjusted performance** with Sharpe ratio of {sharpe:.2f}.\n\n"
    else:
        report += f"**Poor risk-adjusted performance** with Sharpe ratio of {sharpe:.2f}.\n\n"

    # Return analysis
    ann_return = metrics.get('ann_return', 0)
    if ann_return > 0.15:
        report += f"**Outstanding absolute returns** of {ann_return:.1%} annualized.\n\n"
    elif ann_return > 0.08:
        report += f"**Solid absolute returns** of {ann_return:.1%} annualized.\n\n"
    else:
        report += f"**Below-average returns** of {ann_return:.1%} annualized.\n\n"

    # Risk analysis
    max_dd = metrics.get('max_dd', 0)
    if abs(max_dd) < 0.15:
        report += f"**Low risk** with maximum drawdown of {max_dd:.1%}.\n\n"
    elif abs(max_dd) < 0.25:
        report += f"**Moderate risk** with maximum drawdown of {max_dd:.1%}.\n\n"
    else:
        report += f"**High risk** with maximum drawdown of {max_dd:.1%}.\n\n"

    # Save report
    with open(output_path, 'w') as f:
        f.write(report)

    print(f"✅ Report saved to {output_path}")

    # Generate equity curve chart
    if 'returns' in result:
        plt.figure(figsize=(12, 6))
        cumulative_returns = (1 + result['returns']).cumprod()
        cumulative_returns.plot(linewidth=2)
        plt.title(f'{strategy_spec.get("name", "Strategy")} - Equity Curve')
        plt.ylabel('Portfolio Value')
        plt.grid(True, alpha=0.3)

        chart_path = output_path.replace('.md', '_equity.png')
        plt.savefig(chart_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"✅ Chart saved to {chart_path}")

def run_phase0_test():
    """Run Phase 0 testing"""
    print("🚀 Starting Phase 0 Test")
    print("=" * 50)

    # Test imports
    if not test_basic_imports():
        print("❌ Basic imports failed - cannot continue")
        return

    # Configuration
    config = {
        'start_date': '2020-01-01',
        'end_date': '2024-01-01',
        'commission_bps': 2.0,
        'slippage_bps': 1.0,
    }

    # Data ingestion test
    tickers = ['SPY', 'QQQ', 'TQQQ', 'XLE', 'XLF', 'XLK']
    market_data = download_market_data(tickers, config['start_date'], config['end_date'])

    if not market_data:
        print("❌ Data download failed")
        return

    # Backtester test
    SimpleBacktester = create_simple_backtester()
    backtester = SimpleBacktester(config)

    # Test with TQQQ strategy
    from utils.strategy_dsl import EXAMPLE_TQQQ_STRATEGY
    strategy_spec = EXAMPLE_TQQQ_STRATEGY

    print(f"🔄 Backtesting {strategy_spec['name']}...")
    result = backtester.run_backtest(strategy_spec, market_data)

    if result:
        print("✅ Backtest completed")
        print(".2%")

        # Generate report
        reports_dir = Path('reports')
        reports_dir.mkdir(exist_ok=True)

        report_path = reports_dir / f"{strategy_spec['name']}_phase0_report.md"
        generate_basic_report(result, strategy_spec, str(report_path))

    print("✅ Phase 0 test completed!")

if __name__ == "__main__":
    run_phase0_test()

