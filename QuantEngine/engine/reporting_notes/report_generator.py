"""
Research Report and Notes Generator for AI Quant Trading System

Creates comprehensive research notes including:
- Strategy overview and hypothesis
- Performance analysis and charts
- Risk metrics and robustness tests
- Implementation details and next steps
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional
import logging
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import json

logger = logging.getLogger(__name__)


class ReportGenerator:
    """Generates research reports and trading notes"""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.output_dir = Path(config.get('output_dir', 'reports'))
        self.output_dir.mkdir(exist_ok=True)

        # Set up matplotlib style
        plt.style.use('seaborn-v0_8')
        sns.set_palette("husl")

    def generate_research_note(self, strategy_spec: Any, backtest_result: Any,
                              robustness_report: Dict[str, Any],
                              market_data: Dict[str, pd.DataFrame]) -> str:
        """
        Generate comprehensive research note for a strategy

        Args:
            strategy_spec: Validated strategy specification
            backtest_result: Backtest results
            robustness_report: Robustness testing results
            market_data: Market data used

        Returns:
            Path to generated report
        """

        # Generate filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        strategy_name = strategy_spec.name
        filename = f"{strategy_name}_research_note_{timestamp}.md"
        filepath = self.output_dir / filename

        # Generate report sections
        sections = []

        # Title and overview
        sections.append(self._generate_title_section(strategy_spec))

        # Hypothesis and strategy logic
        sections.append(self._generate_hypothesis_section(strategy_spec))

        # Data and methodology
        sections.append(self._generate_methodology_section(strategy_spec, market_data))

        # Performance analysis
        sections.append(self._generate_performance_section(backtest_result))

        # Risk analysis
        sections.append(self._generate_risk_section(backtest_result, robustness_report))

        # Robustness testing
        sections.append(self._generate_robustness_section(robustness_report))

        # Implementation details
        sections.append(self._generate_implementation_section(strategy_spec))

        # Charts and visualizations
        chart_paths = self._generate_charts(strategy_spec, backtest_result, market_data)
        sections.append(self._generate_charts_section(chart_paths))

        # Next steps and recommendations
        sections.append(self._generate_next_steps_section(robustness_report))

        # Combine all sections
        full_report = "\n\n".join(sections)

        # Write to file
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(full_report)

        logger.info(f"Generated research note: {filepath}")

        return str(filepath)

    def _generate_title_section(self, strategy_spec: Any) -> str:
        """Generate title and overview section"""

        title = f"# {strategy_spec.name.replace('_', ' ').title()}\n"
        title += f"**Strategy Research Note**\n\n"
        title += f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"

        if strategy_spec.description:
            title += f"**Overview:** {strategy_spec.description}\n\n"

        # Strategy metadata
        title += "## Strategy Metadata\n\n"
        title += f"- **Universe:** {', '.join(strategy_spec.universe)}\n"
        title += f"- **Signals:** {len(strategy_spec.signals)}\n"
        title += f"- **Entry Conditions:** {len(strategy_spec.entry.all) if strategy_spec.entry.all else 0} required\n"

        if strategy_spec.overlays:
            title += f"- **Options Overlays:** {list(strategy_spec.overlays.keys())}\n"

        title += f"- **Risk Limits:** Max DD {strategy_spec.risk.max_dd_pct:.1%}\n\n"

        return title

    def _generate_hypothesis_section(self, strategy_spec: Any) -> str:
        """Generate hypothesis and strategy logic section"""

        section = "## Hypothesis & Strategy Logic\n\n"

        section += "### Investment Hypothesis\n\n"

        # Infer hypothesis from strategy components
        hypotheses = []

        for signal in strategy_spec.signals:
            if signal.type.value == "MA_cross":
                params = signal.params
                hypotheses.append(f"Price trends persist: {params['fast']}d MA crossing {params['slow']}d MA signals sustained moves")
            elif signal.type.value == "IV_proxy":
                hypotheses.append("Low implied volatility periods offer better risk-adjusted returns")
            elif signal.type.value == "sentiment":
                hypotheses.append("Market sentiment indicators predict short-term reversals")

        if strategy_spec.overlays:
            hypotheses.append("Options overlays provide efficient risk management and enhance returns")

        for i, hyp in enumerate(hypotheses, 1):
            section += f"{i}. {hyp}\n"

        section += "\n### Entry Logic\n\n"
        if strategy_spec.entry.all:
            section += "Enter when ALL conditions are met:\n\n"
            for condition in strategy_spec.entry.all:
                section += f"- {condition}\n"
        elif strategy_spec.entry.any:
            section += "Enter when ANY condition is met:\n\n"
            for condition in strategy_spec.entry.any:
                section += f"- {condition}\n"

        section += "\n### Exit Logic\n\n"
        if strategy_spec.exit:
            if strategy_spec.exit.all:
                section += "Exit when ALL conditions are met:\n\n"
                for condition in strategy_spec.exit.all:
                    section += f"- {condition}\n"
            elif strategy_spec.exit.any:
                section += "Exit when ANY condition is met:\n\n"
                for condition in strategy_spec.exit.any:
                    section += f"- {condition}\n"
        else:
            section += "Hold until entry conditions reverse or risk limits hit\n"

        return section

    def _generate_methodology_section(self, strategy_spec: Any, market_data: Dict[str, pd.DataFrame]) -> str:
        """Generate data and methodology section"""

        section = "## Data & Methodology\n\n"

        # Data description
        section += "### Data Sources\n\n"
        section += f"- **Universe:** {', '.join(strategy_spec.universe)}\n"
        section += f"- **Time Period:** {min([df.index.min() for df in market_data.values()])} to {max([df.index.max() for df in market_data.values()])}\n"
        section += f"- **Frequency:** Daily\n"
        section += f"- **Source:** Yahoo Finance\n\n"

        # Backtesting methodology
        section += "### Backtesting Methodology\n\n"
        section += "- **Execution:** Vectorized pandas/NumPy implementation\n"
        section += "- **Costs:**\n"
        section += f"  - Commissions: {strategy_spec.costs.commission_bps} bps per trade\n"
        section += f"  - Slippage: {strategy_spec.costs.slippage_bps} bps per trade\n"
        section += f"  - Options fees: ${strategy_spec.costs.fee_per_option} per contract\n"
        section += "- **Position Sizing:**\n"
        section += f"  - Volatility targeting: {strategy_spec.sizing.vol_target_ann:.1%} annualized\n"
        section += f"  - Max weight: {strategy_spec.sizing.max_weight:.1%} per position\n"
        section += "- **Rebalancing:** Daily\n\n"

        return section

    def _generate_performance_section(self, backtest_result: Any) -> str:
        """Generate performance analysis section"""

        if not hasattr(backtest_result, 'metrics') or not backtest_result.metrics:
            return "## Performance Analysis\n\nNo performance data available\n"

        metrics = backtest_result.metrics

        section = "## Performance Analysis\n\n"

        # Key metrics table
        section += "### Key Performance Metrics\n\n"
        section += "| Metric | Value |\n"
        section += "|--------|-------|\n"
        section += f"| Total Return | {metrics.get('total_return', 0):.2%} |\n"
        section += f"| Annualized Return | {metrics.get('ann_return', 0):.2%} |\n"
        section += f"| Annualized Volatility | {metrics.get('ann_vol', 0):.2%} |\n"
        section += f"| Sharpe Ratio | {metrics.get('sharpe', 0):.2f} |\n"
        section += f"| Max Drawdown | {metrics.get('max_dd', 0):.2%} |\n"
        section += f"| Win Rate | {metrics.get('win_rate', 0):.1%} |\n"
        section += f"| Profit Factor | {metrics.get('profit_factor', float('inf')):.2f} |\n"
        section += f"| Calmar Ratio | {metrics.get('calmar', 0):.2f} |\n"
        section += "\n"

        # Performance commentary
        section += "### Performance Commentary\n\n"

        sharpe = metrics.get('sharpe', 0)
        max_dd = metrics.get('max_dd', 0)
        ann_return = metrics.get('ann_return', 0)

        if sharpe > 1.5:
            section += f"**Excellent risk-adjusted returns** with Sharpe ratio of {sharpe:.2f}. "
        elif sharpe > 1.0:
            section += f"**Strong risk-adjusted performance** with Sharpe ratio of {sharpe:.2f}. "
        elif sharpe > 0.5:
            section += f"**Acceptable risk-adjusted returns** with Sharpe ratio of {sharpe:.2f}. "
        else:
            section += f"**Poor risk-adjusted performance** with Sharpe ratio of {sharpe:.2f}. "

        if abs(max_dd) < 0.15:
            section += f"Drawdown of {max_dd:.1%} is well within acceptable limits. "
        elif abs(max_dd) < 0.25:
            section += f"Drawdown of {max_dd:.1%} is manageable but noteworthy. "
        else:
            section += f"Drawdown of {max_dd:.1%} is concerning and may require risk management adjustments. "

        if ann_return > 0.15:
            section += f"Annualized returns of {ann_return:.1%} exceed typical equity benchmarks. "
        elif ann_return > 0.08:
            section += f"Annualized returns of {ann_return:.1%} are solid. "
        else:
            section += f"Annualized returns of {ann_return:.1%} are below expectations. "

        section += "\n\n"

        return section

    def _generate_risk_section(self, backtest_result: Any, robustness_report: Dict[str, Any]) -> str:
        """Generate risk analysis section"""

        section = "## Risk Analysis\n\n"

        if hasattr(backtest_result, 'metrics') and backtest_result.metrics:
            metrics = backtest_result.metrics

            section += "### Risk Metrics\n\n"
            section += "| Risk Measure | Value | Assessment |\n"
            section += "|--------------|-------|------------|\n"

            max_dd = metrics.get('max_dd', 0)
            var_95 = metrics.get('var_95', 0)
            ann_vol = metrics.get('ann_vol', 0)

            dd_assessment = "✅ Low" if abs(max_dd) < 0.15 else "⚠️ Moderate" if abs(max_dd) < 0.25 else "❌ High"
            section += f"| Max Drawdown | {max_dd:.2%} | {dd_assessment} |\n"

            vol_assessment = "✅ Low" if ann_vol < 0.15 else "⚠️ Moderate" if ann_vol < 0.25 else "❌ High"
            section += f"| Annual Volatility | {ann_vol:.2%} | {vol_assessment} |\n"

            var_assessment = "✅ Low" if abs(var_95) < 0.02 else "⚠️ Moderate" if abs(var_95) < 0.03 else "❌ High"
            section += f"| 95% VaR | {var_95:.2%} | {var_assessment} |\n"

            section += "\n"

        # Position and sizing risk
        section += "### Position & Sizing Risk\n\n"
        if hasattr(backtest_result, 'positions') and not backtest_result.positions.empty:
            positions = backtest_result.positions
            avg_exposure = positions['target_weight'].abs().mean()
            max_exposure = positions['target_weight'].abs().max()
            turnover = positions['target_weight'].diff().abs().sum()

            section += f"- **Average Exposure:** {avg_exposure:.1%}\n"
            section += f"- **Maximum Exposure:** {max_exposure:.1%}\n"
            section += f"- **Portfolio Turnover:** {turnover:.1f}x\n\n"

        return section

    def _generate_robustness_section(self, robustness_report: Dict[str, Any]) -> str:
        """Generate robustness testing section"""

        section = "## Robustness Testing\n\n"

        if not robustness_report:
            section += "No robustness testing performed\n\n"
            return section

        # Walk-forward analysis
        wf = robustness_report.get('walk_forward', {})
        if 'oos_metrics' in wf:
            section += "### Walk-Forward Validation\n\n"
            oos_sharpe = wf['oos_metrics'].get('sharpe', 0)
            is_sharpe = wf['is_metrics'].get('sharpe', 0)

            section += f"- **IS Sharpe:** {is_sharpe:.2f}\n"
            section += f"- **OOS Sharpe:** {oos_sharpe:.2f}\n"
            section += f"- **OOS/IS Ratio:** {oos_sharpe/is_sharpe:.2f} {'✅' if oos_sharpe/is_sharpe > 0.8 else '⚠️'}\n\n"

        # Regime analysis
        regime = robustness_report.get('regime_analysis', {})
        if regime:
            section += "### Regime Analysis\n\n"
            section += "| Market Regime | Sharpe | Return | Max DD |\n"
            section += "|---------------|--------|--------|--------|\n"

            for regime_name, metrics in regime.items():
                if isinstance(metrics, dict):
                    sharpe = metrics.get('sharpe', 0)
                    ret = metrics.get('ann_return', 0)
                    dd = metrics.get('max_dd', 0)
                    section += f"| {regime_name.replace('_', ' ').title()} | {sharpe:.2f} | {ret:.1%} | {dd:.1%} |\n"

            section += "\n"

        # Green light decision
        green_light = robustness_report.get('green_light', {})
        if green_light:
            approved = green_light.get('approved', False)
            score = green_light.get('score', 0)
            max_score = green_light.get('max_score', 10)
            confidence = green_light.get('confidence', 'unknown')

            section += "### Green Light Decision\n\n"
            status = "✅ **APPROVED**" if approved else "❌ **REJECTED**"
            section += f"**Status:** {status}\n\n"
            section += f"**Robustness Score:** {score}/{max_score} ({confidence} confidence)\n\n"

            if green_light.get('reasons'):
                section += "**Approval Reasons:**\n"
                for reason in green_light['reasons']:
                    section += f"- {reason}\n"
                section += "\n"

            if green_light.get('warnings'):
                section += "**Warnings/Concerns:**\n"
                for warning in green_light['warnings']:
                    section += f"- {warning}\n"
                section += "\n"

        return section

    def _generate_implementation_section(self, strategy_spec: Any) -> str:
        """Generate implementation details section"""

        section = "## Implementation Details\n\n"

        section += "### Strategy DSL Specification\n\n"
        section += "```json\n"
        section += json.dumps(strategy_spec.dict(), indent=2)
        section += "\n```\n\n"

        section += "### Key Parameters\n\n"
        section += f"- **Vol Target:** {strategy_spec.sizing.vol_target_ann:.1%} annualized\n"
        section += f"- **Max Position:** {strategy_spec.sizing.max_weight:.1%}\n"
        section += f"- **Max Drawdown:** {strategy_spec.risk.max_dd_pct:.1%}\n"

        if strategy_spec.overlays:
            for overlay_name, overlay in strategy_spec.overlays.items():
                section += f"- **{overlay_name.title()} Overlay:** {overlay.target_delta:.2f} Δ target\n"

        section += "\n"

        return section

    def _generate_charts(self, strategy_spec: Any, backtest_result: Any,
                        market_data: Dict[str, pd.DataFrame]) -> List[str]:
        """Generate performance charts"""

        chart_paths = []

        if not hasattr(backtest_result, 'returns') or backtest_result.returns.empty:
            return chart_paths

        try:
            # Equity curve
            fig, ax = plt.subplots(figsize=(12, 6))
            cumulative_returns = (1 + backtest_result.returns).cumprod()
            cumulative_returns.plot(ax=ax, linewidth=2)
            ax.set_title(f'{strategy_spec.name} - Equity Curve')
            ax.set_ylabel('Portfolio Value')
            ax.grid(True, alpha=0.3)

            equity_path = self.output_dir / f"{strategy_spec.name}_equity_curve.png"
            plt.savefig(equity_path, dpi=150, bbox_inches='tight')
            plt.close()
            chart_paths.append(str(equity_path))

            # Drawdown chart
            fig, ax = plt.subplots(figsize=(12, 6))
            running_max = cumulative_returns.expanding().max()
            drawdown = (cumulative_returns - running_max) / running_max
            drawdown.plot(ax=ax, color='red', linewidth=2)
            ax.set_title(f'{strategy_spec.name} - Drawdown')
            ax.set_ylabel('Drawdown')
            ax.grid(True, alpha=0.3)
            ax.fill_between(drawdown.index, drawdown, 0, color='red', alpha=0.3)

            dd_path = self.output_dir / f"{strategy_spec.name}_drawdown.png"
            plt.savefig(dd_path, dpi=150, bbox_inches='tight')
            plt.close()
            chart_paths.append(str(dd_path))

            # Returns distribution
            fig, ax = plt.subplots(figsize=(10, 6))
            backtest_result.returns.hist(ax=ax, bins=50, alpha=0.7, density=True)
            ax.set_title(f'{strategy_spec.name} - Returns Distribution')
            ax.set_xlabel('Daily Return')
            ax.set_ylabel('Density')
            ax.axvline(backtest_result.returns.mean(), color='red', linestyle='--', label='Mean')
            ax.legend()

            dist_path = self.output_dir / f"{strategy_spec.name}_returns_dist.png"
            plt.savefig(dist_path, dpi=150, bbox_inches='tight')
            plt.close()
            chart_paths.append(str(dist_path))

        except Exception as e:
            logger.error(f"Failed to generate charts: {e}")

        return chart_paths

    def _generate_charts_section(self, chart_paths: List[str]) -> str:
        """Generate charts section with image links"""

        if not chart_paths:
            return "## Charts & Visualizations\n\nNo charts generated\n\n"

        section = "## Charts & Visualizations\n\n"

        for path in chart_paths:
            filename = Path(path).name
            section += f"![{filename}]({filename})\n\n"

        return section

    def _generate_next_steps_section(self, robustness_report: Dict[str, Any]) -> str:
        """Generate next steps and recommendations"""

        section = "## Next Steps & Recommendations\n\n"

        green_light = robustness_report.get('green_light', {})
        approved = green_light.get('approved', False)

        if approved:
            confidence = green_light.get('confidence', 'medium')

            if confidence == 'high':
                section += "### ✅ Ready for Live Trading\n\n"
                section += "Strategy meets all robustness criteria and can be deployed to live paper trading.\n\n"
                section += "**Action Items:**\n"
                section += "1. Deploy to paper trading environment\n"
                section += "2. Monitor performance for first 30 days\n"
                section += "3. Set up automated alerts for risk limits\n"
                section += "4. Schedule monthly performance reviews\n\n"

            elif confidence == 'medium':
                section += "### ⚠️ Conditional Approval\n\n"
                section += "Strategy shows promise but requires monitoring and potential adjustments.\n\n"
                section += "**Action Items:**\n"
                section += "1. Deploy to paper trading with reduced capital allocation\n"
                section += "2. Monitor key risk metrics weekly\n"
                section += "3. Consider parameter optimization\n"
                section += "4. Reassess after 60 days of paper trading\n\n"

        else:
            section += "### ❌ Further Development Required\n\n"
            section += "Strategy does not meet minimum robustness criteria.\n\n"
            section += "**Recommended Actions:**\n"
            section += "1. Review and strengthen entry/exit logic\n"
            section += "2. Adjust position sizing parameters\n"
            section += "3. Add additional risk management overlays\n"
            section += "4. Re-test with different parameter combinations\n\n"

        # General recommendations
        section += "### General Recommendations\n\n"
        section += "1. **Data Quality:** Ensure data feeds are reliable and up-to-date\n"
        section += "2. **Execution Quality:** Monitor slippage and transaction costs in live trading\n"
        section += "3. **Risk Management:** Never exceed position limits or risk thresholds\n"
        section += "4. **Performance Tracking:** Maintain detailed logs of all trades and decisions\n"
        section += "5. **Regular Review:** Reassess strategy performance quarterly\n\n"

        section += "---\n\n"
        section += f"*Report generated by AI Quant v1 on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*"

        return section

