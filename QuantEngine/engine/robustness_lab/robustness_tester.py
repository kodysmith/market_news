"""
Robustness Testing Lab for AI Quant Trading System

Implements comprehensive strategy validation including:
- Walk-forward cross-validation
- Out-of-sample testing
- Regime-based analysis
- Parameter sensitivity testing
- Reality checks and statistical validation
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
import logging
from scipy import stats
from sklearn.model_selection import TimeSeriesSplit
import statsmodels.api as sm
from statsmodels.stats.diagnostic import acorr_ljungbox
import warnings

logger = logging.getLogger(__name__)


class RobustnessTester:
    """Comprehensive strategy robustness testing"""

    def __init__(self, config: Dict[str, Any]):
        self.config = config

    def run_full_robustness_suite(self, backtest_results: List[Any],
                                market_data: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
        """
        Run complete robustness testing suite

        Args:
            backtest_results: List of backtest results from walk-forward analysis
            market_data: Original market data

        Returns:
            Comprehensive robustness report
        """

        report = {}

        if not backtest_results:
            logger.warning("No backtest results to analyze")
            return report

        # Combine results into single analysis
        combined_returns = self._combine_results(backtest_results)

        # Basic performance metrics
        report['performance'] = self._calculate_performance_metrics(combined_returns)

        # Walk-forward validation
        report['walk_forward'] = self._analyze_walk_forward_performance(backtest_results)

        # Regime analysis
        report['regime_analysis'] = self._analyze_regime_robustness(backtest_results, market_data)

        # Statistical tests
        report['statistical_tests'] = self._run_statistical_tests(combined_returns)

        # Parameter sensitivity (placeholder for now)
        report['parameter_sensitivity'] = self._analyze_parameter_sensitivity(backtest_results)

        # Reality checks
        report['reality_checks'] = self._perform_reality_checks(combined_returns)

        # Green light decision
        report['green_light'] = self._make_green_light_decision(report)

        return report

    def _combine_results(self, backtest_results: List[Any]) -> pd.Series:
        """Combine multiple backtest results into single return series"""
        if not backtest_results:
            return pd.Series()

        # Concatenate returns from all periods
        all_returns = []
        for result in backtest_results:
            if hasattr(result, 'returns') and not result.returns.empty:
                all_returns.append(result.returns)

        if not all_returns:
            return pd.Series()

        # Combine with proper alignment
        combined = pd.concat(all_returns, axis=0).sort_index()

        # Remove duplicate indices (if any)
        combined = combined[~combined.index.duplicated(keep='first')]

        return combined

    def _calculate_performance_metrics(self, returns: pd.Series) -> Dict[str, float]:
        """Calculate comprehensive performance metrics"""

        if returns.empty:
            return {}

        # Basic return metrics
        cumulative_returns = (1 + returns).cumprod()
        total_return = cumulative_returns.iloc[-1] - 1

        # Annualized metrics
        days = len(returns)
        years = max(days / 252, 0.001)  # Avoid division by zero
        ann_return = (1 + total_return) ** (1 / years) - 1

        # Risk metrics
        ann_vol = returns.std() * np.sqrt(252)
        sharpe = ann_return / ann_vol if ann_vol > 0 else 0

        # Drawdown analysis
        running_max = cumulative_returns.expanding().max()
        drawdown = (cumulative_returns - running_max) / running_max
        max_dd = drawdown.min()

        # Additional metrics
        win_rate = (returns > 0).mean()
        profit_factor = returns[returns > 0].sum() / abs(returns[returns < 0].sum()) if (returns < 0).any() else float('inf')

        # Calmar ratio
        calmar = ann_return / abs(max_dd) if max_dd < 0 else 0

        # Sortino ratio (downside deviation)
        downside_returns = returns[returns < 0]
        downside_vol = downside_returns.std() * np.sqrt(252) if not downside_returns.empty else 0
        sortino = ann_return / downside_vol if downside_vol > 0 else 0

        # Value at Risk (95%)
        var_95 = np.percentile(returns, 5)

        return {
            'total_return': total_return,
            'ann_return': ann_return,
            'ann_vol': ann_vol,
            'sharpe': sharpe,
            'max_dd': max_dd,
            'calmar': calmar,
            'sortino': sortino,
            'win_rate': win_rate,
            'profit_factor': profit_factor,
            'var_95': var_95,
            'num_trades': len(returns),
            'avg_trade': returns.mean(),
        }

    def _analyze_walk_forward_performance(self, backtest_results: List[Any]) -> Dict[str, Any]:
        """Analyze walk-forward validation performance"""

        if len(backtest_results) < 2:
            return {'insufficient_data': True}

        # Extract IS/OOS performance
        is_periods = []
        oos_periods = []

        # Assuming alternating IS/OOS or structured walk-forward
        for i, result in enumerate(backtest_results):
            if i % 2 == 0:  # Even indices = IS
                is_periods.append(result)
            else:  # Odd indices = OOS
                oos_periods.append(result)

        # Calculate metrics for IS and OOS
        is_returns = self._combine_results(is_periods)
        oos_returns = self._combine_results(oos_periods)

        is_metrics = self._calculate_performance_metrics(is_returns)
        oos_metrics = self._calculate_performance_metrics(oos_returns)

        # Degradation analysis
        degradation = {}
        for metric in ['sharpe', 'ann_return', 'win_rate']:
            if metric in is_metrics and metric in oos_metrics:
                is_val = is_metrics[metric]
                oos_val = oos_metrics[metric]
                degradation[metric] = (oos_val - is_val) / abs(is_val) if is_val != 0 else 0

        return {
            'is_metrics': is_metrics,
            'oos_metrics': oos_metrics,
            'degradation': degradation,
            'oos_sharpe_ratio': oos_metrics.get('sharpe', 0) / is_metrics.get('sharpe', 1) if is_metrics.get('sharpe', 0) > 0 else 0
        }

    def _analyze_regime_robustness(self, backtest_results: List[Any],
                                 market_data: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
        """Analyze performance across different market regimes"""

        if not backtest_results or not market_data:
            return {}

        # Combine all returns
        returns = self._combine_results(backtest_results)

        # Identify market regimes (simplified)
        # This would be enhanced with actual regime detection
        spy_data = market_data.get('SPY')
        if spy_data is None or spy_data.empty:
            return {'regime_data_unavailable': True}

        # Simple regime classification based on SPY
        spy_returns = spy_data['close'].pct_change()

        # Trend regime
        sma_200 = spy_data['close'].rolling(200).mean()
        trend_regime = (spy_data['close'] > sma_200).astype(int)

        # Vol regime (high/low vol)
        vol_20 = spy_returns.rolling(20).std() * np.sqrt(252)
        vol_median = vol_20.rolling(252).median()
        vol_regime = (vol_20 > vol_median).astype(int)

        # Align indices
        common_index = returns.index.intersection(trend_regime.index).intersection(vol_regime.index)
        if len(common_index) == 0:
            return {'no_common_dates': True}

        returns_aligned = returns.loc[common_index]
        trend_aligned = trend_regime.loc[common_index]
        vol_aligned = vol_regime.loc[common_index]

        # Performance by regime
        regime_performance = {}

        # Bull/Bear markets
        bull_returns = returns_aligned[trend_aligned == 1]
        bear_returns = returns_aligned[trend_aligned == 0]

        if not bull_returns.empty:
            regime_performance['bull_market'] = self._calculate_performance_metrics(bull_returns)
        if not bear_returns.empty:
            regime_performance['bear_market'] = self._calculate_performance_metrics(bear_returns)

        # High/Low vol periods
        high_vol_returns = returns_aligned[vol_aligned == 1]
        low_vol_returns = returns_aligned[vol_aligned == 0]

        if not high_vol_returns.empty:
            regime_performance['high_vol'] = self._calculate_performance_metrics(high_vol_returns)
        if not low_vol_returns.empty:
            regime_performance['low_vol'] = self._calculate_performance_metrics(low_vol_returns)

        return regime_performance

    def _run_statistical_tests(self, returns: pd.Series) -> Dict[str, Any]:
        """Run statistical tests for robustness"""

        if returns.empty or len(returns) < 30:
            return {'insufficient_data': True}

        tests = {}

        # Normality test (Shapiro-Wilk)
        try:
            stat, p_value = stats.shapiro(returns.dropna())
            tests['shapiro_normality'] = {'statistic': stat, 'p_value': p_value, 'is_normal': p_value > 0.05}
        except:
            tests['shapiro_normality'] = {'error': 'Test failed'}

        # Autocorrelation test (Ljung-Box)
        try:
            lb_stat, lb_p = acorr_ljungbox(returns.dropna(), lags=[5, 10], return_df=False)
            tests['ljung_box'] = {'statistic': lb_stat[0], 'p_value': lb_p[0], 'no_autocorr': lb_p[0] > 0.05}
        except:
            tests['ljung_box'] = {'error': 'Test failed'}

        # Stationarity test (ADF)
        try:
            from statsmodels.tsa.stattools import adfuller
            adf_stat, adf_p, _, _, _, _ = adfuller(returns.dropna())
            tests['adf_stationarity'] = {'statistic': adf_stat, 'p_value': adf_p, 'is_stationary': adf_p < 0.05}
        except:
            tests['adf_stationarity'] = {'error': 'Test failed'}

        # Outlier analysis
        z_scores = np.abs(stats.zscore(returns.dropna()))
        outlier_pct = (z_scores > 3).mean()
        tests['outliers'] = {'percentage': outlier_pct, 'concerning': outlier_pct > 0.05}

        return tests

    def _analyze_parameter_sensitivity(self, backtest_results: List[Any]) -> Dict[str, Any]:
        """Analyze sensitivity to parameter changes (placeholder)"""

        # This would analyze how performance changes with parameter variations
        # For now, return placeholder
        return {
            'sensitivity_tested': False,
            'note': 'Parameter sensitivity analysis not yet implemented'
        }

    def _perform_reality_checks(self, returns: pd.Series) -> Dict[str, Any]:
        """Perform reality checks (Bailey and Lopez de Prado)"""

        if returns.empty or len(returns) < 100:
            return {'insufficient_data': True}

        checks = {}

        # Multiple testing correction
        # Simple Bonferroni correction for Sharpe ratio significance
        n_tests = 10  # Assume 10 different tests/strategies
        sharpe = returns.mean() / returns.std() * np.sqrt(252)
        t_stat = sharpe * np.sqrt(len(returns)) / np.sqrt(252)

        # Adjusted p-value for multiple testing
        checks['multiple_testing'] = {
            'original_sharpe': sharpe,
            't_statistic': t_stat,
            'bonferroni_threshold': stats.t.ppf(0.95, len(returns) - 1) / np.sqrt(n_tests),
            'significant_after_correction': t_stat > stats.t.ppf(0.95, len(returns) - 1) / np.sqrt(n_tests)
        }

        # Deflated Sharpe Ratio (simplified)
        # This is a simplified version - full implementation would be more complex
        checks['sharpe_deflation'] = {
            'note': 'Full Sharpe ratio deflation test requires multiple strategy backtests',
            'estimated_deflation_factor': 0.8  # Placeholder
        }

        return checks

    def _make_green_light_decision(self, report: Dict[str, Any]) -> Dict[str, Any]:
        """Make final green light decision based on all tests"""

        decision = {
            'approved': False,
            'reasons': [],
            'warnings': [],
            'score': 0
        }

        score = 0
        max_score = 10

        # Performance criteria
        perf = report.get('performance', {})
        sharpe = perf.get('sharpe', 0)
        max_dd = perf.get('max_dd', 0)
        win_rate = perf.get('win_rate', 0)

        if sharpe >= 1.0:
            score += 3
            decision['reasons'].append(f"Strong Sharpe ratio: {sharpe:.2f}")
        elif sharpe >= 0.5:
            score += 2
            decision['reasons'].append(f"Acceptable Sharpe ratio: {sharpe:.2f}")
        else:
            decision['warnings'].append(f"Low Sharpe ratio: {sharpe:.2f}")

        if abs(max_dd) <= 0.25:
            score += 2
            decision['reasons'].append(f"Acceptable max drawdown: {max_dd:.2%}")
        elif abs(max_dd) <= 0.35:
            score += 1
            decision['warnings'].append(f"High max drawdown: {max_dd:.2%}")
        else:
            decision['warnings'].append(f"Excessive max drawdown: {max_dd:.2%}")

        # Walk-forward validation
        wf = report.get('walk_forward', {})
        oos_sharpe_ratio = wf.get('oos_sharpe_ratio', 0)

        if oos_sharpe_ratio >= 0.8:
            score += 3
            decision['reasons'].append(f"Good OOS performance: {oos_sharpe_ratio:.2f}")
        elif oos_sharpe_ratio >= 0.5:
            score += 1
            decision['warnings'].append(f"Weak OOS performance: {oos_sharpe_ratio:.2f}")
        else:
            decision['warnings'].append("Poor OOS performance")

        # Statistical tests
        stats_tests = report.get('statistical_tests', {})
        normality = stats_tests.get('shapiro_normality', {})
        if normality.get('is_normal', True):
            score += 1
            decision['reasons'].append("Returns appear normally distributed")
        else:
            decision['warnings'].append("Non-normal return distribution")

        # Final decision
        decision['score'] = score
        decision['max_score'] = max_score

        if score >= 7:
            decision['approved'] = True
            decision['confidence'] = 'high'
        elif score >= 5:
            decision['approved'] = True
            decision['confidence'] = 'medium'
        else:
            decision['approved'] = False
            decision['confidence'] = 'low'

        return decision
