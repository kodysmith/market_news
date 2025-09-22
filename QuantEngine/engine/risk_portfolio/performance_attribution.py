"""
Performance Attribution and Factor Analysis for AI Quant Trading System

Implements:
- Return attribution (security selection, allocation effects)
- Risk factor decomposition
- Multi-factor model regression
- Performance contribution analysis
- Benchmark-relative attribution
- Style analysis and factor exposures
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Tuple, Union
import logging
from datetime import datetime
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import statsmodels.api as sm
from scipy import stats

logger = logging.getLogger(__name__)


class PerformanceAttribution:
    """
    Comprehensive performance attribution analysis

    Decomposes portfolio returns into components:
    - Asset allocation effect
    - Security selection effect
    - Interaction effect
    - Timing effect
    """

    def __init__(self):
        self.benchmark_weights = {}  # Benchmark portfolio weights
        self.factor_exposures = {}   # Factor loading estimates

    def brinson_attribution(self, portfolio_returns: Dict[str, pd.Series],
                          portfolio_weights: Dict[str, Dict[str, float]],
                          benchmark_weights: Dict[str, Dict[str, float]],
                          periods: List[str] = None) -> Dict[str, Any]:
        """
        Brinson attribution analysis (allocation + selection effects)

        Args:
            portfolio_returns: Security return series
            portfolio_weights: Portfolio weights by period
            benchmark_weights: Benchmark weights by period
            periods: Analysis periods (if None, uses all available)

        Returns:
            Attribution breakdown by period and total
        """

        if not periods:
            # Use common dates across all weight periods
            all_dates = set()
            for weights in portfolio_weights.values():
                all_dates.update(weights.keys())
            for weights in benchmark_weights.values():
                all_dates.update(weights.keys())
            periods = sorted(list(all_dates))

        attribution_results = {}

        for period in periods:
            port_weights = portfolio_weights.get(period, {})
            bench_weights = benchmark_weights.get(period, {})

            if not port_weights or not bench_weights:
                continue

            # Calculate returns for the period
            period_returns = {}
            for ticker in set(port_weights.keys()) | set(bench_weights.keys()):
                if ticker in portfolio_returns:
                    ret_series = portfolio_returns[ticker]
                    # Find return for this period (simplified)
                    if len(ret_series) > 0:
                        period_returns[ticker] = ret_series.iloc[-1] if len(ret_series) == 1 else ret_series.mean()

            if not period_returns:
                continue

            # Attribution calculation
            attribution = self._calculate_brinson_attribution_single_period(
                period_returns, port_weights, bench_weights
            )

            attribution_results[period] = attribution

        # Aggregate across periods
        total_attribution = self._aggregate_attribution_results(attribution_results)

        return {
            'period_attribution': attribution_results,
            'total_attribution': total_attribution,
            'method': 'brinson',
            'periods_analyzed': len(attribution_results)
        }

    def _calculate_brinson_attribution_single_period(self, returns: Dict[str, float],
                                                   port_weights: Dict[str, float],
                                                   bench_weights: Dict[str, float]) -> Dict[str, Any]:
        """Calculate Brinson attribution for a single period"""

        # Get overlapping securities
        common_securities = set(port_weights.keys()) & set(bench_weights.keys()) & set(returns.keys())

        if not common_securities:
            return {'error': 'No overlapping securities with returns'}

        # Portfolio return
        port_return = sum(port_weights.get(sec, 0) * returns.get(sec, 0) for sec in common_securities)

        # Benchmark return
        bench_return = sum(bench_weights.get(sec, 0) * returns.get(sec, 0) for sec in common_securities)

        # Active return
        active_return = port_return - bench_return

        # Allocation effect: (wp - wb) * rb
        allocation_effect = sum(
            (port_weights.get(sec, 0) - bench_weights.get(sec, 0)) * returns.get(sec, 0)
            for sec in common_securities
        )

        # Selection effect: wb * (rp - rb)
        selection_effect = sum(
            bench_weights.get(sec, 0) * (returns.get(sec, 0) - returns.get(sec, 0))
            for sec in common_securities
        )

        # For multi-period, selection would be: wb * (rp - rb) where rb is benchmark return
        # Simplified for single period
        selection_effect = sum(
            bench_weights.get(sec, 0) * (returns.get(sec, 0) - bench_return)
            for sec in common_securities
        )

        # Interaction effect (residual)
        interaction_effect = active_return - allocation_effect - selection_effect

        return {
            'portfolio_return': port_return,
            'benchmark_return': bench_return,
            'active_return': active_return,
            'allocation_effect': allocation_effect,
            'selection_effect': selection_effect,
            'interaction_effect': interaction_effect,
            'securities_analyzed': len(common_securities)
        }

    def _aggregate_attribution_results(self, period_results: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        """Aggregate attribution results across periods"""

        if not period_results:
            return {}

        # Sum effects across periods
        total_portfolio_return = sum(r.get('portfolio_return', 0) for r in period_results.values())
        total_benchmark_return = sum(r.get('benchmark_return', 0) for r in period_results.values())
        total_active_return = sum(r.get('active_return', 0) for r in period_results.values())

        total_allocation = sum(r.get('allocation_effect', 0) for r in period_results.values())
        total_selection = sum(r.get('selection_effect', 0) for r in period_results.values())
        total_interaction = sum(r.get('interaction_effect', 0) for r in period_results.values())

        # Attribution quality metrics
        attribution_explanation = total_allocation + total_selection + total_interaction
        unexplained = total_active_return - attribution_explanation

        return {
            'total_portfolio_return': total_portfolio_return,
            'total_benchmark_return': total_benchmark_return,
            'total_active_return': total_active_return,
            'total_allocation_effect': total_allocation,
            'total_selection_effect': total_selection,
            'total_interaction_effect': total_interaction,
            'attribution_quality': attribution_explanation / total_active_return if total_active_return != 0 else 0,
            'unexplained_return': unexplained,
            'periods_contributing': len([r for r in period_results.values() if r.get('active_return', 0) > 0])
        }


class FactorModelAttribution:
    """
    Multi-factor model performance attribution

    Implements Fama-French and custom factor models to decompose returns
    """

    def __init__(self):
        # Standard factor definitions
        self.standard_factors = {
            'MKT': 'Market excess return (RM - RF)',
            'SMB': 'Small minus Big (size factor)',
            'HML': 'High minus Low (value factor)',
            'MOM': 'Momentum factor',
            'BAB': 'Betting Against Beta',
            'QMJ': 'Quality minus Junk',
            'VOL': 'Volatility factor'
        }

    def fama_french_attribution(self, portfolio_returns: pd.Series,
                              factor_returns: Dict[str, pd.Series],
                              risk_free_rate: pd.Series = None) -> Dict[str, Any]:
        """
        Fama-French 3-factor model attribution

        Args:
            portfolio_returns: Portfolio return series
            factor_returns: Dictionary with MKT, SMB, HML factor returns
            risk_free_rate: Risk-free rate series

        Returns:
            Factor attribution results
        """

        # Align all series
        common_index = portfolio_returns.index
        for factor_name, factor_series in factor_returns.items():
            common_index = common_index.intersection(factor_series.index)

        if len(common_index) < 20:
            return {'error': 'Insufficient overlapping data for factor analysis'}

        # Align series
        port_ret = portfolio_returns.loc[common_index]
        factors = {}

        for factor_name in ['MKT', 'SMB', 'HML']:
            if factor_name in factor_returns:
                factors[factor_name] = factor_returns[factor_name].loc[common_index]
            else:
                return {'error': f'Missing required factor: {factor_name}'}

        # Excess returns
        if risk_free_rate is not None:
            rf = risk_free_rate.loc[common_index]
            excess_returns = port_ret - rf
            factor_excess = {name: factors[name] - rf for name, factors[name] in factors.items()}
        else:
            excess_returns = port_ret
            factor_excess = factors

        # Run regression
        X = pd.DataFrame(factor_excess)
        X = sm.add_constant(X)  # Add intercept

        model = sm.OLS(excess_returns, X)
        results = model.fit()

        # Extract factor exposures and contributions
        factor_loadings = {}
        factor_contributions = {}
        total_explained = 0

        for i, factor_name in enumerate(['MKT', 'SMB', 'HML']):
            loading = results.params.iloc[i+1]  # Skip constant
            factor_loadings[factor_name] = loading

            # Annualized contribution
            avg_factor_return = factors[factor_name].mean() * 12  # Monthly to annual
            contribution = loading * avg_factor_return
            factor_contributions[factor_name] = contribution
            total_explained += contribution

        # Alpha and unexplained return
        alpha = results.params.iloc[0] * 12  # Annualized alpha
        unexplained = excess_returns.mean() * 12 - total_explained

        return {
            'model': 'fama_french_3_factor',
            'factor_loadings': factor_loadings,
            'factor_contributions': factor_contributions,
            'alpha': alpha,
            'unexplained_return': unexplained,
            'total_explained': total_explained,
            'r_squared': results.rsquared,
            'r_squared_adj': results.rsquared_adj,
            'f_statistic': results.fvalue,
            'observations': len(excess_returns),
            'regression_summary': {
                'coefficients': results.params.to_dict(),
                't_statistics': results.tvalues.to_dict(),
                'p_values': results.pvalues.to_dict(),
                'confidence_intervals': results.conf_int().to_dict()
            }
        }

    def custom_factor_attribution(self, portfolio_returns: pd.Series,
                                factor_returns: Dict[str, pd.Series],
                                factor_names: List[str] = None) -> Dict[str, Any]:
        """
        Custom multi-factor attribution analysis

        Args:
            portfolio_returns: Portfolio return series
            factor_returns: Dictionary of factor return series
            factor_names: List of factor names to include

        Returns:
            Custom factor attribution results
        """

        if factor_names is None:
            factor_names = list(factor_returns.keys())

        # Align all series
        common_index = portfolio_returns.index
        for factor_name in factor_names:
            if factor_name in factor_returns:
                common_index = common_index.intersection(factor_returns[factor_name].index)

        if len(common_index) < 20:
            return {'error': 'Insufficient data for factor analysis'}

        # Prepare data
        y = portfolio_returns.loc[common_index]
        X = pd.DataFrame({name: factor_returns[name].loc[common_index]
                         for name in factor_names if name in factor_returns})
        X = sm.add_constant(X)

        # Run regression
        model = sm.OLS(y, X)
        results = model.fit()

        # Factor contributions (simplified - assumes factors are in same units)
        factor_contributions = {}
        total_explained = 0

        for factor_name in factor_names:
            if factor_name in factor_returns:
                loading = results.params.get(factor_name, 0)
                avg_factor_return = factor_returns[factor_name].loc[common_index].mean()
                contribution = loading * avg_factor_return
                factor_contributions[factor_name] = contribution
                total_explained += contribution

        alpha = results.params.get('const', 0)

        return {
            'model': 'custom_multi_factor',
            'factors_used': factor_names,
            'factor_contributions': factor_contributions,
            'alpha': alpha,
            'total_explained': total_explained,
            'unexplained_return': y.mean() - total_explained,
            'r_squared': results.rsquared,
            'observations': len(y),
            'factor_loadings': {name: results.params.get(name, 0) for name in factor_names}
        }


class RiskAttribution:
    """
    Risk decomposition and attribution analysis

    Breaks down portfolio risk into factor contributions
    """

    def __init__(self):
        pass

    def risk_decomposition(self, portfolio_weights: Dict[str, float],
                          covariance_matrix: pd.DataFrame,
                          factor_exposures: Dict[str, Dict[str, float]] = None) -> Dict[str, Any]:
        """
        Decompose portfolio risk into components

        Args:
            portfolio_weights: Portfolio weights
            covariance_matrix: Asset covariance matrix
            factor_exposures: Factor exposures by asset (optional)

        Returns:
            Risk decomposition results
        """

        # Portfolio volatility
        weights = np.array([portfolio_weights.get(ticker, 0) for ticker in covariance_matrix.index])
        portfolio_variance = weights.T @ covariance_matrix.values @ weights
        portfolio_vol = np.sqrt(portfolio_variance)

        # Marginal contributions to risk
        marginal_risk = covariance_matrix.values @ weights
        risk_contributions = weights * marginal_risk

        # Percentage contributions
        total_risk = portfolio_vol
        risk_contributions_pct = risk_contributions / total_risk if total_risk > 0 else risk_contributions * 0

        # Diversification ratio
        weighted_vol_sum = np.sum(weights * np.sqrt(np.diag(covariance_matrix.values)))
        diversification_ratio = weighted_vol_sum / portfolio_vol if portfolio_vol > 0 else 1

        decomposition = {
            'portfolio_volatility': portfolio_vol,
            'total_variance': portfolio_variance,
            'asset_risk_contributions': dict(zip(covariance_matrix.index, risk_contributions)),
            'asset_risk_contributions_pct': dict(zip(covariance_matrix.index, risk_contributions_pct)),
            'diversification_ratio': diversification_ratio,
            'largest_risk_contributor': max(zip(covariance_matrix.index, risk_contributions),
                                          key=lambda x: x[1]),
            'most_diversified': diversification_ratio > 1.2  # Arbitrary threshold
        }

        # Factor-based risk attribution (if factor exposures provided)
        if factor_exposures:
            factor_risk = self._factor_risk_attribution(portfolio_weights, covariance_matrix, factor_exposures)
            decomposition.update(factor_risk)

        return decomposition

    def _factor_risk_attribution(self, portfolio_weights: Dict[str, float],
                               covariance_matrix: pd.DataFrame,
                               factor_exposures: Dict[str, Dict[str, float]]) -> Dict[str, Any]:
        """Factor-based risk attribution"""

        # This is a simplified implementation
        # Real factor risk attribution would use a full factor model

        factor_contributions = {}
        total_factor_risk = 0

        # Simplified: assume each factor contributes independently
        for factor_name, exposures in factor_exposures.items():
            factor_variance = 0
            for ticker, exposure in exposures.items():
                weight = portfolio_weights.get(ticker, 0)
                # Simplified factor risk calculation
                factor_variance += (weight * exposure) ** 2

            factor_vol = np.sqrt(factor_variance)
            factor_contributions[factor_name] = factor_vol
            total_factor_risk += factor_vol

        return {
            'factor_risk_contributions': factor_contributions,
            'total_factor_risk': total_factor_risk,
            'systematic_risk_ratio': total_factor_risk / np.sqrt(np.sum(list(portfolio_weights.values()))**2)
        }


# Test functions
def test_performance_attribution():
    """Test performance attribution functionality"""

    print("🧪 Testing Performance Attribution")

    # Create mock data
    dates = pd.date_range('2023-01-01', '2023-12-31', freq='M')
    np.random.seed(42)

    # Mock portfolio and benchmark weights (simplified - single period)
    portfolio_weights = {
        '2023-12-31': {'SPY': 0.6, 'QQQ': 0.3, 'TQQQ': 0.1}
    }

    benchmark_weights = {
        '2023-12-31': {'SPY': 0.7, 'QQQ': 0.2, 'TQQQ': 0.1}
    }

    # Mock returns
    portfolio_returns = {
        'SPY': pd.Series(np.random.normal(0.01, 0.05, len(dates)), index=dates),
        'QQQ': pd.Series(np.random.normal(0.012, 0.06, len(dates)), index=dates),
        'TQQQ': pd.Series(np.random.normal(0.025, 0.12, len(dates)), index=dates)
    }

    # Test Brinson attribution
    print("📊 Testing Brinson Attribution...")

    attr = PerformanceAttribution()
    brinson_results = attr.brinson_attribution(
        portfolio_returns, portfolio_weights, benchmark_weights, ['2023-12-31']
    )

    if 'total_attribution' in brinson_results:
        total_attr = brinson_results['total_attribution']
        print(".2%")
        print(".2%")
        print(".2%")

    # Test factor attribution (simplified)
    print("\n📈 Testing Factor Attribution...")

    # Mock factor returns
    factor_returns = {
        'MKT': pd.Series(np.random.normal(0.008, 0.04, len(dates)), index=dates),
        'SMB': pd.Series(np.random.normal(0.002, 0.02, len(dates)), index=dates),
        'HML': pd.Series(np.random.normal(0.003, 0.025, len(dates)), index=dates)
    }

    portfolio_returns_series = pd.Series(np.random.normal(0.015, 0.08, len(dates)), index=dates)

    factor_attr = FactorModelAttribution()
    fama_french_results = factor_attr.fama_french_attribution(
        portfolio_returns_series, factor_returns
    )

    if 'factor_loadings' in fama_french_results:
        loadings = fama_french_results['factor_loadings']
        print(".2f")
        print(".2f")
        print(".2f")
        print(".3f")

    print("\n✅ Performance attribution tests completed!")

    return {
        'brinson': brinson_results,
        'fama_french': fama_french_results
    }


if __name__ == "__main__":
    test_performance_attribution()

