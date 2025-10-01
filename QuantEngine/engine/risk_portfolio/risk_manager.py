"""
Advanced Risk Management for AI Quant Trading System

Implements:
- Value at Risk (VaR) calculations (parametric, historical, Monte Carlo)
- Expected Shortfall (CVaR)
- Stress testing scenarios
- Dynamic position sizing
- Portfolio optimization
- Risk attribution and decomposition
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Tuple, Union
import logging
from datetime import datetime, timedelta
from scipy import stats
from scipy.optimize import minimize
import warnings

logger = logging.getLogger(__name__)


class RiskManager:
    """
    Advanced risk management system for portfolio and position-level risk control
    """

    def __init__(self, config: Dict[str, Any]):
        self.config = config

        # Risk limits
        self.max_portfolio_var = config.get('max_portfolio_var', 0.02)  # 2% daily VaR limit
        self.max_position_var = config.get('max_position_var', 0.05)   # 5% position VaR limit
        self.max_drawdown_limit = config.get('max_drawdown_limit', 0.20)  # 20% drawdown limit
        self.confidence_level = config.get('confidence_level', 0.95)   # 95% confidence for VaR
        self.var_horizon_days = config.get('var_horizon_days', 1)      # 1-day VaR

        # Position sizing parameters
        self.kelly_fraction = config.get('kelly_fraction', 0.5)        # 50% Kelly fraction
        self.max_leverage = config.get('max_leverage', 2.0)            # 2x max leverage

        # Stress testing scenarios
        self.stress_scenarios = self._define_stress_scenarios()

    def calculate_portfolio_var(self, portfolio: Dict[str, float], returns_data: Dict[str, pd.Series],
                               method: str = 'parametric', confidence: float = 0.95) -> Dict[str, Any]:
        """
        Calculate portfolio Value at Risk using multiple methods

        Args:
            portfolio: Dictionary of ticker -> weight mappings
            returns_data: Dictionary of ticker -> returns series
            method: 'parametric', 'historical', or 'monte_carlo'
            confidence: Confidence level (0.95 = 95%)

        Returns:
            Dictionary with VaR calculations and risk metrics
        """

        if not portfolio or not returns_data:
            return {'error': 'Insufficient data for VaR calculation'}

        # Align all return series to common dates
        returns_df = pd.DataFrame(returns_data).dropna()

        if returns_df.empty or len(returns_df) < 30:
            return {'error': 'Insufficient historical data'}

        weights = np.array([portfolio.get(ticker, 0) for ticker in returns_df.columns])

        if np.sum(np.abs(weights)) == 0:
            return {'error': 'Zero portfolio weights'}

        # Normalize weights
        weights = weights / np.sum(np.abs(weights))

        if method == 'parametric':
            return self._parametric_var(returns_df, weights, confidence)
        elif method == 'historical':
            return self._historical_var(returns_df, weights, confidence)
        elif method == 'monte_carlo':
            return self._monte_carlo_var(returns_df, weights, confidence)
        else:
            return {'error': f'Unknown VaR method: {method}'}

    def _parametric_var(self, returns_df: pd.DataFrame, weights: np.ndarray,
                       confidence: float) -> Dict[str, Any]:
        """Calculate parametric VaR assuming normal distribution"""

        # Calculate portfolio returns
        portfolio_returns = returns_df.dot(weights)

        # Portfolio mean and volatility
        mu = portfolio_returns.mean()
        sigma = portfolio_returns.std()

        # VaR using normal distribution
        z_score = stats.norm.ppf(confidence)
        var = -(mu * self.var_horizon_days + z_score * sigma * np.sqrt(self.var_horizon_days))

        # Expected Shortfall (CVaR)
        es_z = stats.norm.pdf(z_score) / (1 - confidence)
        expected_shortfall = -(mu * self.var_horizon_days + es_z * sigma * np.sqrt(self.var_horizon_days))

        # Component VaR (risk contribution by asset)
        marginal_var = z_score * sigma * returns_df.cov().dot(weights) / (sigma * np.sqrt(self.var_horizon_days))
        component_var = weights * marginal_var

        return {
            'method': 'parametric',
            'confidence': confidence,
            'var_1d': var,
            'var_annual': var * np.sqrt(252),
            'expected_shortfall': expected_shortfall,
            'portfolio_volatility': sigma * np.sqrt(252),
            'component_var': dict(zip(returns_df.columns, component_var)),
            'diversification_ratio': np.sum(component_var) / var if var != 0 else 0
        }

    def _historical_var(self, returns_df: pd.DataFrame, weights: np.ndarray,
                       confidence: float) -> Dict[str, Any]:
        """Calculate historical simulation VaR"""

        # Calculate portfolio returns
        portfolio_returns = returns_df.dot(weights)

        # Sort returns and find percentile
        sorted_returns = np.sort(portfolio_returns.values)
        var_index = int((1 - confidence) * len(sorted_returns))
        var = -sorted_returns[var_index]

        # Expected Shortfall (average of losses beyond VaR)
        tail_losses = sorted_returns[:var_index]
        expected_shortfall = -np.mean(tail_losses) if len(tail_losses) > 0 else var

        return {
            'method': 'historical',
            'confidence': confidence,
            'var_1d': var,
            'var_annual': var * np.sqrt(252),
            'expected_shortfall': expected_shortfall,
            'sample_size': len(portfolio_returns)
        }

    def _monte_carlo_var(self, returns_df: pd.DataFrame, weights: np.ndarray,
                        confidence: float, n_simulations: int = 10000) -> Dict[str, Any]:
        """Calculate Monte Carlo VaR"""

        # Fit multivariate normal distribution
        mu = returns_df.mean().values
        cov = returns_df.cov().values

        # Generate random scenarios
        np.random.seed(42)  # For reproducibility
        scenarios = np.random.multivariate_normal(mu, cov, n_simulations)

        # Calculate portfolio returns for each scenario
        portfolio_scenarios = scenarios.dot(weights)

        # Calculate VaR
        sorted_scenarios = np.sort(portfolio_scenarios)
        var_index = int((1 - confidence) * n_simulations)
        var = -sorted_scenarios[var_index]

        # Expected Shortfall
        tail_losses = sorted_scenarios[:var_index]
        expected_shortfall = -np.mean(tail_losses) if len(tail_losses) > 0 else var

        return {
            'method': 'monte_carlo',
            'confidence': confidence,
            'var_1d': var,
            'var_annual': var * np.sqrt(252),
            'expected_shortfall': expected_shortfall,
            'n_simulations': n_simulations
        }

    def dynamic_position_sizing(self, strategy_return: pd.Series, current_capital: float,
                              target_vol: float = 0.15) -> Dict[str, Any]:
        """
        Calculate dynamic position sizing based on Kelly criterion and risk limits

        Args:
            strategy_return: Historical strategy returns
            current_capital: Current portfolio capital
            target_vol: Target annualized volatility

        Returns:
            Position sizing recommendations
        """

        if len(strategy_return) < 30:
            return {'error': 'Insufficient return history for position sizing'}

        # Calculate strategy statistics
        mu = strategy_return.mean()
        sigma = strategy_return.std()
        ann_mu = mu * 252
        ann_sigma = sigma * np.sqrt(252)

        # Sharpe ratio
        sharpe = ann_mu / ann_sigma if ann_sigma > 0 else 0

        # Kelly fraction sizing
        kelly_fraction = (ann_mu / (ann_sigma ** 2)) * self.kelly_fraction  # Conservative Kelly
        kelly_fraction = np.clip(kelly_fraction, 0, self.max_leverage)

        # Volatility-targeted sizing
        current_vol = ann_sigma
        vol_scalar = target_vol / current_vol if current_vol > 0 else 1.0
        vol_target_fraction = vol_scalar * self.kelly_fraction
        vol_target_fraction = np.clip(vol_target_fraction, 0, self.max_leverage)

        # Risk parity approach (equal risk contribution)
        risk_parity_fraction = target_vol / ann_sigma if ann_sigma > 0 else 0
        risk_parity_fraction = np.clip(risk_parity_fraction, 0, self.max_leverage)

        # Recommended position size
        recommended_fraction = min(kelly_fraction, vol_target_fraction, risk_parity_fraction)
        recommended_fraction = min(recommended_fraction, self.max_leverage)

        position_size = current_capital * recommended_fraction

        return {
            'strategy_mu': ann_mu,
            'strategy_sigma': ann_sigma,
            'strategy_sharpe': sharpe,
            'kelly_fraction': kelly_fraction,
            'vol_target_fraction': vol_target_fraction,
            'risk_parity_fraction': risk_parity_fraction,
            'recommended_fraction': recommended_fraction,
            'recommended_position_size': position_size,
            'max_position_size': current_capital * self.max_leverage,
            'sizing_method': 'dynamic_kelly_vol_target'
        }

    def stress_test_portfolio(self, portfolio: Dict[str, float], returns_data: Dict[str, pd.Series],
                            scenarios: Optional[List[Dict[str, Any]]] = None) -> Dict[str, Any]:
        """
        Run stress tests on portfolio under various market scenarios

        Args:
            portfolio: Current portfolio weights
            returns_data: Historical returns data
            scenarios: Custom stress scenarios (optional)

        Returns:
            Stress test results for each scenario
        """

        if scenarios is None:
            scenarios = self.stress_scenarios

        results = {}

        for scenario in scenarios:
            scenario_name = scenario['name']
            shock_returns = self._apply_scenario_shock(returns_data, scenario)

            # Calculate portfolio impact
            portfolio_impact = self._calculate_portfolio_impact(portfolio, shock_returns, scenario)

            results[scenario_name] = {
                'scenario': scenario,
                'portfolio_impact': portfolio_impact,
                'breaches': self._check_risk_breaches(portfolio_impact)
            }

        return results

    def _define_stress_scenarios(self) -> List[Dict[str, Any]]:
        """Define standard stress testing scenarios"""

        return [
            {
                'name': '2020_covid_crash',
                'description': 'March 2020 COVID-19 market crash',
                'shocks': {
                    'SPY': -0.34,  # -34% in March 2020
                    'QQQ': -0.37,
                    'TQQQ': -0.65,  # Leveraged moves more
                    'XLE': -0.55,  # Energy sector crash
                    'XLF': -0.45,  # Financials crash
                },
                'type': 'historical'
            },
            {
                'name': 'dot_com_bubble',
                'description': '2000-2002 dot-com bubble burst',
                'shocks': {
                    'SPY': -0.13,
                    'QQQ': -0.22,  # Tech heavy
                    'TQQQ': -0.40,
                    'XLK': -0.25,  # Tech sector
                },
                'type': 'historical'
            },
            {
                'name': 'fed_rate_hike',
                'description': 'Sudden 1% Fed rate hike',
                'shocks': {
                    'XLF': -0.08,  # Banks hurt by rate hikes
                    'XLE': 0.05,   # Energy benefits from higher rates
                },
                'type': 'macro'
            },
            {
                'name': 'tech_sector_dump',
                'description': '20% tech sector decline',
                'shocks': {
                    'QQQ': -0.20,
                    'TQQQ': -0.40,
                    'XLK': -0.20,
                },
                'type': 'sector'
            },
            {
                'name': 'global_recession',
                'description': 'Global recession scenario',
                'shocks': {
                    'SPY': -0.25,
                    'QQQ': -0.30,
                    'TQQQ': -0.60,
                    'XLE': -0.40,
                    'XLF': -0.35,
                },
                'type': 'systemic'
            }
        ]

    def _apply_scenario_shock(self, returns_data: Dict[str, pd.Series],
                             scenario: Dict[str, Any]) -> Dict[str, float]:
        """Apply scenario shocks to return distributions"""

        shocked_returns = {}

        for ticker in returns_data.keys():
            shock = scenario['shocks'].get(ticker, 0)  # Default to 0 if not specified

            # For historical scenarios, use the actual shock
            # For hypothetical scenarios, this represents expected move
            shocked_returns[ticker] = shock

        return shocked_returns

    def _calculate_portfolio_impact(self, portfolio: Dict[str, float],
                                  shocked_returns: Dict[str, float],
                                  scenario: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate portfolio impact under stress scenario"""

        # Calculate portfolio P&L under scenario
        portfolio_pnl = 0.0
        asset_impacts = {}

        for ticker, weight in portfolio.items():
            shock = shocked_returns.get(ticker, 0)
            asset_impact = weight * shock
            asset_impacts[ticker] = asset_impact
            portfolio_pnl += asset_impact

        return {
            'portfolio_pnl': portfolio_pnl,
            'asset_impacts': asset_impacts,
            'scenario_type': scenario.get('type', 'unknown'),
            'worst_asset': min(asset_impacts.items(), key=lambda x: x[1]) if asset_impacts else None,
            'best_asset': max(asset_impacts.items(), key=lambda x: x[1]) if asset_impacts else None
        }

    def _check_risk_breaches(self, portfolio_impact: Dict[str, Any]) -> List[str]:
        """Check if stress scenario breaches risk limits"""

        breaches = []
        portfolio_pnl = portfolio_impact['portfolio_pnl']

        if abs(portfolio_pnl) > self.max_portfolio_var:
            breaches.append(f"Portfolio VaR breach: {portfolio_pnl:.1%} > {self.max_portfolio_var:.1%}")

        if portfolio_pnl < -self.max_drawdown_limit:
            breaches.append(f"Drawdown breach: {portfolio_pnl:.1%} < -{self.max_drawdown_limit:.1%}")

        return breaches

    def optimize_portfolio(self, returns_data: Dict[str, pd.Series], target_return: Optional[float] = None,
                          method: str = 'min_variance') -> Dict[str, Any]:
        """
        Optimize portfolio weights using modern portfolio theory

        Args:
            returns_data: Historical returns for each asset
            target_return: Target portfolio return (optional)
            method: 'min_variance', 'max_sharpe', or 'risk_parity'

        Returns:
            Optimal portfolio weights and metrics
        """

        returns_df = pd.DataFrame(returns_data).dropna()

        if returns_df.empty or len(returns_df) < 30:
            return {'error': 'Insufficient data for portfolio optimization'}

        mu = returns_df.mean().values * 252  # Annualized
        cov = returns_df.cov().values * 252  # Annualized

        n_assets = len(returns_df.columns)

        if method == 'min_variance':
            return self._min_variance_optimization(cov, n_assets, returns_df.columns)
        elif method == 'max_sharpe':
            return self._max_sharpe_optimization(mu, cov, n_assets, returns_df.columns)
        elif method == 'risk_parity':
            return self._risk_parity_optimization(cov, n_assets, returns_df.columns)
        else:
            return {'error': f'Unknown optimization method: {method}'}

    def _min_variance_optimization(self, cov: np.ndarray, n_assets: int, tickers: pd.Index) -> Dict[str, Any]:
        """Minimum variance portfolio optimization"""

        def objective(weights):
            return weights.T @ cov @ weights

        constraints = [
            {'type': 'eq', 'fun': lambda x: np.sum(x) - 1}  # Weights sum to 1
        ]

        bounds = [(0, 1) for _ in range(n_assets)]  # Long only

        result = minimize(objective, np.ones(n_assets) / n_assets,
                         method='SLSQP', bounds=bounds, constraints=constraints)

        if result.success:
            weights = result.x
            portfolio_vol = np.sqrt(weights.T @ cov @ weights)

            return {
                'method': 'min_variance',
                'weights': dict(zip(tickers, weights)),
                'portfolio_volatility': portfolio_vol,
                'optimization_success': True
            }
        else:
            return {'error': 'Optimization failed', 'message': result.message}

    def _max_sharpe_optimization(self, mu: np.ndarray, cov: np.ndarray, n_assets: int, tickers: pd.Index) -> Dict[str, Any]:
        """Maximum Sharpe ratio portfolio optimization"""

        def objective(weights):
            portfolio_return = weights.T @ mu
            portfolio_vol = np.sqrt(weights.T @ cov @ weights)
            return -portfolio_return / portfolio_vol  # Negative for minimization

        constraints = [
            {'type': 'eq', 'fun': lambda x: np.sum(x) - 1}
        ]

        bounds = [(0, 1) for _ in range(n_assets)]

        result = minimize(objective, np.ones(n_assets) / n_assets,
                         method='SLSQP', bounds=bounds, constraints=constraints)

        if result.success:
            weights = result.x
            portfolio_return = weights.T @ mu
            portfolio_vol = np.sqrt(weights.T @ cov @ weights)
            sharpe = portfolio_return / portfolio_vol

            return {
                'method': 'max_sharpe',
                'weights': dict(zip(tickers, weights)),
                'portfolio_return': portfolio_return,
                'portfolio_volatility': portfolio_vol,
                'sharpe_ratio': sharpe,
                'optimization_success': True
            }
        else:
            return {'error': 'Optimization failed', 'message': result.message}

    def _risk_parity_optimization(self, cov: np.ndarray, n_assets: int, tickers: pd.Index) -> Dict[str, Any]:
        """Risk parity portfolio optimization (equal risk contribution)"""

        def objective(weights):
            # Risk contribution of each asset
            portfolio_vol = np.sqrt(weights.T @ cov @ weights)
            marginal_risk = cov @ weights
            risk_contribution = weights * marginal_risk / portfolio_vol

            # Objective: minimize variance of risk contributions
            return np.var(risk_contribution)

        constraints = [
            {'type': 'eq', 'fun': lambda x: np.sum(x) - 1}
        ]

        bounds = [(0.01, 1) for _ in range(n_assets)]  # Minimum 1% per asset

        result = minimize(objective, np.ones(n_assets) / n_assets,
                         method='SLSQP', bounds=bounds, constraints=constraints)

        if result.success:
            weights = result.x
            portfolio_vol = np.sqrt(weights.T @ cov @ weights)

            # Calculate risk contributions
            marginal_risk = cov @ weights
            risk_contribution = weights * marginal_risk / portfolio_vol

            return {
                'method': 'risk_parity',
                'weights': dict(zip(tickers, weights)),
                'portfolio_volatility': portfolio_vol,
                'risk_contributions': dict(zip(tickers, risk_contribution)),
                'optimization_success': True
            }
        else:
            return {'error': 'Optimization failed', 'message': result.message}

    def risk_attribution(self, portfolio_returns: pd.Series, asset_returns: Dict[str, pd.Series],
                        weights: Dict[str, float]) -> Dict[str, Any]:
        """
        Perform risk attribution analysis to understand sources of portfolio risk

        Args:
            portfolio_returns: Portfolio return series
            asset_returns: Individual asset return series
            weights: Portfolio weights

        Returns:
            Risk attribution breakdown
        """

        if len(portfolio_returns) < 30:
            return {'error': 'Insufficient data for risk attribution'}

        # Calculate portfolio volatility
        portfolio_vol = portfolio_returns.std() * np.sqrt(252)

        # Calculate asset volatilities and correlations
        asset_vols = {}
        correlations = {}

        for ticker, returns in asset_returns.items():
            if ticker in weights and len(returns) > 30:
                asset_vols[ticker] = returns.std() * np.sqrt(252)
                correlations[ticker] = returns.corr(portfolio_returns)

        # Risk attribution using Euler decomposition
        total_risk = portfolio_vol
        specific_risk = 0
        systematic_risk = 0

        for ticker, weight in weights.items():
            if ticker in asset_vols:
                asset_vol = asset_vols[ticker]
                corr = correlations.get(ticker, 0)

                # Marginal contribution to risk
                marginal_risk = weight * asset_vol * corr

                if corr > 0.8:  # High correlation = systematic risk
                    systematic_risk += marginal_risk
                else:  # Low correlation = specific risk
                    specific_risk += marginal_risk

        # Diversification effect
        weighted_vol_sum = sum(weights.get(ticker, 0) * asset_vols.get(ticker, 0)
                              for ticker in weights.keys() if ticker in asset_vols)

        diversification_ratio = weighted_vol_sum / portfolio_vol if portfolio_vol > 0 else 0

        return {
            'portfolio_volatility': portfolio_vol,
            'systematic_risk': systematic_risk,
            'specific_risk': specific_risk,
            'diversification_ratio': diversification_ratio,
            'asset_risk_contributions': {
                ticker: weights[ticker] * asset_vols[ticker] * correlations[ticker]
                for ticker in weights.keys()
                if ticker in asset_vols and ticker in correlations
            },
            'top_risk_contributors': sorted(
                [(ticker, weights[ticker] * asset_vols[ticker] * correlations[ticker])
                 for ticker in weights.keys()
                 if ticker in asset_vols and ticker in correlations],
                key=lambda x: x[1], reverse=True
            )[:5]  # Top 5
        }


# Test functions
def test_risk_management():
    """Test risk management functionality"""

    print("🧪 Testing Advanced Risk Management")

    # Mock data
    dates = pd.date_range('2020-01-01', '2024-01-01', freq='D')
    np.random.seed(42)

    n_days = len(dates)
    spy_returns = np.random.normal(0.0001, 0.02, n_days)
    qqq_returns = np.random.normal(0.00015, 0.025, n_days)
    tqqq_returns = np.random.normal(0.0003, 0.05, n_days)  # More volatile

    returns_data = {
        'SPY': pd.Series(spy_returns, index=dates),
        'QQQ': pd.Series(qqq_returns, index=dates),
        'TQQQ': pd.Series(tqqq_returns, index=dates)
    }

    portfolio = {'SPY': 0.4, 'QQQ': 0.4, 'TQQQ': 0.2}

    config = {
        'max_portfolio_var': 0.02,
        'confidence_level': 0.95,
        'kelly_fraction': 0.5
    }

    risk_manager = RiskManager(config)

    # Test VaR calculations
    print("📊 Testing VaR Calculations...")

    parametric_var = risk_manager.calculate_portfolio_var(portfolio, returns_data, 'parametric')
    historical_var = risk_manager.calculate_portfolio_var(portfolio, returns_data, 'historical')
    mc_var = risk_manager.calculate_portfolio_var(portfolio, returns_data, 'monte_carlo')

    print(".2%")
    print(".2%")
    print(".2%")

    # Test position sizing
    print("\n📏 Testing Dynamic Position Sizing...")

    strategy_returns = pd.Series(np.random.normal(0.001, 0.03, 252), index=pd.date_range('2023-01-01', periods=252))
    sizing = risk_manager.dynamic_position_sizing(strategy_returns, 100000, 0.15)

    print(".1%")
    print(f"   Kelly Fraction: {sizing['kelly_fraction']:.1%}")
    print(".1%")

    # Test stress testing
    print("\n🔥 Testing Stress Testing...")

    stress_results = risk_manager.stress_test_portfolio(portfolio, returns_data)
    covid_stress = stress_results.get('2020_covid_crash', {})

    if 'portfolio_impact' in covid_stress:
        impact = covid_stress['portfolio_impact']
        print(".1%")
        if 'breaches' in covid_stress and covid_stress['breaches']:
            print(f"   Risk Breaches: {len(covid_stress['breaches'])} detected")

    print("\n✅ Risk management tests completed!")

    return {
        'parametric_var': parametric_var,
        'historical_var': historical_var,
        'monte_carlo_var': mc_var,
        'position_sizing': sizing,
        'stress_test': stress_results
    }


if __name__ == "__main__":
    test_risk_management()


