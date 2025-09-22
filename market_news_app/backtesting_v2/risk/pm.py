#!/usr/bin/env python3
"""
Risk and Portfolio Management Layer

Handles position sizing, risk limits, drawdown controls, and portfolio optimization.
Implements Kelly criterion, volatility targeting, and regime-aware risk management.
"""

import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar
from scipy.stats import norm
import warnings
warnings.filterwarnings('ignore')

from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime, timedelta
from dataclasses import dataclass

@dataclass
class RiskLimits:
    """Risk limit configuration"""
    max_drawdown: float = 0.20  # 20% max drawdown
    max_volatility: float = 0.25  # 25% annualized volatility target
    max_leverage: float = 2.0  # 2x max leverage
    max_concentration: float = 0.10  # 10% max position size
    var_limit: float = 0.05  # 5% VaR limit (95% confidence)
    stress_test_threshold: float = 0.15  # 15% stress loss threshold

@dataclass
class PortfolioState:
    """Current portfolio state"""
    capital: float
    positions: Dict[str, float]  # symbol -> position_size
    current_value: float
    peak_value: float
    current_drawdown: float
    volatility_target: float
    current_volatility: float

class RiskManager:
    """Risk management and position sizing"""

    def __init__(self, risk_limits: RiskLimits = None):
        self.risk_limits = risk_limits or RiskLimits()

    def calculate_kelly_fraction(self, win_rate: float, win_loss_ratio: float) -> float:
        """
        Calculate Kelly criterion position size

        Args:
            win_rate: Probability of winning trades
            win_loss_ratio: Average win / average loss ratio

        Returns:
            Kelly fraction (0-1)
        """
        if win_rate <= 0 or win_rate >= 1 or win_loss_ratio <= 0:
            return 0.0

        kelly = win_rate - (1 - win_rate) / win_loss_ratio
        return max(0.0, min(kelly, 1.0))  # Bound between 0 and 1

    def calculate_optimal_f(self, returns: pd.Series, confidence_level: float = 0.95) -> float:
        """
        Calculate optimal f using historical returns (deflated Kelly)

        Args:
            returns: Historical returns series
            confidence_level: Confidence level for Kelly fraction

        Returns:
            Optimal position size fraction
        """
        if len(returns) < 30:
            return 0.02  # Conservative default

        # Calculate win rate and win/loss ratio
        wins = returns[returns > 0]
        losses = returns[returns < 0]

        win_rate = len(wins) / len(returns)
        win_loss_ratio = wins.mean() / abs(losses.mean()) if len(losses) > 0 else 1.0

        # Full Kelly
        kelly = self.calculate_kelly_fraction(win_rate, win_loss_ratio)

        # Deflate Kelly based on confidence
        if len(returns) > 100:
            # Use historical distribution to estimate confidence
            kelly_std = np.std([self.calculate_kelly_fraction(
                np.random.choice([1, 0], size=len(returns), p=[win_rate, 1-win_rate]),
                win_loss_ratio
            ) for _ in range(1000)])

            # Deflate by confidence interval
            z_score = norm.ppf(confidence_level)
            deflated_kelly = kelly - z_score * kelly_std
            return max(0.0, min(deflated_kelly, kelly * 0.5))  # Conservative cap
        else:
            # Simple deflation for small samples
            return kelly * 0.5

    def calculate_volatility_target(self, portfolio_state: PortfolioState) -> float:
        """
        Calculate volatility-adjusted position size

        Args:
            portfolio_state: Current portfolio state

        Returns:
            Position size adjustment factor
        """
        target_vol = self.risk_limits.max_volatility
        current_vol = portfolio_state.current_volatility

        if current_vol <= 0:
            return 1.0

        # Volatility scaling factor
        vol_adjustment = target_vol / current_vol

        # Apply limits
        vol_adjustment = min(vol_adjustment, 2.0)  # Max 2x leverage
        vol_adjustment = max(vol_adjustment, 0.1)  # Min 10% of target

        return vol_adjustment

    def check_risk_limits(self, portfolio_state: PortfolioState) -> Dict[str, bool]:
        """
        Check if portfolio violates risk limits

        Args:
            portfolio_state: Current portfolio state

        Returns:
            Dictionary of risk limit violations
        """
        violations = {}

        # Drawdown check
        violations['max_drawdown'] = portfolio_state.current_drawdown > self.risk_limits.max_drawdown

        # Volatility check
        violations['max_volatility'] = portfolio_state.current_volatility > self.risk_limits.max_volatility

        # Leverage check
        total_exposure = sum(abs(size) for size in portfolio_state.positions.values())
        leverage = total_exposure / portfolio_state.capital
        violations['max_leverage'] = leverage > self.risk_limits.max_leverage

        # Concentration check
        for symbol, position in portfolio_state.positions.items():
            concentration = abs(position) / portfolio_state.current_value
            if concentration > self.risk_limits.max_concentration:
                violations[f'concentration_{symbol}'] = True

        return violations

    def calculate_position_size(self,
                              capital: float,
                              volatility: float,
                              risk_per_trade: float = 0.01) -> float:
        """
        Calculate position size based on risk per trade

        Args:
            capital: Available capital
            volatility: Asset volatility
            risk_per_trade: Risk per trade as fraction of capital

        Returns:
            Position size in dollars
        """
        if volatility <= 0:
            return capital * 0.01  # Conservative fallback

        # Kelly-style position sizing based on volatility
        position_risk = capital * risk_per_trade
        position_size = position_risk / volatility

        # Apply risk limits
        max_position = capital * self.risk_limits.max_concentration
        position_size = min(position_size, max_position)

        return position_size

class PortfolioManager:
    """Portfolio construction and optimization"""

    def __init__(self, risk_manager: RiskManager = None):
        self.risk_manager = risk_manager or RiskManager()

    def optimize_portfolio(self,
                          returns: pd.DataFrame,
                          target_return: Optional[float] = None,
                          risk_free_rate: float = 0.02) -> Dict[str, float]:
        """
        Optimize portfolio weights using mean-variance optimization

        Args:
            returns: Historical returns DataFrame (assets as columns)
            target_return: Target portfolio return (optional)
            risk_free_rate: Risk-free rate

        Returns:
            Dictionary of optimal weights
        """
        if len(returns.columns) < 2:
            # Single asset - allocate 100%
            return {returns.columns[0]: 1.0}

        # Calculate expected returns and covariance
        expected_returns = returns.mean() * 252  # Annualized
        cov_matrix = returns.cov() * 252  # Annualized

        n_assets = len(returns.columns)

        def portfolio_volatility(weights):
            return np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))

        def portfolio_return(weights):
            return np.dot(weights, expected_returns)

        def objective_function(weights):
            port_return = portfolio_return(weights)
            port_vol = portfolio_volatility(weights)

            if target_return is not None:
                # Minimize volatility for target return
                return port_vol if port_return >= target_return else 1000
            else:
                # Maximize Sharpe ratio
                sharpe = (port_return - risk_free_rate) / port_vol
                return -sharpe

        # Constraints
        constraints = [
            {'type': 'eq', 'fun': lambda x: np.sum(x) - 1}  # Weights sum to 1
        ]

        bounds = [(0, 0.3) for _ in range(n_assets)]  # Max 30% per asset

        # Initial guess
        initial_weights = np.ones(n_assets) / n_assets

        try:
            result = minimize_scalar(
                lambda x: objective_function(x),
                bounds=bounds,
                constraints=constraints,
                method='SLSQP'
            )

            if result.success:
                weights = result.x
                # Normalize to ensure sum = 1
                weights = weights / np.sum(weights)

                return dict(zip(returns.columns, weights))
            else:
                print(f"Optimization failed: {result.message}")
                # Return equal weight portfolio
                equal_weight = 1.0 / n_assets
                return {col: equal_weight for col in returns.columns}

        except Exception as e:
            print(f"Portfolio optimization error: {e}")
            # Return equal weight portfolio
            equal_weight = 1.0 / n_assets
            return {col: equal_weight for col in returns.columns}

    def apply_regime_adjustments(self,
                                base_weights: Dict[str, float],
                                risk_regime: int,
                                trend_signal: bool) -> Dict[str, float]:
        """
        Apply regime-aware adjustments to portfolio weights

        Args:
            base_weights: Base portfolio weights
            risk_regime: Current risk regime (0=low, 1=normal, 2=high, 3=extreme)
            trend_signal: Current trend direction

        Returns:
            Adjusted portfolio weights
        """
        adjusted_weights = base_weights.copy()

        # Risk regime adjustments
        if risk_regime >= 2:  # High or extreme risk
            # Reduce exposure in high risk regimes
            risk_multiplier = max(0.3, 1.0 - (risk_regime - 1) * 0.3)
            adjusted_weights = {k: v * risk_multiplier for k, v in adjusted_weights.items()}

            # Increase cash allocation
            cash_allocation = 1.0 - sum(adjusted_weights.values())
            adjusted_weights['_cash'] = cash_allocation

        elif risk_regime == 0:  # Low risk
            # Can be more aggressive in low risk regimes
            risk_multiplier = min(1.3, 1.0 + 0.3)
            adjusted_weights = {k: v * risk_multiplier for k, v in adjusted_weights.items()}

        # Trend adjustments
        if not trend_signal:  # Downtrend
            # Reduce equity exposure in downtrends
            equity_multiplier = 0.7
            for asset in adjusted_weights:
                if asset != '_cash' and not asset.startswith(('bond', 'gold')):
                    adjusted_weights[asset] *= equity_multiplier

        # Re-normalize weights
        total_weight = sum(v for k, v in adjusted_weights.items() if k != '_cash')
        if total_weight > 0:
            adjusted_weights = {
                k: v / total_weight * (1 - adjusted_weights.get('_cash', 0))
                for k, v in adjusted_weights.items()
                if k != '_cash'
            }

        return adjusted_weights

    def calculate_rebalance_trades(self,
                                  current_weights: Dict[str, float],
                                  target_weights: Dict[str, float],
                                  portfolio_value: float) -> Dict[str, float]:
        """
        Calculate trades needed to rebalance to target weights

        Args:
            current_weights: Current portfolio weights
            target_weights: Target portfolio weights
            portfolio_value: Current portfolio value

        Returns:
            Dictionary of required trades (positive = buy, negative = sell)
        """
        trades = {}

        # Get all unique assets
        all_assets = set(current_weights.keys()) | set(target_weights.keys())

        for asset in all_assets:
            current_weight = current_weights.get(asset, 0.0)
            target_weight = target_weights.get(asset, 0.0)

            weight_diff = target_weight - current_weight
            trade_value = weight_diff * portfolio_value

            if abs(trade_value) > portfolio_value * 0.001:  # Minimum trade size
                trades[asset] = trade_value

        return trades

class DrawdownController:
    """Drawdown control and portfolio rebalancing"""

    def __init__(self, risk_limits: RiskLimits = None):
        self.risk_limits = risk_limits or RiskLimits()
        self.portfolio_history = []
        self.peak_value = 0
        self.current_drawdown = 0

    def update_drawdown(self, current_value: float) -> float:
        """
        Update drawdown calculation

        Args:
            current_value: Current portfolio value

        Returns:
            Current drawdown as fraction
        """
        self.peak_value = max(self.peak_value, current_value)
        self.current_drawdown = (self.peak_value - current_value) / self.peak_value
        self.portfolio_history.append(current_value)

        return self.current_drawdown

    def should_reduce_risk(self) -> Tuple[bool, float]:
        """
        Determine if risk reduction is needed

        Returns:
            Tuple of (should_reduce, reduction_factor)
        """
        if self.current_drawdown > self.risk_limits.max_drawdown:
            # Severe drawdown - significant risk reduction
            reduction_factor = max(0.3, 1.0 - self.current_drawdown)
            return True, reduction_factor

        elif self.current_drawdown > self.risk_limits.max_drawdown * 0.7:
            # Moderate drawdown - moderate risk reduction
            reduction_factor = 0.8
            return True, reduction_factor

        elif self.current_drawdown > self.risk_limits.max_drawdown * 0.5:
            # Mild drawdown - slight risk reduction
            reduction_factor = 0.9
            return True, reduction_factor

        return False, 1.0

    def calculate_var_reduction(self, returns: pd.Series, var_limit: float = 0.05) -> float:
        """
        Calculate position size reduction based on VaR

        Args:
            returns: Historical returns series
            var_limit: VaR limit (95% confidence)

        Returns:
            Position size reduction factor
        """
        if len(returns) < 30:
            return 1.0

        # Calculate 95% VaR
        var_95 = np.percentile(returns, 5)  # 5th percentile = 95% VaR

        if abs(var_95) > var_limit:
            # Reduce position size to bring VaR within limits
            reduction_factor = var_limit / abs(var_95)
            return min(reduction_factor, 1.0)

        return 1.0

class StressTester:
    """Portfolio stress testing"""

    def __init__(self):
        self.scenarios = {
            'covid_crash': {'equity': -0.33, 'duration': 30},
            'tech_bubble': {'equity': -0.50, 'tech': -0.70, 'duration': 60},
            'interest_rate_shock': {'equity': -0.15, 'bonds': -0.10, 'duration': 90},
            'volatility_explosion': {'vix': 2.0, 'equity': -0.20, 'duration': 20},
        }

    def run_stress_test(self,
                       portfolio_weights: Dict[str, float],
                       historical_returns: pd.DataFrame,
                       scenario: str = 'covid_crash') -> Dict[str, Any]:
        """
        Run stress test on portfolio

        Args:
            portfolio_weights: Current portfolio weights
            historical_returns: Historical returns DataFrame
            scenario: Stress scenario to test

        Returns:
            Stress test results
        """
        if scenario not in self.scenarios:
            raise ValueError(f"Unknown scenario: {scenario}")

        scenario_params = self.scenarios[scenario]
        duration = scenario_params['duration']

        # Simulate stress scenario
        stress_returns = pd.DataFrame(index=range(duration), columns=historical_returns.columns)

        for asset in historical_returns.columns:
            if asset in scenario_params:
                # Apply scenario shock
                shock = scenario_params[asset]
                base_vol = historical_returns[asset].std()

                # Generate stressed returns
                stressed_returns = np.random.normal(
                    shock / duration,  # Mean return over period
                    base_vol * 2,      # Increased volatility
                    duration
                )
                stress_returns[asset] = stressed_returns
            else:
                # Use historical distribution
                stress_returns[asset] = np.random.choice(
                    historical_returns[asset].values,
                    size=duration,
                    replace=True
                )

        # Calculate portfolio returns
        portfolio_returns = stress_returns.dot(pd.Series(portfolio_weights))

        # Calculate stress metrics
        max_drawdown = (portfolio_returns.cumsum() - portfolio_returns.cumsum().expanding().max()).min()
        total_return = portfolio_returns.sum()
        volatility = portfolio_returns.std() * np.sqrt(252)
        var_95 = np.percentile(portfolio_returns, 5)

        return {
            'scenario': scenario,
            'duration': duration,
            'total_return': float(total_return),
            'max_drawdown': float(max_drawdown),
            'volatility': float(volatility),
            'var_95': float(var_95),
            'breach_threshold': abs(total_return) > 0.15  # 15% loss threshold
        }

# Convenience functions for easy use
def create_risk_manager(max_drawdown: float = 0.20,
                       max_volatility: float = 0.25) -> RiskManager:
    """Create risk manager with custom limits"""
    limits = RiskLimits(max_drawdown=max_drawdown, max_volatility=max_volatility)
    return RiskManager(limits)

def create_portfolio_manager(risk_manager: RiskManager = None) -> PortfolioManager:
    """Create portfolio manager"""
    return PortfolioManager(risk_manager)

def create_drawdown_controller(max_drawdown: float = 0.20) -> DrawdownController:
    """Create drawdown controller"""
    limits = RiskLimits(max_drawdown=max_drawdown)
    return DrawdownController(limits)

if __name__ == "__main__":
    # Example usage
    import yfinance as yf

    # Create risk management components
    risk_mgr = create_risk_manager()
    port_mgr = create_portfolio_manager(risk_mgr)
    dd_ctrl = create_drawdown_controller()

    # Example portfolio state
    portfolio_state = PortfolioState(
        capital=1000000,
        positions={'SPY': 50000, 'QQQ': 30000},
        current_value=950000,
        peak_value=1000000,
        current_drawdown=0.05,
        volatility_target=0.15,
        current_volatility=0.18
    )

    # Check risk limits
    violations = risk_mgr.check_risk_limits(portfolio_state)
    print("Risk violations:", violations)

    # Calculate position size
    position_size = risk_mgr.calculate_position_size(100000, 0.25)
    print(f"Position size for 1% risk: ${position_size:,.0f}")

    # Test Kelly criterion
    kelly = risk_mgr.calculate_kelly_fraction(0.55, 1.5)
    print(f"Kelly fraction (55% win rate, 1.5:1 ratio): {kelly:.3f}")

    print("Risk management system initialized successfully!")

