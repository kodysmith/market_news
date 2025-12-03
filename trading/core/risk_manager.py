"""
Risk Manager - Core Module

Risk validation and position limits.
Same logic used in backtest, paper, and live modes.
"""

from typing import Dict, List, Any, Optional
import logging

logger = logging.getLogger(__name__)


class RiskManager:
    """
    Risk management and validation.
    Enforces position limits and risk rules.
    """

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.max_position_size = config.get('max_position_size', 0.5)  # 50% max per position
        self.max_portfolio_var = config.get('max_portfolio_var', 0.02)  # 2% daily VaR
        self.max_leverage = config.get('max_leverage', 1.0)  # No leverage by default

    def validate_weights(self, weights: Dict[str, float]) -> Dict[str, Any]:
        """
        Validate portfolio weights against risk limits

        Args:
            weights: Proposed portfolio weights

        Returns:
            Validation result with any violations
        """
        violations = []
        warnings = []

        # Check individual position limits
        for ticker, weight in weights.items():
            if abs(weight) > self.max_position_size:
                violations.append(f"{ticker}: weight {weight:.1%} exceeds max {self.max_position_size:.1%}")

        # Check total leverage
        total_long = sum(w for w in weights.values() if w > 0)
        total_short = sum(abs(w) for w in weights.values() if w < 0)
        total_exposure = total_long + total_short

        if total_exposure > self.max_leverage:
            violations.append(f"Total exposure {total_exposure:.1%} exceeds max leverage {self.max_leverage:.1%}")

        # Check weight sum
        weight_sum = sum(weights.values())
        if abs(weight_sum - 1.0) > 0.01:  # Allow 1% tolerance
            warnings.append(f"Weight sum {weight_sum:.1%} is not 1.0")

        is_valid = len(violations) == 0

        return {
            'is_valid': is_valid,
            'violations': violations,
            'warnings': warnings
        }

    def validate_trades(self, trades: Dict[str, float],
                       current_positions: Dict[str, float],
                       portfolio_value: float) -> Dict[str, Any]:
        """
        Validate proposed trades against risk limits

        Args:
            trades: Proposed trades (ticker -> trade amount)
            current_positions: Current position values
            portfolio_value: Total portfolio value

        Returns:
            Validation result
        """
        violations = []

        for ticker, trade_amount in trades.items():
            if trade_amount == 0:
                continue

            # Calculate new position value
            current_value = current_positions.get(ticker, 0)
            new_value = current_value + trade_amount
            new_weight = new_value / portfolio_value if portfolio_value > 0 else 0

            # Check position size limit
            if abs(new_weight) > self.max_position_size:
                violations.append(
                    f"{ticker}: trade would create {new_weight:.1%} position, "
                    f"exceeds max {self.max_position_size:.1%}"
                )

        is_valid = len(violations) == 0

        return {
            'is_valid': is_valid,
            'violations': violations
        }

    def check_portfolio_risk(self, positions: Dict[str, float],
                           portfolio_value: float,
                           market_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Check overall portfolio risk

        Args:
            positions: Current positions
            portfolio_value: Total portfolio value
            market_data: Current market data

        Returns:
            Risk assessment
        """
        # Calculate position concentrations
        concentrations = {}
        for ticker, value in positions.items():
            if portfolio_value > 0:
                concentrations[ticker] = value / portfolio_value

        # Find max concentration
        max_concentration = max(concentrations.values()) if concentrations else 0

        # Simple risk assessment
        risk_level = 'low'
        if max_concentration > 0.4:
            risk_level = 'high'
        elif max_concentration > 0.3:
            risk_level = 'medium'

        return {
            'risk_level': risk_level,
            'max_concentration': max_concentration,
            'concentrations': concentrations
        }
