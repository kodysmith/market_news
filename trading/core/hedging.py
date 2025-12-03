"""
Hedging Module - Core Module

Overnight hedging logic using options.
Same logic used in backtest, paper, and live modes.
"""

import numpy as np
from typing import Dict, List, Any, Optional
import logging
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)


class HedgingModule:
    """
    Overnight hedging using options.
    Calculates hedge positions based on portfolio risk.
    """

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.enable_hedging = config.get('enable_hedging', False)
        self.hedge_delta = config.get('hedge_delta', -0.5)
        self.hedge_coverage = config.get('hedge_coverage', 1.0)

    def should_hedge(self, regime: Dict[str, Any], 
                   portfolio_value: float,
                   current_hedge_value: float = 0) -> bool:
        """
        Determine if hedging is needed

        Args:
            regime: Current market regime
            portfolio_value: Total portfolio value
            current_hedge_value: Current hedge position value

        Returns:
            True if hedging should be applied
        """
        if not self.enable_hedging:
            return False

        regime_type = regime.get('regime', 'ranging')
        confidence = regime.get('confidence', 0.5)

        # Hedge in high volatility or bear markets
        if regime_type in ['high_volatility', 'bear_market', 'risk_off']:
            return True

        # Hedge if confidence is low (uncertain regime)
        if confidence < 0.5:
            return True

        return False

    def calculate_hedge_size(self, portfolio_value: float,
                           portfolio_delta: float = 1.0) -> Dict[str, Any]:
        """
        Calculate hedge position size

        Args:
            portfolio_value: Total portfolio value
            portfolio_delta: Portfolio delta (1.0 = fully long)

        Returns:
            Hedge position details
        """
        # Target hedge delta
        target_hedge_delta = portfolio_value * portfolio_delta * self.hedge_delta * self.hedge_coverage

        # For SPY puts, each contract has ~100 shares * put delta
        # Assuming ATM put has delta of ~-0.5
        put_delta_per_contract = -0.5
        contracts_needed = abs(target_hedge_delta / (100 * put_delta_per_contract))

        return {
            'contracts': int(contracts_needed),
            'target_delta': target_hedge_delta,
            'underlying': 'SPY',
            'option_type': 'put',
            'strike_type': 'atm'  # At-the-money
        }

    def calculate_hedge_cost(self, hedge_size: Dict[str, Any],
                           current_price: float,
                           implied_vol: float = 0.20) -> float:
        """
        Estimate hedge cost using Black-Scholes approximation

        Args:
            hedge_size: Hedge position details from calculate_hedge_size
            current_price: Current underlying price
            implied_vol: Implied volatility (annualized)

        Returns:
            Estimated cost per contract
        """
        contracts = hedge_size.get('contracts', 0)
        if contracts == 0:
            return 0.0

        # Simplified Black-Scholes for ATM put
        # Premium ≈ S * σ * sqrt(T) * 0.4 (rough approximation for 1-DTE ATM put)
        days_to_expiry = 1
        time_to_expiry = days_to_expiry / 365.0
        premium_per_contract = current_price * implied_vol * np.sqrt(time_to_expiry) * 0.4

        total_cost = premium_per_contract * contracts * 100  # 100 shares per contract

        return total_cost
