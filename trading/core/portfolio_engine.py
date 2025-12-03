"""
Portfolio Engine - Core Module

Calculates portfolio weights based on regime and strategy rules.
Same logic used in backtest, paper, and live modes.
"""

import numpy as np
from typing import Dict, List, Any, Optional
import logging

logger = logging.getLogger(__name__)


class PortfolioEngine:
    """
    Portfolio weight calculation engine.
    Implements regime-aware portfolio allocation.
    """

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.target_assets = config.get('target_assets', ['SPY', 'QQQ', 'TQQQ'])
        self.base_weights = config.get('base_weights', {'SPY': 0.33, 'QQQ': 0.33, 'TQQQ': 0.34})

    def calculate_weights(self, regime: Dict[str, Any], 
                         current_positions: Optional[Dict[str, float]] = None) -> Dict[str, float]:
        """
        Calculate target portfolio weights based on regime

        Args:
            regime: Current market regime from RegimeDetector
            current_positions: Current position values (optional)

        Returns:
            Dictionary of ticker -> target weight
        """
        regime_type = regime.get('regime', 'ranging')
        confidence = regime.get('confidence', 0.5)

        # Regime-based weight adjustments
        if regime_type == 'bull_market':
            # Increase TQQQ in bull markets
            weights = {'SPY': 0.25, 'QQQ': 0.30, 'TQQQ': 0.45}
        elif regime_type == 'bear_market':
            # Reduce TQQQ, increase SPY in bear markets
            weights = {'SPY': 0.50, 'QQQ': 0.35, 'TQQQ': 0.15}
        elif regime_type == 'high_volatility':
            # Defensive allocation
            weights = {'SPY': 0.60, 'QQQ': 0.30, 'TQQQ': 0.10}
        elif regime_type == 'risk_off':
            # Conservative allocation
            weights = {'SPY': 0.55, 'QQQ': 0.30, 'TQQQ': 0.15}
        else:
            # Default balanced allocation
            weights = self.base_weights.copy()

        # Blend with base weights based on confidence
        if confidence < 0.7:
            # Low confidence: use more conservative base weights
            blend_factor = 0.5
            for ticker in weights:
                weights[ticker] = blend_factor * weights[ticker] + (1 - blend_factor) * self.base_weights.get(ticker, 0)

        # Normalize to sum to 1.0
        total = sum(weights.values())
        if total > 0:
            weights = {k: v / total for k, v in weights.items()}
        else:
            weights = self.base_weights.copy()

        logger.info(f"Calculated weights: {weights}")
        return weights

    def calculate_rebalance_trades(self, current_positions: Dict[str, float],
                                   target_weights: Dict[str, float],
                                   portfolio_value: float) -> Dict[str, float]:
        """
        Calculate trades needed to rebalance to target weights

        Args:
            current_positions: Current position values by ticker
            target_weights: Target weights by ticker
            portfolio_value: Total portfolio value

        Returns:
            Dictionary of ticker -> trade amount (positive = buy, negative = sell)
        """
        trades = {}

        for ticker in target_weights:
            current_value = current_positions.get(ticker, 0)
            target_value = portfolio_value * target_weights[ticker]
            trade_amount = target_value - current_value

            # Only trade if difference is significant (>1% of portfolio)
            if abs(trade_amount) > portfolio_value * 0.01:
                trades[ticker] = trade_amount
            else:
                trades[ticker] = 0.0

        return trades
