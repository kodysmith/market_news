"""
Market Regime Detection - Core Module

Detects market regimes (bull/bear, volatility, risk-on/off) using multiple indicators.
Same logic used in backtest, paper, and live modes.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional
import logging
from datetime import datetime

logger = logging.getLogger(__name__)


class RegimeDetector:
    """
    Market regime detection using multiple indicators.
    Simplified version for production use.
    """

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.regime_history = []

    def detect_regime(self, market_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Detect current market regime

        Args:
            market_data: Dictionary with price data, VIX, etc.

        Returns:
            Regime classification with confidence
        """
        try:
            indicators = self._calculate_indicators(market_data)
            regime = self._classify_regime(indicators)
            confidence = self._calculate_confidence(indicators, regime)

            result = {
                'regime': regime,
                'confidence': confidence,
                'indicators': indicators,
                'timestamp': datetime.now().isoformat()
            }

            self.regime_history.append(result)
            if len(self.regime_history) > 1000:
                self.regime_history = self.regime_history[-1000:]

            logger.info(f"Detected regime: {regime} (confidence: {confidence:.1%})")
            return result

        except Exception as e:
            logger.error(f"Regime detection failed: {e}")
            return {
                'regime': 'unknown',
                'confidence': 0.0,
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }

    def _calculate_indicators(self, market_data: Dict[str, Any]) -> Dict[str, float]:
        """Calculate regime indicators from market data"""
        indicators = {}

        # Get SPY prices if available
        if 'spy' in market_data and 'prices' in market_data['spy']:
            prices = market_data['spy']['prices']
            if len(prices) >= 20:
                # Trend strength (20-day)
                trend = (prices[-1] - prices[-20]) / prices[-20]
                indicators['trend_strength'] = np.clip(trend / 0.1, -1, 1)

                # Volatility (20-day rolling)
                returns = np.diff(np.log(prices[-20:]))
                vol = np.std(returns)
                indicators['volatility'] = np.clip((vol - 0.015) / 0.015, -1, 1)

                # Momentum (5 vs 20-day MA)
                if len(prices) >= 5:
                    short_ma = np.mean(prices[-5:])
                    long_ma = np.mean(prices[-20:])
                    momentum = (short_ma - long_ma) / long_ma
                    indicators['momentum'] = np.clip(momentum / 0.02, -1, 1)

        # VIX level
        if 'vix' in market_data:
            vix_level = market_data['vix'].get('level', 20)
            indicators['vix_level'] = np.clip((vix_level - 20) / 20, -1, 1)

        # Fill missing with neutral
        for key in ['trend_strength', 'volatility', 'momentum', 'vix_level']:
            if key not in indicators:
                indicators[key] = 0.0

        return indicators

    def _classify_regime(self, indicators: Dict[str, float]) -> str:
        """Classify regime from indicators"""
        trend = indicators.get('trend_strength', 0)
        vol = indicators.get('volatility', 0)
        vix = indicators.get('vix_level', 0)

        # Primary classification
        if trend > 0.3 and vol < 0.3:
            return 'bull_market'
        elif trend < -0.3:
            return 'bear_market'
        elif vol > 0.5 or vix > 0.5:
            return 'high_volatility'
        elif trend > 0:
            return 'risk_on'
        elif trend < 0:
            return 'risk_off'
        else:
            return 'ranging'

    def _calculate_confidence(self, indicators: Dict[str, float], regime: str) -> float:
        """Calculate confidence in regime classification"""
        # Base confidence from indicator strength
        trend_strength = abs(indicators.get('trend_strength', 0))
        vol_strength = abs(indicators.get('volatility', 0))

        confidence = max(trend_strength, vol_strength)
        return min(confidence, 1.0)

    def get_regime_history(self) -> List[Dict[str, Any]]:
        """Get regime detection history"""
        return self.regime_history
