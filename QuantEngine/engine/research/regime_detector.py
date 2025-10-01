"""
Market Regime Detection System

Continuously monitors market conditions and identifies:
- Bull vs Bear markets
- High vs Low volatility regimes
- Risk-on vs Risk-off environments
- Trend strength and momentum
- Liquidity conditions

Uses multiple indicators and statistical methods to classify market state.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
import logging
from datetime import datetime, timedelta
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import statsmodels.api as sm
from scipy import stats

logger = logging.getLogger(__name__)


class MarketRegimeDetector:
    """
    Advanced market regime detection using multiple indicators
    """

    def __init__(self, config: Dict[str, Any]):
        self.config = config

        # Regime classification thresholds
        self.regime_thresholds = {
            'trend_strength': {
                'strong_bull': 0.02,    # 2% daily trend
                'weak_bull': 0.005,    # 0.5% daily trend
                'weak_bear': -0.005,   # -0.5% daily trend
                'strong_bear': -0.02   # -2% daily trend
            },
            'volatility': {
                'low_vol': 0.015,      # 1.5% daily vol
                'normal_vol': 0.025,   # 2.5% daily vol
                'high_vol': 0.04       # 4% daily vol
            },
            'momentum': {
                'strong_positive': 1.5,
                'weak_positive': 0.5,
                'weak_negative': -0.5,
                'strong_negative': -1.5
            }
        }

        # Historical regime data
        self.regime_history = []
        self.regime_transition_matrix = {}

        # Market indicators
        self.indicators = [
            'trend_strength', 'volatility', 'momentum', 'volume_trend',
            'correlation_spread', 'put_call_ratio', 'vix_level'
        ]

    async def detect_regime(self, market_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Detect current market regime using comprehensive analysis

        Args:
            market_data: Dictionary containing price, volume, and sentiment data

        Returns:
            Regime classification with confidence scores
        """

        try:
            # Calculate regime indicators
            indicators = await self.calculate_regime_indicators(market_data)

            # Classify regime
            regime_classification = self.classify_regime(indicators)

            # Calculate confidence
            confidence = self.calculate_regime_confidence(indicators, regime_classification)

            # Update regime history
            self.update_regime_history(regime_classification, indicators)

            result = {
                'regime': regime_classification['primary_regime'],
                'confidence': confidence,
                'indicators': indicators,
                'secondary_regimes': regime_classification.get('secondary_regimes', []),
                'regime_score': regime_classification.get('regime_score', 0),
                'timestamp': datetime.now().isoformat(),
                'market_data_quality': self.assess_data_quality(market_data)
            }

            logger.info(f"🎭 Detected regime: {result['regime']} (confidence: {confidence:.1%})")
            return result

        except Exception as e:
            logger.error(f"Regime detection failed: {e}")
            return {
                'regime': 'unknown',
                'confidence': 0.0,
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }

    async def calculate_regime_indicators(self, market_data: Dict[str, Any]) -> Dict[str, float]:
        """
        Calculate comprehensive market regime indicators

        Returns normalized indicators between -1 and 1
        """

        indicators = {}

        # 1. Trend Strength (20-day return)
        if 'spy' in market_data and 'prices' in market_data['spy']:
            prices = market_data['spy']['prices']
            if len(prices) >= 20:
                trend_return = (prices[-1] - prices[-20]) / prices[-20]
                indicators['trend_strength'] = self.normalize_indicator(trend_return, -0.1, 0.1)

        # 2. Volatility (20-day realized vol)
        if 'spy' in market_data and 'prices' in market_data['spy']:
            prices = market_data['spy']['prices']
            if len(prices) >= 20:
                returns = np.diff(np.log(prices[-20:]))
                volatility = np.std(returns)
                indicators['volatility'] = self.normalize_indicator(volatility, 0.01, 0.06)

        # 3. Momentum (MACD-like indicator)
        if 'spy' in market_data and 'prices' in market_data['spy']:
            prices = market_data['spy']['prices']
            if len(prices) >= 26:
                fast_ma = np.mean(prices[-12:])  # 12-day MA
                slow_ma = np.mean(prices[-26:])  # 26-day MA
                momentum = (fast_ma - slow_ma) / slow_ma
                indicators['momentum'] = self.normalize_indicator(momentum, -0.05, 0.05)

        # 4. Volume Trend (relative to 20-day average)
        if 'spy' in market_data and 'volumes' in market_data['spy']:
            volumes = market_data['spy']['volumes']
            if len(volumes) >= 20:
                current_volume = volumes[-1]
                avg_volume = np.mean(volumes[-20:])
                volume_trend = (current_volume - avg_volume) / avg_volume
                indicators['volume_trend'] = self.normalize_indicator(volume_trend, -0.5, 0.5)

        # 5. Correlation Spread (between sectors)
        correlation_spread = await self.calculate_correlation_spread(market_data)
        indicators['correlation_spread'] = correlation_spread

        # 6. Put/Call Ratio (if available)
        if 'options' in market_data and 'put_call_ratio' in market_data['options']:
            pcr = market_data['options']['put_call_ratio']
            indicators['put_call_ratio'] = self.normalize_indicator(pcr, 0.5, 1.5)

        # 7. VIX Level (fear gauge)
        if 'vix' in market_data:
            vix = market_data['vix'].get('level', 20)
            indicators['vix_level'] = self.normalize_indicator(vix, 10, 40)

        # 8. Economic indicators
        if 'economic' in market_data:
            economic_score = self.calculate_economic_score(market_data['economic'])
            indicators['economic_score'] = economic_score

        # 9. Sentiment indicators
        if 'sentiment' in market_data:
            sentiment_score = self.calculate_sentiment_score(market_data['sentiment'])
            indicators['sentiment_score'] = sentiment_score

        # Fill missing indicators with neutral values
        for indicator in self.indicators:
            if indicator not in indicators:
                indicators[indicator] = 0.0

        return indicators

    def normalize_indicator(self, value: float, min_val: float, max_val: float) -> float:
        """Normalize indicator to [-1, 1] range"""
        if max_val == min_val:
            return 0.0

        # Clip to range first
        value = np.clip(value, min_val, max_val)

        # Normalize to [-1, 1]
        normalized = 2 * (value - min_val) / (max_val - min_val) - 1

        return float(normalized)

    async def calculate_correlation_spread(self, market_data: Dict[str, Any]) -> float:
        """Calculate spread in correlations between different assets"""
        try:
            assets = ['spy', 'qqq', 'iwm', 'vti']
            returns_data = {}

            for asset in assets:
                if asset in market_data and 'prices' in market_data[asset]:
                    prices = market_data[asset]['prices']
                    if len(prices) >= 20:
                        returns = np.diff(np.log(prices[-20:]))
                        returns_data[asset] = returns

            if len(returns_data) < 2:
                return 0.0

            # Calculate pairwise correlations
            correlations = []
            asset_list = list(returns_data.keys())

            for i in range(len(asset_list)):
                for j in range(i+1, len(asset_list)):
                    asset1, asset2 = asset_list[i], asset_list[j]
                    if len(returns_data[asset1]) == len(returns_data[asset2]):
                        corr = np.corrcoef(returns_data[asset1], returns_data[asset2])[0, 1]
                        correlations.append(corr)

            if not correlations:
                return 0.0

            # Correlation spread (std of correlations)
            correlation_spread = np.std(correlations)

            # Normalize: higher spread = more decoupled markets = risk-on (positive)
            # Lower spread = highly correlated = risk-off (negative)
            return self.normalize_indicator(correlation_spread, 0.1, 0.8)

        except Exception as e:
            logger.warning(f"Correlation spread calculation failed: {e}")
            return 0.0

    def calculate_economic_score(self, economic_data: Dict[str, Any]) -> float:
        """Calculate economic conditions score"""
        try:
            score = 0.0
            indicators = 0

            # GDP growth
            if 'gdp_growth' in economic_data:
                gdp = economic_data['gdp_growth']
                score += self.normalize_indicator(gdp, -0.02, 0.04)  # -2% to +4%
                indicators += 1

            # Unemployment
            if 'unemployment' in economic_data:
                unemp = economic_data['unemployment']
                score += -self.normalize_indicator(unemp, 3, 10)  # Invert: lower unemployment is positive
                indicators += 1

            # Inflation
            if 'inflation' in economic_data:
                infl = economic_data['inflation']
                # Optimal inflation around 2%
                inflation_score = -abs(infl - 0.02) / 0.02  # Negative deviation from 2%
                score += self.normalize_indicator(inflation_score, -1, 0)
                indicators += 1

            return score / max(indicators, 1)

        except Exception as e:
            logger.warning(f"Economic score calculation failed: {e}")
            return 0.0

    def calculate_sentiment_score(self, sentiment_data: Dict[str, Any]) -> float:
        """Calculate market sentiment score"""
        try:
            score = 0.0
            indicators = 0

            # News sentiment
            if 'news_sentiment' in sentiment_data:
                news_sent = sentiment_data['news_sentiment']  # -1 to 1
                score += news_sent
                indicators += 1

            # Social media sentiment
            if 'social_sentiment' in sentiment_data:
                social_sent = sentiment_data['social_sentiment']  # -1 to 1
                score += social_sent * 0.5  # Weight less than news
                indicators += 1

            # Put/call ratio sentiment (high PCR = bearish)
            if 'put_call_ratio' in sentiment_data:
                pcr = sentiment_data['put_call_ratio']
                pcr_sentiment = -self.normalize_indicator(pcr, 0.5, 1.5)  # Invert PCR
                score += pcr_sentiment * 0.3
                indicators += 1

            return score / max(indicators, 1)

        except Exception as e:
            logger.warning(f"Sentiment score calculation failed: {e}")
            return 0.0

    def classify_regime(self, indicators: Dict[str, float]) -> Dict[str, Any]:
        """Classify market regime based on indicators"""

        # Regime scoring system
        regime_scores = {
            'bull_market': 0.0,
            'bear_market': 0.0,
            'high_volatility': 0.0,
            'low_volatility': 0.0,
            'risk_on': 0.0,
            'risk_off': 0.0,
            'trending': 0.0,
            'ranging': 0.0
        }

        # Trend-based classification
        trend = indicators.get('trend_strength', 0)
        if trend > 0.3:
            regime_scores['bull_market'] += 2
            regime_scores['trending'] += 1
        elif trend > 0:
            regime_scores['bull_market'] += 1
        elif trend < -0.3:
            regime_scores['bear_market'] += 2
            regime_scores['trending'] += 1
        elif trend < 0:
            regime_scores['bear_market'] += 1

        # Volatility-based classification
        vol = indicators.get('volatility', 0)
        if vol > 0.5:
            regime_scores['high_volatility'] += 2
        elif vol > 0.2:
            regime_scores['high_volatility'] += 1
        elif vol < -0.5:
            regime_scores['low_volatility'] += 2
        elif vol < -0.2:
            regime_scores['low_volatility'] += 1

        # Risk-based classification
        vix = indicators.get('vix_level', 0)
        correlation = indicators.get('correlation_spread', 0)
        sentiment = indicators.get('sentiment_score', 0)

        risk_score = (vix + correlation + sentiment) / 3
        if risk_score > 0.3:
            regime_scores['risk_off'] += 2
        elif risk_score > 0:
            regime_scores['risk_off'] += 1
        elif risk_score < -0.3:
            regime_scores['risk_on'] += 2
        elif risk_score < 0:
            regime_scores['risk_on'] += 1

        # Momentum-based classification
        momentum = indicators.get('momentum', 0)
        if abs(momentum) > 0.5:
            regime_scores['trending'] += 1
        else:
            regime_scores['ranging'] += 1

        # Find primary regime
        primary_regime = max(regime_scores.items(), key=lambda x: x[1])[0]
        primary_score = regime_scores[primary_regime]

        # Find secondary regimes (scores within 1 of primary)
        secondary_regimes = [
            regime for regime, score in regime_scores.items()
            if score >= primary_score - 1 and regime != primary_regime
        ]

        return {
            'primary_regime': primary_regime,
            'regime_score': primary_score,
            'secondary_regimes': secondary_regimes,
            'all_scores': regime_scores
        }

    def calculate_regime_confidence(self, indicators: Dict[str, float],
                                   regime_classification: Dict[str, Any]) -> float:
        """Calculate confidence in regime classification"""

        primary_regime = regime_classification['primary_regime']
        regime_score = regime_classification['regime_score']

        # Base confidence from regime score
        base_confidence = min(regime_score / 3.0, 1.0)  # Max score of 3

        # Indicator consistency bonus
        consistent_indicators = 0
        total_indicators = 0

        # Check indicator alignment with regime
        indicator_regime_map = {
            'bull_market': ['trend_strength', 'momentum', 'sentiment_score'],
            'bear_market': ['trend_strength', 'momentum', 'put_call_ratio'],
            'high_volatility': ['volatility', 'vix_level'],
            'low_volatility': ['volatility', 'correlation_spread'],
            'risk_on': ['correlation_spread', 'sentiment_score'],
            'risk_off': ['vix_level', 'put_call_ratio'],
            'trending': ['trend_strength', 'momentum'],
            'ranging': ['volatility', 'volume_trend']
        }

        key_indicators = indicator_regime_map.get(primary_regime, [])
        for indicator in key_indicators:
            if indicator in indicators:
                total_indicators += 1
                # Check if indicator aligns with regime
                if self.indicator_aligns_with_regime(indicator, indicators[indicator], primary_regime):
                    consistent_indicators += 1

        indicator_consistency = consistent_indicators / max(total_indicators, 1)

        # Historical consistency (if we have history)
        historical_consistency = 1.0
        if self.regime_history:
            recent_regimes = [r['regime'] for r in self.regime_history[-5:]]  # Last 5
            consistency_count = recent_regimes.count(primary_regime)
            historical_consistency = consistency_count / len(recent_regimes)

        # Combine confidence factors
        confidence = (base_confidence * 0.5 +
                     indicator_consistency * 0.3 +
                     historical_consistency * 0.2)

        return min(confidence, 1.0)

    def indicator_aligns_with_regime(self, indicator: str, value: float, regime: str) -> bool:
        """Check if indicator value aligns with regime classification"""

        alignments = {
            'bull_market': {
                'trend_strength': lambda x: x > 0,
                'momentum': lambda x: x > 0,
                'sentiment_score': lambda x: x > 0
            },
            'bear_market': {
                'trend_strength': lambda x: x < 0,
                'momentum': lambda x: x < 0,
                'put_call_ratio': lambda x: x > 0
            },
            'high_volatility': {
                'volatility': lambda x: x > 0,
                'vix_level': lambda x: x > 0
            },
            'low_volatility': {
                'volatility': lambda x: x < 0,
                'correlation_spread': lambda x: x < 0
            },
            'risk_on': {
                'correlation_spread': lambda x: x > 0,
                'sentiment_score': lambda x: x > 0
            },
            'risk_off': {
                'vix_level': lambda x: x > 0,
                'put_call_ratio': lambda x: x > 0
            },
            'trending': {
                'trend_strength': lambda x: abs(x) > 0.2,
                'momentum': lambda x: abs(x) > 0.3
            },
            'ranging': {
                'volatility': lambda x: abs(x) < 0.3,
                'volume_trend': lambda x: abs(x) < 0.2
            }
        }

        regime_alignments = alignments.get(regime, {})
        check_func = regime_alignments.get(indicator)

        return check_func(value) if check_func else True

    def update_regime_history(self, regime_classification: Dict[str, Any],
                             indicators: Dict[str, float]):
        """Update regime history for learning"""

        history_entry = {
            'timestamp': datetime.now().isoformat(),
            'regime': regime_classification['primary_regime'],
            'confidence': regime_classification.get('regime_score', 0),
            'indicators': indicators.copy()
        }

        self.regime_history.append(history_entry)

        # Keep only recent history (last 1000 entries)
        if len(self.regime_history) > 1000:
            self.regime_history = self.regime_history[-1000:]

    def assess_data_quality(self, market_data: Dict[str, Any]) -> str:
        """Assess quality of market data"""

        quality_score = 0
        max_score = 4

        # Check for key data sources
        if 'spy' in market_data and 'prices' in market_data['spy']:
            quality_score += 1

        if 'vix' in market_data:
            quality_score += 1

        if 'options' in market_data:
            quality_score += 1

        if 'sentiment' in market_data:
            quality_score += 1

        quality_pct = quality_score / max_score

        if quality_pct >= 0.75:
            return "excellent"
        elif quality_pct >= 0.5:
            return "good"
        elif quality_pct >= 0.25:
            return "fair"
        else:
            return "poor"

    def get_regime_statistics(self) -> Dict[str, Any]:
        """Get statistics about regime detection"""

        if not self.regime_history:
            return {'error': 'No regime history available'}

        # Count regime occurrences
        regime_counts = {}
        for entry in self.regime_history:
            regime = entry['regime']
            regime_counts[regime] = regime_counts.get(regime, 0) + 1

        # Calculate regime stability
        recent_regimes = [r['regime'] for r in self.regime_history[-20:]]
        stability_score = max([recent_regimes.count(r) for r in set(recent_regimes)]) / len(recent_regimes)

        # Average confidence
        avg_confidence = np.mean([r.get('confidence', 0) for r in self.regime_history])

        return {
            'total_observations': len(self.regime_history),
            'regime_distribution': regime_counts,
            'most_common_regime': max(regime_counts.items(), key=lambda x: x[1])[0],
            'regime_stability': stability_score,
            'average_confidence': avg_confidence,
            'current_regime': self.regime_history[-1]['regime'] if self.regime_history else 'unknown'
        }

    def predict_regime_transition(self, current_regime: str) -> Dict[str, float]:
        """Predict likely next regimes based on historical transitions"""

        if len(self.regime_history) < 10:
            return {'error': 'Insufficient history for transition prediction'}

        # Build transition matrix
        transitions = {}
        for i in range(1, len(self.regime_history)):
            from_regime = self.regime_history[i-1]['regime']
            to_regime = self.regime_history[i]['regime']

            if from_regime not in transitions:
                transitions[from_regime] = {}

            transitions[from_regime][to_regime] = transitions[from_regime].get(to_regime, 0) + 1

        # Convert to probabilities
        if current_regime in transitions:
            total_transitions = sum(transitions[current_regime].values())
            probabilities = {
                regime: count / total_transitions
                for regime, count in transitions[current_regime].items()
            }

            return probabilities

        return {'error': 'No transition data for current regime'}


