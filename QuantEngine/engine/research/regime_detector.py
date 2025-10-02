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

        # Enhanced market indicators
        self.indicators = [
            # Core indicators
            'trend_strength', 'volatility', 'momentum', 'volume_trend',
            # Multi-timeframe indicators
            'short_trend', 'medium_trend', 'long_trend',
            'short_volatility', 'medium_volatility',
            'short_momentum', 'medium_momentum', 'long_momentum',
            'short_volume_trend', 'medium_volume_trend',
            # Market structure indicators
            'correlation_spread', 'put_call_ratio',
            'vix_level', 'vix_momentum', 'vix_combined',
            # Macro indicators
            'yield_10y', 'yield_curve', 'yield_momentum',
            'dollar_strength', 'dollar_momentum',
            'gold_level', 'oil_level',
            # Sector indicators
            'tech_performance', 'financial_performance', 'energy_performance',
            # Sentiment indicators
            'economic_score', 'sentiment_score'
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
        Calculate comprehensive market regime indicators with multi-timeframe analysis

        Returns normalized indicators between -1 and 1
        """

        indicators = {}

        # Multi-timeframe trend analysis
        if 'spy' in market_data and 'prices' in market_data['spy']:
            prices = market_data['spy']['prices']
            
            # 1. Short-term trend (5-day)
            if len(prices) >= 5:
                short_trend = (prices[-1] - prices[-5]) / prices[-5]
                indicators['short_trend'] = self.normalize_indicator(short_trend, -0.05, 0.05)
            
            # 2. Medium-term trend (20-day)
            if len(prices) >= 20:
                medium_trend = (prices[-1] - prices[-20]) / prices[-20]
                indicators['medium_trend'] = self.normalize_indicator(medium_trend, -0.1, 0.1)
            
            # 3. Long-term trend (50-day)
            if len(prices) >= 50:
                long_trend = (prices[-1] - prices[-50]) / prices[-50]
                indicators['long_trend'] = self.normalize_indicator(long_trend, -0.2, 0.2)
            
            # 4. Combined trend strength (weighted average)
            trend_weights = {'short_trend': 0.4, 'medium_trend': 0.4, 'long_trend': 0.2}
            trend_strength = 0
            total_weight = 0
            for trend_name, weight in trend_weights.items():
                if trend_name in indicators:
                    trend_strength += indicators[trend_name] * weight
                    total_weight += weight
            indicators['trend_strength'] = trend_strength / max(total_weight, 1)

            # 5. Multi-timeframe volatility analysis
            if len(prices) >= 20:
                # Short-term volatility (5-day)
                if len(prices) >= 5:
                    short_returns = np.diff(np.log(prices[-5:]))
                    short_vol = np.std(short_returns)
                    indicators['short_volatility'] = self.normalize_indicator(short_vol, 0.005, 0.03)
                
                # Medium-term volatility (20-day)
                medium_returns = np.diff(np.log(prices[-20:]))
                medium_vol = np.std(medium_returns)
                indicators['medium_volatility'] = self.normalize_indicator(medium_vol, 0.01, 0.06)
                
                # Combined volatility score
                vol_weights = {'short_volatility': 0.3, 'medium_volatility': 0.7}
                volatility_score = 0
                vol_total_weight = 0
                for vol_name, weight in vol_weights.items():
                    if vol_name in indicators:
                        volatility_score += indicators[vol_name] * weight
                        vol_total_weight += weight
                indicators['volatility'] = volatility_score / max(vol_total_weight, 1)

            # 6. Enhanced momentum (multiple timeframes)
            if len(prices) >= 50:
                # Short momentum (5 vs 12-day MA)
                if len(prices) >= 12:
                    short_ma = np.mean(prices[-5:])
                    medium_ma = np.mean(prices[-12:])
                    short_momentum = (short_ma - medium_ma) / medium_ma
                    indicators['short_momentum'] = self.normalize_indicator(short_momentum, -0.02, 0.02)
                
                # Medium momentum (12 vs 26-day MA)
                if len(prices) >= 26:
                    medium_ma = np.mean(prices[-12:])
                    long_ma = np.mean(prices[-26:])
                    medium_momentum = (medium_ma - long_ma) / long_ma
                    indicators['medium_momentum'] = self.normalize_indicator(medium_momentum, -0.03, 0.03)
                
                # Long momentum (26 vs 50-day MA)
                if len(prices) >= 50:
                    long_ma = np.mean(prices[-26:])
                    very_long_ma = np.mean(prices[-50:])
                    long_momentum = (long_ma - very_long_ma) / very_long_ma
                    indicators['long_momentum'] = self.normalize_indicator(long_momentum, -0.05, 0.05)
                
                # Combined momentum score
                momentum_weights = {'short_momentum': 0.5, 'medium_momentum': 0.3, 'long_momentum': 0.2}
                momentum_score = 0
                momentum_total_weight = 0
                for mom_name, weight in momentum_weights.items():
                    if mom_name in indicators:
                        momentum_score += indicators[mom_name] * weight
                        momentum_total_weight += weight
                indicators['momentum'] = momentum_score / max(momentum_total_weight, 1)

            # 7. Enhanced volume analysis (multiple timeframes)
            if 'volumes' in market_data['spy']:
                volumes = market_data['spy']['volumes']
                if len(volumes) >= 20:
                    # Short-term volume trend (5-day)
                    if len(volumes) >= 5:
                        short_vol_avg = np.mean(volumes[-5:])
                        current_volume = volumes[-1]
                        short_vol_trend = (current_volume - short_vol_avg) / short_vol_avg
                        indicators['short_volume_trend'] = self.normalize_indicator(short_vol_trend, -0.3, 0.3)
                    
                    # Medium-term volume trend (20-day)
                    medium_vol_avg = np.mean(volumes[-20:])
                    current_volume = volumes[-1]
                    medium_vol_trend = (current_volume - medium_vol_avg) / medium_vol_avg
                    indicators['medium_volume_trend'] = self.normalize_indicator(medium_vol_trend, -0.5, 0.5)
                    
                    # Combined volume trend
                    vol_trend_weights = {'short_volume_trend': 0.6, 'medium_volume_trend': 0.4}
                    volume_trend_score = 0
                    vol_trend_total_weight = 0
                    for vol_trend_name, weight in vol_trend_weights.items():
                        if vol_trend_name in indicators:
                            volume_trend_score += indicators[vol_trend_name] * weight
                            vol_trend_total_weight += weight
                    indicators['volume_trend'] = volume_trend_score / max(vol_trend_total_weight, 1)

        # 8. Enhanced correlation analysis (multiple assets)
        correlation_spread = await self.calculate_correlation_spread(market_data)
        indicators['correlation_spread'] = correlation_spread

        # 9. Put/Call Ratio (if available)
        if 'options' in market_data and 'put_call_ratio' in market_data['options']:
            pcr = market_data['options']['put_call_ratio']
            indicators['put_call_ratio'] = self.normalize_indicator(pcr, 0.5, 1.5)

        # 10. Enhanced VIX analysis (multiple timeframes)
        if 'vix' in market_data:
            vix_level = market_data['vix'].get('level', 20)
            vix_change = market_data['vix'].get('change', 0)
            
            # VIX level (fear gauge)
            indicators['vix_level'] = self.normalize_indicator(vix_level, 10, 40)
            
            # VIX momentum (change in VIX)
            indicators['vix_momentum'] = self.normalize_indicator(vix_change, -5, 5)
            
            # Combined VIX score (level + momentum)
            vix_score = (indicators['vix_level'] * 0.7 + indicators['vix_momentum'] * 0.3)
            indicators['vix_combined'] = vix_score

        # 11. Treasury yield analysis (if available)
        if 'treasury' in market_data:
            treasury_data = market_data['treasury']
            
            # 10Y yield level
            if 'yield_10y' in treasury_data:
                yield_10y = treasury_data['yield_10y']
                indicators['yield_10y'] = self.normalize_indicator(yield_10y, 1.0, 6.0)
            
            # Yield curve (10Y - 2Y spread)
            if 'yield_10y' in treasury_data and 'yield_2y' in treasury_data:
                yield_spread = treasury_data['yield_10y'] - treasury_data['yield_2y']
                indicators['yield_curve'] = self.normalize_indicator(yield_spread, -1.0, 3.0)
            
            # Yield momentum (change in yields)
            if 'yield_change' in treasury_data:
                yield_change = treasury_data['yield_change']
                indicators['yield_momentum'] = self.normalize_indicator(yield_change, -0.5, 0.5)

        # 12. Dollar strength analysis (if available)
        if 'dollar' in market_data:
            dollar_data = market_data['dollar']
            
            # Dollar index level
            if 'dxy_level' in dollar_data:
                dxy = dollar_data['dxy_level']
                indicators['dollar_strength'] = self.normalize_indicator(dxy, 90, 110)
            
            # Dollar momentum
            if 'dxy_change' in dollar_data:
                dxy_change = dollar_data['dxy_change']
                indicators['dollar_momentum'] = self.normalize_indicator(dxy_change, -2, 2)

        # 13. Commodity analysis (if available)
        if 'commodities' in market_data:
            comm_data = market_data['commodities']
            
            # Gold level (safe haven)
            if 'gold' in comm_data:
                gold = comm_data['gold']
                indicators['gold_level'] = self.normalize_indicator(gold, 1500, 2500)
            
            # Oil level (inflation/economic indicator)
            if 'oil' in comm_data:
                oil = comm_data['oil']
                indicators['oil_level'] = self.normalize_indicator(oil, 50, 150)

        # 14. Sector rotation analysis (if available)
        if 'sectors' in market_data:
            sector_data = market_data['sectors']
            
            # Technology sector performance
            if 'technology' in sector_data:
                tech_perf = sector_data['technology']
                indicators['tech_performance'] = self.normalize_indicator(tech_perf, -0.1, 0.1)
            
            # Financial sector performance
            if 'financials' in sector_data:
                fin_perf = sector_data['financials']
                indicators['financial_performance'] = self.normalize_indicator(fin_perf, -0.1, 0.1)
            
            # Energy sector performance
            if 'energy' in sector_data:
                energy_perf = sector_data['energy']
                indicators['energy_performance'] = self.normalize_indicator(energy_perf, -0.1, 0.1)

        # 15. Economic indicators (enhanced)
        if 'economic' in market_data:
            economic_score = self.calculate_economic_score(market_data['economic'])
            indicators['economic_score'] = economic_score

        # 16. Sentiment indicators (enhanced)
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
        """Enhanced regime classification with weighted scoring"""

        # Enhanced regime scoring system with weights
        regime_scores = {
            'bull_market': 0.0,
            'bear_market': 0.0,
            'high_volatility': 0.0,
            'low_volatility': 0.0,
            'risk_on': 0.0,
            'risk_off': 0.0,
            'trending': 0.0,
            'ranging': 0.0,
            'growth_market': 0.0,
            'value_market': 0.0,
            'inflation_market': 0.0,
            'deflation_market': 0.0
        }

        # Enhanced trend-based classification with multi-timeframe analysis
        short_trend = indicators.get('short_trend', 0)
        medium_trend = indicators.get('medium_trend', 0)
        long_trend = indicators.get('long_trend', 0)
        trend_strength = indicators.get('trend_strength', 0)
        
        # Weighted trend analysis
        trend_score = (short_trend * 0.4 + medium_trend * 0.4 + long_trend * 0.2)
        
        if trend_score > 0.3:
            regime_scores['bull_market'] += 3
            regime_scores['trending'] += 2
        elif trend_score > 0.1:
            regime_scores['bull_market'] += 2
            regime_scores['trending'] += 1
        elif trend_score > 0:
            regime_scores['bull_market'] += 1
        elif trend_score < -0.3:
            regime_scores['bear_market'] += 3
            regime_scores['trending'] += 2
        elif trend_score < -0.1:
            regime_scores['bear_market'] += 2
            regime_scores['trending'] += 1
        elif trend_score < 0:
            regime_scores['bear_market'] += 1
        else:
            regime_scores['ranging'] += 2

        # Enhanced volatility classification with multi-timeframe analysis
        short_vol = indicators.get('short_volatility', 0)
        medium_vol = indicators.get('medium_volatility', 0)
        volatility = indicators.get('volatility', 0)
        
        # Weighted volatility analysis
        vol_score = (short_vol * 0.3 + medium_vol * 0.7)
        
        if vol_score > 0.5:
            regime_scores['high_volatility'] += 3
        elif vol_score > 0.2:
            regime_scores['high_volatility'] += 2
        elif vol_score > 0:
            regime_scores['high_volatility'] += 1
        elif vol_score < -0.5:
            regime_scores['low_volatility'] += 3
        elif vol_score < -0.2:
            regime_scores['low_volatility'] += 2
        elif vol_score < 0:
            regime_scores['low_volatility'] += 1

        # Enhanced risk-based classification with macro indicators
        vix_level = indicators.get('vix_level', 0)
        vix_momentum = indicators.get('vix_momentum', 0)
        correlation = indicators.get('correlation_spread', 0)
        sentiment = indicators.get('sentiment_score', 0)
        yield_curve = indicators.get('yield_curve', 0)
        dollar_strength = indicators.get('dollar_strength', 0)
        
        # Weighted risk score
        risk_score = (
            vix_level * 0.25 +           # VIX level (fear gauge)
            vix_momentum * 0.15 +        # VIX momentum
            correlation * 0.20 +         # Market correlation
            sentiment * 0.20 +           # Market sentiment
            yield_curve * 0.10 +         # Yield curve (economic outlook)
            dollar_strength * 0.10       # Dollar strength (risk appetite)
        )
        
        if risk_score > 0.3:
            regime_scores['risk_off'] += 3
        elif risk_score > 0.1:
            regime_scores['risk_off'] += 2
        elif risk_score > 0:
            regime_scores['risk_off'] += 1
        elif risk_score < -0.3:
            regime_scores['risk_on'] += 3
        elif risk_score < -0.1:
            regime_scores['risk_on'] += 2
        elif risk_score < 0:
            regime_scores['risk_on'] += 1

        # Enhanced market type classification
        tech_perf = indicators.get('tech_performance', 0)
        fin_perf = indicators.get('financial_performance', 0)
        energy_perf = indicators.get('energy_performance', 0)
        gold_level = indicators.get('gold_level', 0)
        oil_level = indicators.get('oil_level', 0)
        
        # Growth vs Value market
        if tech_perf > 0.1 and fin_perf < 0.1:
            regime_scores['growth_market'] += 2
        elif fin_perf > 0.1 and tech_perf < 0.1:
            regime_scores['value_market'] += 2
        elif tech_perf > 0 and fin_perf > 0:
            regime_scores['growth_market'] += 1
            regime_scores['value_market'] += 1
        
        # Inflation vs Deflation market
        if oil_level > 0.3 and gold_level > 0.3:
            regime_scores['inflation_market'] += 2
        elif oil_level < -0.3 and gold_level < -0.3:
            regime_scores['deflation_market'] += 2
        elif oil_level > 0 or gold_level > 0:
            regime_scores['inflation_market'] += 1
        elif oil_level < 0 or gold_level < 0:
            regime_scores['deflation_market'] += 1

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


