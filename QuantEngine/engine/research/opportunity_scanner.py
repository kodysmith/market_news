"""
Opportunity Discovery System

Continuously scans markets for trading opportunities by:
- Analyzing price action and technical patterns
- Detecting unusual volume and volatility
- Finding statistical arbitrage opportunities
- Identifying regime-based alpha opportunities
- Scanning for news-driven dislocations
- Monitoring options market anomalies

Returns ranked opportunities with expected value estimates.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
import logging
from datetime import datetime, timedelta
from scipy import stats
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
import asyncio

logger = logging.getLogger(__name__)


class OpportunityScanner:
    """
    Real-time opportunity discovery and ranking system
    """

    def __init__(self, config: Dict[str, Any]):
        self.config = config

        # Opportunity types to scan
        self.opportunity_types = [
            'technical_breakout', 'mean_reversion', 'momentum_divergence',
            'volume_anomaly', 'volatility_expansion', 'statistical_arbitrage',
            'options_anomaly', 'news_dislocation', 'regime_shift'
        ]

        # Minimum thresholds for opportunity detection
        self.thresholds = {
            'min_expected_return': 0.005,    # 0.5% minimum expected return
            'min_sharpe_ratio': 0.5,         # Minimum risk-adjusted return
            'max_position_size': 0.05,       # 5% max position size
            'min_liquidity': 1000000,        # Minimum daily volume
            'max_volatility': 0.08,          # Maximum acceptable volatility
            'confidence_threshold': 0.6      # Minimum confidence level
        }

        # Opportunity cache to avoid duplicates
        self.opportunity_cache = {}
        self.cache_timeout = timedelta(hours=4)

    async def scan_opportunities(self, market_state: Dict[str, Any],
                               current_regime: str,
                               strategy_portfolio: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Comprehensive opportunity scan across all markets

        Args:
            market_state: Current market data and conditions
            current_regime: Current market regime
            strategy_portfolio: Active trading strategies

        Returns:
            List of discovered opportunities ranked by expected value
        """

        logger.info("🔍 Scanning for trading opportunities...")

        opportunities = []

        # 1. Technical pattern opportunities
        technical_opps = await self.scan_technical_opportunities(market_state, current_regime)
        opportunities.extend(technical_opps)

        # 2. Volume and volatility anomalies
        volume_opps = await self.scan_volume_anomalies(market_state)
        opportunities.extend(volume_opps)

        # 3. Statistical arbitrage opportunities
        stat_arb_opps = await self.scan_statistical_arbitrage(market_state)
        opportunities.extend(stat_arb_opps)

        # 4. Options market opportunities
        options_opps = await self.scan_options_opportunities(market_state)
        opportunities.extend(options_opps)

        # 5. News and sentiment dislocations
        news_opps = await self.scan_news_dislocations(market_state)
        opportunities.extend(news_opps)

        # 6. Strategy-specific opportunities
        strategy_opps = await self.scan_strategy_opportunities(market_state, strategy_portfolio)
        opportunities.extend(strategy_opps)

        # Filter and rank opportunities
        filtered_opportunities = self.filter_opportunities(opportunities)
        ranked_opportunities = self.rank_opportunities(filtered_opportunities)

        # Cache opportunities to avoid duplicates
        self.update_opportunity_cache(ranked_opportunities)

        logger.info(f"✅ Discovered {len(ranked_opportunities)} valid opportunities")

        return ranked_opportunities

    async def scan_technical_opportunities(self, market_state: Dict[str, Any],
                                        current_regime: str) -> List[Dict[str, Any]]:
        """Scan for technical pattern-based opportunities"""

        opportunities = []

        # Get universe of assets
        universe = self.config['universe'].get('equities', []) + \
                  self.config['universe'].get('sectors', [])

        for symbol in universe:
            try:
                asset_data = market_state.get(symbol, {})
                if not asset_data or 'prices' not in asset_data:
                    continue

                prices = asset_data['prices']
                if len(prices) < 50:  # Need enough data for analysis
                    continue

                # Calculate technical indicators
                indicators = self.calculate_technical_indicators(prices)

                # Detect breakout opportunities
                breakout_opp = self.detect_breakout_opportunity(symbol, prices, indicators, current_regime)
                if breakout_opp:
                    opportunities.append(breakout_opp)

                # Detect mean reversion opportunities
                reversion_opp = self.detect_reversion_opportunity(symbol, prices, indicators, current_regime)
                if reversion_opp:
                    opportunities.append(reversion_opp)

                # Detect momentum opportunities
                momentum_opp = self.detect_momentum_opportunity(symbol, prices, indicators, current_regime)
                if momentum_opp:
                    opportunities.append(momentum_opp)

            except Exception as e:
                logger.warning(f"Technical scan failed for {symbol}: {e}")
                continue

        return opportunities

    def calculate_technical_indicators(self, prices: List[float]) -> Dict[str, Any]:
        """Calculate technical indicators for opportunity detection"""

        prices = np.array(prices)
        indicators = {}

        # Simple moving averages
        indicators['sma_20'] = np.convolve(prices, np.ones(20)/20, mode='valid')
        indicators['sma_50'] = np.convolve(prices, np.ones(50)/50, mode='valid')

        # RSI
        deltas = np.diff(prices)
        gains = np.where(deltas > 0, deltas, 0)
        losses = np.where(deltas < 0, -deltas, 0)

        avg_gain = np.convolve(gains, np.ones(14)/14, mode='valid')
        avg_loss = np.convolve(losses, np.ones(14)/14, mode='valid')

        rs = avg_gain / np.where(avg_loss == 0, 0.001, avg_loss)
        indicators['rsi'] = 100 - (100 / (1 + rs))

        # Bollinger Bands
        sma_20 = indicators['sma_20']
        if len(sma_20) > 0:
            rolling_std = []
            for i in range(20, len(prices)):
                window = prices[i-20:i]
                rolling_std.append(np.std(window))
            rolling_std = np.array(rolling_std)

            indicators['bb_upper'] = sma_20 + (rolling_std * 2)
            indicators['bb_lower'] = sma_20 - (rolling_std * 2)

        # MACD
        ema_12 = self.calculate_ema(prices, 12)
        ema_26 = self.calculate_ema(prices, 26)
        if len(ema_12) > 0 and len(ema_26) > 0:
            macd_line = ema_12[-len(ema_26):] - ema_26
            signal_line = self.calculate_ema(macd_line, 9)
            indicators['macd'] = macd_line[-len(signal_line):] - signal_line

        return indicators

    def calculate_ema(self, prices: np.ndarray, period: int) -> np.ndarray:
        """Calculate exponential moving average"""
        if len(prices) < period:
            return np.array([])

        ema = np.zeros(len(prices))
        ema[period-1] = np.mean(prices[:period])

        multiplier = 2 / (period + 1)
        for i in range(period, len(prices)):
            ema[i] = (prices[i] * multiplier) + (ema[i-1] * (1 - multiplier))

        return ema[period-1:]

    def detect_breakout_opportunity(self, symbol: str, prices: List[float],
                                  indicators: Dict[str, Any], regime: str) -> Optional[Dict[str, Any]]:
        """Detect breakout trading opportunities"""

        if len(prices) < 50 or 'bb_upper' not in indicators:
            return None

        current_price = prices[-1]
        bb_upper = indicators['bb_upper']
        sma_20 = indicators['sma_20']

        if len(bb_upper) == 0 or len(sma_20) == 0:
            return None

        # Check for breakout above Bollinger Band
        if (current_price > bb_upper[-1] and
            prices[-2] <= bb_upper[-2] and  # Just broke out
            current_price > sma_20[-1]):   # Above moving average

            # Calculate expected return (conservative estimate)
            expected_return = min((current_price - sma_20[-1]) / sma_20[-1], 0.05)
            expected_vol = np.std(np.diff(prices[-20:])) * np.sqrt(252)

            if expected_return > self.thresholds['min_expected_return']:
                return {
                    'id': f"breakout_{symbol}_{int(datetime.now().timestamp())}",
                    'type': 'technical_breakout',
                    'symbol': symbol,
                    'strategy': 'breakout_trading',
                    'direction': 'long',
                    'entry_price': current_price,
                    'expected_return': expected_return,
                    'expected_volatility': expected_vol,
                    'confidence': 0.7,
                    'timeframe': '1-5 days',
                    'regime': regime,
                    'signal_strength': (current_price - bb_upper[-1]) / bb_upper[-1],
                    'risk_reward_ratio': expected_return / (expected_vol * 1.5)
                }

        return None

    def detect_reversion_opportunity(self, symbol: str, prices: List[float],
                                   indicators: Dict[str, Any], regime: str) -> Optional[Dict[str, Any]]:
        """Detect mean reversion opportunities"""

        if len(prices) < 50 or 'bb_lower' not in indicators or 'rsi' not in indicators:
            return None

        current_price = prices[-1]
        bb_lower = indicators['bb_lower']
        rsi = indicators['rsi']

        if len(bb_lower) == 0 or len(rsi) == 0:
            return None

        # Check for oversold conditions
        if (current_price < bb_lower[-1] and
            rsi[-1] < 30 and  # RSI oversold
            prices[-2] >= bb_lower[-2]):  # Just broke below

            # Calculate expected return
            expected_return = min((bb_lower[-1] - current_price) / current_price, 0.03)
            expected_vol = np.std(np.diff(prices[-20:])) * np.sqrt(252)

            if expected_return > self.thresholds['min_expected_return']:
                return {
                    'id': f"reversion_{symbol}_{int(datetime.now().timestamp())}",
                    'type': 'mean_reversion',
                    'symbol': symbol,
                    'strategy': 'mean_reversion',
                    'direction': 'long',
                    'entry_price': current_price,
                    'expected_return': expected_return,
                    'expected_volatility': expected_vol,
                    'confidence': 0.65,
                    'timeframe': '2-7 days',
                    'regime': regime,
                    'signal_strength': (bb_lower[-1] - current_price) / current_price,
                    'risk_reward_ratio': expected_return / (expected_vol * 1.2)
                }

        return None

    def detect_momentum_opportunity(self, symbol: str, prices: List[float],
                                  indicators: Dict[str, Any], regime: str) -> Optional[Dict[str, Any]]:
        """Detect momentum-based opportunities"""

        if len(prices) < 50 or 'macd' not in indicators:
            return None

        current_price = prices[-1]
        macd = indicators['macd']

        if len(macd) < 2:
            return None

        # Check for MACD crossover (bullish momentum)
        if (macd[-1] > 0 and
            macd[-2] <= 0 and  # Just crossed above zero
            current_price > indicators['sma_20'][-1]):  # Above moving average

            # Calculate expected return based on momentum
            recent_returns = np.diff(np.log(prices[-10:]))
            momentum = np.mean(recent_returns)
            expected_return = max(momentum * 5, 0.01)  # Extrapolate momentum
            expected_vol = np.std(recent_returns) * np.sqrt(252)

            if expected_return > self.thresholds['min_expected_return']:
                return {
                    'id': f"momentum_{symbol}_{int(datetime.now().timestamp())}",
                    'type': 'momentum_divergence',
                    'symbol': symbol,
                    'strategy': 'momentum_trading',
                    'direction': 'long',
                    'entry_price': current_price,
                    'expected_return': expected_return,
                    'expected_volatility': expected_vol,
                    'confidence': 0.6,
                    'timeframe': '3-10 days',
                    'regime': regime,
                    'signal_strength': abs(macd[-1]),
                    'risk_reward_ratio': expected_return / expected_vol
                }

        return None

    async def scan_volume_anomalies(self, market_state: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Scan for unusual volume patterns"""

        opportunities = []

        for symbol in self.config['universe'].get('equities', []):
            try:
                asset_data = market_state.get(symbol, {})
                if not asset_data or 'volumes' not in asset_data:
                    continue

                volumes = asset_data['volumes']
                if len(volumes) < 20:
                    continue

                current_volume = volumes[-1]
                avg_volume = np.mean(volumes[-20:-1])  # Exclude current day

                if avg_volume > 0:
                    volume_ratio = current_volume / avg_volume

                    # Detect volume spike
                    if volume_ratio > 3.0:  # 3x average volume
                        prices = asset_data.get('prices', [])
                        if prices:
                            current_price = prices[-1]
                            price_change = (current_price - prices[-2]) / prices[-2] if len(prices) > 1 else 0

                            # High volume with price movement
                            if abs(price_change) > 0.01:  # 1% price move
                                direction = 'long' if price_change > 0 else 'short'

                                opportunities.append({
                                    'id': f"volume_spike_{symbol}_{int(datetime.now().timestamp())}",
                                    'type': 'volume_anomaly',
                                    'symbol': symbol,
                                    'strategy': 'volume_breakout',
                                    'direction': direction,
                                    'entry_price': current_price,
                                    'expected_return': abs(price_change) * 1.5,  # Expect continuation
                                    'expected_volatility': np.std(np.diff(np.log(prices[-20:]))) * np.sqrt(252),
                                    'confidence': min(volume_ratio / 5.0, 0.8),  # Higher volume = higher confidence
                                    'timeframe': '1-3 days',
                                    'signal_strength': volume_ratio,
                                    'volume_ratio': volume_ratio
                                })

            except Exception as e:
                logger.warning(f"Volume anomaly scan failed for {symbol}: {e}")
                continue

        return opportunities

    async def scan_statistical_arbitrage(self, market_state: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Scan for statistical arbitrage opportunities"""

        opportunities = []

        try:
            # Get pairs of related assets
            pairs = [
                ('SPY', 'QQQ'),  # Broad market vs tech
                ('XLE', 'XLF'),  # Energy vs Financials
                ('IWM', 'SPY'),  # Small cap vs large cap
            ]

            for asset1, asset2 in pairs:
                if asset1 in market_state and asset2 in market_state:
                    prices1 = market_state[asset1].get('prices', [])
                    prices2 = market_state[asset2].get('prices', [])

                    if len(prices1) >= 60 and len(prices2) >= 60:
                        # Calculate spread
                        spread = np.log(prices1[-60:]) - np.log(prices2[-60:])
                        spread_zscore = (spread[-1] - np.mean(spread)) / np.std(spread)

                        # Check for mean reversion opportunity
                        if abs(spread_zscore) > 2.0:  # 2 standard deviations
                            direction = 'long' if spread_zscore < -2 else 'short'
                            expected_return = abs(spread_zscore) * 0.005  # Expected reversion

                            opportunities.append({
                                'id': f"stat_arb_{asset1}_{asset2}_{int(datetime.now().timestamp())}",
                                'type': 'statistical_arbitrage',
                                'symbol': f"{asset1}/{asset2}",
                                'strategy': 'pairs_trading',
                                'direction': direction,
                                'entry_price': spread[-1],
                                'expected_return': expected_return,
                                'expected_volatility': np.std(spread) * np.sqrt(252),
                                'confidence': min(abs(spread_zscore) / 3.0, 0.75),
                                'timeframe': '5-15 days',
                                'signal_strength': abs(spread_zscore),
                                'z_score': spread_zscore,
                                'pair': (asset1, asset2)
                            })

        except Exception as e:
            logger.warning(f"Statistical arbitrage scan failed: {e}")

        return opportunities

    async def scan_options_opportunities(self, market_state: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Scan for options market anomalies"""

        opportunities = []

        try:
            # Check put/call ratio anomalies
            if 'options' in market_state:
                options_data = market_state['options']

                pcr = options_data.get('put_call_ratio')
                if pcr:
                    # Extreme PCR readings
                    if pcr > 1.5:  # Very bearish sentiment
                        opportunities.append({
                            'id': f"options_pcr_{int(datetime.now().timestamp())}",
                            'type': 'options_anomaly',
                            'symbol': 'SPY',  # Market-wide signal
                            'strategy': 'sentiment_reversal',
                            'direction': 'long',
                            'entry_price': market_state.get('SPY', {}).get('prices', [400])[-1],
                            'expected_return': 0.03,  # Expect mean reversion
                            'expected_volatility': 0.25,
                            'confidence': min(pcr / 2.0, 0.7),
                            'timeframe': '1-2 weeks',
                            'signal_strength': pcr,
                            'pcr': pcr
                        })

                    elif pcr < 0.5:  # Very bullish sentiment
                        opportunities.append({
                            'id': f"options_pcr_bearish_{int(datetime.now().timestamp())}",
                            'type': 'options_anomaly',
                            'symbol': 'SPY',
                            'strategy': 'sentiment_reversal',
                            'direction': 'short',
                            'entry_price': market_state.get('SPY', {}).get('prices', [400])[-1],
                            'expected_return': 0.02,
                            'expected_volatility': 0.25,
                            'confidence': min((1/pcr) / 3.0, 0.7),
                            'timeframe': '1-2 weeks',
                            'signal_strength': 1/pcr,
                            'pcr': pcr
                        })

        except Exception as e:
            logger.warning(f"Options opportunity scan failed: {e}")

        return opportunities

    async def scan_news_dislocations(self, market_state: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Scan for news-driven price dislocations"""

        opportunities = []

        try:
            if 'sentiment' in market_state:
                sentiment_data = market_state['sentiment']

                # Extreme sentiment readings
                news_sentiment = sentiment_data.get('news_sentiment')
                if news_sentiment:
                    if news_sentiment < -0.5:  # Very negative news
                        # Look for oversold conditions
                        spy_data = market_state.get('SPY', {})
                        prices = spy_data.get('prices', [])
                        if prices and len(prices) >= 5:
                            recent_return = (prices[-1] - prices[-5]) / prices[-5]
                            if recent_return < -0.05:  # Down 5%+ recently
                                opportunities.append({
                                    'id': f"news_dislocation_{int(datetime.now().timestamp())}",
                                    'type': 'news_dislocation',
                                    'symbol': 'SPY',
                                    'strategy': 'sentiment_reversal',
                                    'direction': 'long',
                                    'entry_price': prices[-1],
                                    'expected_return': abs(news_sentiment) * 0.02,
                                    'expected_volatility': 0.30,
                                    'confidence': min(abs(news_sentiment) * 2, 0.8),
                                    'timeframe': '3-7 days',
                                    'signal_strength': abs(news_sentiment),
                                    'news_sentiment': news_sentiment
                                })

        except Exception as e:
            logger.warning(f"News dislocation scan failed: {e}")

        return opportunities

    async def scan_strategy_opportunities(self, market_state: Dict[str, Any],
                                        strategy_portfolio: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Scan for opportunities based on existing strategies"""

        opportunities = []

        for strategy_id, strategy in strategy_portfolio.items():
            try:
                # Check if strategy conditions are met
                if self.strategy_conditions_met(strategy, market_state):
                    opp = {
                        'id': f"strategy_{strategy_id}_{int(datetime.now().timestamp())}",
                        'type': 'strategy_signal',
                        'symbol': strategy.get('universe', ['SPY'])[0],
                        'strategy': strategy['name'],
                        'direction': 'long',  # Assume long for now
                        'entry_price': market_state.get('SPY', {}).get('prices', [400])[-1],
                        'expected_return': strategy.get('performance', {}).get('avg_return', 0.02),
                        'expected_volatility': strategy.get('performance', {}).get('volatility', 0.25),
                        'confidence': strategy.get('performance', {}).get('win_rate', 0.5),
                        'timeframe': '1-5 days',
                        'strategy_id': strategy_id
                    }
                    opportunities.append(opp)

            except Exception as e:
                logger.warning(f"Strategy opportunity scan failed for {strategy_id}: {e}")
                continue

        return opportunities

    def strategy_conditions_met(self, strategy: Dict[str, Any], market_state: Dict[str, Any]) -> bool:
        """Check if strategy entry conditions are met"""
        # Simplified condition checking
        # In practice, this would implement the actual strategy logic
        return True  # Placeholder

    def filter_opportunities(self, opportunities: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Filter opportunities based on risk and quality criteria"""

        filtered = []

        for opp in opportunities:
            # Check basic criteria
            if (opp.get('expected_return', 0) >= self.thresholds['min_expected_return'] and
                opp.get('confidence', 0) >= self.thresholds['confidence_threshold'] and
                opp.get('expected_volatility', 1) <= self.thresholds['max_volatility']):

                # Check cache to avoid duplicates
                cache_key = f"{opp['type']}_{opp['symbol']}"
                if cache_key not in self.opportunity_cache:
                    filtered.append(opp)
                else:
                    # Check if enough time has passed
                    cached_time = self.opportunity_cache[cache_key]['timestamp']
                    if datetime.now() - cached_time > self.cache_timeout:
                        filtered.append(opp)

        return filtered

    def rank_opportunities(self, opportunities: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Rank opportunities by expected value and risk-adjusted return"""

        def opportunity_score(opp):
            expected_return = opp.get('expected_return', 0)
            volatility = opp.get('expected_volatility', 0.25)
            confidence = opp.get('confidence', 0.5)

            # Risk-adjusted score
            sharpe_like = expected_return / max(volatility, 0.01)

            # Weighted score
            score = (sharpe_like * 0.6 +
                    expected_return * 0.2 +
                    confidence * 0.2)

            return score

        # Sort by score (descending)
        ranked = sorted(opportunities, key=opportunity_score, reverse=True)

        # Add ranking and timestamp
        for i, opp in enumerate(ranked):
            opp['rank'] = i + 1
            opp['timestamp'] = datetime.now().isoformat()
            opp['expected_value'] = opportunity_score(opp)

        return ranked

    def update_opportunity_cache(self, opportunities: List[Dict[str, Any]]):
        """Update opportunity cache and save to disk"""

        for opp in opportunities[:10]:  # Cache top 10
            cache_key = f"{opp['type']}_{opp['symbol']}"
            self.opportunity_cache[cache_key] = {
                'timestamp': datetime.now(),
                'opportunity': opp
            }

        # Clean old cache entries
        cutoff_time = datetime.now() - self.cache_timeout
        self.opportunity_cache = {
            k: v for k, v in self.opportunity_cache.items()
            if v['timestamp'] > cutoff_time
        }

        # Save opportunities to disk for user visibility
        self.save_opportunities_to_disk()

    def save_opportunities_to_disk(self):
        """Save current opportunities to JSON file for user access"""

        try:
            import json
            from pathlib import Path

            # Prepare opportunities for JSON serialization
            opportunities_for_json = []
            for cache_key, opp_data in self.opportunity_cache.items():
                opp = opp_data['opportunity'].copy()
                opp['discovered_at'] = opp_data['timestamp'].isoformat()
                opp['cache_key'] = cache_key
                opportunities_for_json.append(opp)

            # Save to file
            opportunities_file = Path("opportunity_cache.json")
            with open(opportunities_file, 'w') as f:
                json.dump({
                    'timestamp': datetime.now().isoformat(),
                    'total_opportunities': len(opportunities_for_json),
                    'opportunities': opportunities_for_json
                }, f, indent=2, default=str)

            logger.info(f"💾 Saved {len(opportunities_for_json)} opportunities to opportunity_cache.json")

        except Exception as e:
            logger.warning(f"Could not save opportunities to disk: {e}")

    def get_opportunity_statistics(self) -> Dict[str, Any]:
        """Get statistics about discovered opportunities"""

        if not self.opportunity_cache:
            return {'error': 'No opportunities in cache'}

        # Count by type
        type_counts = {}
        confidence_levels = []
        expected_returns = []

        for opp_data in self.opportunity_cache.values():
            opp = opp_data['opportunity']
            opp_type = opp.get('type', 'unknown')
            type_counts[opp_type] = type_counts.get(opp_type, 0) + 1

            confidence_levels.append(opp.get('confidence', 0))
            expected_returns.append(opp.get('expected_return', 0))

        return {
            'total_opportunities': len(self.opportunity_cache),
            'opportunities_by_type': type_counts,
            'avg_confidence': np.mean(confidence_levels) if confidence_levels else 0,
            'avg_expected_return': np.mean(expected_returns) if expected_returns else 0,
            'cache_size': len(self.opportunity_cache),
            'most_common_type': max(type_counts.items(), key=lambda x: x[1])[0] if type_counts else 'none'
        }
