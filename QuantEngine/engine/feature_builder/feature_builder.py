"""
Feature Builder for AI Quant Trading System

Creates technical indicators, signals, and labels from raw market data.
Handles both price-based and microstructure features.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
import logging
import talib
from scipy import stats
import numba as nb

logger = logging.getLogger(__name__)


class FeatureBuilder:
    """Builds features and signals from market data"""

    def __init__(self, config: Dict[str, Any]):
        self.config = config

    def build_features(self, market_data: Dict[str, pd.DataFrame],
                      feature_config: Optional[Dict[str, Any]] = None) -> Dict[str, pd.DataFrame]:
        """
        Build comprehensive feature set from market data

        Args:
            market_data: Dictionary of price DataFrames
            feature_config: Feature configuration (optional)

        Returns:
            Dictionary of feature DataFrames
        """

        if feature_config is None:
            feature_config = self._get_default_feature_config()

        feature_data = {}

        for ticker, price_df in market_data.items():
            try:
                features_df = self._build_ticker_features(price_df, feature_config)
                feature_data[ticker] = features_df
            except Exception as e:
                logger.error(f"Failed to build features for {ticker}: {e}")
                continue

        return feature_data

    def _get_default_feature_config(self) -> Dict[str, Any]:
        """Get default feature configuration"""
        return {
            'price_features': {
                'returns': {'windows': [1, 5, 20, 60]},
                'volatility': {'windows': [5, 20, 60]},
                'momentum': {'windows': [1, 5, 20]},
                'moving_averages': {'windows': [5, 10, 20, 50, 200]},
                'rsi': {'windows': [14]},
                'macd': {'fast': 12, 'slow': 26, 'signal': 9},
                'bollinger': {'window': 20, 'std_dev': 2},
            },
            'microstructure': {
                'gaps': True,
                'range_compression': True,
                'realized_vol_proxy': {'window': 20}
            },
            'labels': {
                'forward_returns': {'horizons': [1, 5, 20]},
                'regime_labels': True
            }
        }

    def _build_ticker_features(self, price_df: pd.DataFrame,
                              config: Dict[str, Any]) -> pd.DataFrame:
        """Build features for a single ticker"""

        # Start with basic price data
        features = price_df.copy()

        # Price-based features
        if 'price_features' in config:
            features = self._add_price_features(features, config['price_features'])

        # Microstructure features
        if 'microstructure' in config:
            features = self._add_microstructure_features(features, config['microstructure'])

        # Labels
        if 'labels' in config:
            features = self._add_labels(features, config['labels'])

        return features

    def _add_price_features(self, df: pd.DataFrame, config: Dict[str, Any]) -> pd.DataFrame:
        """Add price-based technical features"""

        # Returns at different horizons
        if 'returns' in config:
            for window in config['returns']['windows']:
                df[f'returns_{window}d'] = df['close'].pct_change(window)

        # Volatility measures
        if 'volatility' in config:
            for window in config['volatility']['windows']:
                # Realized volatility (annualized)
                returns = df['close'].pct_change()
                df[f'vol_{window}d'] = returns.rolling(window).std() * np.sqrt(252)

                # Parkinson volatility (if OHLC available)
                if all(col in df.columns for col in ['high', 'low']):
                    hl_ratio = np.log(df['high'] / df['low']) ** 2
                    df[f'parkinson_vol_{window}d'] = np.sqrt(hl_ratio.rolling(window).mean() * 252 / (4 * np.log(2)))

        # Momentum features
        if 'momentum' in config:
            for window in config['momentum']['windows']:
                df[f'momentum_{window}d'] = (df['close'] / df['close'].shift(window) - 1)

        # Moving averages
        if 'moving_averages' in config:
            for window in config['moving_averages']['windows']:
                df[f'sma_{window}'] = df['close'].rolling(window).mean()
                df[f'ema_{window}'] = df['close'].ewm(span=window).mean()

        # RSI
        if 'rsi' in config:
            for window in config['rsi']['windows']:
                df[f'rsi_{window}'] = self._calculate_rsi(df['close'], window)

        # MACD
        if 'macd' in config:
            macd_config = config['macd']
            macd_line, signal_line, hist = self._calculate_macd(
                df['close'],
                macd_config['fast'],
                macd_config['slow'],
                macd_config['signal']
            )
            df['macd'] = macd_line
            df['macd_signal'] = signal_line
            df['macd_hist'] = hist

        # Bollinger Bands
        if 'bollinger' in config:
            bb_config = config['bollinger']
            upper, middle, lower = self._calculate_bollinger_bands(
                df['close'],
                bb_config['window'],
                bb_config['std_dev']
            )
            df['bb_upper'] = upper
            df['bb_middle'] = middle
            df['bb_lower'] = lower
            df['bb_position'] = (df['close'] - lower) / (upper - lower)

        return df

    def _calculate_rsi(self, prices: pd.Series, period: int = 14) -> pd.Series:
        """Calculate RSI indicator"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(period).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))

    def _calculate_macd(self, prices: pd.Series, fast: int = 12, slow: int = 26,
                        signal: int = 9) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """Calculate MACD indicator"""
        ema_fast = prices.ewm(span=fast).mean()
        ema_slow = prices.ewm(span=slow).mean()
        macd_line = ema_fast - ema_slow
        signal_line = macd_line.ewm(span=signal).mean()
        histogram = macd_line - signal_line
        return macd_line, signal_line, histogram

    def _calculate_bollinger_bands(self, prices: pd.Series, window: int = 20,
                                  std_dev: float = 2) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """Calculate Bollinger Bands"""
        middle = prices.rolling(window).mean()
        std = prices.rolling(window).std()
        upper = middle + (std * std_dev)
        lower = middle - (std * std_dev)
        return upper, middle, lower

    def _add_microstructure_features(self, df: pd.DataFrame, config: Dict[str, Any]) -> pd.DataFrame:
        """Add microstructure-based features"""

        # Gap features
        if config.get('gaps', False):
            df['gap_up'] = (df['open'] / df['close'].shift(1) - 1) > 0.02
            df['gap_down'] = (df['open'] / df['close'].shift(1) - 1) < -0.02
            df['gap_size'] = np.log(df['open'] / df['close'].shift(1))

        # Range compression (relative range)
        if config.get('range_compression', False):
            df['daily_range'] = (df['high'] - df['low']) / df['close']
            df['range_compression_20'] = df['daily_range'].rolling(20).mean() / df['daily_range'].rolling(60).mean()

        # Realized volatility proxy (simplified VRP)
        if 'realized_vol_proxy' in config:
            window = config['realized_vol_proxy']['window']
            returns = df['close'].pct_change()

            # Rolling realized vol
            df[f'rv_proxy_{window}'] = returns.rolling(window).std() * np.sqrt(252)

            # VRP: difference between implied and realized vol
            # Note: This is a proxy - real VRP would use options data
            df[f'vrp_{window}'] = df[f'rv_proxy_{window}'].rolling(window).mean()

        return df

    def _add_labels(self, df: pd.DataFrame, config: Dict[str, Any]) -> pd.DataFrame:
        """Add prediction labels"""

        # Forward returns at different horizons
        if 'forward_returns' in config:
            for horizon in config['forward_returns']['horizons']:
                df[f'forward_return_{horizon}d'] = df['close'].shift(-horizon) / df['close'] - 1

                # Hit ratio (binary classification target)
                df[f'hit_ratio_{horizon}d'] = (df[f'forward_return_{horizon}d'] > 0).astype(int)

                # Asymmetry measure
                df[f'asymmetry_{horizon}d'] = df[f'forward_return_{horizon}d'].abs() * np.sign(df[f'forward_return_{horizon}d'])

        # Regime labels (bull/bear/chop)
        if config.get('regime_labels', False):
            df = self._add_regime_labels(df)

        return df

    def _add_regime_labels(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add market regime labels"""

        # Trend regime (based on 200-day MA)
        if 'sma_200' in df.columns:
            df['trend_regime'] = np.where(df['close'] > df['sma_200'], 1, -1)
        else:
            # Fallback: use momentum
            df['trend_regime'] = np.where(df['close'] > df['close'].shift(200), 1, -1)

        # Volatility regime (high/low vol)
        if 'vol_20d' in df.columns:
            vol_median = df['vol_20d'].rolling(252).median()
            df['vol_regime'] = np.where(df['vol_20d'] > vol_median, 1, -1)
        else:
            df['vol_regime'] = 0  # Neutral if no vol data

        # Combined regime (simplified)
        df['market_regime'] = df['trend_regime'] * df['vol_regime']

        # Categorical regime labels
        conditions = [
            (df['trend_regime'] == 1) & (df['vol_regime'] == -1),  # Bull low vol
            (df['trend_regime'] == 1) & (df['vol_regime'] == 1),   # Bull high vol
            (df['trend_regime'] == -1) & (df['vol_regime'] == -1), # Bear low vol
            (df['trend_regime'] == -1) & (df['vol_regime'] == 1),  # Bear high vol
        ]
        choices = ['bull_low_vol', 'bull_high_vol', 'bear_low_vol', 'bear_high_vol']
        df['regime_category'] = np.select(conditions, choices, default='neutral')

        return df

    def create_signal_library(self) -> Dict[str, Dict[str, Any]]:
        """
        Create a library of predefined signal functions

        Returns:
            Dictionary of signal definitions
        """

        signals = {
            'ma_cross': {
                'description': 'Moving average crossover',
                'params': {'fast_period': 20, 'slow_period': 200},
                'function': lambda df, fast, slow: (df['close'].rolling(fast).mean() > df['close'].rolling(slow).mean()).astype(int)
            },

            'rsi_oversold': {
                'description': 'RSI oversold signal',
                'params': {'period': 14, 'threshold': 30},
                'function': lambda df, period, thresh: (self._calculate_rsi(df['close'], period) < thresh).astype(int)
            },

            'vol_breakout': {
                'description': 'Volume breakout signal',
                'params': {'window': 20, 'multiplier': 1.5},
                'function': lambda df, window, mult: (df['volume'] > df['volume'].rolling(window).mean() * mult).astype(int)
            },

            'bb_squeeze': {
                'description': 'Bollinger Band squeeze',
                'params': {'window': 20, 'threshold': 0.1},
                'function': lambda df, window, thresh: ((df['bb_upper'] - df['bb_lower']) / df['bb_middle'] < thresh).astype(int)
            }
        }

        return signals

    def generate_signal_from_template(self, signal_type: str, params: Dict[str, Any],
                                    data: pd.DataFrame) -> pd.Series:
        """
        Generate a signal from predefined templates

        Args:
            signal_type: Type of signal to generate
            params: Signal parameters
            data: Input data

        Returns:
            Signal series
        """

        signal_library = self.create_signal_library()

        if signal_type not in signal_library:
            raise ValueError(f"Unknown signal type: {signal_type}")

        signal_def = signal_library[signal_type]
        signal_func = signal_def['function']

        # Merge default params with provided params
        full_params = {**signal_def['params'], **params}

        try:
            return signal_func(data, **full_params)
        except Exception as e:
            logger.error(f"Failed to generate {signal_type} signal: {e}")
            return pd.Series(0, index=data.index)

