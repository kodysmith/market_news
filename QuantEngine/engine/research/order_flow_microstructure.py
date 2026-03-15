"""
Order Flow and Market Microstructure Analysis

Analyzes trade-level and order book data to extract signals from:
- VWAP/TWAP calculations and deviation signals
- Order book imbalance and Kyle's Lambda (price impact)
- VPIN (Volume-Synchronized Probability of Informed Trading)
- Trade flow classification (Lee-Ready algorithm)
- Spread estimation (effective spread, Roll implied spread)

Provides estimation methods from OHLCV data when full order book
data is not available.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
import logging
from datetime import datetime, timedelta
from scipy import stats

logger = logging.getLogger(__name__)


class OrderFlowAnalyzer:
    """
    Order flow and market microstructure analysis engine
    """

    def __init__(self, config: Dict[str, Any]):
        self.config = config

        # VWAP/TWAP parameters
        self.vwap_config = {
            'deviation_threshold': config.get('vwap_deviation_threshold', 0.02),
            'signal_smoothing': config.get('vwap_signal_smoothing', 5),
        }

        # Order imbalance parameters
        self.imbalance_config = {
            'lookback_window': config.get('imbalance_lookback', 20),
            'significance_threshold': config.get('imbalance_significance', 0.6),
        }

        # VPIN parameters
        self.vpin_config = {
            'default_bucket_size': config.get('vpin_bucket_size', 50000),
            'default_n_buckets': config.get('vpin_n_buckets', 50),
            'alert_threshold': config.get('vpin_alert_threshold', 0.7),
        }

        # Trade flow parameters
        self.flow_config = {
            'large_trade_percentile': config.get('large_trade_percentile', 95),
            'arrival_rate_window': config.get('arrival_rate_window', 60),
            'spike_std_threshold': config.get('spike_std_threshold', 3.0),
        }

    # ─── VWAP / TWAP Analysis ───────────────────────────────────────────

    def calculate_vwap(self, ohlcv: pd.DataFrame) -> pd.Series:
        """
        Calculate Volume-Weighted Average Price.

        Uses typical price (high + low + close) / 3 weighted by volume.
        Computes cumulative VWAP across the session.

        Args:
            ohlcv: DataFrame with columns [open, high, low, close, volume]

        Returns:
            Series of cumulative VWAP values
        """
        try:
            required_cols = {'high', 'low', 'close', 'volume'}
            if not required_cols.issubset(set(ohlcv.columns)):
                logger.error(f"VWAP requires columns {required_cols}, got {list(ohlcv.columns)}")
                return pd.Series(dtype=float)

            typical_price = (ohlcv['high'] + ohlcv['low'] + ohlcv['close']) / 3.0
            cum_tp_volume = (typical_price * ohlcv['volume']).cumsum()
            cum_volume = ohlcv['volume'].cumsum()

            vwap = cum_tp_volume / cum_volume.replace(0, np.nan)
            vwap.name = 'vwap'

            logger.info(f"Calculated VWAP for {len(ohlcv)} bars")
            return vwap

        except Exception as e:
            logger.error(f"VWAP calculation failed: {e}")
            return pd.Series(dtype=float)

    def calculate_twap(self, ohlcv: pd.DataFrame, interval_minutes: int = 5) -> pd.Series:
        """
        Calculate Time-Weighted Average Price.

        Computes a rolling average of typical price over fixed time intervals,
        giving equal weight to each time period regardless of volume.

        Args:
            ohlcv: DataFrame with columns [high, low, close] and a DatetimeIndex
            interval_minutes: Resampling interval in minutes

        Returns:
            Series of TWAP values
        """
        try:
            required_cols = {'high', 'low', 'close'}
            if not required_cols.issubset(set(ohlcv.columns)):
                logger.error(f"TWAP requires columns {required_cols}, got {list(ohlcv.columns)}")
                return pd.Series(dtype=float)

            typical_price = (ohlcv['high'] + ohlcv['low'] + ohlcv['close']) / 3.0

            # If we have a datetime index, resample; otherwise use rolling window
            if isinstance(ohlcv.index, pd.DatetimeIndex):
                resampled = typical_price.resample(f'{interval_minutes}min').mean()
                twap = resampled.reindex(ohlcv.index, method='ffill')
            else:
                # Fall back to rolling mean when no datetime index
                twap = typical_price.rolling(window=max(interval_minutes, 1), min_periods=1).mean()

            twap.name = 'twap'
            logger.info(f"Calculated TWAP ({interval_minutes}min) for {len(ohlcv)} bars")
            return twap

        except Exception as e:
            logger.error(f"TWAP calculation failed: {e}")
            return pd.Series(dtype=float)

    def vwap_deviation_signal(self, ohlcv: pd.DataFrame) -> pd.DataFrame:
        """
        Generate trading signal from price deviation relative to VWAP.

        Above VWAP = bullish bias (positive signal)
        Below VWAP = bearish bias (negative signal)

        The signal magnitude is the percentage deviation from VWAP,
        smoothed with an EMA to reduce noise.

        Args:
            ohlcv: DataFrame with columns [high, low, close, volume]

        Returns:
            DataFrame with columns [vwap, deviation, signal]
        """
        try:
            vwap = self.calculate_vwap(ohlcv)
            if vwap.empty:
                return pd.DataFrame()

            deviation = (ohlcv['close'] - vwap) / vwap

            smoothing = self.vwap_config['signal_smoothing']
            smoothed_deviation = deviation.ewm(span=smoothing, adjust=False).mean()

            threshold = self.vwap_config['deviation_threshold']
            signal = pd.Series(0.0, index=ohlcv.index)
            signal[smoothed_deviation > threshold] = 1.0
            signal[smoothed_deviation < -threshold] = -1.0
            # Proportional signal for intermediate values
            mask_mid = (signal == 0.0) & (smoothed_deviation != 0.0)
            signal[mask_mid] = smoothed_deviation[mask_mid] / threshold

            result = pd.DataFrame({
                'vwap': vwap,
                'deviation': deviation,
                'signal': signal,
            }, index=ohlcv.index)

            bullish = (signal > 0).sum()
            bearish = (signal < 0).sum()
            logger.info(f"VWAP deviation signal: {bullish} bullish, {bearish} bearish bars")
            return result

        except Exception as e:
            logger.error(f"VWAP deviation signal failed: {e}")
            return pd.DataFrame()

    # ─── Order Book Imbalance ────────────────────────────────────────────

    def calculate_order_imbalance(self, trades_df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate buy vs sell volume imbalance using tick rule (Lee-Ready).

        Classifies each trade as buyer- or seller-initiated, then computes
        rolling imbalance ratio: (buy_volume - sell_volume) / total_volume.

        Args:
            trades_df: DataFrame with columns [price, volume] at minimum.
                       Optionally [bid, ask] for quote-based classification.

        Returns:
            DataFrame with columns [trade_sign, buy_volume, sell_volume, imbalance]
        """
        try:
            classified = self.classify_trades_lee_ready(trades_df)
            if classified.empty:
                return pd.DataFrame()

            classified['signed_volume'] = classified['trade_sign'] * classified['volume']
            classified['buy_volume'] = classified['signed_volume'].clip(lower=0)
            classified['sell_volume'] = (-classified['signed_volume']).clip(lower=0)

            window = self.imbalance_config['lookback_window']
            rolling_buy = classified['buy_volume'].rolling(window=window, min_periods=1).sum()
            rolling_sell = classified['sell_volume'].rolling(window=window, min_periods=1).sum()
            rolling_total = rolling_buy + rolling_sell

            imbalance = (rolling_buy - rolling_sell) / rolling_total.replace(0, np.nan)

            result = pd.DataFrame({
                'trade_sign': classified['trade_sign'],
                'buy_volume': classified['buy_volume'],
                'sell_volume': classified['sell_volume'],
                'imbalance': imbalance,
            }, index=trades_df.index)

            logger.info(
                f"Order imbalance calculated: mean={imbalance.mean():.4f}, "
                f"std={imbalance.std():.4f}"
            )
            return result

        except Exception as e:
            logger.error(f"Order imbalance calculation failed: {e}")
            return pd.DataFrame()

    def calculate_kyle_lambda(self, trades_df: pd.DataFrame, returns: pd.Series) -> Dict:
        """
        Estimate Kyle's Lambda — the price impact per unit of signed order flow.

        Regresses returns on net signed volume (OLS):
            r_t = alpha + lambda * signed_volume_t + epsilon_t

        A higher lambda indicates lower liquidity / higher price impact.

        Args:
            trades_df: DataFrame with columns [price, volume] (and optionally [bid, ask])
            returns: Series of returns aligned with trades_df

        Returns:
            Dict with keys: lambda, t_stat, r_squared, p_value, n_obs
        """
        try:
            classified = self.classify_trades_lee_ready(trades_df)
            if classified.empty:
                return {'error': 'Trade classification failed'}

            signed_volume = classified['trade_sign'] * classified['volume']

            # Align series
            common_idx = returns.index.intersection(signed_volume.index)
            if len(common_idx) < 10:
                return {'error': 'Insufficient aligned observations', 'n_obs': len(common_idx)}

            y = returns.loc[common_idx].values
            x = signed_volume.loc[common_idx].values

            # OLS regression with constant
            x_with_const = np.column_stack([np.ones(len(x)), x])
            try:
                beta, residuals, rank, sv = np.linalg.lstsq(x_with_const, y, rcond=None)
            except np.linalg.LinAlgError:
                return {'error': 'Regression failed — singular matrix'}

            kyle_lambda = float(beta[1])
            y_hat = x_with_const @ beta
            ss_res = np.sum((y - y_hat) ** 2)
            ss_tot = np.sum((y - np.mean(y)) ** 2)
            r_squared = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0

            # Standard error and t-stat for lambda
            n = len(y)
            dof = n - 2
            if dof > 0:
                mse = ss_res / dof
                var_beta = mse * np.linalg.inv(x_with_const.T @ x_with_const).diagonal()
                se_lambda = np.sqrt(var_beta[1]) if var_beta[1] > 0 else np.nan
                t_stat = kyle_lambda / se_lambda if se_lambda > 0 else np.nan
                p_value = float(2 * stats.t.sf(abs(t_stat), dof)) if not np.isnan(t_stat) else np.nan
            else:
                t_stat = np.nan
                p_value = np.nan

            result = {
                'lambda': kyle_lambda,
                't_stat': float(t_stat),
                'r_squared': float(r_squared),
                'p_value': float(p_value),
                'n_obs': n,
                'interpretation': (
                    'high_impact' if kyle_lambda > 0 and p_value < 0.05
                    else 'low_impact' if p_value < 0.05
                    else 'insignificant'
                ),
            }

            logger.info(
                f"Kyle's Lambda: {kyle_lambda:.6f} (t={t_stat:.2f}, R²={r_squared:.4f})"
            )
            return result

        except Exception as e:
            logger.error(f"Kyle's Lambda calculation failed: {e}")
            return {'error': str(e)}

    # ─── VPIN ────────────────────────────────────────────────────────────

    def calculate_vpin(self, trades_df: pd.DataFrame,
                       volume_bucket_size: int = 50000,
                       n_buckets: int = 50) -> pd.DataFrame:
        """
        Calculate Volume-Synchronized Probability of Informed Trading.

        1. Classify trades as buy/sell using Lee-Ready.
        2. Group trades into equal-volume buckets.
        3. For each bucket, compute |buy_volume - sell_volume| / bucket_volume.
        4. VPIN = rolling average of bucket imbalance over n_buckets.

        Args:
            trades_df: DataFrame with columns [price, volume]
            volume_bucket_size: Volume per bucket (trade-clock sampling)
            n_buckets: Number of buckets for rolling VPIN average

        Returns:
            DataFrame with columns [bucket_id, bucket_buy_volume, bucket_sell_volume,
                                    bucket_imbalance, vpin]
        """
        try:
            classified = self.classify_trades_lee_ready(trades_df)
            if classified.empty:
                return pd.DataFrame()

            classified = classified.copy()
            classified['buy_volume'] = (classified['trade_sign'].clip(lower=0)) * classified['volume']
            classified['sell_volume'] = ((-classified['trade_sign']).clip(lower=0)) * classified['volume']

            # Assign trades to volume buckets
            cum_volume = classified['volume'].cumsum()
            classified['bucket_id'] = (cum_volume / volume_bucket_size).astype(int)

            # Aggregate by bucket
            bucket_agg = classified.groupby('bucket_id').agg(
                bucket_buy_volume=('buy_volume', 'sum'),
                bucket_sell_volume=('sell_volume', 'sum'),
                bucket_total_volume=('volume', 'sum'),
                bucket_end_time=('price', 'count'),
            ).reset_index()

            # Bucket imbalance
            bucket_agg['bucket_imbalance'] = (
                np.abs(bucket_agg['bucket_buy_volume'] - bucket_agg['bucket_sell_volume'])
                / bucket_agg['bucket_total_volume'].replace(0, np.nan)
            )

            # Rolling VPIN
            bucket_agg['vpin'] = (
                bucket_agg['bucket_imbalance']
                .rolling(window=n_buckets, min_periods=1)
                .mean()
            )

            logger.info(
                f"VPIN calculated: {len(bucket_agg)} buckets, "
                f"latest VPIN={bucket_agg['vpin'].iloc[-1]:.4f}"
            )

            return bucket_agg[['bucket_id', 'bucket_buy_volume', 'bucket_sell_volume',
                               'bucket_imbalance', 'vpin']]

        except Exception as e:
            logger.error(f"VPIN calculation failed: {e}")
            return pd.DataFrame()

    def vpin_alert(self, vpin_series: pd.Series, threshold: float = 0.7) -> Dict:
        """
        Generate alert when VPIN exceeds threshold, indicating high
        probability of informed trading and potential adverse selection risk.

        Args:
            vpin_series: Series of VPIN values
            threshold: Alert threshold (default 0.7)

        Returns:
            Dict with alert status, current VPIN, breach history, and statistics
        """
        try:
            if vpin_series.empty:
                return {'alert': False, 'error': 'Empty VPIN series'}

            current_vpin = float(vpin_series.iloc[-1])
            is_alert = current_vpin >= threshold

            breaches = vpin_series[vpin_series >= threshold]
            breach_pct = len(breaches) / len(vpin_series) if len(vpin_series) > 0 else 0.0

            result = {
                'alert': is_alert,
                'current_vpin': current_vpin,
                'threshold': threshold,
                'breach_count': len(breaches),
                'breach_pct': float(breach_pct),
                'vpin_mean': float(vpin_series.mean()),
                'vpin_std': float(vpin_series.std()),
                'vpin_max': float(vpin_series.max()),
                'vpin_percentile': float(stats.percentileofscore(vpin_series.dropna(), current_vpin)),
                'timestamp': datetime.now().isoformat(),
            }

            if is_alert:
                logger.warning(
                    f"VPIN ALERT: {current_vpin:.4f} >= {threshold} — "
                    f"high informed trading probability"
                )
            else:
                logger.info(f"VPIN check: {current_vpin:.4f} (threshold {threshold})")

            return result

        except Exception as e:
            logger.error(f"VPIN alert check failed: {e}")
            return {'alert': False, 'error': str(e)}

    # ─── Trade Flow Analysis ─────────────────────────────────────────────

    def classify_trades_lee_ready(self, trades_df: pd.DataFrame) -> pd.DataFrame:
        """
        Classify trades as buyer- or seller-initiated using the Lee-Ready algorithm.

        1. Quote test: if bid and ask are available, compare trade price to midpoint.
           - price > midpoint → buy (+1)
           - price < midpoint → sell (-1)
           - price == midpoint → fall through to tick test

        2. Tick test: compare to previous trade price.
           - uptick → buy (+1)
           - downtick → sell (-1)
           - zero tick → use last non-zero tick direction

        Args:
            trades_df: DataFrame with column [price, volume].
                       Optional columns: [bid, ask] for quote test.

        Returns:
            DataFrame with original columns plus [trade_sign] (+1 buy, -1 sell)
        """
        try:
            if 'price' not in trades_df.columns or 'volume' not in trades_df.columns:
                logger.error("Trade classification requires [price, volume] columns")
                return pd.DataFrame()

            df = trades_df.copy()
            n = len(df)
            signs = np.zeros(n, dtype=float)

            # Quote test (if bid/ask available)
            has_quotes = 'bid' in df.columns and 'ask' in df.columns
            if has_quotes:
                midpoint = (df['bid'] + df['ask']) / 2.0
                signs[df['price'] > midpoint] = 1.0
                signs[df['price'] < midpoint] = -1.0
                unclassified = signs == 0.0
            else:
                unclassified = np.ones(n, dtype=bool)

            # Tick test for unclassified trades
            price_diff = df['price'].diff()
            last_sign = 0.0
            for i in range(n):
                if not unclassified[i]:
                    last_sign = signs[i]
                    continue

                if i == 0:
                    # First trade — cannot classify via tick test, default buy
                    signs[i] = 1.0
                    last_sign = 1.0
                    continue

                diff = price_diff.iloc[i]
                if diff > 0:
                    signs[i] = 1.0
                    last_sign = 1.0
                elif diff < 0:
                    signs[i] = -1.0
                    last_sign = -1.0
                else:
                    # Zero tick — use last known direction
                    signs[i] = last_sign if last_sign != 0.0 else 1.0

            df['trade_sign'] = signs

            buys = (signs > 0).sum()
            sells = (signs < 0).sum()
            logger.info(f"Lee-Ready classification: {buys} buys, {sells} sells out of {n} trades")
            return df

        except Exception as e:
            logger.error(f"Lee-Ready trade classification failed: {e}")
            return pd.DataFrame()

    def calculate_flow_toxicity(self, trades_df: pd.DataFrame) -> Dict:
        """
        Calculate order flow toxicity metrics measuring adverse selection cost.

        Toxicity is estimated as the average absolute order imbalance
        relative to total volume, combined with price impact analysis.

        Args:
            trades_df: DataFrame with columns [price, volume]

        Returns:
            Dict with toxicity_score, adverse_selection_cost, flow_metrics
        """
        try:
            classified = self.classify_trades_lee_ready(trades_df)
            if classified.empty:
                return {'error': 'Trade classification failed'}

            signed_volume = classified['trade_sign'] * classified['volume']
            total_volume = classified['volume'].sum()

            # Net order flow
            net_flow = signed_volume.sum()
            flow_imbalance = abs(net_flow) / total_volume if total_volume > 0 else 0.0

            # Rolling toxicity (short windows)
            window = min(50, len(classified) // 5) if len(classified) >= 10 else len(classified)
            rolling_signed = signed_volume.rolling(window=max(window, 1), min_periods=1).sum()
            rolling_total = classified['volume'].rolling(window=max(window, 1), min_periods=1).sum()
            rolling_imbalance = (rolling_signed.abs() / rolling_total.replace(0, np.nan)).dropna()

            # Adverse selection cost: average price move in direction of trade sign
            price_changes = classified['price'].pct_change().dropna()
            if len(price_changes) > 0:
                aligned_signs = classified['trade_sign'].iloc[1:]
                adverse_cost = (price_changes.values * aligned_signs.values).mean()
            else:
                adverse_cost = 0.0

            toxicity_score = float(rolling_imbalance.mean()) if len(rolling_imbalance) > 0 else 0.0

            result = {
                'toxicity_score': toxicity_score,
                'adverse_selection_cost': float(adverse_cost),
                'flow_imbalance': float(flow_imbalance),
                'net_flow': float(net_flow),
                'total_volume': float(total_volume),
                'mean_rolling_imbalance': float(rolling_imbalance.mean()) if len(rolling_imbalance) > 0 else 0.0,
                'max_rolling_imbalance': float(rolling_imbalance.max()) if len(rolling_imbalance) > 0 else 0.0,
                'interpretation': (
                    'high_toxicity' if toxicity_score > 0.5
                    else 'moderate_toxicity' if toxicity_score > 0.3
                    else 'low_toxicity'
                ),
            }

            logger.info(
                f"Flow toxicity: score={toxicity_score:.4f}, "
                f"adverse_selection={adverse_cost:.6f}"
            )
            return result

        except Exception as e:
            logger.error(f"Flow toxicity calculation failed: {e}")
            return {'error': str(e)}

    def detect_large_trades(self, trades_df: pd.DataFrame,
                            threshold_percentile: float = 95) -> pd.DataFrame:
        """
        Identify institutional-size trades above a volume percentile threshold.

        Args:
            trades_df: DataFrame with columns [price, volume]
            threshold_percentile: Percentile above which trades are flagged (default 95th)

        Returns:
            DataFrame of large trades with columns from trades_df plus
            [volume_percentile, volume_zscore]
        """
        try:
            if trades_df.empty or 'volume' not in trades_df.columns:
                logger.warning("No trade data or missing volume column for large trade detection")
                return pd.DataFrame()

            threshold = np.percentile(trades_df['volume'], threshold_percentile)
            large_mask = trades_df['volume'] >= threshold

            large_trades = trades_df[large_mask].copy()

            # Annotate
            vol_mean = trades_df['volume'].mean()
            vol_std = trades_df['volume'].std()
            large_trades['volume_percentile'] = large_trades['volume'].apply(
                lambda v: float(stats.percentileofscore(trades_df['volume'], v))
            )
            large_trades['volume_zscore'] = (
                (large_trades['volume'] - vol_mean) / vol_std if vol_std > 0 else 0.0
            )

            logger.info(
                f"Detected {len(large_trades)} large trades "
                f"(>= {threshold_percentile}th percentile, threshold={threshold:.0f})"
            )
            return large_trades

        except Exception as e:
            logger.error(f"Large trade detection failed: {e}")
            return pd.DataFrame()

    def analyze_trade_arrival_rate(self, trades_df: pd.DataFrame) -> Dict:
        """
        Analyze trade arrival intensity and detect unusual activity spikes.

        Groups trades into time windows and models the arrival rate.
        Flags windows where trade count exceeds mean + k * std.

        Args:
            trades_df: DataFrame with a DatetimeIndex and columns [price, volume]

        Returns:
            Dict with mean_rate, current_rate, spike_detected, spike_windows, stats
        """
        try:
            if trades_df.empty:
                return {'error': 'Empty trades DataFrame'}

            # Determine time-based or index-based binning
            if isinstance(trades_df.index, pd.DatetimeIndex):
                window_seconds = self.flow_config['arrival_rate_window']
                counts = trades_df.resample(f'{window_seconds}s').size()
                volumes = trades_df['volume'].resample(f'{window_seconds}s').sum()
            else:
                # Fall back to fixed-size windows
                window_size = max(self.flow_config['arrival_rate_window'], 1)
                n = len(trades_df)
                bucket_ids = np.arange(n) // window_size
                counts = pd.Series(bucket_ids).value_counts().sort_index()
                volumes = trades_df['volume'].groupby(bucket_ids).sum()

            mean_rate = float(counts.mean())
            std_rate = float(counts.std()) if len(counts) > 1 else 0.0
            current_rate = float(counts.iloc[-1]) if len(counts) > 0 else 0.0

            spike_threshold = mean_rate + self.flow_config['spike_std_threshold'] * std_rate
            spike_windows = counts[counts > spike_threshold] if std_rate > 0 else pd.Series(dtype=float)
            spike_detected = len(spike_windows) > 0

            result = {
                'mean_rate': mean_rate,
                'std_rate': std_rate,
                'current_rate': current_rate,
                'spike_threshold': float(spike_threshold),
                'spike_detected': spike_detected,
                'spike_window_count': len(spike_windows),
                'total_windows': len(counts),
                'total_trades': len(trades_df),
                'mean_volume_per_window': float(volumes.mean()) if len(volumes) > 0 else 0.0,
                'arrival_rate_percentile': float(
                    stats.percentileofscore(counts.values, current_rate)
                ) if len(counts) > 0 else 0.0,
                'timestamp': datetime.now().isoformat(),
            }

            if spike_detected:
                logger.warning(
                    f"Trade arrival spike detected: {len(spike_windows)} windows "
                    f"above threshold ({spike_threshold:.1f} trades/window)"
                )
            else:
                logger.info(
                    f"Trade arrival rate: {mean_rate:.1f} +/- {std_rate:.1f} per window, "
                    f"current={current_rate:.0f}"
                )

            return result

        except Exception as e:
            logger.error(f"Trade arrival rate analysis failed: {e}")
            return {'error': str(e)}

    # ─── Spread Analysis ─────────────────────────────────────────────────

    def calculate_effective_spread(self, trades_df: pd.DataFrame) -> pd.Series:
        """
        Calculate effective half-spread from trade and quote data.

        Effective spread = 2 * |trade_price - midpoint| / midpoint

        When bid/ask data is not available, estimates the midpoint as the
        rolling median of trade prices.

        Args:
            trades_df: DataFrame with columns [price, volume].
                       Optional: [bid, ask] for exact midpoint.

        Returns:
            Series of effective spread values per trade
        """
        try:
            if 'price' not in trades_df.columns:
                logger.error("Effective spread requires [price] column")
                return pd.Series(dtype=float)

            has_quotes = 'bid' in trades_df.columns and 'ask' in trades_df.columns

            if has_quotes:
                midpoint = (trades_df['bid'] + trades_df['ask']) / 2.0
            else:
                # Estimate midpoint from rolling median of prices
                window = min(21, len(trades_df))
                midpoint = trades_df['price'].rolling(window=max(window, 1), min_periods=1).median()
                logger.info("No bid/ask data — estimating midpoint from rolling median")

            effective_spread = 2.0 * (trades_df['price'] - midpoint).abs() / midpoint.replace(0, np.nan)
            effective_spread.name = 'effective_spread'

            logger.info(
                f"Effective spread: mean={effective_spread.mean():.6f}, "
                f"median={effective_spread.median():.6f}"
            )
            return effective_spread

        except Exception as e:
            logger.error(f"Effective spread calculation failed: {e}")
            return pd.Series(dtype=float)

    def estimate_roll_spread(self, returns: pd.Series) -> float:
        """
        Estimate the implied bid-ask spread using the Roll (1984) model.

        Roll spread = 2 * sqrt(-Cov(r_t, r_{t-1}))

        If serial covariance is positive (no spread signal), returns 0.0.

        Args:
            returns: Series of log or simple returns

        Returns:
            Estimated implied spread (float)
        """
        try:
            if len(returns) < 3:
                logger.warning("Roll spread requires at least 3 return observations")
                return 0.0

            returns_clean = returns.dropna()
            if len(returns_clean) < 3:
                return 0.0

            # Serial covariance: Cov(r_t, r_{t-1})
            r_t = returns_clean.iloc[1:].values
            r_t_minus_1 = returns_clean.iloc[:-1].values
            serial_cov = np.cov(r_t, r_t_minus_1, ddof=1)[0, 1]

            if serial_cov >= 0:
                # Positive serial covariance — no spread signal in the Roll model
                logger.info(
                    f"Roll spread: serial covariance is non-negative ({serial_cov:.8f}), "
                    f"returning 0.0"
                )
                return 0.0

            roll_spread = 2.0 * np.sqrt(-serial_cov)

            logger.info(
                f"Roll implied spread: {roll_spread:.6f} "
                f"(serial_cov={serial_cov:.8f})"
            )
            return float(roll_spread)

        except Exception as e:
            logger.error(f"Roll spread estimation failed: {e}")
            return 0.0
