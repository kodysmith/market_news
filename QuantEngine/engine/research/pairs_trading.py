"""
Statistical Arbitrage & Pairs Trading Engine

Implements quantitative pairs trading strategies including:
- Engle-Granger and Johansen cointegration testing
- Ornstein-Uhlenbeck mean reversion modeling
- Dynamic hedge ratio estimation (Rolling OLS, Kalman filter)
- Z-score based signal generation with configurable thresholds
- PCA-based statistical arbitrage basket construction

Uses statsmodels for cointegration, sklearn for PCA, numpy/scipy for
parameter estimation. Includes graceful fallbacks for optional dependencies.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
import logging
from datetime import datetime
from itertools import combinations

try:
    import statsmodels.api as sm
    from statsmodels.tsa.stattools import adfuller, coint
    HAS_STATSMODELS = True
except ImportError:
    HAS_STATSMODELS = False

try:
    from statsmodels.tsa.vector_ar.vecm import coint_johansen
    HAS_JOHANSEN = True
except ImportError:
    HAS_JOHANSEN = False

try:
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False

from scipy import stats as sp_stats
from scipy.optimize import minimize

logger = logging.getLogger(__name__)


class PairsTradingEngine:
    """
    Statistical arbitrage and pairs trading engine with cointegration
    analysis, mean reversion modeling, and signal generation.
    """

    def __init__(self, config: Dict[str, Any]):
        self.config = config

        # Cointegration testing parameters
        self.coint_config = {
            'p_threshold': config.get('p_threshold', 0.05),
            'min_observations': config.get('min_observations', 252),
            'max_half_life': config.get('max_half_life', 126),
            'min_half_life': config.get('min_half_life', 5),
        }

        # Signal generation parameters
        self.signal_config = {
            'entry_z': config.get('entry_z', 2.0),
            'exit_z': config.get('exit_z', 0.5),
            'stop_z': config.get('stop_z', 4.0),
            'lookback': config.get('lookback', 60),
        }

        # Hedge ratio estimation
        self.hedge_config = {
            'method': config.get('hedge_method', 'rolling_ols'),
            'window': config.get('hedge_window', 60),
            'kalman_delta': config.get('kalman_delta', 1e-4),
            'kalman_ve': config.get('kalman_ve', 1e-3),
        }

        # PCA basket construction
        self.pca_config = {
            'n_components': config.get('n_components', 3),
            'min_explained_variance': config.get('min_explained_variance', 0.5),
        }

        # Pair tracking
        self.active_pairs: List[Dict] = []
        self.pair_history: List[Dict] = []

    # ------------------------------------------------------------------ #
    #  1. Cointegration Testing
    # ------------------------------------------------------------------ #

    def test_cointegration_eg(
        self, series_a: pd.Series, series_b: pd.Series
    ) -> Dict[str, Any]:
        """
        Engle-Granger two-step cointegration test.

        Step 1: Regress series_a on series_b to obtain residuals.
        Step 2: Test residuals for stationarity via ADF.

        Args:
            series_a: Price series for asset A
            series_b: Price series for asset B

        Returns:
            Dictionary with test_stat, p_value, critical_values,
            is_cointegrated flag, hedge_ratio, and residual_adf details.
        """

        if not HAS_STATSMODELS:
            logger.error("statsmodels is required for Engle-Granger test")
            return {'error': 'statsmodels not available', 'is_cointegrated': False}

        try:
            # Align series
            aligned = pd.concat([series_a, series_b], axis=1).dropna()
            if len(aligned) < self.coint_config['min_observations']:
                logger.warning(
                    f"Insufficient observations: {len(aligned)} "
                    f"(need {self.coint_config['min_observations']})"
                )
                return {
                    'error': 'insufficient_observations',
                    'is_cointegrated': False,
                    'n_observations': len(aligned),
                }

            a_vals = aligned.iloc[:, 0].values
            b_vals = aligned.iloc[:, 1].values

            # Use statsmodels coint for Engle-Granger test
            test_stat, p_value, crit_values = coint(a_vals, b_vals)

            # Also compute hedge ratio via OLS regression
            b_with_const = sm.add_constant(b_vals)
            model = sm.OLS(a_vals, b_with_const).fit()
            hedge_ratio = model.params[1]
            intercept = model.params[0]
            residuals = model.resid

            # ADF on residuals for additional detail
            adf_result = adfuller(residuals, maxlag=None, autolag='AIC')

            is_cointegrated = p_value < self.coint_config['p_threshold']

            result = {
                'test_stat': float(test_stat),
                'p_value': float(p_value),
                'critical_values': {
                    '1%': float(crit_values[0]),
                    '5%': float(crit_values[1]),
                    '10%': float(crit_values[2]),
                },
                'is_cointegrated': is_cointegrated,
                'hedge_ratio': float(hedge_ratio),
                'intercept': float(intercept),
                'residual_adf': {
                    'test_stat': float(adf_result[0]),
                    'p_value': float(adf_result[1]),
                    'n_lags': int(adf_result[2]),
                    'n_observations': int(adf_result[3]),
                },
                'n_observations': len(aligned),
                'r_squared': float(model.rsquared),
                'timestamp': datetime.now().isoformat(),
            }

            logger.info(
                f"Engle-Granger test: stat={test_stat:.4f}, "
                f"p={p_value:.4f}, cointegrated={is_cointegrated}"
            )
            return result

        except Exception as e:
            logger.error(f"Engle-Granger test failed: {e}")
            return {'error': str(e), 'is_cointegrated': False}

    def test_cointegration_johansen(
        self, price_data: pd.DataFrame
    ) -> Dict[str, Any]:
        """
        Johansen cointegration test for multiple time series.

        Tests for the number of cointegrating relationships among
        all columns in the provided price DataFrame.

        Args:
            price_data: DataFrame with each column a price series

        Returns:
            Dictionary with trace_stats, max_eigen_stats, critical_values,
            cointegration_rank, and eigenvalues.
        """

        if not HAS_JOHANSEN:
            logger.error("statsmodels VECM module required for Johansen test")
            return {'error': 'johansen not available', 'cointegration_rank': 0}

        try:
            clean_data = price_data.dropna()
            if len(clean_data) < self.coint_config['min_observations']:
                logger.warning(
                    f"Insufficient observations for Johansen test: {len(clean_data)}"
                )
                return {
                    'error': 'insufficient_observations',
                    'cointegration_rank': 0,
                    'n_observations': len(clean_data),
                }

            # det_order=-1: no deterministic terms; k_ar_diff=1: VAR(1) in differences
            joh_result = coint_johansen(clean_data.values, det_order=0, k_ar_diff=1)

            n_series = price_data.shape[1]

            # Determine cointegration rank from trace test at 5% level
            trace_stats = joh_result.lr1  # trace statistics
            trace_crit = joh_result.cvt  # critical values (90%, 95%, 99%)
            max_eigen_stats = joh_result.lr2  # max eigenvalue statistics
            max_eigen_crit = joh_result.cvm

            # Count rank using 5% critical values (column index 1)
            rank = 0
            for i in range(n_series):
                if trace_stats[i] > trace_crit[i, 1]:
                    rank += 1
                else:
                    break

            result = {
                'trace_stats': trace_stats.tolist(),
                'trace_critical_values': {
                    '90%': trace_crit[:, 0].tolist(),
                    '95%': trace_crit[:, 1].tolist(),
                    '99%': trace_crit[:, 2].tolist(),
                },
                'max_eigen_stats': max_eigen_stats.tolist(),
                'max_eigen_critical_values': {
                    '90%': max_eigen_crit[:, 0].tolist(),
                    '95%': max_eigen_crit[:, 1].tolist(),
                    '99%': max_eigen_crit[:, 2].tolist(),
                },
                'cointegration_rank': int(rank),
                'eigenvalues': joh_result.eig.tolist(),
                'eigenvectors': joh_result.evec.tolist(),
                'n_series': n_series,
                'n_observations': len(clean_data),
                'timestamp': datetime.now().isoformat(),
            }

            logger.info(
                f"Johansen test: rank={rank}/{n_series}, "
                f"trace_stats={[f'{s:.2f}' for s in trace_stats]}"
            )
            return result

        except Exception as e:
            logger.error(f"Johansen test failed: {e}")
            return {'error': str(e), 'cointegration_rank': 0}

    def scan_pairs(
        self, price_data: pd.DataFrame, p_threshold: float = 0.05
    ) -> List[Dict[str, Any]]:
        """
        Scan all pairwise combinations in a price universe for cointegration.

        Ranks discovered pairs by statistical strength (p-value) and filters
        by half-life feasibility.

        Args:
            price_data: DataFrame with each column a different asset's prices
            p_threshold: Maximum p-value for cointegration acceptance

        Returns:
            List of cointegrated pair dictionaries sorted by p-value ascending.
        """

        if not HAS_STATSMODELS:
            logger.error("statsmodels required for pair scanning")
            return []

        try:
            columns = price_data.columns.tolist()
            n_assets = len(columns)
            n_pairs = n_assets * (n_assets - 1) // 2

            logger.info(
                f"Scanning {n_pairs} pairs across {n_assets} assets..."
            )

            cointegrated_pairs: List[Dict[str, Any]] = []

            for asset_a, asset_b in combinations(columns, 2):
                series_a = price_data[asset_a]
                series_b = price_data[asset_b]

                eg_result = self.test_cointegration_eg(series_a, series_b)

                if eg_result.get('is_cointegrated', False):
                    # Calculate half-life for the spread
                    spread = series_a - eg_result['hedge_ratio'] * series_b
                    half_life = self.calculate_half_life(spread)

                    # Filter by half-life feasibility
                    if (
                        self.coint_config['min_half_life']
                        <= half_life
                        <= self.coint_config['max_half_life']
                    ):
                        pair_info = {
                            'asset_a': asset_a,
                            'asset_b': asset_b,
                            'p_value': eg_result['p_value'],
                            'test_stat': eg_result['test_stat'],
                            'hedge_ratio': eg_result['hedge_ratio'],
                            'half_life': half_life,
                            'r_squared': eg_result.get('r_squared', 0.0),
                            'n_observations': eg_result.get('n_observations', 0),
                            'strength_score': self._calculate_pair_strength(
                                eg_result['p_value'], half_life
                            ),
                        }
                        cointegrated_pairs.append(pair_info)

            # Sort by p-value ascending (strongest cointegration first)
            cointegrated_pairs.sort(key=lambda x: x['p_value'])

            logger.info(
                f"Found {len(cointegrated_pairs)} cointegrated pairs "
                f"out of {n_pairs} scanned (threshold p<{p_threshold})"
            )
            return cointegrated_pairs

        except Exception as e:
            logger.error(f"Pair scanning failed: {e}")
            return []

    def _calculate_pair_strength(self, p_value: float, half_life: float) -> float:
        """
        Composite strength score combining statistical significance and
        mean reversion speed.
        """

        # Lower p-value -> higher score (max 50 points)
        p_score = max(0.0, (0.05 - p_value) / 0.05) * 50.0

        # Ideal half-life around 15-30 days (max 50 points)
        ideal_hl = 20.0
        hl_deviation = abs(half_life - ideal_hl) / ideal_hl
        hl_score = max(0.0, 1.0 - hl_deviation) * 50.0

        return p_score + hl_score

    # ------------------------------------------------------------------ #
    #  2. Mean Reversion Modeling
    # ------------------------------------------------------------------ #

    def fit_ornstein_uhlenbeck(self, spread: pd.Series) -> Dict[str, Any]:
        """
        Estimate Ornstein-Uhlenbeck process parameters for a spread series.

        The OU process: dX = theta * (mu - X) * dt + sigma * dW

        Parameters are estimated via maximum likelihood on the discretized
        process, with half-life derived as ln(2) / theta.

        Args:
            spread: Time series of the spread (e.g. residual from hedge)

        Returns:
            Dictionary with theta (speed of reversion), mu (long-run mean),
            sigma (volatility), half_life, and log_likelihood.
        """

        try:
            spread_clean = spread.dropna()
            if len(spread_clean) < 30:
                logger.warning("Insufficient data for OU estimation")
                return {'error': 'insufficient_data'}

            vals = spread_clean.values
            dt = 1.0  # daily frequency

            # Estimate via AR(1) regression: X_t = a + b * X_{t-1} + eps
            x_lag = vals[:-1]
            x_now = vals[1:]

            x_lag_const = sm.add_constant(x_lag) if HAS_STATSMODELS else np.column_stack([np.ones(len(x_lag)), x_lag])

            if HAS_STATSMODELS:
                ar_model = sm.OLS(x_now, x_lag_const).fit()
                a_hat = ar_model.params[0]
                b_hat = ar_model.params[1]
                residuals = ar_model.resid
            else:
                # Fallback: numpy least squares
                coeffs, _, _, _ = np.linalg.lstsq(x_lag_const, x_now, rcond=None)
                a_hat = coeffs[0]
                b_hat = coeffs[1]
                residuals = x_now - x_lag_const @ coeffs

            # Convert AR(1) params to continuous OU params
            if b_hat >= 1.0:
                logger.warning("Spread is not mean-reverting (b >= 1)")
                return {
                    'theta': 0.0,
                    'mu': float(np.mean(vals)),
                    'sigma': float(np.std(np.diff(vals))),
                    'half_life': float('inf'),
                    'is_mean_reverting': False,
                    'log_likelihood': float('nan'),
                }

            theta = -np.log(b_hat) / dt
            mu = a_hat / (1.0 - b_hat)
            sigma_eq = np.std(residuals)
            sigma = sigma_eq * np.sqrt(2.0 * theta / (1.0 - b_hat ** 2)) if theta > 0 else sigma_eq

            # Half-life of mean reversion
            half_life = np.log(2.0) / theta if theta > 0 else float('inf')

            # Log-likelihood (Gaussian assumption)
            n = len(residuals)
            ll = -0.5 * n * np.log(2.0 * np.pi) - n * np.log(sigma_eq) - 0.5 * np.sum(residuals ** 2) / (sigma_eq ** 2)

            result = {
                'theta': float(theta),
                'mu': float(mu),
                'sigma': float(sigma),
                'half_life': float(half_life),
                'is_mean_reverting': theta > 0 and half_life < self.coint_config['max_half_life'],
                'ar1_coefficient': float(b_hat),
                'log_likelihood': float(ll),
                'n_observations': len(spread_clean),
                'timestamp': datetime.now().isoformat(),
            }

            logger.info(
                f"OU fit: theta={theta:.4f}, mu={mu:.4f}, "
                f"sigma={sigma:.4f}, half_life={half_life:.1f}d"
            )
            return result

        except Exception as e:
            logger.error(f"OU parameter estimation failed: {e}")
            return {'error': str(e)}

    def calculate_half_life(self, spread: pd.Series) -> float:
        """
        Calculate the half-life of mean reversion via AR(1) regression.

        Regresses spread changes on lagged spread levels:
            delta_spread_t = alpha + beta * spread_{t-1} + eps

        Half-life = -ln(2) / beta  (beta should be negative for mean reversion)

        Args:
            spread: Time series of the spread

        Returns:
            Half-life in periods (days). Returns inf if not mean-reverting.
        """

        try:
            spread_clean = spread.dropna()
            if len(spread_clean) < 20:
                return float('inf')

            vals = spread_clean.values
            delta = np.diff(vals)
            lagged = vals[:-1]

            # Regress delta on lagged level
            lagged_const = np.column_stack([np.ones(len(lagged)), lagged])
            coeffs, _, _, _ = np.linalg.lstsq(lagged_const, delta, rcond=None)
            beta = coeffs[1]

            if beta >= 0:
                return float('inf')

            half_life = -np.log(2.0) / beta
            return float(half_life)

        except Exception as e:
            logger.error(f"Half-life calculation failed: {e}")
            return float('inf')

    def calculate_spread_zscore(
        self,
        series_a: pd.Series,
        series_b: pd.Series,
        lookback: int = 60,
    ) -> pd.DataFrame:
        """
        Calculate the z-score of the spread between two series using a
        dynamic hedge ratio estimated via rolling OLS.

        Args:
            series_a: Price series for asset A
            series_b: Price series for asset B
            lookback: Rolling window for OLS and z-score normalization

        Returns:
            DataFrame with columns: hedge_ratio, spread, spread_mean,
            spread_std, zscore.
        """

        try:
            aligned = pd.concat(
                [series_a.rename('a'), series_b.rename('b')], axis=1
            ).dropna()

            if len(aligned) < lookback + 10:
                logger.warning("Insufficient data for spread z-score calculation")
                return pd.DataFrame()

            hedge_ratios = []
            for i in range(lookback, len(aligned)):
                window_a = aligned['a'].iloc[i - lookback:i].values
                window_b = aligned['b'].iloc[i - lookback:i].values
                b_const = np.column_stack([np.ones(len(window_b)), window_b])
                coeffs, _, _, _ = np.linalg.lstsq(b_const, window_a, rcond=None)
                hedge_ratios.append(coeffs[1])

            result_index = aligned.index[lookback:]
            hr_series = pd.Series(hedge_ratios, index=result_index, name='hedge_ratio')

            # Compute spread
            spread = (
                aligned['a'].loc[result_index]
                - hr_series * aligned['b'].loc[result_index]
            )

            # Rolling mean and std of spread for z-score
            spread_mean = spread.rolling(window=lookback, min_periods=lookback // 2).mean()
            spread_std = spread.rolling(window=lookback, min_periods=lookback // 2).std()

            zscore = (spread - spread_mean) / spread_std.replace(0, np.nan)

            result = pd.DataFrame({
                'hedge_ratio': hr_series,
                'spread': spread,
                'spread_mean': spread_mean,
                'spread_std': spread_std,
                'zscore': zscore,
            })

            logger.info(
                f"Spread z-score computed: latest z={zscore.iloc[-1]:.2f}, "
                f"hedge_ratio={hr_series.iloc[-1]:.4f}"
            )
            return result

        except Exception as e:
            logger.error(f"Spread z-score calculation failed: {e}")
            return pd.DataFrame()

    # ------------------------------------------------------------------ #
    #  3. Signal Generation
    # ------------------------------------------------------------------ #

    def generate_pairs_signals(
        self,
        series_a: pd.Series,
        series_b: pd.Series,
        entry_z: float = 2.0,
        exit_z: float = 0.5,
    ) -> pd.DataFrame:
        """
        Generate entry and exit signals based on spread z-score thresholds.

        Long spread (buy A, sell B) when z-score < -entry_z.
        Short spread (sell A, buy B) when z-score > +entry_z.
        Exit when |z-score| < exit_z.

        Args:
            series_a: Price series for asset A
            series_b: Price series for asset B
            entry_z: Z-score threshold for entry (absolute value)
            exit_z: Z-score threshold for exit (absolute value)

        Returns:
            DataFrame with columns: zscore, signal, position, hedge_ratio.
            signal: 1 = long spread, -1 = short spread, 0 = no signal.
            position: current position state carried forward.
        """

        try:
            lookback = self.signal_config['lookback']
            stop_z = self.signal_config['stop_z']

            spread_data = self.calculate_spread_zscore(series_a, series_b, lookback)
            if spread_data.empty:
                return pd.DataFrame()

            zscore = spread_data['zscore']
            signals = pd.Series(0, index=zscore.index, name='signal')
            positions = pd.Series(0, index=zscore.index, name='position')

            current_position = 0

            for i in range(len(zscore)):
                z = zscore.iloc[i]

                if pd.isna(z):
                    positions.iloc[i] = current_position
                    continue

                # Stop-loss
                if abs(z) > stop_z and current_position != 0:
                    signals.iloc[i] = 0
                    current_position = 0
                # Entry signals
                elif z < -entry_z and current_position <= 0:
                    signals.iloc[i] = 1  # long spread
                    current_position = 1
                elif z > entry_z and current_position >= 0:
                    signals.iloc[i] = -1  # short spread
                    current_position = -1
                # Exit signals
                elif abs(z) < exit_z and current_position != 0:
                    signals.iloc[i] = 0
                    current_position = 0

                positions.iloc[i] = current_position

            result = pd.DataFrame({
                'zscore': zscore,
                'hedge_ratio': spread_data['hedge_ratio'],
                'spread': spread_data['spread'],
                'signal': signals,
                'position': positions,
            })

            n_long = (positions == 1).sum()
            n_short = (positions == -1).sum()
            n_flat = (positions == 0).sum()

            logger.info(
                f"Pairs signals generated: long={n_long}, short={n_short}, "
                f"flat={n_flat} periods"
            )
            return result

        except Exception as e:
            logger.error(f"Signal generation failed: {e}")
            return pd.DataFrame()

    def calculate_dynamic_hedge_ratio(
        self,
        series_a: pd.Series,
        series_b: pd.Series,
        method: str = 'rolling_ols',
        window: int = 60,
    ) -> pd.Series:
        """
        Calculate a time-varying hedge ratio between two price series.

        Supports rolling OLS and Kalman filter estimation methods.

        Args:
            series_a: Price series for asset A
            series_b: Price series for asset B
            method: Estimation method - 'rolling_ols' or 'kalman'
            window: Rolling window size (used by rolling_ols)

        Returns:
            Series of dynamic hedge ratios indexed to the input series.
        """

        try:
            aligned = pd.concat(
                [series_a.rename('a'), series_b.rename('b')], axis=1
            ).dropna()

            if method == 'kalman':
                return self._kalman_hedge_ratio(aligned['a'], aligned['b'])
            else:
                return self._rolling_ols_hedge_ratio(
                    aligned['a'], aligned['b'], window
                )

        except Exception as e:
            logger.error(f"Dynamic hedge ratio calculation failed: {e}")
            return pd.Series(dtype=float)

    def _rolling_ols_hedge_ratio(
        self, a: pd.Series, b: pd.Series, window: int
    ) -> pd.Series:
        """Estimate hedge ratio via rolling OLS regression."""

        hedge_ratios = pd.Series(np.nan, index=a.index, name='hedge_ratio')

        for i in range(window, len(a)):
            window_a = a.iloc[i - window:i].values
            window_b = b.iloc[i - window:i].values
            b_const = np.column_stack([np.ones(len(window_b)), window_b])
            coeffs, _, _, _ = np.linalg.lstsq(b_const, window_a, rcond=None)
            hedge_ratios.iloc[i] = coeffs[1]

        logger.info(
            f"Rolling OLS hedge ratio: window={window}, "
            f"latest={hedge_ratios.iloc[-1]:.4f}"
        )
        return hedge_ratios

    def _kalman_hedge_ratio(self, a: pd.Series, b: pd.Series) -> pd.Series:
        """
        Estimate time-varying hedge ratio using a Kalman filter.

        State: beta (hedge ratio)
        Observation: a_t = beta_t * b_t + eps_t
        Transition: beta_t = beta_{t-1} + eta_t
        """

        delta = self.hedge_config['kalman_delta']
        ve = self.hedge_config['kalman_ve']

        n = len(a)
        hedge_ratios = np.zeros(n)

        # Initial state
        beta = 0.0
        P = 1.0  # state covariance

        for t in range(n):
            x_t = b.iloc[t]
            y_t = a.iloc[t]

            # Prediction
            beta_pred = beta
            P_pred = P + delta

            # Update
            y_hat = beta_pred * x_t
            innovation = y_t - y_hat
            S = x_t * P_pred * x_t + ve  # innovation covariance
            K = P_pred * x_t / S  # Kalman gain

            beta = beta_pred + K * innovation
            P = (1.0 - K * x_t) * P_pred

            hedge_ratios[t] = beta

        result = pd.Series(hedge_ratios, index=a.index, name='hedge_ratio')

        logger.info(
            f"Kalman hedge ratio: delta={delta}, ve={ve}, "
            f"latest={result.iloc[-1]:.4f}"
        )
        return result

    # ------------------------------------------------------------------ #
    #  4. PCA-Based Basket Construction
    # ------------------------------------------------------------------ #

    def construct_stat_arb_basket(
        self, price_data: pd.DataFrame, n_components: int = 3
    ) -> Dict[str, Any]:
        """
        Construct statistical arbitrage baskets via PCA decomposition.

        Decomposes returns into principal components, identifies
        mean-reverting residual portfolios, and returns tradeable
        basket weights with stationarity statistics.

        Args:
            price_data: DataFrame with each column a different asset's prices
            n_components: Number of principal components to extract

        Returns:
            Dictionary with components, explained_variance, basket_weights,
            stationarity_tests, and mean_reversion_stats for each basket.
        """

        if not HAS_SKLEARN:
            logger.error("sklearn is required for PCA basket construction")
            return {'error': 'sklearn not available'}

        try:
            clean_data = price_data.dropna()
            if len(clean_data) < self.coint_config['min_observations']:
                logger.warning("Insufficient data for PCA basket construction")
                return {
                    'error': 'insufficient_data',
                    'n_observations': len(clean_data),
                }

            # Compute log returns
            log_returns = np.log(clean_data / clean_data.shift(1)).dropna()

            # Standardize returns
            scaler = StandardScaler()
            returns_scaled = scaler.fit_transform(log_returns)

            # PCA decomposition
            n_components = min(n_components, len(price_data.columns))
            pca = PCA(n_components=n_components)
            pca.fit(returns_scaled)

            # Residual returns (what PCA doesn't explain)
            factors = pca.transform(returns_scaled)
            reconstructed = pca.inverse_transform(factors)
            residual_returns = returns_scaled - reconstructed

            # Construct baskets from residual components
            baskets = []
            for i in range(residual_returns.shape[1]):
                residual_series = pd.Series(
                    residual_returns[:, i],
                    index=log_returns.index,
                    name=f'residual_{price_data.columns[i]}',
                )

                # Cumulative residual as spread proxy
                cum_residual = residual_series.cumsum()

                # Test stationarity
                half_life = self.calculate_half_life(cum_residual)

                adf_result = None
                if HAS_STATSMODELS:
                    try:
                        adf_out = adfuller(cum_residual.dropna(), autolag='AIC')
                        adf_result = {
                            'test_stat': float(adf_out[0]),
                            'p_value': float(adf_out[1]),
                            'is_stationary': adf_out[1] < 0.05,
                        }
                    except Exception:
                        pass

                basket_info = {
                    'asset': str(price_data.columns[i]),
                    'weight': float(pca.components_[0][i]) if n_components > 0 else 0.0,
                    'half_life': half_life,
                    'adf_test': adf_result,
                    'residual_std': float(residual_series.std()),
                    'is_mean_reverting': (
                        half_life < self.coint_config['max_half_life']
                        and half_life > self.coint_config['min_half_life']
                    ),
                }
                baskets.append(basket_info)

            # Build tradeable basket weights from PCA loadings
            basket_weights = {}
            for comp_idx in range(n_components):
                loadings = pca.components_[comp_idx]
                weights = dict(zip(price_data.columns, loadings.tolist()))
                basket_weights[f'basket_{comp_idx + 1}'] = weights

            # Identify most mean-reverting baskets
            mean_reverting_baskets = [
                b for b in baskets if b['is_mean_reverting']
            ]
            mean_reverting_baskets.sort(key=lambda x: x['half_life'])

            result = {
                'n_components': n_components,
                'explained_variance_ratio': pca.explained_variance_ratio_.tolist(),
                'total_explained_variance': float(
                    pca.explained_variance_ratio_.sum()
                ),
                'basket_weights': basket_weights,
                'asset_analysis': baskets,
                'mean_reverting_baskets': mean_reverting_baskets,
                'n_mean_reverting': len(mean_reverting_baskets),
                'n_assets': len(price_data.columns),
                'n_observations': len(clean_data),
                'timestamp': datetime.now().isoformat(),
            }

            logger.info(
                f"PCA baskets: {n_components} components explain "
                f"{result['total_explained_variance']:.1%} variance, "
                f"{len(mean_reverting_baskets)} mean-reverting baskets found"
            )
            return result

        except Exception as e:
            logger.error(f"PCA basket construction failed: {e}")
            return {'error': str(e)}
