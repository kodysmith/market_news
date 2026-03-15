"""
Causal and Structural Econometric Models

Implements causal inference and structural modeling for quantitative analysis:
- Granger causality testing and pairwise causality matrices
- Vector Autoregression (VAR) with impulse response and variance decomposition
- Lead-lag detection via cross-correlation and transfer entropy
- Structural VAR (SVAR) and Vector Error Correction Models (VECM)
- Macro-to-equity transmission channel estimation

Uses statsmodels for VAR/Granger, scipy for statistical tests, numpy for numerics.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
import logging
from scipy import stats
from statsmodels.tsa.api import VAR
from statsmodels.tsa.stattools import grangercausalitytests, adfuller, coint
from statsmodels.tsa.vector_ar.vecm import VECM, coint_johansen

logger = logging.getLogger(__name__)


class CausalModelAnalyzer:
    """
    Causal and structural econometric analysis for quantitative trading systems
    """

    def __init__(self, config: Dict[str, Any]):
        self.config = config

        # Significance level for hypothesis tests
        self.significance_level = config.get('significance_level', 0.05)

        # Default maximum lags for causality and VAR tests
        self.default_max_lags = config.get('default_max_lags', 10)

        # Minimum observations required for reliable estimation
        self.min_observations = config.get('min_observations', 50)

        # Transfer entropy estimation parameters
        self.te_bins = config.get('transfer_entropy_bins', 10)
        self.te_history_length = config.get('transfer_entropy_history', 1)

        # Information criterion for lag selection
        self.info_criterion = config.get('info_criterion', 'bic')

    # ------------------------------------------------------------------ #
    #  1. Granger Causality
    # ------------------------------------------------------------------ #

    def test_granger_causality(
        self,
        x: pd.Series,
        y: pd.Series,
        max_lags: int = 10,
    ) -> Dict[str, Any]:
        """
        Test whether *x* Granger-causes *y*.

        Returns F-statistics and p-values per lag, the optimal lag chosen by
        BIC, and a human-readable conclusion.
        """
        x = x.dropna()
        y = y.dropna()

        # Align indices
        common_idx = x.index.intersection(y.index)
        if len(common_idx) < self.min_observations:
            logger.warning(
                "Insufficient observations (%d) for Granger test; need %d.",
                len(common_idx),
                self.min_observations,
            )
            return {
                'f_stats': {},
                'p_values': {},
                'optimal_lag': None,
                'conclusion': 'insufficient_data',
                'significant': False,
            }

        x = x.loc[common_idx].values
        y = y.loc[common_idx].values

        # statsmodels expects a 2-column array [y, x] (y in col-0)
        data = np.column_stack([y, x])

        # Cap max_lags to avoid over-fitting with small samples
        max_lags = min(max_lags, len(common_idx) // 3)
        if max_lags < 1:
            max_lags = 1

        try:
            results = grangercausalitytests(data, maxlag=max_lags, verbose=False)
        except Exception as e:
            logger.error("Granger causality test failed: %s", e)
            return {
                'f_stats': {},
                'p_values': {},
                'optimal_lag': None,
                'conclusion': f'error: {e}',
                'significant': False,
            }

        f_stats: Dict[int, float] = {}
        p_values: Dict[int, float] = {}

        for lag in range(1, max_lags + 1):
            if lag not in results:
                continue
            test_result = results[lag]
            # test_result is a tuple (dict_of_tests, [restricted_ols, unrestricted_ols])
            f_test = test_result[0]['ssr_ftest']
            f_stats[lag] = float(f_test[0])
            p_values[lag] = float(f_test[1])

        # Optimal lag via BIC: use the VAR framework for BIC-based selection
        optimal_lag = self._select_optimal_lag_bic(
            np.column_stack([y, x]), max_lags
        )

        # Determine significance at the optimal lag
        sig = False
        conclusion = 'no_causality'
        if optimal_lag is not None and optimal_lag in p_values:
            if p_values[optimal_lag] < self.significance_level:
                sig = True
                conclusion = (
                    f'x Granger-causes y at lag {optimal_lag} '
                    f'(p={p_values[optimal_lag]:.4f})'
                )
            else:
                conclusion = (
                    f'x does NOT Granger-cause y at lag {optimal_lag} '
                    f'(p={p_values[optimal_lag]:.4f})'
                )

        return {
            'f_stats': f_stats,
            'p_values': p_values,
            'optimal_lag': optimal_lag,
            'conclusion': conclusion,
            'significant': sig,
        }

    def pairwise_granger_matrix(
        self,
        data: pd.DataFrame,
        max_lags: int = 5,
    ) -> pd.DataFrame:
        """
        Compute an NxN matrix of Granger causality p-values.

        Entry (i, j) is the minimum p-value across tested lags for the null
        hypothesis that column *j* does NOT Granger-cause column *i*.
        """
        cols = data.columns.tolist()
        n = len(cols)
        matrix = pd.DataFrame(np.ones((n, n)), index=cols, columns=cols)

        for i, target in enumerate(cols):
            for j, cause in enumerate(cols):
                if i == j:
                    matrix.iloc[i, j] = np.nan
                    continue
                result = self.test_granger_causality(
                    data[cause], data[target], max_lags=max_lags
                )
                pvals = result.get('p_values', {})
                if pvals:
                    matrix.iloc[i, j] = min(pvals.values())
                else:
                    matrix.iloc[i, j] = np.nan

        return matrix

    def find_lead_lag_relationships(
        self,
        data: pd.DataFrame,
        max_lags: int = 10,
    ) -> List[Dict[str, Any]]:
        """
        Identify statistically significant lead-lag (Granger) pairs.

        Returns a list of dicts sorted by p-value (strongest first).
        """
        relationships: List[Dict[str, Any]] = []
        cols = data.columns.tolist()

        for cause in cols:
            for target in cols:
                if cause == target:
                    continue
                result = self.test_granger_causality(
                    data[cause], data[target], max_lags=max_lags
                )
                if not result['significant']:
                    continue

                optimal_lag = result['optimal_lag']
                p_val = result['p_values'].get(optimal_lag, 1.0)
                f_stat = result['f_stats'].get(optimal_lag, 0.0)

                relationships.append({
                    'cause': cause,
                    'target': target,
                    'optimal_lag': optimal_lag,
                    'p_value': p_val,
                    'f_stat': f_stat,
                    'conclusion': result['conclusion'],
                })

        # Sort by p-value ascending (strongest evidence first)
        relationships.sort(key=lambda r: r['p_value'])
        return relationships

    # ------------------------------------------------------------------ #
    #  2. Vector Autoregression (VAR)
    # ------------------------------------------------------------------ #

    def fit_var(
        self,
        data: pd.DataFrame,
        max_lags: int = 10,
    ) -> Dict[str, Any]:
        """
        Fit a VAR model with automatic lag selection via AIC/BIC.

        Returns coefficients, residuals, selected lag order, and the fitted
        statsmodels VARResults object.
        """
        data = data.dropna()
        if len(data) < self.min_observations:
            logger.warning("Insufficient data for VAR (%d rows).", len(data))
            return {'error': 'insufficient_data'}

        max_lags = min(max_lags, len(data) // (2 * data.shape[1]))
        if max_lags < 1:
            max_lags = 1

        try:
            model = VAR(data)
            result = model.select_order(maxlags=max_lags)
            selected_lag = getattr(result, self.info_criterion, None)
            if selected_lag is None or selected_lag < 1:
                selected_lag = 1

            fitted = model.fit(maxlags=selected_lag)

            coef_dict: Dict[str, Any] = {}
            for i, col in enumerate(data.columns):
                coef_dict[col] = {
                    'params': fitted.params[:, i].tolist(),
                    'stderr': fitted.stderr[:, i].tolist(),
                    'pvalues': fitted.pvalues[:, i].tolist(),
                }

            return {
                'lag_order': int(fitted.k_ar),
                'coefficients': coef_dict,
                'residuals': pd.DataFrame(
                    fitted.resid, columns=data.columns
                ),
                'aic': float(fitted.aic),
                'bic': float(fitted.bic),
                'fitted_model': fitted,
                'column_names': data.columns.tolist(),
            }

        except Exception as e:
            logger.error("VAR fitting failed: %s", e)
            return {'error': str(e)}

    def var_forecast(
        self,
        var_result: Dict[str, Any],
        data: pd.DataFrame,
        horizon: int = 10,
    ) -> pd.DataFrame:
        """
        Generate multi-step-ahead forecasts with 95% confidence intervals.

        Returns a DataFrame with columns <var>_forecast, <var>_lower, <var>_upper.
        """
        fitted = var_result.get('fitted_model')
        if fitted is None:
            logger.error("No fitted VAR model provided.")
            return pd.DataFrame()

        lag_order = var_result['lag_order']
        data = data.dropna()
        last_obs = data.values[-lag_order:]

        try:
            fc = fitted.forecast(y=last_obs, steps=horizon)
            # Forecast intervals via simulation
            fc_interval = fitted.forecast_interval(
                y=last_obs, steps=horizon, alpha=0.05
            )
            point = fc_interval[0]
            lower = fc_interval[1]
            upper = fc_interval[2]

            cols = var_result.get('column_names', data.columns.tolist())
            out: Dict[str, np.ndarray] = {}
            for i, col in enumerate(cols):
                out[f'{col}_forecast'] = point[:, i]
                out[f'{col}_lower'] = lower[:, i]
                out[f'{col}_upper'] = upper[:, i]

            return pd.DataFrame(out, index=range(1, horizon + 1))

        except Exception as e:
            logger.error("VAR forecast failed: %s", e)
            return pd.DataFrame()

    def impulse_response(
        self,
        var_result: Dict[str, Any],
        periods: int = 20,
        orthogonalized: bool = True,
    ) -> Dict[str, Any]:
        """
        Compute impulse response functions from a fitted VAR.

        Returns IRF arrays and cumulative IRFs keyed by (impulse, response).
        """
        fitted = var_result.get('fitted_model')
        if fitted is None:
            logger.error("No fitted VAR model provided.")
            return {'error': 'no_model'}

        try:
            irf = fitted.irf(periods=periods)
            cols = var_result.get('column_names', [])
            n = len(cols)

            irf_dict: Dict[str, Any] = {}
            cum_dict: Dict[str, Any] = {}

            if orthogonalized:
                irf_data = irf.orth_irfs  # (periods+1, n, n)
            else:
                irf_data = irf.irfs

            cum_data = irf.orth_cum_effects if orthogonalized else irf.cum_effects

            for i, impulse in enumerate(cols):
                for j, response in enumerate(cols):
                    key = f'{impulse} -> {response}'
                    irf_dict[key] = irf_data[:, j, i].tolist()
                    cum_dict[key] = cum_data[:, j, i].tolist()

            return {
                'irf': irf_dict,
                'cumulative_irf': cum_dict,
                'periods': periods,
                'orthogonalized': orthogonalized,
            }

        except Exception as e:
            logger.error("Impulse response computation failed: %s", e)
            return {'error': str(e)}

    def variance_decomposition(
        self,
        var_result: Dict[str, Any],
        periods: int = 20,
    ) -> Dict[str, Any]:
        """
        Forecast error variance decomposition (FEVD).

        Shows how much of each variable's forecast-error variance is explained
        by shocks to every variable in the system.
        """
        fitted = var_result.get('fitted_model')
        if fitted is None:
            logger.error("No fitted VAR model provided.")
            return {'error': 'no_model'}

        try:
            fevd = fitted.fevd(periods=periods)
            cols = var_result.get('column_names', [])

            decomposition: Dict[str, Any] = {}
            # fevd.decomp is (n_vars, periods, n_vars)
            for i, var_name in enumerate(cols):
                decomposition[var_name] = pd.DataFrame(
                    fevd.decomp[i],
                    columns=cols,
                    index=range(1, periods + 1),
                ).to_dict()

            return {
                'decomposition': decomposition,
                'periods': periods,
                'column_names': cols,
            }

        except Exception as e:
            logger.error("Variance decomposition failed: %s", e)
            return {'error': str(e)}

    # ------------------------------------------------------------------ #
    #  3. Lead-Lag Detection
    # ------------------------------------------------------------------ #

    def cross_correlation_analysis(
        self,
        x: pd.Series,
        y: pd.Series,
        max_lag: int = 20,
    ) -> Dict[str, Any]:
        """
        Compute cross-correlation between *x* and *y* at multiple lags.

        Positive optimal lag means *x* leads *y*; negative means *y* leads *x*.
        """
        x = x.dropna()
        y = y.dropna()
        common_idx = x.index.intersection(y.index)

        if len(common_idx) < self.min_observations:
            logger.warning("Insufficient data for cross-correlation analysis.")
            return {'error': 'insufficient_data'}

        x_vals = x.loc[common_idx].values.astype(float)
        y_vals = y.loc[common_idx].values.astype(float)

        # Standardise
        x_vals = (x_vals - x_vals.mean()) / (x_vals.std() + 1e-12)
        y_vals = (y_vals - y_vals.mean()) / (y_vals.std() + 1e-12)

        n = len(x_vals)
        lags = range(-max_lag, max_lag + 1)
        correlations: Dict[int, float] = {}

        for lag in lags:
            if lag >= 0:
                corr = np.corrcoef(x_vals[:n - lag], y_vals[lag:])[0, 1]
            else:
                corr = np.corrcoef(x_vals[-lag:], y_vals[:n + lag])[0, 1]
            correlations[lag] = float(corr) if np.isfinite(corr) else 0.0

        # Find optimal lag (highest absolute correlation)
        optimal_lag = max(correlations, key=lambda k: abs(correlations[k]))
        peak_corr = correlations[optimal_lag]

        # Approximate significance threshold (Bartlett)
        threshold = 1.96 / np.sqrt(n)

        return {
            'correlations': correlations,
            'optimal_lag': int(optimal_lag),
            'peak_correlation': float(peak_corr),
            'significance_threshold': float(threshold),
            'significant': bool(abs(peak_corr) > threshold),
            'interpretation': (
                f'x leads y by {optimal_lag} periods'
                if optimal_lag > 0
                else (
                    f'y leads x by {-optimal_lag} periods'
                    if optimal_lag < 0
                    else 'contemporaneous relationship'
                )
            ),
        }

    def delaney_lead_lag(
        self,
        returns_df: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Systematic lead-lag detection across an asset universe.

        Returns an NxN DataFrame where entry (i, j) is the optimal lag at which
        asset *j* leads asset *i* (positive value means *j* leads).
        """
        cols = returns_df.columns.tolist()
        n = len(cols)
        lag_matrix = pd.DataFrame(
            np.zeros((n, n)), index=cols, columns=cols, dtype=int
        )
        corr_matrix = pd.DataFrame(
            np.zeros((n, n)), index=cols, columns=cols, dtype=float
        )

        max_lag = self.config.get('lead_lag_max_lag', 10)

        for i, col_i in enumerate(cols):
            for j, col_j in enumerate(cols):
                if i == j:
                    continue
                result = self.cross_correlation_analysis(
                    returns_df[col_j], returns_df[col_i], max_lag=max_lag
                )
                if 'error' in result:
                    continue
                lag_matrix.iloc[i, j] = result['optimal_lag']
                corr_matrix.iloc[i, j] = result['peak_correlation']

        # Attach correlation strengths as an attribute for downstream use
        lag_matrix.attrs['correlation_strength'] = corr_matrix
        return lag_matrix

    def information_flow(
        self,
        x: pd.Series,
        y: pd.Series,
    ) -> Dict[str, Any]:
        """
        Estimate transfer entropy from *x* to *y* (directional information flow).

        Uses a binned estimator: TE(X -> Y) = H(Y_t | Y_{t-1}) - H(Y_t | Y_{t-1}, X_{t-1}).
        """
        x = x.dropna()
        y = y.dropna()
        common_idx = x.index.intersection(y.index)

        if len(common_idx) < self.min_observations:
            logger.warning("Insufficient data for transfer entropy estimation.")
            return {'error': 'insufficient_data'}

        x_vals = x.loc[common_idx].values.astype(float)
        y_vals = y.loc[common_idx].values.astype(float)

        k = self.te_history_length
        n_bins = self.te_bins

        # Discretise into equiprobable bins
        x_binned = pd.qcut(x_vals, q=n_bins, labels=False, duplicates='drop')
        y_binned = pd.qcut(y_vals, q=n_bins, labels=False, duplicates='drop')

        n = len(x_binned)
        if n <= k + 1:
            return {'error': 'insufficient_data_after_binning'}

        # Build lagged arrays
        y_t = y_binned[k:]
        y_past = y_binned[k - 1 : -1] if k == 1 else np.column_stack(
            [y_binned[k - j - 1 : n - j - 1] for j in range(k)]
        )
        x_past = x_binned[k - 1 : -1] if k == 1 else np.column_stack(
            [x_binned[k - j - 1 : n - j - 1] for j in range(k)]
        )

        te_xy = self._compute_transfer_entropy(y_t, y_past, x_past)

        # Reverse direction for comparison
        x_t = x_binned[k:]
        te_yx = self._compute_transfer_entropy(x_t, x_past if k == 1 else np.column_stack(
            [x_binned[k - j - 1 : n - j - 1] for j in range(k)]
        ), y_past)

        net_flow = te_xy - te_yx

        return {
            'transfer_entropy_x_to_y': float(te_xy),
            'transfer_entropy_y_to_x': float(te_yx),
            'net_information_flow': float(net_flow),
            'dominant_direction': 'x -> y' if net_flow > 0 else 'y -> x',
            'bins_used': int(n_bins),
            'history_length': int(k),
        }

    # ------------------------------------------------------------------ #
    #  4. Structural Models
    # ------------------------------------------------------------------ #

    def estimate_macro_transmission(
        self,
        macro_data: pd.DataFrame,
        equity_returns: pd.Series,
    ) -> Dict[str, Any]:
        """
        Estimate how macro variables (rates, inflation, growth, etc.) transmit
        to equity returns, including the lag structure.

        Fits a VAR of [macro_vars, equity_returns] and extracts IRFs.
        """
        equity_returns = equity_returns.dropna()
        macro_data = macro_data.dropna()
        common_idx = macro_data.index.intersection(equity_returns.index)

        if len(common_idx) < self.min_observations:
            logger.warning("Insufficient data for macro transmission estimation.")
            return {'error': 'insufficient_data'}

        combined = macro_data.loc[common_idx].copy()
        combined['equity_returns'] = equity_returns.loc[common_idx]

        # Fit VAR
        var_result = self.fit_var(combined, max_lags=self.default_max_lags)
        if 'error' in var_result:
            return var_result

        # Impulse responses: shock each macro variable, observe equity response
        irf_result = self.impulse_response(var_result, periods=20, orthogonalized=True)
        if 'error' in irf_result:
            return irf_result

        # Extract equity-specific responses
        transmission: Dict[str, Any] = {}
        macro_cols = macro_data.columns.tolist()
        for macro_var in macro_cols:
            key = f'{macro_var} -> equity_returns'
            irf_values = irf_result['irf'].get(key, [])
            cum_values = irf_result['cumulative_irf'].get(key, [])

            if irf_values:
                peak_idx = int(np.argmax(np.abs(irf_values)))
                transmission[macro_var] = {
                    'irf': irf_values,
                    'cumulative_irf': cum_values,
                    'peak_impact_lag': peak_idx,
                    'peak_impact_value': float(irf_values[peak_idx]),
                    'total_cumulative_impact': float(cum_values[-1]) if cum_values else 0.0,
                }

        # Variance decomposition for equity returns
        fevd_result = self.variance_decomposition(var_result, periods=20)
        equity_fevd = fevd_result.get('decomposition', {}).get('equity_returns', {})

        return {
            'transmission_channels': transmission,
            'equity_variance_decomposition': equity_fevd,
            'var_lag_order': var_result.get('lag_order'),
            'macro_variables': macro_cols,
        }

    def structural_var(
        self,
        data: pd.DataFrame,
        identification: str = 'cholesky',
    ) -> Dict[str, Any]:
        """
        Estimate a Structural VAR (SVAR) using Cholesky or short-run restrictions.

        The ordering of columns in *data* determines the Cholesky ordering
        (first column = most exogenous).
        """
        data = data.dropna()
        if len(data) < self.min_observations:
            logger.warning("Insufficient data for SVAR (%d rows).", len(data))
            return {'error': 'insufficient_data'}

        # First fit a reduced-form VAR
        var_result = self.fit_var(data, max_lags=self.default_max_lags)
        if 'error' in var_result:
            return var_result

        fitted = var_result['fitted_model']
        cols = var_result['column_names']
        n = len(cols)

        try:
            if identification == 'cholesky':
                # Cholesky decomposition of residual covariance
                sigma_u = fitted.sigma_u  # (n, n) covariance matrix
                A0_inv = np.linalg.cholesky(sigma_u)  # lower-triangular

                # Structural IRFs
                irf_result = self.impulse_response(
                    var_result, periods=20, orthogonalized=True
                )

                return {
                    'identification': 'cholesky',
                    'A0_inverse': A0_inv.tolist(),
                    'column_order': cols,
                    'structural_irf': irf_result.get('irf', {}),
                    'cumulative_irf': irf_result.get('cumulative_irf', {}),
                    'residual_covariance': sigma_u.tolist(),
                    'var_result': var_result,
                }

            elif identification == 'short_run':
                # Short-run restrictions: A0 * e_t = B * u_t
                # Use identity A0 with Cholesky B as default
                sigma_u = fitted.sigma_u
                B = np.linalg.cholesky(sigma_u)
                A0 = np.eye(n)

                irf_result = self.impulse_response(
                    var_result, periods=20, orthogonalized=True
                )

                return {
                    'identification': 'short_run',
                    'A0': A0.tolist(),
                    'B': B.tolist(),
                    'column_order': cols,
                    'structural_irf': irf_result.get('irf', {}),
                    'cumulative_irf': irf_result.get('cumulative_irf', {}),
                    'var_result': var_result,
                }

            else:
                logger.error("Unknown identification scheme: %s", identification)
                return {'error': f'unknown identification: {identification}'}

        except Exception as e:
            logger.error("Structural VAR estimation failed: %s", e)
            return {'error': str(e)}

    def cointegrated_var(
        self,
        data: pd.DataFrame,
        det_order: int = 0,
    ) -> Dict[str, Any]:
        """
        Fit a Vector Error Correction Model (VECM) for cointegrated time series.

        Parameters
        ----------
        data : pd.DataFrame
            Levels (not returns) of potentially cointegrated variables.
        det_order : int
            Deterministic term order (-1 = no constant, 0 = constant,
            1 = constant + trend).
        """
        data = data.dropna()
        if len(data) < self.min_observations:
            logger.warning("Insufficient data for VECM (%d rows).", len(data))
            return {'error': 'insufficient_data'}

        cols = data.columns.tolist()
        n = len(cols)

        # Johansen cointegration test
        try:
            johansen_result = coint_johansen(
                data.values, det_order=det_order, k_ar_diff=1
            )
        except Exception as e:
            logger.error("Johansen test failed: %s", e)
            return {'error': f'johansen_test_failed: {e}'}

        # Determine cointegration rank at 5% significance
        trace_stats = johansen_result.lr1  # trace statistic
        trace_crit = johansen_result.cvt[:, 1]  # 5% critical values
        coint_rank = int(np.sum(trace_stats > trace_crit))

        if coint_rank == 0:
            logger.info("No cointegration found; VECM not appropriate.")
            return {
                'cointegration_rank': 0,
                'trace_statistics': trace_stats.tolist(),
                'critical_values_5pct': trace_crit.tolist(),
                'conclusion': 'no_cointegration',
            }

        # Fit VECM
        try:
            det_order_map = {-1: 'nc', 0: 'co', 1: 'colo'}
            deterministic = det_order_map.get(det_order, 'co')

            vecm = VECM(data, k_ar_diff=1, coint_rank=coint_rank, deterministic=deterministic)
            vecm_result = vecm.fit()

            # Extract cointegrating vectors (beta) and adjustment coefficients (alpha)
            alpha = vecm_result.alpha  # (n, r)
            beta = vecm_result.beta   # (n, r)
            gamma = vecm_result.gamma  # short-run coefficients

            return {
                'cointegration_rank': coint_rank,
                'trace_statistics': trace_stats.tolist(),
                'critical_values_5pct': trace_crit.tolist(),
                'alpha': alpha.tolist(),
                'beta': beta.tolist(),
                'gamma': gamma.tolist() if gamma is not None else None,
                'column_names': cols,
                'det_order': det_order,
                'summary': str(vecm_result.summary()),
            }

        except Exception as e:
            logger.error("VECM estimation failed: %s", e)
            return {
                'cointegration_rank': coint_rank,
                'error': f'vecm_fit_failed: {e}',
            }

    # ------------------------------------------------------------------ #
    #  Private helpers
    # ------------------------------------------------------------------ #

    def _select_optimal_lag_bic(
        self, data: np.ndarray, max_lags: int
    ) -> Optional[int]:
        """Select optimal lag order using BIC via a VAR model."""
        try:
            model = VAR(data)
            max_lags = min(max_lags, len(data) // (2 * data.shape[1]))
            if max_lags < 1:
                return 1
            result = model.select_order(maxlags=max_lags)
            lag = getattr(result, 'bic', 1)
            return max(int(lag), 1) if lag is not None else 1
        except Exception as e:
            logger.debug("BIC lag selection failed: %s", e)
            return 1

    @staticmethod
    def _compute_transfer_entropy(
        target: np.ndarray,
        target_past: np.ndarray,
        source_past: np.ndarray,
    ) -> float:
        """
        Binned transfer entropy estimator.

        TE = H(Y_t | Y_past) - H(Y_t | Y_past, X_past)
        """
        # Ensure 2-D
        if target_past.ndim == 1:
            target_past = target_past.reshape(-1, 1)
        if source_past.ndim == 1:
            source_past = source_past.reshape(-1, 1)

        n = len(target)

        # Joint and marginal counts via histogram hashing
        def _encode_rows(arr: np.ndarray) -> np.ndarray:
            """Hash each row into a single integer for fast grouping."""
            if arr.ndim == 1:
                return arr.astype(int)
            multipliers = np.array(
                [arr[:, c].max() + 1 for c in range(arr.shape[1])],
                dtype=int,
            )
            result = np.zeros(len(arr), dtype=int)
            for c in range(arr.shape[1]):
                stride = int(np.prod(multipliers[c + 1:])) if c + 1 < len(multipliers) else 1
                result += arr[:, c].astype(int) * stride
            return result

        y_t = target.astype(int)
        y_past_enc = _encode_rows(target_past)
        x_past_enc = _encode_rows(source_past)

        # H(Y_t | Y_past) via conditional entropy
        h_y_given_ypast = CausalModelAnalyzer._conditional_entropy(y_t, y_past_enc, n)

        # H(Y_t | Y_past, X_past)
        joint_past = _encode_rows(
            np.column_stack([target_past, source_past])
        )
        h_y_given_joint = CausalModelAnalyzer._conditional_entropy(y_t, joint_past, n)

        te = h_y_given_ypast - h_y_given_joint
        return max(te, 0.0)  # TE is non-negative in expectation

    @staticmethod
    def _conditional_entropy(
        target: np.ndarray, condition: np.ndarray, n: int
    ) -> float:
        """H(target | condition) using empirical frequencies."""
        # Build joint counts
        joint_keys = target.astype(np.int64) * (condition.max() + 1) + condition.astype(np.int64)
        joint_vals, joint_counts = np.unique(joint_keys, return_counts=True)

        cond_vals, cond_counts = np.unique(condition, return_counts=True)
        cond_map = dict(zip(cond_vals, cond_counts))

        h = 0.0
        for jk, jc in zip(joint_vals, joint_counts):
            c_val = int(jk % (condition.max() + 1))
            p_joint = jc / n
            p_cond = cond_map.get(c_val, 1) / n
            if p_joint > 0 and p_cond > 0:
                h -= p_joint * np.log2(p_joint / p_cond)
        return h
