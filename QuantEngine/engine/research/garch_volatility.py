"""
GARCH Family Volatility Forecasting System

Implements GARCH, EGARCH, and GJR-GARCH models for volatility forecasting:
- Symmetric GARCH(1,1) for baseline conditional variance modeling
- EGARCH for asymmetric/leverage effects with log-variance specification
- GJR-GARCH (threshold GARCH) for leverage effect via indicator function
- Multi-step ahead volatility forecasting with confidence intervals
- Rolling window out-of-sample forecast evaluation
- Model comparison via AIC/BIC/log-likelihood criteria
- Forecast accuracy assessment (Mincer-Zarnowitz, QLIKE loss)
- ARCH effect detection and volatility clustering diagnostics

Falls back to EWMA volatility estimation when the arch library is unavailable.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
import logging
from datetime import datetime
from scipy import stats

logger = logging.getLogger(__name__)

# Optional dependency: arch library for GARCH model estimation
try:
    from arch import arch_model
    HAS_ARCH = True
    logger.debug("arch library available - full GARCH model support enabled")
except ImportError:
    HAS_ARCH = False
    logger.warning(
        "arch library not available - falling back to EWMA volatility estimation. "
        "Install with: pip install arch"
    )


class GARCHVolatilityForecaster:
    """
    Production-grade GARCH family volatility forecasting with fallback support
    """

    def __init__(self, config: Dict[str, Any]):
        self.config = config

        # GARCH model defaults
        self.default_p = config.get('garch_p', 1)
        self.default_q = config.get('garch_q', 1)
        self.default_dist = config.get('error_distribution', 'normal')
        self.rescale = config.get('rescale', True)

        # EWMA fallback parameters
        self.ewma_span = config.get('ewma_span', 30)
        self.ewma_decay = config.get('ewma_decay', 0.94)

        # Rolling forecast defaults
        self.default_window = config.get('rolling_window', 252)
        self.default_horizon = config.get('forecast_horizon', 10)

        # Fitted model cache
        self._fitted_models: Dict[str, Any] = {}

    # ------------------------------------------------------------------
    # Core model fitting
    # ------------------------------------------------------------------

    def fit_garch(self, returns: pd.Series, model_type: str = 'GARCH') -> Dict[str, Any]:
        """
        Fit a GARCH(1,1) model to a return series.

        Args:
            returns: Log-return series (typically daily)
            model_type: Volatility process type ('GARCH', 'EGARCH', 'GJR-GARCH')

        Returns:
            Dictionary containing fitted parameters, diagnostics, and
            conditional volatility series.
        """
        try:
            returns = self._validate_returns(returns)

            if not HAS_ARCH:
                logger.info("arch library unavailable - using EWMA fallback for fit_garch")
                return self._ewma_fallback_fit(returns, label=model_type)

            vol_type = self._resolve_vol_type(model_type)
            am = arch_model(
                returns,
                vol=vol_type,
                p=self.default_p,
                q=self.default_q,
                dist=self.default_dist,
                rescale=self.rescale,
            )
            result = am.fit(disp='off', show_warning=False)
            self._fitted_models[model_type] = result

            cond_vol = result.conditional_volatility
            std_resid = result.std_resid

            # Ljung-Box test on standardised squared residuals
            lb_stat, lb_pval = self._ljung_box_squared(std_resid, lags=10)

            diagnostics = {
                'model_type': model_type,
                'aic': float(result.aic),
                'bic': float(result.bic),
                'log_likelihood': float(result.loglikelihood),
                'params': {k: float(v) for k, v in result.params.items()},
                'pvalues': {k: float(v) for k, v in result.pvalues.items()},
                'conditional_volatility': cond_vol,
                'std_residuals': std_resid,
                'ljung_box_stat': float(lb_stat),
                'ljung_box_pval': float(lb_pval),
                'nobs': int(result.nobs),
                'converged': result.convergence_flag == 0,
                'timestamp': datetime.now().isoformat(),
            }

            logger.info(
                f"Fitted {model_type} model - AIC: {result.aic:.2f}, "
                f"BIC: {result.bic:.2f}, converged: {diagnostics['converged']}"
            )
            return diagnostics

        except Exception as e:
            logger.error(f"fit_garch failed for {model_type}: {e}")
            return self._ewma_fallback_fit(returns, label=model_type, error=str(e))

    def fit_egarch(self, returns: pd.Series) -> Dict[str, Any]:
        """
        Fit an EGARCH model to capture asymmetric volatility (leverage effects).

        The EGARCH specification models log(sigma^2) so conditional variance
        is guaranteed positive without parameter constraints.

        Args:
            returns: Log-return series

        Returns:
            Dictionary with fitted parameters, diagnostics, and leverage
            effect magnitude.
        """
        try:
            returns = self._validate_returns(returns)

            if not HAS_ARCH:
                logger.info("arch library unavailable - using EWMA fallback for fit_egarch")
                return self._ewma_fallback_fit(returns, label='EGARCH')

            am = arch_model(
                returns,
                vol='EGARCH',
                p=self.default_p,
                q=self.default_q,
                dist=self.default_dist,
                rescale=self.rescale,
            )
            result = am.fit(disp='off', show_warning=False)
            self._fitted_models['EGARCH'] = result

            cond_vol = result.conditional_volatility
            std_resid = result.std_resid
            lb_stat, lb_pval = self._ljung_box_squared(std_resid, lags=10)

            # Extract leverage coefficient (gamma / alpha term)
            params = {k: float(v) for k, v in result.params.items()}
            leverage_coeff = None
            for key in params:
                if 'gamma' in key.lower() or 'alpha' in key.lower():
                    leverage_coeff = params[key]
                    break

            diagnostics = {
                'model_type': 'EGARCH',
                'aic': float(result.aic),
                'bic': float(result.bic),
                'log_likelihood': float(result.loglikelihood),
                'params': params,
                'pvalues': {k: float(v) for k, v in result.pvalues.items()},
                'conditional_volatility': cond_vol,
                'std_residuals': std_resid,
                'leverage_coefficient': leverage_coeff,
                'leverage_effect_present': leverage_coeff is not None and leverage_coeff < 0,
                'ljung_box_stat': float(lb_stat),
                'ljung_box_pval': float(lb_pval),
                'nobs': int(result.nobs),
                'converged': result.convergence_flag == 0,
                'timestamp': datetime.now().isoformat(),
            }

            logger.info(
                f"Fitted EGARCH model - AIC: {result.aic:.2f}, "
                f"leverage_coeff: {leverage_coeff}"
            )
            return diagnostics

        except Exception as e:
            logger.error(f"fit_egarch failed: {e}")
            return self._ewma_fallback_fit(returns, label='EGARCH', error=str(e))

    def fit_gjr_garch(self, returns: pd.Series) -> Dict[str, Any]:
        """
        Fit a GJR-GARCH (threshold GARCH) model for leverage effects.

        The GJR-GARCH adds an indicator function I(e_{t-1} < 0) so that
        negative shocks contribute more to future variance than positive
        shocks of the same magnitude.

        Args:
            returns: Log-return series

        Returns:
            Dictionary with fitted parameters, diagnostics, and threshold
            (gamma) coefficient.
        """
        try:
            returns = self._validate_returns(returns)

            if not HAS_ARCH:
                logger.info("arch library unavailable - using EWMA fallback for fit_gjr_garch")
                return self._ewma_fallback_fit(returns, label='GJR-GARCH')

            am = arch_model(
                returns,
                vol='GARCH',
                p=self.default_p,
                o=1,  # one threshold/asymmetry term
                q=self.default_q,
                dist=self.default_dist,
                rescale=self.rescale,
            )
            result = am.fit(disp='off', show_warning=False)
            self._fitted_models['GJR-GARCH'] = result

            cond_vol = result.conditional_volatility
            std_resid = result.std_resid
            lb_stat, lb_pval = self._ljung_box_squared(std_resid, lags=10)

            params = {k: float(v) for k, v in result.params.items()}
            gamma_coeff = None
            for key in params:
                if 'gamma' in key.lower():
                    gamma_coeff = params[key]
                    break

            diagnostics = {
                'model_type': 'GJR-GARCH',
                'aic': float(result.aic),
                'bic': float(result.bic),
                'log_likelihood': float(result.loglikelihood),
                'params': params,
                'pvalues': {k: float(v) for k, v in result.pvalues.items()},
                'conditional_volatility': cond_vol,
                'std_residuals': std_resid,
                'gamma_coefficient': gamma_coeff,
                'leverage_effect_present': gamma_coeff is not None and gamma_coeff > 0,
                'ljung_box_stat': float(lb_stat),
                'ljung_box_pval': float(lb_pval),
                'nobs': int(result.nobs),
                'converged': result.convergence_flag == 0,
                'timestamp': datetime.now().isoformat(),
            }

            logger.info(
                f"Fitted GJR-GARCH model - AIC: {result.aic:.2f}, "
                f"gamma: {gamma_coeff}"
            )
            return diagnostics

        except Exception as e:
            logger.error(f"fit_gjr_garch failed: {e}")
            return self._ewma_fallback_fit(returns, label='GJR-GARCH', error=str(e))

    # ------------------------------------------------------------------
    # Forecasting
    # ------------------------------------------------------------------

    def forecast_volatility(
        self,
        returns: pd.Series,
        horizon: int = 10,
        model_type: str = 'GARCH',
    ) -> Dict[str, Any]:
        """
        Produce multi-step ahead volatility forecasts.

        Args:
            returns: Log-return series used for model estimation
            horizon: Number of steps ahead to forecast
            model_type: One of 'GARCH', 'EGARCH', 'GJR-GARCH'

        Returns:
            Dictionary with point forecasts, confidence intervals, and
            annualised volatility estimates.
        """
        try:
            returns = self._validate_returns(returns)

            if not HAS_ARCH:
                logger.info("arch library unavailable - using EWMA fallback for forecast")
                return self._ewma_fallback_forecast(returns, horizon, label=model_type)

            # Fit or re-use cached model
            if model_type in self._fitted_models:
                result = self._fitted_models[model_type]
            else:
                fit_result = self._fit_model_by_type(returns, model_type)
                if 'error' in fit_result and 'fallback' in fit_result.get('model_type', ''):
                    return self._ewma_fallback_forecast(returns, horizon, label=model_type)
                result = self._fitted_models.get(model_type)
                if result is None:
                    return self._ewma_fallback_forecast(returns, horizon, label=model_type)

            forecasts = result.forecast(horizon=horizon)
            variance_forecast = forecasts.variance.iloc[-1].values
            vol_forecast = np.sqrt(variance_forecast)

            # Annualise
            annualised_vol = vol_forecast * np.sqrt(252)

            # Approximate confidence intervals (assuming log-normal variance)
            ci_lower = vol_forecast * np.exp(-1.96 * 0.5)
            ci_upper = vol_forecast * np.exp(1.96 * 0.5)

            forecast_index = list(range(1, horizon + 1))

            output = {
                'model_type': model_type,
                'horizon': horizon,
                'daily_vol_forecast': pd.Series(vol_forecast, index=forecast_index, name='daily_vol'),
                'annualised_vol_forecast': pd.Series(annualised_vol, index=forecast_index, name='ann_vol'),
                'variance_forecast': pd.Series(variance_forecast, index=forecast_index, name='variance'),
                'ci_lower': pd.Series(ci_lower, index=forecast_index, name='ci_lower'),
                'ci_upper': pd.Series(ci_upper, index=forecast_index, name='ci_upper'),
                'current_daily_vol': float(vol_forecast[0]),
                'current_annualised_vol': float(annualised_vol[0]),
                'timestamp': datetime.now().isoformat(),
            }

            logger.info(
                f"{model_type} {horizon}-step forecast - "
                f"1-step vol: {vol_forecast[0]:.6f}, "
                f"annualised: {annualised_vol[0]:.4f}"
            )
            return output

        except Exception as e:
            logger.error(f"forecast_volatility failed for {model_type}: {e}")
            return self._ewma_fallback_forecast(returns, horizon, label=model_type, error=str(e))

    # ------------------------------------------------------------------
    # Model comparison
    # ------------------------------------------------------------------

    def compare_models(self, returns: pd.Series) -> Dict[str, Any]:
        """
        Fit GARCH, EGARCH, and GJR-GARCH models and compare them using
        information criteria and log-likelihood.

        Args:
            returns: Log-return series

        Returns:
            Dictionary containing per-model results, rankings, and the
            name of the best model by AIC and BIC.
        """
        try:
            returns = self._validate_returns(returns)

            results = {}
            model_specs = {
                'GARCH': self.fit_garch,
                'EGARCH': self.fit_egarch,
                'GJR-GARCH': self.fit_gjr_garch,
            }

            for name, fit_fn in model_specs.items():
                if name == 'GARCH':
                    res = fit_fn(returns, model_type='GARCH')
                else:
                    res = fit_fn(returns)
                results[name] = res

            # Build comparison table
            comparison = {}
            for name, res in results.items():
                comparison[name] = {
                    'aic': res.get('aic', np.inf),
                    'bic': res.get('bic', np.inf),
                    'log_likelihood': res.get('log_likelihood', -np.inf),
                    'converged': res.get('converged', False),
                }

            # Rank by AIC and BIC (lower is better)
            aic_ranking = sorted(comparison.items(), key=lambda x: x[1]['aic'])
            bic_ranking = sorted(comparison.items(), key=lambda x: x[1]['bic'])

            best_aic = aic_ranking[0][0]
            best_bic = bic_ranking[0][0]

            output = {
                'model_results': results,
                'comparison': comparison,
                'aic_ranking': [name for name, _ in aic_ranking],
                'bic_ranking': [name for name, _ in bic_ranking],
                'best_model_aic': best_aic,
                'best_model_bic': best_bic,
                'recommended_model': best_bic,  # BIC penalises complexity more
                'timestamp': datetime.now().isoformat(),
            }

            logger.info(
                f"Model comparison complete - best AIC: {best_aic}, best BIC: {best_bic}"
            )
            return output

        except Exception as e:
            logger.error(f"compare_models failed: {e}")
            return {
                'error': str(e),
                'recommended_model': 'EWMA_fallback',
                'timestamp': datetime.now().isoformat(),
            }

    # ------------------------------------------------------------------
    # Rolling forecast
    # ------------------------------------------------------------------

    def rolling_forecast(
        self,
        returns: pd.Series,
        window: int = 252,
        horizon: int = 1,
    ) -> pd.DataFrame:
        """
        Perform rolling-window out-of-sample volatility forecasts.

        At each step the model is re-estimated on the trailing *window*
        observations and a *horizon*-step-ahead forecast is produced.

        Args:
            returns: Full log-return series
            window: Estimation window size (default 252 ~ 1 trading year)
            horizon: Forecast horizon in periods

        Returns:
            DataFrame with columns: date, forecasted_vol, realised_vol
        """
        try:
            returns = self._validate_returns(returns)

            if len(returns) < window + horizon:
                raise ValueError(
                    f"Insufficient data: need at least {window + horizon} observations, "
                    f"got {len(returns)}"
                )

            forecasted_vols = []
            realised_vols = []
            dates = []

            n = len(returns)
            for end in range(window, n - horizon + 1):
                train = returns.iloc[end - window:end]

                if HAS_ARCH:
                    try:
                        am = arch_model(
                            train,
                            vol='GARCH',
                            p=self.default_p,
                            q=self.default_q,
                            dist=self.default_dist,
                            rescale=self.rescale,
                        )
                        result = am.fit(disp='off', show_warning=False)
                        fcast = result.forecast(horizon=horizon)
                        fc_var = fcast.variance.iloc[-1].values[-1]
                        fc_vol = np.sqrt(fc_var)
                    except Exception:
                        fc_vol = train.ewm(span=self.ewma_span).std().iloc[-1]
                else:
                    fc_vol = train.ewm(span=self.ewma_span).std().iloc[-1]

                # Realised volatility over the forecast horizon
                actual = returns.iloc[end:end + horizon]
                rv = float(actual.std())

                forecasted_vols.append(float(fc_vol))
                realised_vols.append(rv)
                dates.append(returns.index[end] if hasattr(returns.index, '__getitem__') else end)

            df = pd.DataFrame({
                'date': dates,
                'forecasted_vol': forecasted_vols,
                'realised_vol': realised_vols,
            })

            logger.info(
                f"Rolling forecast complete - {len(df)} forecasts, "
                f"window={window}, horizon={horizon}"
            )
            return df

        except Exception as e:
            logger.error(f"rolling_forecast failed: {e}")
            return pd.DataFrame(columns=['date', 'forecasted_vol', 'realised_vol'])

    # ------------------------------------------------------------------
    # Forecast accuracy
    # ------------------------------------------------------------------

    def calculate_vol_forecast_accuracy(
        self,
        returns: pd.Series,
        forecasted_vol: pd.Series,
    ) -> Dict[str, Any]:
        """
        Evaluate volatility forecast accuracy using standard loss functions.

        Metrics computed:
        - Mincer-Zarnowitz regression (realised vol ~ alpha + beta * forecast)
        - QLIKE loss: E[sigma^2_r / sigma^2_f - log(sigma^2_r / sigma^2_f) - 1]
        - MSE and MAE of volatility forecasts
        - R-squared of Mincer-Zarnowitz regression

        Args:
            returns: Realised return series aligned with forecasted_vol
            forecasted_vol: Forecasted volatility series (same index)

        Returns:
            Dictionary with all accuracy metrics.
        """
        try:
            # Align series
            common_idx = returns.index.intersection(forecasted_vol.index)
            if len(common_idx) < 10:
                raise ValueError(
                    f"Need at least 10 overlapping observations, got {len(common_idx)}"
                )

            ret = returns.loc[common_idx]
            fvol = forecasted_vol.loc[common_idx]

            # Proxy for realised volatility: absolute returns
            realised_vol = ret.abs()

            # --- Mincer-Zarnowitz regression ---
            X = np.column_stack([np.ones(len(fvol)), fvol.values])
            y = realised_vol.values
            beta, residuals, _, _ = np.linalg.lstsq(X, y, rcond=None)
            y_hat = X @ beta
            ss_res = np.sum((y - y_hat) ** 2)
            ss_tot = np.sum((y - np.mean(y)) ** 2)
            r_squared = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0

            mz_alpha = float(beta[0])
            mz_beta = float(beta[1])

            # Standard errors (OLS)
            n = len(y)
            k = 2
            if n > k:
                sigma2 = ss_res / (n - k)
                var_beta = sigma2 * np.linalg.inv(X.T @ X)
                se_alpha = float(np.sqrt(var_beta[0, 0]))
                se_beta = float(np.sqrt(var_beta[1, 1]))
                # t-stats for H0: alpha=0, beta=1
                t_alpha = mz_alpha / se_alpha if se_alpha > 0 else 0.0
                t_beta = (mz_beta - 1.0) / se_beta if se_beta > 0 else 0.0
                p_alpha = float(2 * (1 - stats.t.cdf(abs(t_alpha), df=n - k)))
                p_beta = float(2 * (1 - stats.t.cdf(abs(t_beta), df=n - k)))
            else:
                se_alpha = se_beta = t_alpha = t_beta = p_alpha = p_beta = np.nan

            # --- QLIKE loss ---
            # Use squared values as variance proxies
            realised_var = realised_vol.values ** 2
            forecast_var = fvol.values ** 2
            # Avoid division by zero
            mask = forecast_var > 1e-16
            if mask.sum() > 0:
                ratio = realised_var[mask] / forecast_var[mask]
                qlike = float(np.mean(ratio - np.log(ratio) - 1))
            else:
                qlike = np.nan

            # --- MSE / MAE ---
            mse = float(np.mean((realised_vol.values - fvol.values) ** 2))
            mae = float(np.mean(np.abs(realised_vol.values - fvol.values)))

            output = {
                'mincer_zarnowitz': {
                    'alpha': mz_alpha,
                    'beta': mz_beta,
                    'se_alpha': se_alpha,
                    'se_beta': se_beta,
                    't_stat_alpha_eq_0': t_alpha,
                    't_stat_beta_eq_1': t_beta,
                    'p_value_alpha': p_alpha,
                    'p_value_beta': p_beta,
                    'r_squared': float(r_squared),
                    'efficient_forecast': p_alpha > 0.05 and p_beta > 0.05,
                },
                'qlike_loss': qlike,
                'mse': mse,
                'mae': mae,
                'nobs': int(n),
                'timestamp': datetime.now().isoformat(),
            }

            logger.info(
                f"Forecast accuracy - R2: {r_squared:.4f}, QLIKE: {qlike:.6f}, "
                f"MZ beta: {mz_beta:.4f}"
            )
            return output

        except Exception as e:
            logger.error(f"calculate_vol_forecast_accuracy failed: {e}")
            return {'error': str(e), 'timestamp': datetime.now().isoformat()}

    # ------------------------------------------------------------------
    # Volatility clustering detection
    # ------------------------------------------------------------------

    def detect_vol_clustering(self, returns: pd.Series) -> Dict[str, Any]:
        """
        Test for ARCH effects and volatility clustering in a return series.

        Diagnostics computed:
        - Engle's ARCH-LM test (H0: no ARCH effects)
        - Ljung-Box test on squared returns
        - Autocorrelation function of squared returns (first 20 lags)
        - Kurtosis and Jarque-Bera normality test on returns

        Args:
            returns: Log-return series

        Returns:
            Dictionary with test statistics, p-values, and interpretation.
        """
        try:
            returns = self._validate_returns(returns)
            squared_returns = returns ** 2

            # --- Engle's ARCH-LM test ---
            arch_lags = min(10, len(returns) // 5)
            arch_stat, arch_pval = self._arch_lm_test(returns, lags=arch_lags)

            # --- Ljung-Box on squared returns ---
            lb_lags = min(20, len(returns) // 5)
            lb_stat, lb_pval = self._ljung_box_squared(returns, lags=lb_lags)

            # --- Autocorrelation of squared returns ---
            max_acf_lags = min(20, len(squared_returns) - 1)
            acf_values = self._compute_acf(squared_returns, nlags=max_acf_lags)

            # --- Distributional properties ---
            kurt = float(stats.kurtosis(returns.dropna(), fisher=True))
            skew = float(stats.skew(returns.dropna()))
            jb_stat, jb_pval = stats.jarque_bera(returns.dropna())

            # Interpretation
            arch_effects_present = arch_pval < 0.05
            vol_clustering_present = lb_pval < 0.05
            fat_tails = kurt > 0
            significant_acf_lags = int(np.sum(np.abs(acf_values[1:]) > 1.96 / np.sqrt(len(returns))))

            output = {
                'arch_lm_test': {
                    'statistic': float(arch_stat),
                    'p_value': float(arch_pval),
                    'lags': arch_lags,
                    'arch_effects_present': arch_effects_present,
                },
                'ljung_box_squared': {
                    'statistic': float(lb_stat),
                    'p_value': float(lb_pval),
                    'lags': lb_lags,
                    'vol_clustering_present': vol_clustering_present,
                },
                'squared_return_acf': {
                    'values': acf_values.tolist(),
                    'significant_lags': significant_acf_lags,
                },
                'distribution': {
                    'kurtosis': kurt,
                    'skewness': skew,
                    'jarque_bera_stat': float(jb_stat),
                    'jarque_bera_pval': float(jb_pval),
                    'fat_tails': fat_tails,
                    'non_normal': float(jb_pval) < 0.05,
                },
                'summary': {
                    'arch_effects': arch_effects_present,
                    'volatility_clustering': vol_clustering_present,
                    'garch_modelling_recommended': arch_effects_present or vol_clustering_present,
                },
                'nobs': len(returns),
                'timestamp': datetime.now().isoformat(),
            }

            logger.info(
                f"Vol clustering detection - ARCH effects: {arch_effects_present}, "
                f"clustering: {vol_clustering_present}, kurtosis: {kurt:.2f}"
            )
            return output

        except Exception as e:
            logger.error(f"detect_vol_clustering failed: {e}")
            return {'error': str(e), 'timestamp': datetime.now().isoformat()}

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _validate_returns(self, returns: pd.Series) -> pd.Series:
        """Validate and clean a return series."""
        if not isinstance(returns, pd.Series):
            returns = pd.Series(returns)

        returns = returns.dropna()

        if len(returns) < 30:
            raise ValueError(
                f"Insufficient observations for GARCH estimation: {len(returns)} < 30"
            )

        # Warn about constant series
        if returns.std() == 0:
            raise ValueError("Return series has zero variance")

        return returns

    @staticmethod
    def _resolve_vol_type(model_type: str) -> str:
        """Map user-facing model name to arch library vol parameter."""
        mapping = {
            'GARCH': 'GARCH',
            'EGARCH': 'EGARCH',
            'GJR-GARCH': 'GARCH',  # GJR uses o=1 with GARCH vol
        }
        return mapping.get(model_type, 'GARCH')

    def _fit_model_by_type(self, returns: pd.Series, model_type: str) -> Dict[str, Any]:
        """Dispatch to the appropriate fit method."""
        if model_type == 'EGARCH':
            return self.fit_egarch(returns)
        elif model_type == 'GJR-GARCH':
            return self.fit_gjr_garch(returns)
        else:
            return self.fit_garch(returns, model_type='GARCH')

    @staticmethod
    def _ljung_box_squared(resid: pd.Series, lags: int = 10) -> Tuple[float, float]:
        """Ljung-Box test on squared (standardised) residuals."""
        try:
            resid = resid.dropna()
            squared = resid ** 2
            n = len(squared)
            if n < lags + 1:
                return 0.0, 1.0

            mean_sq = squared.mean()
            denom = np.sum((squared - mean_sq) ** 2)
            if denom == 0:
                return 0.0, 1.0

            acf_vals = np.array([
                np.sum((squared.iloc[k:] - mean_sq) * (squared.iloc[:-k] - mean_sq)) / denom
                for k in range(1, lags + 1)
            ])

            q_stat = float(n * (n + 2) * np.sum(acf_vals ** 2 / np.arange(n - 1, n - lags - 1, -1)))
            p_value = float(1 - stats.chi2.cdf(q_stat, df=lags))
            return q_stat, p_value
        except Exception:
            return 0.0, 1.0

    @staticmethod
    def _arch_lm_test(returns: pd.Series, lags: int = 10) -> Tuple[float, float]:
        """
        Engle's ARCH-LM test.

        Regresses squared residuals on their own lags and tests whether
        the R-squared is significantly different from zero.
        """
        try:
            returns = returns.dropna()
            resid = returns - returns.mean()
            squared = (resid ** 2).values
            n = len(squared)

            if n < lags + 1:
                return 0.0, 1.0

            y = squared[lags:]
            X = np.column_stack([squared[lags - i - 1: n - i - 1] for i in range(lags)])
            X = np.column_stack([np.ones(len(y)), X])

            beta, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
            y_hat = X @ beta
            ss_res = np.sum((y - y_hat) ** 2)
            ss_tot = np.sum((y - np.mean(y)) ** 2)
            r_squared = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0

            lm_stat = float(len(y) * r_squared)
            p_value = float(1 - stats.chi2.cdf(lm_stat, df=lags))
            return lm_stat, p_value
        except Exception:
            return 0.0, 1.0

    @staticmethod
    def _compute_acf(series: pd.Series, nlags: int = 20) -> np.ndarray:
        """Compute autocorrelation function up to nlags."""
        series = series.dropna().values
        n = len(series)
        mean = np.mean(series)
        denom = np.sum((series - mean) ** 2)
        if denom == 0:
            return np.zeros(nlags + 1)

        acf = np.array([
            np.sum((series[k:] - mean) * (series[:n - k] - mean)) / denom
            if k > 0 else 1.0
            for k in range(nlags + 1)
        ])
        return acf

    # ------------------------------------------------------------------
    # EWMA fallback
    # ------------------------------------------------------------------

    def _ewma_fallback_fit(
        self,
        returns: pd.Series,
        label: str = 'EWMA_fallback',
        error: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Simple EWMA volatility estimation as a fallback when the arch
        library is unavailable or model fitting fails.
        """
        try:
            returns = returns.dropna() if isinstance(returns, pd.Series) else pd.Series(returns).dropna()
            ewma_var = returns.ewm(span=self.ewma_span).var()
            ewma_vol = np.sqrt(ewma_var)

            result = {
                'model_type': f'{label} (EWMA fallback)',
                'aic': np.nan,
                'bic': np.nan,
                'log_likelihood': np.nan,
                'params': {'lambda': self.ewma_decay, 'span': self.ewma_span},
                'pvalues': {},
                'conditional_volatility': ewma_vol,
                'std_residuals': returns / ewma_vol.replace(0, np.nan),
                'converged': True,
                'nobs': len(returns),
                'fallback': True,
                'timestamp': datetime.now().isoformat(),
            }
            if error:
                result['original_error'] = error

            logger.info(f"EWMA fallback fit complete for {label} - span={self.ewma_span}")
            return result

        except Exception as e:
            logger.error(f"EWMA fallback fit failed: {e}")
            return {
                'model_type': f'{label} (EWMA fallback)',
                'error': str(e),
                'fallback': True,
                'timestamp': datetime.now().isoformat(),
            }

    def _ewma_fallback_forecast(
        self,
        returns: pd.Series,
        horizon: int,
        label: str = 'EWMA_fallback',
        error: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Multi-step EWMA volatility forecast using RiskMetrics-style
        exponential smoothing with flat extrapolation.
        """
        try:
            returns = returns.dropna() if isinstance(returns, pd.Series) else pd.Series(returns).dropna()
            ewma_vol = returns.ewm(span=self.ewma_span).std()
            last_vol = float(ewma_vol.iloc[-1])

            # Flat forecast (EWMA is a martingale for variance)
            vol_forecast = np.full(horizon, last_vol)
            annualised_vol = vol_forecast * np.sqrt(252)

            # Simple confidence intervals based on estimation uncertainty
            ci_lower = vol_forecast * 0.7
            ci_upper = vol_forecast * 1.3

            forecast_index = list(range(1, horizon + 1))

            result = {
                'model_type': f'{label} (EWMA fallback)',
                'horizon': horizon,
                'daily_vol_forecast': pd.Series(vol_forecast, index=forecast_index, name='daily_vol'),
                'annualised_vol_forecast': pd.Series(annualised_vol, index=forecast_index, name='ann_vol'),
                'variance_forecast': pd.Series(vol_forecast ** 2, index=forecast_index, name='variance'),
                'ci_lower': pd.Series(ci_lower, index=forecast_index, name='ci_lower'),
                'ci_upper': pd.Series(ci_upper, index=forecast_index, name='ci_upper'),
                'current_daily_vol': last_vol,
                'current_annualised_vol': float(last_vol * np.sqrt(252)),
                'fallback': True,
                'timestamp': datetime.now().isoformat(),
            }
            if error:
                result['original_error'] = error

            logger.info(
                f"EWMA fallback forecast for {label} - "
                f"1-step vol: {last_vol:.6f}, horizon: {horizon}"
            )
            return result

        except Exception as e:
            logger.error(f"EWMA fallback forecast failed: {e}")
            return {
                'model_type': f'{label} (EWMA fallback)',
                'error': str(e),
                'fallback': True,
                'timestamp': datetime.now().isoformat(),
            }
