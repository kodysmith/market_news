"""
Extreme Value Theory and Advanced Tail Risk Models for AI Quant Trading System

Implements:
- Peaks Over Threshold (POT) with Generalized Pareto Distribution
- Block Maxima with Generalized Extreme Value distribution
- Copula models for joint tail dependence (Gaussian, Student-t)
- Merton jump-diffusion model for crash risk
- Tail risk parity portfolio allocation
- Monte Carlo simulation from fitted copulas
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Tuple, Union
import logging
from scipy import stats
from scipy.optimize import minimize, minimize_scalar
from scipy.special import gamma as gamma_func
import warnings

logger = logging.getLogger(__name__)


class TailRiskAnalyzer:
    """
    Advanced tail risk analysis using Extreme Value Theory, copula models,
    and jump-diffusion processes for robust portfolio risk management
    """

    def __init__(self, config: Dict[str, Any]):
        self.config = config

        # EVT parameters
        self.default_threshold_percentile = config.get('threshold_percentile', 95.0)
        self.default_confidence = config.get('confidence_level', 0.99)
        self.default_block_size = config.get('block_size', 21)  # ~1 month of trading days

        # Copula parameters
        self.n_copula_simulations = config.get('n_copula_simulations', 10000)

        # Jump-diffusion parameters
        self.jump_threshold_sigma = config.get('jump_threshold_sigma', 3.0)

        # Tail risk parity parameters
        self.trp_confidence = config.get('trp_confidence', 0.99)
        self.min_weight = config.get('min_weight', 0.01)

    # ----------------------------------------------------------------
    # 1. Extreme Value Theory - Peaks Over Threshold (POT)
    # ----------------------------------------------------------------

    def fit_gpd(self, losses: pd.Series, threshold_percentile: float = 95.0) -> Dict[str, Any]:
        """
        Fit Generalized Pareto Distribution to exceedances above a threshold

        Args:
            losses: Series of loss values (positive = loss)
            threshold_percentile: Percentile for threshold selection

        Returns:
            Dictionary with GPD shape (xi), scale (beta), threshold, and diagnostics
        """
        losses_clean = losses.dropna()
        if len(losses_clean) < 50:
            return {'error': 'Insufficient data for GPD fitting (need >= 50 observations)'}

        threshold = np.percentile(losses_clean, threshold_percentile)
        exceedances = losses_clean[losses_clean > threshold] - threshold

        if len(exceedances) < 10:
            return {'error': f'Too few exceedances ({len(exceedances)}) above threshold {threshold:.6f}'}

        # Fit GPD using scipy MLE
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                shape, loc, scale = stats.genpareto.fit(exceedances, floc=0)
        except Exception as e:
            logger.warning(f"GPD fitting failed: {e}")
            return {'error': f'GPD fitting failed: {str(e)}'}

        # Goodness of fit via Kolmogorov-Smirnov test
        ks_stat, ks_pvalue = stats.kstest(exceedances, 'genpareto', args=(shape, 0, scale))

        logger.info(f"GPD fit: xi={shape:.4f}, beta={scale:.6f}, threshold={threshold:.6f}, "
                     f"n_exceedances={len(exceedances)}, KS p-value={ks_pvalue:.4f}")

        return {
            'xi': shape,               # Shape parameter (tail index)
            'beta': scale,             # Scale parameter
            'threshold': threshold,
            'threshold_percentile': threshold_percentile,
            'n_exceedances': len(exceedances),
            'n_total': len(losses_clean),
            'exceedance_rate': len(exceedances) / len(losses_clean),
            'ks_statistic': ks_stat,
            'ks_pvalue': ks_pvalue,
            'mean_excess': float(exceedances.mean()),
            'tail_type': 'heavy' if shape > 0 else ('light' if shape < 0 else 'exponential')
        }

    def evt_var(self, losses: pd.Series, confidence: float = 0.99) -> Dict[str, Any]:
        """
        Calculate Value at Risk using EVT/GPD (superior tail estimation vs parametric)

        Args:
            losses: Series of loss values (positive = loss)
            confidence: Confidence level (e.g. 0.99 for 99% VaR)

        Returns:
            Dictionary with EVT VaR, comparison to parametric, and parameters
        """
        gpd_fit = self.fit_gpd(losses, self.default_threshold_percentile)
        if 'error' in gpd_fit:
            return gpd_fit

        xi = gpd_fit['xi']
        beta = gpd_fit['beta']
        threshold = gpd_fit['threshold']
        n_total = gpd_fit['n_total']
        n_exceed = gpd_fit['n_exceedances']
        exceedance_rate = gpd_fit['exceedance_rate']

        # EVT VaR formula: u + (beta/xi) * ((n/Nu * (1-p))^(-xi) - 1)
        p = confidence
        if abs(xi) < 1e-10:
            # Exponential tail case
            evt_var_value = threshold + beta * np.log(exceedance_rate / (1 - p))
        else:
            evt_var_value = threshold + (beta / xi) * (
                (exceedance_rate / (1 - p)) ** xi - 1
            )

        # Parametric VaR for comparison
        losses_clean = losses.dropna()
        mu = losses_clean.mean()
        sigma = losses_clean.std()
        parametric_var = mu + stats.norm.ppf(confidence) * sigma

        logger.info(f"EVT VaR({confidence:.1%}): {evt_var_value:.6f} vs Parametric: {parametric_var:.6f}")

        return {
            'evt_var': evt_var_value,
            'parametric_var': parametric_var,
            'confidence': confidence,
            'evt_parametric_ratio': evt_var_value / parametric_var if parametric_var != 0 else np.inf,
            'gpd_params': gpd_fit
        }

    def evt_cvar(self, losses: pd.Series, confidence: float = 0.99) -> Dict[str, Any]:
        """
        Calculate CVaR (Expected Shortfall) using EVT/GPD

        Args:
            losses: Series of loss values (positive = loss)
            confidence: Confidence level

        Returns:
            Dictionary with EVT CVaR and comparison metrics
        """
        var_result = self.evt_var(losses, confidence)
        if 'error' in var_result:
            return var_result

        xi = var_result['gpd_params']['xi']
        beta = var_result['gpd_params']['beta']
        threshold = var_result['gpd_params']['threshold']
        evt_var_value = var_result['evt_var']

        # EVT CVaR formula: VaR/(1-xi) + (beta - xi*u)/(1-xi)
        if xi >= 1.0:
            return {'error': 'CVaR undefined for xi >= 1 (infinite mean)'}

        evt_cvar_value = evt_var_value / (1 - xi) + (beta - xi * threshold) / (1 - xi)

        # Parametric CVaR for comparison
        losses_clean = losses.dropna()
        mu = losses_clean.mean()
        sigma = losses_clean.std()
        z = stats.norm.ppf(confidence)
        parametric_cvar = mu + sigma * stats.norm.pdf(z) / (1 - confidence)

        logger.info(f"EVT CVaR({confidence:.1%}): {evt_cvar_value:.6f} vs Parametric: {parametric_cvar:.6f}")

        return {
            'evt_cvar': evt_cvar_value,
            'parametric_cvar': parametric_cvar,
            'evt_var': evt_var_value,
            'confidence': confidence,
            'cvar_var_ratio': evt_cvar_value / evt_var_value if evt_var_value != 0 else np.inf,
            'gpd_params': var_result['gpd_params']
        }

    def select_threshold(self, losses: pd.Series) -> Dict[str, Any]:
        """
        Automated threshold selection using mean residual life plot analysis

        Examines stability of GPD parameter estimates across threshold candidates
        to find the optimal threshold for POT modeling.

        Args:
            losses: Series of loss values

        Returns:
            Dictionary with recommended threshold, MRL data, and parameter stability
        """
        losses_clean = losses.dropna().sort_values()
        n = len(losses_clean)

        if n < 100:
            return {'error': 'Insufficient data for threshold selection (need >= 100)'}

        # Mean residual life analysis across candidate thresholds
        percentiles = np.arange(80, 98, 0.5)
        mrl_data = []
        gpd_stability = []

        for pct in percentiles:
            u = np.percentile(losses_clean, pct)
            exceedances = losses_clean[losses_clean > u] - u
            n_exceed = len(exceedances)

            if n_exceed < 10:
                continue

            mean_excess = exceedances.mean()
            se_excess = exceedances.std() / np.sqrt(n_exceed)

            mrl_data.append({
                'percentile': pct,
                'threshold': u,
                'mean_excess': mean_excess,
                'se': se_excess,
                'n_exceedances': n_exceed
            })

            # Fit GPD at this threshold for parameter stability check
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    shape, _, scale = stats.genpareto.fit(exceedances, floc=0)
                gpd_stability.append({
                    'percentile': pct,
                    'threshold': u,
                    'xi': shape,
                    'beta': scale,
                    'modified_scale': scale - shape * u  # Should be stable
                })
            except Exception:
                pass

        if len(gpd_stability) < 3:
            return {'error': 'Could not fit GPD across enough thresholds'}

        # Find optimal threshold: where modified scale and shape stabilize
        # Use rolling variance of xi estimates to detect stability
        xi_values = [s['xi'] for s in gpd_stability]
        window = max(3, len(xi_values) // 5)
        xi_rolling_var = []

        for i in range(window, len(xi_values)):
            xi_rolling_var.append(np.var(xi_values[i - window:i]))

        if xi_rolling_var:
            # Optimal threshold is where xi variance is minimized (most stable)
            optimal_idx = np.argmin(xi_rolling_var) + window
            optimal_idx = min(optimal_idx, len(gpd_stability) - 1)
            recommended = gpd_stability[optimal_idx]
        else:
            # Fallback to 90th percentile
            recommended = gpd_stability[len(gpd_stability) // 2]

        logger.info(f"Threshold selection: recommended percentile={recommended['percentile']:.1f}, "
                     f"threshold={recommended['threshold']:.6f}")

        return {
            'recommended_percentile': recommended['percentile'],
            'recommended_threshold': recommended['threshold'],
            'recommended_xi': recommended['xi'],
            'recommended_beta': recommended['beta'],
            'mrl_data': mrl_data,
            'gpd_stability': gpd_stability,
            'n_observations': n
        }

    # ----------------------------------------------------------------
    # 2. Block Maxima (GEV)
    # ----------------------------------------------------------------

    def fit_gev(self, returns: pd.Series, block_size: int = 21) -> Dict[str, Any]:
        """
        Fit Generalized Extreme Value distribution to block maxima of losses

        Args:
            returns: Return series (losses computed as negative returns)
            block_size: Block size in trading days (21 ~ 1 month)

        Returns:
            Dictionary with GEV parameters (location, scale, shape) and diagnostics
        """
        losses = -returns.dropna()
        n = len(losses)

        if n < block_size * 5:
            return {'error': f'Insufficient data: need >= {block_size * 5} observations'}

        # Extract block maxima
        n_blocks = n // block_size
        block_maxima = []

        for i in range(n_blocks):
            block = losses.iloc[i * block_size:(i + 1) * block_size]
            block_maxima.append(block.max())

        block_maxima = np.array(block_maxima)

        if len(block_maxima) < 10:
            return {'error': 'Too few blocks for reliable GEV fitting'}

        # Fit GEV using scipy MLE
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                shape, loc, scale = stats.genextreme.fit(block_maxima)
        except Exception as e:
            logger.warning(f"GEV fitting failed: {e}")
            return {'error': f'GEV fitting failed: {str(e)}'}

        # Note: scipy uses sign convention c = -xi, so xi = -shape
        xi = -shape

        # Goodness of fit
        ks_stat, ks_pvalue = stats.kstest(block_maxima, 'genextreme', args=(shape, loc, scale))

        # Determine GEV type
        if abs(xi) < 0.05:
            gev_type = 'Gumbel (Type I)'
        elif xi > 0:
            gev_type = 'Frechet (Type II) - heavy tail'
        else:
            gev_type = 'Weibull (Type III) - bounded tail'

        logger.info(f"GEV fit: xi={xi:.4f}, mu={loc:.6f}, sigma={scale:.6f}, type={gev_type}")

        return {
            'xi': xi,                   # Shape (tail index)
            'mu': loc,                  # Location
            'sigma': scale,             # Scale
            'scipy_shape': shape,       # scipy sign convention
            'block_size': block_size,
            'n_blocks': len(block_maxima),
            'gev_type': gev_type,
            'block_maxima_mean': float(np.mean(block_maxima)),
            'block_maxima_std': float(np.std(block_maxima)),
            'ks_statistic': ks_stat,
            'ks_pvalue': ks_pvalue
        }

    def gev_return_level(self, gev_params: Dict[str, Any], return_period: int = 100) -> float:
        """
        Calculate the expected maximum loss for a given return period

        Args:
            gev_params: Output from fit_gev containing xi, mu, sigma, block_size
            return_period: Return period in blocks (e.g., 100 blocks)

        Returns:
            Expected maximum loss over the return period
        """
        xi = gev_params['xi']
        mu = gev_params['mu']
        sigma = gev_params['sigma']

        # Return level formula: mu + (sigma/xi) * ((-log(1-1/m))^(-xi) - 1)
        yp = -np.log(1 - 1.0 / return_period)

        if abs(xi) < 1e-10:
            # Gumbel case
            return_level = mu - sigma * np.log(yp)
        else:
            return_level = mu + (sigma / xi) * (yp ** (-xi) - 1)

        logger.info(f"GEV return level: {return_level:.6f} for {return_period}-block return period")

        return float(return_level)

    # ----------------------------------------------------------------
    # 3. Copula Models for Joint Tail Dependence
    # ----------------------------------------------------------------

    def fit_gaussian_copula(self, returns_df: pd.DataFrame) -> Dict[str, Any]:
        """
        Fit Gaussian copula to multivariate returns

        Args:
            returns_df: DataFrame of asset returns (columns = assets)

        Returns:
            Dictionary with correlation matrix and tail dependence assessment
        """
        returns_clean = returns_df.dropna()
        n, d = returns_clean.shape

        if n < 50:
            return {'error': 'Insufficient data for copula fitting (need >= 50)'}

        # Step 1: Transform to uniform marginals using empirical CDF (PIT)
        uniform_data = np.zeros_like(returns_clean.values)
        for j in range(d):
            col = returns_clean.iloc[:, j].values
            # Rank-based empirical CDF (scaled to avoid 0 and 1)
            ranks = stats.rankdata(col)
            uniform_data[:, j] = ranks / (n + 1)

        # Step 2: Transform to standard normal
        normal_data = stats.norm.ppf(uniform_data)

        # Step 3: Estimate correlation matrix (MLE for Gaussian copula)
        correlation_matrix = np.corrcoef(normal_data.T)

        # Gaussian copula has zero tail dependence
        tickers = list(returns_clean.columns)

        logger.info(f"Gaussian copula fit: {d} assets, {n} observations")

        return {
            'copula_type': 'gaussian',
            'correlation_matrix': correlation_matrix.tolist(),
            'tickers': tickers,
            'n_observations': n,
            'n_assets': d,
            'upper_tail_dependence': 0.0,  # Always zero for Gaussian copula
            'lower_tail_dependence': 0.0,  # Always zero for Gaussian copula
            'tail_dependence_note': 'Gaussian copula has zero tail dependence by construction'
        }

    def fit_t_copula(self, returns_df: pd.DataFrame) -> Dict[str, Any]:
        """
        Fit Student-t copula to multivariate returns via maximum likelihood

        Args:
            returns_df: DataFrame of asset returns (columns = assets)

        Returns:
            Dictionary with correlation, degrees of freedom, and tail dependence
        """
        returns_clean = returns_df.dropna()
        n, d = returns_clean.shape

        if n < 50:
            return {'error': 'Insufficient data for copula fitting (need >= 50)'}

        # Step 1: Transform to uniform marginals using PIT
        uniform_data = np.zeros((n, d))
        for j in range(d):
            col = returns_clean.iloc[:, j].values
            ranks = stats.rankdata(col)
            uniform_data[:, j] = ranks / (n + 1)

        # Step 2: Estimate degrees of freedom and correlation via profile likelihood
        # For each candidate nu, transform to t-quantiles and compute correlation
        best_nu = 5.0
        best_ll = -np.inf

        for nu_candidate in np.arange(2.5, 30.5, 0.5):
            # Transform uniform to t-quantiles
            t_data = stats.t.ppf(uniform_data, df=nu_candidate)

            # Correlation matrix from t-transformed data
            corr = np.corrcoef(t_data.T)

            # Approximate log-likelihood for t-copula
            try:
                det_corr = np.linalg.det(corr)
                if det_corr <= 0:
                    continue
                inv_corr = np.linalg.inv(corr)

                ll = 0.0
                for i in range(n):
                    x = t_data[i, :]
                    # t-copula density contribution
                    quad_form = x @ inv_corr @ x
                    quad_indep = np.sum(x ** 2)

                    ll += (
                        np.log(gamma_func((nu_candidate + d) / 2))
                        + (d - 1) * np.log(gamma_func(nu_candidate / 2))
                        - d * np.log(gamma_func((nu_candidate + 1) / 2))
                        - 0.5 * np.log(det_corr)
                        - ((nu_candidate + d) / 2) * np.log(1 + quad_form / nu_candidate)
                        + np.sum(((nu_candidate + 1) / 2) * np.log(1 + x ** 2 / nu_candidate))
                    )

                if ll > best_ll:
                    best_ll = ll
                    best_nu = nu_candidate
            except (np.linalg.LinAlgError, ValueError):
                continue

        # Final fit with optimal nu
        t_data = stats.t.ppf(uniform_data, df=best_nu)
        correlation_matrix = np.corrcoef(t_data.T)

        # Tail dependence coefficient for t-copula (pairwise)
        # lambda = 2 * t_{nu+1}(-sqrt((nu+1)(1-rho)/(1+rho)))
        tail_dependence_matrix = np.zeros((d, d))
        for i in range(d):
            for j in range(d):
                if i != j:
                    rho = correlation_matrix[i, j]
                    arg = -np.sqrt((best_nu + 1) * (1 - rho) / (1 + rho))
                    tail_dependence_matrix[i, j] = 2 * stats.t.cdf(arg, df=best_nu + 1)

        tickers = list(returns_clean.columns)

        logger.info(f"t-copula fit: nu={best_nu:.1f}, {d} assets, {n} observations")

        return {
            'copula_type': 't',
            'correlation_matrix': correlation_matrix.tolist(),
            'degrees_of_freedom': best_nu,
            'tickers': tickers,
            'n_observations': n,
            'n_assets': d,
            'tail_dependence_matrix': tail_dependence_matrix.tolist(),
            'log_likelihood': best_ll
        }

    def calculate_tail_dependence(self, returns_a: pd.Series, returns_b: pd.Series) -> Dict[str, Any]:
        """
        Calculate empirical upper and lower tail dependence coefficients

        Args:
            returns_a: Return series for asset A
            returns_b: Return series for asset B

        Returns:
            Dictionary with upper/lower tail dependence at multiple quantile levels
        """
        aligned = pd.concat([returns_a, returns_b], axis=1).dropna()
        if len(aligned) < 100:
            return {'error': 'Insufficient data for tail dependence (need >= 100)'}

        a_vals = aligned.iloc[:, 0].values
        b_vals = aligned.iloc[:, 1].values
        n = len(a_vals)

        # Convert to uniform marginals
        u_a = stats.rankdata(a_vals) / (n + 1)
        u_b = stats.rankdata(b_vals) / (n + 1)

        # Compute tail dependence at multiple quantile levels
        quantile_levels = [0.01, 0.02, 0.05, 0.10]
        lower_td = {}
        upper_td = {}

        for q in quantile_levels:
            # Lower tail: P(U2 <= q | U1 <= q)
            lower_mask = u_a <= q
            if lower_mask.sum() > 0:
                lower_td[q] = float(np.mean(u_b[lower_mask] <= q))
            else:
                lower_td[q] = np.nan

            # Upper tail: P(U2 > 1-q | U1 > 1-q)
            upper_mask = u_a > (1 - q)
            if upper_mask.sum() > 0:
                upper_td[q] = float(np.mean(u_b[upper_mask] > (1 - q)))
            else:
                upper_td[q] = np.nan

        # Chi-plot based summary (average across levels for a single estimate)
        valid_lower = [v for v in lower_td.values() if not np.isnan(v)]
        valid_upper = [v for v in upper_td.values() if not np.isnan(v)]

        logger.info(f"Tail dependence: lower~{np.mean(valid_lower):.3f}, upper~{np.mean(valid_upper):.3f}")

        return {
            'lower_tail_dependence': lower_td,
            'upper_tail_dependence': upper_td,
            'lower_td_estimate': float(np.mean(valid_lower)) if valid_lower else np.nan,
            'upper_td_estimate': float(np.mean(valid_upper)) if valid_upper else np.nan,
            'n_observations': n,
            'linear_correlation': float(np.corrcoef(a_vals, b_vals)[0, 1]),
            'rank_correlation': float(stats.spearmanr(a_vals, b_vals).correlation)
        }

    def simulate_joint_losses(self, copula_params: Dict[str, Any],
                               n_simulations: int = 10000) -> pd.DataFrame:
        """
        Monte Carlo simulation from a fitted copula for stress testing

        Args:
            copula_params: Output from fit_gaussian_copula or fit_t_copula
            n_simulations: Number of simulation paths

        Returns:
            DataFrame of simulated uniform marginals (apply inverse CDF for actual returns)
        """
        copula_type = copula_params.get('copula_type', 'gaussian')
        corr = np.array(copula_params['correlation_matrix'])
        tickers = copula_params.get('tickers', [f'asset_{i}' for i in range(corr.shape[0])])
        d = corr.shape[0]

        # Ensure correlation matrix is positive definite
        eigvals = np.linalg.eigvalsh(corr)
        if np.min(eigvals) <= 0:
            # Nearest positive definite matrix
            corr = self._nearest_positive_definite(corr)

        if copula_type == 'gaussian':
            # Simulate from multivariate normal, then transform to uniform
            normal_samples = np.random.multivariate_normal(np.zeros(d), corr, size=n_simulations)
            uniform_samples = stats.norm.cdf(normal_samples)

        elif copula_type == 't':
            nu = copula_params.get('degrees_of_freedom', 5.0)

            # Simulate from multivariate t via normal/chi-squared mixture
            normal_samples = np.random.multivariate_normal(np.zeros(d), corr, size=n_simulations)
            chi2_samples = np.random.chisquare(df=nu, size=n_simulations)
            t_samples = normal_samples / np.sqrt(chi2_samples[:, np.newaxis] / nu)
            uniform_samples = stats.t.cdf(t_samples, df=nu)

        else:
            logger.error(f"Unknown copula type: {copula_type}")
            return pd.DataFrame()

        logger.info(f"Simulated {n_simulations} paths from {copula_type} copula ({d} assets)")

        return pd.DataFrame(uniform_samples, columns=tickers)

    # ----------------------------------------------------------------
    # 4. Jump-Diffusion Model (Merton)
    # ----------------------------------------------------------------

    def fit_merton_jump_diffusion(self, returns: pd.Series) -> Dict[str, Any]:
        """
        Estimate Merton jump-diffusion model parameters from return series

        Model: dS/S = (mu - lambda*k)dt + sigma*dW + J*dN
        where J ~ N(mu_j, sigma_j^2), N is Poisson(lambda)

        Args:
            returns: Log return series

        Returns:
            Dictionary with drift, diffusion vol, jump intensity, jump mean, jump vol
        """
        returns_clean = returns.dropna()
        n = len(returns_clean)

        if n < 100:
            return {'error': 'Insufficient data for jump-diffusion estimation (need >= 100)'}

        r = returns_clean.values
        dt = 1.0 / 252  # Daily

        # Step 1: Identify jump days using threshold
        sigma_est = np.std(r)
        threshold = self.jump_threshold_sigma * sigma_est
        jump_mask = np.abs(r - np.mean(r)) > threshold
        jump_days = r[jump_mask]
        normal_days = r[~jump_mask]

        # Step 2: Estimate parameters
        # Jump intensity (lambda): average number of jumps per unit time
        n_jumps = len(jump_days)
        jump_intensity = n_jumps / (n * dt)  # Annualized jump frequency

        # Jump size distribution
        if n_jumps > 1:
            jump_mean = float(np.mean(jump_days))
            jump_vol = float(np.std(jump_days))
        else:
            jump_mean = 0.0
            jump_vol = 0.0

        # Diffusion parameters (from non-jump days)
        if len(normal_days) > 10:
            diffusion_vol = float(np.std(normal_days)) * np.sqrt(252)  # Annualized
            drift = float(np.mean(normal_days)) * 252  # Annualized
        else:
            diffusion_vol = float(sigma_est) * np.sqrt(252)
            drift = float(np.mean(r)) * 252

        # Compensated drift (risk-neutral adjustment)
        k = np.exp(jump_mean + 0.5 * jump_vol ** 2) - 1 if jump_vol > 0 else jump_mean
        compensated_drift = drift + jump_intensity * k

        # Total variance decomposition
        total_var = float(np.var(r)) * 252
        diffusion_var = (diffusion_vol ** 2)
        jump_var = jump_intensity * (jump_mean ** 2 + jump_vol ** 2) if n_jumps > 1 else 0.0

        logger.info(f"Merton JD: drift={drift:.4f}, diffusion_vol={diffusion_vol:.4f}, "
                     f"lambda={jump_intensity:.2f}, jump_mean={jump_mean:.6f}, jump_vol={jump_vol:.6f}")

        return {
            'drift': drift,
            'diffusion_vol': diffusion_vol,
            'jump_intensity': jump_intensity,
            'jump_mean': jump_mean,
            'jump_vol': jump_vol,
            'compensated_drift': compensated_drift,
            'n_jumps_detected': n_jumps,
            'jump_fraction': n_jumps / n,
            'total_variance_annual': total_var,
            'diffusion_variance_share': diffusion_var / total_var if total_var > 0 else 1.0,
            'jump_variance_share': jump_var / total_var if total_var > 0 else 0.0,
            'jump_threshold_sigma': self.jump_threshold_sigma,
            'n_observations': n
        }

    def simulate_merton_paths(self, params: Dict[str, Any], n_paths: int = 1000,
                               n_steps: int = 252) -> np.ndarray:
        """
        Simulate price paths under Merton jump-diffusion model

        Args:
            params: Output from fit_merton_jump_diffusion
            n_paths: Number of simulation paths
            n_steps: Number of time steps (252 = 1 year of trading days)

        Returns:
            ndarray of shape (n_paths, n_steps + 1) with simulated price levels (S0 = 1)
        """
        dt = 1.0 / 252
        mu = params['drift']
        sigma = params['diffusion_vol']
        lam = params['jump_intensity']
        mu_j = params['jump_mean']
        sigma_j = params['jump_vol']

        # Compensator for risk-neutral pricing
        k = np.exp(mu_j + 0.5 * sigma_j ** 2) - 1 if sigma_j > 0 else mu_j

        # Initialize paths
        paths = np.ones((n_paths, n_steps + 1))

        for t in range(1, n_steps + 1):
            # Diffusion component
            z = np.random.standard_normal(n_paths)
            diffusion = (mu - lam * k - 0.5 * sigma ** 2) * dt + sigma * np.sqrt(dt) * z

            # Jump component (Poisson arrivals)
            n_jumps = np.random.poisson(lam * dt, n_paths)
            jump_sizes = np.zeros(n_paths)
            for i in range(n_paths):
                if n_jumps[i] > 0:
                    jumps = np.random.normal(mu_j, sigma_j, n_jumps[i])
                    jump_sizes[i] = np.sum(jumps)

            # Log return for this step
            log_return = diffusion + jump_sizes
            paths[:, t] = paths[:, t - 1] * np.exp(log_return)

        logger.info(f"Simulated {n_paths} Merton JD paths over {n_steps} steps")

        return paths

    def jump_risk_metrics(self, returns: pd.Series) -> Dict[str, Any]:
        """
        Identify likely jump days and estimate jump contribution to total risk

        Args:
            returns: Log return series

        Returns:
            Dictionary with jump day identification, variance decomposition, metrics
        """
        jd_params = self.fit_merton_jump_diffusion(returns)
        if 'error' in jd_params:
            return jd_params

        returns_clean = returns.dropna()
        r = returns_clean.values
        sigma_est = np.std(r)
        threshold = self.jump_threshold_sigma * sigma_est

        # Identify jump days with details
        jump_mask = np.abs(r - np.mean(r)) > threshold
        jump_indices = np.where(jump_mask)[0]

        jump_days = []
        for idx in jump_indices:
            jump_days.append({
                'date': str(returns_clean.index[idx]),
                'return': float(r[idx]),
                'z_score': float((r[idx] - np.mean(r)) / sigma_est)
            })

        # Sort by absolute return magnitude
        jump_days.sort(key=lambda x: abs(x['return']), reverse=True)

        # Tail statistics
        negative_jumps = [j for j in jump_days if j['return'] < 0]
        positive_jumps = [j for j in jump_days if j['return'] > 0]

        return {
            'jd_params': jd_params,
            'jump_days': jump_days[:20],  # Top 20 by magnitude
            'n_total_jumps': len(jump_days),
            'n_negative_jumps': len(negative_jumps),
            'n_positive_jumps': len(positive_jumps),
            'avg_negative_jump': float(np.mean([j['return'] for j in negative_jumps])) if negative_jumps else 0.0,
            'avg_positive_jump': float(np.mean([j['return'] for j in positive_jumps])) if positive_jumps else 0.0,
            'worst_jump': jump_days[0] if jump_days else None,
            'jump_contribution_to_variance': jd_params['jump_variance_share'],
            'diffusion_contribution_to_variance': jd_params['diffusion_variance_share']
        }

    # ----------------------------------------------------------------
    # 5. Tail Risk Parity
    # ----------------------------------------------------------------

    def tail_risk_parity_weights(self, returns_df: pd.DataFrame,
                                  confidence: float = 0.99) -> Dict[str, Any]:
        """
        Calculate portfolio weights so each asset contributes equally to portfolio CVaR

        Args:
            returns_df: DataFrame of asset returns
            confidence: Confidence level for CVaR calculation

        Returns:
            Dictionary with optimal weights, CVaR contributions, and diagnostics
        """
        returns_clean = returns_df.dropna()
        n, d = returns_clean.shape
        tickers = list(returns_clean.columns)

        if n < 50:
            return {'error': 'Insufficient data for tail risk parity (need >= 50)'}

        def portfolio_cvar(weights: np.ndarray) -> float:
            """Compute portfolio CVaR at given confidence level"""
            port_returns = returns_clean.values @ weights
            sorted_returns = np.sort(port_returns)
            cutoff = int((1 - confidence) * n)
            if cutoff < 1:
                cutoff = 1
            tail_losses = -sorted_returns[:cutoff]
            return float(np.mean(tail_losses))

        def cvar_contributions(weights: np.ndarray) -> np.ndarray:
            """Compute marginal CVaR contribution of each asset"""
            port_returns = returns_clean.values @ weights
            sorted_indices = np.argsort(port_returns)
            cutoff = int((1 - confidence) * n)
            if cutoff < 1:
                cutoff = 1
            tail_indices = sorted_indices[:cutoff]

            # CVaR contribution = w_i * E[r_i | portfolio in tail]
            tail_asset_returns = returns_clean.values[tail_indices, :]
            mean_tail_returns = np.mean(tail_asset_returns, axis=0)
            contributions = -weights * mean_tail_returns
            return contributions

        def objective(weights: np.ndarray) -> float:
            """Minimize variance of CVaR contributions (equal risk contribution)"""
            contributions = cvar_contributions(weights)
            total_cvar = np.sum(contributions)
            if total_cvar <= 0:
                return 1e10
            # Target: each asset contributes 1/d of total CVaR
            target = total_cvar / d
            return float(np.sum((contributions - target) ** 2))

        # Optimization
        constraints = [
            {'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0}
        ]
        bounds = [(self.min_weight, 1.0) for _ in range(d)]
        x0 = np.ones(d) / d

        result = minimize(objective, x0, method='SLSQP', bounds=bounds, constraints=constraints,
                          options={'maxiter': 500, 'ftol': 1e-12})

        if not result.success:
            logger.warning(f"Tail risk parity optimization did not converge: {result.message}")

        weights = result.x
        weights = weights / np.sum(weights)  # Normalize

        # Final metrics
        contributions = cvar_contributions(weights)
        total_cvar = portfolio_cvar(weights)

        logger.info(f"Tail risk parity: CVaR={total_cvar:.6f}, max weight={np.max(weights):.2%}")

        return {
            'weights': dict(zip(tickers, weights.tolist())),
            'cvar_contributions': dict(zip(tickers, contributions.tolist())),
            'portfolio_cvar': total_cvar,
            'confidence': confidence,
            'contribution_dispersion': float(np.std(contributions)),
            'max_contribution_ratio': float(np.max(contributions) / np.min(contributions)) if np.min(contributions) > 0 else np.inf,
            'optimization_converged': result.success,
            'n_observations': n
        }

    def tail_risk_budget(self, returns_df: pd.DataFrame, weights: np.ndarray) -> Dict[str, Any]:
        """
        Decompose portfolio tail risk by asset contribution for given weights

        Args:
            returns_df: DataFrame of asset returns
            weights: Portfolio weight array

        Returns:
            Dictionary with tail risk decomposition by asset
        """
        returns_clean = returns_df.dropna()
        n, d = returns_clean.shape
        tickers = list(returns_clean.columns)
        confidence = self.trp_confidence

        if n < 50:
            return {'error': 'Insufficient data for tail risk budgeting (need >= 50)'}

        if len(weights) != d:
            return {'error': f'Weight dimension mismatch: got {len(weights)}, expected {d}'}

        # Portfolio returns
        port_returns = returns_clean.values @ weights
        sorted_indices = np.argsort(port_returns)
        cutoff = max(1, int((1 - confidence) * n))
        tail_indices = sorted_indices[:cutoff]

        # Portfolio CVaR
        tail_port_returns = port_returns[tail_indices]
        portfolio_cvar = float(-np.mean(tail_port_returns))

        # Portfolio VaR
        portfolio_var = float(-port_returns[sorted_indices[cutoff - 1]])

        # Asset-level decomposition
        tail_asset_returns = returns_clean.values[tail_indices, :]
        mean_tail_returns = np.mean(tail_asset_returns, axis=0)

        # Marginal CVaR contributions
        marginal_cvar = -mean_tail_returns
        component_cvar = weights * marginal_cvar
        total_component = np.sum(component_cvar)

        # Percentage contributions
        pct_contributions = component_cvar / total_component if total_component > 0 else np.zeros(d)

        # Individual asset CVaR (standalone)
        standalone_cvar = {}
        for j in range(d):
            asset_returns = returns_clean.iloc[:, j].values
            sorted_asset = np.sort(asset_returns)
            asset_tail = -sorted_asset[:cutoff]
            standalone_cvar[tickers[j]] = float(np.mean(asset_tail))

        # Diversification benefit
        weighted_standalone = sum(
            weights[j] * standalone_cvar[tickers[j]] for j in range(d)
        )
        diversification_benefit = 1 - portfolio_cvar / weighted_standalone if weighted_standalone > 0 else 0.0

        logger.info(f"Tail risk budget: CVaR={portfolio_cvar:.6f}, diversification={diversification_benefit:.2%}")

        return {
            'portfolio_cvar': portfolio_cvar,
            'portfolio_var': portfolio_var,
            'confidence': confidence,
            'weights': dict(zip(tickers, weights.tolist())),
            'marginal_cvar': dict(zip(tickers, marginal_cvar.tolist())),
            'component_cvar': dict(zip(tickers, component_cvar.tolist())),
            'pct_contributions': dict(zip(tickers, pct_contributions.tolist())),
            'standalone_cvar': standalone_cvar,
            'diversification_benefit': diversification_benefit,
            'n_tail_observations': cutoff,
            'n_observations': n
        }

    # ----------------------------------------------------------------
    # Utility Methods
    # ----------------------------------------------------------------

    @staticmethod
    def _nearest_positive_definite(matrix: np.ndarray) -> np.ndarray:
        """Find nearest positive definite matrix using Higham's algorithm"""
        B = (matrix + matrix.T) / 2
        _, s, V = np.linalg.svd(B)
        H = V.T @ np.diag(s) @ V
        A2 = (B + H) / 2
        A3 = (A2 + A2.T) / 2

        if np.all(np.linalg.eigvalsh(A3) > 0):
            return A3

        spacing = np.spacing(np.linalg.norm(matrix))
        identity = np.eye(matrix.shape[0])
        k = 1
        while not np.all(np.linalg.eigvalsh(A3) > 0):
            min_eig = np.min(np.linalg.eigvalsh(A3))
            A3 += identity * (-min_eig * k ** 2 + spacing)
            k += 1

        return A3
