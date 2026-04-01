"""
Bayesian and Probabilistic Regime Switching Models

Implements advanced regime detection and parameter estimation using:
- Markov Switching Models (Hamilton Filter)
- Bayesian Online Changepoint Detection (Adams & MacKay 2007)
- Bayesian Parameter Estimation with conjugate priors
- Regime-Aware Strategy Adjustment

Provides probabilistic frameworks for identifying structural breaks,
estimating time-varying parameters, and adapting trading strategies
to current market regimes.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
import logging
from datetime import datetime
from scipy import stats
from scipy.special import gammaln, logsumexp

logger = logging.getLogger(__name__)


class BayesianRegimeSwitcher:
    """
    Bayesian regime switching and changepoint detection system
    """

    def __init__(self, config: Dict[str, Any]):
        self.config = config

        # Markov switching defaults
        self.default_n_regimes = config.get('n_regimes', 2)
        self.em_max_iter = config.get('em_max_iter', 100)
        self.em_tol = config.get('em_tol', 1e-6)

        # Changepoint detection defaults
        self.default_hazard_rate = config.get('hazard_rate', 1 / 250)
        self.changepoint_threshold = config.get('changepoint_threshold', 0.5)

        # Bayesian estimation defaults
        self.n_posterior_samples = config.get('n_posterior_samples', 10000)
        self.credible_interval = config.get('credible_interval', 0.95)

        # Risk multiplier defaults per regime
        self.default_risk_multipliers = config.get('risk_multipliers', {
            'low_vol': 1.2,
            'normal': 1.0,
            'high_vol': 0.5,
            'crisis': 0.1
        })

        # Cache for fitted models
        self._fitted_models = {}

    # ------------------------------------------------------------------ #
    #  1. Markov Switching Model (Hamilton Filter)
    # ------------------------------------------------------------------ #

    def fit_markov_switching(self, returns: pd.Series, n_regimes: int = 2) -> Dict[str, Any]:
        """
        Fit Markov-switching model to return series.

        Attempts to use statsmodels MarkovRegression first; falls back to a
        custom EM/Hamilton filter implementation when the library is unavailable
        or fails to converge.

        Args:
            returns: Series of asset returns
            n_regimes: Number of hidden regimes

        Returns:
            Dict with regime_means, regime_variances, transition_matrix,
            smoothed_probabilities, log_likelihood, n_regimes, converged
        """
        try:
            result = self._fit_markov_statsmodels(returns, n_regimes)
            logger.info(f"Markov switching model fitted via statsmodels ({n_regimes} regimes)")
            return result
        except Exception as e:
            logger.warning(f"statsmodels MarkovRegression failed ({e}); using EM fallback")

        try:
            result = self._fit_markov_em(returns, n_regimes)
            logger.info(f"Markov switching model fitted via EM fallback ({n_regimes} regimes)")
            return result
        except Exception as e:
            logger.error(f"Markov switching model fitting failed: {e}")
            return {
                'regime_means': {},
                'regime_variances': {},
                'transition_matrix': np.array([]),
                'smoothed_probabilities': pd.DataFrame(),
                'log_likelihood': np.nan,
                'n_regimes': n_regimes,
                'converged': False,
                'error': str(e)
            }

    def _fit_markov_statsmodels(self, returns: pd.Series, n_regimes: int) -> Dict[str, Any]:
        """Fit using statsmodels MarkovRegression."""
        from statsmodels.tsa.regime_switching.markov_regression import MarkovRegression

        model = MarkovRegression(
            returns.dropna(),
            k_regimes=n_regimes,
            switching_variance=True
        )
        fitted = model.fit(maxiter=self.em_max_iter, disp=False)

        regime_means = {f'regime_{i}': float(fitted.params[f'const[{i}]'])
                        for i in range(n_regimes)}
        regime_variances = {f'regime_{i}': float(fitted.params[f'sigma2[{i}]'])
                           for i in range(n_regimes)}

        transition_matrix = np.array(fitted.regime_transition)
        smoothed = pd.DataFrame(
            fitted.smoothed_marginal_probabilities,
            index=returns.dropna().index,
            columns=[f'regime_{i}' for i in range(n_regimes)]
        )

        return {
            'regime_means': regime_means,
            'regime_variances': regime_variances,
            'transition_matrix': transition_matrix,
            'smoothed_probabilities': smoothed,
            'log_likelihood': float(fitted.llf),
            'n_regimes': n_regimes,
            'converged': True
        }

    def _fit_markov_em(self, returns: pd.Series, n_regimes: int) -> Dict[str, Any]:
        """Custom EM / Hamilton filter fallback."""
        y = returns.dropna().values.astype(np.float64)
        T = len(y)

        # Initialise parameters via quantile-based heuristic
        sorted_y = np.sort(y)
        boundaries = np.linspace(0, T, n_regimes + 1, dtype=int)
        means = np.array([sorted_y[boundaries[i]:boundaries[i + 1]].mean()
                          for i in range(n_regimes)])
        variances = np.array([max(sorted_y[boundaries[i]:boundaries[i + 1]].var(), 1e-10)
                              for i in range(n_regimes)])

        # Uniform transition matrix with slight self-persistence
        transition = np.full((n_regimes, n_regimes), 0.1 / (n_regimes - 1))
        np.fill_diagonal(transition, 0.9)
        transition = transition / transition.sum(axis=1, keepdims=True)

        # Stationary distribution as initial state probs
        state_probs = np.ones(n_regimes) / n_regimes

        prev_ll = -np.inf
        converged = False

        for iteration in range(self.em_max_iter):
            # --- E-step: Hamilton filter (forward) ---
            filtered = np.zeros((T, n_regimes))
            predicted = np.zeros((T, n_regimes))
            log_likelihood = 0.0

            for t in range(T):
                if t == 0:
                    pred = state_probs.copy()
                else:
                    pred = filtered[t - 1] @ transition

                predicted[t] = pred

                # Observation densities
                densities = np.array([
                    stats.norm.pdf(y[t], loc=means[j], scale=np.sqrt(variances[j]))
                    for j in range(n_regimes)
                ])
                joint = pred * densities
                marginal = joint.sum()

                if marginal < 1e-300:
                    filtered[t] = np.ones(n_regimes) / n_regimes
                else:
                    filtered[t] = joint / marginal
                    log_likelihood += np.log(marginal)

            # --- E-step: Kim smoother (backward) ---
            smoothed = np.zeros((T, n_regimes))
            smoothed[-1] = filtered[-1]

            for t in range(T - 2, -1, -1):
                pred_next = filtered[t] @ transition
                pred_next = np.maximum(pred_next, 1e-300)
                ratio = smoothed[t + 1] / pred_next
                smoothed[t] = filtered[t] * (transition @ ratio)
                smoothed[t] = smoothed[t] / smoothed[t].sum()

            # Check convergence
            if abs(log_likelihood - prev_ll) < self.em_tol:
                converged = True
                break
            prev_ll = log_likelihood

            # --- M-step ---
            weights = smoothed.sum(axis=0)
            weights = np.maximum(weights, 1e-10)

            for j in range(n_regimes):
                w = smoothed[:, j]
                means[j] = np.sum(w * y) / weights[j]
                variances[j] = np.sum(w * (y - means[j]) ** 2) / weights[j]
                variances[j] = max(variances[j], 1e-10)

            # Update transition matrix
            for i in range(n_regimes):
                for j in range(n_regimes):
                    num = 0.0
                    for t in range(1, T):
                        pred_next = filtered[t - 1] @ transition
                        pred_next_j = max(pred_next[j], 1e-300)
                        num += smoothed[t, j] * transition[i, j] * filtered[t - 1, i] / pred_next_j
                    transition[i, j] = num
                row_sum = transition[i].sum()
                if row_sum > 0:
                    transition[i] = transition[i] / row_sum

            state_probs = smoothed[0]

        # Sort regimes by mean (ascending)
        order = np.argsort(means)
        means = means[order]
        variances = variances[order]
        transition = transition[np.ix_(order, order)]
        smoothed = smoothed[:, order]

        regime_means = {f'regime_{i}': float(means[i]) for i in range(n_regimes)}
        regime_variances = {f'regime_{i}': float(variances[i]) for i in range(n_regimes)}
        smoothed_df = pd.DataFrame(
            smoothed,
            index=returns.dropna().index,
            columns=[f'regime_{i}' for i in range(n_regimes)]
        )

        self._fitted_models['markov'] = {
            'means': means,
            'variances': variances,
            'transition': transition,
            'n_regimes': n_regimes
        }

        return {
            'regime_means': regime_means,
            'regime_variances': regime_variances,
            'transition_matrix': transition,
            'smoothed_probabilities': smoothed_df,
            'log_likelihood': float(log_likelihood),
            'n_regimes': n_regimes,
            'converged': converged
        }

    def get_regime_probabilities(self, returns: pd.Series) -> pd.DataFrame:
        """
        Compute filtered and smoothed regime probabilities for each time step.

        Args:
            returns: Series of asset returns

        Returns:
            DataFrame with filtered and smoothed probabilities per regime
        """
        result = self.fit_markov_switching(returns, self.default_n_regimes)

        if result.get('error'):
            logger.warning("Cannot compute regime probabilities due to fitting error")
            return pd.DataFrame()

        smoothed = result['smoothed_probabilities']

        # Re-run Hamilton filter for filtered (non-smoothed) probabilities
        y = returns.dropna().values.astype(np.float64)
        n_regimes = result['n_regimes']
        means = np.array([result['regime_means'][f'regime_{i}'] for i in range(n_regimes)])
        variances = np.array([result['regime_variances'][f'regime_{i}'] for i in range(n_regimes)])
        transition = result['transition_matrix']

        T = len(y)
        filtered = np.zeros((T, n_regimes))
        state_probs = np.ones(n_regimes) / n_regimes

        for t in range(T):
            pred = state_probs if t == 0 else filtered[t - 1] @ transition
            densities = np.array([
                stats.norm.pdf(y[t], loc=means[j], scale=np.sqrt(variances[j]))
                for j in range(n_regimes)
            ])
            joint = pred * densities
            marginal = joint.sum()
            filtered[t] = joint / max(marginal, 1e-300)

        filtered_df = pd.DataFrame(
            filtered,
            index=returns.dropna().index,
            columns=[f'filtered_regime_{i}' for i in range(n_regimes)]
        )

        combined = pd.concat([filtered_df, smoothed], axis=1)
        return combined

    def predict_regime(self, returns: pd.Series, horizon: int = 5) -> Dict[str, Any]:
        """
        Predict future regime probabilities using transition matrix powers.

        Args:
            returns: Series of asset returns
            horizon: Number of periods ahead to forecast

        Returns:
            Dict with current_probs, forecast_probs (per step), most_likely_regime
        """
        result = self.fit_markov_switching(returns, self.default_n_regimes)

        if result.get('error'):
            return {'error': result['error']}

        smoothed = result['smoothed_probabilities']
        current_probs = smoothed.iloc[-1].values
        transition = result['transition_matrix']
        n_regimes = result['n_regimes']

        forecast = {}
        probs = current_probs.copy()

        for h in range(1, horizon + 1):
            probs = probs @ transition
            forecast[f'step_{h}'] = {f'regime_{i}': float(probs[i])
                                     for i in range(n_regimes)}

        most_likely = int(np.argmax(probs))

        return {
            'current_probs': {f'regime_{i}': float(current_probs[i])
                              for i in range(n_regimes)},
            'forecast_probs': forecast,
            'most_likely_regime': f'regime_{most_likely}',
            'horizon': horizon,
            'transition_matrix': transition
        }

    def regime_conditional_statistics(self, returns: pd.Series) -> Dict[str, Any]:
        """
        Compute mean return, volatility, and Sharpe ratio conditional on each regime.

        Args:
            returns: Series of asset returns

        Returns:
            Dict keyed by regime with mean_return, volatility, annualized_sharpe,
            n_observations, fraction_of_time
        """
        result = self.fit_markov_switching(returns, self.default_n_regimes)

        if result.get('error'):
            return {'error': result['error']}

        smoothed = result['smoothed_probabilities']
        y = returns.dropna()
        n_regimes = result['n_regimes']

        regime_stats = {}
        for i in range(n_regimes):
            col = f'regime_{i}'
            weights = smoothed[col].values

            weighted_mean = np.average(y.values, weights=weights)
            weighted_var = np.average((y.values - weighted_mean) ** 2, weights=weights)
            weighted_vol = np.sqrt(weighted_var)

            ann_mean = weighted_mean * 252
            ann_vol = weighted_vol * np.sqrt(252)
            sharpe = ann_mean / ann_vol if ann_vol > 0 else 0.0

            regime_stats[col] = {
                'mean_return': float(weighted_mean),
                'volatility': float(weighted_vol),
                'annualized_mean': float(ann_mean),
                'annualized_volatility': float(ann_vol),
                'annualized_sharpe': float(sharpe),
                'avg_probability': float(weights.mean()),
                'fraction_of_time': float((weights > 0.5).mean())
            }

        return regime_stats

    # ------------------------------------------------------------------ #
    #  2. Bayesian Online Changepoint Detection (Adams & MacKay 2007)
    # ------------------------------------------------------------------ #

    def detect_changepoints(self, series: pd.Series,
                            hazard_rate: float = None) -> pd.DataFrame:
        """
        Online changepoint detection using the Adams & MacKay (2007) algorithm.

        Uses a normal-inverse-gamma conjugate prior so that the predictive
        distribution of each run-length segment is a Student-t.

        Args:
            series: Observed time series (e.g. returns)
            hazard_rate: Prior probability of a changepoint at each step (1/expected_run_length)

        Returns:
            DataFrame with columns for run-length probabilities, most probable
            run length, and changepoint probability at each time step
        """
        if hazard_rate is None:
            hazard_rate = self.default_hazard_rate

        x = series.dropna().values.astype(np.float64)
        T = len(x)

        # Hyperparameters for normal-inverse-gamma prior
        mu0 = x[:min(20, T)].mean() if T > 0 else 0.0
        kappa0 = 1.0
        alpha0 = 1.0
        beta0 = max(x[:min(20, T)].var(), 1e-10) if T > 1 else 1e-4

        # Sufficient statistics arrays (indexed by run length r = 0 .. T)
        mu_params = np.zeros(T + 1)
        kappa_params = np.zeros(T + 1)
        alpha_params = np.zeros(T + 1)
        beta_params = np.zeros(T + 1)

        mu_params[0] = mu0
        kappa_params[0] = kappa0
        alpha_params[0] = alpha0
        beta_params[0] = beta0

        # Log run-length posterior: log R(r, t)
        # At each step we only keep the current column
        log_R = np.full(T + 1, -np.inf)
        log_R[0] = 0.0  # log(1)

        log_hazard = np.log(hazard_rate)
        log_1m_hazard = np.log(1.0 - hazard_rate)

        changepoint_prob = np.zeros(T)
        max_run_length = np.zeros(T, dtype=int)
        run_length_probs_list = []

        for t in range(T):
            obs = x[t]

            # Predictive probabilities under each run length (Student-t)
            active = np.where(log_R > -np.inf)[0]
            log_pred = np.full(T + 1, -np.inf)

            for r in active:
                nu = 2.0 * alpha_params[r]
                mu_r = mu_params[r]
                scale = np.sqrt(beta_params[r] * (kappa_params[r] + 1.0) /
                                (alpha_params[r] * kappa_params[r]))
                scale = max(scale, 1e-15)
                log_pred[r] = stats.t.logpdf(obs, df=max(nu, 1e-10), loc=mu_r, scale=scale)

            # Growth probabilities (run length increases)
            log_R_new = np.full(T + 2, -np.inf)

            # Changepoint probability: sum over all run lengths
            log_cp_vals = log_R[active] + log_pred[active] + log_hazard
            log_cp = logsumexp(log_cp_vals) if len(log_cp_vals) > 0 else -np.inf

            # Growth: shift run lengths up by 1
            for r in active:
                log_R_new[r + 1] = log_R[r] + log_pred[r] + log_1m_hazard

            # Changepoint resets run length to 0
            log_R_new[0] = log_cp

            # Normalise
            active_new = np.where(log_R_new > -np.inf)[0]
            if len(active_new) > 0:
                log_evidence = logsumexp(log_R_new[active_new])
                log_R_new[active_new] -= log_evidence

            log_R = log_R_new[:T + 1]

            # Prune low-probability run lengths to prevent O(T^2) blowup
            active_new = np.where(log_R > -30)[0]  # drop run lengths with negligible probability
            pruned = np.full_like(log_R, -np.inf)
            pruned[active_new] = log_R[active_new]
            log_R = pruned

            # Update sufficient statistics for all active run lengths
            mu_new = np.zeros(T + 1)
            kappa_new = np.zeros(T + 1)
            alpha_new = np.zeros(T + 1)
            beta_new = np.zeros(T + 1)

            # Run length 0 resets to prior
            mu_new[0] = mu0
            kappa_new[0] = kappa0
            alpha_new[0] = alpha0
            beta_new[0] = beta0

            active_grown = np.where(log_R > -np.inf)[0]
            for r in active_grown:
                if r == 0:
                    continue
                r_prev = r - 1
                kp = kappa_params[r_prev]
                mu_new[r] = (kp * mu_params[r_prev] + obs) / (kp + 1.0)
                kappa_new[r] = kp + 1.0
                alpha_new[r] = alpha_params[r_prev] + 0.5
                beta_new[r] = (beta_params[r_prev] +
                               0.5 * kp * (obs - mu_params[r_prev]) ** 2 / (kp + 1.0))

            mu_params = mu_new
            kappa_params = kappa_new
            alpha_params = alpha_new
            beta_params = beta_new

            # Record outputs
            cp_prob = np.exp(log_R[0]) if log_R[0] > -np.inf else 0.0
            changepoint_prob[t] = float(cp_prob)

            active_final = np.where(log_R > -np.inf)[0]
            if len(active_final) > 0:
                probs = np.exp(log_R[active_final])
                max_run_length[t] = int(active_final[np.argmax(probs)])
            else:
                max_run_length[t] = 0

            run_length_probs_list.append(changepoint_prob[t])

        index = series.dropna().index
        result_df = pd.DataFrame({
            'changepoint_probability': changepoint_prob,
            'max_run_length': max_run_length
        }, index=index)

        logger.info(f"Changepoint detection complete: {(changepoint_prob > self.changepoint_threshold).sum()} "
                    f"changepoints detected (threshold={self.changepoint_threshold})")

        return result_df

    def get_changepoint_dates(self, series: pd.Series,
                              threshold: float = None) -> List[datetime]:
        """
        Return dates where changepoint probability exceeds threshold.

        Args:
            series: Observed time series
            threshold: Probability threshold for declaring a changepoint

        Returns:
            List of datetime objects where changepoints were detected
        """
        if threshold is None:
            threshold = self.changepoint_threshold

        cp_df = self.detect_changepoints(series)

        if cp_df.empty:
            return []

        mask = cp_df['changepoint_probability'] > threshold
        dates = cp_df.index[mask].tolist()

        # Convert to datetime if not already
        changepoint_dates = []
        for d in dates:
            if isinstance(d, datetime):
                changepoint_dates.append(d)
            elif isinstance(d, pd.Timestamp):
                changepoint_dates.append(d.to_pydatetime())
            else:
                changepoint_dates.append(d)

        logger.info(f"Found {len(changepoint_dates)} changepoints above threshold {threshold}")
        return changepoint_dates

    def adaptive_parameters(self, series: pd.Series) -> Dict[str, Any]:
        """
        Estimate time-varying mean and variance using changepoint posterior.

        Segments the series at detected changepoints and estimates local
        parameters for each segment.

        Args:
            series: Observed time series

        Returns:
            Dict with time_varying_mean, time_varying_variance (as Series),
            segments list, and current_params
        """
        cp_df = self.detect_changepoints(series)

        if cp_df.empty:
            return {'error': 'No data for adaptive parameter estimation'}

        x = series.dropna()
        index = x.index

        # Identify segment boundaries from changepoints
        cp_mask = cp_df['changepoint_probability'] > self.changepoint_threshold
        cp_indices = np.where(cp_mask.values)[0]

        # Build segments
        boundaries = [0] + cp_indices.tolist() + [len(x)]
        boundaries = sorted(set(boundaries))

        time_varying_mean = np.zeros(len(x))
        time_varying_var = np.zeros(len(x))
        segments = []

        for k in range(len(boundaries) - 1):
            start = boundaries[k]
            end = boundaries[k + 1]
            segment_data = x.values[start:end]

            if len(segment_data) == 0:
                continue

            seg_mean = float(np.mean(segment_data))
            seg_var = float(np.var(segment_data, ddof=1)) if len(segment_data) > 1 else 1e-10

            time_varying_mean[start:end] = seg_mean
            time_varying_var[start:end] = seg_var

            segments.append({
                'start_idx': int(start),
                'end_idx': int(end),
                'start_date': str(index[start]),
                'end_date': str(index[end - 1]),
                'mean': seg_mean,
                'variance': seg_var,
                'n_obs': int(end - start)
            })

        current_segment = segments[-1] if segments else {}

        return {
            'time_varying_mean': pd.Series(time_varying_mean, index=index),
            'time_varying_variance': pd.Series(time_varying_var, index=index),
            'segments': segments,
            'n_changepoints': len(cp_indices),
            'current_params': {
                'mean': current_segment.get('mean', np.nan),
                'variance': current_segment.get('variance', np.nan)
            }
        }

    # ------------------------------------------------------------------ #
    #  3. Bayesian Parameter Estimation
    # ------------------------------------------------------------------ #

    def bayesian_sharpe_ratio(self, returns: pd.Series,
                              prior_mean: float = 0.0,
                              prior_std: float = 0.5) -> Dict[str, Any]:
        """
        Posterior distribution of the annualised Sharpe ratio.

        Uses a normal prior on the mean return and known-variance likelihood
        to derive the posterior, then transforms samples to Sharpe space.

        Args:
            returns: Series of asset returns
            prior_mean: Prior mean for the Sharpe ratio
            prior_std: Prior standard deviation for the Sharpe ratio

        Returns:
            Dict with posterior_mean, posterior_std, credible_interval,
            probability_positive, posterior_samples
        """
        y = returns.dropna().values.astype(np.float64)
        n = len(y)

        if n < 5:
            return {'error': 'Insufficient data for Bayesian Sharpe estimation'}

        sample_mean = y.mean()
        sample_std = y.std(ddof=1)

        if sample_std < 1e-15:
            return {'error': 'Zero variance in returns'}

        # Observed annualised Sharpe
        obs_sharpe = (sample_mean * 252) / (sample_std * np.sqrt(252))

        # Standard error of the Sharpe ratio (Lo, 2002 approximation)
        se_sharpe = np.sqrt((1.0 + 0.5 * obs_sharpe ** 2) / n) * np.sqrt(252)

        # Bayesian update: normal prior * normal likelihood -> normal posterior
        prior_precision = 1.0 / (prior_std ** 2)
        likelihood_precision = 1.0 / (se_sharpe ** 2)

        posterior_precision = prior_precision + likelihood_precision
        posterior_var = 1.0 / posterior_precision
        posterior_mean = posterior_var * (prior_precision * prior_mean +
                                         likelihood_precision * obs_sharpe)
        posterior_std = np.sqrt(posterior_var)

        # Credible interval
        alpha = 1.0 - self.credible_interval
        ci_lower = stats.norm.ppf(alpha / 2, loc=posterior_mean, scale=posterior_std)
        ci_upper = stats.norm.ppf(1 - alpha / 2, loc=posterior_mean, scale=posterior_std)

        # Posterior samples
        samples = np.random.normal(posterior_mean, posterior_std, self.n_posterior_samples)
        prob_positive = float((samples > 0).mean())

        return {
            'posterior_mean': float(posterior_mean),
            'posterior_std': float(posterior_std),
            'credible_interval': (float(ci_lower), float(ci_upper)),
            'credible_level': self.credible_interval,
            'probability_positive': prob_positive,
            'observed_sharpe': float(obs_sharpe),
            'prior_mean': prior_mean,
            'prior_std': prior_std,
            'n_observations': n,
            'posterior_samples': samples
        }

    def bayesian_volatility_estimate(self, returns: pd.Series) -> Dict[str, Any]:
        """
        Posterior volatility estimate using inverse-gamma conjugate prior.

        The variance of normally distributed returns with known mean has an
        inverse-gamma posterior when combined with an inverse-gamma prior.

        Args:
            returns: Series of asset returns

        Returns:
            Dict with posterior_mean_vol, credible_interval, posterior_alpha,
            posterior_beta, annualized values
        """
        y = returns.dropna().values.astype(np.float64)
        n = len(y)

        if n < 5:
            return {'error': 'Insufficient data for Bayesian volatility estimation'}

        sample_mean = y.mean()
        ss = np.sum((y - sample_mean) ** 2)

        # Inverse-gamma prior (weakly informative)
        alpha0 = 2.0
        beta0 = max(np.var(y[:min(20, n)]) * (alpha0 - 1), 1e-10)

        # Posterior parameters
        alpha_post = alpha0 + n / 2.0
        beta_post = beta0 + ss / 2.0

        # Posterior mean and mode of variance
        posterior_mean_var = beta_post / (alpha_post - 1.0) if alpha_post > 1 else beta_post / alpha_post
        posterior_mode_var = beta_post / (alpha_post + 1.0)

        # Credible interval for variance via inverse-gamma quantiles
        alpha_ci = 1.0 - self.credible_interval
        var_lower = stats.invgamma.ppf(alpha_ci / 2, a=alpha_post, scale=beta_post)
        var_upper = stats.invgamma.ppf(1 - alpha_ci / 2, a=alpha_post, scale=beta_post)

        # Convert to volatility (daily and annualised)
        vol_mean = np.sqrt(posterior_mean_var)
        vol_lower = np.sqrt(var_lower)
        vol_upper = np.sqrt(var_upper)

        ann_factor = np.sqrt(252)

        return {
            'posterior_mean_variance': float(posterior_mean_var),
            'posterior_mode_variance': float(posterior_mode_var),
            'posterior_mean_volatility': float(vol_mean),
            'annualized_volatility': float(vol_mean * ann_factor),
            'credible_interval_variance': (float(var_lower), float(var_upper)),
            'credible_interval_volatility': (float(vol_lower), float(vol_upper)),
            'annualized_credible_interval': (float(vol_lower * ann_factor),
                                             float(vol_upper * ann_factor)),
            'credible_level': self.credible_interval,
            'posterior_alpha': float(alpha_post),
            'posterior_beta': float(beta_post),
            'sample_volatility': float(np.std(y, ddof=1)),
            'n_observations': n
        }

    def bayesian_correlation(self, returns_a: pd.Series,
                             returns_b: pd.Series) -> Dict[str, Any]:
        """
        Posterior distribution of the correlation coefficient.

        Uses Fisher z-transform to map correlation to an approximately normal
        space, applies a flat prior, and returns posterior credible intervals.

        Args:
            returns_a: Series of returns for asset A
            returns_b: Series of returns for asset B

        Returns:
            Dict with posterior_mean, credible_interval, probability_positive,
            sample_correlation
        """
        a = returns_a.dropna()
        b = returns_b.dropna()

        # Align
        common_idx = a.index.intersection(b.index)
        a = a.loc[common_idx].values.astype(np.float64)
        b = b.loc[common_idx].values.astype(np.float64)
        n = len(a)

        if n < 5:
            return {'error': 'Insufficient overlapping data for Bayesian correlation'}

        # Sample correlation
        sample_corr = float(np.corrcoef(a, b)[0, 1])

        # Fisher z-transform
        z = np.arctanh(np.clip(sample_corr, -0.9999, 0.9999))
        se_z = 1.0 / np.sqrt(n - 3) if n > 3 else 1.0

        # Posterior in z-space (flat prior -> posterior is N(z, se^2))
        alpha_ci = 1.0 - self.credible_interval
        z_lower = z - stats.norm.ppf(1 - alpha_ci / 2) * se_z
        z_upper = z + stats.norm.ppf(1 - alpha_ci / 2) * se_z

        # Transform back
        corr_lower = float(np.tanh(z_lower))
        corr_upper = float(np.tanh(z_upper))

        # Posterior samples
        z_samples = np.random.normal(z, se_z, self.n_posterior_samples)
        corr_samples = np.tanh(z_samples)

        posterior_mean = float(np.mean(corr_samples))
        posterior_std = float(np.std(corr_samples))
        prob_positive = float((corr_samples > 0).mean())

        return {
            'posterior_mean': posterior_mean,
            'posterior_std': posterior_std,
            'sample_correlation': sample_corr,
            'credible_interval': (corr_lower, corr_upper),
            'credible_level': self.credible_interval,
            'probability_positive': prob_positive,
            'n_observations': n,
            'posterior_samples': corr_samples
        }

    # ------------------------------------------------------------------ #
    #  4. Regime-Aware Strategy Adjustment
    # ------------------------------------------------------------------ #

    def regime_adjusted_position_size(self, base_size: float,
                                     regime_probs: Dict[str, float],
                                     regime_risk_multipliers: Dict[str, float] = None) -> float:
        """
        Scale a base position size by the current regime probability mix.

        Args:
            base_size: Baseline position size (e.g. number of shares or dollar amount)
            regime_probs: Mapping of regime name to probability (should sum to ~1)
            regime_risk_multipliers: Mapping of regime name to scaling factor.
                                     Defaults to self.default_risk_multipliers.

        Returns:
            Adjusted position size (float)
        """
        if regime_risk_multipliers is None:
            regime_risk_multipliers = self.default_risk_multipliers

        weighted_multiplier = 0.0
        total_weight = 0.0

        for regime, prob in regime_probs.items():
            multiplier = regime_risk_multipliers.get(regime, 1.0)
            weighted_multiplier += prob * multiplier
            total_weight += prob

        if total_weight > 0:
            weighted_multiplier /= total_weight

        adjusted = base_size * weighted_multiplier

        logger.info(f"Position size adjusted: {base_size:.2f} -> {adjusted:.2f} "
                    f"(multiplier={weighted_multiplier:.3f})")
        return float(adjusted)

    def regime_transition_alert(self, transition_matrix: np.ndarray,
                                current_probs: np.ndarray) -> Dict[str, Any]:
        """
        Generate alerts when regime switching probability is elevated.

        Computes one-step-ahead regime probabilities and flags large
        expected shifts.

        Args:
            transition_matrix: K x K transition probability matrix
            current_probs: Current regime probability vector (length K)

        Returns:
            Dict with next_step_probs, most_likely_transition, alert_level,
            transition_entropy
        """
        current_probs = np.asarray(current_probs, dtype=np.float64)
        transition_matrix = np.asarray(transition_matrix, dtype=np.float64)

        n_regimes = len(current_probs)
        next_probs = current_probs @ transition_matrix

        # Current most likely regime
        current_regime = int(np.argmax(current_probs))

        # Probability of staying in current regime
        stay_prob = transition_matrix[current_regime, current_regime]
        switch_prob = 1.0 - stay_prob

        # Transition entropy (higher = more uncertain about next regime)
        row = transition_matrix[current_regime]
        row_clipped = np.clip(row, 1e-15, 1.0)
        entropy = -np.sum(row_clipped * np.log(row_clipped))
        max_entropy = np.log(n_regimes) if n_regimes > 1 else 1.0
        normalised_entropy = entropy / max_entropy if max_entropy > 0 else 0.0

        # Determine alert level
        if switch_prob > 0.5:
            alert_level = 'high'
        elif switch_prob > 0.3:
            alert_level = 'medium'
        elif switch_prob > 0.15:
            alert_level = 'low'
        else:
            alert_level = 'none'

        # Most likely transition target (excluding self)
        transition_row = transition_matrix[current_regime].copy()
        transition_row[current_regime] = 0.0
        most_likely_target = int(np.argmax(transition_row))

        alert = {
            'current_regime': f'regime_{current_regime}',
            'current_regime_probability': float(current_probs[current_regime]),
            'next_step_probs': {f'regime_{i}': float(next_probs[i])
                                for i in range(n_regimes)},
            'stay_probability': float(stay_prob),
            'switch_probability': float(switch_prob),
            'most_likely_transition_target': f'regime_{most_likely_target}',
            'transition_target_probability': float(transition_row[most_likely_target]),
            'transition_entropy': float(normalised_entropy),
            'alert_level': alert_level
        }

        if alert_level in ('high', 'medium'):
            logger.warning(f"Regime transition alert [{alert_level}]: "
                           f"P(switch)={switch_prob:.2%} from regime_{current_regime} "
                           f"-> regime_{most_likely_target}")

        return alert

    def optimal_regime_strategy(self, returns: pd.Series,
                                strategy_returns: Dict[str, pd.Series]) -> Dict[str, Any]:
        """
        Find optimal regime-conditional allocation across multiple strategies.

        For each regime, computes each strategy's Sharpe ratio weighted by
        regime membership probabilities, then recommends an allocation.

        Args:
            returns: Market return series used for regime detection
            strategy_returns: Mapping of strategy name to its return series

        Returns:
            Dict with regime_allocations (per-regime optimal strategy weights),
            current_allocation, and strategy performance per regime
        """
        result = self.fit_markov_switching(returns, self.default_n_regimes)

        if result.get('error'):
            return {'error': result['error']}

        smoothed = result['smoothed_probabilities']
        n_regimes = result['n_regimes']
        strategy_names = list(strategy_returns.keys())

        # Align all strategy returns with smoothed probabilities
        common_idx = smoothed.index
        for name in strategy_names:
            common_idx = common_idx.intersection(strategy_returns[name].dropna().index)

        if len(common_idx) < 10:
            return {'error': 'Insufficient overlapping data between returns and strategies'}

        smoothed_aligned = smoothed.loc[common_idx]

        # Compute regime-conditional Sharpe for each strategy
        regime_performance = {}
        regime_allocations = {}

        for i in range(n_regimes):
            regime_key = f'regime_{i}'
            weights = smoothed_aligned[regime_key].values
            perf = {}

            for name in strategy_names:
                strat_ret = strategy_returns[name].loc[common_idx].values.astype(np.float64)
                w_mean = np.average(strat_ret, weights=weights)
                w_var = np.average((strat_ret - w_mean) ** 2, weights=weights)
                w_vol = np.sqrt(w_var)

                ann_mean = w_mean * 252
                ann_vol = w_vol * np.sqrt(252)
                sharpe = ann_mean / ann_vol if ann_vol > 0 else 0.0

                perf[name] = {
                    'mean_return': float(w_mean),
                    'volatility': float(w_vol),
                    'annualized_sharpe': float(sharpe)
                }

            regime_performance[regime_key] = perf

            # Allocate proportionally to positive Sharpe ratios
            sharpes = {name: max(perf[name]['annualized_sharpe'], 0.0) for name in strategy_names}
            total_sharpe = sum(sharpes.values())

            if total_sharpe > 0:
                allocation = {name: sharpes[name] / total_sharpe for name in strategy_names}
            else:
                # Equal weight fallback
                allocation = {name: 1.0 / len(strategy_names) for name in strategy_names}

            regime_allocations[regime_key] = allocation

        # Current allocation based on latest regime probabilities
        current_regime_probs = smoothed_aligned.iloc[-1].values
        current_allocation = {name: 0.0 for name in strategy_names}

        for i in range(n_regimes):
            regime_key = f'regime_{i}'
            prob = current_regime_probs[i]
            for name in strategy_names:
                current_allocation[name] += prob * regime_allocations[regime_key][name]

        return {
            'regime_allocations': regime_allocations,
            'current_allocation': current_allocation,
            'current_regime_probs': {f'regime_{i}': float(current_regime_probs[i])
                                     for i in range(n_regimes)},
            'regime_performance': regime_performance,
            'n_strategies': len(strategy_names),
            'n_regimes': n_regimes
        }
