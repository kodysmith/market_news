"""
Advanced Portfolio Optimization for AI Quant Trading System

Implements:
- Black-Litterman model (equilibrium returns, investor views, posterior optimization)
- Hierarchical Risk Parity (Lopez de Prado clustering-based allocation)
- Risk budgeting and risk parity optimization
- Robust optimization (minimax, resampled frontier, minimum CVaR)
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Tuple, Union
import logging
from scipy.optimize import minimize
from scipy.cluster.hierarchy import linkage, leaves_list
from scipy.spatial.distance import squareform
import warnings

logger = logging.getLogger(__name__)


class AdvancedPortfolioOptimizer:
    """
    Advanced portfolio optimization combining Black-Litterman, Hierarchical Risk Parity,
    risk budgeting, and robust optimization techniques
    """

    def __init__(self, config: Dict[str, Any]):
        self.config = config

        # Black-Litterman parameters
        self.default_risk_aversion = config.get('risk_aversion', 2.5)
        self.default_tau = config.get('tau', 0.05)

        # HRP parameters
        self.linkage_method = config.get('linkage_method', 'single')

        # Optimization parameters
        self.max_weight = config.get('max_weight', 1.0)
        self.min_weight = config.get('min_weight', 0.0)
        self.risk_free_rate = config.get('risk_free_rate', 0.0)

    # -------------------------------------------------------------------------
    # Black-Litterman Model
    # -------------------------------------------------------------------------

    def calculate_equilibrium_returns(self, market_weights: np.ndarray, cov_matrix: np.ndarray,
                                     risk_aversion: float = 2.5) -> np.ndarray:
        """
        Calculate implied equilibrium returns from market capitalisation weights

        Uses the reverse-optimization formula: pi = delta * Sigma * w_mkt
        where delta is the risk aversion coefficient, Sigma is the covariance
        matrix, and w_mkt are the market capitalisation weights.

        Args:
            market_weights: Market capitalisation weight vector
            cov_matrix: Asset covariance matrix
            risk_aversion: Risk aversion coefficient (default 2.5)

        Returns:
            Implied equilibrium excess return vector
        """

        pi = risk_aversion * cov_matrix @ market_weights
        logger.debug("Calculated equilibrium returns for %d assets", len(market_weights))
        return pi

    def incorporate_views(self, equilibrium_returns: np.ndarray, cov_matrix: np.ndarray,
                         P: np.ndarray, Q: np.ndarray, omega: np.ndarray = None,
                         tau: float = 0.05) -> Dict[str, Any]:
        """
        Incorporate investor views into equilibrium returns using Black-Litterman

        Combines the prior (equilibrium) distribution with investor views to
        produce a posterior distribution of expected returns.

        Args:
            equilibrium_returns: Prior equilibrium returns (pi)
            cov_matrix: Asset covariance matrix (Sigma)
            P: View picking matrix (K x N) identifying assets in each view
            Q: View return vector (K x 1) with expected returns for each view
            omega: View uncertainty matrix (K x K). If None, derived from
                   P, tau, and Sigma using the proportional-to-variance method
            tau: Scalar uncertainty of the prior (default 0.05)

        Returns:
            Dictionary with posterior_returns, posterior_covariance, and diagnostics
        """

        n_assets = len(equilibrium_returns)
        tau_sigma = tau * cov_matrix

        # Default omega: proportional to the variance implied by the views
        if omega is None:
            omega = np.diag(np.diag(P @ tau_sigma @ P.T))

        # Posterior mean: combined formula
        # mu_BL = [(tau * Sigma)^-1 + P' Omega^-1 P]^-1
        #         * [(tau * Sigma)^-1 pi + P' Omega^-1 Q]
        tau_sigma_inv = np.linalg.inv(tau_sigma)
        omega_inv = np.linalg.inv(omega)

        posterior_precision = tau_sigma_inv + P.T @ omega_inv @ P
        posterior_cov = np.linalg.inv(posterior_precision)
        posterior_returns = posterior_cov @ (tau_sigma_inv @ equilibrium_returns + P.T @ omega_inv @ Q)

        # Posterior covariance of returns (for use in portfolio optimization)
        # M = Sigma + [(tau * Sigma)^-1 + P' Omega^-1 P]^-1
        posterior_total_cov = cov_matrix + posterior_cov

        logger.info("Incorporated %d views into equilibrium model for %d assets",
                     P.shape[0], n_assets)

        return {
            'posterior_returns': posterior_returns,
            'posterior_covariance': posterior_total_cov,
            'posterior_precision_cov': posterior_cov,
            'equilibrium_returns': equilibrium_returns,
            'view_returns': Q,
            'n_views': P.shape[0],
            'n_assets': n_assets,
            'tau': tau
        }

    def optimize_bl_portfolio(self, market_weights: np.ndarray, cov_matrix: np.ndarray,
                              views: List[Dict], asset_names: List[str]) -> Dict[str, Any]:
        """
        Full Black-Litterman workflow: equilibrium -> views -> posterior -> optimize

        Args:
            market_weights: Market capitalisation weights
            cov_matrix: Asset covariance matrix
            views: List of view dicts with keys:
                   'assets' (list of asset names), 'weights' (list of floats),
                   'return' (float expected return), 'confidence' (float 0-1)
            asset_names: List of asset name strings matching matrix columns

        Returns:
            Optimised portfolio weights, expected return, risk, and diagnostics
        """

        n_assets = len(asset_names)

        # Step 1: equilibrium returns
        pi = self.calculate_equilibrium_returns(market_weights, cov_matrix,
                                                self.default_risk_aversion)

        # Step 2: build view matrices
        n_views = len(views)
        P = np.zeros((n_views, n_assets))
        Q = np.zeros(n_views)
        confidences = np.zeros(n_views)

        for k, view in enumerate(views):
            Q[k] = view['return']
            confidences[k] = view.get('confidence', 0.5)
            for asset, weight in zip(view['assets'], view['weights']):
                if asset in asset_names:
                    idx = asset_names.index(asset)
                    P[k, idx] = weight

        # Omega scaled by inverse confidence (higher confidence -> lower uncertainty)
        tau_sigma = self.default_tau * cov_matrix
        base_omega = np.diag(np.diag(P @ tau_sigma @ P.T))
        # Scale uncertainty inversely with confidence
        confidence_scale = np.diag(1.0 / np.clip(confidences, 0.01, 1.0))
        omega = confidence_scale @ base_omega

        # Step 3: posterior
        bl_result = self.incorporate_views(pi, cov_matrix, P, Q, omega, self.default_tau)
        posterior_returns = bl_result['posterior_returns']
        posterior_cov = bl_result['posterior_covariance']

        # Step 4: mean-variance optimisation on posterior
        def neg_utility(w):
            ret = w @ posterior_returns
            risk = np.sqrt(w @ posterior_cov @ w)
            return -(ret - 0.5 * self.default_risk_aversion * (w @ posterior_cov @ w))

        constraints = [{'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0}]
        bounds = [(self.min_weight, self.max_weight) for _ in range(n_assets)]
        x0 = market_weights.copy()

        result = minimize(neg_utility, x0, method='SLSQP',
                          bounds=bounds, constraints=constraints)

        if not result.success:
            logger.warning("BL optimisation did not converge: %s", result.message)

        optimal_weights = result.x
        expected_return = float(optimal_weights @ posterior_returns)
        expected_risk = float(np.sqrt(optimal_weights @ posterior_cov @ optimal_weights))

        logger.info("BL portfolio optimised: return=%.4f, risk=%.4f", expected_return, expected_risk)

        return {
            'weights': dict(zip(asset_names, optimal_weights)),
            'expected_return': expected_return,
            'expected_risk': expected_risk,
            'sharpe_ratio': (expected_return - self.risk_free_rate) / expected_risk if expected_risk > 0 else 0.0,
            'equilibrium_returns': dict(zip(asset_names, pi)),
            'posterior_returns': dict(zip(asset_names, posterior_returns)),
            'views_applied': n_views,
            'optimization_success': result.success,
            'method': 'black_litterman'
        }

    # -------------------------------------------------------------------------
    # Hierarchical Risk Parity (Lopez de Prado)
    # -------------------------------------------------------------------------

    def hrp_portfolio(self, returns_df: pd.DataFrame) -> Dict[str, Any]:
        """
        Hierarchical Risk Parity allocation (Lopez de Prado)

        Pipeline: correlation distance -> hierarchical clustering ->
        quasi-diagonalisation -> recursive bisection allocation.

        Args:
            returns_df: DataFrame of asset returns (columns = assets)

        Returns:
            Portfolio weights, expected return, risk, and clustering diagnostics
        """

        if returns_df.empty or returns_df.shape[1] < 2:
            return {'error': 'Need at least 2 assets with return history'}

        corr_matrix = returns_df.corr()
        cov_matrix = returns_df.cov()
        asset_names = list(returns_df.columns)

        # Step 1: hierarchical clustering on correlation distance
        link = self._cluster_assets(corr_matrix)

        # Step 2: quasi-diagonalise
        sorted_order = self._quasi_diagonalize(link, len(asset_names))

        # Step 3: recursive bisection
        weights_series = self._recursive_bisection(cov_matrix, sorted_order)

        # Reindex to original column order
        weights_series = weights_series.reindex(cov_matrix.columns).fillna(0.0)

        weights_arr = weights_series.values
        expected_return = float(weights_arr @ returns_df.mean().values * 252)
        expected_risk = float(np.sqrt(weights_arr @ (cov_matrix.values * 252) @ weights_arr))

        logger.info("HRP portfolio built for %d assets: risk=%.4f", len(asset_names), expected_risk)

        return {
            'weights': weights_series.to_dict(),
            'expected_return': expected_return,
            'expected_risk': expected_risk,
            'sharpe_ratio': (expected_return - self.risk_free_rate) / expected_risk if expected_risk > 0 else 0.0,
            'sorted_order': [asset_names[i] for i in sorted_order],
            'n_assets': len(asset_names),
            'method': 'hierarchical_risk_parity'
        }

    def _cluster_assets(self, corr_matrix: pd.DataFrame) -> Any:
        """
        Hierarchical clustering on correlation distance matrix

        Converts correlation to distance: d = sqrt(0.5 * (1 - rho))
        then applies agglomerative clustering.

        Args:
            corr_matrix: Correlation matrix as DataFrame

        Returns:
            Linkage matrix from scipy hierarchical clustering
        """

        distance = np.sqrt(0.5 * (1.0 - corr_matrix.values))
        np.fill_diagonal(distance, 0.0)
        condensed = squareform(distance, checks=False)
        link = linkage(condensed, method=self.linkage_method)
        return link

    def _quasi_diagonalize(self, link: np.ndarray, n_assets: int) -> List[int]:
        """
        Reorder assets so that correlated assets are adjacent (quasi-diagonal)

        Args:
            link: Linkage matrix from hierarchical clustering
            n_assets: Number of assets

        Returns:
            List of asset indices in quasi-diagonalised order
        """

        return list(leaves_list(link).astype(int))

    def _recursive_bisection(self, cov_matrix: pd.DataFrame,
                             sorted_order: List[int]) -> pd.Series:
        """
        Recursive bisection allocation by inverse cluster variance

        Splits the sorted asset list in half repeatedly, allocating weight
        proportional to inverse variance of each sub-cluster.

        Args:
            cov_matrix: Covariance matrix as DataFrame
            sorted_order: Quasi-diagonalised asset index order

        Returns:
            pd.Series of portfolio weights indexed by asset name
        """

        weights = pd.Series(1.0, index=cov_matrix.columns)
        cluster_items = [sorted_order]

        while cluster_items:
            next_level = []
            for subset in cluster_items:
                if len(subset) <= 1:
                    continue

                mid = len(subset) // 2
                left = subset[:mid]
                right = subset[mid:]

                # Cluster variance using inverse-variance weighting within each half
                left_var = self._cluster_variance(cov_matrix, left)
                right_var = self._cluster_variance(cov_matrix, right)

                # Allocate inversely proportional to variance
                total_inv_var = 1.0 / left_var + 1.0 / right_var
                alpha_left = (1.0 / left_var) / total_inv_var
                alpha_right = 1.0 - alpha_left

                left_names = [cov_matrix.columns[i] for i in left]
                right_names = [cov_matrix.columns[i] for i in right]

                weights[left_names] *= alpha_left
                weights[right_names] *= alpha_right

                if len(left) > 1:
                    next_level.append(left)
                if len(right) > 1:
                    next_level.append(right)

            cluster_items = next_level

        return weights

    def _cluster_variance(self, cov_matrix: pd.DataFrame, indices: List[int]) -> float:
        """
        Calculate variance of a cluster using inverse-variance weighted portfolio

        Args:
            cov_matrix: Full covariance matrix
            indices: Indices of assets in the cluster

        Returns:
            Cluster variance (scalar)
        """

        sub_cov = cov_matrix.iloc[indices, indices].values
        inv_diag = 1.0 / np.diag(sub_cov)
        w = inv_diag / inv_diag.sum()
        return float(w @ sub_cov @ w)

    # -------------------------------------------------------------------------
    # Risk Budgeting
    # -------------------------------------------------------------------------

    def risk_parity_portfolio(self, cov_matrix: np.ndarray,
                              risk_budget: np.ndarray = None) -> Dict[str, Any]:
        """
        Risk parity (or custom risk budget) portfolio allocation

        Finds weights such that each asset's risk contribution matches the
        target risk budget. Equal risk contribution when budget is uniform.

        Args:
            cov_matrix: Asset covariance matrix (N x N)
            risk_budget: Target risk budget per asset (sums to 1).
                         If None, equal risk contribution is used.

        Returns:
            Weights, expected risk, risk contributions, and diagnostics
        """

        n_assets = cov_matrix.shape[0]

        if risk_budget is None:
            risk_budget = np.ones(n_assets) / n_assets

        risk_budget = risk_budget / risk_budget.sum()

        def objective(w):
            portfolio_vol = np.sqrt(w @ cov_matrix @ w)
            marginal = cov_matrix @ w
            risk_contrib = w * marginal / portfolio_vol
            # Minimise squared deviation from target budget
            target = risk_budget * portfolio_vol
            return np.sum((risk_contrib - target) ** 2)

        constraints = [{'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0}]
        bounds = [(1e-6, 1.0) for _ in range(n_assets)]
        x0 = np.ones(n_assets) / n_assets

        result = minimize(objective, x0, method='SLSQP',
                          bounds=bounds, constraints=constraints)

        weights = result.x
        portfolio_risk = float(np.sqrt(weights @ cov_matrix @ weights))
        risk_info = self.calculate_risk_contributions(weights, cov_matrix)

        logger.info("Risk parity portfolio: risk=%.4f, success=%s",
                     portfolio_risk, result.success)

        return {
            'weights': weights,
            'expected_risk': portfolio_risk,
            'risk_contributions': risk_info['risk_contributions'],
            'risk_contributions_pct': risk_info['risk_contributions_pct'],
            'target_budget': risk_budget,
            'optimization_success': result.success,
            'method': 'risk_parity'
        }

    def calculate_risk_contributions(self, weights: np.ndarray,
                                     cov_matrix: np.ndarray) -> Dict[str, Any]:
        """
        Calculate marginal and total risk contribution per asset

        Args:
            weights: Portfolio weight vector
            cov_matrix: Asset covariance matrix

        Returns:
            Marginal risk, total risk contribution, and percentage contributions
        """

        portfolio_vol = np.sqrt(weights @ cov_matrix @ weights)
        marginal_risk = cov_matrix @ weights / portfolio_vol
        risk_contributions = weights * marginal_risk
        risk_contributions_pct = risk_contributions / portfolio_vol if portfolio_vol > 0 else risk_contributions * 0

        return {
            'portfolio_risk': float(portfolio_vol),
            'marginal_risk': marginal_risk,
            'risk_contributions': risk_contributions,
            'risk_contributions_pct': risk_contributions_pct,
            'max_contribution_idx': int(np.argmax(risk_contributions)),
            'min_contribution_idx': int(np.argmin(risk_contributions))
        }

    def risk_budget_optimization(self, cov_matrix: np.ndarray,
                                 expected_returns: np.ndarray,
                                 risk_budgets: np.ndarray) -> Dict[str, Any]:
        """
        Maximise expected return subject to risk budget constraints

        Args:
            cov_matrix: Asset covariance matrix
            expected_returns: Expected return vector
            risk_budgets: Target risk budget per asset (sums to 1)

        Returns:
            Optimised weights, expected return, risk, and diagnostics
        """

        n_assets = len(expected_returns)
        risk_budgets = risk_budgets / risk_budgets.sum()

        def neg_return(w):
            return -w @ expected_returns

        def risk_budget_constraint(w):
            port_vol = np.sqrt(w @ cov_matrix @ w)
            if port_vol < 1e-12:
                return 0.0
            marginal = cov_matrix @ w
            rc = w * marginal / port_vol
            target = risk_budgets * port_vol
            return -np.sum((rc - target) ** 2)

        constraints = [
            {'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0},
            {'type': 'ineq', 'fun': risk_budget_constraint}
        ]
        bounds = [(1e-6, 1.0) for _ in range(n_assets)]
        x0 = np.ones(n_assets) / n_assets

        result = minimize(neg_return, x0, method='SLSQP',
                          bounds=bounds, constraints=constraints)

        weights = result.x
        expected_ret = float(weights @ expected_returns)
        expected_risk = float(np.sqrt(weights @ cov_matrix @ weights))
        risk_info = self.calculate_risk_contributions(weights, cov_matrix)

        logger.info("Risk budget optimisation: return=%.4f, risk=%.4f",
                     expected_ret, expected_risk)

        return {
            'weights': weights,
            'expected_return': expected_ret,
            'expected_risk': expected_risk,
            'sharpe_ratio': (expected_ret - self.risk_free_rate) / expected_risk if expected_risk > 0 else 0.0,
            'risk_contributions': risk_info['risk_contributions'],
            'risk_budgets': risk_budgets,
            'optimization_success': result.success,
            'method': 'risk_budget_optimization'
        }

    # -------------------------------------------------------------------------
    # Robust Optimization
    # -------------------------------------------------------------------------

    def robust_mean_variance(self, returns_df: pd.DataFrame,
                             epsilon: float = 0.1) -> Dict[str, Any]:
        """
        Worst-case (minimax) mean-variance optimisation with uncertainty set

        Accounts for estimation error in expected returns by optimising for
        the worst-case return within an ellipsoidal uncertainty set of radius
        epsilon around the sample mean.

        Args:
            returns_df: DataFrame of asset returns (columns = assets)
            epsilon: Radius of the uncertainty set around the mean estimate

        Returns:
            Robust portfolio weights, expected return, risk, and diagnostics
        """

        if returns_df.empty or len(returns_df) < 30:
            return {'error': 'Insufficient data for robust optimisation'}

        mu = returns_df.mean().values * 252
        cov = returns_df.cov().values * 252
        n_assets = len(mu)

        def objective(w):
            portfolio_return = w @ mu
            portfolio_var = w @ cov @ w
            # Worst-case return: subtract epsilon * ||Sigma^{1/2} w||
            penalty = epsilon * np.sqrt(portfolio_var)
            return -(portfolio_return - penalty - 0.5 * self.default_risk_aversion * portfolio_var)

        constraints = [{'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0}]
        bounds = [(self.min_weight, self.max_weight) for _ in range(n_assets)]
        x0 = np.ones(n_assets) / n_assets

        result = minimize(objective, x0, method='SLSQP',
                          bounds=bounds, constraints=constraints)

        weights = result.x
        expected_ret = float(weights @ mu)
        expected_risk = float(np.sqrt(weights @ cov @ weights))
        worst_case_return = expected_ret - epsilon * expected_risk

        asset_names = list(returns_df.columns)

        logger.info("Robust MV portfolio: return=%.4f, worst_case=%.4f, risk=%.4f",
                     expected_ret, worst_case_return, expected_risk)

        return {
            'weights': dict(zip(asset_names, weights)),
            'expected_return': expected_ret,
            'worst_case_return': worst_case_return,
            'expected_risk': expected_risk,
            'sharpe_ratio': (expected_ret - self.risk_free_rate) / expected_risk if expected_risk > 0 else 0.0,
            'epsilon': epsilon,
            'optimization_success': result.success,
            'method': 'robust_mean_variance'
        }

    def resampled_efficient_frontier(self, returns_df: pd.DataFrame,
                                     n_simulations: int = 1000) -> Dict[str, Any]:
        """
        Michaud resampled efficient frontier

        Generates Monte Carlo samples of the return distribution, solves a
        max-Sharpe portfolio for each sample, and averages the resulting
        weights to produce a more stable allocation.

        Args:
            returns_df: DataFrame of asset returns (columns = assets)
            n_simulations: Number of Monte Carlo resamples

        Returns:
            Averaged portfolio weights, expected return, risk, and diagnostics
        """

        if returns_df.empty or len(returns_df) < 30:
            return {'error': 'Insufficient data for resampled frontier'}

        mu = returns_df.mean().values * 252
        cov = returns_df.cov().values * 252
        n_assets = len(mu)
        n_obs = len(returns_df)
        asset_names = list(returns_df.columns)

        all_weights = np.zeros((n_simulations, n_assets))
        successful = 0

        for i in range(n_simulations):
            # Resample returns
            sampled_returns = returns_df.sample(n=n_obs, replace=True)
            sim_mu = sampled_returns.mean().values * 252
            sim_cov = sampled_returns.cov().values * 252

            # Regularise covariance to ensure positive-definiteness
            sim_cov += np.eye(n_assets) * 1e-6

            # Max Sharpe on simulated parameters
            def neg_sharpe(w):
                ret = w @ sim_mu
                vol = np.sqrt(w @ sim_cov @ w)
                return -(ret - self.risk_free_rate) / vol if vol > 1e-12 else 0.0

            constraints = [{'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0}]
            bounds = [(self.min_weight, self.max_weight) for _ in range(n_assets)]
            x0 = np.ones(n_assets) / n_assets

            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                res = minimize(neg_sharpe, x0, method='SLSQP',
                               bounds=bounds, constraints=constraints)

            if res.success:
                all_weights[successful] = res.x
                successful += 1

        if successful == 0:
            return {'error': 'All resampling optimisations failed'}

        avg_weights = all_weights[:successful].mean(axis=0)
        avg_weights = avg_weights / avg_weights.sum()  # renormalise

        expected_ret = float(avg_weights @ mu)
        expected_risk = float(np.sqrt(avg_weights @ cov @ avg_weights))
        weight_std = all_weights[:successful].std(axis=0)

        logger.info("Resampled frontier: %d/%d successful, return=%.4f, risk=%.4f",
                     successful, n_simulations, expected_ret, expected_risk)

        return {
            'weights': dict(zip(asset_names, avg_weights)),
            'expected_return': expected_ret,
            'expected_risk': expected_risk,
            'sharpe_ratio': (expected_ret - self.risk_free_rate) / expected_risk if expected_risk > 0 else 0.0,
            'weight_stability': dict(zip(asset_names, weight_std)),
            'successful_simulations': successful,
            'total_simulations': n_simulations,
            'method': 'resampled_efficient_frontier'
        }

    def minimum_cvar_portfolio(self, returns_df: pd.DataFrame,
                               confidence: float = 0.95) -> Dict[str, Any]:
        """
        Minimise Conditional Value-at-Risk (Expected Shortfall) portfolio

        Uses linear-programming-style reformulation: minimise CVaR by
        introducing an auxiliary variable alpha (VaR threshold) and
        penalising tail losses beyond alpha.

        Args:
            returns_df: DataFrame of asset returns (columns = assets)
            confidence: Confidence level for CVaR (e.g. 0.95)

        Returns:
            Minimum-CVaR portfolio weights, expected return, CVaR, and diagnostics
        """

        if returns_df.empty or len(returns_df) < 30:
            return {'error': 'Insufficient data for CVaR optimisation'}

        returns_matrix = returns_df.values
        n_obs, n_assets = returns_matrix.shape
        asset_names = list(returns_df.columns)

        # Decision variables: [w_1, ..., w_n, alpha]
        # where alpha is the VaR threshold
        def cvar_objective(x):
            w = x[:n_assets]
            alpha = x[n_assets]
            portfolio_returns = returns_matrix @ w
            losses = -portfolio_returns
            # CVaR = alpha + (1 / ((1 - confidence) * T)) * sum(max(loss - alpha, 0))
            excess = np.maximum(losses - alpha, 0)
            cvar = alpha + excess.mean() / (1.0 - confidence)
            return cvar

        # Weights sum to 1
        constraints = [
            {'type': 'eq', 'fun': lambda x: np.sum(x[:n_assets]) - 1.0}
        ]

        # Bounds: weights in [min, max], alpha unbounded
        bounds = [(self.min_weight, self.max_weight) for _ in range(n_assets)]
        bounds.append((-1.0, 1.0))  # alpha (VaR) bounds

        x0 = np.zeros(n_assets + 1)
        x0[:n_assets] = 1.0 / n_assets
        x0[n_assets] = 0.0

        result = minimize(cvar_objective, x0, method='SLSQP',
                          bounds=bounds, constraints=constraints)

        weights = result.x[:n_assets]
        var_threshold = result.x[n_assets]

        # Compute portfolio stats
        portfolio_returns = returns_matrix @ weights
        mu_annual = float(np.mean(portfolio_returns) * 252)
        vol_annual = float(np.std(portfolio_returns) * np.sqrt(252))

        sorted_port = np.sort(portfolio_returns)
        var_idx = int((1.0 - confidence) * n_obs)
        historical_var = float(-sorted_port[max(var_idx, 0)])
        tail = sorted_port[:max(var_idx, 1)]
        historical_cvar = float(-np.mean(tail))

        logger.info("Min-CVaR portfolio: CVaR=%.4f, return=%.4f, risk=%.4f",
                     historical_cvar, mu_annual, vol_annual)

        return {
            'weights': dict(zip(asset_names, weights)),
            'expected_return': mu_annual,
            'expected_risk': vol_annual,
            'var': historical_var,
            'cvar': historical_cvar,
            'var_threshold': float(var_threshold),
            'confidence': confidence,
            'sharpe_ratio': (mu_annual - self.risk_free_rate) / vol_annual if vol_annual > 0 else 0.0,
            'optimization_success': result.success,
            'method': 'minimum_cvar'
        }
