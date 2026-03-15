"""
Academic Factor Model Analysis for AI Quant Trading System

Implements standard academic factor models and risk decomposition:
- Fama-French 3-factor and 5-factor models
- Carhart 4-factor model (momentum)
- Factor construction from cross-sectional stock data
- Rolling and time-varying factor exposure analysis
- Factor momentum and timing signals
- Barra-style risk model decomposition

Uses OLS regression via statsmodels for factor loading estimation
and provides proxy factor construction when external data is unavailable.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
import logging
from datetime import datetime
import statsmodels.api as sm
from scipy import stats

logger = logging.getLogger(__name__)


class FactorModelAnalyzer:
    """
    Academic factor model implementation for systematic strategy research.

    Supports Fama-French 3/5-factor, Carhart 4-factor regressions,
    custom factor construction, and Barra-style risk decomposition.
    """

    def __init__(self, config: Dict[str, Any]):
        self.config = config

        # Regression settings
        self.min_observations = config.get('min_observations', 36)
        self.confidence_level = config.get('confidence_level', 0.95)
        self.annualization_factor = config.get('annualization_factor', 12)

        # Factor construction parameters
        self.size_breakpoint = config.get('size_breakpoint', 0.5)
        self.value_breakpoints = config.get('value_breakpoints', [0.3, 0.7])
        self.momentum_quantiles = config.get('momentum_quantiles', [0.3, 0.7])

        # Rolling window defaults
        self.default_rolling_window = config.get('default_rolling_window', 60)

        # Risk model settings
        self.newey_west_lags = config.get('newey_west_lags', None)
        self.shrinkage_target = config.get('shrinkage_target', 'constant_correlation')
        self.shrinkage_intensity = config.get('shrinkage_intensity', 0.2)

        logger.info("FactorModelAnalyzer initialized with config: min_obs=%d, confidence=%.2f",
                     self.min_observations, self.confidence_level)

    # -------------------------------------------------------------------------
    # Fama-French Models
    # -------------------------------------------------------------------------

    def run_ff3_regression(self, returns: pd.Series,
                          factor_data: pd.DataFrame) -> Dict[str, Any]:
        """
        Run Fama-French 3-factor regression.

        Estimates: R_i - R_f = alpha + beta_mkt * MktRf + beta_smb * SMB
                   + beta_hml * HML + epsilon

        Args:
            returns: Excess return series for the asset/portfolio
            factor_data: DataFrame with columns ['MktRf', 'SMB', 'HML']

        Returns:
            Dict with alpha, betas, t-stats, R-squared, residuals
        """
        required_factors = ['MktRf', 'SMB', 'HML']
        return self._run_factor_regression(returns, factor_data, required_factors,
                                           model_name='FF3')

    def run_ff5_regression(self, returns: pd.Series,
                          factor_data: pd.DataFrame) -> Dict[str, Any]:
        """
        Run Fama-French 5-factor regression.

        Extends FF3 with profitability (RMW) and investment (CMA) factors.

        Args:
            returns: Excess return series for the asset/portfolio
            factor_data: DataFrame with columns ['MktRf', 'SMB', 'HML', 'RMW', 'CMA']

        Returns:
            Dict with alpha, betas, t-stats, R-squared, residuals
        """
        required_factors = ['MktRf', 'SMB', 'HML', 'RMW', 'CMA']
        return self._run_factor_regression(returns, factor_data, required_factors,
                                           model_name='FF5')

    def run_carhart4_regression(self, returns: pd.Series,
                               factor_data: pd.DataFrame) -> Dict[str, Any]:
        """
        Run Carhart 4-factor regression (FF3 + momentum).

        Adds UMD (Up Minus Down) momentum factor to the FF3 model.

        Args:
            returns: Excess return series for the asset/portfolio
            factor_data: DataFrame with columns ['MktRf', 'SMB', 'HML', 'UMD']

        Returns:
            Dict with alpha, betas, t-stats, R-squared, residuals
        """
        required_factors = ['MktRf', 'SMB', 'HML', 'UMD']
        return self._run_factor_regression(returns, factor_data, required_factors,
                                           model_name='Carhart4')

    def calculate_factor_exposures(self, returns_df: pd.DataFrame,
                                   factor_data: pd.DataFrame) -> pd.DataFrame:
        """
        Compute factor loading matrix for multiple assets.

        Runs separate regressions for each asset column in returns_df against
        the provided factor data. Returns a DataFrame of betas indexed by
        asset name with factor names as columns.

        Args:
            returns_df: DataFrame where each column is an asset return series
            factor_data: DataFrame of factor return series

        Returns:
            DataFrame of factor loadings (assets x factors)
        """
        available_factors = [c for c in factor_data.columns
                             if c in returns_df.index.__class__.__name__ or True]
        available_factors = list(factor_data.columns)
        exposures = {}

        for asset in returns_df.columns:
            try:
                aligned_returns, aligned_factors = self._align_data(
                    returns_df[asset], factor_data
                )
                if len(aligned_returns) < self.min_observations:
                    logger.warning("Insufficient data for %s (%d obs), skipping",
                                   asset, len(aligned_returns))
                    continue

                X = sm.add_constant(aligned_factors)
                model = sm.OLS(aligned_returns, X).fit(
                    cov_type='HAC',
                    cov_kwds={'maxlags': self.newey_west_lags or max(1, len(aligned_returns) // 20)}
                )
                exposures[asset] = model.params.drop('const')

            except Exception as e:
                logger.error("Factor exposure estimation failed for %s: %s", asset, e)
                continue

        if not exposures:
            logger.warning("No factor exposures could be estimated")
            return pd.DataFrame()

        result = pd.DataFrame(exposures).T
        result.index.name = 'asset'
        logger.info("Computed factor exposures for %d assets across %d factors",
                     len(result), len(result.columns))
        return result

    # -------------------------------------------------------------------------
    # Factor Construction
    # -------------------------------------------------------------------------

    def construct_smb_hml(self, universe_data: pd.DataFrame) -> pd.DataFrame:
        """
        Construct SMB (size) and HML (value) factors from stock-level data.

        Follows Fama-French methodology:
        - Sort stocks by market cap into Small/Big at the median
        - Sort stocks by book-to-market into Low/Mid/High at 30th/70th percentiles
        - Form 6 portfolios (SL, SM, SH, BL, BM, BH) using intersections
        - SMB = avg(Small) - avg(Big)
        - HML = avg(High B/M) - avg(Low B/M)

        Args:
            universe_data: DataFrame with columns ['date', 'ticker', 'returns',
                           'market_cap', 'book_to_market']

        Returns:
            DataFrame with columns ['SMB', 'HML'] indexed by date
        """
        required_cols = ['date', 'ticker', 'returns', 'market_cap', 'book_to_market']
        missing = [c for c in required_cols if c not in universe_data.columns]
        if missing:
            raise ValueError(f"Missing required columns: {missing}")

        results = []

        for date, group in universe_data.groupby('date'):
            group = group.dropna(subset=['market_cap', 'book_to_market', 'returns'])
            if len(group) < 10:
                logger.debug("Skipping date %s: only %d stocks", date, len(group))
                continue

            # Size breakpoint: median market cap
            size_median = group['market_cap'].quantile(self.size_breakpoint)
            is_small = group['market_cap'] <= size_median

            # Value breakpoints: 30th and 70th percentiles of B/M
            bm_low = group['book_to_market'].quantile(self.value_breakpoints[0])
            bm_high = group['book_to_market'].quantile(self.value_breakpoints[1])

            is_low_bm = group['book_to_market'] <= bm_low
            is_high_bm = group['book_to_market'] >= bm_high
            is_mid_bm = ~is_low_bm & ~is_high_bm

            # Form 6 value-weighted portfolios
            sl = group.loc[is_small & is_low_bm, 'returns'].mean()
            sm = group.loc[is_small & is_mid_bm, 'returns'].mean()
            sh = group.loc[is_small & is_high_bm, 'returns'].mean()
            bl = group.loc[~is_small & is_low_bm, 'returns'].mean()
            bm = group.loc[~is_small & is_mid_bm, 'returns'].mean()
            bh = group.loc[~is_small & is_high_bm, 'returns'].mean()

            # Handle empty portfolios
            small_avg = np.nanmean([sl, sm, sh])
            big_avg = np.nanmean([bl, bm, bh])
            high_avg = np.nanmean([sh, bh])
            low_avg = np.nanmean([sl, bl])

            smb = small_avg - big_avg
            hml = high_avg - low_avg

            results.append({'date': date, 'SMB': smb, 'HML': hml})

        if not results:
            logger.warning("Could not construct SMB/HML: insufficient data")
            return pd.DataFrame(columns=['SMB', 'HML'])

        df = pd.DataFrame(results).set_index('date')
        logger.info("Constructed SMB/HML factors: %d periods", len(df))
        return df

    def construct_momentum_factor(self, universe_data: pd.DataFrame,
                                  lookback: int = 12,
                                  skip: int = 1) -> pd.Series:
        """
        Construct UMD (Up Minus Down) momentum factor.

        Standard 12-1 momentum: rank stocks by cumulative return over the past
        `lookback` months, skipping the most recent `skip` month(s) to avoid
        short-term reversal. Long top quantile, short bottom quantile.

        Args:
            universe_data: DataFrame with columns ['date', 'ticker', 'returns']
            lookback: Number of periods for momentum calculation
            skip: Number of recent periods to skip

        Returns:
            Series of momentum factor returns indexed by date
        """
        required_cols = ['date', 'ticker', 'returns']
        missing = [c for c in required_cols if c not in universe_data.columns]
        if missing:
            raise ValueError(f"Missing required columns: {missing}")

        # Pivot to wide format: dates x tickers
        returns_wide = universe_data.pivot_table(
            index='date', columns='ticker', values='returns'
        )

        dates = sorted(returns_wide.index)
        momentum_returns = {}

        for i in range(lookback + skip, len(dates)):
            current_date = dates[i]

            # Cumulative return over lookback window, skipping recent period(s)
            start_idx = i - lookback - skip
            end_idx = i - skip
            window_returns = returns_wide.iloc[start_idx:end_idx]

            cum_returns = (1 + window_returns).prod() - 1
            cum_returns = cum_returns.dropna()

            if len(cum_returns) < 10:
                continue

            # Quantile breakpoints
            low_q = cum_returns.quantile(self.momentum_quantiles[0])
            high_q = cum_returns.quantile(self.momentum_quantiles[1])

            winners = cum_returns[cum_returns >= high_q].index
            losers = cum_returns[cum_returns <= low_q].index

            # Current period return of the long-short portfolio
            current_returns = returns_wide.loc[current_date]
            winner_ret = current_returns[winners].mean()
            loser_ret = current_returns[losers].mean()

            if not np.isnan(winner_ret) and not np.isnan(loser_ret):
                momentum_returns[current_date] = winner_ret - loser_ret

        result = pd.Series(momentum_returns, name='UMD')
        result.index.name = 'date'
        logger.info("Constructed momentum factor: %d periods, lookback=%d, skip=%d",
                     len(result), lookback, skip)
        return result

    def construct_quality_factor(self, universe_data: pd.DataFrame) -> pd.Series:
        """
        Construct RMW (Robust Minus Weak) quality factor proxy.

        Uses ROE or gross profitability as the quality measure. Sorts stocks
        into high/low quality portfolios each period and returns the
        long-short spread.

        Args:
            universe_data: DataFrame with columns ['date', 'ticker', 'returns']
                           and at least one of ['roe', 'gross_profitability']

        Returns:
            Series of quality factor returns indexed by date
        """
        # Determine quality metric
        if 'roe' in universe_data.columns:
            quality_col = 'roe'
        elif 'gross_profitability' in universe_data.columns:
            quality_col = 'gross_profitability'
        else:
            raise ValueError("universe_data must contain 'roe' or 'gross_profitability'")

        quality_returns = {}

        for date, group in universe_data.groupby('date'):
            group = group.dropna(subset=[quality_col, 'returns'])
            if len(group) < 10:
                continue

            low_q = group[quality_col].quantile(self.momentum_quantiles[0])
            high_q = group[quality_col].quantile(self.momentum_quantiles[1])

            robust = group.loc[group[quality_col] >= high_q, 'returns'].mean()
            weak = group.loc[group[quality_col] <= low_q, 'returns'].mean()

            if not np.isnan(robust) and not np.isnan(weak):
                quality_returns[date] = robust - weak

        result = pd.Series(quality_returns, name='RMW')
        result.index.name = 'date'
        logger.info("Constructed quality factor (RMW via %s): %d periods",
                     quality_col, len(result))
        return result

    # -------------------------------------------------------------------------
    # Factor Analysis
    # -------------------------------------------------------------------------

    def calculate_factor_attribution(self, portfolio_returns: pd.Series,
                                     factor_data: pd.DataFrame) -> Dict[str, Any]:
        """
        Decompose portfolio returns into factor contributions and alpha.

        For each factor, contribution = beta_i * mean(factor_i).
        Alpha is the unexplained component.

        Args:
            portfolio_returns: Portfolio excess return series
            factor_data: DataFrame of factor returns

        Returns:
            Dict with factor contributions, alpha contribution, total return,
            and percentage breakdown
        """
        aligned_returns, aligned_factors = self._align_data(
            portfolio_returns, factor_data
        )

        if len(aligned_returns) < self.min_observations:
            logger.warning("Insufficient data for attribution (%d obs)",
                           len(aligned_returns))
            return {'error': 'insufficient_data', 'n_obs': len(aligned_returns)}

        # Run regression to get factor loadings
        X = sm.add_constant(aligned_factors)
        model = sm.OLS(aligned_returns, X).fit()

        alpha = model.params['const']
        betas = model.params.drop('const')
        factor_means = aligned_factors.mean()

        # Factor contributions (per period)
        contributions = {}
        total_factor_contribution = 0.0

        for factor_name in betas.index:
            contrib = betas[factor_name] * factor_means[factor_name]
            contributions[factor_name] = {
                'beta': float(betas[factor_name]),
                'factor_mean_return': float(factor_means[factor_name]),
                'contribution': float(contrib),
            }
            total_factor_contribution += contrib

        total_return = float(aligned_returns.mean())
        alpha_contribution = float(alpha)

        # Percentage attribution
        if abs(total_return) > 1e-10:
            for factor_name in contributions:
                contributions[factor_name]['pct_of_total'] = (
                    contributions[factor_name]['contribution'] / total_return * 100
                )
            alpha_pct = alpha_contribution / total_return * 100
        else:
            for factor_name in contributions:
                contributions[factor_name]['pct_of_total'] = 0.0
            alpha_pct = 0.0

        result = {
            'model': 'factor_attribution',
            'n_obs': len(aligned_returns),
            'alpha': alpha_contribution,
            'alpha_annualized': alpha_contribution * self.annualization_factor,
            'alpha_pct_of_total': alpha_pct,
            'factor_contributions': contributions,
            'total_factor_contribution': float(total_factor_contribution),
            'total_mean_return': total_return,
            'r_squared': float(model.rsquared),
            'residual_vol': float(model.resid.std()),
        }

        logger.info("Factor attribution: alpha=%.4f, R²=%.3f, %d factors",
                     alpha_contribution, model.rsquared, len(contributions))
        return result

    def rolling_factor_exposure(self, returns: pd.Series,
                                factor_data: pd.DataFrame,
                                window: int = 60) -> pd.DataFrame:
        """
        Estimate time-varying factor betas using rolling-window OLS.

        Args:
            returns: Asset/portfolio excess return series
            factor_data: DataFrame of factor returns
            window: Rolling window size (number of periods)

        Returns:
            DataFrame of rolling betas indexed by date, columns are factor names
        """
        aligned_returns, aligned_factors = self._align_data(returns, factor_data)

        if len(aligned_returns) < window:
            logger.warning("Insufficient data for rolling regression: %d < %d",
                           len(aligned_returns), window)
            return pd.DataFrame()

        rolling_betas = []
        dates = aligned_returns.index

        for i in range(window, len(dates) + 1):
            window_returns = aligned_returns.iloc[i - window:i]
            window_factors = aligned_factors.iloc[i - window:i]

            try:
                X = sm.add_constant(window_factors)
                model = sm.OLS(window_returns, X).fit()
                betas = model.params.drop('const')
                betas.name = dates[i - 1]
                rolling_betas.append(betas)
            except Exception as e:
                logger.debug("Rolling regression failed at %s: %s", dates[i - 1], e)
                continue

        if not rolling_betas:
            return pd.DataFrame()

        result = pd.DataFrame(rolling_betas)
        result.index.name = 'date'
        logger.info("Rolling factor exposure: %d windows of size %d",
                     len(result), window)
        return result

    def factor_timing_signal(self, factor_returns: pd.DataFrame,
                             lookback: int = 12) -> Dict[str, Any]:
        """
        Generate factor momentum signals.

        Factors with positive recent returns tend to continue performing well.
        Computes trailing Sharpe ratio and cumulative return for each factor.

        Args:
            factor_returns: DataFrame of factor return series
            lookback: Number of periods for momentum lookback

        Returns:
            Dict with signal strengths, rankings, and recommended factor tilts
        """
        if len(factor_returns) < lookback:
            logger.warning("Insufficient data for factor timing: %d < %d",
                           len(factor_returns), lookback)
            return {'error': 'insufficient_data'}

        recent = factor_returns.iloc[-lookback:]
        signals = {}

        for factor in factor_returns.columns:
            factor_rets = recent[factor].dropna()
            if len(factor_rets) < lookback // 2:
                continue

            cum_return = float((1 + factor_rets).prod() - 1)
            mean_return = float(factor_rets.mean())
            vol = float(factor_rets.std())
            sharpe = mean_return / vol if vol > 1e-10 else 0.0

            signals[factor] = {
                'cumulative_return': cum_return,
                'mean_return': mean_return,
                'volatility': vol,
                'sharpe_ratio': sharpe,
                'hit_rate': float((factor_rets > 0).mean()),
            }

        if not signals:
            return {'error': 'no_valid_signals'}

        # Rank factors by trailing Sharpe
        ranked = sorted(signals.items(), key=lambda x: x[1]['sharpe_ratio'], reverse=True)
        rankings = {name: rank + 1 for rank, (name, _) in enumerate(ranked)}

        # Simple tilt recommendation: overweight positive Sharpe, underweight negative
        tilts = {}
        for factor, data in signals.items():
            if data['sharpe_ratio'] > 0.5:
                tilts[factor] = 'overweight'
            elif data['sharpe_ratio'] < -0.5:
                tilts[factor] = 'underweight'
            else:
                tilts[factor] = 'neutral'

        result = {
            'lookback': lookback,
            'signals': signals,
            'rankings': rankings,
            'recommended_tilts': tilts,
            'as_of_date': str(factor_returns.index[-1]),
        }

        logger.info("Factor timing signals: %d factors analyzed, lookback=%d",
                     len(signals), lookback)
        return result

    def cross_sectional_momentum(self, returns_df: pd.DataFrame,
                                 lookback: int = 252,
                                 hold: int = 21) -> Dict[str, Any]:
        """
        Cross-sectional momentum strategy: rank stocks by past returns.

        Computes long/short portfolio weights based on momentum ranking.
        Top quintile goes long, bottom quintile goes short.

        Args:
            returns_df: DataFrame of asset returns (dates x assets)
            lookback: Lookback window in trading days
            hold: Holding period in trading days

        Returns:
            Dict with portfolio weights, signal scores, and rebalance dates
        """
        if len(returns_df) < lookback + hold:
            logger.warning("Insufficient data for cross-sectional momentum")
            return {'error': 'insufficient_data'}

        rebalance_dates = []
        all_weights = {}

        # Identify rebalance points
        dates = returns_df.index
        rebalance_indices = range(lookback, len(dates), hold)

        for idx in rebalance_indices:
            rebal_date = dates[idx]
            # Lookback period cumulative returns
            lookback_rets = returns_df.iloc[idx - lookback:idx]
            cum_rets = (1 + lookback_rets).prod() - 1
            cum_rets = cum_rets.dropna()

            if len(cum_rets) < 5:
                continue

            # Rank and assign quintile weights
            ranks = cum_rets.rank(pct=True)
            n_assets = len(ranks)

            weights = pd.Series(0.0, index=ranks.index)
            long_mask = ranks >= 0.8
            short_mask = ranks <= 0.2
            n_long = long_mask.sum()
            n_short = short_mask.sum()

            if n_long > 0:
                weights[long_mask] = 1.0 / n_long
            if n_short > 0:
                weights[short_mask] = -1.0 / n_short

            all_weights[rebal_date] = weights.to_dict()
            rebalance_dates.append(rebal_date)

        # Compute strategy returns
        strategy_returns = []
        for i, rebal_date in enumerate(rebalance_dates[:-1]):
            w = pd.Series(all_weights[rebal_date])
            next_rebal = rebalance_dates[i + 1]
            period_rets = returns_df.loc[rebal_date:next_rebal, w.index]

            # Daily portfolio return during holding period
            daily_port_ret = period_rets.multiply(w).sum(axis=1)
            strategy_returns.append(daily_port_ret)

        if strategy_returns:
            strat = pd.concat(strategy_returns)
            strat = strat[~strat.index.duplicated(keep='first')]
            cumulative = float((1 + strat).prod() - 1)
            ann_return = float(strat.mean() * 252)
            ann_vol = float(strat.std() * np.sqrt(252))
            sharpe = ann_return / ann_vol if ann_vol > 1e-10 else 0.0
        else:
            cumulative = 0.0
            ann_return = 0.0
            ann_vol = 0.0
            sharpe = 0.0

        result = {
            'lookback': lookback,
            'hold_period': hold,
            'n_rebalances': len(rebalance_dates),
            'latest_weights': all_weights.get(rebalance_dates[-1], {}) if rebalance_dates else {},
            'rebalance_dates': [str(d) for d in rebalance_dates],
            'cumulative_return': cumulative,
            'annualized_return': ann_return,
            'annualized_vol': ann_vol,
            'sharpe_ratio': sharpe,
        }

        logger.info("Cross-sectional momentum: lookback=%d, hold=%d, %d rebalances, Sharpe=%.2f",
                     lookback, hold, len(rebalance_dates), sharpe)
        return result

    # -------------------------------------------------------------------------
    # Barra-Style Risk Model
    # -------------------------------------------------------------------------

    def build_risk_model(self, returns_df: pd.DataFrame,
                         factor_data: pd.DataFrame) -> Dict[str, Any]:
        """
        Estimate a Barra-style multi-factor risk model.

        Steps:
        1. Regress each asset on factor returns to get factor loadings
        2. Estimate factor covariance matrix from factor return history
        3. Compute specific (idiosyncratic) risk for each asset from residuals

        Args:
            returns_df: DataFrame of asset returns (dates x assets)
            factor_data: DataFrame of factor returns (dates x factors)

        Returns:
            Dict with factor_loadings, factor_covariance, specific_risk,
            and model diagnostics
        """
        # Step 1: Estimate factor loadings for each asset
        factor_loadings = self.calculate_factor_exposures(returns_df, factor_data)

        if factor_loadings.empty:
            logger.error("Cannot build risk model: no factor loadings estimated")
            return {'error': 'no_factor_loadings'}

        # Step 2: Factor covariance matrix (with optional shrinkage)
        aligned_returns, aligned_factors = self._align_data(
            returns_df.iloc[:, 0], factor_data
        )
        factor_cov_raw = aligned_factors.cov().values
        factor_names = list(aligned_factors.columns)

        # Ledoit-Wolf style shrinkage toward constant correlation
        factor_cov = self._shrink_covariance(factor_cov_raw)

        # Step 3: Specific risk from regression residuals
        specific_risk = {}
        assets_in_model = list(factor_loadings.index)

        for asset in assets_in_model:
            try:
                asset_aligned, factors_aligned = self._align_data(
                    returns_df[asset], factor_data
                )
                X = sm.add_constant(factors_aligned)
                model = sm.OLS(asset_aligned, X).fit()
                residual_vol = float(model.resid.std())
                specific_risk[asset] = residual_vol
            except Exception as e:
                logger.debug("Specific risk estimation failed for %s: %s", asset, e)
                specific_risk[asset] = float(returns_df[asset].std())

        specific_risk_array = np.array([specific_risk[a] for a in assets_in_model])

        result = {
            'factor_loadings': factor_loadings,
            'factor_covariance': factor_cov,
            'factor_names': factor_names,
            'specific_risk': pd.Series(specific_risk, name='specific_risk'),
            'specific_risk_array': specific_risk_array,
            'assets': assets_in_model,
            'n_assets': len(assets_in_model),
            'n_factors': len(factor_names),
            'n_observations': len(aligned_factors),
        }

        logger.info("Risk model built: %d assets, %d factors, %d observations",
                     len(assets_in_model), len(factor_names), len(aligned_factors))
        return result

    def decompose_portfolio_risk(self, weights: np.ndarray,
                                 factor_loadings: pd.DataFrame,
                                 factor_cov: np.ndarray,
                                 specific_risk: np.ndarray) -> Dict[str, Any]:
        """
        Decompose portfolio risk into systematic and idiosyncratic components.

        Total variance = w' B F B' w + w' D w
        where B = factor loadings, F = factor covariance, D = diag(specific_var)

        Args:
            weights: Portfolio weight vector (n_assets,)
            factor_loadings: DataFrame of factor loadings (assets x factors)
            factor_cov: Factor covariance matrix (n_factors x n_factors)
            specific_risk: Array of idiosyncratic volatilities (n_assets,)

        Returns:
            Dict with total risk, systematic risk, idiosyncratic risk,
            factor risk contributions, and marginal risk contributions
        """
        B = factor_loadings.values  # (n_assets, n_factors)
        w = np.asarray(weights).flatten()
        D = np.diag(specific_risk ** 2)  # Specific variance matrix

        # Portfolio factor exposure
        portfolio_factor_exposure = B.T @ w  # (n_factors,)

        # Systematic variance
        systematic_var = float(portfolio_factor_exposure @ factor_cov @ portfolio_factor_exposure)

        # Idiosyncratic variance
        idio_var = float(w @ D @ w)

        # Total variance
        total_var = systematic_var + idio_var
        total_vol = np.sqrt(total_var) if total_var > 0 else 0.0
        systematic_vol = np.sqrt(systematic_var) if systematic_var > 0 else 0.0
        idio_vol = np.sqrt(idio_var) if idio_var > 0 else 0.0

        # Factor-level risk contributions
        factor_names = list(factor_loadings.columns)
        factor_risk_contributions = {}

        for i, factor_name in enumerate(factor_names):
            # Marginal contribution of factor i to systematic variance
            factor_exposure_i = portfolio_factor_exposure[i]
            cov_with_portfolio = factor_cov[i, :] @ portfolio_factor_exposure
            contribution_to_var = factor_exposure_i * cov_with_portfolio
            factor_risk_contributions[factor_name] = {
                'portfolio_exposure': float(factor_exposure_i),
                'variance_contribution': float(contribution_to_var),
                'pct_of_systematic': float(contribution_to_var / systematic_var * 100)
                    if systematic_var > 1e-10 else 0.0,
                'pct_of_total': float(contribution_to_var / total_var * 100)
                    if total_var > 1e-10 else 0.0,
            }

        # Marginal risk contribution per asset
        total_cov = B @ factor_cov @ B.T + D
        marginal_contrib = total_cov @ w
        if total_vol > 1e-10:
            marginal_contrib = marginal_contrib / total_vol

        result = {
            'total_variance': float(total_var),
            'total_volatility': float(total_vol),
            'systematic_variance': float(systematic_var),
            'systematic_volatility': float(systematic_vol),
            'idiosyncratic_variance': float(idio_var),
            'idiosyncratic_volatility': float(idio_vol),
            'systematic_pct': float(systematic_var / total_var * 100)
                if total_var > 1e-10 else 0.0,
            'idiosyncratic_pct': float(idio_var / total_var * 100)
                if total_var > 1e-10 else 0.0,
            'factor_risk_contributions': factor_risk_contributions,
            'marginal_risk_contributions': pd.Series(
                marginal_contrib, index=factor_loadings.index, name='marginal_risk'
            ),
            'portfolio_factor_exposure': pd.Series(
                portfolio_factor_exposure, index=factor_names, name='factor_exposure'
            ),
        }

        logger.info("Risk decomposition: total_vol=%.4f, systematic=%.1f%%, idio=%.1f%%",
                     total_vol, result['systematic_pct'], result['idiosyncratic_pct'])
        return result

    # -------------------------------------------------------------------------
    # Internal Helpers
    # -------------------------------------------------------------------------

    def _run_factor_regression(self, returns: pd.Series,
                               factor_data: pd.DataFrame,
                               required_factors: List[str],
                               model_name: str = 'factor_model') -> Dict[str, Any]:
        """
        Core OLS factor regression with Newey-West standard errors.

        Args:
            returns: Excess return series
            factor_data: DataFrame containing factor columns
            required_factors: List of required factor column names
            model_name: Label for logging

        Returns:
            Dict with regression results
        """
        # Validate factor columns
        missing_factors = [f for f in required_factors if f not in factor_data.columns]
        if missing_factors:
            raise ValueError(f"{model_name} requires factors {missing_factors} "
                             f"not found in factor_data columns: {list(factor_data.columns)}")

        factors = factor_data[required_factors]

        # Align dates
        aligned_returns, aligned_factors = self._align_data(returns, factors)

        if len(aligned_returns) < self.min_observations:
            logger.warning("%s regression: insufficient data (%d < %d)",
                           model_name, len(aligned_returns), self.min_observations)
            return {
                'model': model_name,
                'error': 'insufficient_data',
                'n_obs': len(aligned_returns),
                'min_required': self.min_observations,
            }

        # OLS with Newey-West HAC standard errors
        X = sm.add_constant(aligned_factors)
        nw_lags = self.newey_west_lags or max(1, int(len(aligned_returns) ** (1 / 3)))

        try:
            model = sm.OLS(aligned_returns, X).fit(
                cov_type='HAC',
                cov_kwds={'maxlags': nw_lags}
            )
        except Exception as e:
            logger.error("%s regression failed: %s", model_name, e)
            return {'model': model_name, 'error': str(e)}

        # Extract results
        alpha = float(model.params['const'])
        alpha_tstat = float(model.tvalues['const'])
        alpha_pval = float(model.pvalues['const'])

        betas = {}
        t_stats = {}
        p_values = {}

        for factor in required_factors:
            betas[factor] = float(model.params[factor])
            t_stats[factor] = float(model.tvalues[factor])
            p_values[factor] = float(model.pvalues[factor])

        result = {
            'model': model_name,
            'n_obs': int(model.nobs),
            'alpha': alpha,
            'alpha_annualized': alpha * self.annualization_factor,
            'alpha_t_stat': alpha_tstat,
            'alpha_p_value': alpha_pval,
            'alpha_significant': alpha_pval < (1 - self.confidence_level),
            'betas': betas,
            't_stats': t_stats,
            'p_values': p_values,
            'r_squared': float(model.rsquared),
            'r_squared_adj': float(model.rsquared_adj),
            'residuals': model.resid,
            'residual_std': float(model.resid.std()),
            'durbin_watson': float(sm.stats.durbin_watson(model.resid)),
            'aic': float(model.aic),
            'bic': float(model.bic),
        }

        logger.info("%s regression: alpha=%.4f (t=%.2f), R²=%.3f, n=%d",
                     model_name, alpha, alpha_tstat, model.rsquared, model.nobs)
        return result

    def _align_data(self, returns: pd.Series,
                    factor_data: pd.DataFrame) -> Tuple[pd.Series, pd.DataFrame]:
        """
        Align return series and factor data on their common date index.

        Drops any rows with NaN values after alignment.

        Args:
            returns: Return series
            factor_data: Factor return DataFrame

        Returns:
            Tuple of (aligned_returns, aligned_factors)
        """
        combined = pd.concat([returns.rename('_target_'), factor_data], axis=1)
        combined = combined.dropna()

        aligned_returns = combined['_target_']
        aligned_factors = combined.drop(columns=['_target_'])

        return aligned_returns, aligned_factors

    def _shrink_covariance(self, cov_matrix: np.ndarray) -> np.ndarray:
        """
        Apply shrinkage to a covariance matrix toward a structured target.

        Uses a constant-correlation shrinkage target (Ledoit-Wolf style).

        Args:
            cov_matrix: Raw sample covariance matrix

        Returns:
            Shrunk covariance matrix
        """
        n = cov_matrix.shape[0]
        if n <= 1:
            return cov_matrix

        intensity = self.shrinkage_intensity

        # Constant correlation target
        vols = np.sqrt(np.diag(cov_matrix))
        corr_matrix = cov_matrix / np.outer(vols, vols)
        np.fill_diagonal(corr_matrix, 1.0)

        avg_corr = (corr_matrix.sum() - n) / (n * (n - 1))
        target = np.full_like(cov_matrix, avg_corr)
        np.fill_diagonal(target, 1.0)
        target = target * np.outer(vols, vols)

        shrunk = (1 - intensity) * cov_matrix + intensity * target
        return shrunk
