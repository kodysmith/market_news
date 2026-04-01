"""Quant signal layer for the daily cashflow bot.

Integrates cherry-picked QuantEngine modules to improve trading decisions:
- GARCH: vol forecast → better entry timing (skip when vol expanding)
- Bayesian regime: probability-based regime → smoother than binary SMA50
- Order flow: VPIN proxy → detect dangerous microstructure days
- Tail risk EVT: dynamic stop loss sizing based on tail distribution

All signals use T-1 data only (no look-ahead bias).
"""
from __future__ import annotations

import logging
import sys
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

logger = logging.getLogger("bot.quant_signals")


# ---------------------------------------------------------------------------
# GARCH volatility signal
# ---------------------------------------------------------------------------
def get_garch_signal(spx_returns: pd.Series) -> dict:
    """Forecast volatility using GARCH. Returns signal dict.

    Args:
        spx_returns: daily log returns (T-1 and earlier)

    Returns:
        {
            'garch_vol_1d': float,     # 1-day ahead vol forecast (annualized %)
            'garch_vol_5d': float,     # 5-day ahead vol forecast
            'vol_expanding': bool,     # True if forecast > realized
            'vol_ratio': float,        # forecast / realized (>1 = expanding)
        }
    """
    result = {
        'garch_vol_1d': 0.0, 'garch_vol_5d': 0.0,
        'vol_expanding': False, 'vol_ratio': 1.0,
    }

    if len(spx_returns) < 100:
        return result

    try:
        from QuantEngine.engine.research.garch_volatility import GARCHVolatilityForecaster

        config = {
            'garch_p': 1, 'garch_q': 1,
            'forecast_horizon': 5,
            'rolling_window': 252,
            'ewma_span': 60,
        }
        forecaster = GARCHVolatilityForecaster(config)

        # Scale returns to percentage for GARCH
        returns_pct = spx_returns * 100
        forecast = forecaster.forecast_volatility(returns_pct, model_type='GARCH', horizon=5)

        if forecast and 'daily_vol_forecast' in forecast:
            vol_series = forecast['daily_vol_forecast']
            result['garch_vol_1d'] = float(vol_series.iloc[0]) * np.sqrt(252)  # annualize
            result['garch_vol_5d'] = float(vol_series.mean()) * np.sqrt(252)

            # Compare to realized
            rv_20d = float(spx_returns.tail(20).std() * np.sqrt(252) * 100)
            if rv_20d > 0:
                result['vol_ratio'] = result['garch_vol_1d'] / rv_20d
                result['vol_expanding'] = result['vol_ratio'] > 1.1

        logger.info("GARCH: 1d=%.1f%% 5d=%.1f%% ratio=%.2f expanding=%s",
                    result['garch_vol_1d'], result['garch_vol_5d'],
                    result['vol_ratio'], result['vol_expanding'])

    except Exception as e:
        logger.warning("GARCH signal failed: %s", e)

    return result


# ---------------------------------------------------------------------------
# Bayesian regime signal
# ---------------------------------------------------------------------------
def get_regime_signal(spx_returns: pd.Series) -> dict:
    """Detect market regime using Bayesian switching model.

    Returns:
        {
            'regime': int,              # 0=low vol, 1=high vol
            'regime_prob': float,       # probability of current regime (0-1)
            'regime_label': str,        # 'low_vol' | 'high_vol' | 'crisis'
            'transition_prob': float,   # probability of regime change
            'position_scale': float,    # suggested position scaling (0-1)
        }
    """
    result = {
        'regime': 0, 'regime_prob': 0.5, 'regime_label': 'unknown',
        'transition_prob': 0.0, 'position_scale': 1.0,
    }

    if len(spx_returns) < 200:
        return result

    try:
        from QuantEngine.engine.research.bayesian_regime_switching import BayesianRegimeSwitcher

        config = {
            'n_regimes': 2,
            'em_max_iter': 100,
            'em_tolerance': 1e-6,
            'changepoint_hazard': 1/100,
            'n_posterior_samples': 1000,
        }
        switcher = BayesianRegimeSwitcher(config)

        # Scale returns
        returns_pct = spx_returns * 100

        # Get regime probabilities
        probs = switcher.get_regime_probabilities(returns_pct)

        if probs is not None and not probs.empty:
            # Latest regime probabilities
            latest = probs.iloc[-1]
            regime = int(latest.argmax())
            regime_prob = float(latest.max())

            # Regime statistics
            stats = switcher.regime_conditional_statistics(returns_pct)

            # Label based on which regime has higher vol
            if stats and 'regime_stats' in stats:
                regime_stats = stats['regime_stats']
                vols = {k: v.get('volatility', 0) for k, v in regime_stats.items()}
                sorted_regimes = sorted(vols.items(), key=lambda x: x[1])
                low_vol_regime = int(sorted_regimes[0][0].replace('regime_', ''))
                result['regime'] = regime
                result['regime_prob'] = regime_prob

                if regime == low_vol_regime:
                    result['regime_label'] = 'low_vol'
                    result['position_scale'] = 1.0
                else:
                    result['regime_label'] = 'high_vol'
                    result['position_scale'] = 0.5  # reduce size in high vol regime

            # Transition probability
            predict = switcher.predict_regime(returns_pct, horizon=1)
            if predict and 'transition_probs' in predict:
                trans = predict['transition_probs']
                if hasattr(trans, 'shape') and trans.shape[0] > regime:
                    # Probability of leaving current regime
                    result['transition_prob'] = float(1.0 - trans[regime, regime])

        logger.info("REGIME: %s (prob=%.0f%% scale=%.1f transition=%.0f%%)",
                    result['regime_label'], result['regime_prob'] * 100,
                    result['position_scale'], result['transition_prob'] * 100)

    except Exception as e:
        logger.warning("Regime signal failed: %s", e)

    return result


# ---------------------------------------------------------------------------
# Order flow / VPIN proxy signal
# ---------------------------------------------------------------------------
def get_flow_signal(spx_df: pd.DataFrame) -> dict:
    """Compute order flow signals from daily OHLCV data.

    Uses daily high/low/close/volume as proxies for intraday flow.

    Returns:
        {
            'vpin_proxy': float,        # volume-clock imbalance proxy (0-1)
            'order_imbalance': float,   # buy/sell imbalance (-1 to +1)
            'toxic_flow': bool,         # True if high adverse selection risk
        }
    """
    result = {'vpin_proxy': 0.5, 'order_imbalance': 0.0, 'toxic_flow': False}

    if len(spx_df) < 20:
        return result

    try:
        # VPIN proxy: use close-to-close direction + volume
        # High VPIN = lots of informed trading = dangerous for market makers (us)
        close = spx_df['Close']
        volume = spx_df.get('Volume', pd.Series(1.0, index=close.index))

        # Direction: 1 if up, -1 if down
        direction = np.sign(close.diff())

        # Volume-weighted buy/sell imbalance over last 20 bars
        buy_vol = (volume * (direction > 0)).tail(20).sum()
        sell_vol = (volume * (direction < 0)).tail(20).sum()
        total_vol = buy_vol + sell_vol

        if total_vol > 0:
            result['vpin_proxy'] = abs(buy_vol - sell_vol) / total_vol
            result['order_imbalance'] = (buy_vol - sell_vol) / total_vol

        # Toxic flow: VPIN > 0.7 AND volume spike
        vol_avg = volume.tail(20).mean()
        vol_recent = volume.tail(3).mean()
        vol_spike = vol_recent > vol_avg * 1.5

        result['toxic_flow'] = result['vpin_proxy'] > 0.7 and vol_spike

        logger.info("FLOW: vpin=%.2f imbal=%.2f toxic=%s",
                    result['vpin_proxy'], result['order_imbalance'], result['toxic_flow'])

    except Exception as e:
        logger.warning("Flow signal failed: %s", e)

    return result


# ---------------------------------------------------------------------------
# Tail risk / dynamic stop loss
# ---------------------------------------------------------------------------
def get_tail_risk_signal(spx_returns: pd.Series, current_credit: float = 5.0) -> dict:
    """Compute dynamic stop loss using EVT tail analysis.

    Instead of fixed 2x stop, size the stop based on actual tail distribution.

    Returns:
        {
            'evt_var_95': float,        # 95% VaR from EVT (daily % move)
            'evt_var_99': float,        # 99% VaR
            'evt_cvar_95': float,       # 95% CVaR (expected shortfall)
            'dynamic_stop_mult': float, # suggested stop multiplier (replaces fixed 2.0)
            'tail_index': float,        # shape parameter (higher = fatter tails)
        }
    """
    result = {
        'evt_var_95': 2.0, 'evt_var_99': 3.0, 'evt_cvar_95': 2.5,
        'dynamic_stop_mult': 2.0, 'tail_index': 0.0,
    }

    if len(spx_returns) < 252:
        return result

    try:
        from QuantEngine.engine.risk_portfolio.tail_risk_evt import TailRiskAnalyzer

        config = {
            'evt_threshold_pct': 90,
            'n_bootstrap': 500,
            'copula_type': 'gaussian',
        }
        analyzer = TailRiskAnalyzer(config)

        # Fit EVT to losses (negative returns)
        returns_pct = spx_returns * 100
        losses = -returns_pct[returns_pct < 0]  # positive values = loss magnitude

        if len(losses) < 50:
            return result

        evt_result = analyzer.fit_evt(losses)

        if evt_result and 'gpd_params' in evt_result:
            params = evt_result['gpd_params']
            xi = params.get('shape', 0)  # tail index
            result['tail_index'] = float(xi)

            # Compute VaR/CVaR
            var_result = analyzer.compute_evt_var(
                losses, confidence_levels=[0.95, 0.99]
            )
            if var_result:
                result['evt_var_95'] = float(var_result.get('var_95', 2.0))
                result['evt_var_99'] = float(var_result.get('var_99', 3.0))
                result['evt_cvar_95'] = float(var_result.get('cvar_95', 2.5))

            # Dynamic stop: scale based on tail fatness
            # Fat tails (xi > 0.2) → wider stops to avoid getting stopped by noise
            # Thin tails (xi < 0) → tighter stops, moves are bounded
            if xi > 0.2:
                result['dynamic_stop_mult'] = 2.5  # wider stop for fat tails
            elif xi > 0:
                result['dynamic_stop_mult'] = 2.0  # normal
            else:
                result['dynamic_stop_mult'] = 1.5  # tighter for thin tails

            # Also scale by recent tail activity
            recent_max_loss = abs(returns_pct.tail(20).min())
            if recent_max_loss > result['evt_var_95']:
                result['dynamic_stop_mult'] = min(3.0, result['dynamic_stop_mult'] + 0.5)

        logger.info("TAIL: VaR95=%.1f%% VaR99=%.1f%% CVaR=%.1f%% stop=%.1fx tail_idx=%.2f",
                    result['evt_var_95'], result['evt_var_99'],
                    result['evt_cvar_95'], result['dynamic_stop_mult'], result['tail_index'])

    except Exception as e:
        logger.warning("Tail risk signal failed: %s", e)

    return result


# ---------------------------------------------------------------------------
# Combined quant signal for the daily cashflow bot
# ---------------------------------------------------------------------------
def get_all_quant_signals(spx_df: pd.DataFrame, vix_level: float) -> dict:
    """Run all quant signals and return combined assessment.

    Args:
        spx_df: DataFrame with OHLCV columns (T-1 data, no look-ahead)
        vix_level: current VIX level

    Returns combined dict with all signals + overall recommendation.
    """
    close = spx_df['Close']
    returns = np.log(close / close.shift(1)).dropna()

    # Run all signals
    garch = get_garch_signal(returns)
    regime = get_regime_signal(returns)
    flow = get_flow_signal(spx_df)
    tail = get_tail_risk_signal(returns)

    # Combined assessment
    danger_score = 0  # 0-10, higher = more dangerous

    # GARCH: vol expanding = danger
    if garch['vol_expanding']:
        danger_score += 2
    if garch['vol_ratio'] > 1.3:
        danger_score += 1

    # Regime: high vol regime = danger
    if regime['regime_label'] == 'high_vol':
        danger_score += 2
    if regime['transition_prob'] > 0.3:
        danger_score += 1  # regime may be changing

    # Flow: toxic = danger
    if flow['toxic_flow']:
        danger_score += 2
    if flow['vpin_proxy'] > 0.6:
        danger_score += 1

    # VIX: elevated = danger
    if vix_level > 30:
        danger_score += 1

    # Position sizing based on danger
    if danger_score >= 6:
        position_scale = 0.0  # skip
        action = "SKIP"
    elif danger_score >= 4:
        position_scale = 0.5  # half size
        action = "HALF_SIZE"
    elif danger_score >= 2:
        position_scale = 0.75
        action = "REDUCE"
    else:
        position_scale = 1.0
        action = "FULL_SIZE"

    # Dynamic stop from tail risk
    stop_mult = tail['dynamic_stop_mult']

    combined = {
        'garch': garch,
        'regime': regime,
        'flow': flow,
        'tail': tail,
        'danger_score': danger_score,
        'position_scale': position_scale,
        'action': action,
        'dynamic_stop_mult': stop_mult,
    }

    logger.info(
        "QUANT SIGNALS: danger=%d/10 action=%s scale=%.0f%% stop=%.1fx | "
        "GARCH(ratio=%.2f) REGIME(%s,%.0f%%) FLOW(vpin=%.2f) TAIL(idx=%.2f)",
        danger_score, action, position_scale * 100, stop_mult,
        garch['vol_ratio'], regime['regime_label'], regime['regime_prob'] * 100,
        flow['vpin_proxy'], tail['tail_index'],
    )

    return combined
