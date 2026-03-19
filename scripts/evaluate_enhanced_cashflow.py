#!/usr/bin/env python3
"""Enhanced SPX Monthly Cash Flow Evaluator — Sharpe-optimized.

Builds on the original evaluator with key improvements:
- Dynamic IV: reprices spreads using VIX on each day (not just entry IV)
- VIX regime filter: skip entries when VIX is spiking or too high
- Trend filter: only enter when SPX > N-day SMA
- Adaptive delta: adjust strike selection based on VIX level
- Max concurrent positions: cap overlapping risk
- Time stop: close before expiry to avoid gamma explosion
- Consecutive loss cooldown: pause after streaks
- Proper annualized Sharpe, Sortino, Calmar from monthly returns
"""
from __future__ import annotations

import argparse
import math
import sys
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import norm

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.optimize_spx_verticals_historical import (
    load_spx_vix,
    friday_expirations,
    get_trading_days,
    bs_put_price,
    bs_call_price,
    bs_put_delta,
    bs_call_delta,
    strike_for_delta,
    R,
    MULT,
)

SLIPPAGE_PER_LEG = 0.10
COMMISSION_PER_LEG = 0.65
ES_MULT = 50.0  # dollars per point for 1 ES (mini S&P) contract; 1 ES point ≈ 1 SPX point

NEWS_DAYS_CSV = ROOT / "data" / "news_days.csv"


# ---------------------------------------------------------------------------
# News-day calendar (risk-off: skip new entry on high-impact release days)
# ---------------------------------------------------------------------------
def get_news_dates(start: date, end: date) -> set[date]:
    """Return set of dates in [start, end] that are high-impact news days.
    If data/news_days.csv exists with a 'date' column (YYYY-MM-DD), use it.
    Else use heuristic: NFP (first Fri), CPI/PPI (second Tue), FOMC (second Wed in 8 months).
    """
    if NEWS_DAYS_CSV.is_file():
        try:
            df = pd.read_csv(NEWS_DAYS_CSV)
            if "date" not in df.columns:
                raise ValueError("no date column")
            df["date"] = pd.to_datetime(df["date"]).dt.date
            out = set(d for d in df["date"].unique() if start <= d <= end)
            return out
        except Exception:
            pass
    out: set[date] = set()
    fomc_months = (1, 3, 5, 6, 7, 9, 11, 12)
    d = date(start.year, start.month, 1)
    while d <= end:
        # First Friday (NFP)
        for day in range(1, 8):
            cand = date(d.year, d.month, day)
            if cand.weekday() == 4 and start <= cand <= end:
                out.add(cand)
                break
        # Second Tuesday (CPI/PPI)
        for day in range(8, 15):
            cand = date(d.year, d.month, day)
            if cand.weekday() == 1 and start <= cand <= end:
                out.add(cand)
                break
        # Second Wednesday in FOMC months (FOMC)
        if d.month in fomc_months:
            for day in range(8, 15):
                cand = date(d.year, d.month, day)
                if cand.weekday() == 2 and start <= cand <= end:
                    out.add(cand)
                    break
        if d.month == 12:
            d = date(d.year + 1, 1, 1)
        else:
            d = date(d.year, d.month + 1, 1)
    return out


# ---------------------------------------------------------------------------
# GEX Regime Proxy
# ---------------------------------------------------------------------------
def compute_gex_proxy(
    spx_df: pd.DataFrame,
    vix_close: pd.Series,
    start: date,
    end: date,
) -> pd.Series:
    """Build a daily GEX regime proxy from SPX + VIX + VIX3M + VIX9D.

    Positive GEX (dealers long gamma) = mean-reverting, range-bound → safe for
    short premium.  Negative GEX = trending, accelerating → dangerous.

    Three signals (each +1 for positive, 0 for negative):
      1. VIX term structure contango: VIX < VIX3M  (dealers hedged, calm)
      2. Volatility risk premium:     VIX > 20-day realized vol  (premium sellers paid)
      3. Low VIX momentum:            VIX 5-day change ≤ +15%  (no vol spike)

    Returns a Series indexed by date with integer score 0-3.
    Score ≥ 2 → "positive GEX" (safe to enter).
    """
    import yfinance as yf

    pad_start = (start - timedelta(days=60)).isoformat()
    pad_end = (end + timedelta(days=10)).isoformat()

    vix3m_df = yf.Ticker("^VIX3M").history(start=pad_start, end=pad_end)
    vix9d_df = yf.Ticker("^VIX9D").history(start=pad_start, end=pad_end)

    vix3m = vix3m_df["Close"].copy() if not vix3m_df.empty else pd.Series(dtype=float)
    vix9d = vix9d_df["Close"].copy() if not vix9d_df.empty else pd.Series(dtype=float)
    for s in (vix3m, vix9d):
        if not s.empty:
            s.index = pd.to_datetime(s.index).tz_localize(None).normalize()
            s.index = s.index.date

    # Build a common date index from spx_df
    spx_close = spx_df["Close"].copy()
    spx_close.index = pd.to_datetime(spx_close.index).tz_localize(None).normalize()
    spx_close.index = spx_close.index.date

    # Realized vol: 20-day annualized from SPX log returns
    log_ret = np.log(spx_close / spx_close.shift(1))
    rv_20d = log_ret.rolling(20).std() * np.sqrt(252) * 100  # as VIX-comparable %

    vix_series = vix_close.copy()

    dates = sorted(set(spx_close.index))
    # Precompute VIX 20-day SMA once for the contango fallback (avoid O(N²) recompute per date)
    vix_sma_20 = vix_series.rolling(20).mean()
    scores = {}
    for d in dates:
        score = 0

        vix_val = float(vix_series.get(d, np.nan))
        if np.isnan(vix_val):
            scores[d] = 0
            continue

        # Signal 1: term structure contango (VIX < VIX3M)
        vix3m_val = float(vix3m.get(d, np.nan)) if not vix3m.empty else np.nan
        if not np.isnan(vix3m_val) and vix3m_val > 0:
            if vix_val < vix3m_val:
                score += 1
        else:
            # Fallback: VIX below 20-day SMA as contango proxy
            if d in vix_sma_20.index:
                sma_val = float(vix_sma_20.get(d, np.nan))
                if not np.isnan(sma_val) and vix_val < sma_val:
                    score += 1

        # Signal 2: volatility risk premium (VIX > realized vol)
        rv_val = float(rv_20d.get(d, np.nan)) if d in rv_20d.index else np.nan
        if not np.isnan(rv_val) and rv_val > 0:
            if vix_val > rv_val:
                score += 1

        # Signal 3: low VIX momentum (5-day change ≤ 15%)
        try:
            idx = dates.index(d)
            if idx >= 5:
                d_5ago = dates[idx - 5]
                vix_5ago = float(vix_series.get(d_5ago, np.nan))
                if not np.isnan(vix_5ago) and vix_5ago > 0:
                    pct_chg = (vix_val - vix_5ago) / vix_5ago * 100
                    if pct_chg <= 15:
                        score += 1
        except (ValueError, IndexError):
            pass

        scores[d] = score

    return pd.Series(scores).sort_index()


# ---------------------------------------------------------------------------
# Filters
# ---------------------------------------------------------------------------
@dataclass
class EntryFilters:
    vix_max: float | None = None
    vix_min: float | None = None
    vix_spike_5d_pct: float | None = None      # skip if VIX rose > X% in 5 trading days
    trend_sma_period: int | None = None         # SPX must be above N-day SMA
    max_concurrent: int | None = None           # cap overlapping open positions
    consecutive_loss_cooldown: int | None = None # skip N entries after M consecutive losses
    consecutive_loss_trigger: int = 2
    gex_min_score: int | None = None            # require GEX proxy score >= this (0-3)
    rv_expansion_max: float | None = None       # skip if rv_5d/rv_10d > this (vol expanding)
    gap_down_no_entry: float | None = None      # skip entry when pre-market gap < this (e.g. -0.5 = skip if gap <= -0.5%)
    skip_news_days: bool = False               # skip entry on high-impact news days (PPI, CPI, NFP, FOMC)
    trend_sma_invert: bool = False             # if True, enter when SPX is BELOW SMA (bearish strategies)


@dataclass
class ExitRules:
    profit_target_pct: float = 0.50
    stop_mult: float = 2.0
    time_stop_dte: int | None = None          # close when DTE remaining <= this (e.g. 4 = close 4 days before expiration)
    vix_spike_exit_pct: float | None = None   # close if VIX rises > X% from entry VIX
    spx_drop_exit_pct: float | None = None    # close if SPX drops > X% from entry in 3 days
    spx_drop_lookback: int = 3                # days to look back for SPX drop
    rv_expansion_exit: float | None = None    # close if rv_5d/rv_10d exceeds this during trade
    close_friday: bool = False                # close position at Friday close (avoid weekend gap)
    weekend_close_if_loss: bool = False       # close before weekend only if position is underwater
    hedge_on_gap_down: float | None = None    # go short ES when pre-market gap < this (e.g. -1.0)
    hedge_off_gap_threshold: float = -0.3     # lift ES hedge when next day gap >= this (e.g. -0.3%)
    hedge_weekends: bool = False              # short ES Fri close -> Mon open every weekend in trade (downside only)
    spx_drop_from_entry_pct: float | None = None  # close if SPX drops > X% from entry spot (cumulative, not rolling)
    vix_backwardation_exit: bool = False           # close if VIX9D > VIX during hold (term structure inverts)
    vix_drop_exit_pct: float | None = None        # close early if VIX drops > X% from entry VIX (IV contraction profit capture)


@dataclass
class StrategyConfig:
    config_id: str
    strategy: str            # iron_condor | put_vertical | call_vertical
    dte: int
    short_delta: float
    width: float
    exit_rules: ExitRules
    filters: EntryFilters
    adaptive_delta: bool = False  # adjust delta based on VIX level
    # Dynamic sizing: when entry is in "best zone", multiply trade PnL by this factor
    size_mult_in_zone: float = 1.0
    zone_vix_min: float | None = None
    zone_vix_max: float | None = None
    zone_rv_max: float | None = None
    zone_gex_min: int | None = None
    zone_gap_min: float | None = None  # e.g. -0.3 = no gap down worse than -0.3%


# ---------------------------------------------------------------------------
# Enhanced simulation with dynamic IV
# ---------------------------------------------------------------------------
def simulate_enhanced(
    strategy: str,
    spx_close: pd.Series,
    vix_close: pd.Series,
    entry_date: date,
    exp_date: date,
    short_put_delta: float,
    short_call_delta: float,
    width: float,
    exit_rules: ExitRules,
    trading_days: list[date],
    rv_ratio: pd.Series | None = None,
    spx_open: pd.Series | None = None,
    gap_pct: pd.Series | None = None,
    enable_es_hedge: bool = True,
    trading_idx: dict | None = None,
    vix9d: pd.Series | None = None,
) -> dict | None:
    """Simulate a single trade with dynamic IV repricing from VIX."""
    if entry_date not in spx_close.index or exp_date not in spx_close.index:
        return None
    S_entry = float(spx_close[entry_date])
    if S_entry <= 0:
        return None

    if trading_idx is not None:
        entry_idx = trading_idx.get(entry_date)
        exp_idx = trading_idx.get(exp_date)
        if entry_idx is None or exp_idx is None:
            return None
    else:
        entry_idx = trading_days.index(entry_date)
        exp_idx = trading_days.index(exp_date)
    dte_trading = exp_idx - entry_idx
    if dte_trading <= 0:
        return None
    T_entry = dte_trading / 252.0

    sigma_entry = float(vix_close.get(entry_date, 20.0))
    sigma_entry = sigma_entry / 100.0 if sigma_entry > 3 else sigma_entry

    put_credit = 0.0
    K_put_short = K_put_long = None
    call_credit = 0.0
    K_call_short = K_call_long = None
    n_legs = 4

    if strategy != "call_vertical":
        # Put side (iron_condor and put_vertical)
        K_put_short = strike_for_delta(S_entry, T_entry, short_put_delta, sigma_entry, is_put=True)
        if K_put_short is None:
            return None
        K_put_long = K_put_short - width
        if K_put_long <= 0:
            return None
        put_short_px = bs_put_price(S_entry, K_put_short, T_entry, sigma_entry)
        put_long_px = bs_put_price(S_entry, K_put_long, T_entry, sigma_entry)
        put_credit = put_short_px - put_long_px

    if strategy in ("iron_condor", "call_vertical"):
        # Call side (iron_condor and call_vertical)
        K_call_short = strike_for_delta(S_entry, T_entry, short_call_delta, sigma_entry, is_put=False)
        if K_call_short is None:
            return None
        K_call_long = K_call_short + width
        call_short_px = bs_call_price(S_entry, K_call_short, T_entry, sigma_entry)
        call_long_px = bs_call_price(S_entry, K_call_long, T_entry, sigma_entry)
        call_credit = call_short_px - call_long_px
        if strategy == "iron_condor":
            n_legs = 8

    total_credit = put_credit + call_credit
    entry_slippage = n_legs * SLIPPAGE_PER_LEG / 2.0
    credit_adj = total_credit - entry_slippage
    if credit_adj <= 0:
        return None

    if strategy == "iron_condor":
        max_loss = max(width - put_credit, width - call_credit)
    elif strategy == "call_vertical":
        max_loss = width - call_credit
    else:
        max_loss = width - put_credit
    if max_loss <= 0:
        return None

    close_threshold = (1.0 - exit_rules.profit_target_pct) * credit_adj
    stop_threshold = exit_rules.stop_mult * credit_adj

    exit_date = exp_date
    exit_reason = "expiry"
    exit_debit = None
    total_hedge_pnl = 0.0
    hedged = False

    vix_entry_raw = float(vix_close.get(entry_date, sigma_entry * 100.0))

    for i in range(entry_idx + 1, exp_idx):
        d = trading_days[i]
        if d not in spx_close.index:
            continue
        S_d = float(spx_close[d])
        dte_left = exp_idx - i
        T_d = dte_left / 252.0
        if T_d <= 0:
            break

        # Dynamic IV: use VIX on this day instead of entry IV
        sigma_d = float(vix_close.get(d, sigma_entry * 100.0))
        sigma_d = sigma_d / 100.0 if sigma_d > 3 else sigma_d

        spread_val = 0.0
        if K_put_short is not None:
            put_short_d = bs_put_price(S_d, K_put_short, T_d, sigma_d)
            put_long_d = bs_put_price(S_d, K_put_long, T_d, sigma_d)
            spread_val += put_short_d - put_long_d
        if K_call_short is not None:
            call_short_d = bs_call_price(S_d, K_call_short, T_d, sigma_d)
            call_long_d = bs_call_price(S_d, K_call_long, T_d, sigma_d)
            spread_val += call_short_d - call_long_d

        # --- ES downside-only hedge: short ES when open is down, hold for day, re-assess every morning ---
        if enable_es_hedge and exit_rules.hedge_on_gap_down is not None and spx_open is not None and gap_pct is not None:
            open_d = float(spx_open.get(d, S_d))
            prev_day = trading_days[i - 1]
            close_prev = float(spx_close.get(prev_day, S_d))
            if close_prev > 0:
                gap_today = (open_d - close_prev) / close_prev * 100.0
                # Overnight PnL when we were hedged yesterday (held into this morning)
                # SHORT ES: profit = (entry - exit) = (close_prev - open_d)
                if hedged and i > entry_idx + 1:
                    total_hedge_pnl += (close_prev - open_d) * ES_MULT
                # If opening down, put on (or keep) hedge for the day
                if gap_today <= exit_rules.hedge_on_gap_down:
                    hedged = True
                if hedged:
                    total_hedge_pnl += (open_d - S_d) * ES_MULT  # intraday
                    if i + 1 < exp_idx:
                        next_d = trading_days[i + 1]
                        open_next = spx_open.get(next_d)
                        if open_next is not None and not np.isnan(open_next):
                            gap_tomorrow = (float(open_next) - S_d) / S_d * 100.0
                            if gap_tomorrow >= exit_rules.hedge_off_gap_threshold:
                                total_hedge_pnl += (S_d - float(open_next)) * ES_MULT  # overnight then lift (short ES)
                                hedged = False
        # --- Weekend downside hedge: short ES Fri close -> Mon open (black swan weekend gaps) ---
        if enable_es_hedge and exit_rules.hedge_weekends and spx_open is not None and d.weekday() == 4 and i + 1 < exp_idx:
            next_d = trading_days[i + 1]
            open_next = spx_open.get(next_d)
            if open_next is not None and not np.isnan(open_next):
                total_hedge_pnl += (S_d - float(open_next)) * ES_MULT  # short: gain when Mon open < Fri close

        # --- Circuit breakers fire FIRST (pre-empt regular stop for smaller loss) ---

        # Vol expansion circuit breaker (rv_5d/rv_10d ratio spike)
        if exit_rules.rv_expansion_exit is not None and rv_ratio is not None:
            rv_val = float(rv_ratio.get(d, 1.0))
            if not np.isnan(rv_val) and rv_val > exit_rules.rv_expansion_exit:
                exit_date = d
                exit_reason = f"rv_expand_{rv_val:.2f}"
                exit_debit = spread_val
                break

        # VIX spike circuit breaker
        if exit_rules.vix_spike_exit_pct is not None and vix_entry_raw > 0:
            vix_now = float(vix_close.get(d, vix_entry_raw))
            vix_chg = (vix_now - vix_entry_raw) / vix_entry_raw * 100
            if vix_chg >= exit_rules.vix_spike_exit_pct:
                exit_date = d
                exit_reason = f"vix_spike_{int(vix_chg)}pct"
                exit_debit = spread_val
                break

        # SPX acceleration circuit breaker
        if exit_rules.spx_drop_exit_pct is not None:
            lb = exit_rules.spx_drop_lookback
            if i >= lb:
                d_lb = trading_days[i - lb]
                if d_lb in spx_close.index:
                    S_lb = float(spx_close[d_lb])
                    if S_lb > 0:
                        spx_chg = (S_d - S_lb) / S_lb * 100
                        if spx_chg <= -exit_rules.spx_drop_exit_pct:
                            exit_date = d
                            exit_reason = f"spx_drop_{abs(spx_chg):.1f}pct"
                            exit_debit = spread_val
                            break

        # SPX cumulative drop from entry (intra-trade momentum break)
        if exit_rules.spx_drop_from_entry_pct is not None and S_entry > 0:
            drop_from_entry = (S_d - S_entry) / S_entry * 100
            if drop_from_entry <= -exit_rules.spx_drop_from_entry_pct:
                exit_date = d
                exit_reason = f"spx_entry_drop_{abs(drop_from_entry):.1f}pct"
                exit_debit = spread_val
                break

        # VIX9D > VIX backwardation exit (front vol elevated vs 30d = stress signal)
        if exit_rules.vix_backwardation_exit and vix9d is not None:
            vix9d_now = float(vix9d.get(d, np.nan))
            vix_now_ts = float(vix_close.get(d, np.nan))
            if not np.isnan(vix9d_now) and not np.isnan(vix_now_ts) and vix9d_now > vix_now_ts:
                exit_date = d
                exit_reason = "vix9d_backwardation"
                exit_debit = spread_val
                break

        # IV contraction profit capture: close if VIX drops X% from entry VIX
        # When vol sold at high VIX normalizes, book the vega gain early
        if exit_rules.vix_drop_exit_pct is not None and vix_entry_raw > 0:
            vix_now = float(vix_close.get(d, vix_entry_raw))
            vix_chg = (vix_now - vix_entry_raw) / vix_entry_raw * 100
            if vix_chg <= -exit_rules.vix_drop_exit_pct:
                exit_date = d
                exit_reason = f"vix_drop_{abs(vix_chg):.0f}pct"
                exit_debit = spread_val
                break

        # --- Weekend management exits ---
        day_of_week = d.weekday()  # 0=Mon, 4=Fri

        # Unconditional Friday close: exit at Friday close to avoid weekend gap
        if exit_rules.close_friday and day_of_week == 4:
            exit_date = d
            exit_reason = "fri_close"
            exit_debit = spread_val
            break

        # Conditional weekend close: exit on Thu/Fri only if position is underwater
        if exit_rules.weekend_close_if_loss and day_of_week >= 3:
            if spread_val > credit_adj:
                exit_date = d
                exit_reason = "wknd_loss_close"
                exit_debit = spread_val
                break

        # --- Standard exits ---
        if spread_val <= close_threshold:
            exit_date = d
            exit_reason = f"tp_{int(exit_rules.profit_target_pct * 100)}"
            exit_debit = spread_val
            break
        if spread_val >= stop_threshold:
            exit_date = d
            exit_reason = f"sl_{exit_rules.stop_mult}"
            exit_debit = spread_val
            break
        if exit_rules.time_stop_dte is not None and dte_left <= exit_rules.time_stop_dte:
            exit_date = d
            exit_reason = f"time_stop_{exit_rules.time_stop_dte}dte"
            exit_debit = spread_val
            break

    S_exp = float(spx_close[exp_date])
    if exit_reason == "expiry":
        settlement = 0.0
        if K_put_short is not None:
            put_settlement = max(0.0, K_put_short - S_exp) - max(0.0, K_put_long - S_exp)
            settlement += put_settlement
        if K_call_short is not None:
            call_settlement = max(0.0, S_exp - K_call_short) - max(0.0, S_exp - K_call_long)
            settlement += call_settlement
        commission = (n_legs / 2) * COMMISSION_PER_LEG
        pnl = (credit_adj - settlement) * MULT - commission
    else:
        exit_slippage = n_legs * SLIPPAGE_PER_LEG / 2.0
        debit_adj = (exit_debit or 0) + exit_slippage
        commission = n_legs * COMMISSION_PER_LEG
        pnl = (credit_adj - debit_adj) * MULT - commission

    pnl += total_hedge_pnl

    return {
        "entry_date": entry_date,
        "exit_date": exit_date,
        "exp_date": exp_date,
        "dte": dte_trading,
        "spot_entry": S_entry,
        "iv_entry": sigma_entry,
        "entry_credit": credit_adj,
        "exit_reason": exit_reason,
        "pnl": pnl,
        "win": pnl > 0,
    }


# ---------------------------------------------------------------------------
# Adaptive delta lookup
# ---------------------------------------------------------------------------
def adaptive_delta_for_vix(vix_level: float, base_delta: float) -> float:
    """Shift delta based on VIX regime — go further OTM when vol is high."""
    if vix_level > 25:
        return max(0.08, base_delta - 0.05)
    if vix_level > 20:
        return max(0.08, base_delta - 0.03)
    if vix_level < 14:
        return min(0.25, base_delta + 0.03)
    return base_delta


# ---------------------------------------------------------------------------
# Run backtest with filters
# ---------------------------------------------------------------------------
def run_filtered_backtest(
    configs: list[StrategyConfig],
    spx_close: pd.Series,
    vix_close: pd.Series,
    spx_df: pd.DataFrame,
    trading: list[date],
    expirations: list[date],
    gex_scores: pd.Series | None = None,
    rv_ratio: pd.Series | None = None,
    gap_pct: pd.Series | None = None,
    spx_open: pd.Series | None = None,
    news_dates: set[date] | None = None,
    enable_es_hedge: bool = True,
    vix9d: pd.Series | None = None,
    entry_filter_model_path: Path | None = None,
    entry_filter_threshold: float = 0.5,
    entry_filter_optional_series: dict[str, pd.Series] | None = None,
    entry_filter_log_path: Path | None = None,
) -> pd.DataFrame:
    """Run all configs with entry filters and return trade-level DataFrame.

    If entry_filter_model_path is set, load the XGBoost entry-risk model and skip
    trades where P(loss) > entry_filter_threshold (reduces drawdowns).
    If entry_filter_log_path is set, append one row per candidate entry (entry_date,
    config_id, p_loss, decision, key features) when the filter is active.
    """
    trading_set = set(trading)
    exp_list = [e for e in expirations if e in trading_set]
    trading_idx: dict[date, int] = {d: i for i, d in enumerate(trading)}

    entry_filter_model: Any = None
    entry_filter_feature_cols: list[str] | None = None
    entry_filter_config_encoder: Any = None
    if entry_filter_model_path and Path(entry_filter_model_path).exists():
        try:
            import joblib
            payload = joblib.load(entry_filter_model_path)
            entry_filter_model = payload.get("model")
            entry_filter_feature_cols = payload.get("feature_cols")
            entry_filter_config_encoder = payload.get("config_encoder")
            if entry_filter_model is None or not entry_filter_feature_cols:
                entry_filter_model = None
                entry_filter_feature_cols = None
        except Exception:
            entry_filter_model = None
            entry_filter_feature_cols = None
            entry_filter_config_encoder = None

    entry_filter_log_file = None
    if entry_filter_log_path and entry_filter_model is not None:
        try:
            entry_filter_log_path.parent.mkdir(parents=True, exist_ok=True)
            write_header = not entry_filter_log_path.exists()
            entry_filter_log_file = open(entry_filter_log_path, "a", newline="", encoding="utf-8")
            if write_header:
                import csv
                csv.writer(entry_filter_log_file).writerow(
                    ["entry_date", "config_id", "p_loss", "decision", "vix_at_entry", "regime", "vix_change_5d"]
                )
                entry_filter_log_file.flush()
        except Exception:
            entry_filter_log_file = None

    # Precompute SMA series at various periods
    sma_cache: dict[int, pd.Series] = {}
    for cfg in configs:
        p = cfg.filters.trend_sma_period
        if p and p not in sma_cache:
            sma = spx_df["Close"].rolling(p).mean()
            sma.index = pd.to_datetime(sma.index).tz_localize(None).normalize()
            sma.index = sma.index.date
            sma_cache[p] = sma

    # Precompute VIX 5-day pct change (pct_change(5) compares to 5 trading days prior)
    vix_pct_5d = vix_close.pct_change(5) * 100 if not vix_close.empty else pd.Series(dtype=float)

    rows: list[dict] = []

    for cfg in configs:
        open_positions: list[dict] = []
        consecutive_losses = 0
        cooldown_remaining = 0

        for exp in exp_list:
            exp_pos = trading_idx.get(exp)
            if exp_pos is None:
                continue
            if exp_pos < cfg.dte:
                continue
            entry_date = trading[exp_pos - cfg.dte]
            if entry_date not in spx_close.index or exp not in spx_close.index:
                continue

            # Close expired positions from tracking (exp_date == entry_date settles at close, after entry)
            open_positions = [p for p in open_positions if p["exp_date"] > entry_date]

            # --- Apply filters ---
            vix_val = float(vix_close.get(entry_date, 20.0))

            # VIX band
            if cfg.filters.vix_max is not None and vix_val > cfg.filters.vix_max:
                continue
            if cfg.filters.vix_min is not None and vix_val < cfg.filters.vix_min:
                continue

            # VIX spike
            if cfg.filters.vix_spike_5d_pct is not None and entry_date in vix_pct_5d.index:
                spike = float(vix_pct_5d.get(entry_date, 0.0))
                if not np.isnan(spike) and spike > cfg.filters.vix_spike_5d_pct:
                    continue

            # Trend filter
            if cfg.filters.trend_sma_period is not None:
                sma = sma_cache.get(cfg.filters.trend_sma_period)
                if sma is not None and entry_date in sma.index:
                    ma_val = sma.get(entry_date)
                    if ma_val is not None and not np.isnan(ma_val):
                        if cfg.filters.trend_sma_invert:
                            # Bearish: enter only when SPX is BELOW SMA
                            if float(spx_close[entry_date]) > float(ma_val):
                                continue
                        else:
                            # Bullish: enter only when SPX is ABOVE SMA
                            if float(spx_close[entry_date]) <= float(ma_val):
                                continue

            # Max concurrent
            if cfg.filters.max_concurrent is not None:
                if len(open_positions) >= cfg.filters.max_concurrent:
                    continue

            # Consecutive loss cooldown
            if cooldown_remaining > 0:
                cooldown_remaining -= 1
                continue

            # GEX proxy circuit breaker
            if cfg.filters.gex_min_score is not None and gex_scores is not None:
                gex_score = int(gex_scores.get(entry_date, 0))
                if gex_score < cfg.filters.gex_min_score:
                    continue

            # Vol expansion entry filter (skip when short-term vol > medium-term vol)
            if cfg.filters.rv_expansion_max is not None and rv_ratio is not None:
                rv_val = float(rv_ratio.get(entry_date, 1.0))
                if not np.isnan(rv_val) and rv_val > cfg.filters.rv_expansion_max:
                    continue

            # No entry on pre-market gap down (e.g. skip when gap <= -0.5%)
            if cfg.filters.gap_down_no_entry is not None and gap_pct is not None:
                gap_val = float(gap_pct.get(entry_date, 0.0))
                if not np.isnan(gap_val) and gap_val < cfg.filters.gap_down_no_entry:
                    continue

            # Risk-off for news: skip entry on high-impact release days
            if cfg.filters.skip_news_days and news_dates is not None and entry_date in news_dates:
                continue

            # Regime at entry (for dynamic sizing zone check)
            rv_val = float(rv_ratio.get(entry_date, 1.0)) if rv_ratio is not None else 1.0
            if rv_ratio is not None and entry_date in rv_ratio.index and np.isnan(rv_val):
                rv_val = 1.0
            gex_score = int(gex_scores.get(entry_date, 0)) if gex_scores is not None else 0
            gap_val = float(gap_pct.get(entry_date, 0.0)) if gap_pct is not None else 0.0
            if gap_pct is not None and (entry_date not in gap_pct.index or np.isnan(gap_val)):
                gap_val = 0.0

            # --- Determine delta ---
            delta = cfg.short_delta
            if cfg.adaptive_delta:
                delta = adaptive_delta_for_vix(vix_val, cfg.short_delta)

            # --- Simulate ---
            result = simulate_enhanced(
                strategy=cfg.strategy,
                spx_close=spx_close,
                vix_close=vix_close,
                entry_date=entry_date,
                exp_date=exp,
                short_put_delta=delta,
                short_call_delta=delta,
                width=cfg.width,
                exit_rules=cfg.exit_rules,
                trading_days=trading,
                rv_ratio=rv_ratio,
                spx_open=spx_open,
                gap_pct=gap_pct,
                enable_es_hedge=enable_es_hedge,
                trading_idx=trading_idx,
                vix9d=vix9d,
            )
            if result is None:
                continue

            # Dynamic sizing: scale PnL when entry is in "best zone"
            pnl = result["pnl"]
            if cfg.size_mult_in_zone != 1.0:
                in_zone = True
                if cfg.zone_vix_min is not None:
                    in_zone = in_zone and (vix_val >= cfg.zone_vix_min)
                if cfg.zone_vix_max is not None:
                    in_zone = in_zone and (vix_val <= cfg.zone_vix_max)
                if cfg.zone_rv_max is not None:
                    in_zone = in_zone and (rv_val <= cfg.zone_rv_max)
                if cfg.zone_gex_min is not None:
                    in_zone = in_zone and (gex_score >= cfg.zone_gex_min)
                if cfg.zone_gap_min is not None:
                    in_zone = in_zone and (gap_val >= cfg.zone_gap_min)
                if in_zone:
                    pnl = pnl * cfg.size_mult_in_zone

            # XGBoost entry filter: skip if P(loss) > threshold
            if entry_filter_model is not None and entry_filter_feature_cols is not None:
                gap_ser = gap_pct if gap_pct is not None else pd.Series(dtype=float)
                rv_ser = rv_ratio if rv_ratio is not None else pd.Series(dtype=float)
                if entry_filter_optional_series is not None:
                    feats = compute_entry_features_extended(
                        entry_date, spx_close, vix_close, gap_ser, rv_ser,
                        optional_series=entry_filter_optional_series,
                        iv_at_entry=result["iv_entry"],
                        gex_score_at_entry=gex_score,
                        rv_ratio_at_entry=rv_val,
                    )
                else:
                    feats = compute_rolling_features_for_entry(
                        entry_date, spx_close, vix_close, gap_ser, rv_ser,
                    )
                entry_year = getattr(entry_date, "year", 0)
                entry_month = getattr(entry_date, "month", 0)
                entry_day_of_week = getattr(entry_date, "weekday", lambda: 0)()
                feat_row = {
                    "spot_entry": result["spot_entry"],
                    "vix_at_entry": vix_val,
                    "gap_pct_at_entry": gap_val,
                    "gex_score_at_entry": gex_score,
                    "rv_ratio_at_entry": rv_val,
                    "entry_year": entry_year,
                    "entry_month": entry_month,
                    "entry_day_of_week": entry_day_of_week,
                    "dte": cfg.dte,
                    "short_delta": delta,
                    "width": cfg.width,
                    "profit_target": cfg.exit_rules.profit_target_pct,
                    "stop_mult": cfg.exit_rules.stop_mult,
                    "iv_entry": result["iv_entry"],
                    "entry_credit": result["entry_credit"],
                    **feats,
                }
                if "config_id" in entry_filter_feature_cols:
                    cfg_id = cfg.config_id
                    if entry_filter_config_encoder is not None:
                        try:
                            cfg_id = int(entry_filter_config_encoder.transform([str(cfg_id)])[0])
                        except Exception:
                            cfg_id = -1
                    feat_row["config_id"] = cfg_id
                X = pd.DataFrame([{c: feat_row.get(c, np.nan) for c in entry_filter_feature_cols}])
                try:
                    p_loss = entry_filter_model.predict_proba(X)[0, 1]
                except Exception:
                    p_loss = np.nan
                if entry_filter_log_file is not None:
                    try:
                        import csv
                        decision = 0 if (not np.isnan(p_loss) and p_loss > entry_filter_threshold) else 1
                        csv.writer(entry_filter_log_file).writerow([
                            entry_date.isoformat(),
                            cfg.config_id,
                            p_loss if not np.isnan(p_loss) else "",
                            decision,
                            feat_row.get("vix_at_entry", ""),
                            feat_row.get("regime", ""),
                            feat_row.get("vix_change_5d", ""),
                        ])
                        entry_filter_log_file.flush()
                    except Exception:
                        pass
                if not np.isnan(p_loss) and p_loss > entry_filter_threshold:
                    continue

            open_positions.append({"exp_date": exp, "entry_date": entry_date})

            # Track consecutive losses (use raw result pnl for streak)
            if result["pnl"] > 0:
                consecutive_losses = 0
            else:
                consecutive_losses += 1
                if (cfg.filters.consecutive_loss_cooldown is not None
                        and consecutive_losses >= cfg.filters.consecutive_loss_trigger):
                    cooldown_remaining = cfg.filters.consecutive_loss_cooldown
                    consecutive_losses = 0

            rows.append({
                "config_id": cfg.config_id,
                "strategy": cfg.strategy,
                "dte": cfg.dte,
                "short_delta": delta,
                "width": cfg.width,
                "profit_target": cfg.exit_rules.profit_target_pct,
                "stop_mult": cfg.exit_rules.stop_mult,
                "entry_date": result["entry_date"],
                "exit_date": result["exit_date"],
                "exp_date": result["exp_date"],
                "pnl": pnl,
                "win": pnl > 0,
                "exit_reason": result["exit_reason"],
                "iv_entry": result["iv_entry"],
                "entry_credit": result["entry_credit"],
                "spot_entry": result["spot_entry"],
                "vix_at_entry": vix_val,
                "gap_pct_at_entry": gap_val,
                "gex_score_at_entry": gex_score,
                "rv_ratio_at_entry": rv_val,
            })

    if entry_filter_log_file is not None:
        try:
            entry_filter_log_file.close()
        except Exception:
            pass
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Enhanced metrics
# ---------------------------------------------------------------------------
def compute_enhanced_metrics(trades: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Compute monthly P&L, equity curves, and comprehensive risk metrics."""
    if trades.empty:
        return pd.DataFrame(), pd.DataFrame(), {}

    trades = trades.copy()
    trades["exit_dt"] = pd.to_datetime(trades["exit_date"])
    trades["year_month"] = trades["exit_dt"].dt.to_period("M").astype(str)
    trades["year"] = trades["exit_dt"].dt.year
    trades["month"] = trades["exit_dt"].dt.month

    monthly = trades.groupby(["config_id", "year_month"], as_index=False).agg(
        pnl=("pnl", "sum"),
        year=("year", "first"),
        month=("month", "first"),
    )

    # Equity curve
    trades_sorted = trades.sort_values(["config_id", "exit_date"])
    equity_rows: list[dict] = []
    for config_id, grp in trades_sorted.groupby("config_id"):
        grp = grp.sort_values("exit_date")
        cum = grp["pnl"].cumsum()
        peak = cum.cummax()
        dd = peak - cum
        for i, (_, row) in enumerate(grp.iterrows()):
            equity_rows.append({
                "config_id": config_id,
                "exit_date": row["exit_date"],
                "cum_pnl": cum.iloc[i],
                "peak": peak.iloc[i],
                "drawdown": dd.iloc[i],
            })
    equity_df = pd.DataFrame(equity_rows)

    summary: dict[str, dict] = {}
    for config_id in trades["config_id"].unique():
        pnls = trades.loc[trades["config_id"] == config_id, "pnl"].values
        n = len(pnls)
        if n == 0:
            summary[config_id] = _empty_summary()
            continue

        monthly_cfg = monthly[monthly["config_id"] == config_id].sort_values("year_month")
        monthly_pnls = monthly_cfg["pnl"].values
        n_months = len(monthly_pnls)

        total_pnl = float(np.sum(pnls))
        avg_monthly = total_pnl / n_months if n_months else 0.0
        worst_month = float(np.min(monthly_pnls)) if n_months else 0.0
        best_month = float(np.max(monthly_pnls)) if n_months else 0.0
        profitable_months = int((monthly_pnls > 0).sum())
        pct_prof_months = 100.0 * profitable_months / n_months if n_months else 0.0

        # Per-trade metrics
        std_pnl = float(np.std(pnls)) if n > 1 else 0.0
        trade_sharpe = float(np.mean(pnls)) / std_pnl if std_pnl > 0 else 0.0

        # Monthly annualized Sharpe
        monthly_mean = float(np.mean(monthly_pnls)) if n_months else 0.0
        monthly_std = float(np.std(monthly_pnls, ddof=1)) if n_months > 1 else 0.0
        ann_sharpe = (monthly_mean / monthly_std) * math.sqrt(12) if monthly_std > 0 else 0.0

        # Sortino (downside deviation from monthly)
        neg_monthly = monthly_pnls[monthly_pnls < 0]
        downside_std = float(np.std(neg_monthly, ddof=1)) if len(neg_monthly) > 1 else (float(np.std(neg_monthly)) if len(neg_monthly) == 1 else 0.0)
        sortino = (monthly_mean / downside_std) * math.sqrt(12) if downside_std > 0 else 0.0

        # Max drawdown
        equity_cfg = equity_df[equity_df["config_id"] == config_id]
        max_dd = float(equity_cfg["drawdown"].max()) if not equity_cfg.empty else 0.0

        # Calmar (annualized return / max DD)
        years = n_months / 12.0 if n_months else 1.0
        ann_return = total_pnl / years if years > 0 else 0.0
        calmar = ann_return / max_dd if max_dd > 0 else 0.0

        # Profit factor
        wins = pnls[pnls > 0]
        losses = pnls[pnls < 0]
        gross_win = float(np.sum(wins)) if len(wins) else 0.0
        gross_loss = abs(float(np.sum(losses))) if len(losses) else 1e-9
        pf = gross_win / gross_loss if gross_loss > 0 else 0.0

        # CVaR 5%
        k = max(1, int(np.ceil(n * 0.05)))
        cvar_5 = float(np.mean(np.sort(pnls)[:k]))

        # Recovery factor (total pnl / max DD) = capital efficiency (profit per $ of max DD)
        recovery = total_pnl / max_dd if max_dd > 0 else 0.0
        capital_efficiency = recovery  # same: total P&L per dollar of max drawdown

        win_rate = 100.0 * (pnls > 0).sum() / n

        summary[config_id] = {
            "n_trades": n,
            "total_pnl": round(total_pnl, 2),
            "avg_monthly_pnl": round(avg_monthly, 2),
            "worst_month": round(worst_month, 2),
            "best_month": round(best_month, 2),
            "pct_profitable_months": round(pct_prof_months, 1),
            "win_rate": round(win_rate, 1),
            "profit_factor": round(pf, 2),
            "trade_sharpe": round(trade_sharpe, 4),
            "ann_sharpe": round(ann_sharpe, 4),
            "sortino": round(sortino, 4),
            "calmar": round(calmar, 4),
            "max_drawdown": round(max_dd, 2),
            "cvar_5": round(cvar_5, 2),
            "recovery_factor": round(recovery, 2),
            "capital_efficiency": round(capital_efficiency, 2),
            "avg_pnl_per_trade": round(float(np.mean(pnls)), 2),
            "worst_trade": round(float(np.min(pnls)), 2),
        }

    return monthly, equity_df, summary


def _empty_summary() -> dict:
    return {k: 0 for k in [
        "n_trades", "total_pnl", "avg_monthly_pnl", "worst_month", "best_month",
        "pct_profitable_months", "win_rate", "profit_factor", "trade_sharpe",
        "ann_sharpe", "sortino", "calmar", "max_drawdown", "cvar_5",
        "recovery_factor", "capital_efficiency", "avg_pnl_per_trade", "worst_trade",
    ]}


# ---------------------------------------------------------------------------
# Ensemble portfolio metrics
# ---------------------------------------------------------------------------
def compute_ensemble_metrics(
    trades: pd.DataFrame,
    config_weights: dict[str, float],
    ensemble_id: str = "ENS",
) -> dict[str, Any]:
    """Combine selected configs into a single portfolio; return summary metrics.
    config_weights: config_id -> weight (will be normalized to sum to 1).
    """
    if trades.empty or not config_weights:
        return _empty_summary()
    subset = trades[trades["config_id"].isin(config_weights)].copy()
    if subset.empty:
        return _empty_summary()
    total_w = sum(config_weights.get(c, 0.0) for c in subset["config_id"].unique())
    if total_w <= 0:
        return _empty_summary()
    weights = {c: config_weights[c] / total_w for c in config_weights}
    subset["weighted_pnl"] = subset.apply(lambda r: weights.get(r["config_id"], 0.0) * r["pnl"], axis=1)
    by_date = subset.groupby("exit_date", as_index=False)["weighted_pnl"].sum()
    by_date = by_date.sort_values("exit_date")
    portfolio_pnls = by_date["weighted_pnl"].values
    n_trades = len(subset)
    total_pnl = float(by_date["weighted_pnl"].sum())
    cum = np.cumsum(portfolio_pnls)
    peak = np.maximum.accumulate(cum)
    dd = peak - cum
    max_dd = float(np.max(dd)) if len(dd) else 0.0
    by_date["exit_dt"] = pd.to_datetime(by_date["exit_date"])
    by_date["year_month"] = by_date["exit_dt"].dt.to_period("M").astype(str)
    monthly = by_date.groupby("year_month", as_index=False)["weighted_pnl"].sum()
    monthly_pnls = monthly["weighted_pnl"].values
    n_months = len(monthly_pnls)
    avg_monthly = total_pnl / n_months if n_months else 0.0
    worst_month = float(np.min(monthly_pnls)) if n_months else 0.0
    best_month = float(np.max(monthly_pnls)) if n_months else 0.0
    profitable_months = int((monthly_pnls > 0).sum())
    pct_prof_months = 100.0 * profitable_months / n_months if n_months else 0.0
    monthly_mean = float(np.mean(monthly_pnls)) if n_months else 0.0
    monthly_std = float(np.std(monthly_pnls, ddof=1)) if n_months > 1 else 0.0
    ann_sharpe = (monthly_mean / monthly_std) * math.sqrt(12) if monthly_std > 0 else 0.0
    neg_monthly = monthly_pnls[monthly_pnls < 0]
    downside_std = float(np.std(neg_monthly, ddof=1)) if len(neg_monthly) > 1 else (float(np.std(neg_monthly)) if len(neg_monthly) == 1 else 0.0)
    sortino = (monthly_mean / downside_std) * math.sqrt(12) if downside_std > 0 else 0.0
    years = n_months / 12.0 if n_months else 1.0
    ann_return = total_pnl / years if years > 0 else 0.0
    calmar = ann_return / max_dd if max_dd > 0 else 0.0
    wins = portfolio_pnls[portfolio_pnls > 0]
    losses = portfolio_pnls[portfolio_pnls < 0]
    gross_win = float(np.sum(wins)) if len(wins) else 0.0
    gross_loss = abs(float(np.sum(losses))) if len(losses) else 1e-9
    pf = gross_win / gross_loss if gross_loss > 0 else 0.0
    all_pnls = subset["weighted_pnl"].values
    k = max(1, int(np.ceil(len(all_pnls) * 0.05)))
    cvar_5 = float(np.mean(np.sort(all_pnls)[:k]))
    win_rate = 100.0 * (all_pnls > 0).sum() / len(all_pnls) if len(all_pnls) else 0.0
    return {
        "n_trades": n_trades,
        "total_pnl": round(total_pnl, 2),
        "avg_monthly_pnl": round(avg_monthly, 2),
        "worst_month": round(worst_month, 2),
        "best_month": round(best_month, 2),
        "pct_profitable_months": round(pct_prof_months, 1),
        "win_rate": round(win_rate, 1),
        "profit_factor": round(pf, 2),
        "trade_sharpe": 0.0,
        "ann_sharpe": round(ann_sharpe, 4),
        "sortino": round(sortino, 4),
        "calmar": round(calmar, 4),
        "max_drawdown": round(max_dd, 2),
        "cvar_5": round(cvar_5, 2),
        "recovery_factor": round(total_pnl / max_dd, 2) if max_dd > 0 else 0.0,
        "capital_efficiency": round(total_pnl / max_dd, 2) if max_dd > 0 else 0.0,
        "avg_pnl_per_trade": round(float(np.mean(all_pnls)), 2) if len(all_pnls) else 0.0,
        "worst_trade": round(float(np.min(all_pnls)), 2) if len(all_pnls) else 0.0,
    }


# ---------------------------------------------------------------------------
# Report builder
# ---------------------------------------------------------------------------
def build_enhanced_report(
    trades: pd.DataFrame,
    monthly: pd.DataFrame,
    equity_df: pd.DataFrame,
    summary: dict,
    output_path: Path,
    report_title: str | None = None,
    *,
    trades_filtered: pd.DataFrame | None = None,
    monthly_filtered: pd.DataFrame | None = None,
    equity_filtered: pd.DataFrame | None = None,
    summary_filtered: dict | None = None,
    entry_filter_label: str | None = None,
) -> None:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    config_ids = sorted(trades["config_id"].unique()) if not trades.empty else []
    palette = [
        "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
        "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf",
        "#636efa", "#ef553b", "#00cc96", "#ab63fa", "#ffa15a",
        "#19d3f3",
    ]
    cmap = {c: palette[i % len(palette)] for i, c in enumerate(config_ids)}
    figs: list[go.Figure] = []

    # 0. Entry filter comparison (baseline vs filtered)
    if summary_filtered is not None and entry_filter_label is not None:
        skeys = sorted(summary.keys(), key=lambda k: summary[k]["ann_sharpe"], reverse=True)
        # Table: Config | Baseline Trades | Filtered Trades | Baseline PnL | Filtered PnL | Baseline Max DD | Filtered Max DD | Baseline Sharpe | Filtered Sharpe
        header_comp = [
            "Config", "Trades (base)", "Trades (filter)", "Total P&L (base)", "Total P&L (filter)",
            "Max DD (base)", "Max DD (filter)", "Sharpe (base)", "Sharpe (filter)",
        ]
        cell_comp = [
            skeys,
            [summary[k]["n_trades"] for k in skeys],
            [summary_filtered.get(k, {}).get("n_trades", 0) for k in skeys],
            [f"${summary[k]['total_pnl']:,.0f}" for k in skeys],
            [f"${summary_filtered.get(k, {}).get('total_pnl', 0):,.0f}" for k in skeys],
            [f"${summary[k]['max_drawdown']:,.0f}" for k in skeys],
            [f"${summary_filtered.get(k, {}).get('max_drawdown', 0):,.0f}" for k in skeys],
            [f"{summary[k]['ann_sharpe']:.3f}" for k in skeys],
            [f"{summary_filtered.get(k, {}).get('ann_sharpe', 0):.3f}" for k in skeys],
        ]
        fig_comp = go.Figure(data=[go.Table(
            header=dict(values=header_comp, fill_color="lavender", align="left", font=dict(size=11)),
            cells=dict(values=cell_comp, align="left", font=dict(size=10)),
        )])
        fig_comp.update_layout(
            title=f"Entry Filter Comparison — Baseline vs {entry_filter_label}",
            height=400 + min(len(skeys), 30) * 18,
        )
        figs.append(fig_comp)
        # Portfolio equity: baseline vs filtered (all trades combined by exit_date)
        if not trades.empty and trades_filtered is not None and not trades_filtered.empty:
            base_by_date = trades.sort_values("exit_date").groupby("exit_date")["pnl"].sum().cumsum()
            filt_by_date = trades_filtered.sort_values("exit_date").groupby("exit_date")["pnl"].sum().cumsum()
            fig_port = go.Figure()
            fig_port.add_trace(go.Scatter(
                x=base_by_date.index, y=base_by_date.values, mode="lines",
                name="Baseline (no filter)", line=dict(color="#1f77b4", width=2),
            ))
            fig_port.add_trace(go.Scatter(
                x=filt_by_date.index, y=filt_by_date.values, mode="lines",
                name=entry_filter_label, line=dict(color="#2ca02c", width=2),
            ))
            fig_port.update_layout(
                title="Portfolio Cumulative P&L — Baseline vs Entry Filter",
                xaxis_title="Date", yaxis_title="Cumulative P&L ($)",
                legend=dict(orientation="h", yanchor="bottom", y=1.02),
                height=450,
            )
            figs.append(fig_port)

    # 1. Equity curves
    fig_eq = go.Figure()
    for cid in config_ids:
        eq = equity_df[equity_df["config_id"] == cid].sort_values("exit_date")
        if eq.empty:
            continue
        fig_eq.add_trace(go.Scatter(
            x=eq["exit_date"], y=eq["cum_pnl"], mode="lines",
            name=cid, line=dict(color=cmap[cid], width=2),
        ))
    fig_eq.update_layout(
        title="Equity Curves — Cumulative P&L ($, per 1-lot)",
        xaxis_title="Date", yaxis_title="Cumulative P&L ($)",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        height=500,
    )
    figs.append(fig_eq)

    # 2. Sharpe comparison bar chart
    if summary:
        skeys = sorted(summary.keys())
        fig_sharpe = go.Figure()
        fig_sharpe.add_trace(go.Bar(
            x=skeys,
            y=[summary[k]["ann_sharpe"] for k in skeys],
            name="Annualized Sharpe",
            marker_color=[cmap.get(k, "#333") for k in skeys],
        ))
        fig_sharpe.update_layout(
            title="Annualized Sharpe Ratio (from monthly returns)",
            yaxis_title="Sharpe", height=400,
        )
        figs.append(fig_sharpe)

    # 3. Sortino comparison
    if summary:
        skeys = sorted(summary.keys())
        fig_sort = go.Figure()
        fig_sort.add_trace(go.Bar(
            x=skeys,
            y=[summary[k]["sortino"] for k in skeys],
            name="Sortino",
            marker_color=[cmap.get(k, "#333") for k in skeys],
        ))
        fig_sort.update_layout(
            title="Sortino Ratio (annualized, downside vol only)",
            yaxis_title="Sortino", height=400,
        )
        figs.append(fig_sort)

    # 4. Drawdown comparison
    fig_dd = go.Figure()
    for cid in config_ids:
        eq = equity_df[equity_df["config_id"] == cid].sort_values("exit_date")
        if eq.empty:
            continue
        fig_dd.add_trace(go.Scatter(
            x=eq["exit_date"], y=-eq["drawdown"], mode="lines",
            name=cid, line=dict(color=cmap[cid], width=2),
        ))
    fig_dd.update_layout(
        title="Drawdown (negative = underwater)",
        xaxis_title="Date", yaxis_title="Drawdown ($)",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        height=450,
    )
    figs.append(fig_dd)

    # 5. PnL distribution histograms (overlay)
    if not trades.empty:
        fig_hist = go.Figure()
        for cid in config_ids:
            pnls = trades.loc[trades["config_id"] == cid, "pnl"].values
            fig_hist.add_trace(go.Histogram(
                x=pnls, name=cid, opacity=0.5,
                marker_color=cmap.get(cid, "#333"),
                nbinsx=60,
            ))
        fig_hist.update_layout(
            barmode="overlay",
            title="Trade P&L Distribution",
            xaxis_title="P&L ($)", yaxis_title="Count",
            height=400,
        )
        figs.append(fig_hist)

    # 6. Monthly P&L heatmap for top 4 configs by ann_sharpe
    if summary and not monthly.empty:
        top_ids = sorted(summary.keys(), key=lambda k: summary[k]["ann_sharpe"], reverse=True)[:4]
        n_cols = 2
        n_rows = 2
        fig_heat = make_subplots(rows=n_rows, cols=n_cols, subplot_titles=top_ids,
                                  vertical_spacing=0.12, horizontal_spacing=0.08)
        for idx, cid in enumerate(top_ids):
            m = monthly[monthly["config_id"] == cid]
            if m.empty:
                continue
            r, c = idx // n_cols + 1, idx % n_cols + 1
            pivot = m.pivot_table(index="year", columns="month", values="pnl", aggfunc="sum")
            pivot = pivot.reindex(columns=range(1, 13), fill_value=np.nan)
            fig_heat.add_trace(
                go.Heatmap(z=pivot.values, x=pivot.columns, y=pivot.index,
                           colorscale="RdYlGn", zmid=0,
                           colorbar=dict(len=0.4, y=1 - (r - 0.5) / n_rows)),
                row=r, col=c,
            )
        fig_heat.update_layout(title="Monthly P&L Heatmap (Top 4 by Sharpe)", height=600)
        figs.append(fig_heat)

    # 7. Win rate by year for top configs
    if not trades.empty:
        top_ids = sorted(summary.keys(), key=lambda k: summary[k]["ann_sharpe"], reverse=True)[:6]
        tc = trades[trades["config_id"].isin(top_ids)].copy()
        tc["year"] = pd.to_datetime(tc["exit_date"]).dt.year
        wr = tc.groupby(["config_id", "year"]).agg(
            wins=("win", "sum"), n=("win", "count")).reset_index()
        wr["win_rate_pct"] = 100.0 * wr["wins"] / wr["n"]
        fig_wr = go.Figure()
        for cid in top_ids:
            w = wr[wr["config_id"] == cid]
            if w.empty:
                continue
            fig_wr.add_trace(go.Bar(x=w["year"], y=w["win_rate_pct"], name=cid))
        fig_wr.update_layout(barmode="group", title="Win Rate by Year (Top configs by Sharpe)",
                              yaxis_title="Win Rate (%)", height=400)
        figs.append(fig_wr)

    # 8. Summary table
    if summary:
        skeys = sorted(summary.keys(), key=lambda k: summary[k]["ann_sharpe"], reverse=True)
        header_vals = [
            "Config", "Trades", "Total P&L", "Avg Monthly", "Win %", "PF",
            "Trade Sharpe", "Ann Sharpe", "Sortino", "Calmar",
            "Max DD", "Worst Trade", "CVaR 5%", "Cap eff",
            "% Prof Mo", "Worst Mo",
        ]
        cell_vals = [
            skeys,
            [summary[k]["n_trades"] for k in skeys],
            [f"${summary[k]['total_pnl']:,.0f}" for k in skeys],
            [f"${summary[k]['avg_monthly_pnl']:,.0f}" for k in skeys],
            [f"{summary[k]['win_rate']:.1f}" for k in skeys],
            [f"{summary[k]['profit_factor']:.2f}" for k in skeys],
            [f"{summary[k]['trade_sharpe']:.3f}" for k in skeys],
            [f"{summary[k]['ann_sharpe']:.3f}" for k in skeys],
            [f"{summary[k]['sortino']:.3f}" for k in skeys],
            [f"{summary[k]['calmar']:.2f}" for k in skeys],
            [f"${summary[k]['max_drawdown']:,.0f}" for k in skeys],
            [f"${summary[k]['worst_trade']:,.0f}" for k in skeys],
            [f"${summary[k]['cvar_5']:,.0f}" for k in skeys],
            [f"{summary[k].get('capital_efficiency', summary[k].get('recovery_factor', 0)):.2f}" for k in skeys],
            [f"{summary[k]['pct_profitable_months']:.1f}" for k in skeys],
            [f"${summary[k]['worst_month']:,.0f}" for k in skeys],
        ]
        fig_tbl = go.Figure(data=[go.Table(
            header=dict(values=header_vals, fill_color="paleturquoise", align="left",
                        font=dict(size=11)),
            cells=dict(values=cell_vals, align="left", font=dict(size=10)),
        )])
        fig_tbl.update_layout(title="Enhanced Summary Table (sorted by Ann. Sharpe)", height=450)
        figs.append(fig_tbl)

    title_label = report_title if report_title else "SPX"
    with open(output_path, "w") as f:
        f.write("<!DOCTYPE html><html><head><meta charset='utf-8'>"
                f"<title>Enhanced Cash Flow Report — {title_label}</title></head><body>\n")
        f.write(f"<h1>Enhanced Cash Flow — {title_label}</h1>\n")
        for i, fig in enumerate(figs):
            f.write(fig.to_html(full_html=False, include_plotlyjs=("cdn" if i == 0 else False)))
        f.write("</body></html>\n")


# ---------------------------------------------------------------------------
# Strategy definitions
# ---------------------------------------------------------------------------
def build_strategy_configs() -> list[StrategyConfig]:
    """Define the strategy matrix: baselines + enhanced variants."""
    configs = [
        # --- Baselines (no filters, static IV in original; here dynamic IV but no entry filters) ---
        StrategyConfig(
            config_id="01_BASELINE_IC14",
            strategy="iron_condor", dte=14, short_delta=0.15, width=50.0,
            exit_rules=ExitRules(profit_target_pct=0.50, stop_mult=2.0),
            filters=EntryFilters(),
        ),
        StrategyConfig(
            config_id="02_BASELINE_IC21",
            strategy="iron_condor", dte=21, short_delta=0.20, width=50.0,
            exit_rules=ExitRules(profit_target_pct=0.60, stop_mult=1.5),
            filters=EntryFilters(),
        ),
        # Close N days before expiration (take profit early or avoid gamma) — variants 1–7 DTE
        StrategyConfig(
            config_id="02_IC21_CLOSE_1DTE",
            strategy="iron_condor", dte=21, short_delta=0.20, width=50.0,
            exit_rules=ExitRules(profit_target_pct=0.60, stop_mult=2.0, time_stop_dte=1),
            filters=EntryFilters(),
        ),
        # 02_IC21_CLOSE_1DTE risk-off for news (no new entry on PPI/CPI/NFP/FOMC days)
        StrategyConfig(
            config_id="02_IC21_CLOSE_1DTE_NO_NEWS",
            strategy="iron_condor", dte=21, short_delta=0.20, width=50.0,
            exit_rules=ExitRules(profit_target_pct=0.60, stop_mult=2.0, time_stop_dte=1),
            filters=EntryFilters(skip_news_days=True),
        ),
        # 02_IC21_CLOSE_1DTE with tighter 1.5x credit stop loss
        StrategyConfig(
            config_id="02_IC21_CLOSE_1DTE_SL15",
            strategy="iron_condor", dte=21, short_delta=0.20, width=50.0,
            exit_rules=ExitRules(profit_target_pct=0.60, stop_mult=1.5, time_stop_dte=1),
            filters=EntryFilters(),
        ),
        StrategyConfig(
            config_id="02_IC21_CLOSE_2DTE",
            strategy="iron_condor", dte=21, short_delta=0.20, width=50.0,
            exit_rules=ExitRules(profit_target_pct=0.60, stop_mult=1.5, time_stop_dte=2),
            filters=EntryFilters(),
        ),
        StrategyConfig(
            config_id="02_IC21_CLOSE_3DTE",
            strategy="iron_condor", dte=21, short_delta=0.20, width=50.0,
            exit_rules=ExitRules(profit_target_pct=0.60, stop_mult=1.5, time_stop_dte=3),
            filters=EntryFilters(),
        ),
        StrategyConfig(
            config_id="02_IC21_CLOSE_4DTE",
            strategy="iron_condor", dte=21, short_delta=0.20, width=50.0,
            exit_rules=ExitRules(profit_target_pct=0.60, stop_mult=1.5, time_stop_dte=4),
            filters=EntryFilters(),
        ),
        StrategyConfig(
            config_id="02_IC21_CLOSE_5DTE",
            strategy="iron_condor", dte=21, short_delta=0.20, width=50.0,
            exit_rules=ExitRules(profit_target_pct=0.60, stop_mult=1.5, time_stop_dte=5),
            filters=EntryFilters(),
        ),
        StrategyConfig(
            config_id="02_IC21_CLOSE_6DTE",
            strategy="iron_condor", dte=21, short_delta=0.20, width=50.0,
            exit_rules=ExitRules(profit_target_pct=0.60, stop_mult=1.5, time_stop_dte=6),
            filters=EntryFilters(),
        ),
        StrategyConfig(
            config_id="02_IC21_CLOSE_7DTE",
            strategy="iron_condor", dte=21, short_delta=0.20, width=50.0,
            exit_rules=ExitRules(profit_target_pct=0.60, stop_mult=1.5, time_stop_dte=7),
            filters=EntryFilters(),
        ),

        # --- VIX regime filter only ---
        StrategyConfig(
            config_id="03_VIX_FILTER",
            strategy="iron_condor", dte=14, short_delta=0.15, width=50.0,
            exit_rules=ExitRules(profit_target_pct=0.50, stop_mult=2.0),
            filters=EntryFilters(vix_max=28),
        ),

        # --- VIX + trend filter ---
        StrategyConfig(
            config_id="04_VIX_TREND",
            strategy="iron_condor", dte=14, short_delta=0.15, width=50.0,
            exit_rules=ExitRules(profit_target_pct=0.50, stop_mult=2.0),
            filters=EntryFilters(vix_max=28, trend_sma_period=50),
        ),

        # --- VIX + trend + spike filter ---
        StrategyConfig(
            config_id="05_VIX_TREND_SPIKE",
            strategy="iron_condor", dte=14, short_delta=0.15, width=50.0,
            exit_rules=ExitRules(profit_target_pct=0.50, stop_mult=2.0),
            filters=EntryFilters(vix_max=28, vix_spike_5d_pct=25, trend_sma_period=50),
        ),

        # --- Full filters: VIX + trend + spike + max 2 concurrent + loss cooldown ---
        StrategyConfig(
            config_id="06_FULL_FILTER",
            strategy="iron_condor", dte=14, short_delta=0.15, width=50.0,
            exit_rules=ExitRules(profit_target_pct=0.50, stop_mult=2.0),
            filters=EntryFilters(
                vix_max=28, vix_spike_5d_pct=25, trend_sma_period=50,
                max_concurrent=2, consecutive_loss_cooldown=1, consecutive_loss_trigger=2,
            ),
        ),

        # --- Adaptive delta (auto-adjusts to VIX) ---
        StrategyConfig(
            config_id="07_ADAPTIVE_DELTA",
            strategy="iron_condor", dte=14, short_delta=0.15, width=50.0,
            exit_rules=ExitRules(profit_target_pct=0.50, stop_mult=2.0),
            filters=EntryFilters(vix_max=35, trend_sma_period=50),
            adaptive_delta=True,
        ),

        # --- Time stop: close at 2 DTE to avoid gamma ---
        StrategyConfig(
            config_id="08_TIME_STOP",
            strategy="iron_condor", dte=14, short_delta=0.15, width=50.0,
            exit_rules=ExitRules(profit_target_pct=0.50, stop_mult=2.0, time_stop_dte=2),
            filters=EntryFilters(vix_max=28, trend_sma_period=50),
        ),

        # --- Aggressive: wider wing, higher premium, filtered ---
        StrategyConfig(
            config_id="09_AGGRESSIVE",
            strategy="iron_condor", dte=21, short_delta=0.20, width=75.0,
            exit_rules=ExitRules(profit_target_pct=0.50, stop_mult=2.0),
            filters=EntryFilters(vix_max=28, trend_sma_period=50),
        ),

        # --- Conservative: tight, far OTM, all filters ---
        StrategyConfig(
            config_id="10_CONSERVATIVE",
            strategy="iron_condor", dte=14, short_delta=0.10, width=30.0,
            exit_rules=ExitRules(profit_target_pct=0.50, stop_mult=2.0),
            filters=EntryFilters(
                vix_max=28, vix_spike_5d_pct=25, trend_sma_period=50,
                max_concurrent=2,
            ),
        ),

        # --- Adaptive + time stop + full filters (the "kitchen sink") ---
        StrategyConfig(
            config_id="11_KITCHEN_SINK",
            strategy="iron_condor", dte=14, short_delta=0.15, width=50.0,
            exit_rules=ExitRules(profit_target_pct=0.50, stop_mult=2.0, time_stop_dte=2),
            filters=EntryFilters(
                vix_max=30, vix_spike_5d_pct=25, trend_sma_period=50,
                max_concurrent=2, consecutive_loss_cooldown=1, consecutive_loss_trigger=2,
            ),
            adaptive_delta=True,
        ),

        # --- Longer DTE with adaptive + filters ---
        StrategyConfig(
            config_id="12_ADAPTIVE_21DTE",
            strategy="iron_condor", dte=21, short_delta=0.15, width=50.0,
            exit_rules=ExitRules(profit_target_pct=0.50, stop_mult=2.0, time_stop_dte=3),
            filters=EntryFilters(vix_max=28, trend_sma_period=50),
            adaptive_delta=True,
        ),

        # --- Tighter profit target for higher win rate ---
        StrategyConfig(
            config_id="13_TIGHT_TP",
            strategy="iron_condor", dte=14, short_delta=0.15, width=50.0,
            exit_rules=ExitRules(profit_target_pct=0.40, stop_mult=2.0, time_stop_dte=2),
            filters=EntryFilters(vix_max=28, trend_sma_period=50),
        ),

        # --- Put vertical only (for comparison) ---
        StrategyConfig(
            config_id="14_PCS_FILTERED",
            strategy="put_vertical", dte=14, short_delta=0.15, width=50.0,
            exit_rules=ExitRules(profit_target_pct=0.50, stop_mult=2.0),
            filters=EntryFilters(vix_max=28, trend_sma_period=50),
        ),

        # --- VIX sweet spot: only enter when VIX is 16-24 (premium-rich but not crisis) ---
        StrategyConfig(
            config_id="15_VIX_SWEET_SPOT",
            strategy="iron_condor", dte=14, short_delta=0.15, width=50.0,
            exit_rules=ExitRules(profit_target_pct=0.50, stop_mult=2.0, time_stop_dte=2),
            filters=EntryFilters(vix_min=16, vix_max=24, trend_sma_period=50),
        ),

        # --- Max positions=1 for simplicity ---
        StrategyConfig(
            config_id="16_SINGLE_POS",
            strategy="iron_condor", dte=14, short_delta=0.15, width=50.0,
            exit_rules=ExitRules(profit_target_pct=0.50, stop_mult=2.0, time_stop_dte=2),
            filters=EntryFilters(
                vix_max=28, trend_sma_period=50, max_concurrent=1,
            ),
        ),

        # ===================================================================
        # GEX PROXY CIRCUIT BREAKER STRATEGIES
        # Only enter when GEX regime proxy is positive (score >= 2 of 3)
        # ===================================================================

        # --- GEX gate on baseline (no other filters) ---
        StrategyConfig(
            config_id="G1_GEX_BASELINE",
            strategy="iron_condor", dte=14, short_delta=0.15, width=50.0,
            exit_rules=ExitRules(profit_target_pct=0.50, stop_mult=2.0),
            filters=EntryFilters(gex_min_score=2),
        ),

        # --- GEX gate + VIX + trend ---
        StrategyConfig(
            config_id="G2_GEX_VIX_TREND",
            strategy="iron_condor", dte=14, short_delta=0.15, width=50.0,
            exit_rules=ExitRules(profit_target_pct=0.50, stop_mult=2.0),
            filters=EntryFilters(vix_max=28, trend_sma_period=50, gex_min_score=2),
        ),

        # --- GEX gate + time stop ---
        StrategyConfig(
            config_id="G3_GEX_TIME_STOP",
            strategy="iron_condor", dte=14, short_delta=0.15, width=50.0,
            exit_rules=ExitRules(profit_target_pct=0.50, stop_mult=2.0, time_stop_dte=2),
            filters=EntryFilters(vix_max=28, trend_sma_period=50, gex_min_score=2),
        ),

        # --- GEX + adaptive delta + time stop (best-of-everything) ---
        StrategyConfig(
            config_id="G4_GEX_ADAPTIVE",
            strategy="iron_condor", dte=14, short_delta=0.15, width=50.0,
            exit_rules=ExitRules(profit_target_pct=0.50, stop_mult=2.0, time_stop_dte=2),
            filters=EntryFilters(vix_max=30, trend_sma_period=50, gex_min_score=2),
            adaptive_delta=True,
        ),

        # --- GEX strict (score >= 3, all signals must agree) ---
        StrategyConfig(
            config_id="G5_GEX_STRICT",
            strategy="iron_condor", dte=14, short_delta=0.15, width=50.0,
            exit_rules=ExitRules(profit_target_pct=0.50, stop_mult=2.0, time_stop_dte=2),
            filters=EntryFilters(trend_sma_period=50, gex_min_score=3),
        ),

        # --- GEX + VIX sweet spot (double selectivity) ---
        StrategyConfig(
            config_id="G6_GEX_SWEET_SPOT",
            strategy="iron_condor", dte=14, short_delta=0.15, width=50.0,
            exit_rules=ExitRules(profit_target_pct=0.50, stop_mult=2.0, time_stop_dte=2),
            filters=EntryFilters(vix_min=16, vix_max=24, trend_sma_period=50, gex_min_score=2),
        ),

        # --- GEX + 21 DTE for more premium ---
        StrategyConfig(
            config_id="G7_GEX_21DTE",
            strategy="iron_condor", dte=21, short_delta=0.15, width=50.0,
            exit_rules=ExitRules(profit_target_pct=0.50, stop_mult=2.0, time_stop_dte=3),
            filters=EntryFilters(vix_max=28, trend_sma_period=50, gex_min_score=2),
            adaptive_delta=True,
        ),

        # --- GEX + wider wings for bigger credit ---
        StrategyConfig(
            config_id="G8_GEX_WIDE",
            strategy="iron_condor", dte=14, short_delta=0.15, width=75.0,
            exit_rules=ExitRules(profit_target_pct=0.50, stop_mult=2.0, time_stop_dte=2),
            filters=EntryFilters(vix_max=28, trend_sma_period=50, gex_min_score=2),
        ),

        # ===================================================================
        # INTRA-TRADE CIRCUIT BREAKER STRATEGIES
        # VIX spike exit + SPX acceleration exit kill losses before stop hit
        # ===================================================================

        # --- VIX spike exit only (close if VIX jumps 30%+ from entry) ---
        StrategyConfig(
            config_id="CB1_VIX_SPIKE_EXIT",
            strategy="iron_condor", dte=14, short_delta=0.15, width=50.0,
            exit_rules=ExitRules(profit_target_pct=0.50, stop_mult=2.0,
                                 vix_spike_exit_pct=30),
            filters=EntryFilters(vix_max=28, trend_sma_period=50),
        ),

        # --- SPX drop exit only (close if SPX drops 2%+ in 3 days) ---
        StrategyConfig(
            config_id="CB2_SPX_DROP_EXIT",
            strategy="iron_condor", dte=14, short_delta=0.15, width=50.0,
            exit_rules=ExitRules(profit_target_pct=0.50, stop_mult=2.0,
                                 spx_drop_exit_pct=2.0, spx_drop_lookback=3),
            filters=EntryFilters(vix_max=28, trend_sma_period=50),
        ),

        # --- Both circuit breakers + time stop ---
        StrategyConfig(
            config_id="CB3_DUAL_BREAKER",
            strategy="iron_condor", dte=14, short_delta=0.15, width=50.0,
            exit_rules=ExitRules(profit_target_pct=0.50, stop_mult=2.0,
                                 time_stop_dte=2,
                                 vix_spike_exit_pct=30, spx_drop_exit_pct=2.0),
            filters=EntryFilters(vix_max=28, trend_sma_period=50),
        ),

        # --- Tighter circuit breakers (VIX +25%, SPX -1.5% in 3d) ---
        StrategyConfig(
            config_id="CB4_TIGHT_BREAKER",
            strategy="iron_condor", dte=14, short_delta=0.15, width=50.0,
            exit_rules=ExitRules(profit_target_pct=0.50, stop_mult=2.0,
                                 time_stop_dte=2,
                                 vix_spike_exit_pct=25, spx_drop_exit_pct=1.5),
            filters=EntryFilters(vix_max=28, trend_sma_period=50),
        ),

        # --- Full package: adaptive + GEX + circuit breakers ---
        StrategyConfig(
            config_id="CB5_FULL_PACKAGE",
            strategy="iron_condor", dte=14, short_delta=0.15, width=50.0,
            exit_rules=ExitRules(profit_target_pct=0.50, stop_mult=2.0,
                                 time_stop_dte=2,
                                 vix_spike_exit_pct=30, spx_drop_exit_pct=2.0),
            filters=EntryFilters(vix_max=28, trend_sma_period=50, gex_min_score=2),
            adaptive_delta=True,
        ),

        # --- Circuit breakers on VIX sweet spot (can we make best even better?) ---
        StrategyConfig(
            config_id="CB6_SWEET_BREAKER",
            strategy="iron_condor", dte=14, short_delta=0.15, width=50.0,
            exit_rules=ExitRules(profit_target_pct=0.50, stop_mult=2.0,
                                 time_stop_dte=2,
                                 vix_spike_exit_pct=25, spx_drop_exit_pct=1.5),
            filters=EntryFilters(vix_min=16, vix_max=24, trend_sma_period=50),
        ),

        # --- Ultra-conservative: tight breakers + loss cooldown ---
        StrategyConfig(
            config_id="CB7_ULTRA_SAFE",
            strategy="iron_condor", dte=14, short_delta=0.15, width=50.0,
            exit_rules=ExitRules(profit_target_pct=0.50, stop_mult=1.5,
                                 time_stop_dte=2,
                                 vix_spike_exit_pct=20, spx_drop_exit_pct=1.5),
            filters=EntryFilters(vix_max=28, trend_sma_period=50,
                                 consecutive_loss_cooldown=2, consecutive_loss_trigger=2),
        ),

        # --- Wider wings + circuit breakers (bigger credit, protected) ---
        StrategyConfig(
            config_id="CB8_WIDE_PROTECTED",
            strategy="iron_condor", dte=21, short_delta=0.15, width=75.0,
            exit_rules=ExitRules(profit_target_pct=0.50, stop_mult=2.0,
                                 time_stop_dte=3,
                                 vix_spike_exit_pct=30, spx_drop_exit_pct=2.0),
            filters=EntryFilters(vix_max=28, trend_sma_period=50),
            adaptive_delta=True,
        ),

        # ===================================================================
        # VOL EXPANSION DRAWDOWN DETECTOR STRATEGIES
        # Skip entry when 5d realized vol is expanding vs 10d (early warning)
        # Exit intra-trade when vol expansion spikes (catches surprise moves)
        # ===================================================================

        # --- Entry filter only: skip when rv_5d/rv_10d > 1.02 (catches 85% of big losses) ---
        StrategyConfig(
            config_id="VD1_ENTRY_FILTER",
            strategy="iron_condor", dte=14, short_delta=0.15, width=50.0,
            exit_rules=ExitRules(profit_target_pct=0.50, stop_mult=2.0),
            filters=EntryFilters(vix_max=28, trend_sma_period=50, rv_expansion_max=1.02),
        ),

        # --- Entry filter at 1.05 threshold (slightly more permissive, still catches 85%) ---
        StrategyConfig(
            config_id="VD2_ENTRY_105",
            strategy="iron_condor", dte=14, short_delta=0.15, width=50.0,
            exit_rules=ExitRules(profit_target_pct=0.50, stop_mult=2.0),
            filters=EntryFilters(vix_max=28, trend_sma_period=50, rv_expansion_max=1.05),
        ),

        # --- Entry filter + time stop (Sharpe-optimized with gamma protection) ---
        StrategyConfig(
            config_id="VD3_ENTRY_TIME",
            strategy="iron_condor", dte=14, short_delta=0.15, width=50.0,
            exit_rules=ExitRules(profit_target_pct=0.50, stop_mult=2.0, time_stop_dte=2),
            filters=EntryFilters(vix_max=28, trend_sma_period=50, rv_expansion_max=1.02),
        ),

        # --- Intra-trade exit only: exit if rv_ratio > 1.3 during trade ---
        StrategyConfig(
            config_id="VD4_INTRATRADE",
            strategy="iron_condor", dte=14, short_delta=0.15, width=50.0,
            exit_rules=ExitRules(profit_target_pct=0.50, stop_mult=2.0,
                                 rv_expansion_exit=1.3),
            filters=EntryFilters(vix_max=28, trend_sma_period=50),
        ),

        # --- Combined: entry filter 1.02 + intra-trade exit at 1.25 ---
        StrategyConfig(
            config_id="VD5_DUAL_DETECT",
            strategy="iron_condor", dte=14, short_delta=0.15, width=50.0,
            exit_rules=ExitRules(profit_target_pct=0.50, stop_mult=2.0,
                                 rv_expansion_exit=1.25),
            filters=EntryFilters(vix_max=28, trend_sma_period=50, rv_expansion_max=1.02),
        ),

        # --- Combined + time stop (maximum protection) ---
        StrategyConfig(
            config_id="VD6_DUAL_TIME",
            strategy="iron_condor", dte=14, short_delta=0.15, width=50.0,
            exit_rules=ExitRules(profit_target_pct=0.50, stop_mult=2.0,
                                 time_stop_dte=2, rv_expansion_exit=1.25),
            filters=EntryFilters(vix_max=28, trend_sma_period=50, rv_expansion_max=1.02),
        ),

        # --- Full stack: vol detector + VIX circuit breakers + time stop ---
        StrategyConfig(
            config_id="VD7_FULL_STACK",
            strategy="iron_condor", dte=14, short_delta=0.15, width=50.0,
            exit_rules=ExitRules(profit_target_pct=0.50, stop_mult=2.0,
                                 time_stop_dte=2,
                                 vix_spike_exit_pct=30, spx_drop_exit_pct=2.0,
                                 rv_expansion_exit=1.25),
            filters=EntryFilters(vix_max=28, trend_sma_period=50, rv_expansion_max=1.02),
        ),

        # --- Vol detector + adaptive delta (best risk-adjusted attempt) ---
        StrategyConfig(
            config_id="VD8_ADAPTIVE",
            strategy="iron_condor", dte=14, short_delta=0.15, width=50.0,
            exit_rules=ExitRules(profit_target_pct=0.50, stop_mult=2.0,
                                 time_stop_dte=2, rv_expansion_exit=1.25),
            filters=EntryFilters(vix_max=30, trend_sma_period=50, rv_expansion_max=1.05),
            adaptive_delta=True,
        ),

        # --- Vol detector on 21 DTE (does it work on longer duration?) ---
        StrategyConfig(
            config_id="VD9_21DTE",
            strategy="iron_condor", dte=21, short_delta=0.15, width=50.0,
            exit_rules=ExitRules(profit_target_pct=0.50, stop_mult=2.0,
                                 time_stop_dte=3, rv_expansion_exit=1.25),
            filters=EntryFilters(vix_max=28, trend_sma_period=50, rv_expansion_max=1.05),
            adaptive_delta=True,
        ),

        # --- Ultra-conservative: tight entry + tight intra-trade + tight stop ---
        StrategyConfig(
            config_id="VD10_ULTRA_SAFE",
            strategy="iron_condor", dte=14, short_delta=0.15, width=50.0,
            exit_rules=ExitRules(profit_target_pct=0.50, stop_mult=1.5,
                                 time_stop_dte=2, rv_expansion_exit=1.20),
            filters=EntryFilters(vix_max=28, trend_sma_period=50,
                                 rv_expansion_max=1.00,
                                 consecutive_loss_cooldown=1, consecutive_loss_trigger=2),
        ),

        # ===================================================================
        # 7-DTE SHORT-DURATION STRATEGIES
        # Enter Wednesday, expire Friday. Max 1 weekend exposure.
        # Eliminates all multi-weekend drawdowns structurally.
        # ===================================================================

        # --- 7-DTE baseline with VIX + trend filters ---
        StrategyConfig(
            config_id="SD1_7DTE_BASE",
            strategy="iron_condor", dte=7, short_delta=0.15, width=50.0,
            exit_rules=ExitRules(profit_target_pct=0.50, stop_mult=2.0),
            filters=EntryFilters(vix_max=28, trend_sma_period=50),
        ),

        # --- 7-DTE + vol expansion entry filter ---
        StrategyConfig(
            config_id="SD2_7DTE_VOLDET",
            strategy="iron_condor", dte=7, short_delta=0.15, width=50.0,
            exit_rules=ExitRules(profit_target_pct=0.50, stop_mult=2.0),
            filters=EntryFilters(vix_max=28, trend_sma_period=50, rv_expansion_max=1.02),
        ),

        # --- 7-DTE + vol detector + time stop at 1 DTE ---
        StrategyConfig(
            config_id="SD3_7DTE_VOLDET_TS",
            strategy="iron_condor", dte=7, short_delta=0.15, width=50.0,
            exit_rules=ExitRules(profit_target_pct=0.50, stop_mult=2.0, time_stop_dte=1),
            filters=EntryFilters(vix_max=28, trend_sma_period=50, rv_expansion_max=1.02),
        ),

        # --- 7-DTE + vol detector + intra-trade rv expansion exit ---
        StrategyConfig(
            config_id="SD4_7DTE_FULL",
            strategy="iron_condor", dte=7, short_delta=0.15, width=50.0,
            exit_rules=ExitRules(profit_target_pct=0.50, stop_mult=2.0,
                                 rv_expansion_exit=1.25),
            filters=EntryFilters(vix_max=28, trend_sma_period=50, rv_expansion_max=1.02),
        ),

        # ===================================================================
        # FRIDAY CLOSE STRATEGIES (unconditional weekend avoidance)
        # Close all positions at Friday close to eliminate weekend gap risk.
        # ===================================================================

        # --- 14-DTE + unconditional Friday close + VIX/trend ---
        StrategyConfig(
            config_id="WK1_FRI_CLOSE",
            strategy="iron_condor", dte=14, short_delta=0.15, width=50.0,
            exit_rules=ExitRules(profit_target_pct=0.50, stop_mult=2.0,
                                 close_friday=True),
            filters=EntryFilters(vix_max=28, trend_sma_period=50),
        ),

        # --- 14-DTE + Friday close + vol detector ---
        StrategyConfig(
            config_id="WK2_FRI_CLOSE_VD",
            strategy="iron_condor", dte=14, short_delta=0.15, width=50.0,
            exit_rules=ExitRules(profit_target_pct=0.50, stop_mult=2.0,
                                 close_friday=True),
            filters=EntryFilters(vix_max=28, trend_sma_period=50, rv_expansion_max=1.02),
        ),

        # --- 14-DTE + Friday close + vol detector + time stop ---
        StrategyConfig(
            config_id="WK3_FRI_CLOSE_VD_TS",
            strategy="iron_condor", dte=14, short_delta=0.15, width=50.0,
            exit_rules=ExitRules(profit_target_pct=0.50, stop_mult=2.0,
                                 close_friday=True, time_stop_dte=2),
            filters=EntryFilters(vix_max=28, trend_sma_period=50, rv_expansion_max=1.02),
        ),

        # ===================================================================
        # CONDITIONAL WEEKEND CLOSE STRATEGIES
        # Close before weekend ONLY if position is underwater.
        # Winners keep running (theta decay over weekend is free money).
        # ===================================================================

        # --- 14-DTE + close losers before weekend ---
        StrategyConfig(
            config_id="WK4_COND_WEEKEND",
            strategy="iron_condor", dte=14, short_delta=0.15, width=50.0,
            exit_rules=ExitRules(profit_target_pct=0.50, stop_mult=2.0,
                                 weekend_close_if_loss=True),
            filters=EntryFilters(vix_max=28, trend_sma_period=50),
        ),

        # --- 14-DTE + conditional weekend close + vol detector ---
        StrategyConfig(
            config_id="WK5_COND_WKND_VD",
            strategy="iron_condor", dte=14, short_delta=0.15, width=50.0,
            exit_rules=ExitRules(profit_target_pct=0.50, stop_mult=2.0,
                                 weekend_close_if_loss=True),
            filters=EntryFilters(vix_max=28, trend_sma_period=50, rv_expansion_max=1.02),
        ),

        # --- 14-DTE + conditional weekend + vol detector + intra-trade rv exit ---
        StrategyConfig(
            config_id="WK6_BEST_BLEND",
            strategy="iron_condor", dte=14, short_delta=0.15, width=50.0,
            exit_rules=ExitRules(profit_target_pct=0.50, stop_mult=2.0,
                                 weekend_close_if_loss=True,
                                 rv_expansion_exit=1.25),
            filters=EntryFilters(vix_max=28, trend_sma_period=50, rv_expansion_max=1.02),
        ),

        # ===================================================================
        # AGGRESSIVE STRATEGY VARIANTS (based on 09_AGGRESSIVE)
        # Base: IC, 21 DTE, 0.20 delta, 75-wide, VIX<28, trend SMA50
        # Goal: keep the $248k profit engine, tame the $17.7k drawdown
        # ===================================================================

        # --- AG1: Tighter stop (1.5x instead of 2x) ---
        # Cuts avg loss from ~$2,547 to ~$1,014 per trade.
        # Even if win rate drops slightly, Sharpe should jump.
        StrategyConfig(
            config_id="AG1_TIGHT_STOP",
            strategy="iron_condor", dte=21, short_delta=0.20, width=75.0,
            exit_rules=ExitRules(profit_target_pct=0.50, stop_mult=1.5),
            filters=EntryFilters(vix_max=28, trend_sma_period=50),
        ),

        # --- AG2: Vol expansion entry filter ---
        # Skip entries when rv_5d/rv_10d > 1.05. Catches 50% of big losses at entry.
        StrategyConfig(
            config_id="AG2_VOLDET",
            strategy="iron_condor", dte=21, short_delta=0.20, width=75.0,
            exit_rules=ExitRules(profit_target_pct=0.50, stop_mult=2.0),
            filters=EntryFilters(vix_max=28, trend_sma_period=50, rv_expansion_max=1.05),
        ),

        # --- AG3: Tighter stop + time stop at 3 DTE ---
        # Prevents the -$5,279 expiry loss and all late-trade gamma.
        StrategyConfig(
            config_id="AG3_STOP_TIME",
            strategy="iron_condor", dte=21, short_delta=0.20, width=75.0,
            exit_rules=ExitRules(profit_target_pct=0.50, stop_mult=1.5, time_stop_dte=3),
            filters=EntryFilters(vix_max=28, trend_sma_period=50),
        ),

        # --- AG4: Full protection (tight stop + vol detector + time stop + weekend close) ---
        # Maximum drawdown reduction. Every defense layer active.
        StrategyConfig(
            config_id="AG4_FULL_PROTECT",
            strategy="iron_condor", dte=21, short_delta=0.20, width=75.0,
            exit_rules=ExitRules(profit_target_pct=0.50, stop_mult=1.5,
                                 time_stop_dte=3, weekend_close_if_loss=True),
            filters=EntryFilters(vix_max=28, trend_sma_period=50, rv_expansion_max=1.05),
        ),

        # --- AG5: Keep 2x stop but add all circuit breakers ---
        # VIX spike + SPX drop + rv expansion exits fire before stop hits.
        StrategyConfig(
            config_id="AG5_CIRCUIT_BREAK",
            strategy="iron_condor", dte=21, short_delta=0.20, width=75.0,
            exit_rules=ExitRules(profit_target_pct=0.50, stop_mult=2.0,
                                 vix_spike_exit_pct=30, spx_drop_exit_pct=2.0,
                                 rv_expansion_exit=1.25),
            filters=EntryFilters(vix_max=28, trend_sma_period=50),
        ),

        # ===================================================================
        # GAP-DOWN + ES HEDGE STRATEGIES
        # No entry on pre-market gap down; go short ES when gap down in trade;
        # hold hedge until next day open is not a gap down (recovery).
        # ===================================================================

        # --- H1: No new trade on gap down (aggressive base) ---
        StrategyConfig(
            config_id="H1_GAP_NO_ENTRY",
            strategy="iron_condor", dte=21, short_delta=0.20, width=75.0,
            exit_rules=ExitRules(profit_target_pct=0.50, stop_mult=2.0),
            filters=EntryFilters(vix_max=28, trend_sma_period=50,
                                 gap_down_no_entry=-0.5),
        ),

        # --- H2: ES hedge when in trade and gap < -1%; lift when next day gap >= -0.3% ---
        StrategyConfig(
            config_id="H2_ES_HEDGE",
            strategy="iron_condor", dte=21, short_delta=0.20, width=75.0,
            exit_rules=ExitRules(profit_target_pct=0.50, stop_mult=2.0,
                                 hedge_on_gap_down=-1.0, hedge_off_gap_threshold=-0.3),
            filters=EntryFilters(vix_max=28, trend_sma_period=50),
        ),
        # H2 + close N days before expiration — variants 1–7 DTE
        StrategyConfig(
            config_id="H2_ES_HEDGE_1DTE",
            strategy="iron_condor", dte=21, short_delta=0.20, width=75.0,
            exit_rules=ExitRules(profit_target_pct=0.50, stop_mult=2.0,
                                 hedge_on_gap_down=-1.0, hedge_off_gap_threshold=-0.3,
                                 time_stop_dte=1),
            filters=EntryFilters(vix_max=28, trend_sma_period=50),
        ),
        StrategyConfig(
            config_id="H2_ES_HEDGE_2DTE",
            strategy="iron_condor", dte=21, short_delta=0.20, width=75.0,
            exit_rules=ExitRules(profit_target_pct=0.50, stop_mult=2.0,
                                 hedge_on_gap_down=-1.0, hedge_off_gap_threshold=-0.3,
                                 time_stop_dte=2),
            filters=EntryFilters(vix_max=28, trend_sma_period=50),
        ),
        # H2_ES_HEDGE_2DTE risk-off for news
        StrategyConfig(
            config_id="H2_ES_HEDGE_2DTE_NO_NEWS",
            strategy="iron_condor", dte=21, short_delta=0.20, width=75.0,
            exit_rules=ExitRules(profit_target_pct=0.50, stop_mult=2.0,
                                 hedge_on_gap_down=-1.0, hedge_off_gap_threshold=-0.3,
                                 time_stop_dte=2),
            filters=EntryFilters(vix_max=28, trend_sma_period=50, skip_news_days=True),
        ),
        # H2_ES_HEDGE_2DTE with tighter 1.5x credit stop loss
        StrategyConfig(
            config_id="H2_ES_HEDGE_2DTE_SL15",
            strategy="iron_condor", dte=21, short_delta=0.20, width=75.0,
            exit_rules=ExitRules(profit_target_pct=0.50, stop_mult=1.5,
                                 hedge_on_gap_down=-1.0, hedge_off_gap_threshold=-0.3,
                                 time_stop_dte=2),
            filters=EntryFilters(vix_max=28, trend_sma_period=50),
        ),
        StrategyConfig(
            config_id="H2_ES_HEDGE_3DTE",
            strategy="iron_condor", dte=21, short_delta=0.20, width=75.0,
            exit_rules=ExitRules(profit_target_pct=0.50, stop_mult=2.0,
                                 hedge_on_gap_down=-1.0, hedge_off_gap_threshold=-0.3,
                                 time_stop_dte=3),
            filters=EntryFilters(vix_max=28, trend_sma_period=50),
        ),
        StrategyConfig(
            config_id="H2_ES_HEDGE_4DTE",
            strategy="iron_condor", dte=21, short_delta=0.20, width=75.0,
            exit_rules=ExitRules(profit_target_pct=0.50, stop_mult=2.0,
                                 hedge_on_gap_down=-1.0, hedge_off_gap_threshold=-0.3,
                                 time_stop_dte=4),
            filters=EntryFilters(vix_max=28, trend_sma_period=50),
        ),
        StrategyConfig(
            config_id="H2_ES_HEDGE_5DTE",
            strategy="iron_condor", dte=21, short_delta=0.20, width=75.0,
            exit_rules=ExitRules(profit_target_pct=0.50, stop_mult=2.0,
                                 hedge_on_gap_down=-1.0, hedge_off_gap_threshold=-0.3,
                                 time_stop_dte=5),
            filters=EntryFilters(vix_max=28, trend_sma_period=50),
        ),
        StrategyConfig(
            config_id="H2_ES_HEDGE_6DTE",
            strategy="iron_condor", dte=21, short_delta=0.20, width=75.0,
            exit_rules=ExitRules(profit_target_pct=0.50, stop_mult=2.0,
                                 hedge_on_gap_down=-1.0, hedge_off_gap_threshold=-0.3,
                                 time_stop_dte=6),
            filters=EntryFilters(vix_max=28, trend_sma_period=50),
        ),
        StrategyConfig(
            config_id="H2_ES_HEDGE_7DTE",
            strategy="iron_condor", dte=21, short_delta=0.20, width=75.0,
            exit_rules=ExitRules(profit_target_pct=0.50, stop_mult=2.0,
                                 hedge_on_gap_down=-1.0, hedge_off_gap_threshold=-0.3,
                                 time_stop_dte=7),
            filters=EntryFilters(vix_max=28, trend_sma_period=50),
        ),

        # --- H3: No entry on gap down + ES hedge when in trade ---
        StrategyConfig(
            config_id="H3_GAP_AND_HEDGE",
            strategy="iron_condor", dte=21, short_delta=0.20, width=75.0,
            exit_rules=ExitRules(profit_target_pct=0.50, stop_mult=2.0,
                                 hedge_on_gap_down=-1.0, hedge_off_gap_threshold=-0.3),
            filters=EntryFilters(vix_max=28, trend_sma_period=50,
                                 gap_down_no_entry=-0.5),
        ),

        # --- H4: Vol detector + gap no entry + ES hedge (best of all) ---
        StrategyConfig(
            config_id="H4_VD_GAP_HEDGE",
            strategy="iron_condor", dte=21, short_delta=0.20, width=75.0,
            exit_rules=ExitRules(profit_target_pct=0.50, stop_mult=2.0,
                                 hedge_on_gap_down=-1.0, hedge_off_gap_threshold=-0.3),
            filters=EntryFilters(vix_max=28, trend_sma_period=50,
                                 rv_expansion_max=1.05, gap_down_no_entry=-0.5),
        ),
        # H4 + close N days before expiration — variants 1–7 DTE
        StrategyConfig(
            config_id="H4_VD_GAP_HEDGE_1DTE",
            strategy="iron_condor", dte=21, short_delta=0.20, width=75.0,
            exit_rules=ExitRules(profit_target_pct=0.50, stop_mult=2.0,
                                 hedge_on_gap_down=-1.0, hedge_off_gap_threshold=-0.3,
                                 time_stop_dte=1),
            filters=EntryFilters(vix_max=28, trend_sma_period=50,
                                 rv_expansion_max=1.05, gap_down_no_entry=-0.5),
        ),
        StrategyConfig(
            config_id="H4_VD_GAP_HEDGE_2DTE",
            strategy="iron_condor", dte=21, short_delta=0.20, width=75.0,
            exit_rules=ExitRules(profit_target_pct=0.50, stop_mult=2.0,
                                 hedge_on_gap_down=-1.0, hedge_off_gap_threshold=-0.3,
                                 time_stop_dte=2),
            filters=EntryFilters(vix_max=28, trend_sma_period=50,
                                 rv_expansion_max=1.05, gap_down_no_entry=-0.5),
        ),
        StrategyConfig(
            config_id="H4_VD_GAP_HEDGE_3DTE",
            strategy="iron_condor", dte=21, short_delta=0.20, width=75.0,
            exit_rules=ExitRules(profit_target_pct=0.50, stop_mult=2.0,
                                 hedge_on_gap_down=-1.0, hedge_off_gap_threshold=-0.3,
                                 time_stop_dte=3),
            filters=EntryFilters(vix_max=28, trend_sma_period=50,
                                 rv_expansion_max=1.05, gap_down_no_entry=-0.5),
        ),
        StrategyConfig(
            config_id="H4_VD_GAP_HEDGE_4DTE",
            strategy="iron_condor", dte=21, short_delta=0.20, width=75.0,
            exit_rules=ExitRules(profit_target_pct=0.50, stop_mult=2.0,
                                 hedge_on_gap_down=-1.0, hedge_off_gap_threshold=-0.3,
                                 time_stop_dte=4),
            filters=EntryFilters(vix_max=28, trend_sma_period=50,
                                 rv_expansion_max=1.05, gap_down_no_entry=-0.5),
        ),
        StrategyConfig(
            config_id="H4_VD_GAP_HEDGE_5DTE",
            strategy="iron_condor", dte=21, short_delta=0.20, width=75.0,
            exit_rules=ExitRules(profit_target_pct=0.50, stop_mult=2.0,
                                 hedge_on_gap_down=-1.0, hedge_off_gap_threshold=-0.3,
                                 time_stop_dte=5),
            filters=EntryFilters(vix_max=28, trend_sma_period=50,
                                 rv_expansion_max=1.05, gap_down_no_entry=-0.5),
        ),
        # H4_VD_GAP_HEDGE_5DTE risk-off for news
        StrategyConfig(
            config_id="H4_VD_GAP_HEDGE_5DTE_NO_NEWS",
            strategy="iron_condor", dte=21, short_delta=0.20, width=75.0,
            exit_rules=ExitRules(profit_target_pct=0.50, stop_mult=2.0,
                                 hedge_on_gap_down=-1.0, hedge_off_gap_threshold=-0.3,
                                 time_stop_dte=5),
            filters=EntryFilters(vix_max=28, trend_sma_period=50,
                                 rv_expansion_max=1.05, gap_down_no_entry=-0.5, skip_news_days=True),
        ),
        # H4_VD_GAP_HEDGE_5DTE with tighter 1.5x credit stop loss
        StrategyConfig(
            config_id="H4_VD_GAP_HEDGE_5DTE_SL15",
            strategy="iron_condor", dte=21, short_delta=0.20, width=75.0,
            exit_rules=ExitRules(profit_target_pct=0.50, stop_mult=1.5,
                                 hedge_on_gap_down=-1.0, hedge_off_gap_threshold=-0.3,
                                 time_stop_dte=5),
            filters=EntryFilters(vix_max=28, trend_sma_period=50,
                                 rv_expansion_max=1.05, gap_down_no_entry=-0.5),
        ),
        # Dynamic sizing: 1.5x or 2x when VIX 16–24, rv_ratio ≤ 1.05, GEX ≥ 2, gap ≥ -0.3%
        StrategyConfig(
            config_id="02_IC21_CLOSE_1DTE_DS15",
            strategy="iron_condor", dte=21, short_delta=0.20, width=50.0,
            exit_rules=ExitRules(profit_target_pct=0.60, stop_mult=2.0, time_stop_dte=1),
            filters=EntryFilters(),
            size_mult_in_zone=1.5,
            zone_vix_min=16.0, zone_vix_max=24.0, zone_rv_max=1.05,
            zone_gex_min=2, zone_gap_min=-0.3,
        ),
        StrategyConfig(
            config_id="H2_ES_HEDGE_2DTE_DS15",
            strategy="iron_condor", dte=21, short_delta=0.20, width=75.0,
            exit_rules=ExitRules(profit_target_pct=0.50, stop_mult=2.0,
                                 hedge_on_gap_down=-1.0, hedge_off_gap_threshold=-0.3,
                                 time_stop_dte=2),
            filters=EntryFilters(vix_max=28, trend_sma_period=50),
            size_mult_in_zone=1.5,
            zone_vix_min=16.0, zone_vix_max=24.0, zone_rv_max=1.05,
            zone_gex_min=2, zone_gap_min=-0.3,
        ),
        StrategyConfig(
            config_id="H4_VD_GAP_HEDGE_5DTE_DS15",
            strategy="iron_condor", dte=21, short_delta=0.20, width=75.0,
            exit_rules=ExitRules(profit_target_pct=0.50, stop_mult=2.0,
                                 hedge_on_gap_down=-1.0, hedge_off_gap_threshold=-0.3,
                                 time_stop_dte=5),
            filters=EntryFilters(vix_max=28, trend_sma_period=50,
                                 rv_expansion_max=1.05, gap_down_no_entry=-0.5),
            size_mult_in_zone=1.5,
            zone_vix_min=16.0, zone_vix_max=24.0, zone_rv_max=1.05,
            zone_gex_min=2, zone_gap_min=-0.3,
        ),
        StrategyConfig(
            config_id="H4_VD_GAP_HEDGE_5DTE_DS2",
            strategy="iron_condor", dte=21, short_delta=0.20, width=75.0,
            exit_rules=ExitRules(profit_target_pct=0.50, stop_mult=2.0,
                                 hedge_on_gap_down=-1.0, hedge_off_gap_threshold=-0.3,
                                 time_stop_dte=5),
            filters=EntryFilters(vix_max=28, trend_sma_period=50,
                                 rv_expansion_max=1.05, gap_down_no_entry=-0.5),
            size_mult_in_zone=2.0,
            zone_vix_min=16.0, zone_vix_max=24.0, zone_rv_max=1.05,
            zone_gex_min=2, zone_gap_min=-0.3,
        ),
        StrategyConfig(
            config_id="H4_VD_GAP_HEDGE_6DTE",
            strategy="iron_condor", dte=21, short_delta=0.20, width=75.0,
            exit_rules=ExitRules(profit_target_pct=0.50, stop_mult=2.0,
                                 hedge_on_gap_down=-1.0, hedge_off_gap_threshold=-0.3,
                                 time_stop_dte=6),
            filters=EntryFilters(vix_max=28, trend_sma_period=50,
                                 rv_expansion_max=1.05, gap_down_no_entry=-0.5),
        ),
        StrategyConfig(
            config_id="H4_VD_GAP_HEDGE_7DTE",
            strategy="iron_condor", dte=21, short_delta=0.20, width=75.0,
            exit_rules=ExitRules(profit_target_pct=0.50, stop_mult=2.0,
                                 hedge_on_gap_down=-1.0, hedge_off_gap_threshold=-0.3,
                                 time_stop_dte=7),
            filters=EntryFilters(vix_max=28, trend_sma_period=50,
                                 rv_expansion_max=1.05, gap_down_no_entry=-0.5),
        ),

        # --- H5: Stricter gap (no entry if gap < -0.3%) + ES hedge ---
        StrategyConfig(
            config_id="H5_GAP_STRICT_HEDGE",
            strategy="iron_condor", dte=21, short_delta=0.20, width=75.0,
            exit_rules=ExitRules(profit_target_pct=0.50, stop_mult=2.0,
                                 hedge_on_gap_down=-0.8, hedge_off_gap_threshold=-0.2),
            filters=EntryFilters(vix_max=28, trend_sma_period=50,
                                 gap_down_no_entry=-0.3),
        ),

        # --- H2P: PUT-only version of H2_ES_HEDGE ---
        StrategyConfig(
            config_id="H2P_ES_HEDGE_PUT",
            strategy="put_vertical", dte=21, short_delta=0.20, width=75.0,
            exit_rules=ExitRules(profit_target_pct=0.50, stop_mult=2.0,
                                 hedge_on_gap_down=-1.0, hedge_off_gap_threshold=-0.3),
            filters=EntryFilters(vix_max=28, trend_sma_period=50),
        ),

        # --- H4P: PUT-only version of H4_VD_GAP_HEDGE ---
        StrategyConfig(
            config_id="H4P_VD_GAP_HEDGE_PUT",
            strategy="put_vertical", dte=21, short_delta=0.20, width=75.0,
            exit_rules=ExitRules(profit_target_pct=0.50, stop_mult=2.0,
                                 hedge_on_gap_down=-1.0, hedge_off_gap_threshold=-0.3),
            filters=EntryFilters(vix_max=28, trend_sma_period=50,
                                 rv_expansion_max=1.05, gap_down_no_entry=-0.5),
        ),

        # --- H5P: PUT-only version of H5_GAP_STRICT_HEDGE ---
        StrategyConfig(
            config_id="H5P_GAP_STRICT_HEDGE_PUT",
            strategy="put_vertical", dte=21, short_delta=0.20, width=75.0,
            exit_rules=ExitRules(profit_target_pct=0.50, stop_mult=2.0,
                                 hedge_on_gap_down=-0.8, hedge_off_gap_threshold=-0.2),
            filters=EntryFilters(vix_max=28, trend_sma_period=50,
                                 gap_down_no_entry=-0.3),
        ),

        # ===================================================================
        # BEAR CALL VERTICALS — enter only when SPX < SMA50 (bearish regime)
        # Sell call credit spreads into downtrends for directional premium.
        # ===================================================================

        # BC1: 14 DTE bear call, 15-delta, $50 wide, VIX 16-28
        StrategyConfig(
            config_id="BC1_BEAR_CALL_14DTE",
            strategy="call_vertical", dte=14, short_delta=0.15, width=50.0,
            exit_rules=ExitRules(profit_target_pct=0.50, stop_mult=2.0, time_stop_dte=2),
            filters=EntryFilters(vix_min=16, vix_max=28, trend_sma_period=50, trend_sma_invert=True),
        ),

        # BC2: 7 DTE bear call, faster cycle
        StrategyConfig(
            config_id="BC2_BEAR_CALL_7DTE",
            strategy="call_vertical", dte=7, short_delta=0.15, width=50.0,
            exit_rules=ExitRules(profit_target_pct=0.50, stop_mult=2.0),
            filters=EntryFilters(vix_min=16, vix_max=28, trend_sma_period=50, trend_sma_invert=True),
        ),

        # BC3: 21 DTE bear call, 20-delta, $75 wide (more premium, aggressive)
        StrategyConfig(
            config_id="BC3_BEAR_CALL_21DTE",
            strategy="call_vertical", dte=21, short_delta=0.20, width=75.0,
            exit_rules=ExitRules(profit_target_pct=0.50, stop_mult=2.0),
            filters=EntryFilters(vix_max=28, trend_sma_period=50, trend_sma_invert=True,
                                 rv_expansion_max=1.05),
        ),

        # BC4: 14 DTE bear call, 10-delta (far OTM, conservative)
        StrategyConfig(
            config_id="BC4_BEAR_CALL_CONSERVATIVE",
            strategy="call_vertical", dte=14, short_delta=0.10, width=50.0,
            exit_rules=ExitRules(profit_target_pct=0.50, stop_mult=2.0, time_stop_dte=2),
            filters=EntryFilters(vix_max=28, trend_sma_period=50, trend_sma_invert=True),
        ),

        # BC5: 14 DTE bear call, VIX sweet spot only (16-24)
        StrategyConfig(
            config_id="BC5_BEAR_CALL_SWEET",
            strategy="call_vertical", dte=14, short_delta=0.15, width=50.0,
            exit_rules=ExitRules(profit_target_pct=0.50, stop_mult=2.0, time_stop_dte=2),
            filters=EntryFilters(vix_min=16, vix_max=24, trend_sma_period=50, trend_sma_invert=True),
        ),

        # BC6: 21 DTE bear call, no VIX min (enters even in low vol downtrends)
        StrategyConfig(
            config_id="BC6_BEAR_CALL_21DTE_WIDE",
            strategy="call_vertical", dte=21, short_delta=0.15, width=50.0,
            exit_rules=ExitRules(profit_target_pct=0.50, stop_mult=2.0, time_stop_dte=3),
            filters=EntryFilters(vix_max=35, trend_sma_period=50, trend_sma_invert=True),
        ),

        # ===================================================================
        # DOWNSIDE-ONLY HEDGE: nights + weekends, re-assess every morning
        # Short ES only when open is down; hold for the day; lift when next
        # morning not gap down. Optional: short ES every Fri close -> Mon open.
        # ===================================================================

        # --- H2W: H2 + weekend hedge (Fri close -> Mon open) ---
        StrategyConfig(
            config_id="H2W_ES_HEDGE_WEEKENDS",
            strategy="iron_condor", dte=21, short_delta=0.20, width=75.0,
            exit_rules=ExitRules(profit_target_pct=0.50, stop_mult=2.0,
                                 hedge_on_gap_down=-1.0, hedge_off_gap_threshold=-0.3,
                                 hedge_weekends=True),
            filters=EntryFilters(vix_max=28, trend_sma_period=50),
        ),

        # --- H4W: H4 + weekend hedge ---
        StrategyConfig(
            config_id="H4W_VD_GAP_WEEKENDS",
            strategy="iron_condor", dte=21, short_delta=0.20, width=75.0,
            exit_rules=ExitRules(profit_target_pct=0.50, stop_mult=2.0,
                                 hedge_on_gap_down=-1.0, hedge_off_gap_threshold=-0.3,
                                 hedge_weekends=True),
            filters=EntryFilters(vix_max=28, trend_sma_period=50,
                                 rv_expansion_max=1.05, gap_down_no_entry=-0.5),
        ),

        # --- H5W: H5 + weekend hedge ---
        StrategyConfig(
            config_id="H5W_GAP_STRICT_WEEKENDS",
            strategy="iron_condor", dte=21, short_delta=0.20, width=75.0,
            exit_rules=ExitRules(profit_target_pct=0.50, stop_mult=2.0,
                                 hedge_on_gap_down=-0.8, hedge_off_gap_threshold=-0.2,
                                 hedge_weekends=True),
            filters=EntryFilters(vix_max=28, trend_sma_period=50,
                                 gap_down_no_entry=-0.3),
        ),

        # -----------------------------------------------------------------------
        # IH_MOM — SPX cumulative drop from entry (momentum break flatten)
        # Close position if SPX falls more than X% from the entry spot price.
        # Tighter than the rolling spx_drop_exit_pct which measures last N days.
        # -----------------------------------------------------------------------
        StrategyConfig(
            config_id="IH_MOM1_2P5PCT",
            strategy="iron_condor", dte=21, short_delta=0.20, width=75.0,
            exit_rules=ExitRules(profit_target_pct=0.50, stop_mult=2.0,
                                 spx_drop_from_entry_pct=2.5),
            filters=EntryFilters(),
        ),
        StrategyConfig(
            config_id="IH_MOM2_3P5PCT",
            strategy="iron_condor", dte=21, short_delta=0.20, width=75.0,
            exit_rules=ExitRules(profit_target_pct=0.50, stop_mult=2.0,
                                 spx_drop_from_entry_pct=3.5),
            filters=EntryFilters(),
        ),
        StrategyConfig(
            config_id="IH_MOM3_5PCT",
            strategy="iron_condor", dte=21, short_delta=0.20, width=75.0,
            exit_rules=ExitRules(profit_target_pct=0.50, stop_mult=2.0,
                                 spx_drop_from_entry_pct=5.0),
            filters=EntryFilters(),
        ),

        # -----------------------------------------------------------------------
        # IH_VXN — VIX9D > VIX backwardation exit
        # Close position when the 9-day IV exceeds the 30-day IV (term structure
        # inverts into backwardation), signalling acute short-term fear.
        # -----------------------------------------------------------------------
        StrategyConfig(
            config_id="IH_VXN1_IC14",
            strategy="iron_condor", dte=14, short_delta=0.15, width=50.0,
            exit_rules=ExitRules(profit_target_pct=0.50, stop_mult=2.0,
                                 vix_backwardation_exit=True),
            filters=EntryFilters(),
        ),
        StrategyConfig(
            config_id="IH_VXN2_IC21",
            strategy="iron_condor", dte=21, short_delta=0.20, width=75.0,
            exit_rules=ExitRules(profit_target_pct=0.50, stop_mult=2.0,
                                 vix_backwardation_exit=True),
            filters=EntryFilters(),
        ),
        StrategyConfig(
            config_id="IH_VXN3_IC21_FILT",
            strategy="iron_condor", dte=21, short_delta=0.20, width=75.0,
            exit_rules=ExitRules(profit_target_pct=0.50, stop_mult=2.0,
                                 vix_backwardation_exit=True),
            filters=EntryFilters(vix_max=28, trend_sma_period=50),
        ),

        # -----------------------------------------------------------------------
        # IH_VSP — VIX spike from entry exit (existing field, new threshold sweep)
        # Close position when VIX has risen more than X% from entry-day VIX.
        # -----------------------------------------------------------------------
        StrategyConfig(
            config_id="IH_VSP1_25PCT",
            strategy="iron_condor", dte=21, short_delta=0.20, width=75.0,
            exit_rules=ExitRules(profit_target_pct=0.50, stop_mult=2.0,
                                 vix_spike_exit_pct=25),
            filters=EntryFilters(),
        ),
        StrategyConfig(
            config_id="IH_VSP2_40PCT",
            strategy="iron_condor", dte=21, short_delta=0.20, width=75.0,
            exit_rules=ExitRules(profit_target_pct=0.50, stop_mult=2.0,
                                 vix_spike_exit_pct=40),
            filters=EntryFilters(),
        ),
        StrategyConfig(
            config_id="IH_VSP3_60PCT",
            strategy="iron_condor", dte=21, short_delta=0.20, width=75.0,
            exit_rules=ExitRules(profit_target_pct=0.50, stop_mult=2.0,
                                 vix_spike_exit_pct=60),
            filters=EntryFilters(),
        ),

        # -----------------------------------------------------------------------
        # BX_ — Baseline experiment: vix_min=14 filter + VIX spike exit sweep
        # Testing worst-month fix for 01_BASELINE_IC14:
        #   BX_VMIN14         : skip entries when VIX < 14 (compressed vol = thin credit)
        #   BX_VMIN14_VSP30   : same + close if VIX rises 30% from entry
        #   BX_VMIN14_VSP40   : same + close if VIX rises 40% from entry
        # These are NOT modified by the 5% momentum-break post-processing loop.
        # -----------------------------------------------------------------------
        StrategyConfig(
            config_id="BX_VMIN14",
            strategy="iron_condor", dte=14, short_delta=0.15, width=50.0,
            exit_rules=ExitRules(profit_target_pct=0.50, stop_mult=2.0),
            filters=EntryFilters(vix_min=14),
        ),
        StrategyConfig(
            config_id="BX_VMIN14_VSP30",
            strategy="iron_condor", dte=14, short_delta=0.15, width=50.0,
            exit_rules=ExitRules(profit_target_pct=0.50, stop_mult=2.0,
                                 vix_spike_exit_pct=30),
            filters=EntryFilters(vix_min=14),
        ),
        StrategyConfig(
            config_id="BX_VMIN14_VSP40",
            strategy="iron_condor", dte=14, short_delta=0.15, width=50.0,
            exit_rules=ExitRules(profit_target_pct=0.50, stop_mult=2.0,
                                 vix_spike_exit_pct=40),
            filters=EntryFilters(vix_min=14),
        ),

        # -----------------------------------------------------------------------
        # VX_ — Vol-regime: enter in elevated VIX, exit on IV contraction
        # Strategy: only enter when VIX >= vix_min (collect elevated premium),
        # then exit early when VIX falls X% from entry VIX (book the vega gain).
        # Both exits compete with profit_target (50% credit captured = also exit).
        #
        # VX1: IC14, vix>=16, close if VIX drops 20%  (e.g. 20→16)
        # VX2: IC14, vix>=16, close if VIX drops 30%  (tighter hold)
        # VX3: IC14, vix>=18, close if VIX drops 25%  (higher quality entries)
        # VX4: IC21, vix>=18, close if VIX drops 25%  (longer DTE, more room)
        # VX5: IC14, vix>=20, close if VIX drops 30%  (stress-entry only)
        # VX6: IC21, vix>=20, close if VIX drops 30%  (stress-entry, longer DTE)
        # -----------------------------------------------------------------------
        StrategyConfig(
            config_id="VX1_IC14_V16_D20",
            strategy="iron_condor", dte=14, short_delta=0.15, width=50.0,
            exit_rules=ExitRules(profit_target_pct=0.50, stop_mult=2.0,
                                 vix_drop_exit_pct=20),
            filters=EntryFilters(vix_min=16),
        ),
        StrategyConfig(
            config_id="VX2_IC14_V16_D30",
            strategy="iron_condor", dte=14, short_delta=0.15, width=50.0,
            exit_rules=ExitRules(profit_target_pct=0.50, stop_mult=2.0,
                                 vix_drop_exit_pct=30),
            filters=EntryFilters(vix_min=16),
        ),
        StrategyConfig(
            config_id="VX3_IC14_V18_D25",
            strategy="iron_condor", dte=14, short_delta=0.15, width=50.0,
            exit_rules=ExitRules(profit_target_pct=0.50, stop_mult=2.0,
                                 vix_drop_exit_pct=25),
            filters=EntryFilters(vix_min=18),
        ),
        StrategyConfig(
            config_id="VX4_IC21_V18_D25",
            strategy="iron_condor", dte=21, short_delta=0.20, width=75.0,
            exit_rules=ExitRules(profit_target_pct=0.50, stop_mult=2.0,
                                 vix_drop_exit_pct=25),
            filters=EntryFilters(vix_min=18),
        ),
        StrategyConfig(
            config_id="VX5_IC14_V20_D30",
            strategy="iron_condor", dte=14, short_delta=0.15, width=50.0,
            exit_rules=ExitRules(profit_target_pct=0.50, stop_mult=2.0,
                                 vix_drop_exit_pct=30),
            filters=EntryFilters(vix_min=20),
        ),
        StrategyConfig(
            config_id="VX6_IC21_V20_D30",
            strategy="iron_condor", dte=21, short_delta=0.20, width=75.0,
            exit_rules=ExitRules(profit_target_pct=0.50, stop_mult=2.0,
                                 vix_drop_exit_pct=30),
            filters=EntryFilters(vix_min=20),
        ),
    ]

    # -----------------------------------------------------------------------
    # Apply 5% SPX-from-entry momentum break to all baseline configs that
    # don't already have an explicit SPX drop exit or momentum break exit.
    # Skipped families: H* (ES hedge), CB* (circuit breaker, already has
    # spx_drop_exit_pct), VD* (rv_expansion exits), WK* (weekend logic),
    # IH_* (experiment configs), BX_* (baseline experiments), ENS_* (ensemble).
    # -----------------------------------------------------------------------
    _skip_prefixes = ("H", "CB", "VD", "WK", "IH_", "BX_", "VX_", "ENS")
    for cfg in configs:
        if cfg.config_id.startswith(_skip_prefixes):
            continue
        if cfg.exit_rules.spx_drop_exit_pct is not None:
            continue
        if cfg.exit_rules.spx_drop_from_entry_pct is not None:
            continue
        cfg.exit_rules.spx_drop_from_entry_pct = 5.0

    return configs


# ---------------------------------------------------------------------------
# Rolling features for ML (training export and entry filter)
# ---------------------------------------------------------------------------
def compute_regime(
    vix_percentile_252d: float,
    vix_change_5d: float,
    term_structure_contango: float,
    spike_pct: float = 15.0,
) -> int:
    """Deterministic regime 0-3 at entry. 0=calm, 1=elevated, 2=spiking/flat, 3=stress."""
    vp = float(vix_percentile_252d) if not (isinstance(vix_percentile_252d, float) and np.isnan(vix_percentile_252d)) else 50.0
    vc = float(vix_change_5d) if not (isinstance(vix_change_5d, float) and np.isnan(vix_change_5d)) else 0.0
    contango = bool(term_structure_contango == 1) if not (isinstance(term_structure_contango, float) and np.isnan(term_structure_contango)) else True
    if vp >= 90 or not contango:
        return 3
    if vc > spike_pct:
        return 2
    if 50 <= vp < 80 and contango:
        return 1
    if vp < 50 and vc <= 0 and contango:
        return 0
    return 2


# Lookback feature keys (leakage-safe: all use data up to and including entry_date only)
ROLLING_FEATURE_KEYS_LB = (
    "ret_5d_lb", "ret_10d_lb", "ret_21d_lb",
    "vix_5d_mean_lb", "vix_5d_max_lb", "vix_21d_mean_lb", "vix_21d_max_lb",
    "gap_5d_mean_lb", "gap_5d_std_lb", "rv_ratio_5d_mean_lb",
    "spot_vs_sma10_lb", "spot_vs_sma21_lb",
)


def compute_rolling_features_for_entry(
    entry_date: date,
    spx_close: pd.Series,
    vix_close: pd.Series,
    gap_pct: pd.Series,
    rv_ratio_series: pd.Series,
) -> dict[str, float]:
    """Compute rolling/window features at entry_date for ML. All series must have date index.

    All features are lookback-only (data up to and including entry_date). Keys use _lb suffix.
    Returns dict with keys: ret_5d_lb, ret_10d_lb, ret_21d_lb, vix_5d_mean_lb, vix_5d_max_lb,
    vix_21d_mean_lb, vix_21d_max_lb, gap_5d_mean_lb, gap_5d_std_lb, rv_ratio_5d_mean_lb,
    spot_vs_sma10_lb, spot_vs_sma21_lb. Values are float or np.nan when insufficient history.
    """
    out: dict[str, float] = {}
    if entry_date not in spx_close.index:
        for k in ROLLING_FEATURE_KEYS_LB:
            out[k] = np.nan
        return out

    close_up_to = spx_close.loc[:entry_date].sort_index()
    if len(close_up_to) < 2:
        for k in ROLLING_FEATURE_KEYS_LB:
            out[k] = np.nan
        return out

    # Returns (log return over N trading days ending on entry_date) — lookback only
    for n, key in [(5, "ret_5d_lb"), (10, "ret_10d_lb"), (21, "ret_21d_lb")]:
        if len(close_up_to) >= n + 1:
            c0 = float(close_up_to.iloc[-(n + 1)])
            c1 = float(close_up_to.iloc[-1])
            out[key] = float(np.log(c1 / c0)) if c0 > 0 else np.nan
        else:
            out[key] = np.nan

    # VIX rolling (lookback)
    vix_up_to = vix_close.loc[:entry_date].sort_index() if not vix_close.empty else pd.Series(dtype=float)
    for n, mean_key, max_key in [(5, "vix_5d_mean_lb", "vix_5d_max_lb"), (21, "vix_21d_mean_lb", "vix_21d_max_lb")]:
        if len(vix_up_to) >= n:
            tail = vix_up_to.tail(n)
            out[mean_key] = float(tail.mean())
            out[max_key] = float(tail.max())
        else:
            out[mean_key] = out[max_key] = np.nan

    # Gap rolling (lookback)
    gap_up_to = gap_pct.loc[:entry_date].sort_index() if not gap_pct.empty else pd.Series(dtype=float)
    if len(gap_up_to) >= 5:
        tail = gap_up_to.tail(5)
        out["gap_5d_mean_lb"] = float(tail.mean())
        out["gap_5d_std_lb"] = float(tail.std()) if len(tail) > 1 else 0.0
    else:
        out["gap_5d_mean_lb"] = out["gap_5d_std_lb"] = np.nan

    # RV ratio 5d mean (lookback)
    rv_up_to = rv_ratio_series.loc[:entry_date].sort_index() if not rv_ratio_series.empty else pd.Series(dtype=float)
    if len(rv_up_to) >= 5:
        out["rv_ratio_5d_mean_lb"] = float(rv_up_to.tail(5).mean())
    else:
        out["rv_ratio_5d_mean_lb"] = np.nan

    # Spot vs SMA (lookback)
    for n, key in [(10, "spot_vs_sma10_lb"), (21, "spot_vs_sma21_lb")]:
        if len(close_up_to) >= n:
            spot = float(close_up_to.iloc[-1])
            sma = float(close_up_to.tail(n).mean())
            out[key] = spot / sma if sma > 0 else np.nan
        else:
            out[key] = np.nan

    return out


# Extended feature series keys (built from full history; lookback applied at entry_date in helper)
EXTENDED_FEATURE_SERIES_KEYS = (
    "vix_change_1d", "vix_change_5d", "vix_percentile_252d",
    "rv_5", "rv_10", "rv_20",
    "vix9d_vix_ratio", "term_structure_contango",
    "spot_vs_sma50", "spot_vs_sma200", "atr_pct_14",
    "gap_abs_z", "gap_direction",
)


def build_extended_feature_series(
    spx_close: pd.Series,
    vix_close: pd.Series,
    gap_pct: pd.Series,
    log_ret: pd.Series,
    *,
    price_df: pd.DataFrame | None = None,
    vix9d: pd.Series | None = None,
    vix3m: pd.Series | None = None,
) -> dict[str, pd.Series]:
    """Build date-indexed series for extended entry features. Used by export and backtest."""
    out: dict[str, pd.Series] = {}
    # VIX changes (pct)
    if not vix_close.empty:
        vix_shift1 = vix_close.shift(1)
        vix_shift5 = vix_close.shift(5)
        out["vix_change_1d"] = (vix_close - vix_shift1) / np.maximum(vix_shift1, 1e-8) * 100
        out["vix_change_5d"] = (vix_close - vix_shift5) / np.maximum(vix_shift5, 1e-8) * 100
        out["vix_percentile_252d"] = vix_close.rolling(252, min_periods=1).apply(
            lambda x: (x <= x.iloc[-1]).sum() / max(1, len(x)) * 100, raw=False
        )
    else:
        out["vix_change_1d"] = out["vix_change_5d"] = out["vix_percentile_252d"] = pd.Series(index=spx_close.index, dtype=float)

    rv_5 = log_ret.rolling(5).std() * np.sqrt(252) * 100
    rv_10 = log_ret.rolling(10).std() * np.sqrt(252) * 100
    rv_20 = log_ret.rolling(20).std() * np.sqrt(252) * 100
    out["rv_5"] = rv_5
    out["rv_10"] = rv_10
    out["rv_20"] = rv_20

    if vix9d is not None and not vix9d.empty and not vix_close.empty:
        common = vix_close.index.intersection(vix9d.index)
        out["vix9d_vix_ratio"] = (vix9d.reindex(common).ffill().bfill() / vix_close.reindex(common).ffill().bfill()).reindex(spx_close.index)
    else:
        out["vix9d_vix_ratio"] = pd.Series(index=spx_close.index, dtype=float)

    if vix3m is not None and not vix3m.empty and not vix_close.empty:
        common = vix_close.index.intersection(vix3m.index)
        out["term_structure_contango"] = (vix_close.reindex(common).ffill().bfill() < vix3m.reindex(common).ffill().bfill()).astype(int).reindex(spx_close.index)
    else:
        out["term_structure_contango"] = pd.Series(index=spx_close.index, dtype=float)

    out["spot_vs_sma50"] = spx_close / spx_close.rolling(50, min_periods=1).mean()
    out["spot_vs_sma200"] = spx_close / spx_close.rolling(200, min_periods=1).mean()

    if price_df is not None and not price_df.empty and "High" in price_df.columns and "Low" in price_df.columns and "Close" in price_df.columns:
        pdf = price_df.copy()
        idx = pd.to_datetime(pdf.index).tz_localize(None).normalize()
        pdf.index = pd.Index(idx.date) if hasattr(idx, "date") else idx
        close = pdf["Close"]
        high = pdf["High"]
        low = pdf["Low"]
        prev_close = close.shift(1)
        tr = pd.concat([high - low, (high - prev_close).abs(), (low - prev_close).abs()], axis=1).max(axis=1)
        atr = tr.rolling(14, min_periods=1).mean()
        out["atr_pct_14"] = (atr / np.maximum(close, 1e-8) * 100).reindex(spx_close.index)
    else:
        out["atr_pct_14"] = pd.Series(index=spx_close.index, dtype=float)

    gap_roll20_mean = gap_pct.rolling(20, min_periods=1).mean()
    gap_roll20_std = gap_pct.rolling(20, min_periods=1).std()
    out["gap_abs_z"] = (gap_pct - gap_roll20_mean) / np.where(gap_roll20_std > 0, gap_roll20_std, np.nan)
    out["gap_abs_z"] = out["gap_abs_z"].fillna(0)
    out["gap_direction"] = np.sign(gap_pct).replace(0, 0)

    return out


def compute_entry_features_extended(
    entry_date: date,
    spx_close: pd.Series,
    vix_close: pd.Series,
    gap_pct: pd.Series,
    rv_ratio_series: pd.Series,
    *,
    optional_series: dict[str, pd.Series] | None = None,
    iv_at_entry: float | None = None,
    gex_score_at_entry: int | float | None = None,
    rv_ratio_at_entry: float | None = None,
) -> dict[str, float]:
    """Base rolling (_lb) features plus extended (vol trend, term structure, regime, interactions).

    optional_series: name -> date-indexed series for vix_change_1d, vix_change_5d,
    vix_percentile_252d, rv_5, rv_10, rv_20, vix9d_vix_ratio, term_structure_contango,
    spot_vs_sma50, spot_vs_sma200, atr_pct_14, gap_abs_z, gap_direction.
    """
    out = compute_rolling_features_for_entry(
        entry_date, spx_close, vix_close, gap_pct, rv_ratio_series
    )
    optional_series = optional_series or {}
    for k, ser in optional_series.items():
        if ser is not None and not ser.empty and entry_date in ser.index:
            try:
                out[k] = float(ser.loc[entry_date])
            except (TypeError, ValueError):
                out[k] = np.nan
        else:
            out[k] = np.nan

    rv_5 = out.get("rv_5", np.nan)
    rv_20 = out.get("rv_20", np.nan)
    if not np.isnan(rv_5) and not np.isnan(rv_20):
        out["rv_trend"] = float(rv_5 - rv_20)
    else:
        out["rv_trend"] = np.nan

    if iv_at_entry is not None and not np.isnan(rv_20):
        out["iv_rv_spread"] = float(iv_at_entry) - float(rv_20)
    else:
        out["iv_rv_spread"] = np.nan

    vp = out.get("vix_percentile_252d", np.nan)
    vc5 = out.get("vix_change_5d", np.nan)
    cont = out.get("term_structure_contango", np.nan)
    out["regime"] = float(compute_regime(vp, vc5, cont))

    vix_at_entry_val = float(iv_at_entry) if iv_at_entry is not None else None
    if rv_ratio_at_entry is not None and vix_at_entry_val is not None:
        out["vix_at_entry_x_rv_ratio_at_entry"] = vix_at_entry_val * float(rv_ratio_at_entry)
    else:
        out["vix_at_entry_x_rv_ratio_at_entry"] = np.nan

    gex = float(gex_score_at_entry) if gex_score_at_entry is not None else None
    vc5_val = out.get("vix_change_5d", np.nan)
    if gex is not None and not np.isnan(vc5_val):
        out["gex_score_at_entry_x_vix_change_5d"] = gex * vc5_val
    else:
        out["gex_score_at_entry_x_vix_change_5d"] = np.nan

    sma21 = out.get("spot_vs_sma21_lb", np.nan)
    if not np.isnan(sma21) and not np.isnan(vc5_val):
        out["spot_vs_sma21_lb_x_vix_change_5d"] = sma21 * vc5_val
    else:
        out["spot_vs_sma21_lb_x_vix_change_5d"] = np.nan

    return out


# ---------------------------------------------------------------------------
# Get trades only (for training data export)
# ---------------------------------------------------------------------------
def get_backtest_trades(
    price_df: pd.DataFrame,
    iv_series: pd.Series,
    start: date,
    end: date,
    *,
    gex_scores: pd.Series | None = None,
    enable_es_hedge: bool = True,
    entry_filter_model_path: Path | None = None,
    entry_filter_threshold: float = 0.5,
    entry_filter_log_path: Path | None = None,
) -> pd.DataFrame:
    """Run backtest and return the trades DataFrame (no report, no CSV write).

    Same data setup as run_backtest_and_report; use this when you only need
    the enriched trade rows for training data or analysis.
    If entry_filter_model_path is set, trades with P(loss) > entry_filter_threshold are skipped.
    If entry_filter_log_path is set, append per-candidate log when the filter is active.
    """
    if price_df.empty or "Close" not in price_df.columns or "Open" not in price_df.columns:
        return pd.DataFrame()

    spx_close = price_df["Close"].copy()
    spx_close.index = pd.to_datetime(spx_close.index).tz_localize(None).normalize()
    spx_close.index = spx_close.index.date
    spx_open = price_df["Open"].copy()
    spx_open.index = pd.to_datetime(spx_open.index).tz_localize(None).normalize()
    spx_open.index = spx_open.index.date
    prev_close = spx_close.shift(1)
    gap_pct = (spx_open - prev_close) / prev_close * 100.0

    vix_close = iv_series.copy()
    if not vix_close.empty:
        vix_close.index = pd.to_datetime(vix_close.index).tz_localize(None).normalize()
        vix_close.index = vix_close.index.date

    trading = get_trading_days(start, end + timedelta(days=60))
    expirations = [e for e in friday_expirations(start, end) if e in set(trading)]

    log_ret = np.log(spx_close / spx_close.shift(1))
    rv_5d = log_ret.rolling(5).std() * np.sqrt(252) * 100
    rv_10d = log_ret.rolling(10).std() * np.sqrt(252) * 100
    rv_ratio_series = rv_5d / rv_10d

    if gex_scores is None:
        gex_scores = pd.Series(2, index=spx_close.index)

    configs = build_strategy_configs()
    news_dates = get_news_dates(start, end)

    entry_filter_optional_series = None
    if entry_filter_model_path and Path(entry_filter_model_path).exists():
        try:
            import yfinance as yf
            pad_start = (start - timedelta(days=400)).isoformat()
            pad_end = (end + timedelta(days=10)).isoformat()
            vix3m_df = yf.Ticker("^VIX3M").history(start=pad_start, end=pad_end)
            vix9d_df = yf.Ticker("^VIX9D").history(start=pad_start, end=pad_end)
            vix3m_ser = vix3m_df["Close"].copy() if not vix3m_df.empty else None
            vix9d_ser = vix9d_df["Close"].copy() if not vix9d_df.empty else None
            if vix3m_ser is not None:
                vix3m_ser.index = pd.to_datetime(vix3m_ser.index).tz_localize(None).normalize()
                vix3m_ser.index = vix3m_ser.index.date
            if vix9d_ser is not None:
                vix9d_ser.index = pd.to_datetime(vix9d_ser.index).tz_localize(None).normalize()
                vix9d_ser.index = vix9d_ser.index.date
            entry_filter_optional_series = build_extended_feature_series(
                spx_close, vix_close, gap_pct, log_ret,
                price_df=price_df, vix9d=vix9d_ser, vix3m=vix3m_ser,
            )
        except Exception:
            entry_filter_optional_series = None

    trades = run_filtered_backtest(
        configs, spx_close, vix_close, price_df, trading, expirations,
        gex_scores, rv_ratio_series, gap_pct, spx_open,
        news_dates=news_dates,
        enable_es_hedge=enable_es_hedge,
        entry_filter_model_path=entry_filter_model_path,
        entry_filter_threshold=entry_filter_threshold,
        entry_filter_optional_series=entry_filter_optional_series,
        entry_filter_log_path=entry_filter_log_path,
    )
    return trades


# ---------------------------------------------------------------------------
# Shared run: backtest + metrics + report (for CLI and pipeline)
# ---------------------------------------------------------------------------
def run_backtest_and_report(
    price_df: pd.DataFrame,
    iv_series: pd.Series,
    start: date,
    end: date,
    output_path: Path,
    *,
    gex_scores: pd.Series | None = None,
    enable_es_hedge: bool = True,
    report_title: str | None = None,
    trades_csv: Path | None = None,
    verbose: bool = True,
    entry_filter_model_path: Path | None = None,
    entry_filter_threshold: float = 0.5,
    entry_filter_log_path: Path | None = None,
) -> int:
    """Run backtest, compute metrics and ensemble, write HTML report.

    price_df must have Close and Open columns; index will be normalized to date.
    iv_series is VIX or other IV proxy (index = date). If gex_scores is None,
    a pass-all series (score 2 every day) is used so all configs run.

    If entry_filter_model_path is set, a second backtest is run with the XGBoost
    entry filter and the report includes a comparison section (baseline vs filtered).
    If entry_filter_log_path is set, append per-candidate entry log when the filter is active.
    """
    if price_df.empty or "Close" not in price_df.columns or "Open" not in price_df.columns:
        if verbose:
            print("No price data or missing Close/Open.", file=sys.stderr)
        return 1

    spx_close = price_df["Close"].copy()
    spx_close.index = pd.to_datetime(spx_close.index).tz_localize(None).normalize()
    spx_close.index = spx_close.index.date
    spx_open = price_df["Open"].copy()
    spx_open.index = pd.to_datetime(spx_open.index).tz_localize(None).normalize()
    spx_open.index = spx_open.index.date
    prev_close = spx_close.shift(1)
    gap_pct = (spx_open - prev_close) / prev_close * 100.0

    vix_close = iv_series.copy()
    if not vix_close.empty:
        vix_close.index = pd.to_datetime(vix_close.index).tz_localize(None).normalize()
        vix_close.index = vix_close.index.date

    trading = get_trading_days(start, end + timedelta(days=60))
    expirations = [e for e in friday_expirations(start, end) if e in set(trading)]

    log_ret = np.log(spx_close / spx_close.shift(1))
    rv_5d = log_ret.rolling(5).std() * np.sqrt(252) * 100
    rv_10d = log_ret.rolling(10).std() * np.sqrt(252) * 100
    rv_ratio_series = rv_5d / rv_10d

    if gex_scores is None:
        gex_scores = pd.Series(2, index=spx_close.index)

    configs = build_strategy_configs()
    news_dates = get_news_dates(start, end)
    if verbose:
        print(f"Configs: {len(configs)}, Expirations: {len(expirations)}", file=sys.stderr)
        print(f"News-day filter: {len(news_dates)} dates in [{start}, {end}]", file=sys.stderr)

    # Load VIX9D once for all backtest runs (used by vix_backwardation_exit)
    vix9d_ser: pd.Series | None = None
    try:
        import yfinance as yf
        _pad_s = (start - timedelta(days=30)).isoformat()
        _pad_e = (end + timedelta(days=10)).isoformat()
        _vix9d_df = yf.download("^VIX9D", start=_pad_s, end=_pad_e, auto_adjust=True, progress=False)
        if not _vix9d_df.empty:
            vix9d_ser = _vix9d_df["Close"].squeeze().copy()
            vix9d_ser.index = pd.to_datetime(vix9d_ser.index).tz_localize(None).normalize()
            vix9d_ser.index = vix9d_ser.index.date
    except Exception:
        vix9d_ser = None

    trades = run_filtered_backtest(
        configs, spx_close, vix_close, price_df, trading, expirations,
        gex_scores, rv_ratio_series, gap_pct, spx_open,
        news_dates=news_dates,
        enable_es_hedge=enable_es_hedge,
        vix9d=vix9d_ser,
    )
    if verbose:
        print(f"Total trades (baseline): {len(trades)}", file=sys.stderr)

    # Optional: run again with entry filter for comparison
    trades_filtered: pd.DataFrame | None = None
    monthly_filtered: pd.DataFrame | None = None
    equity_filtered: pd.DataFrame | None = None
    summary_filtered: dict | None = None
    entry_filter_label: str | None = None
    if entry_filter_model_path and Path(entry_filter_model_path).exists():
        if verbose:
            print(f"Running with entry filter (threshold={entry_filter_threshold})...", file=sys.stderr)
        entry_filter_optional_series = None
        try:
            import yfinance as yf
            pad_start = (start - timedelta(days=400)).isoformat()
            pad_end = (end + timedelta(days=10)).isoformat()
            vix3m_df = yf.Ticker("^VIX3M").history(start=pad_start, end=pad_end)
            vix9d_df = yf.Ticker("^VIX9D").history(start=pad_start, end=pad_end)
            vix3m_ser = vix3m_df["Close"].copy() if not vix3m_df.empty else None
            vix9d_ser = vix9d_df["Close"].copy() if not vix9d_df.empty else None
            if vix3m_ser is not None:
                vix3m_ser.index = pd.to_datetime(vix3m_ser.index).tz_localize(None).normalize()
                vix3m_ser.index = vix3m_ser.index.date
            if vix9d_ser is not None:
                vix9d_ser.index = pd.to_datetime(vix9d_ser.index).tz_localize(None).normalize()
                vix9d_ser.index = vix9d_ser.index.date
            entry_filter_optional_series = build_extended_feature_series(
                spx_close, vix_close, gap_pct, log_ret,
                price_df=price_df, vix9d=vix9d_ser, vix3m=vix3m_ser,
            )
        except Exception:
            entry_filter_optional_series = None
        trades_filtered = run_filtered_backtest(
            configs, spx_close, vix_close, price_df, trading, expirations,
            gex_scores, rv_ratio_series, gap_pct, spx_open,
            news_dates=news_dates,
            enable_es_hedge=enable_es_hedge,
            vix9d=vix9d_ser,
            entry_filter_model_path=entry_filter_model_path,
            entry_filter_threshold=entry_filter_threshold,
            entry_filter_optional_series=entry_filter_optional_series,
            entry_filter_log_path=entry_filter_log_path,
        )
        if verbose:
            print(f"Total trades (filtered): {len(trades_filtered)}", file=sys.stderr)
        monthly_filtered, equity_filtered, summary_filtered = compute_enhanced_metrics(trades_filtered)
        ensemble_weights = {
            "02_IC21_CLOSE_1DTE": 1.0 / 3.0,
            "H2_ES_HEDGE_2DTE": 1.0 / 3.0,
            "H4_VD_GAP_HEDGE_5DTE": 1.0 / 3.0,
        }
        summary_filtered["ENS_02_H2_H4"] = compute_ensemble_metrics(
            trades_filtered, ensemble_weights, "ENS_02_H2_H4"
        )
        entry_filter_label = f"P(loss)>{entry_filter_threshold}"

    if trades_csv is not None:
        trades_csv.parent.mkdir(parents=True, exist_ok=True)
        trades.to_csv(trades_csv, index=False)
        if verbose:
            print(f"Wrote {trades_csv}", file=sys.stderr)

    monthly, equity_df, summary = compute_enhanced_metrics(trades)
    ensemble_weights = {
        "02_IC21_CLOSE_1DTE": 1.0 / 3.0,
        "H2_ES_HEDGE_2DTE": 1.0 / 3.0,
        "H4_VD_GAP_HEDGE_5DTE": 1.0 / 3.0,
    }
    summary["ENS_02_H2_H4"] = compute_ensemble_metrics(trades, ensemble_weights, "ENS_02_H2_H4")

    if summary and verbose:
        print("\n" + "=" * 100, file=sys.stderr)
        print("RESULTS RANKED BY ANNUALIZED SHARPE", file=sys.stderr)
        print("=" * 100, file=sys.stderr)
        ranked = sorted(summary.items(), key=lambda x: x[1]["ann_sharpe"], reverse=True)
        fmt = "{:<22s} {:>6} {:>12s} {:>10s} {:>7s} {:>6s} {:>8s} {:>8s} {:>8s} {:>10s} {:>8s}"
        print(fmt.format("Config", "Trades", "Total P&L", "Avg/Mo", "Win%", "PF",
                          "Sharpe", "Sortino", "Calmar", "Max DD", "Cap eff"), file=sys.stderr)
        print("-" * 110, file=sys.stderr)
        for cid, m in ranked:
            print(fmt.format(
                cid, m["n_trades"],
                f"${m['total_pnl']:,.0f}", f"${m['avg_monthly_pnl']:,.0f}",
                f"{m['win_rate']:.1f}", f"{m['profit_factor']:.2f}",
                f"{m['ann_sharpe']:.3f}", f"{m['sortino']:.3f}", f"{m['calmar']:.2f}",
                f"${m['max_drawdown']:,.0f}",
                f"{m.get('capital_efficiency', m.get('recovery_factor', 0)):.2f}",
            ), file=sys.stderr)
        print("=" * 100, file=sys.stderr)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    build_enhanced_report(
        trades, monthly, equity_df, summary, output_path, report_title=report_title,
        trades_filtered=trades_filtered,
        monthly_filtered=monthly_filtered,
        equity_filtered=equity_filtered,
        summary_filtered=summary_filtered,
        entry_filter_label=entry_filter_label,
    )
    if verbose:
        print(f"\nWrote {output_path}", file=sys.stderr)
    return 0


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> int:
    parser = argparse.ArgumentParser(description="Enhanced SPX Cash Flow Evaluator")
    parser.add_argument("--start", type=str, default="2015-01-01")
    parser.add_argument("--end", type=str, default="2025-12-31")
    parser.add_argument("--trades-csv", type=Path,
                        default=ROOT / "data" / "enhanced_cashflow_trades.csv")
    parser.add_argument("--report-html", type=Path,
                        default=ROOT / "data" / "enhanced_cashflow_report.html")
    parser.add_argument("--entry-filter-model", type=Path,
                        default=None,
                        help="XGBoost entry filter joblib; enables baseline vs filtered comparison in report.")
    parser.add_argument("--entry-filter-threshold", type=float, default=0.5,
                        help="P(loss) threshold when using --entry-filter-model.")
    parser.add_argument("--entry-filter-log", type=Path, default=None,
                        help="When using entry filter, append one row per candidate entry to this CSV (entry_date, config_id, p_loss, decision, key features).")
    args = parser.parse_args()

    start = datetime.strptime(args.start, "%Y-%m-%d").date()
    end = datetime.strptime(args.end, "%Y-%m-%d").date()

    print("Loading SPX + VIX...", file=sys.stderr)
    spx_df, vix_df = load_spx_vix(start, end)
    if spx_df.empty:
        print("No SPX data.", file=sys.stderr)
        return 1

    vix_close = vix_df["Close"].copy() if not vix_df.empty else pd.Series(dtype=float)
    if not vix_close.empty:
        vix_close.index = vix_close.index.date

    configs = build_strategy_configs()
    any_gex = any(c.filters.gex_min_score is not None for c in configs)
    gex_scores = None
    if any_gex:
        print("Computing GEX regime proxy (VIX term structure + RV + momentum)...", file=sys.stderr)
        gex_scores = compute_gex_proxy(spx_df, vix_close, start, end)
        if not gex_scores.empty:
            pos_days = (gex_scores >= 2).sum()
            total_days = len(gex_scores)
            print(f"  GEX proxy: {pos_days}/{total_days} days positive (score>=2), "
                  f"{(gex_scores >= 3).sum()} days strongly positive (score=3)", file=sys.stderr)

    return run_backtest_and_report(
        spx_df,
        vix_close,
        start,
        end,
        args.report_html,
        gex_scores=gex_scores,
        enable_es_hedge=True,
        report_title="SPX",
        trades_csv=args.trades_csv,
        verbose=True,
        entry_filter_model_path=args.entry_filter_model,
        entry_filter_threshold=args.entry_filter_threshold,
        entry_filter_log_path=args.entry_filter_log,
    )


if __name__ == "__main__":
    sys.exit(main())
