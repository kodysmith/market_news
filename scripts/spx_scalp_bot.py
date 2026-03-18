#!/usr/bin/env python3
"""
SPX Day-Trading Scalp Bot (Paper/Simulation Mode)

Evaluates SPX scalping opportunities every 60 seconds using:
  - GEX regime & flip line (from Massive.com options chains)
  - VIX level & term structure
  - SPX price action (1-min bars via yfinance)
  - SMA/EMA momentum indicators

All trades are logged to a CSV file for post-processing analysis.
No real orders are placed — this is a signal-generation & logging bot.

Usage:
    source venv/bin/activate
    python scripts/spx_scalp_bot.py [--interval 60] [--log-dir logs/scalp_bot]
"""

from __future__ import annotations

import argparse
import atexit
import csv
import json
import logging
import math
import os
import signal
import sys
import time
from dataclasses import dataclass, asdict, field
from datetime import datetime, date, timedelta
from pathlib import Path
from typing import Optional
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import yfinance as yf
from scipy.stats import norm

# Add project root to path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from QuantEngine.gex_calculator import (
    get_option_chain_snapshot,
    get_spot_price,
    compute_cockpit_state,
)

ET = ZoneInfo("America/New_York")

# ─── Slippage & Commission Constants (defaults, overridden by account profile)
SLIPPAGE_PCT = 0.02       # 0.02% per fill
COMMISSION_PER_TRADE = 0.65  # dollars per side

# ─── Black-Scholes for 0DTE option pricing ───────────────────────────────────
_BS_R = 0.05

def _bs_d1(S, K, T, sigma):
    if T <= 0 or sigma <= 0:
        return 0.0
    return (math.log(S / K) + (_BS_R + 0.5 * sigma**2) * T) / (sigma * math.sqrt(T))

def bs_call(S, K, T, sigma):
    if T <= 1e-10: return max(S - K, 0.0)
    d1 = _bs_d1(S, K, T, sigma)
    d2 = d1 - sigma * math.sqrt(T)
    return S * norm.cdf(d1) - K * math.exp(-_BS_R * T) * norm.cdf(d2)

def bs_put(S, K, T, sigma):
    if T <= 1e-10: return max(K - S, 0.0)
    d1 = _bs_d1(S, K, T, sigma)
    d2 = d1 - sigma * math.sqrt(T)
    return K * math.exp(-_BS_R * T) * norm.cdf(-d2) - S * norm.cdf(-d1)

def bs_call_delta(S, K, T, sigma):
    if T <= 1e-10 or sigma <= 0:
        return 1.0 if S > K else 0.0
    d1 = _bs_d1(S, K, T, sigma)
    return norm.cdf(d1)

def bs_put_delta(S, K, T, sigma):
    return bs_call_delta(S, K, T, sigma) - 1.0

def strike_for_delta(S, T, sigma, target_delta, is_put=False, inc=1.0):
    """Find the strike closest to target_delta (e.g. -0.10 for 10-delta put).

    For 0DTE, the range is very tight since 1-sigma move is tiny.
    We search up to 3 standard deviations from spot.
    """
    best_K = S
    best_diff = 999
    # Dynamic range based on vol: search ±3 sigma
    one_sigma = S * sigma * math.sqrt(max(T, 1e-8))
    search_range = max(3 * one_sigma, 3 * inc)  # at least 3 increments
    if is_put:
        lo = S - search_range
        # Nearest OTM put strike (below spot, snapped to grid)
        hi = math.floor(S / inc) * inc
        if hi >= S:
            hi -= inc
    else:
        # Nearest OTM call strike (above spot, snapped to grid)
        lo = math.ceil(S / inc) * inc
        if lo <= S:
            lo += inc
        hi = S + search_range
    K = math.ceil(lo / inc) * inc
    while K <= hi:
        d = bs_put_delta(S, K, T, sigma) if is_put else bs_call_delta(S, K, T, sigma)
        diff = abs(d - target_delta)
        if diff < best_diff:
            best_diff = diff
            best_K = K
        K = round(K + inc, 2)
    return best_K

def atm_strike(spot, inc=5.0):
    return round(spot / inc) * inc

def tte_years(ts_str: str) -> float:
    try:
        dt = datetime.strptime(ts_str, "%Y-%m-%d %H:%M:%S")
    except ValueError:
        return 0.01
    close = dt.replace(hour=16, minute=0, second=0)
    mins = max(1, (close - dt).total_seconds() / 60)
    return mins / (365 * 24 * 60)

def tte_minutes(ts_str: str) -> float:
    try:
        dt = datetime.strptime(ts_str, "%Y-%m-%d %H:%M:%S")
    except ValueError:
        return 60
    close = dt.replace(hour=16, minute=0, second=0)
    return max(0, (close - dt).total_seconds() / 60)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("scalp_bot")

# ─── Config ──────────────────────────────────────────────────────────────────

CONFIG_PATH = ROOT / "data" / "config.json"

def load_config() -> dict:
    with open(CONFIG_PATH) as f:
        return json.load(f)

CFG = load_config()
MASSIVE_KEY = CFG.get("MASSIVE_API_KEY", "")
AV_KEY = CFG.get("ALPHAVANTAGE_API_KEY", "")

# ─── Data Structures ─────────────────────────────────────────────────────────

@dataclass
class ScalpSignal:
    timestamp: str
    ticker: str
    spx_price: float
    vix: float
    gex_regime: str          # POSITIVE / NEGATIVE / UNKNOWN
    flip_line: float
    distance_to_flip: float  # points from spot to flip
    net_gex: float
    call_wall: float
    put_wall: float
    ema9: float
    ema21: float
    rsi14: float
    atr14: float             # ATR(14) on 1-min bars
    price_vs_vwap: float     # positive = above VWAP
    signal: str              # LONG / SHORT / FLAT
    reason: str
    confidence: float        # 0-1

@dataclass
class Trade:
    entry_time: str
    entry_price: float
    direction: str           # LONG / SHORT
    signal_reason: str
    ticker: str = ""
    exit_time: str = ""
    exit_price: float = 0.0
    exit_reason: str = ""
    pnl_points: float = 0.0
    gross_pnl_points: float = 0.0
    slippage_cost: float = 0.0
    commission_cost: float = 0.0
    duration_sec: int = 0
    gex_regime_at_entry: str = ""
    vix_at_entry: float = 0.0
    # 0DTE option fields (only populated in option mode)
    mode: str = "spot"             # "spot" or "0dte"
    opt_type: str = ""             # "CALL" or "PUT"
    strike: float = 0.0
    opt_entry: float = 0.0        # option premium at entry
    opt_exit: float = 0.0         # option premium at exit
    contracts: int = 1
    net_dollar: float = 0.0       # net P&L in dollars

@dataclass
class IronCondor:
    """Tracks an open 0DTE iron condor position."""
    entry_time: str
    ticker: str
    spot_at_entry: float
    vix_at_entry: float
    expiry: str                    # "20260311"
    put_short: float               # short put strike
    put_long: float                # long put strike (lower)
    call_short: float              # short call strike
    call_long: float               # long call strike (higher)
    width: float = 5.0             # spread width
    contracts: int = 1
    credit: float = 0.0            # net credit received per contract
    exit_time: str = ""
    exit_debit: float = 0.0        # debit paid to close
    exit_reason: str = ""
    net_pnl: float = 0.0           # (credit - debit - commissions) * contracts * 100

@dataclass
class Straddle:
    """Tracks an open 0DTE straddle for gamma scalping."""
    entry_time: str
    ticker: str
    spot_at_entry: float
    vix_at_entry: float
    expiry: str
    strike: float                  # ATM strike
    contracts: int = 1
    call_entry: float = 0.0        # call premium paid
    put_entry: float = 0.0         # put premium paid
    total_cost: float = 0.0        # call + put per contract
    hedge_count: int = 0           # number of rebalances done
    realized_pnl: float = 0.0     # locked in from closed legs
    exit_time: str = ""
    exit_reason: str = ""
    net_pnl: float = 0.0

# ─── Indicators ───────────────────────────────────────────────────────────────

def compute_atr(df: pd.DataFrame, period: int = 14) -> float:
    """Average True Range from OHLC bars."""
    if len(df) < period + 1:
        # Fallback: use recent bar range
        if not df.empty:
            return float((df["High"] - df["Low"]).tail(min(len(df), period)).mean())
        return 0.0
    high = df["High"]
    low = df["Low"]
    prev_close = df["Close"].shift(1)
    tr = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low - prev_close).abs(),
    ], axis=1).max(axis=1)
    return float(tr.rolling(period).mean().iloc[-1])


def compute_ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span, adjust=False).mean()

def compute_rsi(series: pd.Series, period: int = 14) -> float:
    if len(series) < period + 1:
        return 50.0
    delta = series.diff().dropna()
    gains = delta.where(delta > 0, 0.0)
    losses = -delta.where(delta < 0, 0.0)
    avg_gain = gains.rolling(period).mean().iloc[-1]
    avg_loss = losses.rolling(period).mean().iloc[-1]
    if avg_loss < 1e-9:
        return 100.0
    rs = avg_gain / avg_loss
    return 100.0 - (100.0 / (1.0 + rs))

def compute_vwap(df: pd.DataFrame) -> float:
    """VWAP from intraday bars (needs High, Low, Close, Volume)."""
    if df.empty or "Volume" not in df.columns:
        return 0.0
    typical = (df["High"] + df["Low"] + df["Close"]) / 3.0
    cum_vol = df["Volume"].cumsum()
    cum_tp_vol = (typical * df["Volume"]).cumsum()
    if cum_vol.iloc[-1] < 1:
        return float(typical.iloc[-1])
    return float(cum_tp_vol.iloc[-1] / cum_vol.iloc[-1])

# ─── GEX Fetcher (cached, refreshes every 5 min) ─────────────────────────────

_gex_cache: dict = {"state": None, "fetched_at": 0.0}
GEX_CACHE_TTL = 300  # 5 minutes

def fetch_gex_state(spot: float, ticker: str = "SPX") -> dict:
    """Fetch GEX cockpit state from Massive.com, with 5-min caching."""
    now = time.time()
    if _gex_cache["state"] and (now - _gex_cache["fetched_at"]) < GEX_CACHE_TTL:
        return _gex_cache["state"]

    # Determine strike params based on ticker type
    is_index = ticker.upper() in ("SPX", "NDX", "DJX", "RUT", "XSP", "SPY", "QQQ", "IWM")
    strike_inc = 5.0 if ticker.upper() in ("SPX", "NDX") else 1.0
    strike_range = 50 if is_index else max(10, int(spot * 0.08))  # 8% range for stocks

    try:
        log.info("Fetching GEX for %s from Massive.com ...", ticker)
        snap = get_option_chain_snapshot(
            api_key=MASSIVE_KEY,
            underlying=ticker,
            limit=250,
            days_out=14,
            spot_price=spot,
            strike_range=strike_range,
        )
        state = compute_cockpit_state(
            snap=snap,
            spot=spot,
            ticker=ticker,
            num_strikes_each_side=30,
            strike_increment=strike_inc,
        )
        _gex_cache["state"] = state
        _gex_cache["fetched_at"] = now
        log.info(
            "GEX: regime=%s  flip=%.1f  net_gex=%.0f",
            state.get("regime", "?"),
            state.get("flip") or 0,
            state.get("net_gex") or 0,
        )
        return state
    except Exception as e:
        log.warning("GEX fetch failed: %s", e)
        return _gex_cache["state"] or {}

# ─── Price Bars ───────────────────────────────────────────────────────────────

def yf_symbol(ticker: str) -> str:
    """Map our ticker names to yfinance symbols."""
    INDEX_MAP = {"SPX": "^GSPC", "NDX": "^NDX", "VIX": "^VIX", "DJX": "^DJI", "RUT": "^RUT"}
    return INDEX_MAP.get(ticker.upper(), ticker.upper())


def get_intraday_bars(ticker: str = "SPX") -> pd.DataFrame:
    """Fetch today's 1-min bars from yfinance."""
    sym = yf_symbol(ticker)
    t = yf.Ticker(sym)
    df = t.history(period="1d", interval="1m", auto_adjust=True)
    return df

def get_vix() -> float:
    try:
        v = yf.Ticker("^VIX").history(period="1d", interval="1m", auto_adjust=True)
        return float(v["Close"].iloc[-1]) if not v.empty else 0.0
    except Exception:
        return 0.0


def get_realtime_price(ticker: str) -> Optional[float]:
    """Get real-time price via Massive.com snapshot, fall back to yfinance bar close."""
    try:
        price = get_spot_price(ticker, massive_api_key=MASSIVE_KEY, alphavantage_api_key=AV_KEY)
        if price and price > 0:
            return float(price)
    except Exception as e:
        log.warning("Real-time price fetch failed for %s: %s", ticker, e)
    # Fallback to yfinance
    try:
        bars = get_intraday_bars(ticker)
        if not bars.empty:
            return float(bars["Close"].iloc[-1])
    except Exception:
        pass
    return None


# ─── Lockfile ────────────────────────────────────────────────────────────────

def acquire_lockfile(log_dir: Path) -> Path:
    """Create a PID lockfile. Exit if another instance is running."""
    lockfile = log_dir / ".scalp_bot.lock"
    log_dir.mkdir(parents=True, exist_ok=True)

    if lockfile.exists():
        try:
            old_pid = int(lockfile.read_text().strip())
            os.kill(old_pid, 0)  # Check if alive
            log.error("Another scalp bot instance is running (PID %d). Exiting.", old_pid)
            sys.exit(1)
        except (ProcessLookupError, ValueError):
            # Stale lockfile — process is dead
            log.warning("Removing stale lockfile (PID no longer running)")
        except PermissionError:
            log.error("Another scalp bot instance is running (PID check blocked). Exiting.")
            sys.exit(1)

    lockfile.write_text(str(os.getpid()))

    def _remove_lock():
        try:
            lockfile.unlink(missing_ok=True)
        except Exception:
            pass

    atexit.register(_remove_lock)
    return lockfile


# ─── Signal Logic ─────────────────────────────────────────────────────────────

def generate_signal(
    bars: pd.DataFrame,
    gex: dict,
    vix: float,
    spx_price: float,
    ticker: str = "SPX",
    timestamp: str = "",
) -> ScalpSignal:
    """
    Core signal generation combining GEX regime + momentum + VIX.

    Rules:
      POSITIVE GEX regime (mean-reverting, dealers stabilize):
        - Fade moves away from flip line
        - Buy dips near put wall, sell rips near call wall
      NEGATIVE GEX regime (trending, dealers amplify):
        - Momentum trades in direction of move
        - Wider stops, trend-following
      VIX filter:
        - VIX > 30: reduce size / skip (too volatile for scalps)
        - VIX < 13: skip (too dead, no edge)
    """
    now_str = timestamp or datetime.now(ET).strftime("%Y-%m-%d %H:%M:%S")

    # --- Indicators from 1-min bars ---
    closes = bars["Close"] if not bars.empty else pd.Series([spx_price])
    ema9 = float(compute_ema(closes, 9).iloc[-1])
    ema21 = float(compute_ema(closes, 21).iloc[-1])
    rsi = compute_rsi(closes, 14)
    atr = compute_atr(bars, 14) if not bars.empty else 0.0
    vwap = compute_vwap(bars) if not bars.empty else spx_price
    price_vs_vwap = spx_price - vwap

    # --- GEX data ---
    raw_regime = gex.get("regime", "UNKNOWN") or "UNKNOWN"
    # Normalize: "POSITIVE_GAMMA" → "POSITIVE", "NEGATIVE_GAMMA" → "NEGATIVE"
    regime = raw_regime.replace("_GAMMA", "")

    # Fallback regime classifier when Massive.com returns UNKNOWN
    # Uses realized vol / implied vol ratio (same proxy as backtest)
    if regime == "UNKNOWN" and vix > 0 and len(bars) >= 20:
        recent_returns = bars["Close"].pct_change().dropna().tail(20)
        if len(recent_returns) >= 10:
            realized_vol = float(recent_returns.std()) * np.sqrt(78 * 252) * 100
            rv_ratio = realized_vol / vix
            if rv_ratio < 0.70:
                regime = "POSITIVE"
                log.info("GEX fallback: rv_ratio=%.2f → POSITIVE", rv_ratio)
            elif rv_ratio > 0.95:
                regime = "NEGATIVE"
                log.info("GEX fallback: rv_ratio=%.2f → NEGATIVE", rv_ratio)
            else:
                log.info("GEX fallback: rv_ratio=%.2f → stays UNKNOWN", rv_ratio)

    flip_line = gex.get("flip", 0.0) or 0.0
    net_gex = gex.get("net_gex", 0.0) or 0.0
    dist_to_flip = spx_price - flip_line if flip_line else 0.0

    # Walls — extract from GEX state
    walls_regime = gex.get("walls_regime", {})
    call_wall = walls_regime.get("call_wall", {}).get("strike", 0.0) if isinstance(walls_regime.get("call_wall"), dict) else 0.0
    put_wall = walls_regime.get("put_wall", {}).get("strike", 0.0) if isinstance(walls_regime.get("put_wall"), dict) else 0.0

    signal = "FLAT"
    reason = ""
    confidence = 0.0

    # --- VIX filter ---
    if vix > 30:
        return ScalpSignal(
            timestamp=now_str, ticker=ticker, spx_price=spx_price, vix=vix,
            gex_regime=regime, flip_line=flip_line, distance_to_flip=dist_to_flip,
            net_gex=net_gex, call_wall=call_wall, put_wall=put_wall,
            ema9=ema9, ema21=ema21, rsi14=rsi, atr14=atr, price_vs_vwap=price_vs_vwap,
            signal="FLAT", reason="VIX too high (>30), skip", confidence=0.0,
        )
    if vix < 13:
        return ScalpSignal(
            timestamp=now_str, ticker=ticker, spx_price=spx_price, vix=vix,
            gex_regime=regime, flip_line=flip_line, distance_to_flip=dist_to_flip,
            net_gex=net_gex, call_wall=call_wall, put_wall=put_wall,
            ema9=ema9, ema21=ema21, rsi14=rsi, atr14=atr, price_vs_vwap=price_vs_vwap,
            signal="FLAT", reason="VIX too low (<13), no edge", confidence=0.0,
        )

    # --- POSITIVE GAMMA: mean-reversion / fade extremes ---
    if regime == "POSITIVE":
        # Tier 1 (0.8): GEX wall + RSI extreme — strongest mean-reversion signal
        if put_wall and spx_price < put_wall + 3 and rsi < 30 and price_vs_vwap < 0:
            signal, reason, confidence = "LONG", "POS_GAMMA: put wall + RSI<30 + below VWAP", 0.8
        elif call_wall and spx_price > call_wall - 3 and rsi > 70 and price_vs_vwap > 0:
            signal, reason, confidence = "SHORT", "POS_GAMMA: call wall + RSI>70 + above VWAP", 0.8
        # Tier 2 (0.7): flip line + RSI + VWAP confluence
        elif dist_to_flip < -5 and rsi < 35 and price_vs_vwap < 0 and ema9 < ema21:
            signal, reason, confidence = "LONG", "POS_GAMMA: below flip + oversold + below VWAP", 0.7
        elif dist_to_flip > 5 and rsi > 65 and price_vs_vwap > 0 and ema9 > ema21:
            signal, reason, confidence = "SHORT", "POS_GAMMA: above flip + overbought + above VWAP", 0.7
        # Tier 3 (0.65): RSI extreme + VWAP (no wall/flip needed)
        elif rsi < 25 and price_vs_vwap < 0 and ema9 < ema21:
            signal, reason, confidence = "LONG", "POS_GAMMA: RSI<25 extreme + below VWAP", 0.65
        elif rsi > 75 and price_vs_vwap > 0 and ema9 > ema21:
            signal, reason, confidence = "SHORT", "POS_GAMMA: RSI>75 extreme + above VWAP", 0.65
        else:
            reason = "POS_GAMMA: no setup"

    # --- NEGATIVE GAMMA: trend-following / momentum ---
    elif regime == "NEGATIVE":
        # Tier 1 (0.8): wall break + trend alignment — strongest momentum
        if put_wall and spx_price < put_wall - 2 and ema9 < ema21 and price_vs_vwap < 0:
            signal, reason, confidence = "SHORT", "NEG_GAMMA: broke put wall + trend + VWAP", 0.8
        elif call_wall and spx_price > call_wall + 2 and ema9 > ema21 and price_vs_vwap > 0:
            signal, reason, confidence = "LONG", "NEG_GAMMA: broke call wall + trend + VWAP", 0.8
        # Tier 2 (0.7): EMA + VWAP + strong RSI trend — requires 3 confirmations
        elif ema9 > ema21 and price_vs_vwap > 0 and 45 < rsi < 70:
            signal, reason, confidence = "LONG", "NEG_GAMMA: uptrend (EMA+VWAP+RSI)", 0.7
        elif ema9 < ema21 and price_vs_vwap < 0 and 30 < rsi < 55:
            signal, reason, confidence = "SHORT", "NEG_GAMMA: downtrend (EMA+VWAP+RSI)", 0.7
        else:
            reason = "NEG_GAMMA: no clean setup"

    else:
        # Unknown regime — only trade extreme RSI extremes with full confluence
        if rsi < 20 and price_vs_vwap < -3 and ema9 < ema21:
            signal, reason, confidence = "LONG", "UNKNOWN: extreme oversold (RSI<20 + VWAP + EMA)", 0.65
        elif rsi > 80 and price_vs_vwap > 3 and ema9 > ema21:
            signal, reason, confidence = "SHORT", "UNKNOWN: extreme overbought (RSI>80 + VWAP + EMA)", 0.65
        else:
            reason = "GEX regime unknown, no extreme setup"

    return ScalpSignal(
        timestamp=now_str, ticker=ticker, spx_price=spx_price, vix=vix,
        gex_regime=regime, flip_line=flip_line, distance_to_flip=dist_to_flip,
        net_gex=net_gex, call_wall=call_wall, put_wall=put_wall,
        ema9=ema9, ema21=ema21, rsi14=rsi, atr14=atr, price_vs_vwap=price_vs_vwap,
        signal=signal, reason=reason, confidence=confidence,
    )

# ─── Trade Manager ────────────────────────────────────────────────────────────

class TradeManager:
    """Manages open position, stop-loss, take-profit, and trade logging.

    Supports two modes:
      - "spot": ATR-based stops on underlying price (default)
      - "0dte": 0DTE option buying with premium-based TP/SL + scale-on-trail
    """

    # ─── Default params (overridden by account profile if provided) ──────
    SL_ATR_MULT_POS = 2.0
    SL_ATR_MULT_NEG = 2.5
    TP_ATR_MULT_POS = 3.0
    TP_ATR_MULT_NEG = 5.0
    MIN_NET_PROFIT = 1.0
    CONFIDENCE_THRESHOLD = 0.65
    MAX_HOLD_MINUTES = 20
    COOLDOWN_SEC = 300
    MAX_TRADES_PER_DAY = 8
    TRAIL_ACTIVATE_ATR = 2.0
    TRAIL_DISTANCE_ATR = 1.0
    TRADE_WINDOWS = [(9, 35, 11, 30), (14, 0, 15, 30)]

    def __init__(self, log_dir: Path, profile: dict = None, ibkr=None):
        self.open_trade: Optional[Trade] = None
        self.log_dir = log_dir
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.profile = profile or {}
        self.mode = self.profile.get("mode", "spot")
        self.ibkr = ibkr  # ScalpOrderManager or None

        # Override defaults from profile
        if profile:
            self.CONFIDENCE_THRESHOLD = profile.get("min_confidence", self.CONFIDENCE_THRESHOLD)
            self.COOLDOWN_SEC = profile.get("cooldown_sec", self.COOLDOWN_SEC)
            self.MAX_TRADES_PER_DAY = profile.get("max_trades_per_day",
                                                   profile.get("target_trades_per_day", self.MAX_TRADES_PER_DAY))
            self.MAX_HOLD_MINUTES = profile.get("max_hold_minutes", self.MAX_HOLD_MINUTES)
            # Widen trade windows when using account profiles (signal quality is the filter)
            self.TRADE_WINDOWS = profile.get("trade_windows", [(9, 35, 15, 55)])
            # Spot-mode ATR overrides
            if "sl_atr_mult" in profile:
                self.SL_ATR_MULT_POS = profile["sl_atr_mult"]
                self.SL_ATR_MULT_NEG = profile["sl_atr_mult"]
            if "tp_atr_mult" in profile:
                self.TP_ATR_MULT_POS = profile["tp_atr_mult"]
                self.TP_ATR_MULT_NEG = profile["tp_atr_mult"]
            if "trail_activate_atr" in profile:
                self.TRAIL_ACTIVATE_ATR = profile["trail_activate_atr"]
            if "trail_distance_atr" in profile:
                self.TRAIL_DISTANCE_ATR = profile["trail_distance_atr"]

        today_str = datetime.now(ET).strftime("%Y%m%d")
        self.trade_log_path = log_dir / f"trades_{today_str}.csv"
        self.signal_log_path = log_dir / f"signals_{today_str}.csv"

        self._init_trade_log()
        self._init_signal_log()

        self.last_exit_time: float = 0.0
        self._last_exit_ts: str = ""
        self._trade_atr: float = 0.0
        self._peak_pnl: float = 0.0
        self._scaled: bool = False       # scale-on-trail fired for current trade
        self._today_trades: int = 0
        self._today_date: str = ""
        self._daily_pnl: float = 0.0     # track daily P&L for daily loss limit
        self.stats = {"total": 0, "wins": 0, "losses": 0, "total_pnl": 0.0}
        # IBKR live-order tracking
        self._ibkr_expiry: str = ""      # "20260311"
        self._ibkr_strike: float = 0.0
        self._ibkr_right: str = ""       # "C" or "P"
        self._ibkr_qty: int = 0          # total contracts held

    def _init_trade_log(self):
        if not self.trade_log_path.exists():
            with open(self.trade_log_path, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow([
                    "entry_time", "entry_price", "direction", "signal_reason",
                    "ticker", "exit_time", "exit_price", "exit_reason",
                    "pnl_points", "gross_pnl_points", "slippage_cost",
                    "commission_cost", "duration_sec", "gex_regime", "vix_at_entry",
                    "mode", "opt_type", "strike", "opt_entry", "opt_exit",
                    "contracts", "net_dollar",
                ])

    def _init_signal_log(self):
        if not self.signal_log_path.exists():
            with open(self.signal_log_path, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow([
                    "timestamp", "ticker", "spx_price", "vix", "gex_regime", "flip_line",
                    "distance_to_flip", "net_gex", "call_wall", "put_wall",
                    "ema9", "ema21", "rsi14", "atr14", "price_vs_vwap",
                    "signal", "reason", "confidence",
                ])

    def log_signal(self, sig: ScalpSignal):
        with open(self.signal_log_path, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                sig.timestamp, sig.ticker, f"{sig.spx_price:.2f}", f"{sig.vix:.2f}",
                sig.gex_regime, f"{sig.flip_line:.1f}", f"{sig.distance_to_flip:.1f}",
                f"{sig.net_gex:.0f}", f"{sig.call_wall:.0f}", f"{sig.put_wall:.0f}",
                f"{sig.ema9:.2f}", f"{sig.ema21:.2f}", f"{sig.rsi14:.1f}",
                f"{sig.atr14:.4f}", f"{sig.price_vs_vwap:.2f}", sig.signal, sig.reason,
                f"{sig.confidence:.2f}",
            ])

    def _log_trade(self, trade: Trade):
        with open(self.trade_log_path, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                trade.entry_time, f"{trade.entry_price:.2f}", trade.direction,
                trade.signal_reason, trade.ticker, trade.exit_time,
                f"{trade.exit_price:.2f}", trade.exit_reason,
                f"{trade.pnl_points:.2f}", f"{trade.gross_pnl_points:.2f}",
                f"{trade.slippage_cost:.4f}", f"{trade.commission_cost:.4f}",
                trade.duration_sec, trade.gex_regime_at_entry,
                f"{trade.vix_at_entry:.2f}",
                trade.mode, trade.opt_type, f"{trade.strike:.0f}" if trade.strike else "",
                f"{trade.opt_entry:.2f}" if trade.opt_entry else "",
                f"{trade.opt_exit:.2f}" if trade.opt_exit else "",
                trade.contracts, f"{trade.net_dollar:.2f}" if trade.net_dollar else "",
            ])

    def _in_cooldown(self, current_ts: str = "") -> bool:
        # Timestamp-based cooldown (works in both live and replay mode)
        if current_ts and self._last_exit_ts:
            try:
                now_dt = datetime.strptime(current_ts, "%Y-%m-%d %H:%M:%S")
                exit_dt = datetime.strptime(self._last_exit_ts, "%Y-%m-%d %H:%M:%S")
                return (now_dt - exit_dt).total_seconds() < self.COOLDOWN_SEC
            except ValueError:
                pass
        return (time.time() - self.last_exit_time) < self.COOLDOWN_SEC

    def _in_trade_window(self, ts: str) -> bool:
        """Check if timestamp falls within allowed trading windows."""
        try:
            dt = datetime.strptime(ts, "%Y-%m-%d %H:%M:%S")
        except ValueError:
            return True  # allow if unparseable
        t = dt.time()
        from datetime import time as dtime
        for h1, m1, h2, m2 in self.TRADE_WINDOWS:
            if dtime(h1, m1) <= t <= dtime(h2, m2):
                return True
        return False

    def _check_cost_viable(self, sig: ScalpSignal) -> bool:
        """Pre-flight: will TP after costs exceed MIN_NET_PROFIT?"""
        atr = sig.atr14
        if atr <= 0:
            return False
        if sig.gex_regime == "POSITIVE":
            tp = atr * self.TP_ATR_MULT_POS
        else:
            tp = atr * self.TP_ATR_MULT_NEG
        total_costs = 2 * COMMISSION_PER_TRADE
        net_tp = tp - total_costs
        return net_tp >= self.MIN_NET_PROFIT

    def process_tick(self, sig: ScalpSignal):
        """Called every minute — manage open trade or open new one."""
        self.log_signal(sig)

        # Reset daily counter on new day
        today = sig.timestamp[:10]
        if today != self._today_date:
            self._today_date = today
            self._today_trades = 0
            self._daily_pnl = 0.0

        if self.open_trade:
            if self.mode == "0dte":
                self._check_exit_option(sig)
            else:
                self._check_exit(sig)
        elif sig.signal in ("LONG", "SHORT") and sig.confidence >= self.CONFIDENCE_THRESHOLD:
            # Check daily loss limit
            max_daily = self.profile.get("max_daily_loss")
            if max_daily and self._daily_pnl <= -max_daily:
                log.info("  Daily loss limit ($%.0f) hit, no more entries", max_daily)
                return
            if self._in_cooldown(sig.timestamp):
                log.debug("  Cooldown active, skipping entry")
            elif self._today_trades >= self.MAX_TRADES_PER_DAY:
                log.debug("  Daily trade cap reached (%d), skipping", self.MAX_TRADES_PER_DAY)
            elif not self._in_trade_window(sig.timestamp):
                log.debug("  Outside trade window, skipping")
            elif self.mode == "0dte":
                self._open_option_trade(sig)
            elif not self._check_cost_viable(sig):
                log.debug("  ATR too low — TP after costs < %.1f, skipping", self.MIN_NET_PROFIT)
            else:
                self._open_trade(sig)

    def _open_trade(self, sig: ScalpSignal):
        # Apply slippage: longs fill higher, shorts fill lower
        if sig.signal == "LONG":
            fill_price = sig.spx_price * (1 + SLIPPAGE_PCT / 100)
        else:
            fill_price = sig.spx_price * (1 - SLIPPAGE_PCT / 100)
        self.open_trade = Trade(
            entry_time=sig.timestamp,
            entry_price=round(fill_price, 2),
            direction=sig.signal,
            signal_reason=sig.reason,
            ticker=sig.ticker,
            gex_regime_at_entry=sig.gex_regime,
            vix_at_entry=sig.vix,
        )
        self._trade_atr = sig.atr14
        self._peak_pnl = 0.0
        self._today_trades += 1

        # Log the ATR-based TP/SL levels
        if sig.gex_regime == "POSITIVE":
            sl_pts = sig.atr14 * self.SL_ATR_MULT_POS
            tp_pts = sig.atr14 * self.TP_ATR_MULT_POS
        else:
            sl_pts = sig.atr14 * self.SL_ATR_MULT_NEG
            tp_pts = sig.atr14 * self.TP_ATR_MULT_NEG
        log.info(
            "  >>> ENTRY %s @ %.2f (slipped from %.2f) | ATR=%.2f SL=%.1f TP=%.1f | %s",
            sig.signal, fill_price, sig.spx_price, sig.atr14, sl_pts, tp_pts, sig.reason,
        )

    def _check_exit(self, sig: ScalpSignal):
        t = self.open_trade
        is_long = t.direction == "LONG"
        raw_pnl = (sig.spx_price - t.entry_price) if is_long else (t.entry_price - sig.spx_price)

        atr = self._trade_atr
        if atr <= 0:
            atr = sig.atr14 if sig.atr14 > 0 else t.entry_price * 0.002  # fallback

        # ATR-based stop/target
        if t.gex_regime_at_entry == "POSITIVE":
            sl = atr * self.SL_ATR_MULT_POS
            tp = atr * self.TP_ATR_MULT_POS
        else:
            sl = atr * self.SL_ATR_MULT_NEG
            tp = atr * self.TP_ATR_MULT_NEG

        entry_dt = datetime.strptime(t.entry_time, "%Y-%m-%d %H:%M:%S")
        now_dt = datetime.strptime(sig.timestamp, "%Y-%m-%d %H:%M:%S")
        hold_sec = int((now_dt - entry_dt).total_seconds())

        # Update peak PnL for trailing stop
        self._peak_pnl = max(self._peak_pnl, raw_pnl)

        exit_reason = ""
        exit_price = sig.spx_price  # default: use actual price
        use_actual_price = False     # for signal/regime/time/trail exits

        if raw_pnl <= -sl:
            exit_reason = f"STOP_LOSS ({sl:.1f}pt)"
            exit_price = (t.entry_price - sl) if is_long else (t.entry_price + sl)
        elif raw_pnl >= tp:
            exit_reason = f"TAKE_PROFIT ({tp:.1f}pt)"
            exit_price = (t.entry_price + tp) if is_long else (t.entry_price - tp)
        # Trailing stop: activates after TRAIL_ACTIVATE_ATR×ATR profit
        elif (self._peak_pnl >= atr * self.TRAIL_ACTIVATE_ATR
              and raw_pnl < self._peak_pnl - atr * self.TRAIL_DISTANCE_ATR):
            exit_reason = f"TRAIL_STOP (peak={self._peak_pnl:.1f} now={raw_pnl:.1f})"
            use_actual_price = True
        elif hold_sec >= self.MAX_HOLD_MINUTES * 60:
            exit_reason = f"TIME_EXIT ({self.MAX_HOLD_MINUTES}min)"
            use_actual_price = True
        elif sig.gex_regime != t.gex_regime_at_entry and sig.gex_regime != "UNKNOWN":
            exit_reason = "REGIME_FLIP"
            use_actual_price = True
        elif (is_long and sig.signal == "SHORT") or (not is_long and sig.signal == "LONG"):
            exit_reason = "SIGNAL_REVERSAL"
            use_actual_price = True

        if exit_reason:
            # Apply exit slippage (reverse direction: longs sell lower, shorts buy higher)
            if use_actual_price:
                if is_long:
                    exit_price = sig.spx_price * (1 - SLIPPAGE_PCT / 100)
                else:
                    exit_price = sig.spx_price * (1 + SLIPPAGE_PCT / 100)

            exit_price = round(exit_price, 2)
            gross_pnl = (exit_price - t.entry_price) if is_long else (t.entry_price - exit_price)

            slippage_entry = abs(t.entry_price * SLIPPAGE_PCT / 100)
            slippage_exit = abs(exit_price * SLIPPAGE_PCT / 100) if use_actual_price else 0.0
            total_slippage = slippage_entry + slippage_exit
            total_commission = 2 * COMMISSION_PER_TRADE
            net_pnl = gross_pnl - total_commission

            t.exit_time = sig.timestamp
            t.exit_price = exit_price
            t.exit_reason = exit_reason
            t.gross_pnl_points = round(gross_pnl, 2)
            t.slippage_cost = round(total_slippage, 4)
            t.commission_cost = round(total_commission, 4)
            t.pnl_points = round(net_pnl, 2)
            t.duration_sec = hold_sec

            self._log_trade(t)
            self.stats["total"] += 1
            self.stats["total_pnl"] += net_pnl
            if net_pnl > 0:
                self.stats["wins"] += 1
            else:
                self.stats["losses"] += 1

            wr = (self.stats["wins"] / self.stats["total"] * 100) if self.stats["total"] else 0
            log.info(
                "  <<< EXIT %s @ %.2f | Gross: %+.2f Net: %+.2f pts (slip=%.2f comm=%.2f) | %s | "
                "Session: %d trades, %.1f%% WR, %+.1f pts total",
                t.direction, exit_price, gross_pnl, net_pnl,
                total_slippage, total_commission, exit_reason,
                self.stats["total"], wr, self.stats["total_pnl"],
            )

            self.open_trade = None
            self.last_exit_time = time.time()
            self._last_exit_ts = sig.timestamp
            self._daily_pnl += net_pnl

    # ─── 0DTE Option Mode ──────────────────────────────────────────────────

    def _open_option_trade(self, sig: ScalpSignal):
        """Enter a 0DTE option trade (buy ATM call or put)."""
        p = self.profile
        min_tte = p.get("min_tte_minutes", 30)
        if tte_minutes(sig.timestamp) < min_tte:
            log.debug("  < %d min to expiry, skipping option entry", min_tte)
            return

        # XSP uses SPX/10 as underlying
        ticker = p.get("ticker", sig.ticker)
        if ticker == "XSP":
            opt_spot = sig.spx_price / 10.0
            strike_inc = 1.0
        else:
            opt_spot = sig.spx_price
            strike_inc = 5.0

        K = atm_strike(opt_spot, strike_inc)
        T = tte_years(sig.timestamp)
        sigma = sig.vix / 100.0
        if sigma < 0.05:
            log.debug("  VIX too low for option pricing, skipping")
            return

        if sig.signal == "LONG":
            bs_price = bs_call(opt_spot, K, T, sigma)
            opt_type = "CALL"
            right = "C"
        else:
            bs_price = bs_put(opt_spot, K, T, sigma)
            opt_type = "PUT"
            right = "P"

        min_prem = p.get("min_premium", 2.0)
        if bs_price < min_prem:
            log.debug("  Premium $%.2f < min $%.2f, skipping", bs_price, min_prem)
            return
        mult = p.get("multiplier", 100)
        max_prem = p.get("max_premium", 1500)
        if bs_price * mult > max_prem:
            log.debug("  Premium $%.0f > max $%.0f, skipping", bs_price * mult, max_prem)
            return

        contracts = p.get("contracts", 1)
        comm_per_side = p.get("commission_per_side", COMMISSION_PER_TRADE)

        # ── IBKR execution ──
        opt_price = bs_price  # default: BS model price
        expiry = datetime.now(ET).strftime("%Y%m%d")

        if self.ibkr:
            # Get real quote from IBKR
            quote = self.ibkr.get_quote(expiry, K, right)
            if quote and quote.ask > 0:
                opt_price = quote.ask  # buy at ask
                log.info("  IBKR quote: bid=$%.2f ask=$%.2f mid=$%.2f (BS=$%.2f)",
                         quote.bid, quote.ask, quote.mid, bs_price)
            else:
                log.warning("  No IBKR quote, using BS price $%.2f", bs_price)

            # Check premium limits with real price
            if opt_price < min_prem:
                log.debug("  Real premium $%.2f < min, skipping", opt_price)
                return
            if opt_price * mult > max_prem:
                log.debug("  Real premium $%.0f > max, skipping", opt_price * mult)
                return

            # Place the order
            result = self.ibkr.buy_option(expiry, K, right, contracts, limit_price=opt_price)
            if not result.success:
                log.warning("  IBKR entry order FAILED: %s — skipping trade", result.message)
                return
            opt_price = result.fill_price
            log.info("  IBKR FILL: %d x %s @ $%.2f", contracts, opt_type, opt_price)

            # Store IBKR order state for exit/scale
            self._ibkr_expiry = expiry
            self._ibkr_strike = K
            self._ibkr_right = right
            self._ibkr_qty = contracts

        self.open_trade = Trade(
            entry_time=sig.timestamp,
            entry_price=sig.spx_price,
            direction=sig.signal,
            signal_reason=sig.reason,
            ticker=sig.ticker,
            gex_regime_at_entry=sig.gex_regime,
            vix_at_entry=sig.vix,
            mode="0dte",
            opt_type=opt_type,
            strike=K,
            opt_entry=round(opt_price, 2),
            contracts=contracts,
        )
        self._peak_pnl = 0.0
        self._scaled = False
        self._today_trades += 1

        sl_pct = p.get("option_sl_pct", 35)
        tp_pct = p.get("option_tp_pct", 50)
        log.info(
            "  >>> OPTION ENTRY: Buy %d %s %s K=%.0f @ $%.2f/pt ($%.0f) | "
            "SL=%d%% TP=%d%% | %s%s",
            contracts, opt_type, sig.signal, K, opt_price, opt_price * mult,
            sl_pct, tp_pct, sig.reason,
            " [IBKR LIVE]" if self.ibkr else "",
        )

    def _check_exit_option(self, sig: ScalpSignal):
        """Check exit conditions for an open 0DTE option trade."""
        t = self.open_trade
        p = self.profile
        mult = p.get("multiplier", 100)

        # XSP uses SPX/10 as underlying
        ticker = p.get("ticker", sig.ticker)
        if ticker == "XSP":
            opt_spot = sig.spx_price / 10.0
        else:
            opt_spot = sig.spx_price

        T = tte_years(sig.timestamp)
        sigma = sig.vix / 100.0 if sig.vix > 0 else t.vix_at_entry / 100.0

        # Get current option price — prefer IBKR quote, fall back to BS
        current = None
        if self.ibkr:
            quote = self.ibkr.get_quote(self._ibkr_expiry, t.strike, self._ibkr_right)
            if quote and quote.mid > 0:
                current = quote.mid
                log.debug("  IBKR mid=$%.2f (bid=$%.2f ask=$%.2f)", quote.mid, quote.bid, quote.ask)

        if current is None:
            if t.opt_type == "CALL":
                current = bs_call(opt_spot, t.strike, T, sigma)
            else:
                current = bs_put(opt_spot, t.strike, T, sigma)

        pnl_per_contract = (current - t.opt_entry) * mult
        self._peak_pnl = max(self._peak_pnl, pnl_per_contract)

        # Scale-on-trail: add contracts when gain hits trigger %
        if (p.get("scale_on_trail") and not self._scaled
                and pnl_per_contract >= t.opt_entry * mult
                * p.get("scale_trigger_pct", 20) / 100):
            add_qty = p.get("scale_add_contracts", 1)

            # IBKR: place a real scale-up order
            if self.ibkr:
                result = self.ibkr.scale_up(
                    self._ibkr_expiry, t.strike, self._ibkr_right, add_qty,
                )
                if result.success:
                    t.contracts += add_qty
                    self._ibkr_qty += add_qty
                    self._scaled = True
                    log.info(
                        "  ↑↑ SCALED to %d contracts via IBKR (fill=$%.2f)",
                        t.contracts, result.fill_price,
                    )
                else:
                    log.warning("  Scale-up FAILED: %s — staying at %d cts", result.message, t.contracts)
            else:
                t.contracts += add_qty
                self._scaled = True
                log.info(
                    "  ↑↑ SCALED to %d contracts (pnl/ct=$%+.0f, trigger=%d%%)",
                    t.contracts, pnl_per_contract, p.get("scale_trigger_pct", 20),
                )

        contracts = t.contracts

        entry_dt = datetime.strptime(t.entry_time, "%Y-%m-%d %H:%M:%S")
        now_dt = datetime.strptime(sig.timestamp, "%Y-%m-%d %H:%M:%S")
        hold_sec = int((now_dt - entry_dt).total_seconds())

        is_long = t.direction == "LONG"
        tp_pct = p.get("option_tp_pct", 50)
        sl_pct = p.get("option_sl_pct", 35)
        trail_pct = p.get("option_trail_pct", 25)
        tp_thresh = t.opt_entry * mult * tp_pct / 100
        sl_thresh = t.opt_entry * mult * sl_pct / 100

        exit_reason = ""
        if pnl_per_contract >= tp_thresh:
            exit_reason = f"TAKE_PROFIT ({tp_pct}%)"
        elif pnl_per_contract <= -sl_thresh:
            exit_reason = f"STOP_LOSS ({sl_pct}%)"
        elif (self._peak_pnl > t.opt_entry * mult * 0.20
              and pnl_per_contract < self._peak_pnl * (1 - trail_pct / 100)):
            exit_reason = f"TRAIL_STOP (peak=${self._peak_pnl:.0f})"
        elif hold_sec >= self.MAX_HOLD_MINUTES * 60:
            exit_reason = f"TIME_EXIT ({self.MAX_HOLD_MINUTES}min)"
        elif (is_long and sig.signal == "SHORT") or (not is_long and sig.signal == "LONG"):
            exit_reason = "SIGNAL_REVERSAL"

        if not exit_reason:
            return

        # ── IBKR exit order ──
        actual_exit_price = current
        if self.ibkr:
            sell_qty = self._ibkr_qty
            quote = self.ibkr.get_quote(self._ibkr_expiry, t.strike, self._ibkr_right)
            limit = quote.bid if quote and quote.bid > 0 else None
            result = self.ibkr.sell_option(
                self._ibkr_expiry, t.strike, self._ibkr_right,
                sell_qty, limit_price=limit,
            )
            if result.success:
                actual_exit_price = result.fill_price
                log.info("  IBKR SELL FILL: %d x %s @ $%.2f", sell_qty, t.opt_type, actual_exit_price)
            else:
                log.error("  IBKR SELL FAILED: %s — logging BS price", result.message)

            self._ibkr_qty = 0
            self._ibkr_expiry = ""
            self._ibkr_strike = 0.0
            self._ibkr_right = ""

        comm_per_side = p.get("commission_per_side", COMMISSION_PER_TRADE)
        total_comm = 2 * comm_per_side * contracts
        pnl_per_contract = (actual_exit_price - t.opt_entry) * mult
        gross_dollar = pnl_per_contract * contracts
        net_dollar = gross_dollar - total_comm

        t.exit_time = sig.timestamp
        t.exit_price = sig.spx_price
        t.exit_reason = exit_reason
        t.opt_exit = round(actual_exit_price, 2)
        t.gross_pnl_points = round(gross_dollar, 2)
        t.commission_cost = round(total_comm, 2)
        t.pnl_points = round(net_dollar, 2)
        t.net_dollar = round(net_dollar, 2)
        t.duration_sec = hold_sec

        self._log_trade(t)
        self.stats["total"] += 1
        self.stats["total_pnl"] += net_dollar
        if net_dollar > 0:
            self.stats["wins"] += 1
        else:
            self.stats["losses"] += 1

        wr = (self.stats["wins"] / self.stats["total"] * 100) if self.stats["total"] else 0
        log.info(
            "  <<< OPTION EXIT %s %s | %d cts | $%.2f→$%.2f | "
            "Net: $%+.2f (comm=$%.2f) | %s | Session: %d trades, %.1f%% WR, $%+.0f%s",
            t.opt_type, t.direction, contracts, t.opt_entry, actual_exit_price,
            net_dollar, total_comm, exit_reason,
            self.stats["total"], wr, self.stats["total_pnl"],
            " [IBKR]" if self.ibkr else "",
        )

        self.open_trade = None
        self.last_exit_time = time.time()
        self._last_exit_ts = sig.timestamp
        self._daily_pnl += net_dollar

# ─── Iron Condor Manager (sells 0DTE ICs when FLAT + positive GEX) ───────────

class IronCondorManager:
    """Opens 0DTE iron condors when the scalp signal is FLAT and GEX is positive.

    Positive GEX = dealers hedge in a stabilizing way → market pinned in range
    → ideal for selling premium via iron condors.

    Exits: 50% max profit, directional signal fires, or 15:30 ET (30 min before close).
    """

    IC_CONTRACTS = 1               # number of IC combos
    IC_WIDTH = 5.0                 # $5 wide spreads
    IC_TARGET_DELTA = 0.10         # 10-delta wings
    IC_PROFIT_TARGET_PCT = 50      # close at 50% of credit
    IC_CLOSE_TIME = (15, 30)       # close by 15:30 ET
    IC_MIN_CREDIT = 0.20           # minimum credit to bother ($0.20)
    IC_MIN_TTE_MINUTES = 60        # don't open IC with < 60 min to expiry
    IC_COOLDOWN_SEC = 600          # 10 min cooldown after closing an IC
    IC_MAX_PER_DAY = 3             # max ICs per day
    IC_MAX_VIX = 21.0              # skip ICs when VIX > 21 (big losses in high-vol)
    IC_ENTRY_AFTER = (10, 0)       # wait until 10:00 (let range establish)
    IC_RSI_RANGE = (35, 65)        # RSI must be in this range (confirms range-bound)
    COMM_PER_LEG = 0.65            # commission per leg per side

    def __init__(self, log_dir: Path, ibkr=None, profile: dict = None):
        self.ibkr = ibkr
        self.profile = profile or {}
        self.open_ic: Optional[IronCondor] = None
        self.log_dir = log_dir
        self._last_close_time: float = 0.0
        self._today_count: int = 0
        self._today_date: str = ""
        self.stats = {"total": 0, "wins": 0, "losses": 0, "total_pnl": 0.0}

        # IC trade log
        today_str = datetime.now(ET).strftime("%Y%m%d")
        self.ic_log_path = log_dir / f"ic_trades_{today_str}.csv"
        if not self.ic_log_path.exists():
            with open(self.ic_log_path, "w", newline="") as f:
                csv.writer(f).writerow([
                    "entry_time", "ticker", "spot", "vix", "expiry",
                    "put_short", "put_long", "call_short", "call_long",
                    "width", "contracts", "credit", "exit_time", "exit_debit",
                    "exit_reason", "commission", "net_pnl",
                ])

    def _log_ic(self, ic: IronCondor, commission: float):
        with open(self.ic_log_path, "a", newline="") as f:
            csv.writer(f).writerow([
                ic.entry_time, ic.ticker, f"{ic.spot_at_entry:.2f}",
                f"{ic.vix_at_entry:.1f}", ic.expiry,
                f"{ic.put_short:.0f}", f"{ic.put_long:.0f}",
                f"{ic.call_short:.0f}", f"{ic.call_long:.0f}",
                f"{ic.width:.0f}", ic.contracts, f"{ic.credit:.2f}",
                ic.exit_time, f"{ic.exit_debit:.2f}", ic.exit_reason,
                f"{commission:.2f}", f"{ic.net_pnl:.2f}",
            ])

    def process_tick(self, sig: ScalpSignal, has_directional_trade: bool):
        """Called every tick. Opens IC when flat+positive GEX, manages exits."""
        # Reset daily counter
        today = sig.timestamp[:10]
        if today != self._today_date:
            self._today_date = today
            self._today_count = 0

        if self.open_ic:
            self._check_ic_exit(sig)
        elif not has_directional_trade:
            self._check_ic_entry(sig)

    def _check_ic_entry(self, sig: ScalpSignal):
        """Open an IC if: FLAT signal, positive GEX, low VIX, range-bound RSI."""
        if sig.signal != "FLAT":
            return
        if sig.gex_regime != "POSITIVE":
            return
        if sig.vix > self.IC_MAX_VIX:
            log.debug("  IC: VIX %.1f > %.1f, skipping", sig.vix, self.IC_MAX_VIX)
            return
        if not (self.IC_RSI_RANGE[0] <= sig.rsi14 <= self.IC_RSI_RANGE[1]):
            log.debug("  IC: RSI %.0f outside %s, skipping", sig.rsi14, self.IC_RSI_RANGE)
            return
        if tte_minutes(sig.timestamp) < self.IC_MIN_TTE_MINUTES:
            return
        if self._today_count >= self.IC_MAX_PER_DAY:
            return
        if (time.time() - self._last_close_time) < self.IC_COOLDOWN_SEC:
            return

        # Check time window: not before IC_ENTRY_AFTER and not after IC_CLOSE_TIME
        try:
            dt = datetime.strptime(sig.timestamp, "%Y-%m-%d %H:%M:%S")
            from datetime import time as dtime
            if dt.time() < dtime(*self.IC_ENTRY_AFTER):
                return
            if dt.time() >= dtime(*self.IC_CLOSE_TIME):
                return
        except ValueError:
            return

        # Compute strikes: use actual index level for option pricing
        # sig.spx_price is already at the correct index level (XSP ~676, SPX ~6760)
        ticker = self.profile.get("ticker", sig.ticker)
        spot = sig.spx_price
        inc = 1.0 if ticker == "XSP" else 5.0

        T = tte_years(sig.timestamp)
        sigma = sig.vix / 100.0
        if sigma < 0.05:
            return

        # Find 10-delta strikes
        put_short_K = strike_for_delta(spot, T, sigma, -self.IC_TARGET_DELTA, is_put=True, inc=inc)
        call_short_K = strike_for_delta(spot, T, sigma, self.IC_TARGET_DELTA, is_put=False, inc=inc)
        put_long_K = put_short_K - self.IC_WIDTH
        call_long_K = call_short_K + self.IC_WIDTH

        # Price the IC via BS
        credit = (
            bs_put(spot, put_short_K, T, sigma) - bs_put(spot, put_long_K, T, sigma)
            + bs_call(spot, call_short_K, T, sigma) - bs_call(spot, call_long_K, T, sigma)
        )

        if credit < self.IC_MIN_CREDIT:
            log.debug("  IC credit $%.2f < min $%.2f, skipping", credit, self.IC_MIN_CREDIT)
            return

        expiry = datetime.now(ET).strftime("%Y%m%d")
        contracts = self.IC_CONTRACTS

        # IBKR execution
        actual_credit = credit
        if self.ibkr:
            result = self.ibkr.sell_iron_condor(
                expiry, put_short_K, put_long_K, call_short_K, call_long_K,
                contracts, limit_credit=round(credit, 2),
            )
            if not result.success:
                log.warning("  IC IBKR entry FAILED: %s", result.message)
                return
            actual_credit = result.fill_price

        self.open_ic = IronCondor(
            entry_time=sig.timestamp,
            ticker=sig.ticker,
            spot_at_entry=sig.spx_price,
            vix_at_entry=sig.vix,
            expiry=expiry,
            put_short=put_short_K,
            put_long=put_long_K,
            call_short=call_short_K,
            call_long=call_long_K,
            width=self.IC_WIDTH,
            contracts=contracts,
            credit=round(actual_credit, 2),
        )
        self._today_count += 1

        log.info(
            "  >>> IC ENTRY: Sell %d XSP IC  P%.0f/P%.0f  C%.0f/C%.0f  "
            "credit=$%.2f ($%.0f/ct)  spot=%.2f  VIX=%.1f%s",
            contracts, put_short_K, put_long_K, call_short_K, call_long_K,
            actual_credit, actual_credit * 100, sig.spx_price, sig.vix,
            " [IBKR]" if self.ibkr else "",
        )

    def _check_ic_exit(self, sig: ScalpSignal):
        """Exit IC on: 50% profit, directional signal, time cutoff, or max loss."""
        ic = self.open_ic

        # Get current IC value — use actual index level
        ticker = self.profile.get("ticker", sig.ticker)
        spot = sig.spx_price

        T = tte_years(sig.timestamp)
        sigma = sig.vix / 100.0 if sig.vix > 0 else ic.vix_at_entry / 100.0

        # Current debit to close (BS model)
        current_debit = (
            bs_put(spot, ic.put_short, T, sigma) - bs_put(spot, ic.put_long, T, sigma)
            + bs_call(spot, ic.call_short, T, sigma) - bs_call(spot, ic.call_long, T, sigma)
        )
        current_debit = max(0, current_debit)

        profit_pct = ((ic.credit - current_debit) / ic.credit * 100) if ic.credit > 0 else 0

        exit_reason = ""

        # 1. Take profit at 50%
        if profit_pct >= self.IC_PROFIT_TARGET_PCT:
            exit_reason = f"TAKE_PROFIT ({profit_pct:.0f}%)"

        # 2. Directional signal fires — close IC to make room for scalp
        elif sig.signal in ("LONG", "SHORT") and sig.confidence >= 0.60:
            exit_reason = f"DIRECTIONAL_SIGNAL ({sig.signal})"

        # 3. Time cutoff — 15:30 ET
        else:
            try:
                dt = datetime.strptime(sig.timestamp, "%Y-%m-%d %H:%M:%S")
                from datetime import time as dtime
                if dt.time() >= dtime(*self.IC_CLOSE_TIME):
                    exit_reason = "TIME_EXIT (15:30)"
            except ValueError:
                pass

        # 4. Max loss — cut at 3x credit (wider stop for 0DTE; theta decay helps)
        if not exit_reason and current_debit >= ic.credit * 3:
            exit_reason = f"STOP_LOSS (debit ${current_debit:.2f} > 3x credit)"

        if not exit_reason:
            # Log position status periodically
            log.debug(
                "  IC open: credit=$%.2f  current=$%.2f  profit=%.0f%%",
                ic.credit, current_debit, profit_pct,
            )
            return

        # Execute close
        actual_debit = current_debit
        if self.ibkr:
            result = self.ibkr.close_iron_condor(
                ic.expiry, ic.put_short, ic.put_long,
                ic.call_short, ic.call_long,
                ic.contracts, limit_debit=round(current_debit, 2),
            )
            if result.success:
                actual_debit = result.fill_price
            else:
                log.error("  IC close FAILED: %s — logging BS price", result.message)

        mult = 100
        commission = 4 * self.COMM_PER_LEG * 2 * ic.contracts  # 4 legs × 2 sides
        gross = (ic.credit - actual_debit) * mult * ic.contracts
        net = gross - commission

        ic.exit_time = sig.timestamp
        ic.exit_debit = round(actual_debit, 2)
        ic.exit_reason = exit_reason
        ic.net_pnl = round(net, 2)

        self._log_ic(ic, commission)
        self.stats["total"] += 1
        self.stats["total_pnl"] += net
        if net > 0:
            self.stats["wins"] += 1
        else:
            self.stats["losses"] += 1

        wr = (self.stats["wins"] / self.stats["total"] * 100) if self.stats["total"] else 0
        log.info(
            "  <<< IC EXIT: credit=$%.2f → debit=$%.2f | Gross: $%+.2f Net: $%+.2f "
            "(comm=$%.2f) | %s | ICs: %d trades, %.0f%% WR, $%+.0f%s",
            ic.credit, actual_debit, gross, net, commission, exit_reason,
            self.stats["total"], wr, self.stats["total_pnl"],
            " [IBKR]" if self.ibkr else "",
        )

        self.open_ic = None
        self._last_close_time = time.time()

    def force_close(self, sig: ScalpSignal):
        """Force-close any open IC (market close, etc)."""
        if self.open_ic:
            # Temporarily set close time to now to trigger exit
            orig = self.IC_CLOSE_TIME
            self.IC_CLOSE_TIME = (0, 0)
            self._check_ic_exit(sig)
            self.IC_CLOSE_TIME = orig


# ─── Gamma Scalp Manager (buys 0DTE straddle when FLAT + negative GEX) ───────

class GammaScalpManager:
    """Buys ATM straddle and scalps delta when market oscillates in negative GEX.

    Negative GEX = dealers hedge in a destabilizing way → market whips around
    → realized vol > implied vol → long gamma profits.

    Mechanics:
      1. Buy ATM straddle (call + put at same strike)
      2. When one leg gains > REBALANCE_PCT, sell it and buy new ATM on that side
         (locks in oscillation profit, resets delta to ~0)
      3. Repeat until exit (time, profit target, or regime change)
    """

    GS_CONTRACTS = 1
    GS_REBALANCE_PCT = 30          # rebalance when one leg is up 30%
    GS_PROFIT_TARGET_PCT = 25      # close straddle at 25% net gain
    GS_STOP_LOSS_PCT = 40          # close if straddle value drops 40%
    GS_CLOSE_TIME = (15, 30)       # close by 15:30 ET
    GS_MIN_TTE_MINUTES = 90        # need more TTE for gamma scalp (oscillations take time)
    GS_COOLDOWN_SEC = 600
    GS_MAX_PER_DAY = 2
    GS_MIN_VIX = 16                # don't buy straddle if VIX < 16
    GS_HIGH_VIX_UNKNOWN = 22       # treat UNKNOWN as NEGATIVE when VIX > this
    COMM_PER_LEG = 0.65

    def __init__(self, log_dir: Path, ibkr=None, profile: dict = None):
        self.ibkr = ibkr
        self.profile = profile or {}
        self.open_straddle: Optional[Straddle] = None
        self.log_dir = log_dir
        self._last_close_time: float = 0.0
        self._today_count: int = 0
        self._today_date: str = ""
        self.stats = {"total": 0, "wins": 0, "losses": 0, "total_pnl": 0.0,
                      "rebalances": 0}

        # IBKR tracking for open legs
        self._ibkr_expiry: str = ""
        self._ibkr_strike: float = 0.0
        self._ibkr_call_qty: int = 0
        self._ibkr_put_qty: int = 0

        today_str = datetime.now(ET).strftime("%Y%m%d")
        self.gs_log_path = log_dir / f"gamma_scalp_{today_str}.csv"
        if not self.gs_log_path.exists():
            with open(self.gs_log_path, "w", newline="") as f:
                csv.writer(f).writerow([
                    "entry_time", "ticker", "spot", "vix", "expiry", "strike",
                    "contracts", "call_entry", "put_entry", "total_cost",
                    "hedge_count", "realized_pnl", "exit_time", "exit_reason",
                    "commission", "net_pnl",
                ])

    def _log_gs(self, gs: Straddle, commission: float):
        with open(self.gs_log_path, "a", newline="") as f:
            csv.writer(f).writerow([
                gs.entry_time, gs.ticker, f"{gs.spot_at_entry:.2f}",
                f"{gs.vix_at_entry:.1f}", gs.expiry, f"{gs.strike:.0f}",
                gs.contracts, f"{gs.call_entry:.2f}", f"{gs.put_entry:.2f}",
                f"{gs.total_cost:.2f}", gs.hedge_count,
                f"{gs.realized_pnl:.2f}", gs.exit_time, gs.exit_reason,
                f"{commission:.2f}", f"{gs.net_pnl:.2f}",
            ])

    def process_tick(self, sig: ScalpSignal, has_other_trade: bool):
        today = sig.timestamp[:10]
        if today != self._today_date:
            self._today_date = today
            self._today_count = 0

        if self.open_straddle:
            self._check_rebalance(sig)
            self._check_exit(sig)
        elif not has_other_trade:
            self._check_entry(sig)

    def _check_entry(self, sig: ScalpSignal):
        if sig.signal != "FLAT":
            return
        # Trigger on NEGATIVE GEX, or UNKNOWN with elevated VIX (likely negative gamma)
        is_negative = sig.gex_regime == "NEGATIVE"
        is_unknown_high_vix = (sig.gex_regime == "UNKNOWN"
                               and sig.vix >= self.GS_HIGH_VIX_UNKNOWN)
        if not (is_negative or is_unknown_high_vix):
            return
        if sig.vix < self.GS_MIN_VIX:
            return
        if tte_minutes(sig.timestamp) < self.GS_MIN_TTE_MINUTES:
            return
        if self._today_count >= self.GS_MAX_PER_DAY:
            return
        if (time.time() - self._last_close_time) < self.GS_COOLDOWN_SEC:
            return

        try:
            dt = datetime.strptime(sig.timestamp, "%Y-%m-%d %H:%M:%S")
            from datetime import time as dtime
            if dt.time() >= dtime(*self.GS_CLOSE_TIME):
                return
        except ValueError:
            return

        spot = sig.spx_price
        ticker = self.profile.get("ticker", sig.ticker)
        inc = 1.0 if ticker == "XSP" else 5.0
        K = atm_strike(spot, inc)

        T = tte_years(sig.timestamp)
        sigma = sig.vix / 100.0
        if sigma < 0.05:
            return

        call_price = bs_call(spot, K, T, sigma)
        put_price = bs_put(spot, K, T, sigma)
        total = call_price + put_price

        if total < 0.50:
            log.debug("  GS: straddle cost $%.2f too cheap, skipping", total)
            return

        expiry = datetime.now(ET).strftime("%Y%m%d")
        contracts = self.GS_CONTRACTS

        # IBKR execution — buy call and put separately
        actual_call = call_price
        actual_put = put_price
        if self.ibkr:
            call_result = self.ibkr.buy_option(expiry, K, "C", contracts)
            if not call_result.success:
                log.warning("  GS: call buy FAILED: %s", call_result.message)
                return
            actual_call = call_result.fill_price

            put_result = self.ibkr.buy_option(expiry, K, "P", contracts)
            if not put_result.success:
                log.warning("  GS: put buy FAILED: %s — unwinding call", put_result.message)
                self.ibkr.sell_option(expiry, K, "C", contracts)
                return
            actual_put = put_result.fill_price

            self._ibkr_expiry = expiry
            self._ibkr_strike = K
            self._ibkr_call_qty = contracts
            self._ibkr_put_qty = contracts

        total_actual = actual_call + actual_put
        self.open_straddle = Straddle(
            entry_time=sig.timestamp,
            ticker=sig.ticker,
            spot_at_entry=spot,
            vix_at_entry=sig.vix,
            expiry=expiry,
            strike=K,
            contracts=contracts,
            call_entry=round(actual_call, 2),
            put_entry=round(actual_put, 2),
            total_cost=round(total_actual, 2),
        )
        self._today_count += 1

        log.info(
            "  >>> GAMMA SCALP ENTRY: Buy %d straddle K=%.0f  "
            "call=$%.2f + put=$%.2f = $%.2f ($%.0f/ct)  spot=%.2f  VIX=%.1f%s",
            contracts, K, actual_call, actual_put, total_actual,
            total_actual * 100, spot, sig.vix,
            " [IBKR]" if self.ibkr else "",
        )

    def _check_rebalance(self, sig: ScalpSignal):
        """Rebalance: when one leg profits > threshold, sell it and re-enter at new ATM."""
        gs = self.open_straddle
        spot = sig.spx_price
        T = tte_years(sig.timestamp)
        sigma = sig.vix / 100.0 if sig.vix > 0 else gs.vix_at_entry / 100.0

        # Current leg values
        call_now = bs_call(spot, gs.strike, T, sigma)
        put_now = bs_put(spot, gs.strike, T, sigma)

        # Check if call leg gained enough to rebalance
        call_gain_pct = ((call_now - gs.call_entry) / gs.call_entry * 100) if gs.call_entry > 0 else 0
        put_gain_pct = ((put_now - gs.put_entry) / gs.put_entry * 100) if gs.put_entry > 0 else 0

        rebalance_leg = None
        if call_gain_pct >= self.GS_REBALANCE_PCT:
            rebalance_leg = "call"
        elif put_gain_pct >= self.GS_REBALANCE_PCT:
            rebalance_leg = "put"

        if not rebalance_leg:
            return

        ticker = self.profile.get("ticker", sig.ticker)
        inc = 1.0 if ticker == "XSP" else 5.0
        new_K = atm_strike(spot, inc)

        if rebalance_leg == "call":
            # Sell the winning call, buy new ATM call
            profit = (call_now - gs.call_entry) * 100 * gs.contracts
            new_call_price = bs_call(spot, new_K, T, sigma)

            if self.ibkr:
                sell_r = self.ibkr.sell_option(gs.expiry, gs.strike, "C", gs.contracts)
                if sell_r.success:
                    profit = (sell_r.fill_price - gs.call_entry) * 100 * gs.contracts
                buy_r = self.ibkr.buy_option(gs.expiry, new_K, "C", gs.contracts)
                if buy_r.success:
                    new_call_price = buy_r.fill_price
                    self._ibkr_strike = new_K

            gs.realized_pnl += profit - (2 * self.COMM_PER_LEG * gs.contracts)
            gs.call_entry = round(new_call_price, 2)
            gs.put_entry = round(put_now, 2)  # mark put to current
            gs.strike = new_K
            gs.hedge_count += 1
            self.stats["rebalances"] += 1

            log.info(
                "  ⟳ GS REBALANCE #%d: sold call +$%.0f, new ATM K=%.0f  "
                "realized=$%.0f  spot=%.2f",
                gs.hedge_count, profit, new_K, gs.realized_pnl, spot,
            )

        elif rebalance_leg == "put":
            profit = (put_now - gs.put_entry) * 100 * gs.contracts
            new_put_price = bs_put(spot, new_K, T, sigma)

            if self.ibkr:
                sell_r = self.ibkr.sell_option(gs.expiry, gs.strike, "P", gs.contracts)
                if sell_r.success:
                    profit = (sell_r.fill_price - gs.put_entry) * 100 * gs.contracts
                buy_r = self.ibkr.buy_option(gs.expiry, new_K, "P", gs.contracts)
                if buy_r.success:
                    new_put_price = buy_r.fill_price
                    self._ibkr_strike = new_K

            gs.realized_pnl += profit - (2 * self.COMM_PER_LEG * gs.contracts)
            gs.put_entry = round(new_put_price, 2)
            gs.call_entry = round(call_now, 2)
            gs.strike = new_K
            gs.hedge_count += 1
            self.stats["rebalances"] += 1

            log.info(
                "  ⟳ GS REBALANCE #%d: sold put +$%.0f, new ATM K=%.0f  "
                "realized=$%.0f  spot=%.2f",
                gs.hedge_count, profit, new_K, gs.realized_pnl, spot,
            )

    def _check_exit(self, sig: ScalpSignal):
        gs = self.open_straddle
        spot = sig.spx_price
        T = tte_years(sig.timestamp)
        sigma = sig.vix / 100.0 if sig.vix > 0 else gs.vix_at_entry / 100.0

        call_now = bs_call(spot, gs.strike, T, sigma)
        put_now = bs_put(spot, gs.strike, T, sigma)
        straddle_now = call_now + put_now
        mult = 100

        # Unrealized P&L = current straddle value - entry cost (of current legs)
        unrealized = (straddle_now - (gs.call_entry + gs.put_entry)) * mult * gs.contracts
        total_pnl = gs.realized_pnl + unrealized

        # Total cost basis = original straddle cost
        cost_basis = gs.total_cost * mult * gs.contracts
        total_return_pct = (total_pnl / cost_basis * 100) if cost_basis > 0 else 0

        exit_reason = ""

        # 1. Profit target on total (realized + unrealized)
        if total_return_pct >= self.GS_PROFIT_TARGET_PCT:
            exit_reason = f"TAKE_PROFIT ({total_return_pct:.0f}%)"

        # 2. Stop loss on total
        elif total_return_pct <= -self.GS_STOP_LOSS_PCT:
            exit_reason = f"STOP_LOSS ({total_return_pct:.0f}%)"

        # 3. GEX regime flipped to POSITIVE → close gamma scalp, IC will take over
        elif sig.gex_regime == "POSITIVE":
            exit_reason = "REGIME_FLIP (→ POSITIVE)"

        # 4. Directional signal → close, let directional trade take over
        elif sig.signal in ("LONG", "SHORT") and sig.confidence >= 0.60:
            exit_reason = f"DIRECTIONAL_SIGNAL ({sig.signal})"

        # 5. Time cutoff
        else:
            try:
                dt = datetime.strptime(sig.timestamp, "%Y-%m-%d %H:%M:%S")
                from datetime import time as dtime
                if dt.time() >= dtime(*self.GS_CLOSE_TIME):
                    exit_reason = "TIME_EXIT (15:30)"
            except ValueError:
                pass

        if not exit_reason:
            log.debug(
                "  GS open: cost=$%.2f  now=$%.2f  realized=$%.0f  total=$%+.0f (%.0f%%)",
                gs.total_cost, straddle_now, gs.realized_pnl, total_pnl, total_return_pct,
            )
            return

        # Close both legs
        actual_call_exit = call_now
        actual_put_exit = put_now
        if self.ibkr:
            if self._ibkr_call_qty > 0:
                r = self.ibkr.sell_option(gs.expiry, gs.strike, "C", self._ibkr_call_qty)
                if r.success:
                    actual_call_exit = r.fill_price
            if self._ibkr_put_qty > 0:
                r = self.ibkr.sell_option(gs.expiry, gs.strike, "P", self._ibkr_put_qty)
                if r.success:
                    actual_put_exit = r.fill_price
            self._ibkr_call_qty = 0
            self._ibkr_put_qty = 0

        # Final P&L
        exit_value = (actual_call_exit + actual_put_exit) * mult * gs.contracts
        unrealized_final = exit_value - (gs.call_entry + gs.put_entry) * mult * gs.contracts
        # Entry commission (2 legs) + exit commission (2 legs) + rebalance commissions (already in realized)
        base_commission = 2 * self.COMM_PER_LEG * gs.contracts * 2  # entry + exit
        total_pnl_final = gs.realized_pnl + unrealized_final - base_commission

        gs.exit_time = sig.timestamp
        gs.exit_reason = exit_reason
        gs.net_pnl = round(total_pnl_final, 2)

        self._log_gs(gs, base_commission)
        self.stats["total"] += 1
        self.stats["total_pnl"] += total_pnl_final
        if total_pnl_final > 0:
            self.stats["wins"] += 1
        else:
            self.stats["losses"] += 1

        wr = (self.stats["wins"] / self.stats["total"] * 100) if self.stats["total"] else 0
        log.info(
            "  <<< GAMMA SCALP EXIT: K=%.0f | cost=$%.2f | %d rebalances | "
            "realized=$%+.0f | Net: $%+.2f (%.0f%%) | %s | "
            "GS: %d trades, %.0f%% WR, $%+.0f%s",
            gs.strike, gs.total_cost, gs.hedge_count,
            gs.realized_pnl, total_pnl_final, total_return_pct, exit_reason,
            self.stats["total"], wr, self.stats["total_pnl"],
            " [IBKR]" if self.ibkr else "",
        )

        self.open_straddle = None
        self._last_close_time = time.time()

    def force_close(self, sig: ScalpSignal):
        if self.open_straddle:
            orig = self.GS_CLOSE_TIME
            self.GS_CLOSE_TIME = (0, 0)
            self._check_exit(sig)
            self.GS_CLOSE_TIME = orig


# ─── Strategy Router ─────────────────────────────────────────────────────────

class StrategyRouter:
    """Routes each tick to the appropriate strategy based on signal + GEX regime.

    Manages transitions: closes current strategy's position before pivoting.

    Strategies:
      - DIRECTIONAL: LONG/SHORT signal → 0DTE option buy (TradeManager)
      - IRON_CONDOR: FLAT + POSITIVE GEX → sell IC (IronCondorManager)
      - GAMMA_SCALP: FLAT + NEGATIVE GEX → buy straddle (GammaScalpManager)
      - IDLE: FLAT + UNKNOWN GEX → no trade
    """

    def __init__(self, trade_mgr: TradeManager, ic_mgr: IronCondorManager,
                 gs_mgr: GammaScalpManager):
        self.trade_mgr = trade_mgr
        self.ic_mgr = ic_mgr
        self.gs_mgr = gs_mgr
        self._last_regime: str = ""

    @property
    def active_strategy(self) -> str:
        if self.trade_mgr.open_trade:
            return "DIRECTIONAL"
        if self.ic_mgr.open_ic:
            return "IRON_CONDOR"
        if self.gs_mgr.open_straddle:
            return "GAMMA_SCALP"
        return "IDLE"

    def _target_strategy(self, sig: ScalpSignal) -> str:
        if sig.signal in ("LONG", "SHORT") and sig.confidence >= self.trade_mgr.CONFIDENCE_THRESHOLD:
            return "DIRECTIONAL"
        if sig.signal == "FLAT" and sig.gex_regime == "POSITIVE" and sig.vix <= self.ic_mgr.IC_MAX_VIX:
            return "IRON_CONDOR"
        if sig.signal == "FLAT" and (
            sig.gex_regime == "NEGATIVE"
            or (sig.gex_regime == "UNKNOWN" and sig.vix >= self.gs_mgr.GS_HIGH_VIX_UNKNOWN)
        ):
            return "GAMMA_SCALP"
        return "IDLE"

    def process_tick(self, sig: ScalpSignal):
        """Main entry point — determines strategy and routes accordingly."""
        active = self.active_strategy
        target = self._target_strategy(sig)

        # Log regime changes
        if sig.gex_regime != self._last_regime and self._last_regime:
            log.info("  REGIME SHIFT: %s → %s  (active=%s, target=%s)",
                     self._last_regime, sig.gex_regime, active, target)
        self._last_regime = sig.gex_regime

        # ── Handle transitions ──
        # Directional trades manage their own exits (TP/SL/trail) — don't force close
        # But IC and gamma scalp should yield to directional signals
        if active == "IRON_CONDOR" and target == "DIRECTIONAL":
            log.info("  PIVOT: closing IC for directional signal")
            # IC manager will close on its own via DIRECTIONAL_SIGNAL exit check

        if active == "GAMMA_SCALP" and target == "DIRECTIONAL":
            log.info("  PIVOT: closing gamma scalp for directional signal")
            # GS manager will close on its own via DIRECTIONAL_SIGNAL exit check

        if active == "IRON_CONDOR" and target == "GAMMA_SCALP":
            log.info("  PIVOT: GEX flipped negative — IC will close, gamma scalp next")
            # IC will close via regime-change check

        if active == "GAMMA_SCALP" and target == "IRON_CONDOR":
            log.info("  PIVOT: GEX flipped positive — gamma scalp will close, IC next")
            # GS will close via REGIME_FLIP check

        # ── Process all managers ──
        # Directional always processes (manages its own open trade)
        self.trade_mgr.process_tick(sig)

        # IC processes when no directional trade is open
        has_directional = self.trade_mgr.open_trade is not None
        has_gamma = self.gs_mgr.open_straddle is not None
        self.ic_mgr.process_tick(sig, has_directional_trade=(has_directional or has_gamma))

        # Gamma scalp processes when no directional trade or IC is open
        has_ic = self.ic_mgr.open_ic is not None
        self.gs_mgr.process_tick(sig, has_other_trade=(has_directional or has_ic))

    def force_close_all(self, sig: ScalpSignal):
        """Force close all open positions (market close, shutdown)."""
        if self.ic_mgr.open_ic:
            self.ic_mgr.force_close(sig)
        if self.gs_mgr.open_straddle:
            self.gs_mgr.force_close(sig)

    def summary(self) -> dict:
        return {
            "directional": self.trade_mgr.stats,
            "iron_condor": self.ic_mgr.stats,
            "gamma_scalp": self.gs_mgr.stats,
        }


# ─── Market Hours ─────────────────────────────────────────────────────────────

def is_market_open() -> bool:
    now = datetime.now(ET)
    if now.weekday() >= 5:
        return False
    from datetime import time as dtime
    t = now.time()
    return dtime(9, 30) <= t <= dtime(16, 0)

def minutes_to_close() -> int:
    now = datetime.now(ET)
    close = now.replace(hour=16, minute=0, second=0, microsecond=0)
    return max(0, int((close - now).total_seconds() / 60))

# ─── Main Loop ────────────────────────────────────────────────────────────────

def run(interval: int = 60, log_dir: str = "logs/scalp_bot", ticker: str = "SPX",
        account: str = None, ibkr_mode: str = None):
    log_path = ROOT / log_dir

    # Fix 4: Single-instance lockfile
    acquire_lockfile(log_path)

    # Load account profile if specified
    profile = None
    if account:
        from scripts.scalp_accounts import ACCOUNTS
        if account not in ACCOUNTS:
            log.error("Unknown account '%s'. Available: %s", account, list(ACCOUNTS.keys()))
            sys.exit(1)
        profile = ACCOUNTS[account]
        ticker = profile.get("ticker", ticker)
        log.info("Loaded account profile: %s (%s mode)", profile["name"], profile["mode"])

    # IBKR execution layer
    ibkr = None
    if ibkr_mode and ibkr_mode != "off":
        from scripts.scalp_ibkr import ScalpOrderManager
        ibkr = ScalpOrderManager(mode=ibkr_mode, ticker=ticker)
        ibkr.connect()
        log.info("IBKR execution: %s mode (ticker=%s)", ibkr_mode.upper(), ticker)
        bp = ibkr.get_buying_power()
        log.info("IBKR buying power: $%s", f"{bp:,.0f}")

    mgr = TradeManager(log_path, profile=profile, ibkr=ibkr)
    ic_mgr = IronCondorManager(log_path, ibkr=ibkr, profile=profile)
    gs_mgr = GammaScalpManager(log_path, ibkr=ibkr, profile=profile)
    router = StrategyRouter(mgr, ic_mgr, gs_mgr)

    log.info("=" * 60)
    log.info("%s Scalp Bot starting — interval=%ds  mode=%s", ticker, interval, mgr.mode)
    if profile:
        log.info("Account: %s ($%s)  Scale-on-trail: %s",
                 profile["name"], f"{profile['account_size']:,}",
                 "ON" if profile.get("scale_on_trail") else "OFF")
    if ibkr and ibkr_mode == "dry-run":
        log.info("IBKR: DRY-RUN — quotes only, no real orders")
    elif ibkr:
        log.info("IBKR: %s — LIVE orders enabled", ibkr_mode.upper())
    else:
        log.info("IBKR: OFF — simulation only (CSV logging)")
    log.info("Trade log: %s", mgr.trade_log_path)
    log.info("Signal log: %s", mgr.signal_log_path)
    log.info("=" * 60)

    # Graceful shutdown
    running = [True]
    def _stop(signum, frame):
        log.info("Shutting down...")
        running[0] = False
    signal.signal(signal.SIGINT, _stop)
    signal.signal(signal.SIGTERM, _stop)

    while running[0]:
        try:
            if not is_market_open():
                mtc = minutes_to_close()
                if mtc == 0:
                    now = datetime.now(ET)
                    if now.time() > __import__("datetime").time(16, 0):
                        # After close — force-exit any open trade
                        if mgr.open_trade:
                            if ibkr and ibkr.is_connected():
                                bars = ibkr.get_bars(ticker)
                            else:
                                bars = get_intraday_bars(ticker)
                            spx = float(bars["Close"].iloc[-1]) if not bars.empty else mgr.open_trade.entry_price
                            if ibkr and ibkr.is_connected():
                                vix_now = ibkr.get_vix()
                            else:
                                vix_now = get_vix()
                            # Build a fake signal to trigger exit
                            close_sig = ScalpSignal(
                                timestamp=now.strftime("%Y-%m-%d %H:%M:%S"),
                                ticker=ticker, spx_price=spx, vix=vix_now,
                                gex_regime=mgr.open_trade.gex_regime_at_entry,
                                flip_line=0, distance_to_flip=0, net_gex=0,
                                call_wall=0, put_wall=0, ema9=0, ema21=0,
                                rsi14=50, atr14=0, price_vs_vwap=0,
                                signal="FLAT", reason="MARKET_CLOSE", confidence=0,
                            )
                            if mgr.mode == "0dte":
                                # Force time-exit via large hold time
                                orig_max = mgr.MAX_HOLD_MINUTES
                                mgr.MAX_HOLD_MINUTES = 0
                                mgr._check_exit_option(close_sig)
                                mgr.MAX_HOLD_MINUTES = orig_max
                            else:
                                is_long = mgr.open_trade.direction == "LONG"
                                if is_long:
                                    exit_px = spx * (1 - SLIPPAGE_PCT / 100)
                                else:
                                    exit_px = spx * (1 + SLIPPAGE_PCT / 100)
                                exit_px = round(exit_px, 2)
                                gross_pnl = (exit_px - mgr.open_trade.entry_price) if is_long else (mgr.open_trade.entry_price - exit_px)
                                slippage_entry = abs(mgr.open_trade.entry_price * SLIPPAGE_PCT / 100)
                                slippage_exit = abs(exit_px * SLIPPAGE_PCT / 100)
                                total_commission = 2 * COMMISSION_PER_TRADE
                                net_pnl = gross_pnl - total_commission
                                mgr.open_trade.exit_time = now.strftime("%Y-%m-%d %H:%M:%S")
                                mgr.open_trade.exit_price = exit_px
                                mgr.open_trade.exit_reason = "MARKET_CLOSE"
                                entry_dt = datetime.strptime(mgr.open_trade.entry_time, "%Y-%m-%d %H:%M:%S")
                                mgr.open_trade.duration_sec = int((now.replace(tzinfo=None) - entry_dt).total_seconds())
                                mgr.open_trade.gross_pnl_points = round(gross_pnl, 2)
                                mgr.open_trade.slippage_cost = round(slippage_entry + slippage_exit, 4)
                                mgr.open_trade.commission_cost = round(total_commission, 4)
                                mgr.open_trade.pnl_points = round(net_pnl, 2)
                                mgr._log_trade(mgr.open_trade)
                                log.info("  <<< MARKET CLOSE EXIT @ %.2f | Net PnL: %+.2f", exit_px, net_pnl)
                            mgr.open_trade = None
                        # Force-close any IC or gamma scalp
                        router.force_close_all(close_sig)
                        log.info("Market closed. Directional: %s", mgr.stats)
                        if ic_mgr.stats["total"]:
                            log.info("IC: %s", ic_mgr.stats)
                        if gs_mgr.stats["total"]:
                            log.info("GS: %s", gs_mgr.stats)
                        break
                log.info("Market not open. Waiting...")
                time.sleep(30)
                continue

            # Skip last 5 minutes (liquidity dries up)
            if minutes_to_close() <= 5 and not mgr.open_trade:
                log.info("Last 5 min — no new entries")
                time.sleep(interval)
                continue

            # --- Fetch data ---
            if ibkr and ibkr.is_connected():
                bars = ibkr.get_bars(ticker)
            else:
                bars = get_intraday_bars(ticker)
            if bars.empty:
                log.warning("No bars returned, skipping tick")
                time.sleep(interval)
                continue

            # Use real-time price for decisions; bars only for indicators
            if ibkr and ibkr.is_connected():
                rt_price = ibkr.get_spot_price()
            else:
                rt_price = get_realtime_price(ticker)
            bar_price = float(bars["Close"].iloc[-1])
            spx_price = rt_price if rt_price else bar_price
            if rt_price:
                log.debug("Real-time: %.2f  bar: %.2f  delta=%.2f", rt_price, bar_price, rt_price - bar_price)

            if ibkr and ibkr.is_connected():
                vix = ibkr.get_vix()
                if vix <= 0:
                    vix = get_vix()  # fallback to yfinance
            else:
                vix = get_vix()

            # GEX (cached 5-min refresh)
            gex = fetch_gex_state(spx_price, ticker)

            # Generate signal
            sig = generate_signal(bars, gex, vix, spx_price, ticker=ticker)

            # Log
            regime_str = sig.gex_regime
            log.info(
                "SPX=%.2f  VIX=%.1f  GEX=%s  flip=%.0f  RSI=%.0f  EMA9/21=%.1f/%.1f  sig=%s",
                spx_price, vix, regime_str, sig.flip_line, sig.rsi14,
                sig.ema9, sig.ema21, sig.signal,
            )

            # Route to appropriate strategy based on signal + GEX regime
            router.process_tick(sig)

        except Exception as e:
            log.error("Tick error: %s", e, exc_info=True)

        time.sleep(interval)

    # Print final summary
    log.info("=" * 60)
    log.info("SESSION SUMMARY")
    log.info("  Trades: %d  |  Wins: %d  |  Losses: %d", mgr.stats["total"], mgr.stats["wins"], mgr.stats["losses"])
    if mgr.stats["total"]:
        log.info("  Win Rate: %.1f%%", mgr.stats["wins"] / mgr.stats["total"] * 100)
    log.info("  Directional PnL: $%s", f"{mgr.stats['total_pnl']:+,.2f}")
    if ic_mgr.stats["total"]:
        log.info("  --- Iron Condors ---")
        log.info("  ICs: %d  |  Wins: %d  |  Losses: %d",
                 ic_mgr.stats["total"], ic_mgr.stats["wins"], ic_mgr.stats["losses"])
        log.info("  IC PnL: $%s", f"{ic_mgr.stats['total_pnl']:+,.2f}")
    if gs_mgr.stats["total"]:
        log.info("  --- Gamma Scalps ---")
        log.info("  GS: %d  |  Wins: %d  |  Losses: %d  |  Rebalances: %d",
                 gs_mgr.stats["total"], gs_mgr.stats["wins"], gs_mgr.stats["losses"],
                 gs_mgr.stats["rebalances"])
        log.info("  GS PnL: $%s", f"{gs_mgr.stats['total_pnl']:+,.2f}")
    combined_pnl = mgr.stats["total_pnl"] + ic_mgr.stats["total_pnl"] + gs_mgr.stats["total_pnl"]
    log.info("  COMBINED PnL: $%s", f"{combined_pnl:+,.2f}")
    log.info("  Logs saved to: %s", mgr.log_dir)
    log.info("=" * 60)

    # Disconnect IBKR
    if ibkr:
        ibkr.disconnect()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SPX Scalp Bot")
    parser.add_argument("--interval", type=int, default=60, help="Seconds between ticks (default: 60)")
    parser.add_argument("--log-dir", type=str, default="logs/scalp_bot", help="Log output directory")
    parser.add_argument("--ticker", type=str, default="SPX", help="Ticker to scalp (e.g. TSLA, NVDA, SPX)")
    parser.add_argument("--account", type=str, default=None,
                        help="Account profile from scalp_accounts.py (robinhood, pm, etrade, etrade_xsp, pm_xsp)")
    parser.add_argument("--ibkr", type=str, default=None, choices=["paper", "live", "dry-run"],
                        help="IBKR execution mode: paper (port 7497), live (port 7496), dry-run (no orders)")
    args = parser.parse_args()
    run(interval=args.interval, log_dir=args.log_dir, ticker=args.ticker.upper(),
        account=args.account, ibkr_mode=args.ibkr)
