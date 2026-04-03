"""Technical indicators for intraday trading — VWAP, ATR, EMA, RSI."""
from __future__ import annotations

import numpy as np
import pandas as pd


def compute_vwap(df: pd.DataFrame) -> pd.Series:
    """Cumulative VWAP from OHLCV bars. Returns a Series aligned with df index."""
    typical = (df["High"] + df["Low"] + df["Close"]) / 3.0
    cum_vol = df["Volume"].cumsum()
    cum_tp_vol = (typical * df["Volume"]).cumsum()
    return cum_tp_vol / cum_vol.replace(0, 1)


def compute_ema(series: pd.Series, span: int) -> pd.Series:
    """Exponential moving average."""
    return series.ewm(span=span, adjust=False).mean()


def compute_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """Average True Range."""
    high = df["High"]
    low = df["Low"]
    close = df["Close"]
    tr = pd.concat([
        high - low,
        (high - close.shift(1)).abs(),
        (low - close.shift(1)).abs(),
    ], axis=1).max(axis=1)
    return tr.rolling(period, min_periods=1).mean()


def compute_rsi(series: pd.Series, period: int = 14) -> pd.Series:
    """Relative Strength Index."""
    delta = series.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = (-delta).where(delta < 0, 0.0)
    avg_gain = gain.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, 1e-10)
    return 100 - (100 / (1 + rs))


def opening_range(df: pd.DataFrame, n_bars: int = 15) -> tuple[float, float]:
    """Compute opening range high and low from first n_bars (1-min bars → 15 min)."""
    or_bars = df.iloc[:n_bars]
    return float(or_bars["High"].max()), float(or_bars["Low"].min())


def is_green(bar) -> bool:
    """Is this a green (bullish) candle?"""
    return bar["Close"] > bar["Open"]


def is_red(bar) -> bool:
    """Is this a red (bearish) candle?"""
    return bar["Close"] < bar["Open"]


def candle_body_position(bar) -> float:
    """Where the close sits in the bar range. 1.0 = closed at high, 0.0 = closed at low."""
    rng = bar["High"] - bar["Low"]
    if rng <= 0:
        return 0.5
    return (bar["Close"] - bar["Low"]) / rng
