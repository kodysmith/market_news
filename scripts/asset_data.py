#!/usr/bin/env python3
"""Asset data loading for multi-asset cashflow pipeline.

Maps symbol (SPX, XSP, GLD, SLV) to yfinance ticker and returns
(price_df, iv_series) for backtest. SPX/XSP use VIX as IV; GLD/SLV use
20-day realized volatility.
"""
from __future__ import annotations

from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent

# Symbol (user-facing) -> yfinance ticker
SYMBOL_TO_TICKER: dict[str, str] = {
    "SPX": "^GSPC",
    "XSP": "XSP",
    "GLD": "GLD",
    "SLV": "SLV",
}


def _normalize_index_to_date(series: pd.Series) -> pd.Series:
    s = series.copy()
    if s.empty:
        return s
    s.index = pd.to_datetime(s.index).tz_localize(None).normalize()
    s.index = s.index.date
    return s


def load_asset_data(
    symbol: str,
    start: date,
    end: date,
) -> tuple[pd.DataFrame, pd.Series]:
    """Load price and IV for the given symbol and date range.

    Returns (price_df, iv_series). price_df has Open, High, Low, Close, etc.;
    iv_series is VIX (for SPX/XSP) or 20-day annualized realized vol * 100 (for GLD/SLV).
    Index is normalized to date for both.
    """
    ticker = SYMBOL_TO_TICKER.get(symbol.upper())
    if not ticker:
        return pd.DataFrame(), pd.Series(dtype=float)

    import yfinance as yf

    pad_end = (end + timedelta(days=10)).isoformat()
    pad_start = (start - timedelta(days=60)).isoformat() if symbol.upper() in ("GLD", "SLV") else start.isoformat()

    if symbol.upper() == "SPX":
        from scripts.optimize_spx_verticals_historical import load_spx_vix
        spx_df, vix_df = load_spx_vix(start, end)
        if spx_df.empty:
            return pd.DataFrame(), pd.Series(dtype=float)
        vix_close = vix_df["Close"].copy() if not vix_df.empty else pd.Series(dtype=float)
        if not vix_close.empty:
            vix_close = _normalize_index_to_date(vix_close)
        if not spx_df.empty:
            spx_df.index = pd.to_datetime(spx_df.index).tz_localize(None).normalize()
        return spx_df, vix_close

    if symbol.upper() == "XSP":
        from scripts.optimize_spx_verticals_historical import load_spx_vix
        _, vix_df = load_spx_vix(start, end)
        vix_close = vix_df["Close"].copy() if not vix_df.empty else pd.Series(dtype=float)
        if not vix_close.empty:
            vix_close = _normalize_index_to_date(vix_close)
        xsp = yf.Ticker("XSP").history(start=start.isoformat(), end=pad_end)
        if xsp.empty:
            return pd.DataFrame(), vix_close
        xsp.index = pd.to_datetime(xsp.index).tz_localize(None).normalize()
        return xsp, vix_close

    # GLD / SLV: price only; IV = 20-day realized vol (annualized, as percent)
    price_df = yf.Ticker(ticker).history(start=pad_start, end=pad_end)
    if price_df.empty or "Close" not in price_df.columns:
        return pd.DataFrame(), pd.Series(dtype=float)
    price_df.index = pd.to_datetime(price_df.index).tz_localize(None).normalize()
    close = price_df["Close"]
    log_ret = np.log(close / close.shift(1))
    rv_20d = log_ret.rolling(20).std() * np.sqrt(252) * 100  # annualized, percent
    iv_series = _normalize_index_to_date(rv_20d).fillna(20.0)  # default 20% vol for early days
    return price_df, iv_series
