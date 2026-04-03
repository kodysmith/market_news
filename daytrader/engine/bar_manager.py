"""Multi-ticker 1-minute bar manager.

Polls for 1-min bars from yfinance (fallback) or Polygon. Provides a
clean DataFrame interface for the entry engine to consume.
"""
from __future__ import annotations

import logging
from datetime import datetime, timedelta
from typing import Optional

import pandas as pd

logger = logging.getLogger("daytrader.bars")


class BarManager:
    """Manages 1-min bars for multiple tickers during a trading session."""

    def __init__(self):
        self._bars: dict[str, pd.DataFrame] = {}
        self._last_fetch: dict[str, datetime] = {}

    def get_bars(self, symbol: str, force_refresh: bool = False) -> pd.DataFrame:
        """Get current session 1-min bars for a symbol. Polls if stale."""
        now = datetime.now()
        last = self._last_fetch.get(symbol)

        # Refresh if never fetched or stale (>10 seconds old)
        if force_refresh or last is None or (now - last).total_seconds() > 10:
            bars = self._fetch_bars(symbol)
            if bars is not None and not bars.empty:
                self._bars[symbol] = bars
                self._last_fetch[symbol] = now

        return self._bars.get(symbol, pd.DataFrame())

    def _fetch_bars(self, symbol: str) -> Optional[pd.DataFrame]:
        """Fetch today's 1-min bars via yfinance."""
        try:
            import yfinance as yf
            ticker = yf.Ticker(symbol)
            df = ticker.history(period="1d", interval="1m")
            if df.empty:
                return None

            # Normalize timezone
            if df.index.tz is not None:
                df.index = df.index.tz_convert("America/New_York")
            else:
                df.index = df.index.tz_localize("America/New_York",
                                                 ambiguous="NaT",
                                                 nonexistent="NaT")

            # Filter to regular session only
            df = df.between_time("09:30", "15:59")
            return df

        except Exception as e:
            logger.warning("Failed to fetch bars for %s: %s", symbol, e)
            return None

    def get_bars_polygon(self, symbol: str, date_str: str, api_key: str) -> pd.DataFrame:
        """Fetch 1-min bars from Polygon for backtesting."""
        from daytrader.scanner.universe_scanner import polygon_get, _cache_path, _read_cache, _write_cache

        cache = _cache_path("minute", f"{symbol}_{date_str}")
        data = _read_cache(cache)
        if data is None:
            data = polygon_get(
                f"/v2/aggs/ticker/{symbol}/range/1/minute/{date_str}/{date_str}",
                {"adjusted": "true", "sort": "asc", "limit": "5000"},
                api_key,
            )
            if data:
                _write_cache(cache, data)

        results = data.get("results", [])
        if not results:
            return pd.DataFrame()

        df = pd.DataFrame(results)
        # Polygon fields: t=timestamp(ms), o=open, h=high, l=low, c=close, v=volume
        df["datetime"] = pd.to_datetime(df["t"], unit="ms", utc=True)
        df["datetime"] = df["datetime"].dt.tz_convert("America/New_York")
        df = df.set_index("datetime")
        df = df.rename(columns={"o": "Open", "h": "High", "l": "Low",
                                 "c": "Close", "v": "Volume"})
        df = df[["Open", "High", "Low", "Close", "Volume"]]

        # Filter to regular session
        df = df.between_time("09:30", "15:59")
        return df

    def clear(self):
        """Clear all cached bars (e.g., at end of day)."""
        self._bars.clear()
        self._last_fetch.clear()
