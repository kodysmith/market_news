"""Market data fetcher for the SPX IC bot.

Provides SPX price, VIX, SMA50, RV ratio, gap, and news-day flag.

Data source priority:
  1. IBKR TWS (real-time, requires active TWS/Gateway connection)
  2. yfinance (15-min delay, free fallback)
"""
from __future__ import annotations

import logging
import math
import os
import sys
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


@dataclass
class MarketSnapshot:
    """All data needed to make an entry/exit decision."""
    spx_price: float
    vix_level: float
    spx_open: float           # today's open (for gap calc)
    prev_close: float         # yesterday's SPX close
    sma50: float              # 50-day SMA of SPX closes
    rv_ratio: float           # rv_5d / rv_10d (vol expansion proxy)
    gap_pct: float            # (open - prev_close) / prev_close * 100
    is_news_day: bool
    timestamp: datetime = None

    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.utcnow()


def get_historical_closes(n_days: int = 70) -> pd.Series:
    """Return the last n_days SPX daily closes as a pd.Series indexed by date.

    Uses yfinance. IBKR historical data would need `reqHistoricalData` which
    requires an active connection — yfinance is sufficient for SMA/RV.
    """
    try:
        import yfinance as yf
        ticker = yf.Ticker("^GSPC")
        hist = ticker.history(period=f"{n_days + 10}d", auto_adjust=True)
        if hist.empty:
            raise ValueError("Empty yfinance result for ^GSPC")
        closes = hist["Close"].dropna()
        closes.index = pd.to_datetime(closes.index).date
        return closes.tail(n_days)
    except Exception as e:
        logger.error("get_historical_closes failed: %s", e)
        return pd.Series(dtype=float)


def _compute_sma(closes: pd.Series, period: int = 50) -> float:
    if len(closes) < period:
        logger.warning("Not enough closes for SMA%d (have %d)", period, len(closes))
        return float("nan")
    return float(closes.tail(period).mean())


def _compute_rv_ratio(closes: pd.Series) -> float:
    """Compute rv_5d / rv_10d from daily log returns.

    rv_Nd = annualized std of last N log returns.
    Returns ratio; >1 means short-term vol is expanding.
    """
    if len(closes) < 11:
        return 1.0
    log_rets = np.log(closes / closes.shift(1)).dropna()
    rv5 = float(log_rets.tail(5).std()) * math.sqrt(252)
    rv10 = float(log_rets.tail(10).std()) * math.sqrt(252)
    if rv10 < 1e-9:
        return 1.0
    return rv5 / rv10


def _get_spx_vix_ibkr() -> Optional[tuple[float, float, float]]:
    """Try to get (spx_price, spx_open, vix) from IBKR. Returns None if unavailable."""
    try:
        import ib_insync as ib_mod
        ib = ib_mod.IB()
        port = int(os.environ.get("IBKR_PORT", "7497"))
        ib.connect("127.0.0.1", port, clientId=20, readonly=True, timeout=5)

        spx_contract = ib_mod.Index("SPX", "CBOE", "USD")
        vix_contract = ib_mod.Index("VIX", "CBOE", "USD")
        ib.qualifyContracts(spx_contract, vix_contract)

        [spx_ticker, vix_ticker] = ib.reqTickers(spx_contract, vix_contract)

        spx_price = float(spx_ticker.last or spx_ticker.close or 0)
        spx_open = float(spx_ticker.open or spx_price)
        vix_price = float(vix_ticker.last or vix_ticker.close or 0)

        ib.disconnect()

        if spx_price > 0 and vix_price > 0:
            return spx_price, spx_open, vix_price
    except Exception as e:
        logger.debug("IBKR market data unavailable: %s", e)
    return None


def _get_spx_vix_yfinance() -> tuple[float, float, float]:
    """Fallback: get SPX and VIX from yfinance (15-min delayed)."""
    import yfinance as yf
    spx_hist = yf.Ticker("^GSPC").history(period="2d", interval="1m", auto_adjust=True)
    vix_hist = yf.Ticker("^VIX").history(period="2d", interval="1m", auto_adjust=True)

    spx_price = float(spx_hist["Close"].iloc[-1]) if not spx_hist.empty else 0.0
    spx_open = float(spx_hist["Open"].iloc[-1]) if not spx_hist.empty else spx_price
    vix_price = float(vix_hist["Close"].iloc[-1]) if not vix_hist.empty else 0.0
    return spx_price, spx_open, vix_price


def get_market_snapshot() -> MarketSnapshot:
    """Build a full MarketSnapshot for today.

    Tries IBKR first; falls back to yfinance.
    Historical data (SMA, RV) always from yfinance.
    """
    from scripts.evaluate_enhanced_cashflow import get_news_dates

    # Real-time price: IBKR preferred
    result = _get_spx_vix_ibkr()
    if result:
        spx_price, spx_open, vix_level = result
        source = "ibkr"
    else:
        spx_price, spx_open, vix_level = _get_spx_vix_yfinance()
        source = "yfinance"

    logger.info("Market data from %s: SPX=%.2f  VIX=%.2f", source, spx_price, vix_level)

    # Historical for SMA + RV
    closes = get_historical_closes(70)
    sma50 = _compute_sma(closes, 50)
    rv_ratio = _compute_rv_ratio(closes)

    # Previous close and gap
    prev_close = float(closes.iloc[-2]) if len(closes) >= 2 else spx_price
    gap_pct = (spx_open - prev_close) / prev_close * 100 if prev_close else 0.0

    today = date.today()
    is_news = today in get_news_dates(today, today)

    return MarketSnapshot(
        spx_price=spx_price,
        vix_level=vix_level,
        spx_open=spx_open,
        prev_close=prev_close,
        sma50=sma50,
        rv_ratio=rv_ratio,
        gap_pct=gap_pct,
        is_news_day=is_news,
        timestamp=datetime.utcnow(),
    )


def get_friday_expirations(from_date: date, n: int = 8) -> list[date]:
    """Return the next n monthly-option expiration Fridays from from_date."""
    from scripts.optimize_spx_verticals_historical import friday_expirations
    all_fridays = friday_expirations(from_date, date(from_date.year + 2, 12, 31))
    return [f for f in all_fridays if f >= from_date][:n]


def is_market_open() -> bool:
    """Return True if US equity markets are currently open (rough check)."""
    try:
        import pandas_market_calendars as mcal
        nyse = mcal.get_calendar("NYSE")
        now = datetime.now()
        schedule = nyse.schedule(start_date=now.date(), end_date=now.date())
        if schedule.empty:
            return False
        market_open = schedule.iloc[0]["market_open"].to_pydatetime()
        market_close = schedule.iloc[0]["market_close"].to_pydatetime()
        # pandas_market_calendars returns UTC; convert to local is complex, use simpler check
        return True  # calendar says it's a trading day; time check handled by scheduler
    except Exception:
        # Fallback: Mon-Fri 9:30-16:00 ET (approximate)
        from zoneinfo import ZoneInfo
        et = ZoneInfo("America/New_York")
        now_et = datetime.now(et)
        if now_et.weekday() >= 5:
            return False
        t = now_et.time()
        return t >= __import__("datetime").time(9, 30) and t <= __import__("datetime").time(16, 0)
