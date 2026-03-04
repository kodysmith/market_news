#!/usr/bin/env python3
"""
Warrior-style Gap + RelVol + 1-min First Pullback backtest using Polygon.

Uses grouped daily bars for gap/relvol screening (1 API call per day), then 1-min bars
only for candidates. All Polygon responses cached under data/polygon_cache/.

Requires: POLYGON_API_KEY in env (or .env at repo root).

Usage:
  python scripts/backtest_warrior_gap_pullback.py --start 2024-01-01 --end 2024-12-31 [options]
  --symbols symbols.txt   OR  --symbols-csv symbols.csv  OR  --scan-all

Outputs: trades.csv, summary.txt (current dir unless --output-dir).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any, Optional

import pandas as pd
import requests

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.trading_calendar import get_trading_days_index, get_trading_days_in_range

# Timezone: match project convention (zoneinfo or pytz)
try:
    from zoneinfo import ZoneInfo
    NY = ZoneInfo("America/New_York")
except ImportError:
    try:
        import pytz
        NY = pytz.timezone("America/New_York")
    except ImportError:
        NY = None  # fallback: naive datetimes

POLYGON_BASE = "https://api.polygon.io"
CACHE_DIR = ROOT / "data" / "polygon_cache"


def _load_env() -> None:
    """Load .env from repo root if present."""
    env_file = ROOT / ".env"
    if env_file.is_file():
        with open(env_file) as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    k, _, v = line.partition("=")
                    k = k.strip()
                    v = v.strip().strip('"').strip("'")
                    if k and k not in os.environ:
                        os.environ[k] = v


def polygon_get(path: str, params: dict[str, Any], api_key: str, max_retries: int = 5) -> dict[str, Any]:
    """GET with retry on 429/5xx."""
    params = dict(params)
    params["apiKey"] = api_key
    url = f"{POLYGON_BASE}{path}"
    backoff = 1.0
    for attempt in range(max_retries):
        r = requests.get(url, params=params, timeout=60)
        if r.status_code == 200:
            return r.json()
        if r.status_code in (429, 500, 502, 503, 504):
            time.sleep(backoff)
            backoff *= 2
            continue
        raise RuntimeError(f"Polygon {r.status_code}: {r.text[:500]}")
    raise RuntimeError(f"Polygon failed after retries: {url}")


# ---------- Disk cache layer ----------

def _cache_path(kind: str, key: str) -> Path:
    """kind: 'grouped_daily', 'minute', 'ticker'. key: e.g. '2024-01-02' or 'AAPL_2024-01-02' or 'AAPL'."""
    safe = key.replace("/", "_")
    return CACHE_DIR / kind / f"{safe}.json"


def _read_cache(path: Path) -> Optional[dict[str, Any]]:
    if not path.is_file():
        return None
    try:
        with open(path) as f:
            return json.load(f)
    except Exception:
        return None


def _write_cache(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, separators=(",", ":"))


def get_grouped_daily_cached(date_str: str, api_key: str, sleep: float = 0.0) -> Optional[dict[str, Any]]:
    """Fetch grouped daily bars for all US stocks for date. Cached under data/polygon_cache/grouped_daily/."""
    path = _cache_path("grouped_daily", date_str)
    cached = _read_cache(path)
    if cached is not None:
        return cached
    # GET /v2/aggs/grouped/locale/us/market/stocks/{date}
    j = polygon_get(
        f"/v2/aggs/grouped/locale/us/market/stocks/{date_str}",
        {"adjusted": "true"},
        api_key,
    )
    _write_cache(path, j)
    if sleep:
        time.sleep(sleep)
    return j


def get_daily_bars_yfinance(date_str: str, symbols: list[str], cache_dir: Path) -> pd.DataFrame:
    """Fetch daily OHLCV for date_str for given symbols via yfinance. Cached to CSV per date."""
    import logging
    import yfinance as yf
    # Suppress yfinance "possibly delisted" / 404 stderr spam for missing or invalid tickers
    logging.getLogger("yfinance").setLevel(logging.WARNING)
    if getattr(yf, "config", None) and getattr(yf.config, "debug", None):
        yf.config.debug.logging = False
    subdir = cache_dir / "daily_yfinance"
    cache_file = subdir / f"{date_str}.csv"
    if cache_file.is_file():
        df = pd.read_csv(cache_file)
        df = df[df["symbol"].isin(symbols)] if symbols else df
        return df
    from datetime import datetime as dt
    target = dt.strptime(date_str, "%Y-%m-%d")
    start_dt = target - timedelta(days=5)
    end_dt = target + timedelta(days=2)
    start_str = start_dt.strftime("%Y-%m-%d")
    end_str = end_dt.strftime("%Y-%m-%d")
    rows = []
    for sym in symbols:
        try:
            t = yf.Ticker(sym)
            hist = t.history(start=start_str, end=end_str, auto_adjust=True)
            if hist is None or hist.empty:
                continue
            hist.index = pd.to_datetime(hist.index)
            if hist.index.tz is not None:
                hist = hist.tz_localize(None)
            target_date = target.date()
            mask = [d == target_date for d in hist.index.date]
            hist = hist.loc[mask]
            if hist.empty or len(hist) == 0:
                continue
            row = hist.iloc[0]
            rows.append({
                "symbol": sym,
                "o": float(row.get("Open", 0) or 0),
                "h": float(row.get("High", 0) or 0),
                "l": float(row.get("Low", 0) or 0),
                "c": float(row.get("Close", 0) or 0),
                "v": int(row.get("Volume", 0) or 0),
                "t": 0,
            })
        except Exception:
            continue
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    subdir.mkdir(parents=True, exist_ok=True)
    df.to_csv(cache_file, index=False)
    return df


def get_daily_bars_ibkr(
    date_str: str,
    symbols: list[str],
    cache_dir: Path,
    host: str = "127.0.0.1",
    port: int = 7497,
    client_id: int = 1,
) -> pd.DataFrame:
    """Fetch daily OHLCV for date_str for given symbols via IBKR reqHistoricalData. Cached to CSV per date."""
    subdir = cache_dir / "daily_ibkr"
    cache_file = subdir / f"{date_str}.csv"
    if cache_file.is_file():
        df = pd.read_csv(cache_file)
        df = df[df["symbol"].isin(symbols)] if symbols else df
        return df
    try:
        import ib_insync as ib
    except ImportError:
        return pd.DataFrame()
    # End of day in ET for that date (market close 16:00)
    if NY:
        from datetime import datetime as dt
        end_dt = dt.strptime(date_str + " 16:00:00", "%Y-%m-%d %H:%M:%S").replace(tzinfo=NY)
        end_str = end_dt.strftime("%Y%m%d %H:%M:%S")
    else:
        end_str = date_str.replace("-", "") + " 16:00:00"
    rows = []
    conn = ib.IB()
    try:
        conn.connect(host, port, clientId=client_id, timeout=30, readonly=True)
        for sym in symbols:
            try:
                contract = ib.Stock(sym, "SMART", "USD")
                bars = conn.reqHistoricalData(
                    contract,
                    endDateTime=end_str,
                    durationStr="1 D",
                    barSizeSetting="1 day",
                    whatToShow="TRADES",
                    useRTH=True,
                    formatDate=1,
                )
                if not bars:
                    continue
                b = bars[0]
                rows.append({
                    "symbol": sym,
                    "o": float(b.open),
                    "h": float(b.high),
                    "l": float(b.low),
                    "c": float(b.close),
                    "v": int(b.volume),
                    "t": 0,
                })
            except Exception:
                continue
    finally:
        try:
            conn.disconnect()
        except Exception:
            pass
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    subdir.mkdir(parents=True, exist_ok=True)
    df.to_csv(cache_file, index=False)
    return df


def get_daily_bars_for_date(
    date_str: str,
    symbols: Optional[list[str]],
    source: str,
    api_key: Optional[str],
    cache_dir: Path,
    sleep: float = 0.0,
    ibkr_host: str = "127.0.0.1",
    ibkr_port: int = 7497,
    ibkr_client_id: int = 1,
) -> pd.DataFrame:
    """Return daily bars for date_str. source in ('polygon','yfinance','ibkr'). Same DataFrame shape: symbol, o, h, l, c, v, t."""
    if source == "polygon":
        if not api_key:
            return pd.DataFrame()
        j = get_grouped_daily_cached(date_str, api_key, sleep)
        if not j:
            return pd.DataFrame()
        df = grouped_daily_to_dataframe(j)
        if symbols is not None:
            df = df[df["symbol"].isin(symbols)]
        return df
    if source == "yfinance":
        if not symbols:
            return pd.DataFrame()
        return get_daily_bars_yfinance(date_str, symbols, cache_dir)
    if source == "ibkr":
        if not symbols:
            return pd.DataFrame()
        return get_daily_bars_ibkr(date_str, symbols, cache_dir, ibkr_host, ibkr_port, ibkr_client_id)
    return pd.DataFrame()


def get_minute_bars_cached(symbol: str, date_str: str, api_key: str, sleep: float = 0.0) -> Optional[dict[str, Any]]:
    """Fetch 1-min bars for symbol on date. Cached under data/polygon_cache/minute/."""
    key = f"{symbol}_{date_str}"
    path = _cache_path("minute", key)
    cached = _read_cache(path)
    if cached is not None:
        return cached
    j = polygon_get(
        f"/v2/aggs/ticker/{symbol}/range/1/minute/{date_str}/{date_str}",
        {"adjusted": "true", "sort": "asc", "limit": 50000},
        api_key,
    )
    _write_cache(path, j)
    if sleep:
        time.sleep(sleep)
    return j


def get_ticker_details_cached(symbol: str, api_key: str, sleep: float = 0.0) -> Optional[dict[str, Any]]:
    """Fetch ticker details (e.g. shares outstanding). Cached under data/polygon_cache/ticker/."""
    path = _cache_path("ticker", symbol.upper())
    cached = _read_cache(path)
    if cached is not None:
        return cached
    j = polygon_get(f"/v3/reference/tickers/{symbol.upper()}", {}, api_key)
    _write_cache(path, j)
    if sleep:
        time.sleep(sleep)
    return j


def load_symbols(path_txt: Optional[str], path_csv: Optional[str]) -> list[str]:
    if path_txt:
        syms = []
        with open(path_txt) as f:
            for line in f:
                s = line.strip().upper()
                if s and not s.startswith("#"):
                    syms.append(s)
        return sorted(set(syms))
    if path_csv:
        df = pd.read_csv(path_csv)
        if "symbol" not in df.columns:
            raise ValueError("CSV must have a 'symbol' column")
        return sorted(set(df["symbol"].astype(str).str.upper().tolist()))
    raise ValueError("Provide --symbols (txt), --symbols-csv (csv), or use --scan-all")


def to_datestr(d: date) -> str:
    return d.strftime("%Y-%m-%d")


def ms_to_dt(ms: int):
    if NY:
        return datetime.fromtimestamp(ms / 1000.0, tz=NY)
    return datetime.fromtimestamp(ms / 1000.0)


# ---------- Gap scanner (grouped daily) ----------

def grouped_daily_to_dataframe(j: dict[str, Any]) -> pd.DataFrame:
    """Parse Polygon grouped daily response into DataFrame with columns symbol, o, h, l, c, v, t."""
    results = j.get("results") or []
    if not results:
        return pd.DataFrame()
    rows = []
    for r in results:
        ticker = r.get("T") or r.get("ticker")
        if not ticker:
            continue
        rows.append({
            "symbol": str(ticker),
            "o": float(r.get("o", 0)),
            "h": float(r.get("h", 0)),
            "l": float(r.get("l", 0)),
            "c": float(r.get("c", 0)),
            "v": int(r.get("v", 0)),
            "t": int(r.get("t", 0)),
        })
    return pd.DataFrame(rows)


def get_prev_trading_day(trading_dates: list[date], d: date) -> Optional[date]:
    """Previous trading day before d (d must be in trading_dates)."""
    try:
        i = trading_dates.index(d)
        if i > 0:
            return trading_dates[i - 1]
        return None
    except ValueError:
        return None


def gap_candidates_for_day(
    date_str: str,
    prev_date_str: str,
    daily_today: pd.DataFrame,
    daily_prev: pd.DataFrame,
    min_gap_pct: float,
    symbol_universe: Optional[list[str]],
) -> pd.DataFrame:
    """
    Merge today vs prev close, compute gap%, filter by min_gap_pct.
    If symbol_universe is set, only keep those symbols; else keep all.
    Returns DataFrame with columns symbol, day_open, prev_close, gap_pct, day_volume.
    """
    if daily_prev.empty or daily_today.empty:
        return pd.DataFrame()
    prev = daily_prev[["symbol", "c"]].rename(columns={"c": "prev_close"})
    today = daily_today[["symbol", "o", "v"]].rename(columns={"o": "day_open", "v": "day_volume"})
    m = today.merge(prev, on="symbol", how="inner")
    m = m[m["prev_close"] > 0]
    m["gap_pct"] = (m["day_open"] - m["prev_close"]) / m["prev_close"]
    m = m[m["gap_pct"] >= min_gap_pct]
    if symbol_universe is not None:
        m = m[m["symbol"].isin(symbol_universe)]
    m["day"] = date_str
    return m


# ---------- Relative volume (daily bars) ----------

def relvol_filter(
    candidates: pd.DataFrame,
    all_trading_dates: list[date],
    current_date: date,
    daily_bars_by_date: dict[str, pd.DataFrame],
    min_relvol: float,
    lookback_days: int,
) -> pd.DataFrame:
    """
    For each candidate, compute lookback_days average daily volume,
    then rel_vol = today_volume / avg_volume. Keep rows where rel_vol >= min_relvol.
    all_trading_dates must include enough history before current_date for lookback.
    """
    if candidates.empty:
        return candidates.copy()
    try:
        idx = all_trading_dates.index(current_date)
    except ValueError:
        return pd.DataFrame()
    if idx < lookback_days:
        return pd.DataFrame()
    date_str = to_datestr(current_date)
    today_vol = candidates.set_index("symbol")["day_volume"]
    relvols = []
    for sym in candidates["symbol"].unique():
        hist_vols = []
        for i in range(1, lookback_days + 1):
            past = all_trading_dates[idx - i]
            past_str = to_datestr(past)
            df = daily_bars_by_date.get(past_str)
            if df is None or df.empty:
                continue
            row = df[df["symbol"] == sym]
            if not row.empty and float(row["v"].iloc[0]) > 0:
                hist_vols.append(float(row["v"].iloc[0]))
        if len(hist_vols) < max(5, lookback_days // 2):
            continue
        avg = sum(hist_vols) / len(hist_vols)
        vol_today = today_vol.get(sym)
        if vol_today is None or avg <= 0 or vol_today <= 0:
            continue
        rv = vol_today / avg
        if rv >= min_relvol:
            relvols.append({"symbol": sym, "rel_vol": rv})
    if not relvols:
        return pd.DataFrame()
    rv_df = pd.DataFrame(relvols)
    return candidates.merge(rv_df, on="symbol", how="inner")


# ---------- Scanner candidates: build candidate rows from (date, symbol) list ----------

def load_scanner_candidates(path: Path) -> dict[str, set[str]]:
    """Load CSV with columns date, symbol. Returns dict date_str -> set(symbols)."""
    df = pd.read_csv(path)
    if "date" not in df.columns or "symbol" not in df.columns:
        raise ValueError("Scanner candidates CSV must have 'date' and 'symbol' columns")
    df["date"] = df["date"].astype(str)
    df["symbol"] = df["symbol"].astype(str).str.upper()
    out: dict[str, set[str]] = {}
    for date_str, sym in zip(df["date"], df["symbol"]):
        out.setdefault(date_str, set()).add(sym)
    return out


def candidates_from_scanner_list(
    date_str: str,
    symbols: set[str],
    daily_today: pd.DataFrame,
    daily_prev: pd.DataFrame,
    all_trading_dates: list[date],
    current_date: date,
    daily_bars_by_date: dict[str, pd.DataFrame],
    lookback_days: int,
) -> pd.DataFrame:
    """
    Build candidates DataFrame for only the given symbols (from historical scanner).
    Each row has symbol, gap_pct, rel_vol, day_open, prev_close, day_volume.
    """
    if not symbols or daily_today.empty or daily_prev.empty:
        return pd.DataFrame()
    prev = daily_prev[["symbol", "c"]].rename(columns={"c": "prev_close"})
    today = daily_today[["symbol", "o", "v"]].rename(columns={"o": "day_open", "v": "day_volume"})
    m = today.merge(prev, on="symbol", how="inner")
    m = m[m["symbol"].isin(symbols)]
    m = m[m["prev_close"] > 0]
    m["gap_pct"] = (m["day_open"] - m["prev_close"]) / m["prev_close"]
    m["day"] = date_str
    try:
        idx = all_trading_dates.index(current_date)
    except ValueError:
        return pd.DataFrame()
    if idx < lookback_days:
        return m.assign(rel_vol=float("nan"))
    relvols = []
    for sym in m["symbol"].unique():
        hist_vols = []
        for i in range(1, lookback_days + 1):
            past = all_trading_dates[idx - i]
            past_str = to_datestr(past)
            df = daily_bars_by_date.get(past_str)
            if df is None or df.empty:
                continue
            row = df[df["symbol"] == sym]
            if not row.empty and float(row["v"].iloc[0]) > 0:
                hist_vols.append(float(row["v"].iloc[0]))
        if len(hist_vols) < max(5, lookback_days // 2):
            relvols.append({"symbol": sym, "rel_vol": float("nan")})
            continue
        avg = sum(hist_vols) / len(hist_vols)
        vol_today = m.loc[m["symbol"] == sym, "day_volume"].iloc[0]
        if avg <= 0 or vol_today <= 0:
            relvols.append({"symbol": sym, "rel_vol": float("nan")})
            continue
        relvols.append({"symbol": sym, "rel_vol": vol_today / avg})
    rv_df = pd.DataFrame(relvols)
    return m.merge(rv_df, on="symbol", how="left")


# ---------- Shares outstanding filter (optional) ----------

def get_shares_outstanding(symbol: str, api_key: str, sleep: float = 0.0) -> Optional[float]:
    j = get_ticker_details_cached(symbol, api_key, sleep)
    if not j:
        return None
    res = j.get("results") or {}
    for k in ("share_class_shares_outstanding", "weighted_shares_outstanding", "shares_outstanding"):
        v = res.get(k)
        if isinstance(v, (int, float)) and v > 0:
            return float(v)
    return None


# ---------- Minute bars and regular session ----------

def minute_response_to_dataframe(j: dict[str, Any]) -> pd.DataFrame:
    rows = j.get("results") or []
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    df["dt"] = df["t"].apply(ms_to_dt)
    df = df.sort_values("dt").reset_index(drop=True)
    return df


def filter_regular_session(df: pd.DataFrame) -> pd.DataFrame:
    """Keep only 9:30 - 16:00 NY regular session bars."""
    if df.empty:
        return df
    d0 = df["dt"].iloc[0]
    if hasattr(d0, "date"):
        d = d0.date()
    else:
        d = d0
    year, month, day = d.year, d.month, d.day
    if NY:
        start = datetime(year, month, day, 9, 30, tzinfo=NY)
        end = datetime(year, month, day, 16, 0, tzinfo=NY)
    else:
        start = datetime(year, month, day, 9, 30)
        end = datetime(year, month, day, 16, 0)
    mask = (df["dt"] >= start) & (df["dt"] < end)
    return df.loc[mask].copy()


# ---------- First pullback detection (opening range breakout, then first pullback) ----------

@dataclass
class TradeResult:
    entry_dt: Any
    entry_price: float
    stop_price: float
    exit_dt: Any
    exit_price: float
    outcome: str  # TP, STOP, EOD, TIMESTOP
    r: float


def find_first_pullback_trade(
    df_reg: pd.DataFrame,
    target_r: float,
    time_stop_minutes: int = 120,
) -> Optional[TradeResult]:
    """
    1) Opening range = first 15 minutes (9:30-9:44).
    2) First bar after 9:45 that breaks above OR high = push confirmation.
    3) First red candle after that = pullback. Stop = low of that candle.
    4) Entry = open of the bar after the first green close after the pullback.
    5) Target = entry + target_r * R. Time stop after time_stop_minutes.
    """
    if df_reg.empty or len(df_reg) < 20:
        return None
    d0 = df_reg["dt"].iloc[0]
    if hasattr(d0, "tzinfo") and d0.tzinfo and NY:
        session_start = d0.replace(hour=9, minute=30, second=0, microsecond=0)
    else:
        session_start = datetime(d0.year, d0.month, d0.day, 9, 30, tzinfo=NY)
    or_end = session_start + timedelta(minutes=15)
    time_stop_dt = session_start + timedelta(minutes=time_stop_minutes)
    session_end = session_start + timedelta(hours=6, minutes=30)

    or_bars = df_reg[(df_reg["dt"] >= session_start) & (df_reg["dt"] < or_end)]
    if or_bars.empty:
        return None
    or_high = float(or_bars["h"].max())
    or_low = float(or_bars["l"].min())

    # Bars after opening range (minute 16+)
    after_or = df_reg[df_reg["dt"] >= or_end].reset_index(drop=True)
    if after_or.empty:
        return None

    # Find first bar that breaks above OR high
    breakout_idx = None
    for i in range(len(after_or)):
        if float(after_or.iloc[i]["h"]) >= or_high:
            breakout_idx = i
            break
    if breakout_idx is None:
        return None

    # First red candle after breakout
    pullback_idx = None
    for j in range(breakout_idx, len(after_or)):
        row = after_or.iloc[j]
        if float(row["c"]) < float(row["o"]):
            pullback_idx = j
            break
    if pullback_idx is None or pullback_idx + 1 >= len(after_or):
        return None

    stop_price = float(after_or.iloc[pullback_idx]["l"])

    # First green close after pullback
    green_idx = None
    for k in range(pullback_idx + 1, len(after_or)):
        row = after_or.iloc[k]
        if float(row["c"]) > float(row["o"]):
            green_idx = k
            break
    if green_idx is None or green_idx + 1 >= len(after_or):
        return None

    entry_row = after_or.iloc[green_idx + 1]
    entry_dt = entry_row["dt"]
    entry_price = float(entry_row["o"])
    r = entry_price - stop_price
    if r <= 0:
        return None
    target_price = entry_price + target_r * r

    # Walk bars for exit
    for idx in range(green_idx + 1, len(after_or)):
        row = after_or.iloc[idx]
        dt = row["dt"]
        if dt >= session_end:
            return TradeResult(
                entry_dt=entry_dt,
                entry_price=entry_price,
                stop_price=stop_price,
                exit_dt=dt,
                exit_price=float(row["c"]),
                outcome="EOD",
                r=r,
            )
        if dt >= time_stop_dt:
            return TradeResult(
                entry_dt=entry_dt,
                entry_price=entry_price,
                stop_price=stop_price,
                exit_dt=dt,
                exit_price=float(row["c"]),
                outcome="TIMESTOP",
                r=r,
            )
        low, high = float(row["l"]), float(row["h"])
        if low <= stop_price:
            return TradeResult(
                entry_dt=entry_dt,
                entry_price=entry_price,
                stop_price=stop_price,
                exit_dt=dt,
                exit_price=stop_price,
                outcome="STOP",
                r=r,
            )
        if high >= target_price:
            return TradeResult(
                entry_dt=entry_dt,
                entry_price=entry_price,
                stop_price=stop_price,
                exit_dt=dt,
                exit_price=target_price,
                outcome="TP",
                r=r,
            )

    last = after_or.iloc[-1]
    return TradeResult(
        entry_dt=entry_dt,
        entry_price=entry_price,
        stop_price=stop_price,
        exit_dt=last["dt"],
        exit_price=float(last["c"]),
        outcome="EOD",
        r=r,
    )


# ---------- Trade record and PnL with slippage ----------

@dataclass
class Trade:
    symbol: str
    day: str
    gap_pct: float
    rel_vol: float
    entry_time: str
    entry: float
    stop: float
    r: float
    target_r: float
    exit_time: str
    exit: float
    outcome: str
    pnl_r: float


def apply_slippage(entry: float, exit_px: float, outcome: str, slippage_per_share: float) -> float:
    """Deduct slippage on entry and exit (each side pays slippage)."""
    gross = exit_px - entry
    deduct = 2 * slippage_per_share
    return gross - deduct


# ---------- Main ----------

def main() -> int:
    _load_env()
    ap = argparse.ArgumentParser(description="Warrior Gap + RelVol + First Pullback backtest (Polygon, cached)")
    ap.add_argument("--symbols", type=str, default=None, help="Txt file, one symbol per line")
    ap.add_argument("--symbols-csv", type=str, default=None, help="CSV with column 'symbol'")
    ap.add_argument("--scan-all", action="store_true", help="Scan all tickers from grouped daily (no symbol list)")
    ap.add_argument("--scanner-candidates", type=Path, default=None, help="CSV from run_historical_scanner (date, symbol) to backtest only those")
    ap.add_argument("--start", type=str, required=True, help="YYYY-MM-DD")
    ap.add_argument("--end", type=str, required=True, help="YYYY-MM-DD")
    ap.add_argument("--gap", type=float, default=0.10, help="Min gap fraction (0.10 = 10%%)")
    ap.add_argument("--relvol", type=float, default=5.0, help="Min relative volume (daily)")
    ap.add_argument("--relvol-lookback-days", type=int, default=20, help="Days for avg volume")
    ap.add_argument("--min-float", type=float, default=0, help="Min shares outstanding (0 = skip filter)")
    ap.add_argument("--target-r", type=float, default=2.0, help="Take profit in R multiples")
    ap.add_argument("--time-stop-min", type=int, default=120, help="Exit after N min if no stop/TP")
    ap.add_argument("--slippage", type=float, default=0.02, help="Slippage per share (dollars)")
    ap.add_argument("--output-dir", type=Path, default=Path("."), help="Directory for trades.csv and summary.txt")
    ap.add_argument("--sleep", type=float, default=0.0, help="Seconds between API calls (rate limit)")
    args = ap.parse_args()

    api_key = os.environ.get("POLYGON_API_KEY")
    if not api_key:
        print("Set POLYGON_API_KEY in env or .env", file=sys.stderr)
        return 1

    symbol_universe = None
    scanner_candidates: Optional[dict[str, set[str]]] = None
    if args.scanner_candidates is not None:
        cand_path = args.scanner_candidates
        if not cand_path.is_file() and not cand_path.is_absolute():
            cand_path = ROOT / cand_path
        if not cand_path.is_file():
            print(f"Scanner candidates file not found: {args.scanner_candidates}", file=sys.stderr)
            return 1
        scanner_candidates = load_scanner_candidates(cand_path)
        print(f"Loaded {sum(len(s) for s in scanner_candidates.values())} (date, symbol) pairs from {cand_path}")
    elif not args.scan_all:
        if not args.symbols and not args.symbols_csv:
            print("Provide --symbols, --symbols-csv, --scan-all, or --scanner-candidates", file=sys.stderr)
            return 1
        symbol_universe = load_symbols(args.symbols, args.symbols_csv)

    start_d = datetime.strptime(args.start, "%Y-%m-%d").date()
    end_d = datetime.strptime(args.end, "%Y-%m-%d").date()
    trading_index = get_trading_days_index(start_d, end_d)
    trading_dates = get_trading_days_in_range(trading_index, start_d, end_d)
    if not trading_dates:
        print("No trading days in range", file=sys.stderr)
        return 1

    # Preload grouped daily for full range (so we have prev day + lookback for relvol)
    extend_start = trading_dates[0]
    for _ in range(args.relvol_lookback_days + 5):
        prev = extend_start - timedelta(days=1)
        while prev.weekday() >= 5:
            prev -= timedelta(days=1)
        extend_start = prev
    extended_index = get_trading_days_index(extend_start, end_d)
    extended_dates = get_trading_days_in_range(extended_index, extend_start, end_d)

    daily_bars_by_date: dict[str, pd.DataFrame] = {}
    for d in extended_dates:
        ds = to_datestr(d)
        j = get_grouped_daily_cached(ds, api_key, args.sleep)
        if j:
            daily_bars_by_date[ds] = grouped_daily_to_dataframe(j)

    trades: list[Trade] = []
    shares_cache: dict[str, Optional[float]] = {}
    total_days = len(trading_dates)

    for day_idx, d in enumerate(trading_dates):
        date_str = to_datestr(d)
        prev_d = get_prev_trading_day(trading_dates, d)
        if prev_d is None:
            continue
        prev_str = to_datestr(prev_d)
        daily_today = daily_bars_by_date.get(date_str)
        daily_prev = daily_bars_by_date.get(prev_str)
        if daily_today is None or daily_prev is None or daily_today.empty or daily_prev.empty:
            continue

        if scanner_candidates is not None:
            symbols_this_day = scanner_candidates.get(date_str, set())
            if not symbols_this_day:
                continue
            candidates = candidates_from_scanner_list(
                date_str,
                symbols_this_day,
                daily_today,
                daily_prev,
                extended_dates,
                d,
                daily_bars_by_date,
                args.relvol_lookback_days,
            )
        else:
            candidates = gap_candidates_for_day(
                date_str, prev_str, daily_today, daily_prev, args.gap, symbol_universe
            )
            if candidates.empty:
                if (day_idx + 1) % 20 == 0 or day_idx == 0:
                    print(f"  [{day_idx+1}/{total_days}] {date_str}: 0 gap candidates")
                continue
            candidates = relvol_filter(
                candidates,
                extended_dates,
                d,
                daily_bars_by_date,
                args.relvol,
                args.relvol_lookback_days,
            )
        if candidates.empty:
            continue

        for _, row in candidates.iterrows():
            sym = row["symbol"]
            if args.min_float > 0:
                if sym not in shares_cache:
                    shares_cache[sym] = get_shares_outstanding(sym, api_key, args.sleep)
                if (shares_cache[sym] or 0) < args.min_float:
                    continue

            j = get_minute_bars_cached(sym, date_str, api_key, args.sleep)
            if not j:
                continue
            m_df = minute_response_to_dataframe(j)
            m_reg = filter_regular_session(m_df)
            if m_reg.empty or len(m_reg) < 20:
                continue

            res = find_first_pullback_trade(
                m_reg,
                target_r=args.target_r,
                time_stop_minutes=args.time_stop_min,
            )
            if not res:
                continue

            raw_pnl = apply_slippage(res.entry_price, res.exit_price, res.outcome, args.slippage)
            pnl_r = raw_pnl / res.r if res.r > 0 else float("nan")
            entry_time_str = res.entry_dt.strftime("%Y-%m-%d %H:%M") if hasattr(res.entry_dt, "strftime") else str(res.entry_dt)
            exit_time_str = res.exit_dt.strftime("%Y-%m-%d %H:%M") if hasattr(res.exit_dt, "strftime") else str(res.exit_dt)

            trades.append(
                Trade(
                    symbol=sym,
                    day=date_str,
                    gap_pct=row["gap_pct"],
                    rel_vol=row["rel_vol"],
                    entry_time=entry_time_str,
                    entry=res.entry_price,
                    stop=res.stop_price,
                    r=res.r,
                    target_r=args.target_r,
                    exit_time=exit_time_str,
                    exit=res.exit_price,
                    outcome=res.outcome,
                    pnl_r=pnl_r,
                )
            )

        print(f"  [{day_idx+1}/{total_days}] {date_str}: {len(candidates)} candidates -> {len([t for t in trades if t.day == date_str])} trades")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    out_csv = args.output_dir / "trades.csv"
    out_txt = args.output_dir / "summary.txt"

    df = pd.DataFrame([t.__dict__ for t in trades])
    if not df.empty:
        df.to_csv(out_csv, index=False)
    else:
        df.to_csv(out_csv, index=False)

    with open(out_txt, "w") as f:
        f.write(f"Trades: {len(df)}\n")
        if len(df) > 0:
            f.write(f"Win% (pnl_r>0): {(df['pnl_r'] > 0).mean() * 100:.2f}%\n")
            f.write(f"Avg R: {df['pnl_r'].mean():.3f}\n")
            f.write(f"Median R: {df['pnl_r'].median():.3f}\n")
            f.write(f"TP count: {(df['outcome'] == 'TP').sum()}\n")
            f.write(f"STOP count: {(df['outcome'] == 'STOP').sum()}\n")
            f.write(f"EOD/TIMESTOP count: {(df['outcome'].isin(['EOD', 'TIMESTOP'])).sum()}\n")

    print(f"Done. Wrote {out_csv} and {out_txt}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
