"""
Market Data Service: normalized premarket snapshot + prior close.
Adapters: IBKR (premarket best), Alpha Vantage, Yahoo/FMP fallbacks.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

import requests

# Optional: ib_insync for IBKR
try:
    import ib_insync as ib
    _IB_AVAILABLE = True
except ImportError:
    ib = None
    _IB_AVAILABLE = False

try:
    import yfinance as yf
    _YF_AVAILABLE = True
except ImportError:
    yf = None
    _YF_AVAILABLE = False


@dataclass
class PremarketSnapshot:
    """Normalized premarket + prior close for one symbol."""
    symbol: str
    pm_last: float
    prior_close: float
    pm_volume: Optional[float] = None  # None if source has no premarket volume
    pm_high: Optional[float] = None
    pm_low: Optional[float] = None
    source: str = "unknown"  # "ibkr" | "alphavantage" | "yfinance" | "fmp"
    prev_day_volume: Optional[float] = None  # for dollar vol / liquidity


def get_prior_close_fmp(api_key: Optional[str], symbol: str) -> Optional[float]:
    if not api_key:
        return None
    try:
        r = requests.get(
            f"https://financialmodelingprep.com/api/v3/quote/{symbol}",
            params={"apikey": api_key},
            timeout=10,
        )
        r.raise_for_status()
        data = r.json()
        if isinstance(data, list) and len(data) > 0:
            prev = data[0].get("previousClose")
            if prev is not None and float(prev) > 0:
                return float(prev)
    except Exception:
        pass
    return None


def get_prior_close_alpha_vantage(api_key: Optional[str], symbol: str) -> Optional[float]:
    if not api_key:
        return None
    try:
        r = requests.get(
            "https://www.alphavantage.co/query",
            params={
                "function": "GLOBAL_QUOTE",
                "symbol": symbol,
                "apikey": api_key,
            },
            timeout=10,
        )
        r.raise_for_status()
        data = r.json()
        quote = data.get("Global Quote", {})
        prev = quote.get("08. previous close")
        if prev is not None and float(prev) > 0:
            return float(prev)
    except Exception:
        pass
    return None


def get_prior_close_yfinance(symbol: str) -> Optional[float]:
    if not _YF_AVAILABLE:
        return None
    try:
        t = yf.Ticker(symbol)
        info = t.info
        prev = info.get("previousClose")
        if prev is not None and float(prev) > 0:
            return float(prev)
        hist = t.history(period="5d")
        if not hist.empty:
            return float(hist["Close"].iloc[-2])
    except Exception:
        pass
    return None


def get_prior_close(symbol: str, config: Dict[str, Any]) -> Optional[float]:
    """Prior session close. Tries FMP -> Alpha Vantage -> yfinance."""
    fmp_key = config.get("FMP_API_KEY") or config.get("fmp_api_key")
    av_key = config.get("ALPHAVANTAGE_API_KEY") or config.get("alphavantage_api_key")
    v = get_prior_close_fmp(fmp_key, symbol)
    if v is not None:
        return v
    v = get_prior_close_alpha_vantage(av_key, symbol)
    if v is not None:
        return v
    return get_prior_close_yfinance(symbol)


def get_premarket_ibkr(symbol: str, host: str = "127.0.0.1", port: int = 7497, client_id: int = 1) -> Optional[PremarketSnapshot]:
    """IBKR premarket snapshot. Returns None if unavailable or not connected."""
    if not _IB_AVAILABLE:
        return None
    try:
        conn = ib.IB()
        conn.connect(host, port, clientId=client_id, timeout=10, readonly=True)
        try:
            stock = ib.Stock(symbol, "SMART", "USD")
            # Request market data (may include premarket if subscription allows)
            ticker = conn.reqMktData(stock, "", False, False)
            conn.sleep(2)
            if ticker.last is not None and ticker.last > 0:
                prior = ticker.close if ticker.close and ticker.close > 0 else ticker.last
                vol = getattr(ticker, "volume", None) or getattr(ticker, "lastSize", None)
                pm_vol = float(vol) if vol is not None else None
                return PremarketSnapshot(
                    symbol=symbol,
                    pm_last=float(ticker.last),
                    prior_close=float(prior),
                    pm_volume=pm_vol,
                    pm_high=float(ticker.high) if getattr(ticker, "high", None) and ticker.high else None,
                    pm_low=float(ticker.low) if getattr(ticker, "low", None) and ticker.low else None,
                    source="ibkr",
                )
        finally:
            conn.disconnect()
    except Exception:
        pass
    return None


def get_premarket_from_quote(symbol: str, config: Dict[str, Any]) -> Optional[PremarketSnapshot]:
    """Use FMP/AV/yfinance quote as 'last' and prior_close. No real premarket volume."""
    prior = get_prior_close(symbol, config)
    if prior is None or prior <= 0:
        return None
    # Current "price" from quote (may be delayed or regular session)
    fmp_key = config.get("FMP_API_KEY") or config.get("fmp_api_key")
    av_key = config.get("ALPHAVANTAGE_API_KEY") or config.get("alphavantage_api_key")
    last = None
    source = "yfinance"
    if fmp_key:
        try:
            r = requests.get(
                f"https://financialmodelingprep.com/api/v3/quote/{symbol}",
                params={"apikey": fmp_key},
                timeout=10,
            )
            r.raise_for_status()
            data = r.json()
            if isinstance(data, list) and len(data) > 0:
                last = data[0].get("price")
                if last is not None and float(last) > 0:
                    last = float(last)
                    source = "fmp"
        except Exception:
            pass
    if last is None and av_key:
        try:
            r = requests.get(
                "https://www.alphavantage.co/query",
                params={"function": "GLOBAL_QUOTE", "symbol": symbol, "apikey": av_key},
                timeout=10,
            )
            r.raise_for_status()
            data = r.json()
            quote = data.get("Global Quote", {})
            p = quote.get("05. price")
            if p is not None and float(p) > 0:
                last = float(p)
                source = "alphavantage"
        except Exception:
            pass
    if last is None and _YF_AVAILABLE:
        try:
            t = yf.Ticker(symbol)
            info = t.info
            last = info.get("regularMarketPrice") or info.get("previousClose")
            if last is not None:
                last = float(last)
        except Exception:
            pass
    if last is None or last <= 0:
        return None
    return PremarketSnapshot(
        symbol=symbol,
        pm_last=last,
        prior_close=prior,
        pm_volume=None,
        pm_high=None,
        pm_low=None,
        source=source,
    )


def get_prev_day_volume(symbol: str, config: Dict[str, Any]) -> Optional[float]:
    """Previous session volume for dollar-vol baseline. FMP historical daily or yfinance."""
    fmp_key = config.get("FMP_API_KEY") or config.get("fmp_api_key")
    if fmp_key:
        try:
            r = requests.get(
                f"https://financialmodelingprep.com/api/v3/historical-price-full/{symbol}",
                params={"apikey": fmp_key, "from": "2020-01-01"},
                timeout=10,
            )
            r.raise_for_status()
            data = r.json()
            if isinstance(data, dict) and "historical" in data:
                hist = data["historical"]
                if isinstance(hist, list) and len(hist) >= 2:
                    v = hist[1].get("volume")
                    if v is not None and int(v) > 0:
                        return float(v)
        except Exception:
            pass
    if _YF_AVAILABLE:
        try:
            t = yf.Ticker(symbol)
            hist = t.history(period="5d")
            if hist is not None and len(hist) >= 2:
                return float(hist["Volume"].iloc[-2])
        except Exception:
            pass
    return None


def get_premarket_snapshot(symbol: str, config: Dict[str, Any]) -> Optional[PremarketSnapshot]:
    """IBKR first; on failure fallback to FMP/AV/yfinance (no premarket volume)."""
    use_ibkr = config.get("SCANNER_USE_IBKR", True)
    if use_ibkr and _IB_AVAILABLE:
        host = config.get("IBKR_HOST", "127.0.0.1")
        port = int(config.get("IBKR_PORT", "7497"))
        cid = int(config.get("IBKR_CLIENT_ID", "1"))
        snap = get_premarket_ibkr(symbol, host=host, port=port, client_id=cid)
        if snap is not None:
            return snap
    snap = get_premarket_from_quote(symbol, config)
    if snap is not None:
        vol = get_prev_day_volume(symbol, config)
        if vol is not None:
            snap.prev_day_volume = vol
    return snap
