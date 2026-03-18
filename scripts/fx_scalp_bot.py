#!/usr/bin/env python3
"""
Forex / Crypto Scalp Bot — IBKR Paper Trading

Strategy: Order Book Imbalance + Cross-Market Correlation + Technicals
  Edge 1 (D): IBKR Level 2 order book — bid/ask size imbalance predicts
              short-term direction. Heavy bids = buy pressure → long.
  Edge 2 (C): ES futures (S&P) lead BTC during risk-on/off moves.
              EUR/USD inversely correlates (dollar strength → BTC weakness).
              When cross-market signals align with technicals → high confidence.
  Base:       Bollinger Bands, RSI, EMA for trend/mean-reversion context.

Instruments: BTC/USD (24/7), EUR/USD (24/5)

Usage:
    python scripts/fx_scalp_bot.py --symbol BTC --ibkr paper
    python scripts/fx_scalp_bot.py --symbol EURUSD --ibkr paper
    python scripts/fx_scalp_bot.py --symbol BTC --ibkr dry-run  # no orders
"""
from __future__ import annotations

import argparse
import csv
import logging
import math
import signal
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

ET = ZoneInfo("America/New_York")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("fx_scalp")

# ─── Instrument Profiles ─────────────────────────────────────────────────────

INSTRUMENTS = {
    "BTC": {
        "name": "BTC/USD",
        "type": "crypto",          # ib_insync.Crypto
        "ib_symbol": "BTC",
        "exchange": "PAXOS",
        "currency": "USD",
        "min_qty": 0.001,          # ~$70
        "qty_decimals": 4,
        "tick_size": 0.25,
        "spread_cost_pct": 0.001,  # ~0.001% spread
        "long_only": True,         # IBKR PAXOS crypto can't be shorted
        # Strategy params
        "position_size_usd": 500,  # $500 per trade
        "max_daily_loss": 200,     # $200
        "max_trades_per_day": 15,
        "sl_atr_mult": 1.5,       # tighter for crypto
        "tp_atr_mult": 2.5,       # 1.67:1 reward/risk
        "trail_activate_atr": 1.5,
        "trail_distance_atr": 0.75,
        "max_hold_minutes": 60,
        "cooldown_sec": 120,       # 2 min between trades
        "bb_period": 20,
        "bb_std": 2.0,
        "rsi_period": 14,
        "ema_fast": 9,
        "ema_slow": 21,
        "atr_period": 14,
    },
    "EURUSD": {
        "name": "EUR/USD",
        "type": "forex",           # ib_insync.Forex
        "ib_symbol": "EURUSD",
        "exchange": "IDEALPRO",
        "currency": "USD",
        "min_qty": 2000,           # 2K units (min for paper test)
        "qty_decimals": 0,
        "tick_size": 0.00001,
        "spread_cost_pct": 0.002,
        "long_only": False,            # forex supports both directions
        "position_size_usd": 2000,   # 2K EUR for paper testing
        "max_daily_loss": 50,
        "max_trades_per_day": 10,
        "sl_atr_mult": 2.0,
        "tp_atr_mult": 3.0,
        "trail_activate_atr": 2.0,
        "trail_distance_atr": 1.0,
        "max_hold_minutes": 60,
        "cooldown_sec": 180,
        "bb_period": 20,
        "bb_std": 2.0,
        "rsi_period": 14,
        "ema_fast": 9,
        "ema_slow": 21,
        "atr_period": 14,
    },
}


# ─── Indicators ──────────────────────────────────────────────────────────────

def compute_ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span, adjust=False).mean()


def compute_rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gains = delta.where(delta > 0, 0.0)
    losses = -delta.where(delta < 0, 0.0)
    avg_gain = gains.rolling(period).mean()
    avg_loss = losses.rolling(period).mean()
    rs = avg_gain / avg_loss.replace(0, 1e-9)
    return 100.0 - (100.0 / (1.0 + rs))


def compute_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    high = df["high"]
    low = df["low"]
    prev_close = df["close"].shift(1)
    tr = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low - prev_close).abs(),
    ], axis=1).max(axis=1)
    return tr.rolling(period).mean()


def compute_bollinger(series: pd.Series, period: int = 20, std: float = 2.0):
    sma = series.rolling(period).mean()
    std_dev = series.rolling(period).std()
    upper = sma + std * std_dev
    lower = sma - std * std_dev
    width = (upper - lower) / sma  # normalized width
    return sma, upper, lower, width


# ─── Signal Generation ───────────────────────────────────────────────────────

@dataclass
class FXSignal:
    timestamp: str
    symbol: str
    price: float
    ema_fast: float
    ema_slow: float
    rsi: float
    atr: float
    bb_upper: float
    bb_lower: float
    bb_mid: float
    bb_width: float
    # Order book imbalance (Edge D)
    dom_imbalance: float       # -1 to +1 (positive = bid heavy = buy pressure)
    dom_bid_size: float        # total bid BTC
    dom_ask_size: float        # total ask BTC
    # Cross-market (Edge C)
    es_momentum: float         # ES 5-bar return (positive = risk-on)
    eur_momentum: float        # EUR/USD 5-bar return (positive = dollar weak = BTC bullish)
    cross_signal: str          # "RISK_ON" / "RISK_OFF" / "NEUTRAL"
    signal: str                # LONG / SHORT / FLAT
    reason: str
    confidence: float


def compute_cross_market_signal(es_bars: pd.DataFrame, eur_bars: pd.DataFrame) -> tuple[float, float, str]:
    """Compute cross-market momentum from ES futures and EUR/USD.

    Returns (es_momentum, eur_momentum, regime).
    - ES rising + EUR rising = risk-on (bullish BTC)
    - ES falling + EUR falling = risk-off (bearish BTC)
    """
    es_mom = 0.0
    eur_mom = 0.0

    if not es_bars.empty and len(es_bars) >= 5:
        closes = es_bars["close"]
        es_mom = (float(closes.iloc[-1]) / float(closes.iloc[-5]) - 1) * 100  # % return

    if not eur_bars.empty and len(eur_bars) >= 5:
        closes = eur_bars["close"]
        eur_mom = (float(closes.iloc[-1]) / float(closes.iloc[-5]) - 1) * 100

    # Classify regime
    if es_mom > 0.02 and eur_mom > 0.002:
        regime = "RISK_ON"
    elif es_mom < -0.02 and eur_mom < -0.002:
        regime = "RISK_OFF"
    elif es_mom > 0.05:
        regime = "RISK_ON"   # strong ES move overrides
    elif es_mom < -0.05:
        regime = "RISK_OFF"
    else:
        regime = "NEUTRAL"

    return es_mom, eur_mom, regime


def compute_dom_imbalance(dom_bids: list, dom_asks: list) -> tuple[float, float, float]:
    """Compute order book imbalance from Level 2 depth.

    Returns (imbalance, total_bid_size, total_ask_size).
    imbalance = (bid_size - ask_size) / (bid_size + ask_size), range [-1, +1].
    Positive = more buy pressure, negative = more sell pressure.
    """
    bid_size = sum(d.size for d in dom_bids) if dom_bids else 0.0
    ask_size = sum(d.size for d in dom_asks) if dom_asks else 0.0
    total = bid_size + ask_size
    if total < 1e-9:
        return 0.0, 0.0, 0.0
    imbalance = (bid_size - ask_size) / total
    return imbalance, bid_size, ask_size


def generate_fx_signal(
    df: pd.DataFrame,
    inst: dict,
    dom_imbalance: float = 0.0,
    dom_bid_size: float = 0.0,
    dom_ask_size: float = 0.0,
    es_momentum: float = 0.0,
    eur_momentum: float = 0.0,
    cross_signal: str = "NEUTRAL",
) -> Optional[FXSignal]:
    """Generate a trading signal combining technicals, order book, and cross-market."""
    if len(df) < 30:
        return None

    closes = df["close"]
    price = float(closes.iloc[-1])
    ts = str(df.index[-1])

    # Indicators
    ema_f = compute_ema(closes, inst["ema_fast"])
    ema_s = compute_ema(closes, inst["ema_slow"])
    rsi_series = compute_rsi(closes, inst["rsi_period"])
    atr_series = compute_atr(df, inst["atr_period"])
    bb_mid, bb_upper, bb_lower, bb_width = compute_bollinger(
        closes, inst["bb_period"], inst["bb_std"]
    )

    ema_fast_val = float(ema_f.iloc[-1])
    ema_slow_val = float(ema_s.iloc[-1])
    rsi = float(rsi_series.iloc[-1]) if not np.isnan(rsi_series.iloc[-1]) else 50.0
    atr = float(atr_series.iloc[-1]) if not np.isnan(atr_series.iloc[-1]) else price * 0.001
    bb_up = float(bb_upper.iloc[-1])
    bb_lo = float(bb_lower.iloc[-1])
    bb_m = float(bb_mid.iloc[-1])
    bb_w = float(bb_width.iloc[-1]) if not np.isnan(bb_width.iloc[-1]) else 0.01

    if atr <= 0 or np.isnan(atr):
        atr = price * 0.001

    sig = FXSignal(
        timestamp=ts, symbol=inst["name"], price=price,
        ema_fast=ema_fast_val, ema_slow=ema_slow_val,
        rsi=rsi, atr=atr,
        bb_upper=bb_up, bb_lower=bb_lo, bb_mid=bb_m, bb_width=bb_w,
        dom_imbalance=dom_imbalance, dom_bid_size=dom_bid_size, dom_ask_size=dom_ask_size,
        es_momentum=es_momentum, eur_momentum=eur_momentum, cross_signal=cross_signal,
        signal="FLAT", reason="", confidence=0.0,
    )

    trend_up = ema_fast_val > ema_slow_val
    trend_down = ema_fast_val < ema_slow_val
    prev_ema_f = float(ema_f.iloc[-2]) if len(ema_f) > 1 else ema_fast_val
    prev_ema_s = float(ema_s.iloc[-2]) if len(ema_s) > 1 else ema_slow_val

    # ── Scoring system: combine all edges into a directional score ──
    # Each edge contributes to a weighted score [-1, +1]
    # Positive = bullish, negative = bearish

    score = 0.0
    reasons = []

    # (D) Order book imbalance — strongest short-term predictor
    # Weight: 0.30
    if abs(dom_imbalance) > 0.15:
        ob_score = dom_imbalance * 0.30  # scales with imbalance magnitude
        score += ob_score
        if dom_imbalance > 0.15:
            reasons.append(f"DOM:bid-heavy({dom_imbalance:+.2f})")
        elif dom_imbalance < -0.15:
            reasons.append(f"DOM:ask-heavy({dom_imbalance:+.2f})")

    # (C) Cross-market momentum — leading indicator
    # Weight: 0.25
    if cross_signal == "RISK_ON":
        score += 0.25
        reasons.append(f"CROSS:risk-on(ES{es_momentum:+.2f}%)")
    elif cross_signal == "RISK_OFF":
        score -= 0.25
        reasons.append(f"CROSS:risk-off(ES{es_momentum:+.2f}%)")

    # Technical: BB position (mean-reversion at extremes)
    # Weight: 0.20
    bb_pct = (price - bb_lo) / (bb_up - bb_lo) if (bb_up - bb_lo) > 0 else 0.5
    if bb_pct < 0.1 and rsi < 35:  # near lower BB + oversold
        score += 0.20
        reasons.append(f"BB:oversold(RSI={rsi:.0f})")
    elif bb_pct > 0.9 and rsi > 65:  # near upper BB + overbought
        score -= 0.20
        reasons.append(f"BB:overbought(RSI={rsi:.0f})")
    elif bb_pct < 0.2:
        score += 0.10
    elif bb_pct > 0.8:
        score -= 0.10

    # Technical: EMA trend alignment
    # Weight: 0.15
    if trend_up:
        score += 0.15
        if prev_ema_f <= prev_ema_s:  # fresh cross
            score += 0.05
            reasons.append("EMA:bull-cross")
    elif trend_down:
        score -= 0.15
        if prev_ema_f >= prev_ema_s:
            score -= 0.05
            reasons.append("EMA:bear-cross")

    # Technical: RSI momentum
    # Weight: 0.10
    if rsi > 60:
        score += 0.10
    elif rsi < 40:
        score -= 0.10

    # ── Convert score to signal ──
    # Higher threshold = fewer but higher-quality trades
    if score >= 0.35:
        sig.signal = "LONG"
        sig.confidence = min(0.90, 0.50 + score)
        sig.reason = " + ".join(reasons) if reasons else "composite bullish"
    elif score <= -0.35:
        sig.signal = "SHORT"
        sig.confidence = min(0.90, 0.50 + abs(score))
        sig.reason = " + ".join(reasons) if reasons else "composite bearish"

    # ── High-conviction overrides ──
    # DOM + cross-market + BB extreme = highest confidence
    if dom_imbalance > 0.30 and cross_signal == "RISK_ON" and bb_pct < 0.3 and rsi < 40:
        sig.signal = "LONG"
        sig.confidence = 0.90
        sig.reason = "ALPHA: strong DOM + risk-on + BB-low + oversold"
    elif dom_imbalance < -0.30 and cross_signal == "RISK_OFF" and bb_pct > 0.7 and rsi > 60:
        sig.signal = "SHORT"
        sig.confidence = 0.90
        sig.reason = "ALPHA: strong DOM + risk-off + BB-high + overbought"

    # BB squeeze breakout with DOM confirmation
    width_history = bb_width.dropna().tail(100)
    if len(width_history) > 20 and sig.signal == "FLAT":
        squeeze_threshold = float(width_history.quantile(0.20))
        prev_width = float(bb_width.iloc[-2]) if len(bb_width) > 1 and not np.isnan(bb_width.iloc[-2]) else bb_w
        was_squeezed = prev_width < squeeze_threshold
        now_expanding = bb_w > squeeze_threshold

        if was_squeezed and now_expanding:
            if price > bb_up and dom_imbalance > 0.10 and trend_up:
                sig.signal = "LONG"
                sig.reason = "SQUEEZE: breakout UP + DOM confirms"
                sig.confidence = 0.75
            elif price < bb_lo and dom_imbalance < -0.10 and trend_down:
                sig.signal = "SHORT"
                sig.reason = "SQUEEZE: breakout DOWN + DOM confirms"
                sig.confidence = 0.75

    return sig


# ─── IBKR Connection ─────────────────────────────────────────────────────────

class FXIBKRClient:
    """Manages IBKR connection, market data, order book, and orders."""

    def __init__(self, mode: str, instrument: dict):
        self.mode = mode
        self.inst = instrument
        self._ib = None
        self._contract = None
        self._es_contract = None   # ES futures for cross-market
        self._eur_contract = None  # EUR/USD for cross-market
        self._dom_subscribed = False

    def connect(self):
        if self.mode == "dry-run":
            log.info("[DRY-RUN] Skipping IBKR connection")
            return
        import ib_insync as ib_mod
        port = int(__import__("os").environ.get("IBKR_PAPER_PORT", "7497"))
        # Each symbol gets its own client ID to allow concurrent bots
        default_ids = {"BTC": "16", "EURUSD": "19"}
        default_id = default_ids.get(self.inst["ib_symbol"], "16")
        client_id = int(__import__("os").environ.get("IBKR_FX_CLIENT_ID", default_id))
        ib = ib_mod.IB()
        ib.connect("127.0.0.1", port, clientId=client_id, timeout=30)
        self._ib = ib
        # Build primary contract
        if self.inst["type"] == "crypto":
            self._contract = ib_mod.Crypto(
                self.inst["ib_symbol"], self.inst["exchange"], self.inst["currency"],
            )
        else:
            self._contract = ib_mod.Forex(self.inst["ib_symbol"])
        ib.qualifyContracts(self._contract)

        # Cross-market contracts (Edge C)
        try:
            # ES front-month futures
            self._es_contract = ib_mod.Future("ES", "20260618", "CME")
            ib.qualifyContracts(self._es_contract)
            log.info("  ES futures contract qualified")
        except Exception as e:
            log.warning("  ES contract setup failed: %s", e)
            self._es_contract = None

        try:
            self._eur_contract = ib_mod.Forex("EURUSD")
            ib.qualifyContracts(self._eur_contract)
            log.info("  EUR/USD contract qualified")
        except Exception as e:
            log.warning("  EUR/USD contract setup failed: %s", e)
            self._eur_contract = None

        # Subscribe to order book (Edge D)
        try:
            ib.reqMktDepth(self._contract, numRows=10)
            self._dom_subscribed = True
            log.info("  Order book (DOM) subscribed — 10 levels")
        except Exception as e:
            log.warning("  DOM subscription failed: %s", e)

        log.info("Connected to IBKR (%s) port=%d — %s", self.mode, port, self.inst["name"])

    def is_connected(self) -> bool:
        """Check if IBKR connection is alive."""
        if self.mode == "dry-run":
            return True
        return self._ib is not None and self._ib.isConnected()

    def reconnect(self):
        """Force disconnect and reconnect to IBKR."""
        log.warning("Forcing IBKR reconnect...")
        if self._ib:
            try:
                self._ib.disconnect()
            except Exception:
                pass
            self._ib = None
        self._contract = None
        self._es_contract = None
        self._eur_contract = None
        self._dom_subscribed = False
        time.sleep(2)
        try:
            self.connect()
            log.info("IBKR reconnect successful")
        except Exception as e:
            log.error("IBKR reconnect failed: %s", e)

    def _ensure_connected(self):
        """Reconnect if connection is stale."""
        if not self.is_connected():
            if self._ib is not None:
                self.reconnect()
            else:
                self.connect()

    def get_position(self) -> tuple[float, float]:
        """Check IBKR for an existing position in this instrument.

        Returns (quantity, avg_cost).  Positive qty = long, zero = flat.
        """
        if self.mode == "dry-run" or not self._ib:
            return 0.0, 0.0
        try:
            positions = self._ib.positions()
            for p in positions:
                if p.contract.symbol == self.inst["ib_symbol"]:
                    qty = float(p.position)
                    avg = float(p.avgCost)
                    return qty, avg
        except Exception as e:
            log.warning("get_position failed: %s", e)
        return 0.0, 0.0

    def cleanup_residual_cash(self):
        """Convert residual non-USD cash balances back to USD.

        IMPORTANT: Only call when the bot has NO open position.
        In IBKR forex, positions and cash are the same thing — a short
        EUR/USD position IS negative EUR cash. So we can only safely
        clean cash when the bot is flat.

        Skips the bot's own trading currency (e.g., EUR for EURUSD bot)
        since any residual there is from the bot's own trades and will
        be managed by the bot itself.
        """
        if self.mode == "dry-run" or not self._ib:
            return
        import ib_insync as ib_mod

        try:
            acct_vals = self._ib.accountValues()
            to_clean = {}
            for av in acct_vals:
                if av.tag == "CashBalance" and av.currency not in ("USD", "BASE", ""):
                    val = float(av.value)
                    if abs(val) > 1:
                        to_clean[av.currency] = val

            if not to_clean:
                log.info("No residual non-USD cash balances to clean up")
                return

            for currency, amount in to_clean.items():
                log.info("Residual %s cash: %.2f — converting to USD", currency, amount)
                try:
                    if currency == "JPY":
                        contract = ib_mod.Forex("USDJPY")
                        self._ib.qualifyContracts(contract)
                        [ticker] = self._ib.reqTickers(contract)
                        self._ib.sleep(0.3)
                        [ticker] = self._ib.reqTickers(contract)
                        rate = float(ticker.midpoint()) if ticker.midpoint() else 159.0
                        usd_qty = round(abs(amount) / rate, 0)
                        if usd_qty < 1:
                            log.info("  JPY residual too small (< $1), skipping")
                            continue
                        action = "SELL" if amount < 0 else "BUY"
                        order = ib_mod.MarketOrder(action=action,
                                                   totalQuantity=usd_qty, tif="IOC")
                    else:
                        pair = f"{currency}USD"
                        contract = ib_mod.Forex(pair)
                        self._ib.qualifyContracts(contract)
                        qty = abs(amount)
                        action = "BUY" if amount < 0 else "SELL"
                        order = ib_mod.MarketOrder(action=action,
                                                   totalQuantity=qty, tif="IOC")

                    trade = self._ib.placeOrder(contract, order)
                    for _ in range(20):
                        self._ib.sleep(0.5)
                        if trade.isDone():
                            break
                    if trade.orderStatus.status == "Filled":
                        fp = float(trade.orderStatus.avgFillPrice)
                        log.info("  Residual %s cleaned: %s %.0f @ %.5f",
                                 currency, action, abs(amount), fp)
                    else:
                        log.warning("  Residual %s cleanup not filled: %s",
                                    currency, trade.orderStatus.status)
                        try:
                            self._ib.cancelOrder(order)
                        except Exception:
                            pass
                except Exception as e:
                    log.warning("  Failed to clean %s residual: %s", currency, e)

        except Exception as e:
            log.warning("cleanup_residual_cash failed: %s", e)

    def disconnect(self):
        if self._ib:
            if self._dom_subscribed:
                try:
                    self._ib.cancelMktDepth(self._contract)
                except Exception:
                    pass
            try:
                self._ib.disconnect()
            except Exception:
                pass
            self._ib = None

    def get_dom(self) -> tuple[list, list]:
        """Get current order book depth (bids, asks)."""
        if self.mode == "dry-run" or not self._ib:
            return [], []
        try:
            ticker = self._ib.ticker(self._contract)
            if ticker:
                return (ticker.domBids or []), (ticker.domAsks or [])
        except Exception as e:
            log.debug("get_dom failed: %s", e)
        return [], []

    def get_es_bars(self, duration: str = "14400 S", bar_size: str = "5 mins") -> pd.DataFrame:
        """Get ES futures historical bars for cross-market correlation."""
        if not self._es_contract or not self._ib:
            return pd.DataFrame()
        try:
            bars = self._ib.reqHistoricalData(
                self._es_contract, endDateTime="", durationStr=duration,
                barSizeSetting=bar_size, whatToShow="TRADES",
                useRTH=False, formatDate=1,
            )
            if not bars:
                return pd.DataFrame()
            data = [{"date": b.date, "open": b.open, "high": b.high,
                     "low": b.low, "close": b.close} for b in bars]
            return pd.DataFrame(data).set_index("date")
        except Exception as e:
            log.debug("get_es_bars failed: %s", e)
            return pd.DataFrame()

    def get_eur_bars(self, duration: str = "14400 S", bar_size: str = "5 mins") -> pd.DataFrame:
        """Get EUR/USD bars for dollar strength proxy."""
        if not self._eur_contract or not self._ib:
            return pd.DataFrame()
        try:
            bars = self._ib.reqHistoricalData(
                self._eur_contract, endDateTime="", durationStr=duration,
                barSizeSetting=bar_size, whatToShow="MIDPOINT",
                useRTH=False, formatDate=1,
            )
            if not bars:
                return pd.DataFrame()
            data = [{"date": b.date, "open": b.open, "high": b.high,
                     "low": b.low, "close": b.close} for b in bars]
            return pd.DataFrame(data).set_index("date")
        except Exception as e:
            log.debug("get_eur_bars failed: %s", e)
            return pd.DataFrame()

    def get_bars(self, duration: str = "1 D", bar_size: str = "1 min") -> pd.DataFrame:
        """Fetch historical bars."""
        if self.mode == "dry-run":
            return pd.DataFrame()
        self._ensure_connected()
        if not self._ib:
            return pd.DataFrame()
        try:
            bars = self._ib.reqHistoricalData(
                self._contract, endDateTime="", durationStr=duration,
                barSizeSetting=bar_size, whatToShow="MIDPOINT",
                useRTH=False, formatDate=1,
            )
            if not bars:
                return pd.DataFrame()
            data = [{
                "date": b.date, "open": b.open, "high": b.high,
                "low": b.low, "close": b.close, "volume": getattr(b, "volume", 0),
            } for b in bars]
            df = pd.DataFrame(data).set_index("date")
            return df
        except Exception as e:
            log.warning("get_bars failed: %s", e)
            return pd.DataFrame()

    def get_price(self) -> Optional[float]:
        if self.mode == "dry-run":
            return None
        self._ensure_connected()
        if not self._ib:
            return None
        try:
            [ticker] = self._ib.reqTickers(self._contract)
            self._ib.sleep(0.3)
            [ticker] = self._ib.reqTickers(self._contract)
            bid = float(ticker.bid) if ticker.bid and ticker.bid > 0 else 0
            ask = float(ticker.ask) if ticker.ask and ticker.ask > 0 else 0
            if bid > 0 and ask > 0:
                return (bid + ask) / 2
            return float(ticker.last) if ticker.last and ticker.last > 0 else None
        except Exception as e:
            log.warning("get_price failed: %s", e)
            return None

    def get_bid_ask(self) -> tuple[float, float]:
        if self.mode == "dry-run" or not self._ib:
            return (0, 0)
        try:
            [ticker] = self._ib.reqTickers(self._contract)
            self._ib.sleep(0.3)
            [ticker] = self._ib.reqTickers(self._contract)
            return (float(ticker.bid or 0), float(ticker.ask or 0))
        except Exception:
            return (0, 0)

    def _order_tif(self, order_type: str = "limit") -> str:
        """Return appropriate TIF for this instrument.
        Crypto (PAXOS) only supports GTC for limit orders and GTC for market.
        Forex (IDEALPRO) supports IOC/DAY normally.
        """
        if self.inst["type"] == "crypto":
            return "GTC"
        return "IOC" if order_type == "limit" else "DAY"

    def buy(self, qty: float, limit: Optional[float] = None):
        import ib_insync as ib_mod
        qty = round(qty, self.inst["qty_decimals"])
        log.info(">>> BUY %s %s @ limit $%s", qty, self.inst["name"],
                 f"{limit:.2f}" if limit else "MKT")
        if self.mode == "dry-run":
            log.info("[DRY-RUN] Simulated buy fill")
            return {"success": True, "fill_price": limit or 0, "qty": qty}
        try:
            if limit:
                order = ib_mod.LimitOrder(action="BUY", totalQuantity=qty, lmtPrice=limit,
                                          tif=self._order_tif("limit"))
            else:
                order = ib_mod.MarketOrder(action="BUY", totalQuantity=qty,
                                           tif=self._order_tif("market"))
            trade = self._ib.placeOrder(self._contract, order)
            for _ in range(30):
                self._ib.sleep(0.5)
                if trade.isDone():
                    break
            if trade.orderStatus.status == "Filled":
                fp = float(trade.orderStatus.avgFillPrice)
                log.info("  BUY FILLED @ %s", fp)
                return {"success": True, "fill_price": fp, "qty": qty}
            else:
                log.warning("  BUY not filled: %s", trade.orderStatus.status)
                try:
                    self._ib.cancelOrder(order)
                except Exception:
                    pass
                return {"success": False, "fill_price": 0, "qty": 0}
        except Exception as e:
            log.error("Buy failed: %s", e)
            return {"success": False, "fill_price": 0, "qty": 0}

    def sell(self, qty: float, limit: Optional[float] = None):
        import ib_insync as ib_mod
        qty = round(qty, self.inst["qty_decimals"])
        log.info("<<< SELL %s %s @ limit $%s", qty, self.inst["name"],
                 f"{limit:.2f}" if limit else "MKT")
        if self.mode == "dry-run":
            log.info("[DRY-RUN] Simulated sell fill")
            return {"success": True, "fill_price": limit or 0, "qty": qty}
        try:
            if limit:
                order = ib_mod.LimitOrder(action="SELL", totalQuantity=qty, lmtPrice=limit,
                                          tif=self._order_tif("limit"))
            else:
                order = ib_mod.MarketOrder(action="SELL", totalQuantity=qty,
                                           tif=self._order_tif("market"))
            trade = self._ib.placeOrder(self._contract, order)
            for _ in range(30):
                self._ib.sleep(0.5)
                if trade.isDone():
                    break
            if trade.orderStatus.status == "Filled":
                fp = float(trade.orderStatus.avgFillPrice)
                log.info("  SELL FILLED @ %s", fp)
                return {"success": True, "fill_price": fp, "qty": qty}
            else:
                log.warning("  SELL limit not filled, trying MKT...")
                try:
                    self._ib.cancelOrder(order)
                    self._ib.sleep(1)
                except Exception:
                    pass
                mkt_order = ib_mod.MarketOrder(action="SELL", totalQuantity=qty,
                                               tif=self._order_tif("market"))
                trade2 = self._ib.placeOrder(self._contract, mkt_order)
                for _ in range(20):
                    self._ib.sleep(0.5)
                    if trade2.isDone():
                        break
                if trade2.orderStatus.status == "Filled":
                    fp = float(trade2.orderStatus.avgFillPrice)
                    log.info("  SELL FILLED (MKT) @ %s", fp)
                    return {"success": True, "fill_price": fp, "qty": qty}
                return {"success": False, "fill_price": 0, "qty": 0}
        except Exception as e:
            log.error("Sell failed: %s", e)
            return {"success": False, "fill_price": 0, "qty": 0}


# ─── Trade Manager ───────────────────────────────────────────────────────────

@dataclass
class FXTrade:
    entry_time: str
    entry_price: float
    direction: str
    qty: float
    reason: str
    exit_time: str = ""
    exit_price: float = 0.0
    exit_reason: str = ""
    pnl_dollar: float = 0.0
    hold_sec: int = 0


class FXTradeManager:
    """Manages trades, risk, and logging."""

    def __init__(self, inst: dict, client: FXIBKRClient, log_dir: Path):
        self.inst = inst
        self.client = client
        self.log_dir = log_dir
        self.log_dir.mkdir(parents=True, exist_ok=True)

        self.open_trade: Optional[FXTrade] = None
        self._peak_pnl = 0.0
        self._last_exit_time = 0.0
        self._today_trades = 0
        self._today_date = ""
        self._daily_pnl = 0.0

        self.stats = {"total": 0, "wins": 0, "losses": 0, "total_pnl": 0.0}

        today_str = datetime.now(ET).strftime("%Y%m%d")
        sym = inst["ib_symbol"].lower()
        self.trade_log = log_dir / f"fx_trades_{sym}_{today_str}.csv"
        self.signal_log = log_dir / f"fx_signals_{sym}_{today_str}.csv"
        self._init_logs()

    def detect_open_position(self):
        """Check IBKR for an existing position and adopt it as open_trade.

        This lets the bot manage exits for positions from previous runs.
        """
        qty, avg_cost = self.client.get_position()
        if qty <= 0:
            if qty < 0:
                log.warning("  Found SHORT position (%.4f) — can't manage short crypto, ignoring", qty)
            return

        # We have a long position — adopt it
        self.open_trade = FXTrade(
            entry_time=datetime.now(ET).strftime("%Y-%m-%d %H:%M:%S"),
            entry_price=avg_cost,
            direction="LONG",
            qty=abs(qty),
            reason="ADOPTED from previous session",
        )
        self._peak_pnl = 0.0

        price = self.client.get_price() or avg_cost
        pnl_per_unit = price - avg_cost
        pnl_dollar = pnl_per_unit * abs(qty)

        log.info("  >>> ADOPTED existing position: LONG %.4f %s @ avg $%s",
                 abs(qty), self.inst["name"],
                 f"{avg_cost:,.2f}" if avg_cost > 100 else f"{avg_cost:.5f}")
        log.info("      Current price: $%s  |  Unrealized PnL: $%s",
                 f"{price:,.2f}" if price > 100 else f"{price:.5f}", f"{pnl_dollar:+,.2f}")

    def _init_logs(self):
        if not self.trade_log.exists():
            with open(self.trade_log, "w", newline="") as f:
                csv.writer(f).writerow([
                    "entry_time", "exit_time", "symbol", "direction",
                    "qty", "entry_price", "exit_price",
                    "pnl_dollar", "exit_reason", "hold_sec", "reason",
                ])
        if not self.signal_log.exists():
            with open(self.signal_log, "w", newline="") as f:
                csv.writer(f).writerow([
                    "timestamp", "symbol", "price", "ema_fast", "ema_slow",
                    "rsi", "atr", "bb_upper", "bb_lower", "bb_width",
                    "dom_imbalance", "dom_bid_size", "dom_ask_size",
                    "es_momentum", "eur_momentum", "cross_signal",
                    "signal", "reason", "confidence",
                ])

    def log_signal(self, sig: FXSignal):
        with open(self.signal_log, "a", newline="") as f:
            csv.writer(f).writerow([
                sig.timestamp, sig.symbol, f"{sig.price:.6f}",
                f"{sig.ema_fast:.6f}", f"{sig.ema_slow:.6f}",
                f"{sig.rsi:.1f}", f"{sig.atr:.6f}",
                f"{sig.bb_upper:.6f}", f"{sig.bb_lower:.6f}", f"{sig.bb_width:.6f}",
                f"{sig.dom_imbalance:.3f}", f"{sig.dom_bid_size:.4f}", f"{sig.dom_ask_size:.4f}",
                f"{sig.es_momentum:.4f}", f"{sig.eur_momentum:.4f}", sig.cross_signal,
                sig.signal, sig.reason, f"{sig.confidence:.2f}",
            ])

    def _log_trade(self, t: FXTrade):
        with open(self.trade_log, "a", newline="") as f:
            csv.writer(f).writerow([
                t.entry_time, t.exit_time, self.inst["name"], t.direction,
                t.qty, t.entry_price, t.exit_price,
                f"{t.pnl_dollar:.2f}", t.exit_reason, t.hold_sec, t.reason,
            ])

    def process(self, sig: FXSignal, current_price: float):
        self.log_signal(sig)

        today = sig.timestamp[:10]
        if today != self._today_date:
            self._today_date = today
            self._today_trades = 0
            self._daily_pnl = 0.0

        if self.open_trade:
            self._check_exit(sig, current_price)
        elif sig.signal in ("LONG", "SHORT") and sig.confidence >= 0.70:
            self._try_entry(sig, current_price)

    def _try_entry(self, sig: FXSignal, price: float):
        # Long-only instruments (e.g. IBKR crypto can't be shorted)
        if self.inst.get("long_only") and sig.signal == "SHORT":
            log.debug("  Skipping SHORT entry — %s is long-only", self.inst["name"])
            return

        # Guards
        if self._today_trades >= self.inst["max_trades_per_day"]:
            return
        if self._daily_pnl <= -self.inst["max_daily_loss"]:
            log.info("  Daily loss limit hit ($%.0f), no more entries", self.inst["max_daily_loss"])
            return
        cooldown = self.inst["cooldown_sec"]
        if time.time() - self._last_exit_time < cooldown:
            return

        # Position sizing
        pos_usd = self.inst["position_size_usd"]
        qty = pos_usd / price
        qty = max(qty, self.inst["min_qty"])
        qty = round(qty, self.inst["qty_decimals"])

        # Place order — long-only instruments always BUY to open
        bid, ask = self.client.get_bid_ask()
        if sig.signal == "LONG":
            limit = ask if ask > 0 else price
        else:
            limit = bid if bid > 0 else price

        result = self.client.buy(qty, limit) if sig.signal == "LONG" else self.client.sell(qty, limit)

        if not result["success"]:
            log.warning("  Entry order failed")
            return

        fill = result["fill_price"] or price
        self.open_trade = FXTrade(
            entry_time=sig.timestamp,
            entry_price=fill,
            direction=sig.signal,
            qty=qty,
            reason=sig.reason,
        )
        self._peak_pnl = 0.0
        self._today_trades += 1

        atr = sig.atr
        sl = atr * self.inst["sl_atr_mult"]
        tp = atr * self.inst["tp_atr_mult"]
        log.info(
            "  >>> ENTRY %s %.4f %s @ %s | ATR=%s SL=%s TP=%s | %s",
            sig.signal, qty, self.inst["name"], fill, f"{atr:.2f}",
            f"{sl:.2f}", f"{tp:.2f}", sig.reason,
        )

    def _check_exit(self, sig: FXSignal, price: float):
        t = self.open_trade
        is_long = t.direction == "LONG"
        pnl_per_unit = (price - t.entry_price) if is_long else (t.entry_price - price)
        pnl_dollar = pnl_per_unit * t.qty
        self._peak_pnl = max(self._peak_pnl, pnl_dollar)

        atr = sig.atr if sig.atr > 0 else t.entry_price * 0.001
        sl_pts = atr * self.inst["sl_atr_mult"]
        tp_pts = atr * self.inst["tp_atr_mult"]
        trail_act = atr * self.inst["trail_activate_atr"] * t.qty
        trail_dist = atr * self.inst["trail_distance_atr"] * t.qty

        hold_sec = int(time.time() - _parse_ts(t.entry_time))

        reason = ""
        if pnl_per_unit <= -sl_pts:
            reason = "STOP_LOSS"
        elif pnl_per_unit >= tp_pts:
            reason = "TAKE_PROFIT"
        elif self._peak_pnl > trail_act and pnl_dollar < self._peak_pnl - trail_dist:
            reason = "TRAIL_STOP"
        elif hold_sec >= self.inst["max_hold_minutes"] * 60:
            reason = "TIME_EXIT"
        elif (is_long and sig.signal == "SHORT") or (not is_long and sig.signal == "LONG"):
            reason = "SIGNAL_REVERSAL"

        if not reason:
            return

        # Place exit order
        bid, ask = self.client.get_bid_ask()
        if is_long:
            limit = bid if bid > 0 else price
            result = self.client.sell(t.qty, limit)
        else:
            limit = ask if ask > 0 else price
            result = self.client.buy(t.qty, limit)

        exit_price = result["fill_price"] if result["success"] else price
        pnl_per_unit = (exit_price - t.entry_price) if is_long else (t.entry_price - exit_price)
        pnl_dollar = pnl_per_unit * t.qty

        t.exit_time = sig.timestamp
        t.exit_price = exit_price
        t.exit_reason = reason
        t.pnl_dollar = round(pnl_dollar, 2)
        t.hold_sec = hold_sec

        self._log_trade(t)
        self.stats["total"] += 1
        self.stats["total_pnl"] += pnl_dollar
        if pnl_dollar > 0:
            self.stats["wins"] += 1
        else:
            self.stats["losses"] += 1

        self._daily_pnl += pnl_dollar
        self._last_exit_time = time.time()

        wr = self.stats["wins"] / self.stats["total"] * 100 if self.stats["total"] else 0
        log.info(
            "  <<< EXIT %s @ %s | PnL: $%+.2f | %s | "
            "Session: %d trades, %.0f%% WR, $%+.2f",
            t.direction, exit_price, pnl_dollar, reason,
            self.stats["total"], wr, self.stats["total_pnl"],
        )
        self.open_trade = None

    def force_close(self, price: float):
        """Force close any open trade."""
        if not self.open_trade:
            return
        t = self.open_trade
        is_long = t.direction == "LONG"
        bid, ask = self.client.get_bid_ask()
        if is_long:
            result = self.client.sell(t.qty, bid if bid > 0 else None)
        else:
            result = self.client.buy(t.qty, ask if ask > 0 else None)
        exit_price = result["fill_price"] if result["success"] else price
        pnl = ((exit_price - t.entry_price) if is_long else (t.entry_price - exit_price)) * t.qty
        t.exit_time = datetime.now(ET).strftime("%Y-%m-%d %H:%M:%S")
        t.exit_price = exit_price
        t.exit_reason = "SHUTDOWN"
        t.pnl_dollar = round(pnl, 2)
        self._log_trade(t)
        self.stats["total"] += 1
        self.stats["total_pnl"] += pnl
        if pnl > 0:
            self.stats["wins"] += 1
        else:
            self.stats["losses"] += 1
        log.info("  <<< FORCE CLOSE %s @ %s | PnL: $%+.2f", t.direction, exit_price, pnl)
        self.open_trade = None


def _parse_ts(ts: str) -> float:
    """Parse timestamp to epoch seconds."""
    try:
        # Handle timezone-aware strings from IBKR
        if "+" in ts or "-0" in ts:
            dt = datetime.fromisoformat(ts)
        else:
            dt = datetime.strptime(ts[:19], "%Y-%m-%d %H:%M:%S")
        return dt.timestamp()
    except Exception:
        return time.time()


# ─── Main Loop ───────────────────────────────────────────────────────────────

def run(symbol: str, ibkr_mode: str, interval: int = 60):
    if symbol not in INSTRUMENTS:
        log.error("Unknown symbol '%s'. Available: %s", symbol, list(INSTRUMENTS.keys()))
        sys.exit(1)

    inst = INSTRUMENTS[symbol]
    log_dir = ROOT / "logs" / "fx_scalp"

    client = FXIBKRClient(mode=ibkr_mode, instrument=inst)
    client.connect()

    mgr = FXTradeManager(inst, client, log_dir)

    log.info("=" * 60)
    log.info("%s Scalp Bot — %s mode", inst["name"], ibkr_mode.upper())
    log.info("  Position size: $%s  |  Max daily loss: $%s",
             f"{inst['position_size_usd']:,}", inst["max_daily_loss"])
    log.info("  SL: %.1fx ATR  |  TP: %.1fx ATR  |  Trail: %.1fx/%.1fx ATR",
             inst["sl_atr_mult"], inst["tp_atr_mult"],
             inst["trail_activate_atr"], inst["trail_distance_atr"])
    log.info("  Trade log: %s", mgr.trade_log)

    # Check for existing positions from previous runs
    mgr.detect_open_position()
    if mgr.open_trade:
        log.info("  Bot will manage exit for adopted position")
    else:
        log.info("  No existing position — waiting for entry signals")
    log.info("=" * 60)

    running = [True]
    def _stop(signum, frame):
        log.info("Shutting down...")
        running[0] = False
    signal.signal(signal.SIGINT, _stop)
    signal.signal(signal.SIGTERM, _stop)

    # Cross-market data cache (refresh every 5 minutes, not every tick)
    _cross_cache = {"es_bars": pd.DataFrame(), "eur_bars": pd.DataFrame(), "last_refresh": 0.0}
    CROSS_REFRESH_SEC = 300  # 5 min

    while running[0]:
        try:
            # Fetch BTC bars
            df = client.get_bars(duration="1 D", bar_size="1 min")
            if df.empty:
                log.debug("No bars, waiting...")
                time.sleep(interval)
                continue

            # Get live price
            price = client.get_price()
            if not price:
                price = float(df["close"].iloc[-1])

            # (D) Order book imbalance — real-time from DOM subscription
            dom_bids, dom_asks = client.get_dom()
            dom_imb, dom_bid_sz, dom_ask_sz = compute_dom_imbalance(dom_bids, dom_asks)

            # (C) Cross-market data — cached, refresh every 5 min
            now = time.time()
            if now - _cross_cache["last_refresh"] > CROSS_REFRESH_SEC:
                _cross_cache["es_bars"] = client.get_es_bars()
                _cross_cache["eur_bars"] = client.get_eur_bars()
                _cross_cache["last_refresh"] = now
                if not _cross_cache["es_bars"].empty:
                    log.debug("  Refreshed cross-market: ES=%d bars, EUR=%d bars",
                              len(_cross_cache["es_bars"]), len(_cross_cache["eur_bars"]))

            es_mom, eur_mom, cross_sig = compute_cross_market_signal(
                _cross_cache["es_bars"], _cross_cache["eur_bars"],
            )

            # Generate signal with all edges
            sig = generate_fx_signal(
                df, inst,
                dom_imbalance=dom_imb, dom_bid_size=dom_bid_sz, dom_ask_size=dom_ask_sz,
                es_momentum=es_mom, eur_momentum=eur_mom, cross_signal=cross_sig,
            )
            if not sig:
                time.sleep(interval)
                continue

            # Update with live price
            sig.price = price
            sig.timestamp = datetime.now(ET).strftime("%Y-%m-%d %H:%M:%S")

            # Build DOM display string
            dom_str = f"DOM={dom_imb:+.2f}" if abs(dom_imb) > 0.01 else "DOM=--"
            cross_str = f"CROSS={cross_sig}" if cross_sig != "NEUTRAL" else ""

            log.info(
                "%s=%s  RSI=%.0f  %s  %s  sig=%s%s",
                inst["name"],
                f"{price:,.2f}" if price > 100 else f"{price:.5f}",
                sig.rsi, dom_str, cross_str,
                sig.signal,
                f" ({sig.confidence:.0%} {sig.reason})" if sig.signal != "FLAT" else "",
            )

            mgr.process(sig, price)

        except Exception as e:
            log.error("Tick error: %s", e, exc_info=True)

        time.sleep(interval)

    # Shutdown
    if mgr.open_trade:
        price = client.get_price() or mgr.open_trade.entry_price
        mgr.force_close(price)

    log.info("=" * 60)
    log.info("SESSION SUMMARY")
    log.info("  Trades: %d  |  Wins: %d  |  Losses: %d",
             mgr.stats["total"], mgr.stats["wins"], mgr.stats["losses"])
    if mgr.stats["total"]:
        log.info("  Win Rate: %.1f%%", mgr.stats["wins"] / mgr.stats["total"] * 100)
    log.info("  Total PnL: $%s", f"{mgr.stats['total_pnl']:+,.2f}")
    log.info("=" * 60)

    client.disconnect()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="FX/Crypto Scalp Bot")
    parser.add_argument("--symbol", type=str, default="BTC",
                        choices=list(INSTRUMENTS.keys()),
                        help="Instrument to trade (BTC, EURUSD)")
    parser.add_argument("--ibkr", type=str, default="paper",
                        choices=["paper", "live", "dry-run"],
                        help="IBKR mode")
    parser.add_argument("--interval", type=int, default=60,
                        help="Seconds between ticks (default: 60)")
    args = parser.parse_args()
    run(symbol=args.symbol, ibkr_mode=args.ibkr, interval=args.interval)
