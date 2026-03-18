#!/usr/bin/env python3
"""
Risk-On / Risk-Off Rotation Bot — IBKR Paper Trading

Strategy: Rotate capital between BTC (risk-on) and USD/JPY (risk-off) based on
composite signal strength from the fx_scalp_bot signal engine.

  Positive score  →  BTC allocation (risk-on)
  Negative score  →  USD/JPY allocation (risk-off)
  |score| < 0.35  →  Flat (cash)

Ladder sizing (scale-up only, never scale down):
  |score| >= 0.70  →  100% allocation
  |score| >= 0.35  →   50% allocation (entry level)

Never shorts — only buys the appropriate leg. On rotation, closes the current
leg entirely and opens the other.

Usage:
    python scripts/rotation_bot.py --ibkr paper --interval 30 --capital 1000
    python scripts/rotation_bot.py --ibkr dry-run
"""
from __future__ import annotations

import argparse
import csv
import logging
import os
import signal
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from scripts.fx_scalp_bot import (  # noqa: E402
    INSTRUMENTS,
    FXSignal,
    _parse_ts,
    compute_atr,
    compute_bollinger,
    compute_cross_market_signal,
    compute_dom_imbalance,
    compute_ema,
    compute_rsi,
    generate_fx_signal,
)

from zoneinfo import ZoneInfo  # noqa: E402

ET = ZoneInfo("America/New_York")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("rotation_bot")

# ─── Ladder thresholds ──────────────────────────────────────────────────────

LADDER = [
    (0.70, 100),
    (0.35, 50),
]

# Hysteresis: wider dead zone for exits than entries to prevent whipsaw
# Enter at ±0.35, but don't exit until score retreats to ±EXIT_THRESHOLD
# Full rotation (flip sides) requires opposite score to cross ENTRY threshold
EXIT_THRESHOLD = 0.10       # score must come back within ±0.10 to go flat
MIN_PROFIT_TO_EXIT = -0.50  # allow small losses on flat exit ($), but not on rotation

# Risk limits
MAX_DAILY_LOSS = 300        # USD
MAX_DAILY_ROTATIONS = 20
ROTATION_COOLDOWN_SEC = 300


def score_to_target(score: float) -> tuple[Optional[str], int]:
    """Map a composite score to (target_leg, alloc_pct).

    Returns:
        ("BTC", 50/100) for positive scores above threshold
        ("JPY", 50/100) for negative scores below threshold
        (None, 0) for scores in the dead zone
    """
    abs_score = abs(score)
    for threshold, alloc in LADDER:
        if abs_score >= threshold:
            leg = "BTC" if score > 0 else "JPY"
            return leg, alloc
    return None, 0


# ─── IBKR Connection ────────────────────────────────────────────────────────

class RotationIBKRClient:
    """Manages IBKR connection for BTC + USD/JPY rotation, plus cross-market data."""

    def __init__(self, mode: str):
        self.mode = mode
        self._ib = None
        self._btc_contract = None
        self._usdjpy_contract = None
        self._es_contract = None
        self._eur_contract = None
        self._dom_subscribed = False

    def connect(self):
        if self.mode == "dry-run":
            log.info("[DRY-RUN] Skipping IBKR connection")
            return
        import ib_insync as ib_mod

        port = int(os.environ.get("IBKR_PAPER_PORT", "7497"))
        client_id = int(os.environ.get("IBKR_ROTATION_CLIENT_ID", "18"))
        ib = ib_mod.IB()
        ib.connect("127.0.0.1", port, clientId=client_id, timeout=30)
        self._ib = ib

        # BTC crypto contract (PAXOS)
        self._btc_contract = ib_mod.Crypto("BTC", "PAXOS", "USD")
        ib.qualifyContracts(self._btc_contract)
        log.info("  BTC/USD (PAXOS) contract qualified")

        # USD/JPY forex contract (IDEALPRO)
        self._usdjpy_contract = ib_mod.Forex("USDJPY")
        ib.qualifyContracts(self._usdjpy_contract)
        log.info("  USD/JPY (IDEALPRO) contract qualified")

        # Cross-market: ES futures
        try:
            self._es_contract = ib_mod.Future("ES", "20260618", "CME")
            ib.qualifyContracts(self._es_contract)
            log.info("  ES futures contract qualified")
        except Exception as e:
            log.warning("  ES contract setup failed: %s", e)
            self._es_contract = None

        # Cross-market: EUR/USD
        try:
            self._eur_contract = ib_mod.Forex("EURUSD")
            ib.qualifyContracts(self._eur_contract)
            log.info("  EUR/USD contract qualified")
        except Exception as e:
            log.warning("  EUR/USD contract setup failed: %s", e)
            self._eur_contract = None

        # Subscribe to BTC DOM (primary signal driver)
        try:
            ib.reqMktDepth(self._btc_contract, numRows=10)
            self._dom_subscribed = True
            log.info("  BTC order book (DOM) subscribed — 10 levels")
        except Exception as e:
            log.warning("  BTC DOM subscription failed: %s", e)

        log.info("Connected to IBKR (%s) port=%d — rotation bot", self.mode, port)

    def disconnect(self):
        if self._ib:
            if self._dom_subscribed:
                try:
                    self._ib.cancelMktDepth(self._btc_contract)
                except Exception:
                    pass
            try:
                self._ib.disconnect()
            except Exception:
                pass
            self._ib = None

    # ── Market Data ──────────────────────────────────────────────────────────

    def get_dom(self) -> tuple[list, list]:
        """Get BTC order book depth (bids, asks)."""
        if self.mode == "dry-run" or not self._ib:
            return [], []
        try:
            ticker = self._ib.ticker(self._btc_contract)
            if ticker:
                return (ticker.domBids or []), (ticker.domAsks or [])
        except Exception as e:
            log.debug("get_dom failed: %s", e)
        return [], []

    def get_btc_bars(self, duration: str = "1 D", bar_size: str = "1 min") -> pd.DataFrame:
        """Fetch BTC historical bars."""
        if self.mode == "dry-run" or not self._ib:
            return pd.DataFrame()
        try:
            bars = self._ib.reqHistoricalData(
                self._btc_contract, endDateTime="", durationStr=duration,
                barSizeSetting=bar_size, whatToShow="MIDPOINT",
                useRTH=False, formatDate=1,
            )
            if not bars:
                return pd.DataFrame()
            data = [{"date": b.date, "open": b.open, "high": b.high,
                     "low": b.low, "close": b.close, "volume": getattr(b, "volume", 0)}
                    for b in bars]
            return pd.DataFrame(data).set_index("date")
        except Exception as e:
            log.warning("get_btc_bars failed: %s", e)
            return pd.DataFrame()

    def get_usdjpy_bars(self, duration: str = "1 D", bar_size: str = "1 min") -> pd.DataFrame:
        """Fetch USD/JPY historical bars."""
        if self.mode == "dry-run" or not self._ib:
            return pd.DataFrame()
        try:
            bars = self._ib.reqHistoricalData(
                self._usdjpy_contract, endDateTime="", durationStr=duration,
                barSizeSetting=bar_size, whatToShow="MIDPOINT",
                useRTH=False, formatDate=1,
            )
            if not bars:
                return pd.DataFrame()
            data = [{"date": b.date, "open": b.open, "high": b.high,
                     "low": b.low, "close": b.close, "volume": getattr(b, "volume", 0)}
                    for b in bars]
            return pd.DataFrame(data).set_index("date")
        except Exception as e:
            log.warning("get_usdjpy_bars failed: %s", e)
            return pd.DataFrame()

    def get_es_bars(self, duration: str = "14400 S", bar_size: str = "5 mins") -> pd.DataFrame:
        """Get ES futures bars for cross-market correlation."""
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

    def get_btc_price(self) -> Optional[float]:
        if self.mode == "dry-run" or not self._ib:
            return None
        try:
            [ticker] = self._ib.reqTickers(self._btc_contract)
            self._ib.sleep(0.3)
            [ticker] = self._ib.reqTickers(self._btc_contract)
            bid = float(ticker.bid) if ticker.bid and ticker.bid > 0 else 0
            ask = float(ticker.ask) if ticker.ask and ticker.ask > 0 else 0
            if bid > 0 and ask > 0:
                return (bid + ask) / 2
            return float(ticker.last) if ticker.last and ticker.last > 0 else None
        except Exception as e:
            log.warning("get_btc_price failed: %s", e)
            return None

    def get_usdjpy_price(self) -> Optional[float]:
        if self.mode == "dry-run" or not self._ib:
            return None
        try:
            [ticker] = self._ib.reqTickers(self._usdjpy_contract)
            self._ib.sleep(0.3)
            [ticker] = self._ib.reqTickers(self._usdjpy_contract)
            bid = float(ticker.bid) if ticker.bid and ticker.bid > 0 else 0
            ask = float(ticker.ask) if ticker.ask and ticker.ask > 0 else 0
            if bid > 0 and ask > 0:
                return (bid + ask) / 2
            return float(ticker.last) if ticker.last and ticker.last > 0 else None
        except Exception as e:
            log.warning("get_usdjpy_price failed: %s", e)
            return None

    def get_btc_bid_ask(self) -> tuple[float, float]:
        if self.mode == "dry-run" or not self._ib:
            return (0, 0)
        try:
            [ticker] = self._ib.reqTickers(self._btc_contract)
            self._ib.sleep(0.3)
            [ticker] = self._ib.reqTickers(self._btc_contract)
            return (float(ticker.bid or 0), float(ticker.ask or 0))
        except Exception:
            return (0, 0)

    def get_usdjpy_bid_ask(self) -> tuple[float, float]:
        if self.mode == "dry-run" or not self._ib:
            return (0, 0)
        try:
            [ticker] = self._ib.reqTickers(self._usdjpy_contract)
            self._ib.sleep(0.3)
            [ticker] = self._ib.reqTickers(self._usdjpy_contract)
            return (float(ticker.bid or 0), float(ticker.ask or 0))
        except Exception:
            return (0, 0)

    # ── Orders ───────────────────────────────────────────────────────────────

    def buy_btc(self, qty: float, limit: Optional[float] = None) -> dict:
        # For fractional BTC on PAXOS, use cashQty instead of totalQuantity
        cash_qty = 0.0
        if qty < 1.0 and limit and limit > 0:
            cash_qty = round(qty * limit, 2)
        return self._place_order(self._btc_contract, "BUY", qty, limit,
                                 qty_decimals=4, tif_limit="GTC", tif_market="GTC",
                                 label="BTC", cash_qty=cash_qty)

    def sell_btc(self, qty: float, limit: Optional[float] = None) -> dict:
        # For fractional BTC on PAXOS, use cashQty instead of totalQuantity
        cash_qty = 0.0
        if qty < 1.0 and limit and limit > 0:
            cash_qty = round(qty * limit, 2)
        return self._place_order(self._btc_contract, "SELL", qty, limit,
                                 qty_decimals=4, tif_limit="GTC", tif_market="GTC",
                                 label="BTC", cash_qty=cash_qty)

    def buy_usdjpy(self, qty: float, limit: Optional[float] = None) -> dict:
        return self._place_order(self._usdjpy_contract, "BUY", qty, limit,
                                 qty_decimals=0, tif_limit="IOC", tif_market="DAY",
                                 label="USD/JPY")

    def sell_usdjpy(self, qty: float, limit: Optional[float] = None) -> dict:
        return self._place_order(self._usdjpy_contract, "SELL", qty, limit,
                                 qty_decimals=0, tif_limit="IOC", tif_market="DAY",
                                 label="USD/JPY")

    def _place_order(self, contract, action: str, qty: float, limit: Optional[float],
                     qty_decimals: int, tif_limit: str, tif_market: str, label: str,
                     cash_qty: float = 0.0) -> dict:
        import ib_insync as ib_mod

        qty = round(qty, qty_decimals)
        arrow = ">>>" if action == "BUY" else "<<<"
        if cash_qty > 0:
            log.info("%s %s %s %s @ %s (cashQty=$%.2f)", arrow, action, qty, label,
                     f"limit ${limit:.2f}" if limit else "MKT", cash_qty)
        else:
            log.info("%s %s %s %s @ %s", arrow, action, qty, label,
                     f"limit ${limit:.2f}" if limit else "MKT")

        if self.mode == "dry-run":
            log.info("[DRY-RUN] Simulated %s fill", action.lower())
            return {"success": True, "fill_price": limit or 0, "qty": qty}

        try:
            if limit:
                order = ib_mod.LimitOrder(action=action, totalQuantity=qty,
                                          lmtPrice=limit, tif=tif_limit)
            else:
                order = ib_mod.MarketOrder(action=action, totalQuantity=qty,
                                           tif=tif_market)

            # PAXOS fractional crypto: use cashQty instead of totalQuantity
            if cash_qty > 0 and label == "BTC":
                order.totalQuantity = 0
                order.cashQty = cash_qty

            trade = self._ib.placeOrder(contract, order)
            for _ in range(30):
                self._ib.sleep(0.5)
                if trade.isDone():
                    break

            if trade.orderStatus.status == "Filled":
                fp = float(trade.orderStatus.avgFillPrice)
                filled_qty = float(trade.orderStatus.filled) if trade.orderStatus.filled else qty
                log.info("  %s FILLED @ %s (qty=%.4f)", action, fp, filled_qty)
                return {"success": True, "fill_price": fp, "qty": filled_qty}

            # Limit not filled — cancel and try market for exits
            log.warning("  %s limit not filled (%s), trying MKT...",
                        action, trade.orderStatus.status)
            try:
                self._ib.cancelOrder(order)
                self._ib.sleep(1)
            except Exception:
                pass
            mkt_order = ib_mod.MarketOrder(action=action, totalQuantity=qty,
                                           tif=tif_market)
            # Also apply cashQty for market fallback on fractional BTC
            if cash_qty > 0 and label == "BTC":
                mkt_order.totalQuantity = 0
                mkt_order.cashQty = cash_qty
            trade2 = self._ib.placeOrder(contract, mkt_order)
            for _ in range(20):
                self._ib.sleep(0.5)
                if trade2.isDone():
                    break
            if trade2.orderStatus.status == "Filled":
                fp = float(trade2.orderStatus.avgFillPrice)
                filled_qty = float(trade2.orderStatus.filled) if trade2.orderStatus.filled else qty
                log.info("  %s FILLED (MKT) @ %s (qty=%.4f)", action, fp, filled_qty)
                return {"success": True, "fill_price": fp, "qty": filled_qty}
            return {"success": False, "fill_price": 0, "qty": 0}

        except Exception as e:
            log.error("%s %s failed: %s", action, label, e)
            return {"success": False, "fill_price": 0, "qty": 0}

    def get_positions(self) -> dict:
        """Return dict with btc_qty, btc_avg, usdjpy_qty, usdjpy_avg."""
        result = {"btc_qty": 0.0, "btc_avg": 0.0, "usdjpy_qty": 0.0, "usdjpy_avg": 0.0}
        if self.mode == "dry-run" or not self._ib:
            return result
        try:
            positions = self._ib.positions()
            for p in positions:
                sym = p.contract.symbol
                if sym == "BTC":
                    result["btc_qty"] = float(p.position)
                    result["btc_avg"] = float(p.avgCost)
                elif sym == "USD" and getattr(p.contract, "currency", "") == "JPY":
                    result["usdjpy_qty"] = float(p.position)
                    result["usdjpy_avg"] = float(p.avgCost)
        except Exception as e:
            log.warning("get_positions failed: %s", e)
        return result


# ─── Rotation Manager ───────────────────────────────────────────────────────

class RotationManager:
    """Manages two-leg rotation between BTC and USD/JPY with ladder sizing."""

    def __init__(self, client: RotationIBKRClient, base_capital_usd: float, log_dir: Path):
        self.client = client
        self.base_capital_usd = base_capital_usd
        self.log_dir = log_dir
        self.log_dir.mkdir(parents=True, exist_ok=True)

        # Current position state
        self.current_leg: Optional[str] = None   # "BTC", "JPY", or None
        self.current_qty: float = 0.0
        self.current_entry_price: float = 0.0
        self.current_alloc_pct: int = 0           # 0, 50, 100

        # Risk management
        self._daily_pnl: float = 0.0
        self._daily_rotations: int = 0
        self._today_date: str = ""
        self._last_rotation_time: float = 0.0

        # Stats
        self.stats = {
            "total_rotations": 0,
            "total_pnl": 0.0,
            "wins": 0,
            "losses": 0,
        }

        # CSV logs
        today_str = datetime.now(ET).strftime("%Y%m%d")
        self.trade_log = log_dir / f"rotation_trades_{today_str}.csv"
        self.signal_log = log_dir / f"rotation_signals_{today_str}.csv"
        self._init_logs()

    def _init_logs(self):
        if not self.trade_log.exists():
            with open(self.trade_log, "w", newline="") as f:
                csv.writer(f).writerow([
                    "timestamp", "action", "leg", "qty", "price",
                    "alloc_pct", "score", "pnl_dollar", "reason",
                ])
        if not self.signal_log.exists():
            with open(self.signal_log, "w", newline="") as f:
                csv.writer(f).writerow([
                    "timestamp", "symbol", "price", "ema_fast", "ema_slow",
                    "rsi", "atr", "bb_upper", "bb_lower", "bb_width",
                    "dom_imbalance", "dom_bid_size", "dom_ask_size",
                    "es_momentum", "eur_momentum", "cross_signal",
                    "signal", "reason", "confidence",
                    "target_leg", "target_alloc",
                ])

    def _log_signal(self, sig: FXSignal, target_leg: Optional[str], target_alloc: int):
        with open(self.signal_log, "a", newline="") as f:
            csv.writer(f).writerow([
                sig.timestamp, sig.symbol, f"{sig.price:.6f}",
                f"{sig.ema_fast:.6f}", f"{sig.ema_slow:.6f}",
                f"{sig.rsi:.1f}", f"{sig.atr:.6f}",
                f"{sig.bb_upper:.6f}", f"{sig.bb_lower:.6f}", f"{sig.bb_width:.6f}",
                f"{sig.dom_imbalance:.3f}", f"{sig.dom_bid_size:.4f}", f"{sig.dom_ask_size:.4f}",
                f"{sig.es_momentum:.4f}", f"{sig.eur_momentum:.4f}", sig.cross_signal,
                sig.signal, sig.reason, f"{sig.confidence:.2f}",
                target_leg or "FLAT", target_alloc,
            ])

    def _log_trade(self, action: str, leg: str, qty: float, price: float,
                   alloc_pct: int, score: float, pnl_dollar: float, reason: str):
        ts = datetime.now(ET).strftime("%Y-%m-%d %H:%M:%S")
        with open(self.trade_log, "a", newline="") as f:
            csv.writer(f).writerow([
                ts, action, leg, f"{qty:.4f}" if leg == "BTC" else f"{qty:.0f}",
                f"{price:.2f}", alloc_pct, f"{score:.3f}",
                f"{pnl_dollar:.2f}", reason,
            ])

    def detect_open_positions(self):
        """On startup, check IBKR for existing BTC or USD/JPY positions and adopt."""
        pos = self.client.get_positions()
        if pos["btc_qty"] > 0:
            self.current_leg = "BTC"
            self.current_qty = pos["btc_qty"]
            self.current_entry_price = pos["btc_avg"]
            # Estimate alloc_pct from position value
            btc_price = self.client.get_btc_price() or pos["btc_avg"]
            value = pos["btc_qty"] * btc_price
            self.current_alloc_pct = self._estimate_alloc_pct(value)
            log.info("  ADOPTED BTC position: %.4f BTC @ avg $%s (est %d%% alloc)",
                     self.current_qty, f"{self.current_entry_price:,.2f}", self.current_alloc_pct)
        elif pos["usdjpy_qty"] > 0:
            self.current_leg = "JPY"
            self.current_qty = pos["usdjpy_qty"]
            self.current_entry_price = pos["usdjpy_avg"]
            value = pos["usdjpy_qty"]  # USD/JPY qty is in USD
            self.current_alloc_pct = self._estimate_alloc_pct(value)
            log.info("  ADOPTED USD/JPY position: %.0f units @ avg %.3f (est %d%% alloc)",
                     self.current_qty, self.current_entry_price, self.current_alloc_pct)

    def _estimate_alloc_pct(self, value: float) -> int:
        """Estimate allocation percentage from position value."""
        if self.base_capital_usd <= 0:
            return 50
        ratio = value / self.base_capital_usd * 100
        # Snap to nearest ladder level
        if ratio >= 75:
            return 100
        else:
            return 50

    def _reset_daily(self, date_str: str):
        """Reset daily counters if the date has changed."""
        if date_str != self._today_date:
            self._today_date = date_str
            self._daily_pnl = 0.0
            self._daily_rotations = 0

    def _compute_score_from_signal(self, sig: FXSignal) -> float:
        """Reconstruct the composite score from the signal.

        The fx_scalp_bot signal engine converts score into LONG/SHORT/FLAT, but
        we need the raw score magnitude for ladder sizing. We reconstruct it
        from the signal components using the same weights.
        """
        score = 0.0

        # (D) DOM imbalance — weight 0.30
        if abs(sig.dom_imbalance) > 0.15:
            score += sig.dom_imbalance * 0.30

        # (C) Cross-market — weight 0.25
        if sig.cross_signal == "RISK_ON":
            score += 0.25
        elif sig.cross_signal == "RISK_OFF":
            score -= 0.25

        # BB position — weight 0.20
        bb_range = sig.bb_upper - sig.bb_lower
        if bb_range > 0:
            bb_pct = (sig.price - sig.bb_lower) / bb_range
        else:
            bb_pct = 0.5
        if bb_pct < 0.1 and sig.rsi < 35:
            score += 0.20
        elif bb_pct > 0.9 and sig.rsi > 65:
            score -= 0.20
        elif bb_pct < 0.2:
            score += 0.10
        elif bb_pct > 0.8:
            score -= 0.10

        # EMA trend — weight 0.15
        if sig.ema_fast > sig.ema_slow:
            score += 0.15
        elif sig.ema_fast < sig.ema_slow:
            score -= 0.15

        # RSI momentum — weight 0.10
        if sig.rsi > 60:
            score += 0.10
        elif sig.rsi < 40:
            score -= 0.10

        return score

    def check_leg_trend(self, leg: str, bars: pd.DataFrame) -> tuple[bool, str]:
        """Check if the target leg's own price action supports entry.

        Returns (ok_to_enter, reason).
        We only buy an asset if its EMA9 > EMA21 (uptrend) or price is
        near support (RSI < 35). Don't buy into a falling knife.
        """
        if bars.empty or len(bars) < 25:
            return True, "insufficient data"  # allow entry if no data

        closes = bars["close"]
        ema9 = compute_ema(closes, 9)
        ema21 = compute_ema(closes, 21)
        rsi = compute_rsi(closes, 14)

        ema9_val = float(ema9.iloc[-1])
        ema21_val = float(ema21.iloc[-1])
        rsi_val = float(rsi.iloc[-1]) if not pd.isna(rsi.iloc[-1]) else 50.0

        in_uptrend = ema9_val > ema21_val
        oversold = rsi_val < 35  # potential reversal / support bounce

        if in_uptrend:
            return True, f"{leg} uptrend (EMA9>{leg[:3]}EMA21, RSI={rsi_val:.0f})"
        elif oversold:
            return True, f"{leg} oversold bounce (RSI={rsi_val:.0f})"
        else:
            return False, f"{leg} downtrend (EMA9<EMA21, RSI={rsi_val:.0f})"

    def _unrealized_pnl(self, btc_price: float, usdjpy_price: float) -> float:
        """Compute unrealized PnL for the current open position."""
        if self.current_leg is None or self.current_qty <= 0:
            return 0.0
        if self.current_leg == "BTC":
            return (btc_price - self.current_entry_price) * self.current_qty
        else:  # JPY
            return (usdjpy_price - self.current_entry_price) / self.current_entry_price * self.current_qty

    def process(self, sig: FXSignal, btc_price: float, usdjpy_price: float,
                btc_trend_ok: bool = True, jpy_trend_ok: bool = True,
                btc_trend_reason: str = "", jpy_trend_reason: str = ""):
        """Main rotation logic with hysteresis and trend confirmation.

        Rules when holding a position:
        1. HOLD if score still favors current leg (even if weakened)
        2. GO FLAT only if score retreats within ±EXIT_THRESHOLD (near zero)
        3. ROTATE only if opposite side crosses entry threshold (±0.35)
           AND we're profitable or the opposing signal is strong (>=0.50)
        4. SCALE UP only within same leg (never scale down to prevent churn)
        5. Never close at a loss just because score weakened slightly
        6. Don't enter a leg if its own price action is in a downtrend
        """
        today = sig.timestamp[:10]
        self._reset_daily(today)

        # Compute score and target
        score = self._compute_score_from_signal(sig)
        target_leg, target_alloc = score_to_target(score)

        # Log signal
        self._log_signal(sig, target_leg, target_alloc)

        # Risk checks
        if self._daily_pnl <= -MAX_DAILY_LOSS:
            log.info("  Daily loss limit hit ($%.0f), closing all and stopping",
                     MAX_DAILY_LOSS)
            self._close_leg(btc_price, usdjpy_price, score, "DAILY_LOSS_LIMIT")
            return

        if self._daily_rotations >= MAX_DAILY_ROTATIONS:
            log.debug("  Max daily rotations (%d) reached, holding", MAX_DAILY_ROTATIONS)
            return

        # Cooldown check
        now = time.time()
        if now - self._last_rotation_time < ROTATION_COOLDOWN_SEC and self.current_leg is not None:
            log.debug("  Rotation cooldown active (%.0fs remaining)",
                      ROTATION_COOLDOWN_SEC - (now - self._last_rotation_time))
            return

        unrealized = self._unrealized_pnl(btc_price, usdjpy_price)

        # ── No position: entry with trend confirmation ──
        if self.current_leg is None:
            if target_leg is not None:
                # Check that the target asset's own trend supports buying
                trend_ok = btc_trend_ok if target_leg == "BTC" else jpy_trend_ok
                trend_reason = btc_trend_reason if target_leg == "BTC" else jpy_trend_reason
                if not trend_ok:
                    log.info("  BLOCKED %s entry — %s", target_leg, trend_reason)
                    return
                self._open_leg(target_leg, target_alloc, btc_price, usdjpy_price, score)
                self._last_rotation_time = now
            return

        # ── Have a position: apply hysteresis ──

        # Same leg — only scale UP (never scale down to prevent churn)
        if target_leg == self.current_leg:
            if target_alloc > self.current_alloc_pct:
                self._scale_position(target_leg, target_alloc, btc_price, usdjpy_price, score)
            # else: target_alloc <= current — just hold, do NOT scale down
            return

        # Score is in dead zone (|score| < 0.35) — should we go flat?
        if target_leg is None:
            # Only go flat if score is truly neutral (within EXIT_THRESHOLD)
            # AND position is profitable (or very small loss)
            if abs(score) <= EXIT_THRESHOLD:
                if unrealized >= MIN_PROFIT_TO_EXIT:
                    log.info("  Score neutral (%.3f), closing %s at $%+.2f",
                             score, self.current_leg, unrealized)
                    self._close_leg(btc_price, usdjpy_price, score, "SCORE_NEUTRAL")
                    self._last_rotation_time = now
                else:
                    log.debug("  Score neutral but position at $%+.2f loss — holding",
                              unrealized)
            # else: score weakened but hasn't reached neutral — hold
            return

        # Opposite side has conviction (target_leg != current_leg, target_leg is not None)
        # Only rotate if:
        #   a) Current position is profitable — take profit and flip, OR
        #   b) Opposing signal is strong (alloc >= 100%) — conviction override
        # Check target leg trend before any rotation
        trend_ok = btc_trend_ok if target_leg == "BTC" else jpy_trend_ok
        trend_reason = btc_trend_reason if target_leg == "BTC" else jpy_trend_reason

        if unrealized >= 0:
            # Profitable — take profit, but only open new leg if trend confirms
            log.info("  Rotating %s -> %s (profitable $%+.2f, score=%+.3f)",
                     self.current_leg, target_leg, unrealized, score)
            self._close_leg(btc_price, usdjpy_price, score, f"ROTATE_TO_{target_leg}")
            self._daily_rotations += 1
            self.stats["total_rotations"] += 1
            if trend_ok:
                self._open_leg(target_leg, target_alloc, btc_price, usdjpy_price, score)
            else:
                log.info("  Took profit on %s but BLOCKED %s entry — %s",
                         self.current_leg, target_leg, trend_reason)
            self._last_rotation_time = now
        elif target_alloc >= 100 and trend_ok:
            # Strong opposing conviction + trend confirms — accept the loss and flip
            log.info("  Strong opposing signal (%s %d%%), closing %s at $%+.2f loss",
                     target_leg, target_alloc, self.current_leg, unrealized)
            self._close_leg(btc_price, usdjpy_price, score, f"STRONG_ROTATE_TO_{target_leg}")
            self._daily_rotations += 1
            self.stats["total_rotations"] += 1
            self._open_leg(target_leg, target_alloc, btc_price, usdjpy_price, score)
            self._last_rotation_time = now
        else:
            # Weak signal, or target in downtrend, or losing — hold
            if not trend_ok:
                log.debug("  Opposing %s blocked — %s", target_leg, trend_reason)
            else:
                log.debug("  Opposing %s at %d%% but position at $%+.2f loss — holding %s",
                          target_leg, target_alloc, unrealized, self.current_leg)

    def _close_leg(self, btc_price: float, usdjpy_price: float,
                   score: float, reason: str):
        """Close the current position entirely."""
        if self.current_leg is None or self.current_qty <= 0:
            return

        leg = self.current_leg
        qty = self.current_qty
        entry = self.current_entry_price

        if leg == "BTC":
            bid, _ = self.client.get_btc_bid_ask()
            limit = bid if bid > 0 else btc_price
            result = self.client.sell_btc(qty, limit)
            exit_price = result["fill_price"] if result["success"] else btc_price
            pnl = (exit_price - entry) * qty
        else:  # JPY
            bid, _ = self.client.get_usdjpy_bid_ask()
            limit = bid if bid > 0 else usdjpy_price
            result = self.client.sell_usdjpy(qty, limit)
            exit_price = result["fill_price"] if result["success"] else usdjpy_price
            # USD/JPY PnL: qty is in USD, PnL is in pips * qty / price
            # Simplified: (exit - entry) / entry * qty for percentage-based
            pnl = (exit_price - entry) / entry * qty

        pnl = round(pnl, 2)
        self._daily_pnl += pnl
        self.stats["total_pnl"] += pnl
        if pnl > 0:
            self.stats["wins"] += 1
        else:
            self.stats["losses"] += 1

        self._log_trade("CLOSE", leg, qty, exit_price, 0, score, pnl, reason)
        log.info("  <<< CLOSE %s %.4f @ %s | PnL: $%+.2f | %s",
                 leg, qty, f"{exit_price:,.2f}" if exit_price > 100 else f"{exit_price:.3f}",
                 pnl, reason)

        self.current_leg = None
        self.current_qty = 0.0
        self.current_entry_price = 0.0
        self.current_alloc_pct = 0

    def _open_leg(self, leg: str, alloc_pct: int, btc_price: float,
                  usdjpy_price: float, score: float):
        """Open a new position in the target leg."""
        if leg == "BTC":
            qty = (self.base_capital_usd * alloc_pct / 100) / btc_price
            qty = max(qty, 0.001)
            qty = round(qty, 4)
            _, ask = self.client.get_btc_bid_ask()
            limit = ask if ask > 0 else btc_price
            result = self.client.buy_btc(qty, limit)
            fill_price = result["fill_price"] if result["success"] else btc_price
        else:  # JPY
            qty = self.base_capital_usd * alloc_pct / 100
            qty = max(qty, 100)
            qty = round(qty, 0)
            _, ask = self.client.get_usdjpy_bid_ask()
            limit = ask if ask > 0 else usdjpy_price
            result = self.client.buy_usdjpy(qty, limit)
            fill_price = result["fill_price"] if result["success"] else usdjpy_price

        if not result["success"]:
            log.warning("  OPEN %s failed — order not filled", leg)
            return

        self.current_leg = leg
        self.current_qty = result["qty"] or qty
        self.current_entry_price = fill_price
        self.current_alloc_pct = alloc_pct

        self._log_trade("OPEN", leg, self.current_qty, fill_price, alloc_pct, score, 0.0,
                        f"score={score:+.3f}")
        log.info("  >>> OPEN %s %.4f @ %s | %d%% alloc ($%.0f) | score=%+.3f",
                 leg, self.current_qty,
                 f"{fill_price:,.2f}" if fill_price > 100 else f"{fill_price:.3f}",
                 alloc_pct, self.base_capital_usd * alloc_pct / 100, score)

    def _scale_position(self, leg: str, new_alloc_pct: int, btc_price: float,
                        usdjpy_price: float, score: float):
        """Adjust position size up or down within the same leg."""
        old_alloc = self.current_alloc_pct
        if new_alloc_pct == old_alloc:
            return

        scaling_up = new_alloc_pct > old_alloc

        if leg == "BTC":
            price = btc_price
            new_target_qty = max((self.base_capital_usd * new_alloc_pct / 100) / price, 0.001)
            new_target_qty = round(new_target_qty, 4)
            delta_qty = round(abs(new_target_qty - self.current_qty), 4)

            if delta_qty < 0.001:
                return  # Too small to adjust

            if scaling_up:
                _, ask = self.client.get_btc_bid_ask()
                limit = ask if ask > 0 else price
                result = self.client.buy_btc(delta_qty, limit)
                action = "SCALE_UP"
            else:
                bid, _ = self.client.get_btc_bid_ask()
                limit = bid if bid > 0 else price
                result = self.client.sell_btc(delta_qty, limit)
                action = "SCALE_DOWN"

            fill_price = result["fill_price"] if result["success"] else price
        else:  # JPY
            price = usdjpy_price
            new_target_qty = max(self.base_capital_usd * new_alloc_pct / 100, 100)
            new_target_qty = round(new_target_qty, 0)
            delta_qty = round(abs(new_target_qty - self.current_qty), 0)

            if delta_qty < 100:
                return  # Too small to adjust

            if scaling_up:
                _, ask = self.client.get_usdjpy_bid_ask()
                limit = ask if ask > 0 else price
                result = self.client.buy_usdjpy(delta_qty, limit)
                action = "SCALE_UP"
            else:
                bid, _ = self.client.get_usdjpy_bid_ask()
                limit = bid if bid > 0 else price
                result = self.client.sell_usdjpy(delta_qty, limit)
                action = "SCALE_DOWN"

            fill_price = result["fill_price"] if result["success"] else price

        if not result["success"]:
            log.warning("  %s %s failed — order not filled", action, leg)
            return

        # Compute blended entry price for scale-ups
        actual_delta = result["qty"] or delta_qty
        if scaling_up:
            total_cost = (self.current_entry_price * self.current_qty) + (fill_price * actual_delta)
            self.current_qty += actual_delta
            self.current_entry_price = total_cost / self.current_qty if self.current_qty > 0 else fill_price
        else:
            # Partial close — compute PnL on the trimmed portion
            if leg == "BTC":
                pnl = (fill_price - self.current_entry_price) * actual_delta
            else:
                pnl = (fill_price - self.current_entry_price) / self.current_entry_price * actual_delta
            pnl = round(pnl, 2)
            self._daily_pnl += pnl
            self.stats["total_pnl"] += pnl
            self.current_qty -= actual_delta
            self.current_qty = max(self.current_qty, 0)

        self.current_alloc_pct = new_alloc_pct

        self._log_trade(action, leg, actual_delta, fill_price, new_alloc_pct, score,
                        0.0 if scaling_up else pnl,
                        f"{old_alloc}%->{new_alloc_pct}%")
        log.info("  %s %s %s %.4f @ %s | %d%% -> %d%% | score=%+.3f",
                 "+++" if scaling_up else "---", action, leg, actual_delta,
                 f"{fill_price:,.2f}" if fill_price > 100 else f"{fill_price:.3f}",
                 old_alloc, new_alloc_pct, score)

    def close_all(self, btc_price: float, usdjpy_price: float):
        """Shutdown: close any open position."""
        if self.current_leg is not None:
            self._close_leg(btc_price, usdjpy_price, 0.0, "SHUTDOWN")


# ─── Main Loop ──────────────────────────────────────────────────────────────

def run(ibkr_mode: str, interval: int = 60, capital: float = 1000):
    log_dir = ROOT / "logs" / "rotation"

    client = RotationIBKRClient(mode=ibkr_mode)
    client.connect()

    mgr = RotationManager(client, capital, log_dir)

    log.info("=" * 60)
    log.info("Risk-On/Risk-Off Rotation Bot — %s mode", ibkr_mode.upper())
    log.info("  Capital: $%s  |  Interval: %ds", f"{capital:,.0f}", interval)
    log.info("  Legs: BTC (risk-on) / USD/JPY (risk-off)")
    log.info("  Ladder: 50%%/100%% at |score| 0.35/0.70 (scale-up only)")
    log.info("  Max daily loss: $%d  |  Max rotations: %d  |  Cooldown: %ds",
             MAX_DAILY_LOSS, MAX_DAILY_ROTATIONS, ROTATION_COOLDOWN_SEC)
    log.info("  Trade log: %s", mgr.trade_log)

    # Adopt any existing positions
    mgr.detect_open_positions()
    if mgr.current_leg:
        log.info("  Managing existing %s position", mgr.current_leg)
    else:
        log.info("  No existing positions — waiting for signals")
    log.info("=" * 60)

    # Signal handling for graceful shutdown
    running = [True]

    def _stop(signum, frame):
        log.info("Shutdown signal received...")
        running[0] = False

    signal.signal(signal.SIGINT, _stop)
    signal.signal(signal.SIGTERM, _stop)

    # Cross-market data cache (refresh every 5 minutes)
    _cross_cache = {"es_bars": pd.DataFrame(), "eur_bars": pd.DataFrame(),
                    "usdjpy_bars": pd.DataFrame(), "last_refresh": 0.0}
    CROSS_REFRESH_SEC = 300

    btc_inst = INSTRUMENTS["BTC"]

    while running[0]:
        try:
            # Fetch BTC bars (primary signal driver)
            btc_bars = client.get_btc_bars(duration="1 D", bar_size="1 min")
            if btc_bars.empty:
                log.debug("No BTC bars, waiting...")
                time.sleep(interval)
                continue

            # Get live prices
            btc_price = client.get_btc_price()
            if not btc_price:
                btc_price = float(btc_bars["close"].iloc[-1])

            usdjpy_price = client.get_usdjpy_price()
            if not usdjpy_price:
                usdjpy_price = 150.0  # fallback placeholder

            # (D) BTC order book imbalance
            dom_bids, dom_asks = client.get_dom()
            dom_imb, dom_bid_sz, dom_ask_sz = compute_dom_imbalance(dom_bids, dom_asks)

            # (C) Cross-market data + USD/JPY bars — cached, refresh every 5 min
            now = time.time()
            if now - _cross_cache["last_refresh"] > CROSS_REFRESH_SEC:
                _cross_cache["es_bars"] = client.get_es_bars()
                _cross_cache["eur_bars"] = client.get_eur_bars()
                _cross_cache["usdjpy_bars"] = client.get_usdjpy_bars(duration="1 D", bar_size="5 mins")
                _cross_cache["last_refresh"] = now
                if not _cross_cache["es_bars"].empty:
                    log.debug("  Refreshed cross-market: ES=%d bars, EUR=%d bars, USDJPY=%d bars",
                              len(_cross_cache["es_bars"]), len(_cross_cache["eur_bars"]),
                              len(_cross_cache["usdjpy_bars"]))

            es_mom, eur_mom, cross_sig = compute_cross_market_signal(
                _cross_cache["es_bars"], _cross_cache["eur_bars"],
            )

            # Generate signal using BTC instrument profile
            sig = generate_fx_signal(
                btc_bars, btc_inst,
                dom_imbalance=dom_imb, dom_bid_size=dom_bid_sz, dom_ask_size=dom_ask_sz,
                es_momentum=es_mom, eur_momentum=eur_mom, cross_signal=cross_sig,
            )
            if not sig:
                time.sleep(interval)
                continue

            # Update with live price and timestamp
            sig.price = btc_price
            sig.timestamp = datetime.now(ET).strftime("%Y-%m-%d %H:%M:%S")

            # Compute rotation target for display
            score = mgr._compute_score_from_signal(sig)
            target_leg, target_alloc = score_to_target(score)

            # Check leg trends — don't buy into a falling asset
            btc_trend_ok, btc_trend_reason = mgr.check_leg_trend("BTC", btc_bars)
            jpy_trend_ok, jpy_trend_reason = mgr.check_leg_trend(
                "USD/JPY", _cross_cache["usdjpy_bars"])

            # Log tick
            dom_str = f"DOM={dom_imb:+.2f}" if abs(dom_imb) > 0.01 else "DOM=--"
            cross_str = f"CROSS={cross_sig}" if cross_sig != "NEUTRAL" else ""
            pos_str = f"[{mgr.current_leg or 'FLAT'} {mgr.current_alloc_pct}%]"
            target_str = f"-> {target_leg or 'FLAT'} {target_alloc}%"
            # Show trend status for target leg
            if target_leg == "BTC" and not btc_trend_ok:
                target_str += " (BTC downtrend!)"
            elif target_leg == "JPY" and not jpy_trend_ok:
                target_str += " (JPY downtrend!)"

            log.info(
                "BTC=$%s  RSI=%.0f  %s  %s  score=%+.3f  %s %s",
                f"{btc_price:,.2f}", sig.rsi, dom_str, cross_str,
                score, pos_str, target_str,
            )

            # Process rotation logic
            mgr.process(sig, btc_price, usdjpy_price,
                        btc_trend_ok=btc_trend_ok, jpy_trend_ok=jpy_trend_ok,
                        btc_trend_reason=btc_trend_reason, jpy_trend_reason=jpy_trend_reason)

        except Exception as e:
            log.error("Tick error: %s", e, exc_info=True)

        time.sleep(interval)

    # Graceful shutdown: close all positions
    log.info("Shutting down — closing all positions...")
    btc_price = client.get_btc_price() or 0
    usdjpy_price = client.get_usdjpy_price() or 0
    mgr.close_all(btc_price, usdjpy_price)

    log.info("=" * 60)
    log.info("SESSION SUMMARY")
    log.info("  Rotations: %d  |  Wins: %d  |  Losses: %d",
             mgr.stats["total_rotations"],
             mgr.stats["wins"], mgr.stats["losses"])
    total_trades = mgr.stats["wins"] + mgr.stats["losses"]
    if total_trades:
        log.info("  Win Rate: %.1f%%", mgr.stats["wins"] / total_trades * 100)
    log.info("  Total PnL: $%s", f"{mgr.stats['total_pnl']:+,.2f}")
    log.info("  Daily PnL: $%s", f"{mgr._daily_pnl:+,.2f}")
    log.info("=" * 60)

    client.disconnect()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Risk-On/Risk-Off Rotation Bot")
    parser.add_argument("--ibkr", type=str, default="paper",
                        choices=["paper", "live", "dry-run"],
                        help="IBKR mode (default: paper)")
    parser.add_argument("--interval", type=int, default=60,
                        help="Seconds between ticks (default: 60)")
    parser.add_argument("--capital", type=float, default=1000,
                        help="Base capital in USD (default: 1000)")
    args = parser.parse_args()
    run(ibkr_mode=args.ibkr, interval=args.interval, capital=args.capital)
