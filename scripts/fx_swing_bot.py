#!/usr/bin/env python3
"""
FX Swing Bot — Carry + Trend Strategy

Academic basis:
  - Carry trade: Burnside, Eichenbaum, Rebelo (2011) — ~4.6% ann, Sharpe 0.84
  - Trend following: Menkhoff et al. (2012, JFE) — ~4.5% ann, Sharpe 0.29-0.70
  - Combined carry + trend: QuantConnect — Sharpe 0.75-0.88
  - Carry and momentum have -31% correlation in crises → smooth combined equity curve

Strategy:
  - Trade G10 pairs where carry + trend align
  - 40% carry signal / 60% trend signal
  - Hold for days to weeks (not minutes)
  - Check once at 17:00 ET (daily bar close) + optional 08:00 ET check
  - VIX filter: reduce/skip carry when VIX > 25
  - Volatility targeting: scale position inversely with realized vol

Pairs: EUR/USD, USD/JPY, GBP/USD, USD/MXN (+ AUD/USD optional)
"""

import logging
import math
import os
import signal
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta, date
from pathlib import Path
from typing import Optional
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import yfinance as yf

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("fx_swing")

ROOT = Path(__file__).resolve().parent.parent
ET = ZoneInfo("America/New_York")

# ─── Rate Differentials (annualized %) ────────────────────────────────────────
# Fed 3.50-3.75%, ECB 2.00%, BOJ 0.75%, RBA 4.10%, BOE 4.50%
# Updated: March 2026. Review monthly.
POLICY_RATES = {
    "USD": 3.625,   # Fed midpoint
    "EUR": 2.00,    # ECB deposit
    "JPY": 0.75,    # BOJ
    "GBP": 4.50,    # BOE
    "AUD": 4.10,    # RBA
    "MXN": 7.00,    # Banxico — highest G20 carry target
}

# ─── Pair Definitions ─────────────────────────────────────────────────────────

@dataclass
class PairConfig:
    pair: str              # "EURUSD"
    base: str              # "EUR" — the currency you buy when going LONG
    quote: str             # "USD"
    yf_symbol: str         # "EURUSD=X"
    ib_symbol: str         # "EURUSD" for ib_insync.Forex()
    pip_size: float        # 0.0001 for EUR/USD, 0.01 for USD/JPY
    min_qty: int           # Minimum IBKR lot
    qty_decimals: int      # Rounding precision
    # Per-pair tuning overrides (None = use global defaults)
    carry_weight: Optional[float] = None
    trend_weight: Optional[float] = None
    entry_threshold: Optional[float] = None
    sl_atr_mult: Optional[float] = None
    trail_activate_atr: Optional[float] = None
    trail_distance_atr: Optional[float] = None
    max_hold_days: Optional[int] = None

    def get_carry_weight(self) -> float:
        return self.carry_weight if self.carry_weight is not None else CARRY_WEIGHT

    def get_trend_weight(self) -> float:
        return self.trend_weight if self.trend_weight is not None else TREND_WEIGHT

    def get_entry_threshold(self) -> float:
        return self.entry_threshold if self.entry_threshold is not None else ENTRY_THRESHOLD

    def get_sl_atr_mult(self) -> float:
        return self.sl_atr_mult if self.sl_atr_mult is not None else SL_ATR_MULT

    def get_trail_activate_atr(self) -> float:
        return self.trail_activate_atr if self.trail_activate_atr is not None else TRAIL_ACTIVATE_ATR

    def get_trail_distance_atr(self) -> float:
        return self.trail_distance_atr if self.trail_distance_atr is not None else TRAIL_DISTANCE_ATR

    def get_max_hold_days(self) -> int:
        return self.max_hold_days if self.max_hold_days is not None else MAX_HOLD_DAYS


PAIRS = {
    # Per-pair optimized parameters (grid search over 5yr data, 2700 combos each)
    "EURUSD": PairConfig("EURUSD", "EUR", "USD", "EURUSD=X", "EURUSD", 0.0001, 20000, 0,
        carry_weight=0.30, trend_weight=0.70, entry_threshold=0.35,
        sl_atr_mult=3.5, trail_activate_atr=2.0, trail_distance_atr=0.8,
        max_hold_days=30),
        # Sharpe=0.46 PnL=$+2,174 WR=60% — trend-dominant, higher entry bar
    "USDJPY": PairConfig("USDJPY", "USD", "JPY", "USDJPY=X", "USDJPY", 0.01, 20000, 0,
        carry_weight=0.40, trend_weight=0.60, entry_threshold=0.25,
        sl_atr_mult=3.5, trail_activate_atr=1.5, trail_distance_atr=1.5,
        max_hold_days=30),
        # Sharpe=1.02 PnL=$+8,687 WR=65% — star pair, carry+trend aligned
    "GBPUSD": PairConfig("GBPUSD", "GBP", "USD", "GBPUSD=X", "GBPUSD", 0.0001, 20000, 0,
        carry_weight=0.40, trend_weight=0.60, entry_threshold=0.30,
        sl_atr_mult=2.0, trail_activate_atr=2.0, trail_distance_atr=1.0,
        max_hold_days=45),
        # Sharpe=0.61 PnL=$+4,362 WR=60% — tight stops, long hold for trends
    "GBPJPY": PairConfig("GBPJPY", "GBP", "JPY", "GBPJPY=X", "GBPJPY", 0.01, 20000, 0,
        carry_weight=0.60, trend_weight=0.40, entry_threshold=0.20,
        sl_atr_mult=2.0, trail_activate_atr=2.0, trail_distance_atr=1.5,
        max_hold_days=45),
        # Sharpe=0.94 PnL=$+7,557 WR=62% — strong JPY carry cross, carry-dominant
    "AUDJPY": PairConfig("AUDJPY", "AUD", "JPY", "AUDJPY=X", "AUDJPY", 0.01, 20000, 0,
        carry_weight=0.30, trend_weight=0.70, entry_threshold=0.35,
        sl_atr_mult=3.5, trail_activate_atr=2.0, trail_distance_atr=0.8,
        max_hold_days=30),
        # Sharpe=0.75 PnL=$+6,123 WR=65% — trend-dominant JPY cross, wide stops
    "AUDUSD": PairConfig("AUDUSD", "AUD", "USD", "AUDUSD=X", "AUDUSD", 0.0001, 20000, 0),
        # Not in default portfolio — degrades Sharpe
    "USDMXN": PairConfig("USDMXN", "USD", "MXN", "USDMXN=X", "USDMXN", 0.0001, 20000, 0,
        carry_weight=0.30, trend_weight=0.70, entry_threshold=0.35,
        sl_atr_mult=3.5, trail_activate_atr=1.0, trail_distance_atr=1.0,
        max_hold_days=45),
        # Sharpe=0.74 PnL=$+5,073 WR=70% — EM carry, patient hold, early trail
}

# ─── Strategy Constants ───────────────────────────────────────────────────────

CARRY_WEIGHT = 0.40
TREND_WEIGHT = 0.60

# Trend: SMA crossover on daily bars
TREND_FAST_MA = 20       # 1-month
TREND_SLOW_MA = 50       # ~2.5 months
TREND_CONFIRM_MA = 200   # Long-term regime filter

# Entry threshold: combined signal must exceed this
ENTRY_THRESHOLD = 0.25   # -1 to +1 scale (lower = more trades)

# Risk management
VIX_CARRY_REDUCE = 25    # Halve carry weight when VIX > 25
VIX_NO_ENTRY = 35        # No new entries above this
MAX_POSITIONS = 6         # Across all pairs (EURUSD, USDJPY, GBPUSD, USDMXN, GBPJPY, AUDJPY)
RISK_PER_TRADE_PCT = 2.0  # 2% of account per trade
MAX_NOTIONAL_PCT = 120.0  # Allow up to 120% notional (FX margin is ~3%)
ATR_PERIOD = 14
SL_ATR_MULT = 2.5        # Stop loss: 2.5x daily ATR (wider to avoid noise stops)
TRAIL_ACTIVATE_ATR = 1.5  # Activate trailing stop at 1.5x ATR profit (earlier)
TRAIL_DISTANCE_ATR = 1.0  # Trail 1.0x ATR behind (tighter once activated)
MAX_HOLD_DAYS = 30        # Close after 30 trading days (let trends play out)
VOL_TARGET_ANN = 0.10     # Target 10% annualized vol for position sizing

# Check schedule (ET hours)
CHECK_HOUR_PRIMARY = 17    # After NY close (daily bar finalized)
CHECK_HOUR_SECONDARY = 8   # London open (optional re-check)

# ─── Data Models ──────────────────────────────────────────────────────────────

@dataclass
class PairSignal:
    pair: str
    timestamp: str
    price: float
    carry_signal: float       # -1 to +1
    trend_signal: float       # -1 to +1
    combined_signal: float    # weighted blend
    direction: str            # "LONG", "SHORT", "FLAT"
    atr: float
    sma_fast: float
    sma_slow: float
    sma_200: float
    vix: float
    carry_rate: float         # annualized carry in %
    realized_vol: float       # annualized realized vol


@dataclass
class SwingPosition:
    pair: str
    direction: str            # "LONG" or "SHORT"
    entry_price: float
    entry_date: str
    qty: float
    atr_at_entry: float
    stop_loss: float
    trailing_stop: Optional[float] = None
    high_water: Optional[float] = None  # Best price since entry
    entry_reason: str = ""
    exit_price: float = 0.0
    exit_date: str = ""
    exit_reason: str = ""
    pnl_dollar: float = 0.0


# ─── Signal Generation ────────────────────────────────────────────────────────

def fetch_daily_bars(yf_symbol: str, period: str = "1y") -> pd.DataFrame:
    """Fetch daily OHLC from yfinance."""
    try:
        tk = yf.Ticker(yf_symbol)
        df = tk.history(period=period, interval="1d")
        if df.empty:
            log.warning("No data for %s", yf_symbol)
        return df
    except Exception as e:
        log.error("yfinance fetch failed for %s: %s", yf_symbol, e)
        return pd.DataFrame()


def fetch_vix() -> float:
    """Get current VIX level."""
    try:
        vix_data = yf.Ticker("^VIX").history(period="5d", interval="1d")
        if not vix_data.empty:
            return float(vix_data["Close"].iloc[-1])
    except Exception as e:
        log.warning("VIX fetch failed: %s", e)
    return 20.0  # default assumption


def compute_carry_signal(pair_cfg: PairConfig) -> tuple[float, float]:
    """Compute carry signal based on interest rate differential.

    Returns (signal, carry_rate).
    signal: +1 = go long (base yields more), -1 = go short
    carry_rate: annualized rate differential in %
    """
    base_rate = POLICY_RATES.get(pair_cfg.base, 0)
    quote_rate = POLICY_RATES.get(pair_cfg.quote, 0)
    diff = base_rate - quote_rate  # positive = long earns carry

    # Normalize to -1/+1 with soft scaling
    # 3% diff → ~0.9 signal, 1% diff → ~0.46
    signal = math.tanh(diff / 2.0)
    return signal, diff


def compute_trend_signal(df: pd.DataFrame) -> tuple[float, float, float, float]:
    """Compute trend signal from daily bars.

    Returns (signal, sma_fast, sma_slow, sma_200).
    Uses SMA crossover strength + price position relative to 200-day.
    """
    if len(df) < TREND_CONFIRM_MA:
        return 0.0, 0.0, 0.0, 0.0

    closes = df["Close"]
    sma_fast = float(closes.rolling(TREND_FAST_MA).mean().iloc[-1])
    sma_slow = float(closes.rolling(TREND_SLOW_MA).mean().iloc[-1])
    sma_200 = float(closes.rolling(TREND_CONFIRM_MA).mean().iloc[-1])
    price = float(closes.iloc[-1])

    # Trend direction: fast MA vs slow MA
    if sma_slow == 0:
        return 0.0, sma_fast, sma_slow, sma_200

    # Normalized crossover distance (% terms)
    cross_pct = (sma_fast - sma_slow) / sma_slow

    # Regime: price vs 200-day MA
    regime_pct = (price - sma_200) / sma_200

    # Combine: cross strength + regime confirmation
    # cross_pct of 1% → ~0.46, 3% → ~0.9
    trend_raw = math.tanh(cross_pct * 50)  # scale up since FX moves are small

    # Dampen if price is on wrong side of 200-day
    if (trend_raw > 0 and price < sma_200) or (trend_raw < 0 and price > sma_200):
        trend_raw *= 0.5  # conflicting regime → halve conviction

    return trend_raw, sma_fast, sma_slow, sma_200


def compute_atr(df: pd.DataFrame, period: int = ATR_PERIOD) -> float:
    """Average True Range on daily bars."""
    if len(df) < period + 1:
        return 0.0
    high = df["High"]
    low = df["Low"]
    close = df["Close"].shift(1)
    tr = pd.concat([
        high - low,
        (high - close).abs(),
        (low - close).abs(),
    ], axis=1).max(axis=1)
    return float(tr.rolling(period).mean().iloc[-1])


def compute_realized_vol(df: pd.DataFrame, window: int = 20) -> float:
    """Annualized realized volatility from daily returns."""
    if len(df) < window + 1:
        return 0.0
    returns = df["Close"].pct_change().dropna().tail(window)
    return float(returns.std() * np.sqrt(252))


def generate_pair_signal(pair_cfg: PairConfig, vix: float) -> Optional[PairSignal]:
    """Generate combined carry + trend signal for one pair."""
    df = fetch_daily_bars(pair_cfg.yf_symbol)
    if df.empty or len(df) < TREND_CONFIRM_MA:
        log.warning("Insufficient data for %s (%d bars)", pair_cfg.pair, len(df))
        return None

    now_str = datetime.now(ET).strftime("%Y-%m-%d %H:%M")
    price = float(df["Close"].iloc[-1])
    atr = compute_atr(df)
    rvol = compute_realized_vol(df)

    carry_sig, carry_rate = compute_carry_signal(pair_cfg)
    trend_sig, sma_fast, sma_slow, sma_200 = compute_trend_signal(df)

    # Adjust carry weight when VIX elevated (use per-pair weights)
    cw = pair_cfg.get_carry_weight()
    tw = pair_cfg.get_trend_weight()
    if vix > VIX_CARRY_REDUCE:
        cw = cw * 0.5
        tw = 1.0 - cw
        log.info("  VIX=%.1f > %d — reducing carry weight to %.0f%%",
                 vix, VIX_CARRY_REDUCE, cw * 100)

    combined = carry_sig * cw + trend_sig * tw

    # Determine direction (use per-pair threshold)
    threshold = pair_cfg.get_entry_threshold()
    if combined > threshold:
        direction = "LONG"
    elif combined < -threshold:
        direction = "SHORT"
    else:
        direction = "FLAT"

    return PairSignal(
        pair=pair_cfg.pair, timestamp=now_str, price=price,
        carry_signal=carry_sig, trend_signal=trend_sig,
        combined_signal=combined, direction=direction,
        atr=atr, sma_fast=sma_fast, sma_slow=sma_slow, sma_200=sma_200,
        vix=vix, carry_rate=carry_rate, realized_vol=rvol,
    )


# ─── Position Sizing ─────────────────────────────────────────────────────────

def compute_position_size(
    account_size: float, price: float, atr: float,
    realized_vol: float, pair_cfg: PairConfig,
) -> int:
    """Size position using volatility targeting + ATR-based risk.

    Two constraints, take the smaller:
    1. Risk budget: risk 1% of account, stop at 2x ATR
    2. Vol target: scale so position contributes ~10% annualized vol
    """
    if atr <= 0 or price <= 0:
        return 0

    # Method 1: risk budget
    stop_distance = pair_cfg.get_sl_atr_mult() * atr
    risk_dollars = account_size * (RISK_PER_TRADE_PCT / 100)
    # For XXX/USD pairs, 1 unit move = $1 per unit of base
    # For USD/JPY, pip_value = lot_size / price (in USD terms)
    if pair_cfg.quote == "USD":
        dollar_per_unit = 1.0  # 1 EUR move = $1 per EUR held
    else:
        dollar_per_unit = 1.0 / price  # 1 JPY move on USDJPY

    qty_risk = risk_dollars / (stop_distance * dollar_per_unit)

    # Method 2: vol targeting
    # We want: (qty * dollar_per_unit * price * realized_vol) / account = VOL_TARGET
    if realized_vol > 0:
        qty_vol = (account_size * VOL_TARGET_ANN) / (realized_vol * price * dollar_per_unit)
    else:
        qty_vol = qty_risk

    # Method 3: max notional cap
    max_notional = account_size * (MAX_NOTIONAL_PCT / 100)
    qty_notional = max_notional / (price * dollar_per_unit)

    qty = min(qty_risk, qty_vol, qty_notional)
    qty = max(qty, pair_cfg.min_qty)
    qty = round(qty, pair_cfg.qty_decimals)

    return int(qty)


# ─── IBKR Client ──────────────────────────────────────────────────────────────

class SwingIBKRClient:
    """Thin IBKR wrapper for swing trading — simpler than scalp bot."""

    def __init__(self, mode: str):
        self.mode = mode
        self._ib = None
        self._contracts = {}  # pair -> qualified contract

    def connect(self):
        if self.mode == "dry-run":
            log.info("[DRY-RUN] Skipping IBKR connection")
            return
        import ib_insync as ib_mod
        port = 7497 if self.mode == "paper" else 7496
        client_id = int(os.environ.get("IBKR_SWING_CLIENT_ID", "20"))
        ib = ib_mod.IB()
        ib.connect("127.0.0.1", port, clientId=client_id, timeout=30)
        self._ib = ib
        log.info("Connected to IBKR (%s) port=%d clientId=%d", self.mode, port, client_id)

    def is_connected(self) -> bool:
        if self.mode == "dry-run":
            return True
        return self._ib is not None and self._ib.isConnected()

    def reconnect(self):
        log.warning("Forcing IBKR reconnect...")
        if self._ib:
            try:
                self._ib.disconnect()
            except Exception:
                pass
            self._ib = None
        self._contracts.clear()
        time.sleep(2)
        try:
            self.connect()
        except Exception as e:
            log.error("Reconnect failed: %s", e)

    def _ensure_connected(self):
        if not self.is_connected():
            self.reconnect()

    def _get_contract(self, pair_cfg: PairConfig):
        import ib_insync as ib_mod
        if pair_cfg.pair not in self._contracts:
            contract = ib_mod.Forex(pair_cfg.ib_symbol)
            self._ib.qualifyContracts(contract)
            self._contracts[pair_cfg.pair] = contract
        return self._contracts[pair_cfg.pair]

    def get_positions(self) -> dict[str, tuple[float, float]]:
        """Returns {pair: (qty, avg_cost)} for all FX positions."""
        if self.mode == "dry-run" or not self._ib:
            return {}
        self._ensure_connected()
        result = {}
        try:
            for p in self._ib.positions():
                sym = p.contract.localSymbol  # "EUR.USD"
                qty = float(p.position)
                avg = float(p.avgCost)
                if qty != 0 and p.contract.secType == "CASH":
                    # Convert "EUR.USD" → "EURUSD"
                    pair = sym.replace(".", "")
                    result[pair] = (qty, avg)
        except Exception as e:
            log.warning("get_positions failed: %s", e)
        return result

    def get_price(self, pair_cfg: PairConfig) -> Optional[float]:
        if self.mode == "dry-run" or not self._ib:
            return None
        self._ensure_connected()
        try:
            contract = self._get_contract(pair_cfg)
            [ticker] = self._ib.reqTickers(contract)
            self._ib.sleep(0.5)
            [ticker] = self._ib.reqTickers(contract)
            bid = float(ticker.bid) if ticker.bid and ticker.bid > 0 else 0
            ask = float(ticker.ask) if ticker.ask and ticker.ask > 0 else 0
            if bid > 0 and ask > 0:
                return (bid + ask) / 2
            return float(ticker.last) if ticker.last and ticker.last > 0 else None
        except Exception as e:
            log.warning("get_price %s failed: %s", pair_cfg.pair, e)
            return None

    def place_order(self, pair_cfg: PairConfig, action: str, qty: int,
                    limit: Optional[float] = None) -> dict:
        """Place a forex order. action: 'BUY' or 'SELL'."""
        import ib_insync as ib_mod
        log.info(">>> %s %d %s @ %s",
                 action, qty, pair_cfg.pair,
                 f"limit {limit:.5f}" if limit else "MKT")

        if self.mode == "dry-run":
            # Use yfinance price for realistic dry-run
            if not limit:
                try:
                    df = yf.Ticker(pair_cfg.yf_symbol).history(period="1d")
                    limit = float(df["Close"].iloc[-1]) if not df.empty else 1.0
                except Exception:
                    limit = 1.0
            log.info("[DRY-RUN] Simulated fill @ %.5f", limit)
            return {"success": True, "fill_price": limit, "qty": qty}

        self._ensure_connected()
        try:
            contract = self._get_contract(pair_cfg)
            if limit:
                order = ib_mod.LimitOrder(action=action, totalQuantity=qty,
                                          lmtPrice=limit, tif="GTC")
            else:
                order = ib_mod.MarketOrder(action=action, totalQuantity=qty, tif="DAY")

            trade = self._ib.placeOrder(contract, order)
            # Wait up to 30 seconds for fill
            for _ in range(60):
                self._ib.sleep(0.5)
                if trade.isDone():
                    break

            if trade.orderStatus.status == "Filled":
                fp = float(trade.orderStatus.avgFillPrice)
                log.info("  FILLED %s %d %s @ %.5f", action, qty, pair_cfg.pair, fp)
                return {"success": True, "fill_price": fp, "qty": qty}
            else:
                log.warning("  Order not filled: %s — trying market order",
                            trade.orderStatus.status)
                try:
                    self._ib.cancelOrder(order)
                    self._ib.sleep(1)
                except Exception:
                    pass
                mkt = ib_mod.MarketOrder(action=action, totalQuantity=qty, tif="DAY")
                trade2 = self._ib.placeOrder(contract, mkt)
                for _ in range(30):
                    self._ib.sleep(0.5)
                    if trade2.isDone():
                        break
                if trade2.orderStatus.status == "Filled":
                    fp = float(trade2.orderStatus.avgFillPrice)
                    log.info("  MKT FILLED %s %d %s @ %.5f", action, qty, pair_cfg.pair, fp)
                    return {"success": True, "fill_price": fp, "qty": qty}
                log.error("  Market order also failed: %s", trade2.orderStatus.status)
                return {"success": False, "fill_price": 0, "qty": 0}
        except Exception as e:
            log.error("Order failed: %s", e)
            return {"success": False, "fill_price": 0, "qty": 0}

    def disconnect(self):
        if self._ib:
            try:
                self._ib.disconnect()
            except Exception:
                pass
            self._ib = None


# ─── Position Manager ─────────────────────────────────────────────────────────

class PositionManager:
    """Manages open swing positions with stop/trail logic."""

    def __init__(self, client: SwingIBKRClient, log_dir: Path, account_size: float):
        self.client = client
        self.account_size = account_size
        self.positions: list[SwingPosition] = []
        self.closed_trades: list[SwingPosition] = []
        self.log_dir = log_dir
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.trade_log = log_dir / f"fx_swing_trades_{date.today():%Y%m%d}.csv"
        self._init_trade_log()

    def _init_trade_log(self):
        if not self.trade_log.exists():
            with open(self.trade_log, "w") as f:
                f.write("entry_date,exit_date,pair,direction,qty,entry_price,"
                        "exit_price,pnl_dollar,exit_reason,hold_days,entry_reason\n")

    def _log_trade(self, pos: SwingPosition):
        hold_days = 0
        if pos.entry_date and pos.exit_date:
            try:
                d1 = datetime.strptime(pos.entry_date[:10], "%Y-%m-%d")
                d2 = datetime.strptime(pos.exit_date[:10], "%Y-%m-%d")
                hold_days = (d2 - d1).days
            except Exception:
                pass
        with open(self.trade_log, "a") as f:
            f.write(f"{pos.entry_date},{pos.exit_date},{pos.pair},{pos.direction},"
                    f"{pos.qty},{pos.entry_price:.5f},{pos.exit_price:.5f},"
                    f"{pos.pnl_dollar:.2f},{pos.exit_reason},{hold_days},"
                    f"{pos.entry_reason}\n")

    def has_position(self, pair: str) -> bool:
        return any(p.pair == pair for p in self.positions)

    def open_count(self) -> int:
        return len(self.positions)

    def enter(self, pair_cfg: PairConfig, sig: PairSignal) -> bool:
        """Open a new swing position."""
        if self.has_position(pair_cfg.pair):
            log.info("  Already positioned in %s, skipping", pair_cfg.pair)
            return False
        if self.open_count() >= MAX_POSITIONS:
            log.info("  Max positions (%d) reached, skipping %s", MAX_POSITIONS, pair_cfg.pair)
            return False

        qty = compute_position_size(
            self.account_size, sig.price, sig.atr,
            sig.realized_vol, pair_cfg,
        )
        if qty <= 0:
            return False

        action = "BUY" if sig.direction == "LONG" else "SELL"
        result = self.client.place_order(pair_cfg, action, qty)
        if not result["success"]:
            return False

        fill = result["fill_price"]
        atr = sig.atr

        sl_mult = pair_cfg.get_sl_atr_mult()
        if sig.direction == "LONG":
            sl = fill - sl_mult * atr
        else:
            sl = fill + sl_mult * atr

        reason = (f"carry={sig.carry_signal:+.2f} trend={sig.trend_signal:+.2f} "
                  f"combined={sig.combined_signal:+.2f} rate={sig.carry_rate:+.1f}%")

        pos = SwingPosition(
            pair=pair_cfg.pair, direction=sig.direction,
            entry_price=fill,
            entry_date=datetime.now(ET).strftime("%Y-%m-%d %H:%M"),
            qty=qty, atr_at_entry=atr, stop_loss=sl,
            high_water=fill, entry_reason=reason,
        )
        self.positions.append(pos)
        log.info("  OPENED %s %s %d @ %.5f | SL=%.5f | %s",
                 sig.direction, pair_cfg.pair, qty, fill, sl, reason)
        return True

    def check_exits(self, pair_prices: dict[str, float]):
        """Check stop loss, trailing stop, and time exits for all positions."""
        now = datetime.now(ET)
        to_close = []

        for pos in self.positions:
            pair_cfg = PAIRS.get(pos.pair)
            if not pair_cfg:
                continue

            price = pair_prices.get(pos.pair)
            if not price:
                # Try IBKR
                price = self.client.get_price(pair_cfg)
            if not price:
                continue

            is_long = pos.direction == "LONG"
            pnl_price = (price - pos.entry_price) if is_long else (pos.entry_price - price)

            # Update high water mark
            if is_long:
                if pos.high_water is None or price > pos.high_water:
                    pos.high_water = price
            else:
                if pos.high_water is None or price < pos.high_water:
                    pos.high_water = price

            # 1. Stop loss
            if is_long and price <= pos.stop_loss:
                to_close.append((pos, price, "STOP_LOSS"))
                continue
            if not is_long and price >= pos.stop_loss:
                to_close.append((pos, price, "STOP_LOSS"))
                continue

            # 2. Trailing stop (per-pair params)
            atr = pos.atr_at_entry
            trail_activate = pair_cfg.get_trail_activate_atr()
            trail_dist = pair_cfg.get_trail_distance_atr()
            activate_dist = trail_activate * atr
            if pnl_price >= activate_dist:
                if is_long:
                    new_trail = pos.high_water - trail_dist * atr
                    if pos.trailing_stop is None or new_trail > pos.trailing_stop:
                        pos.trailing_stop = new_trail
                    if price <= pos.trailing_stop:
                        to_close.append((pos, price, "TRAIL_STOP"))
                        continue
                else:
                    new_trail = pos.high_water + trail_dist * atr
                    if pos.trailing_stop is None or new_trail < pos.trailing_stop:
                        pos.trailing_stop = new_trail
                    if price >= pos.trailing_stop:
                        to_close.append((pos, price, "TRAIL_STOP"))
                        continue

            # 3. Time exit (per-pair max hold)
            max_hold = pair_cfg.get_max_hold_days()
            try:
                entry_dt = datetime.strptime(pos.entry_date[:10], "%Y-%m-%d")
                if (now - entry_dt.replace(tzinfo=ET)).days >= max_hold:
                    to_close.append((pos, price, "TIME_EXIT"))
                    continue
            except Exception:
                pass

        # Execute closes
        for pos, price, reason in to_close:
            self._close_position(pos, price, reason)

    def check_signal_exits(self, signals: dict[str, PairSignal]):
        """Close positions where signal has flipped against us."""
        to_close = []
        for pos in self.positions:
            sig = signals.get(pos.pair)
            if not sig:
                continue
            pair_cfg = PAIRS.get(pos.pair)
            threshold = pair_cfg.get_entry_threshold() if pair_cfg else ENTRY_THRESHOLD
            # If signal flips to opposite direction, close
            if pos.direction == "LONG" and sig.combined_signal < -threshold:
                to_close.append((pos, sig.price, "SIGNAL_FLIP"))
            elif pos.direction == "SHORT" and sig.combined_signal > threshold:
                to_close.append((pos, sig.price, "SIGNAL_FLIP"))

        for pos, price, reason in to_close:
            self._close_position(pos, price, reason)

    def _close_position(self, pos: SwingPosition, price: float, reason: str):
        pair_cfg = PAIRS.get(pos.pair)
        if not pair_cfg:
            return

        action = "SELL" if pos.direction == "LONG" else "BUY"
        result = self.client.place_order(pair_cfg, action, int(pos.qty))

        exit_price = result["fill_price"] if result["success"] else price
        is_long = pos.direction == "LONG"

        if pair_cfg.quote == "USD":
            pnl = ((exit_price - pos.entry_price) if is_long
                    else (pos.entry_price - exit_price)) * pos.qty
        else:
            # USD/JPY: PnL in JPY, convert to USD
            pnl_foreign = ((exit_price - pos.entry_price) if is_long
                           else (pos.entry_price - exit_price)) * pos.qty
            pnl = pnl_foreign / exit_price if exit_price > 0 else 0

        pos.exit_price = exit_price
        pos.exit_date = datetime.now(ET).strftime("%Y-%m-%d %H:%M")
        pos.exit_reason = reason
        pos.pnl_dollar = round(pnl, 2)

        self.positions.remove(pos)
        self.closed_trades.append(pos)
        self._log_trade(pos)

        log.info("  CLOSED %s %s %d @ %.5f | PnL: $%+.2f | %s",
                 pos.direction, pos.pair, int(pos.qty), exit_price, pnl, reason)

    def summary(self) -> str:
        total_pnl = sum(t.pnl_dollar for t in self.closed_trades)
        wins = sum(1 for t in self.closed_trades if t.pnl_dollar > 0)
        losses = sum(1 for t in self.closed_trades if t.pnl_dollar <= 0)
        n = len(self.closed_trades)
        wr = (wins / n * 100) if n > 0 else 0
        return (f"Trades: {n} | W/L: {wins}/{losses} ({wr:.0f}%) | "
                f"PnL: ${total_pnl:+,.2f} | Open: {self.open_count()}")


# ─── Signal Logger ────────────────────────────────────────────────────────────

class SignalLogger:
    def __init__(self, log_dir: Path):
        self.path = log_dir / f"fx_swing_signals_{date.today():%Y%m%d}.csv"
        if not self.path.exists():
            with open(self.path, "w") as f:
                f.write("timestamp,pair,price,carry_sig,trend_sig,combined,"
                        "direction,atr,rvol,vix,carry_rate,sma20,sma50,sma200\n")

    def log(self, sig: PairSignal):
        with open(self.path, "a") as f:
            f.write(f"{sig.timestamp},{sig.pair},{sig.price:.5f},"
                    f"{sig.carry_signal:.3f},{sig.trend_signal:.3f},"
                    f"{sig.combined_signal:.3f},{sig.direction},{sig.atr:.5f},"
                    f"{sig.realized_vol:.4f},{sig.vix:.1f},{sig.carry_rate:.2f},"
                    f"{sig.sma_fast:.5f},{sig.sma_slow:.5f},{sig.sma_200:.5f}\n")


# ─── State Persistence ────────────────────────────────────────────────────────

import json

STATE_FILE = ROOT / "data" / "fx_swing_state.json"

def save_state(positions: list[SwingPosition]):
    """Persist open positions to disk so we can resume after restart."""
    data = []
    for p in positions:
        data.append({
            "pair": p.pair, "direction": p.direction,
            "entry_price": p.entry_price, "entry_date": p.entry_date,
            "qty": p.qty, "atr_at_entry": p.atr_at_entry,
            "stop_loss": p.stop_loss,
            "trailing_stop": p.trailing_stop,
            "high_water": p.high_water,
            "entry_reason": p.entry_reason,
        })
    STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(STATE_FILE, "w") as f:
        json.dump(data, f, indent=2)


def load_state() -> list[SwingPosition]:
    """Load persisted positions from disk."""
    if not STATE_FILE.exists():
        return []
    try:
        with open(STATE_FILE) as f:
            data = json.load(f)
        positions = []
        for d in data:
            positions.append(SwingPosition(
                pair=d["pair"], direction=d["direction"],
                entry_price=d["entry_price"], entry_date=d["entry_date"],
                qty=d["qty"], atr_at_entry=d["atr_at_entry"],
                stop_loss=d["stop_loss"],
                trailing_stop=d.get("trailing_stop"),
                high_water=d.get("high_water"),
                entry_reason=d.get("entry_reason", ""),
            ))
        return positions
    except Exception as e:
        log.warning("Failed to load state: %s", e)
        return []


# ─── Main Loop ────────────────────────────────────────────────────────────────

def run_check(mgr: PositionManager, sig_logger: SignalLogger,
              active_pairs: list[str]):
    """Run one daily check cycle: generate signals, manage exits, enter new."""
    log.info("=" * 60)
    log.info("DAILY CHECK — %s", datetime.now(ET).strftime("%Y-%m-%d %H:%M ET"))

    vix = fetch_vix()
    log.info("VIX: %.1f", vix)

    if vix > VIX_NO_ENTRY:
        log.info("VIX %.1f > %d — no new entries, checking exits only", vix, VIX_NO_ENTRY)

    # Generate signals for all pairs
    signals: dict[str, PairSignal] = {}
    pair_prices: dict[str, float] = {}

    for pair_name in active_pairs:
        pair_cfg = PAIRS[pair_name]
        sig = generate_pair_signal(pair_cfg, vix)
        if sig:
            signals[pair_name] = sig
            pair_prices[pair_name] = sig.price
            sig_logger.log(sig)
            log.info("  %s: %.5f | carry=%+.2f trend=%+.2f → %+.2f %s | "
                     "ATR=%.5f rvol=%.1f%%",
                     pair_name, sig.price, sig.carry_signal, sig.trend_signal,
                     sig.combined_signal, sig.direction, sig.atr,
                     sig.realized_vol * 100)

    # Check exits on existing positions (stop/trail/time)
    mgr.check_exits(pair_prices)

    # Check signal-based exits (regime flip)
    mgr.check_signal_exits(signals)

    # Enter new positions where signal is strong
    if vix <= VIX_NO_ENTRY:
        for pair_name in active_pairs:
            sig = signals.get(pair_name)
            if not sig or sig.direction == "FLAT":
                continue
            if mgr.has_position(pair_name):
                continue
            mgr.enter(PAIRS[pair_name], sig)

    # Persist state
    save_state(mgr.positions)

    log.info("  %s", mgr.summary())
    if mgr.positions:
        for p in mgr.positions:
            price = pair_prices.get(p.pair, p.entry_price)
            is_long = p.direction == "LONG"
            unr = ((price - p.entry_price) if is_long
                   else (p.entry_price - price)) * p.qty
            if PAIRS[p.pair].quote != "USD" and price > 0:
                unr = unr / price
            log.info("    %s %s %d @ %.5f → %.5f | unrealized: $%s | SL=%.5f%s",
                     p.direction, p.pair, int(p.qty), p.entry_price, price,
                     f"{unr:+,.2f}",
                     p.stop_loss,
                     f" trail={p.trailing_stop:.5f}" if p.trailing_stop else "")
    log.info("=" * 60)


def run(pairs: list[str], ibkr_mode: str, account_size: float, once: bool = False):
    log_dir = ROOT / "logs" / "fx_swing"
    log_dir.mkdir(parents=True, exist_ok=True)

    # Set up file logging
    fh = logging.FileHandler(
        log_dir / f"fx_swing_{date.today():%Y%m%d}.log"
    )
    fh.setFormatter(logging.Formatter(
        "%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S"
    ))
    log.addHandler(fh)

    client = SwingIBKRClient(mode=ibkr_mode)
    if ibkr_mode != "dry-run":
        client.connect()

    sig_logger = SignalLogger(log_dir)
    mgr = PositionManager(client, log_dir, account_size)

    # Restore open positions from previous run
    restored = load_state()
    if restored:
        mgr.positions = restored
        log.info("Restored %d open position(s) from previous session:", len(restored))
        for p in restored:
            log.info("  %s %s %d @ %.5f (SL=%.5f)", p.direction, p.pair,
                     int(p.qty), p.entry_price, p.stop_loss)
    else:
        log.info("No open positions from previous session")

    log.info("FX Swing Bot — %s mode — $%s account", ibkr_mode.upper(),
             f"{account_size:,.0f}")
    log.info("Pairs: %s", ", ".join(pairs))
    log.info("Global defaults: %.0f%% carry + %.0f%% trend | threshold: %.2f",
             CARRY_WEIGHT * 100, TREND_WEIGHT * 100, ENTRY_THRESHOLD)
    log.info("Risk: %.1f%% per trade | max %d positions",
             RISK_PER_TRADE_PCT, MAX_POSITIONS)
    for p in pairs:
        cfg = PAIRS.get(p)
        if cfg:
            log.info("  %s: carry=%.0f%% trend=%.0f%% thr=%.2f SL=%.1fxATR "
                     "trail=%.1f/%.1f hold=%dd",
                     p, cfg.get_carry_weight() * 100, cfg.get_trend_weight() * 100,
                     cfg.get_entry_threshold(), cfg.get_sl_atr_mult(),
                     cfg.get_trail_activate_atr(), cfg.get_trail_distance_atr(),
                     cfg.get_max_hold_days())

    if once:
        run_check(mgr, sig_logger, pairs)
        return

    # Continuous mode: check at scheduled times
    running = [True]
    def _stop(signum, frame):
        log.info("Shutting down...")
        running[0] = False
    signal.signal(signal.SIGINT, _stop)
    signal.signal(signal.SIGTERM, _stop)

    last_check_date = None

    while running[0]:
        now = datetime.now(ET)
        today = now.date()
        hour = now.hour

        # Skip weekends (Sat=5, Sun=6)
        if now.weekday() >= 5:
            time.sleep(300)
            continue

        # Primary check: 17:00 ET
        should_check = False
        if hour == CHECK_HOUR_PRIMARY and last_check_date != (today, "primary"):
            should_check = True
            last_check_date = (today, "primary")
        # Secondary check: 08:00 ET (only if we have open positions)
        elif hour == CHECK_HOUR_SECONDARY and mgr.open_count() > 0 and last_check_date != (today, "secondary"):
            should_check = True
            last_check_date = (today, "secondary")

        if should_check:
            try:
                run_check(mgr, sig_logger, pairs)
            except Exception as e:
                log.error("Check failed: %s", e, exc_info=True)

        # Sleep 5 minutes between polls
        time.sleep(300)

    # Graceful shutdown: save state, don't close positions
    save_state(mgr.positions)
    log.info("State saved. %d position(s) will resume on next start.", mgr.open_count())
    if client.is_connected():
        client.disconnect()


# ─── CLI ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="FX Swing Bot — Carry + Trend")
    parser.add_argument("--pairs", nargs="+", default=["EURUSD", "USDJPY", "GBPUSD", "USDMXN"],
                        choices=list(PAIRS.keys()),
                        help="Pairs to trade (default: EURUSD USDJPY GBPUSD USDMXN)")
    parser.add_argument("--ibkr", choices=["paper", "live", "dry-run"],
                        default="dry-run", help="IBKR mode")
    parser.add_argument("--account-size", type=float, default=200_000,
                        help="Account size in USD (default: 200K — box spread capital)")
    parser.add_argument("--once", action="store_true",
                        help="Run one check cycle and exit")
    args = parser.parse_args()
    run(args.pairs, args.ibkr, args.account_size, args.once)
