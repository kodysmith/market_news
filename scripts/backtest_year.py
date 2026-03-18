#!/usr/bin/env python3
"""
Year-long backtest of the 0DTE scalp strategy with scale-on-trail.

Uses hourly bars (1 year) + 5-min bars (60 days) from yfinance.
GEX regime is proxied from VIX level since historical GEX data is unavailable.

Usage:
    python scripts/backtest_year.py [--months 12] [--account pm]
"""

from __future__ import annotations

import argparse
import math
import sys
from datetime import datetime, timedelta, time as dtime
from pathlib import Path
from collections import defaultdict
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import yfinance as yf
from scipy.stats import norm

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from scripts.scalp_accounts import ACCOUNTS

ET = ZoneInfo("America/New_York")

# ─── Black-Scholes ───────────────────────────────────────────────────────────

R = 0.05

def _d1(S, K, T, sigma):
    if T <= 0 or sigma <= 0:
        return 0.0
    return (math.log(S / K) + (R + 0.5 * sigma**2) * T) / (sigma * math.sqrt(T))

def bs_call(S, K, T, sigma):
    if T <= 1e-10: return max(S - K, 0.0)
    d1 = _d1(S, K, T, sigma)
    d2 = d1 - sigma * math.sqrt(T)
    return S * norm.cdf(d1) - K * math.exp(-R * T) * norm.cdf(d2)

def bs_put(S, K, T, sigma):
    if T <= 1e-10: return max(K - S, 0.0)
    d1 = _d1(S, K, T, sigma)
    d2 = d1 - sigma * math.sqrt(T)
    return K * math.exp(-R * T) * norm.cdf(-d2) - S * norm.cdf(-d1)

def atm_strike(spot, inc=5.0):
    return round(spot / inc) * inc

def tte_years_from_dt(dt: datetime) -> float:
    close = dt.replace(hour=16, minute=0, second=0)
    mins = max(1, (close - dt).total_seconds() / 60)
    return mins / (365 * 24 * 60)

def tte_minutes_from_dt(dt: datetime) -> float:
    close = dt.replace(hour=16, minute=0, second=0)
    return max(0, (close - dt).total_seconds() / 60)

# ─── Indicators ──────────────────────────────────────────────────────────────

def compute_ema(series, span):
    return series.ewm(span=span, adjust=False).mean()

def compute_rsi(series, period=14):
    delta = series.diff()
    gains = delta.where(delta > 0, 0.0)
    losses = -delta.where(delta < 0, 0.0)
    avg_gain = gains.rolling(period).mean()
    avg_loss = losses.rolling(period).mean()
    rs = avg_gain / avg_loss.replace(0, 1e-9)
    return 100.0 - (100.0 / (1.0 + rs))

def compute_atr(df, period=14):
    high = df["High"]
    low = df["Low"]
    prev_close = df["Close"].shift(1)
    tr = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low - prev_close).abs(),
    ], axis=1).max(axis=1)
    return tr.rolling(period).mean()

def compute_vwap(df):
    typical = (df["High"] + df["Low"] + df["Close"]) / 3.0
    cum_vol = df["Volume"].cumsum()
    cum_tp_vol = (typical * df["Volume"]).cumsum()
    return cum_tp_vol / cum_vol.replace(0, 1)

# ─── Signal Generation (same logic as spx_scalp_bot.py) ─────────────────────

def generate_signal(price, vix, regime, ema9, ema21, rsi, atr, price_vs_vwap,
                    flip_line=0, call_wall=0, put_wall=0):
    """Generate signal and confidence. Returns (signal, reason, confidence)."""
    dist_to_flip = price - flip_line if flip_line else 0.0

    if vix > 30:
        return "FLAT", "VIX too high", 0.0
    if vix < 13:
        return "FLAT", "VIX too low", 0.0

    if regime == "POSITIVE":
        if put_wall and price < put_wall + 3 and rsi < 30 and price_vs_vwap < 0:
            return "LONG", "POS: put wall + RSI<30", 0.8
        elif call_wall and price > call_wall - 3 and rsi > 70 and price_vs_vwap > 0:
            return "SHORT", "POS: call wall + RSI>70", 0.8
        elif dist_to_flip < -5 and rsi < 35 and price_vs_vwap < 0 and ema9 < ema21:
            return "LONG", "POS: below flip + oversold", 0.7
        elif dist_to_flip > 5 and rsi > 65 and price_vs_vwap > 0 and ema9 > ema21:
            return "SHORT", "POS: above flip + overbought", 0.7
        elif rsi < 25 and price_vs_vwap < 0 and ema9 < ema21:
            return "LONG", "POS: RSI<25 extreme", 0.65
        elif rsi > 75 and price_vs_vwap > 0 and ema9 > ema21:
            return "SHORT", "POS: RSI>75 extreme", 0.65
        # Lower-confidence POS signals (match what bot generates at 0.60)
        elif rsi > 60 and price_vs_vwap > 0 and dist_to_flip > 0:
            return "SHORT", "POS: above flip + VWAP + RSI>60", 0.60
        elif rsi < 40 and price_vs_vwap < 0 and dist_to_flip < 0:
            return "LONG", "POS: below flip + VWAP + RSI<40", 0.60
        return "FLAT", "POS: no setup", 0.0

    elif regime == "NEGATIVE":
        if put_wall and price < put_wall - 2 and ema9 < ema21 and price_vs_vwap < 0:
            return "SHORT", "NEG: broke put wall + trend", 0.8
        elif call_wall and price > call_wall + 2 and ema9 > ema21 and price_vs_vwap > 0:
            return "LONG", "NEG: broke call wall + trend", 0.8
        elif ema9 > ema21 and price_vs_vwap > 0 and 45 < rsi < 70:
            return "LONG", "NEG: uptrend (EMA+VWAP+RSI)", 0.7
        elif ema9 < ema21 and price_vs_vwap < 0 and 30 < rsi < 55:
            return "SHORT", "NEG: downtrend (EMA+VWAP+RSI)", 0.7
        # Lower confidence momentum
        elif ema9 > ema21 and price_vs_vwap > 0:
            return "LONG", "NEG: uptrend momentum", 0.60
        elif ema9 < ema21 and price_vs_vwap < 0:
            return "SHORT", "NEG: downtrend momentum", 0.60
        return "FLAT", "NEG: no setup", 0.0

    return "FLAT", "Unknown regime", 0.0


# ─── GEX Regime Proxy ────────────────────────────────────────────────────────

def estimate_gex_regime(vix: float) -> str:
    """Proxy GEX regime from VIX level.
    Low VIX = dealers long gamma = mean-reversion (POSITIVE)
    High VIX = dealers short gamma = trending (NEGATIVE)
    """
    if vix < 20:
        return "POSITIVE"
    else:
        return "NEGATIVE"


# ─── 0DTE Trade Simulator ───────────────────────────────────────────────────

class TradeSim:
    """Simulates 0DTE option trades for a single account profile."""

    def __init__(self, profile: dict):
        self.p = profile
        self.trades: list[dict] = []
        self.open = None
        self._last_exit_dt = None
        self._peak_pnl = 0.0
        self._scaled = False
        self._today_trades = 0
        self._today_date = None
        self._daily_pnl = 0.0
        self._total_pnl = 0.0
        self._paused = False

    def _in_cooldown(self, dt):
        if not self._last_exit_dt:
            return False
        return (dt - self._last_exit_dt).total_seconds() < self.p.get("cooldown_sec", 300)

    def _opt_price(self, spot: float) -> float:
        """Return the option-underlying price (XSP = SPX/10, SPX = SPX)."""
        if self.p.get("ticker", "SPX") == "XSP":
            return spot / 10.0
        return spot

    def _strike_inc(self) -> float:
        """Strike increment: 5 for SPX, 1 for XSP."""
        return 1.0 if self.p.get("ticker", "SPX") == "XSP" else 5.0

    def tick(self, dt: datetime, price: float, vix: float, signal: str,
             confidence: float, reason: str, atr: float):
        """Process one bar. Price is always SPX-scale; converted for XSP internally."""
        if self._paused:
            return

        today = dt.date()
        if today != self._today_date:
            self._today_date = today
            self._today_trades = 0
            self._daily_pnl = 0.0

        if self.open:
            self._check_exit(dt, price, vix, signal)
            return

        # Entry filters
        if signal not in ("LONG", "SHORT"):
            return
        if confidence < self.p.get("min_confidence", 0.60):
            return
        if self._in_cooldown(dt):
            return
        max_daily = self.p.get("max_trades_per_day", 6)
        if self._today_trades >= max_daily:
            return
        if self._daily_pnl <= -self.p.get("max_daily_loss", 3000):
            return

        # Time filters
        t = dt.time()
        if t < dtime(9, 35) or t > dtime(15, 55):
            return
        if tte_minutes_from_dt(dt) < self.p.get("min_tte_minutes", 30):
            return

        # Price option (scale for XSP)
        opt_spot = self._opt_price(price)
        K = atm_strike(opt_spot, self._strike_inc())
        T = tte_years_from_dt(dt)
        sigma = vix / 100.0
        if sigma < 0.05:
            return

        if signal == "LONG":
            opt_price = bs_call(opt_spot, K, T, sigma)
            opt_type = "CALL"
        else:
            opt_price = bs_put(opt_spot, K, T, sigma)
            opt_type = "PUT"

        min_prem = self.p.get("min_premium", 2.0)
        if opt_price < min_prem:
            return
        mult = self.p.get("multiplier", 100)
        if opt_price * mult > self.p.get("max_premium", 1500):
            return

        self.open = {
            "dt": dt, "spot_entry": opt_spot, "strike": K,
            "opt_type": opt_type, "opt_entry": opt_price,
            "dir": signal, "vix": vix,
            "contracts": self.p.get("contracts", 1),
            "reason": reason,
        }
        self._peak_pnl = 0.0
        self._scaled = False
        self._today_trades += 1

    def _check_exit(self, dt, price, vix, signal):
        o = self.open
        mult = self.p.get("multiplier", 100)
        T = tte_years_from_dt(dt)
        sigma = vix / 100.0 if vix > 0 else o["vix"] / 100.0
        opt_spot = self._opt_price(price)

        if o["opt_type"] == "CALL":
            current = bs_call(opt_spot, o["strike"], T, sigma)
        else:
            current = bs_put(opt_spot, o["strike"], T, sigma)

        pnl_per = (current - o["opt_entry"]) * mult
        self._peak_pnl = max(self._peak_pnl, pnl_per)

        # Scale-on-trail
        if (self.p.get("scale_on_trail") and not self._scaled
                and pnl_per >= o["opt_entry"] * mult
                * self.p.get("scale_trigger_pct", 20) / 100):
            o["contracts"] += self.p.get("scale_add_contracts", 1)
            self._scaled = True

        contracts = o["contracts"]
        hold_sec = (dt - o["dt"]).total_seconds()
        is_long = o["dir"] == "LONG"

        tp_pct = self.p.get("option_tp_pct", 50)
        sl_pct = self.p.get("option_sl_pct", 35)
        trail_pct = self.p.get("option_trail_pct", 25)
        tp_thresh = o["opt_entry"] * mult * tp_pct / 100
        sl_thresh = o["opt_entry"] * mult * sl_pct / 100

        reason = ""
        if pnl_per >= tp_thresh:
            reason = "TAKE_PROFIT"
        elif pnl_per <= -sl_thresh:
            reason = "STOP_LOSS"
        elif (self._peak_pnl > o["opt_entry"] * mult * 0.20
              and pnl_per < self._peak_pnl * (1 - trail_pct / 100)):
            reason = "TRAIL_STOP"
        elif hold_sec >= self.p.get("max_hold_minutes", 30) * 60:
            reason = "TIME_EXIT"
        elif (is_long and signal == "SHORT") or (not is_long and signal == "LONG"):
            reason = "SIGNAL_REVERSAL"

        if not reason:
            return

        comm = 2 * self.p.get("commission_per_side", 0.65) * contracts
        gross = pnl_per * contracts
        net = round(gross - comm, 2)

        self.trades.append({
            "date": o["dt"].strftime("%Y-%m-%d"),
            "entry_time": o["dt"].strftime("%H:%M"),
            "exit_time": dt.strftime("%H:%M"),
            "dir": o["dir"], "opt_type": o["opt_type"],
            "strike": o["strike"],
            "spot_entry": round(o["spot_entry"], 2),
            "spot_exit": round(opt_spot, 2),
            "opt_entry": round(o["opt_entry"], 2),
            "opt_exit": round(current, 2),
            "contracts": contracts,
            "gross": round(gross, 2), "net": net,
            "reason": reason,
            "hold_min": round(hold_sec / 60, 1),
            "vix": round(o["vix"], 1),
        })

        self._daily_pnl += net
        self._total_pnl += net
        if self._total_pnl <= -self.p.get("max_drawdown", 10000):
            self._paused = True
        self._last_exit_dt = dt
        self.open = None

    def force_close_day(self, dt, price, vix):
        """Force-close any open trade at end of day."""
        if not self.open:
            return
        # Trigger time exit
        self._check_exit(dt, price, vix, "FLAT")
        if self.open:
            # Force it
            o = self.open
            mult = self.p.get("multiplier", 100)
            T = tte_years_from_dt(dt)
            sigma = vix / 100.0
            opt_spot = self._opt_price(price)
            if o["opt_type"] == "CALL":
                current = bs_call(opt_spot, o["strike"], T, sigma)
            else:
                current = bs_put(opt_spot, o["strike"], T, sigma)
            contracts = o["contracts"]
            pnl = (current - o["opt_entry"]) * mult * contracts
            comm = 2 * self.p.get("commission_per_side", 0.65) * contracts
            net = round(pnl - comm, 2)
            self.trades.append({
                "date": o["dt"].strftime("%Y-%m-%d"),
                "entry_time": o["dt"].strftime("%H:%M"),
                "exit_time": dt.strftime("%H:%M"),
                "dir": o["dir"], "opt_type": o["opt_type"],
                "strike": o["strike"],
                "spot_entry": round(o["spot_entry"], 2),
                "spot_exit": round(opt_spot, 2),
                "opt_entry": round(o["opt_entry"], 2),
                "opt_exit": round(current, 2),
                "contracts": contracts,
                "gross": round(pnl, 2), "net": net,
                "reason": "EOD_CLOSE",
                "hold_min": round((dt - o["dt"]).total_seconds() / 60, 1),
                "vix": round(o["vix"], 1),
            })
            self._daily_pnl += net
            self._total_pnl += net
            self.open = None


# ─── Data Download ───────────────────────────────────────────────────────────

def download_data(months: int):
    """Download SPX + VIX bars. Uses 5m for recent 60d, hourly for rest."""
    print("Downloading data from yfinance...")

    # Always get hourly for the full period
    period = f"{months}mo" if months <= 24 else "2y"
    spx_h = yf.Ticker("^GSPC").history(period=period, interval="60m", auto_adjust=True)
    vix_h = yf.Ticker("^VIX").history(period=period, interval="60m", auto_adjust=True)
    print(f"  Hourly SPX: {len(spx_h)} bars ({spx_h.index[0].date()} → {spx_h.index[-1].date()})")
    print(f"  Hourly VIX: {len(vix_h)} bars")

    # Get 5-min for recent 60 days (better resolution)
    try:
        spx_5m = yf.Ticker("^GSPC").history(period="60d", interval="5m", auto_adjust=True)
        vix_5m = yf.Ticker("^VIX").history(period="60d", interval="5m", auto_adjust=True)
        print(f"  5-min SPX: {len(spx_5m)} bars ({spx_5m.index[0].date()} → {spx_5m.index[-1].date()})")
        cutoff_5m = spx_5m.index[0].date()
    except Exception:
        spx_5m = pd.DataFrame()
        vix_5m = pd.DataFrame()
        cutoff_5m = None

    return spx_h, vix_h, spx_5m, vix_5m, cutoff_5m


# ─── Main Backtest ───────────────────────────────────────────────────────────

def run_backtest(months: int, account_name: str):
    profile = ACCOUNTS[account_name]
    print(f"\n{'='*75}")
    print(f"  YEAR-LONG BACKTEST: {profile['name']} ({profile['mode'].upper()} mode)")
    print(f"  Scale-on-trail: {'ON' if profile.get('scale_on_trail') else 'OFF'}")
    print(f"  GEX regime: PROXIED from VIX (< 20 = POSITIVE, >= 20 = NEGATIVE)")
    print(f"{'='*75}\n")

    spx_h, vix_h, spx_5m, vix_5m, cutoff_5m = download_data(months)

    # Merge VIX into SPX frames for easy lookup
    # For hourly: align VIX to SPX timestamps
    vix_close_h = vix_h["Close"].reindex(spx_h.index, method="ffill")
    if not spx_5m.empty and not vix_5m.empty:
        vix_close_5m = vix_5m["Close"].reindex(spx_5m.index, method="ffill")
    else:
        vix_close_5m = pd.Series(dtype=float)

    # Get all unique trading days
    all_dates_h = sorted(set(spx_h.index.date))
    all_dates_5m = sorted(set(spx_5m.index.date)) if not spx_5m.empty else []

    # Combine: use 5m data for recent days, hourly for older days
    if cutoff_5m:
        hourly_dates = [d for d in all_dates_h if d < cutoff_5m]
        fivemin_dates = all_dates_5m
    else:
        hourly_dates = all_dates_h
        fivemin_dates = []

    all_dates = sorted(set(hourly_dates + fivemin_dates))
    print(f"  Trading days: {len(all_dates)} ({all_dates[0]} → {all_dates[-1]})")
    print(f"  Hourly days: {len(hourly_dates)}  |  5-min days: {len(fivemin_dates)}")
    print()

    sim = TradeSim(profile)
    daily_results = []
    days_with_trades = 0

    for day_idx, day in enumerate(all_dates):
        # Select appropriate data source
        if day in fivemin_dates and not spx_5m.empty:
            day_bars = spx_5m[spx_5m.index.date == day].copy()
            day_vix = vix_close_5m[vix_close_5m.index.date == day]
        else:
            day_bars = spx_h[spx_h.index.date == day].copy()
            day_vix = vix_close_h[vix_close_h.index.date == day]

        if len(day_bars) < 5:
            continue

        # Compute indicators for this day (with lookback for stability)
        # Get some lookback bars for indicator warmup
        if day in fivemin_dates and not spx_5m.empty:
            lookback_mask = spx_5m.index.date <= day
            lookback = spx_5m[lookback_mask].tail(200)
        else:
            lookback_mask = spx_h.index.date <= day
            lookback = spx_h[lookback_mask].tail(50)

        closes = lookback["Close"]
        ema9_series = compute_ema(closes, 9)
        ema21_series = compute_ema(closes, 21)
        rsi_series = compute_rsi(closes, 14)
        atr_series = compute_atr(lookback, 14)
        vwap_series = compute_vwap(day_bars)

        day_start_trades = len(sim.trades)

        for i, (ts, bar) in enumerate(day_bars.iterrows()):
            ts_naive = ts.tz_localize(None) if ts.tzinfo else ts
            dt = ts_naive

            # Skip pre-market
            if dt.time() < dtime(9, 35):
                continue

            price = float(bar["Close"])
            # Get VIX for this timestamp
            if ts in day_vix.index:
                vix = float(day_vix[ts])
            elif not day_vix.empty:
                vix = float(day_vix.iloc[-1] if ts > day_vix.index[-1] else day_vix.iloc[0])
            else:
                vix = 20.0

            if vix <= 0 or np.isnan(vix):
                vix = 20.0

            # Get indicators at this point
            if ts in ema9_series.index:
                ema9 = float(ema9_series[ts])
                ema21 = float(ema21_series[ts])
                rsi = float(rsi_series[ts]) if not np.isnan(rsi_series[ts]) else 50.0
                atr = float(atr_series[ts]) if not np.isnan(atr_series[ts]) else price * 0.002
            else:
                # Approximate from available data
                ema9 = float(ema9_series.iloc[-1])
                ema21 = float(ema21_series.iloc[-1])
                rsi = float(rsi_series.iloc[-1]) if not np.isnan(rsi_series.iloc[-1]) else 50.0
                atr = float(atr_series.iloc[-1]) if not np.isnan(atr_series.iloc[-1]) else price * 0.002

            if atr <= 0 or np.isnan(atr):
                atr = price * 0.002

            # VWAP
            if ts in vwap_series.index:
                vwap = float(vwap_series[ts])
            else:
                vwap = price
            price_vs_vwap = price - vwap

            # Proxy GEX regime from VIX
            regime = estimate_gex_regime(vix)

            # Generate signal
            sig, reason, conf = generate_signal(
                price, vix, regime, ema9, ema21, rsi, atr, price_vs_vwap
            )

            sim.tick(dt, price, vix, sig, conf, reason, atr)

        # Force close at end of day
        if not day_bars.empty:
            last_ts = day_bars.index[-1]
            last_ts_naive = last_ts.tz_localize(None) if last_ts.tzinfo else last_ts
            last_price = float(day_bars["Close"].iloc[-1])
            last_vix = float(day_vix.iloc[-1]) if not day_vix.empty else 20.0
            if last_vix <= 0 or np.isnan(last_vix):
                last_vix = 20.0
            sim.force_close_day(last_ts_naive, last_price, last_vix)

        day_trades = sim.trades[day_start_trades:]
        day_net = sum(t["net"] for t in day_trades)
        daily_results.append({
            "date": day,
            "trades": len(day_trades),
            "net": day_net,
            "vix": float(day_vix.iloc[0]) if not day_vix.empty else 0,
        })
        if day_trades:
            days_with_trades += 1

        # Progress
        if (day_idx + 1) % 50 == 0:
            print(f"  Processed {day_idx + 1}/{len(all_dates)} days... "
                  f"{len(sim.trades)} trades so far, PnL: ${sim._total_pnl:+,.0f}")

    # ─── Results ─────────────────────────────────────────────────────────────

    trades = sim.trades
    print(f"\n{'='*75}")
    print("BACKTEST RESULTS")
    print(f"{'='*75}")

    if not trades:
        print("  No trades generated.")
        return

    nets = [t["net"] for t in trades]
    wins = [n for n in nets if n > 0]
    losses = [n for n in nets if n <= 0]
    total_pnl = sum(nets)

    # Equity curve stats
    cum = 0.0; peak = 0.0; max_dd = 0.0; max_dd_date = ""
    for r in daily_results:
        cum += r["net"]
        if cum > peak:
            peak = cum
        dd = cum - peak
        if dd < max_dd:
            max_dd = dd
            max_dd_date = str(r["date"])

    # Monthly breakdown
    monthly = defaultdict(lambda: {"trades": 0, "net": 0.0, "days": 0})
    for r in daily_results:
        m = r["date"].strftime("%Y-%m")
        monthly[m]["trades"] += r["trades"]
        monthly[m]["net"] += r["net"]
        monthly[m]["days"] += 1

    # Win/loss streaks
    streak = 0; max_win_streak = 0; max_loss_streak = 0; cur_type = None
    for n in nets:
        t = "W" if n > 0 else "L"
        if t == cur_type:
            streak += 1
        else:
            streak = 1
            cur_type = t
        if t == "W":
            max_win_streak = max(max_win_streak, streak)
        else:
            max_loss_streak = max(max_loss_streak, streak)

    # Exit reason breakdown
    reasons = defaultdict(lambda: {"count": 0, "pnl": 0.0})
    for t in trades:
        r = t["reason"]
        reasons[r]["count"] += 1
        reasons[r]["pnl"] += t["net"]

    # Daily P&L stats
    daily_pnls = [r["net"] for r in daily_results if r["trades"] > 0]
    win_days = len([p for p in daily_pnls if p > 0])
    loss_days = len([p for p in daily_pnls if p <= 0])

    print(f"""
  Period:              {all_dates[0]} → {all_dates[-1]} ({len(all_dates)} trading days)
  Days with trades:    {days_with_trades} ({days_with_trades/len(all_dates)*100:.0f}% of days)
  Total trades:        {len(trades)}
  Avg trades/day:      {len(trades)/days_with_trades:.1f} (on active days)

  Win rate:            {len(wins)}/{len(trades)} = {len(wins)/len(trades)*100:.1f}%
  Profit factor:       {sum(wins)/abs(sum(losses)):.2f}x
  Avg win:             ${sum(wins)/len(wins):+,.2f}
  Avg loss:            ${sum(losses)/len(losses):+,.2f}
  Best trade:          ${max(nets):+,.2f}
  Worst trade:         ${min(nets):+,.2f}

  TOTAL NET P&L:       ${total_pnl:+,.2f}
  Max drawdown:        ${max_dd:+,.2f} (on {max_dd_date})
  Avg daily P&L:       ${total_pnl/days_with_trades:+,.2f} (active days)

  Win days / Loss days: {win_days} / {loss_days} ({win_days/(win_days+loss_days)*100:.0f}% win days)
  Max win streak:      {max_win_streak} trades
  Max loss streak:     {max_loss_streak} trades
  EV per trade:        ${total_pnl/len(trades):+,.2f}
""")

    # Exit reason breakdown
    print("  EXIT REASONS:")
    print(f"  {'Reason':<20} {'Count':>6} {'Total PnL':>12} {'Avg PnL':>10}")
    print(f"  " + "-" * 52)
    for r in sorted(reasons, key=lambda x: reasons[x]["count"], reverse=True):
        d = reasons[r]
        avg = d["pnl"] / d["count"] if d["count"] else 0
        print(f"  {r:<20} {d['count']:>6} ${d['pnl']:>+10,.2f} ${avg:>+9,.2f}")

    # Monthly breakdown
    print(f"\n  MONTHLY BREAKDOWN:")
    print(f"  {'Month':<10} {'Trades':>7} {'Net PnL':>12} {'Avg/Day':>10} {'Days':>5}")
    print(f"  " + "-" * 48)
    for m in sorted(monthly):
        d = monthly[m]
        avg = d["net"] / d["days"] if d["days"] else 0
        print(f"  {m:<10} {d['trades']:>7} ${d['net']:>+10,.2f} ${avg:>+9,.2f} {d['days']:>5}")

    # Equity curve (monthly cumulative)
    print(f"\n  EQUITY CURVE (cumulative):")
    cum = 0.0
    for m in sorted(monthly):
        cum += monthly[m]["net"]
        bar_len = max(0, int(cum / max(abs(total_pnl), 1) * 40))
        bar = "█" * bar_len if cum > 0 else ""
        print(f"  {m}  ${cum:>+12,.0f}  {bar}")

    # VIX regime analysis
    pos_trades = [t for t in trades if t["vix"] < 20]
    neg_trades = [t for t in trades if t["vix"] >= 20]
    print(f"\n  VIX REGIME ANALYSIS:")
    for label, subset in [("POSITIVE (VIX<20)", pos_trades), ("NEGATIVE (VIX≥20)", neg_trades)]:
        if not subset:
            print(f"  {label}: no trades")
            continue
        sn = [t["net"] for t in subset]
        sw = [n for n in sn if n > 0]
        wr = len(sw) / len(sn) * 100
        print(f"  {label}: {len(subset)} trades, {wr:.0f}% WR, ${sum(sn):+,.0f} net")

    if sim._paused:
        print(f"\n  ⚠ BOT PAUSED — max drawdown (${profile['max_drawdown']:,}) hit")

    print(f"\n{'='*75}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Year-long 0DTE scalp backtest")
    parser.add_argument("--months", type=int, default=12, help="Months to backtest (default: 12)")
    parser.add_argument("--account", type=str, default="pm",
                        help="Account profile (pm, etrade, robinhood)")
    args = parser.parse_args()
    run_backtest(args.months, args.account)
