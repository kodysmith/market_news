#!/usr/bin/env python3
"""
Compare the scalp bot strategy across three execution modes:

  1. SPOT with realistic costs (slippage + $0.65/side commission)
  2. SPOT zero-cost (commission-free broker, no slippage)
  3. 0DTE SPX OPTIONS (buy ATM call for LONG, ATM put for SHORT)

Uses saved signal logs that include real GEX data.

Usage:
    python scripts/compare_scalp_modes.py [--date 20260310] [--log-dir logs/scalp_bot]
"""

from __future__ import annotations

import argparse
import csv
import math
import sys
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from scipy.stats import norm

ET = ZoneInfo("America/New_York")

# ─── Black-Scholes for 0DTE option pricing ───────────────────────────────────

R = 0.05  # risk-free rate

def _d1(S, K, T, sigma):
    if T <= 0 or sigma <= 0:
        return 0.0
    return (math.log(S / K) + (R + 0.5 * sigma**2) * T) / (sigma * math.sqrt(T))

def bs_call(S, K, T, sigma):
    if T <= 1e-10:
        return max(S - K, 0.0)
    d1 = _d1(S, K, T, sigma)
    d2 = d1 - sigma * math.sqrt(T)
    return S * norm.cdf(d1) - K * math.exp(-R * T) * norm.cdf(d2)

def bs_put(S, K, T, sigma):
    if T <= 1e-10:
        return max(K - S, 0.0)
    d1 = _d1(S, K, T, sigma)
    d2 = d1 - sigma * math.sqrt(T)
    return K * math.exp(-R * T) * norm.cdf(-d2) - S * norm.cdf(-d1)

def bs_delta_call(S, K, T, sigma):
    if T <= 1e-10:
        return 1.0 if S > K else 0.0
    return norm.cdf(_d1(S, K, T, sigma))

def bs_delta_put(S, K, T, sigma):
    if T <= 1e-10:
        return -1.0 if S < K else 0.0
    return norm.cdf(_d1(S, K, T, sigma)) - 1.0

def atm_strike(spot, increment=5.0):
    """Round to nearest strike increment."""
    return round(spot / increment) * increment

def time_to_close_years(ts: str) -> float:
    """Fraction of year remaining until 4:00 PM ET."""
    try:
        dt = datetime.strptime(ts, "%Y-%m-%d %H:%M:%S")
    except ValueError:
        return 0.01
    close = dt.replace(hour=16, minute=0, second=0)
    minutes_left = max(1, (close - dt).total_seconds() / 60)
    return minutes_left / (365 * 24 * 60)


# ─── Signal loader ───────────────────────────────────────────────────────────

def load_signals(path: Path) -> list[dict]:
    if not path.exists():
        print(f"Error: {path} not found")
        sys.exit(1)
    with open(path) as f:
        return list(csv.DictReader(f))


# ─── Trade simulators ───────────────────────────────────────────────────────

class SpotSimulator:
    """Simulates spot/share trading with configurable costs."""

    def __init__(self, slippage_pct: float = 0.02, commission: float = 0.65):
        self.slippage_pct = slippage_pct
        self.commission = commission
        self.trades: list[dict] = []
        self.open: dict | None = None
        self._last_exit_ts = ""
        self._peak_pnl = 0.0
        self._trade_atr = 0.0
        self._today_trades = 0
        self._today_date = ""

    # ATR-based params (same as v2 TradeManager)
    SL_ATR_POS = 2.0
    SL_ATR_NEG = 2.5
    TP_ATR_POS = 3.0
    TP_ATR_NEG = 5.0
    TRAIL_ACTIVATE = 2.0
    TRAIL_DISTANCE = 1.0
    MAX_HOLD = 20 * 60
    COOLDOWN = 300
    MAX_DAILY = 8
    CONFIDENCE = 0.60      # match v1 signals threshold for comparison
    WINDOWS = [(9, 35, 15, 55)]  # single wide window for comparison

    def _in_window(self, ts):
        try:
            t = datetime.strptime(ts, "%Y-%m-%d %H:%M:%S").time()
        except ValueError:
            return True
        from datetime import time as dtime
        for h1, m1, h2, m2 in self.WINDOWS:
            if dtime(h1, m1) <= t <= dtime(h2, m2):
                return True
        return False

    def _in_cooldown(self, ts):
        if not self._last_exit_ts:
            return False
        try:
            now = datetime.strptime(ts, "%Y-%m-%d %H:%M:%S")
            prev = datetime.strptime(self._last_exit_ts, "%Y-%m-%d %H:%M:%S")
            return (now - prev).total_seconds() < self.COOLDOWN
        except ValueError:
            return False

    def tick(self, row: dict):
        price = float(row["spx_price"])
        sig = row.get("signal", "FLAT")
        conf = float(row.get("confidence", 0))
        regime = (row.get("gex_regime", "UNKNOWN") or "UNKNOWN").replace("_GAMMA", "")
        atr = float(row.get("atr14", 0))
        ts = row["timestamp"]

        today = ts[:10]
        if today != self._today_date:
            self._today_date = today
            self._today_trades = 0

        if self.open:
            self._check_exit(row, price)
        elif sig in ("LONG", "SHORT") and conf >= self.CONFIDENCE:
            if atr <= 0:
                return
            # Cost-viability check
            if regime == "POSITIVE":
                tp_pts = atr * self.TP_ATR_POS
            else:
                tp_pts = atr * self.TP_ATR_NEG
            if tp_pts - 2 * self.commission < 1.0:
                return
            if (self._in_cooldown(ts) or not self._in_window(ts)
                    or self._today_trades >= self.MAX_DAILY):
                return
            # Enter
            if sig == "LONG":
                fill = price * (1 + self.slippage_pct / 100)
            else:
                fill = price * (1 - self.slippage_pct / 100)
            self.open = {
                "entry_ts": ts, "entry": round(fill, 2), "dir": sig,
                "regime": regime, "atr": atr, "reason": row.get("reason", ""),
            }
            self._trade_atr = atr
            self._peak_pnl = 0.0
            self._today_trades += 1

    def _check_exit(self, row, price):
        o = self.open
        is_long = o["dir"] == "LONG"
        pnl = (price - o["entry"]) if is_long else (o["entry"] - price)
        self._peak_pnl = max(self._peak_pnl, pnl)

        atr = self._trade_atr or 1.0
        if o["regime"] == "POSITIVE":
            sl = atr * self.SL_ATR_POS
            tp = atr * self.TP_ATR_POS
        else:
            sl = atr * self.SL_ATR_NEG
            tp = atr * self.TP_ATR_NEG

        try:
            hold = (datetime.strptime(row["timestamp"], "%Y-%m-%d %H:%M:%S")
                    - datetime.strptime(o["entry_ts"], "%Y-%m-%d %H:%M:%S")).total_seconds()
        except ValueError:
            hold = 0

        exit_reason = ""
        exit_px = price
        use_actual = False
        sig = row.get("signal", "FLAT")
        regime = (row.get("gex_regime", "UNKNOWN") or "UNKNOWN").replace("_GAMMA", "")

        if pnl <= -sl:
            exit_reason = "STOP_LOSS"
            exit_px = (o["entry"] - sl) if is_long else (o["entry"] + sl)
        elif pnl >= tp:
            exit_reason = "TAKE_PROFIT"
            exit_px = (o["entry"] + tp) if is_long else (o["entry"] - tp)
        elif (self._peak_pnl >= atr * self.TRAIL_ACTIVATE
              and pnl < self._peak_pnl - atr * self.TRAIL_DISTANCE):
            exit_reason = "TRAIL_STOP"
            use_actual = True
        elif hold >= self.MAX_HOLD:
            exit_reason = "TIME_EXIT"
            use_actual = True
        elif regime != o["regime"] and regime != "UNKNOWN":
            exit_reason = "REGIME_FLIP"
            use_actual = True
        elif (is_long and sig == "SHORT") or (not is_long and sig == "LONG"):
            exit_reason = "SIGNAL_REVERSAL"
            use_actual = True

        if not exit_reason:
            return

        if use_actual:
            if is_long:
                exit_px = price * (1 - self.slippage_pct / 100)
            else:
                exit_px = price * (1 + self.slippage_pct / 100)

        exit_px = round(exit_px, 2)
        gross = (exit_px - o["entry"]) if is_long else (o["entry"] - exit_px)
        comm = 2 * self.commission
        net = gross - comm

        self.trades.append({
            "entry_ts": o["entry_ts"], "exit_ts": row["timestamp"],
            "dir": o["dir"], "entry": o["entry"], "exit": exit_px,
            "gross": round(gross, 2), "net": round(net, 2),
            "commission": round(comm, 2), "reason": exit_reason,
            "regime": o["regime"], "hold_sec": int(hold),
        })
        self._last_exit_ts = row["timestamp"]
        self.open = None

    def force_close(self, row):
        if self.open:
            self._check_exit(row, float(row["spx_price"]))
            if self.open:
                # Force it
                price = float(row["spx_price"])
                o = self.open
                is_long = o["dir"] == "LONG"
                if is_long:
                    exit_px = price * (1 - self.slippage_pct / 100)
                else:
                    exit_px = price * (1 + self.slippage_pct / 100)
                gross = (exit_px - o["entry"]) if is_long else (o["entry"] - exit_px)
                self.trades.append({
                    "entry_ts": o["entry_ts"], "exit_ts": row["timestamp"],
                    "dir": o["dir"], "entry": o["entry"], "exit": round(exit_px, 2),
                    "gross": round(gross, 2), "net": round(gross - 2 * self.commission, 2),
                    "commission": round(2 * self.commission, 2), "reason": "FORCE_CLOSE",
                    "regime": o["regime"], "hold_sec": 0,
                })
                self.open = None


class OptionSimulator:
    """Simulates 0DTE ATM option trading on SPX signals."""

    OPTION_COMMISSION = 0.65  # per contract per side
    CONTRACTS = 1
    MULTIPLIER = 100  # SPX options are $100 per point

    def __init__(self):
        self.trades: list[dict] = []
        self.open: dict | None = None
        self._last_exit_ts = ""
        self._peak_pnl = 0.0
        self._today_trades = 0
        self._today_date = ""

    # Tighter params for options (premium decays fast)
    MAX_HOLD = 15 * 60     # 15 min max for 0DTE
    COOLDOWN = 300
    MAX_DAILY = 8
    CONFIDENCE = 0.60
    TP_PCT = 50.0          # take profit at 50% of premium gain
    SL_PCT = 30.0          # stop at 30% of premium loss
    TRAIL_PCT = 25.0       # trail at 25% pullback from peak
    WINDOWS = [(9, 35, 15, 55)]

    def _in_window(self, ts):
        try:
            t = datetime.strptime(ts, "%Y-%m-%d %H:%M:%S").time()
        except ValueError:
            return True
        from datetime import time as dtime
        for h1, m1, h2, m2 in self.WINDOWS:
            if dtime(h1, m1) <= t <= dtime(h2, m2):
                return True
        return False

    def _in_cooldown(self, ts):
        if not self._last_exit_ts:
            return False
        try:
            now = datetime.strptime(ts, "%Y-%m-%d %H:%M:%S")
            prev = datetime.strptime(self._last_exit_ts, "%Y-%m-%d %H:%M:%S")
            return (now - prev).total_seconds() < self.COOLDOWN
        except ValueError:
            return False

    def tick(self, row: dict):
        price = float(row["spx_price"])
        sig = row.get("signal", "FLAT")
        conf = float(row.get("confidence", 0))
        regime = (row.get("gex_regime", "UNKNOWN") or "UNKNOWN").replace("_GAMMA", "")
        vix = float(row.get("vix", 20))
        ts = row["timestamp"]

        today = ts[:10]
        if today != self._today_date:
            self._today_date = today
            self._today_trades = 0

        if self.open:
            self._check_exit(row, price, vix)
        elif sig in ("LONG", "SHORT") and conf >= self.CONFIDENCE:
            if (self._in_cooldown(ts) or not self._in_window(ts)
                    or self._today_trades >= self.MAX_DAILY):
                return

            # Price the ATM option
            K = atm_strike(price)
            T = time_to_close_years(ts)
            sigma = vix / 100.0
            if T < 1e-8 or sigma < 0.05:
                return

            if sig == "LONG":
                opt_price = bs_call(price, K, T, sigma)
                opt_type = "CALL"
            else:
                opt_price = bs_put(price, K, T, sigma)
                opt_type = "PUT"

            if opt_price < 0.50:  # skip if premium too cheap (near worthless)
                return

            self.open = {
                "entry_ts": ts, "spot_entry": price, "strike": K,
                "opt_type": opt_type, "opt_entry": round(opt_price, 2),
                "dir": sig, "regime": regime, "vix": vix,
                "reason": row.get("reason", ""),
            }
            self._peak_pnl = 0.0
            self._today_trades += 1

    def _check_exit(self, row, price, vix):
        o = self.open
        T = time_to_close_years(row["timestamp"])
        sigma = vix / 100.0 if vix > 0 else o["vix"] / 100.0

        # Re-price the option
        if o["opt_type"] == "CALL":
            current = bs_call(price, o["strike"], T, sigma)
        else:
            current = bs_put(price, o["strike"], T, sigma)

        pnl_per_contract = (current - o["opt_entry"]) * self.MULTIPLIER
        self._peak_pnl = max(self._peak_pnl, pnl_per_contract)

        try:
            hold = (datetime.strptime(row["timestamp"], "%Y-%m-%d %H:%M:%S")
                    - datetime.strptime(o["entry_ts"], "%Y-%m-%d %H:%M:%S")).total_seconds()
        except ValueError:
            hold = 0

        sig = row.get("signal", "FLAT")
        regime = (row.get("gex_regime", "UNKNOWN") or "UNKNOWN").replace("_GAMMA", "")
        is_long = o["dir"] == "LONG"

        exit_reason = ""
        # TP: option gained TP_PCT% of entry premium
        if pnl_per_contract >= o["opt_entry"] * self.MULTIPLIER * self.TP_PCT / 100:
            exit_reason = "TAKE_PROFIT"
        # SL: option lost SL_PCT% of entry premium
        elif pnl_per_contract <= -o["opt_entry"] * self.MULTIPLIER * self.SL_PCT / 100:
            exit_reason = "STOP_LOSS"
        # Trail
        elif (self._peak_pnl > o["opt_entry"] * self.MULTIPLIER * 0.20
              and pnl_per_contract < self._peak_pnl * (1 - self.TRAIL_PCT / 100)):
            exit_reason = "TRAIL_STOP"
        elif hold >= self.MAX_HOLD:
            exit_reason = "TIME_EXIT"
        elif regime != o["regime"] and regime != "UNKNOWN":
            exit_reason = "REGIME_FLIP"
        elif (is_long and sig == "SHORT") or (not is_long and sig == "LONG"):
            exit_reason = "SIGNAL_REVERSAL"

        if not exit_reason:
            return

        commission = 2 * self.OPTION_COMMISSION * self.CONTRACTS
        gross_dollar = round(pnl_per_contract * self.CONTRACTS, 2)
        net_dollar = round(gross_dollar - commission, 2)

        self.trades.append({
            "entry_ts": o["entry_ts"], "exit_ts": row["timestamp"],
            "dir": o["dir"], "opt_type": o["opt_type"],
            "strike": o["strike"],
            "spot_entry": o["spot_entry"], "spot_exit": price,
            "opt_entry": o["opt_entry"], "opt_exit": round(current, 2),
            "gross_dollar": gross_dollar, "net_dollar": net_dollar,
            "commission": round(commission, 2),
            "reason": exit_reason, "regime": o["regime"],
            "hold_sec": int(hold),
        })
        self._last_exit_ts = row["timestamp"]
        self.open = None

    def force_close(self, row):
        if self.open:
            price = float(row["spx_price"])
            vix = float(row.get("vix", self.open["vix"]))
            self._check_exit(row, price, vix)
            if self.open:
                # Force
                T = time_to_close_years(row["timestamp"])
                sigma = vix / 100.0
                if self.open["opt_type"] == "CALL":
                    current = bs_call(price, self.open["strike"], T, sigma)
                else:
                    current = bs_put(price, self.open["strike"], T, sigma)
                pnl = (current - self.open["opt_entry"]) * self.MULTIPLIER * self.CONTRACTS
                comm = 2 * self.OPTION_COMMISSION * self.CONTRACTS
                self.trades.append({
                    "entry_ts": self.open["entry_ts"], "exit_ts": row["timestamp"],
                    "dir": self.open["dir"], "opt_type": self.open["opt_type"],
                    "strike": self.open["strike"],
                    "spot_entry": self.open["spot_entry"], "spot_exit": price,
                    "opt_entry": self.open["opt_entry"], "opt_exit": round(current, 2),
                    "gross_dollar": round(pnl, 2), "net_dollar": round(pnl - comm, 2),
                    "commission": round(comm, 2),
                    "reason": "FORCE_CLOSE", "regime": self.open["regime"],
                    "hold_sec": 0,
                })
                self.open = None


# ─── Summary printer ─────────────────────────────────────────────────────────

def print_spot_summary(name: str, trades: list[dict]):
    if not trades:
        print(f"\n  {name}: No trades")
        return

    pnls = [t["net"] for t in trades]
    gross_pnls = [t["gross"] for t in trades]
    wins = [p for p in pnls if p > 0]
    losses = [p for p in pnls if p <= 0]
    total_comm = sum(t["commission"] for t in trades)

    cum = 0.0; peak = 0.0; max_dd = 0.0
    for p in pnls:
        cum += p
        peak = max(peak, cum)
        max_dd = min(max_dd, cum - peak)

    print(f"\n  {name}")
    print(f"  {'-' * 55}")
    print(f"  Trades:       {len(trades):>4}   |  Win rate:   {len(wins)/len(trades)*100:>5.1f}%")
    print(f"  Gross PnL:  {sum(gross_pnls):>+8.2f} pts  |  Commission: {total_comm:>7.2f} pts")
    print(f"  Net PnL:    {sum(pnls):>+8.2f} pts  |  Max DD:     {max_dd:>+7.2f} pts")
    if wins:
        print(f"  Avg win:    {sum(wins)/len(wins):>+8.2f} pts  |  Avg loss:   {sum(losses)/len(losses) if losses else 0:>+7.2f} pts")
    pf = abs(sum(wins) / sum(losses)) if losses and sum(losses) != 0 else float("inf")
    print(f"  Profit fac: {pf:>8.2f}       |  Avg hold:   {sum(t['hold_sec'] for t in trades)/len(trades)/60:>5.1f} min")

    # Trade detail
    print(f"\n  {'#':>3}  {'Time':>5}  {'Dir':>5}  {'Entry':>8}  {'Exit':>8}  {'Gross':>7}  {'Net':>7}  {'Reason'}")
    print(f"  " + "-" * 70)
    for i, t in enumerate(trades, 1):
        tm = t["entry_ts"].split(" ")[1][:5] if " " in t["entry_ts"] else ""
        print(f"  {i:>3}  {tm:>5}  {t['dir']:>5}  ${t['entry']:>7.2f}  ${t['exit']:>7.2f}  "
              f"{t['gross']:>+7.2f}  {t['net']:>+7.2f}  {t['reason']}")


def print_option_summary(trades: list[dict]):
    if not trades:
        print(f"\n  0DTE OPTIONS: No trades")
        return

    pnls = [t["net_dollar"] for t in trades]
    gross = [t["gross_dollar"] for t in trades]
    wins = [p for p in pnls if p > 0]
    losses = [p for p in pnls if p <= 0]
    total_comm = sum(t["commission"] for t in trades)

    cum = 0.0; peak = 0.0; max_dd = 0.0
    for p in pnls:
        cum += p
        peak = max(peak, cum)
        max_dd = min(max_dd, cum - peak)

    print(f"\n  0DTE SPX OPTIONS (1 contract, $100 multiplier)")
    print(f"  {'-' * 55}")
    print(f"  Trades:       {len(trades):>4}   |  Win rate:   {len(wins)/len(trades)*100:>5.1f}%")
    print(f"  Gross PnL:  ${sum(gross):>+8.2f}  |  Commission: ${total_comm:>6.2f}")
    print(f"  Net PnL:    ${sum(pnls):>+8.2f}  |  Max DD:     ${max_dd:>+7.2f}")
    if wins:
        print(f"  Avg win:    ${sum(wins)/len(wins):>+8.2f}  |  Avg loss:   ${sum(losses)/len(losses) if losses else 0:>+7.2f}")
    pf = abs(sum(wins) / sum(losses)) if losses and sum(losses) != 0 else float("inf")
    print(f"  Profit fac: {pf:>8.2f}       |  Avg hold:   {sum(t['hold_sec'] for t in trades)/len(trades)/60:>5.1f} min")

    # Trade detail
    print(f"\n  {'#':>3}  {'Time':>5}  {'Type':>4}  {'K':>6}  {'Spot':>7}→{'':>7}  "
          f"{'Prem':>6}→{'':>6}  {'Gross$':>7}  {'Net$':>7}  {'Reason'}")
    print(f"  " + "-" * 85)
    for i, t in enumerate(trades, 1):
        tm = t["entry_ts"].split(" ")[1][:5] if " " in t["entry_ts"] else ""
        print(f"  {i:>3}  {tm:>5}  {t['opt_type']:>4}  {t['strike']:>6.0f}  "
              f"${t['spot_entry']:>6.0f}→${t['spot_exit']:>6.0f}  "
              f"${t['opt_entry']:>5.2f}→${t['opt_exit']:>5.2f}  "
              f"{t['gross_dollar']:>+7.2f}  {t['net_dollar']:>+7.2f}  {t['reason']}")


# ─── Main ────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Compare scalp strategy across execution modes")
    parser.add_argument("--date", type=str, default=None, help="Date YYYYMMDD")
    parser.add_argument("--log-dir", type=str, default="logs/scalp_bot")
    parser.add_argument("--ticker", type=str, default=None, help="Filter by ticker")
    args = parser.parse_args()

    log_dir = ROOT / args.log_dir
    date_str = args.date or datetime.now(ET).strftime("%Y%m%d")
    sig_path = log_dir / f"signals_{date_str}.csv"

    rows = load_signals(sig_path)
    if args.ticker:
        rows = [r for r in rows if (r.get("ticker", "SPX")).upper() == args.ticker.upper()]

    print("=" * 65)
    print(f"STRATEGY COMPARISON — {date_str} ({len(rows)} signals)")
    print(f"  Source: {sig_path.name}")
    if args.ticker:
        print(f"  Ticker filter: {args.ticker}")
    print("=" * 65)

    # Run all three simulators
    spot_real = SpotSimulator(slippage_pct=0.02, commission=0.65)
    spot_free = SpotSimulator(slippage_pct=0.0, commission=0.0)
    options = OptionSimulator()

    for row in rows:
        # Estimate ATR if missing from old-format logs
        if not row.get("atr14") or float(row.get("atr14", 0)) <= 0:
            # 1-min ATR ≈ 0.035% of price (empirical: ~2.4 pts for SPX, ~0.14 for TSLA)
            row["atr14"] = str(float(row["spx_price"]) * 0.00035)
        spot_real.tick(row)
        spot_free.tick(row)
        # Only run options sim on SPX-level prices (> $1000)
        if float(row["spx_price"]) > 1000:
            options.tick(row)

    # Force close any open trades
    if rows:
        last = rows[-1]
        spot_real.force_close(last)
        spot_free.force_close(last)
        options.force_close(last)

    # Print results
    print_spot_summary("MODE 1: SPOT + COSTS (slip=0.02%, comm=$0.65/side)", spot_real.trades)
    print_spot_summary("MODE 2: SPOT ZERO-COST (commission-free broker)", spot_free.trades)
    print_option_summary(options.trades)

    # Summary comparison
    print("\n" + "=" * 65)
    print("SIDE-BY-SIDE COMPARISON")
    print("=" * 65)

    headers = ["Metric", "Spot+Costs", "Spot Free", "0DTE Options"]
    def _net(trades, key="net"):
        return sum(t[key] for t in trades) if trades else 0
    def _wr(trades, key="net"):
        if not trades: return 0
        return len([t for t in trades if t[key] > 0]) / len(trades) * 100

    spot_r_net = _net(spot_real.trades)
    spot_f_net = _net(spot_free.trades)
    opt_net = _net(options.trades, "net_dollar")

    rows_data = [
        ("Trades", len(spot_real.trades), len(spot_free.trades), len(options.trades)),
        ("Win Rate", f"{_wr(spot_real.trades):.0f}%", f"{_wr(spot_free.trades):.0f}%",
         f"{_wr(options.trades, 'net_dollar'):.0f}%"),
        ("Net PnL", f"{spot_r_net:+.2f} pts", f"{spot_f_net:+.2f} pts", f"${opt_net:+.2f}"),
        ("Comm paid", f"{sum(t['commission'] for t in spot_real.trades):.2f}",
         f"{sum(t['commission'] for t in spot_free.trades):.2f}",
         f"${sum(t['commission'] for t in options.trades):.2f}"),
    ]
    col_w = [16, 14, 14, 14]
    print(f"  {'':>{col_w[0]}}  {'Spot+Costs':>{col_w[1]}}  {'Spot Free':>{col_w[2]}}  {'0DTE Opts':>{col_w[3]}}")
    print(f"  {'-'*sum(col_w)+'-'*6}")
    for label, *vals in rows_data:
        v = [str(v) for v in vals]
        print(f"  {label:>{col_w[0]}}  {v[0]:>{col_w[1]}}  {v[1]:>{col_w[2]}}  {v[2]:>{col_w[3]}}")

    print("=" * 65)


if __name__ == "__main__":
    main()
