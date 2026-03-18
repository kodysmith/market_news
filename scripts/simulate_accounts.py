#!/usr/bin/env python3
"""
Simulate how each account profile would have performed on a day's signals.

Usage:
    python scripts/simulate_accounts.py [--date 20260310]
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
from scripts.scalp_accounts import ACCOUNTS

ET = ZoneInfo("America/New_York")

# ─── BS pricing (same as compare_scalp_modes.py) ─────────────────────────────

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

def tte_years(ts: str) -> float:
    try:
        dt = datetime.strptime(ts, "%Y-%m-%d %H:%M:%S")
    except ValueError:
        return 0.01
    close = dt.replace(hour=16, minute=0, second=0)
    mins = max(1, (close - dt).total_seconds() / 60)
    return mins / (365 * 24 * 60)

def tte_minutes(ts: str) -> float:
    try:
        dt = datetime.strptime(ts, "%Y-%m-%d %H:%M:%S")
    except ValueError:
        return 60
    close = dt.replace(hour=16, minute=0, second=0)
    return max(0, (close - dt).total_seconds() / 60)


# ─── Account Simulator ───────────────────────────────────────────────────────

class AccountSim:
    def __init__(self, profile: dict):
        self.p = profile
        self.trades: list[dict] = []
        self.open = None
        self._last_exit_ts = ""
        self._peak_pnl = 0.0
        self._trade_atr = 0.0
        self._scaled = False            # True once scale-on-trail has fired
        self._today_trades = 0
        self._today_date = ""
        self._daily_pnl = 0.0
        self._total_pnl = 0.0
        self._day_trade_dates: list[str] = []  # rolling PDT tracker
        self._paused = False

    def _in_cooldown(self, ts):
        if not self._last_exit_ts:
            return False
        try:
            now = datetime.strptime(ts, "%Y-%m-%d %H:%M:%S")
            prev = datetime.strptime(self._last_exit_ts, "%Y-%m-%d %H:%M:%S")
            return (now - prev).total_seconds() < self.p["cooldown_sec"]
        except ValueError:
            return False

    def _pdt_ok(self, ts):
        if not self.p.get("pdt_limited"):
            return True
        # Count day trades in last 5 business days
        today = ts[:10]
        cutoff = datetime.strptime(today, "%Y-%m-%d")
        from datetime import timedelta
        biz_days = 0
        d = cutoff
        while biz_days < 5:
            d -= timedelta(days=1)
            if d.weekday() < 5:
                biz_days += 1
        cutoff_str = d.strftime("%Y-%m-%d")
        recent = [dt for dt in self._day_trade_dates if dt > cutoff_str]
        return len(recent) < self.p.get("max_day_trades_per_week", 3)

    def tick(self, row: dict):
        price = float(row["spx_price"])
        sig = row.get("signal", "FLAT")
        conf = float(row.get("confidence", 0))
        regime = (row.get("gex_regime", "UNKNOWN") or "UNKNOWN").replace("_GAMMA", "")
        atr = float(row.get("atr14", 0))
        vix = float(row.get("vix", 20))
        ts = row["timestamp"]

        # Estimate ATR if missing
        if atr <= 0:
            atr = price * 0.00035

        # Reset daily counter
        today = ts[:10]
        if today != self._today_date:
            self._today_date = today
            self._today_trades = 0
            self._daily_pnl = 0.0

        if self._paused:
            return

        if self.open:
            self._check_exit(row, price, vix, atr)
            return

        # Entry filters
        if sig not in ("LONG", "SHORT") or conf < self.p["min_confidence"]:
            return
        if self.p.get("require_gex_regime") and regime == "UNKNOWN":
            return
        if self._in_cooldown(ts):
            return
        max_daily = self.p.get("target_trades_per_day", self.p.get("max_trades_per_day", 8))
        if self._today_trades >= max_daily:
            return
        if not self._pdt_ok(ts):
            return
        if self._daily_pnl <= -self.p["max_daily_loss"]:
            return

        # Mode-specific entry
        if self.p["mode"] == "spot":
            self._enter_spot(row, price, sig, regime, atr, ts)
        elif self.p["mode"] == "0dte":
            self._enter_option(row, price, sig, regime, atr, vix, ts)

    def _enter_spot(self, row, price, sig, regime, atr, ts):
        slip = self.p["slippage_pct"]
        if sig == "LONG":
            fill = price * (1 + slip / 100)
        else:
            fill = price * (1 - slip / 100)

        # Check risk: would SL exceed max_loss_per_trade?
        sl_mult = self.p.get("sl_atr_mult", 2.0)
        sl_pts = atr * sl_mult
        shares = self.p.get("max_shares", 1)
        potential_loss = sl_pts * shares
        if potential_loss > self.p["max_loss_per_trade"]:
            shares = max(1, int(self.p["max_loss_per_trade"] / sl_pts))

        tp_mult = self.p.get("tp_atr_mult", 3.0)

        self.open = {
            "ts": ts, "entry": round(fill, 2), "dir": sig,
            "regime": regime, "atr": atr, "shares": shares,
            "sl_pts": atr * sl_mult, "tp_pts": atr * tp_mult,
            "mode": "spot", "reason": row.get("reason", ""),
        }
        self._trade_atr = atr
        self._peak_pnl = 0.0
        self._today_trades += 1

    def _opt_spot(self, spx_price: float) -> float:
        """XSP uses SPX/10 as underlying; SPX uses raw price."""
        if self.p.get("ticker", "SPX") == "XSP":
            return spx_price / 10.0
        return spx_price

    def _strike_inc(self) -> float:
        return 1.0 if self.p.get("ticker", "SPX") == "XSP" else 5.0

    def _enter_option(self, row, price, sig, regime, atr, vix, ts):
        # Check time to expiry
        tte_min = tte_minutes(ts)
        if tte_min < self.p.get("min_tte_minutes", 30):
            return

        opt_spot = self._opt_spot(price)
        K = atm_strike(opt_spot, self._strike_inc())
        T = tte_years(ts)
        sigma = vix / 100.0
        if sigma < 0.05:
            return

        if sig == "LONG":
            opt_price = bs_call(opt_spot, K, T, sigma)
            opt_type = "CALL"
        else:
            opt_price = bs_put(opt_spot, K, T, sigma)
            opt_type = "PUT"

        if opt_price < self.p.get("min_premium", 2.0):
            return
        if opt_price * self.p.get("multiplier", 100) > self.p.get("max_premium", 1500):
            return

        self.open = {
            "ts": ts, "spot_entry": opt_spot, "strike": K,
            "opt_type": opt_type, "opt_entry": round(opt_price, 2),
            "dir": sig, "regime": regime, "atr": atr, "vix": vix,
            "contracts": self.p.get("contracts", 1),
            "mode": "0dte", "reason": row.get("reason", ""),
        }
        self._peak_pnl = 0.0
        self._scaled = False
        self._today_trades += 1

    def _check_exit(self, row, price, vix, atr):
        if self.open["mode"] == "spot":
            self._check_exit_spot(row, price, atr)
        else:
            self._check_exit_option(row, price, vix)

    def _check_exit_spot(self, row, price, atr):
        o = self.open
        is_long = o["dir"] == "LONG"
        pnl = (price - o["entry"]) if is_long else (o["entry"] - price)
        self._peak_pnl = max(self._peak_pnl, pnl)

        sl = o["sl_pts"]
        tp = o["tp_pts"]
        trail_act = self.p.get("trail_activate_atr", 2.0) * o["atr"]
        trail_dist = self.p.get("trail_distance_atr", 1.0) * o["atr"]

        try:
            hold = (datetime.strptime(row["timestamp"], "%Y-%m-%d %H:%M:%S")
                    - datetime.strptime(o["ts"], "%Y-%m-%d %H:%M:%S")).total_seconds()
        except ValueError:
            hold = 0

        ts = row["timestamp"]
        sig = row.get("signal", "FLAT")

        reason = ""
        exit_px = price
        if pnl <= -sl:
            reason = "STOP_LOSS"
            exit_px = (o["entry"] - sl) if is_long else (o["entry"] + sl)
        elif pnl >= tp:
            reason = "TAKE_PROFIT"
            exit_px = (o["entry"] + tp) if is_long else (o["entry"] - tp)
        elif self._peak_pnl >= trail_act and pnl < self._peak_pnl - trail_dist:
            reason = "TRAIL_STOP"
        elif hold >= self.p.get("max_hold_minutes", 20) * 60:
            reason = "TIME_EXIT"
        elif (is_long and sig == "SHORT") or (not is_long and sig == "LONG"):
            reason = "SIGNAL_REVERSAL"

        if not reason:
            return

        exit_px = round(exit_px, 2)
        gross = (exit_px - o["entry"]) if is_long else (o["entry"] - exit_px)
        comm = 2 * self.p["commission_per_side"]
        shares = o["shares"]
        net_per_share = gross - comm
        net_dollar = round(net_per_share * shares, 2)

        self.trades.append({
            "entry_ts": o["ts"], "exit_ts": ts, "dir": o["dir"],
            "entry": o["entry"], "exit": exit_px, "shares": shares,
            "gross_pts": round(gross, 2), "net_pts": round(net_per_share, 2),
            "net_dollar": net_dollar, "reason": reason,
            "regime": o["regime"], "hold_sec": int(hold), "mode": "spot",
        })
        self._daily_pnl += net_dollar
        self._total_pnl += net_dollar
        if self._total_pnl <= -self.p["max_drawdown"]:
            self._paused = True
        self._last_exit_ts = ts
        self._day_trade_dates.append(ts[:10])
        self.open = None

    def _check_exit_option(self, row, price, vix):
        o = self.open
        T = tte_years(row["timestamp"])
        sigma = vix / 100.0 if vix > 0 else o["vix"] / 100.0
        mult = self.p.get("multiplier", 100)
        opt_spot = self._opt_spot(price)

        if o["opt_type"] == "CALL":
            current = bs_call(opt_spot, o["strike"], T, sigma)
        else:
            current = bs_put(opt_spot, o["strike"], T, sigma)

        pnl_per_contract = (current - o["opt_entry"]) * mult
        self._peak_pnl = max(self._peak_pnl, pnl_per_contract)

        # Scale-on-trail: add contracts when trade reaches trigger % gain
        if (self.p.get("scale_on_trail") and not self._scaled
                and pnl_per_contract >= o["opt_entry"] * mult
                * self.p.get("scale_trigger_pct", 20) / 100):
            o["contracts"] += self.p.get("scale_add_contracts", 1)
            self._scaled = True

        contracts = o["contracts"]

        try:
            hold = (datetime.strptime(row["timestamp"], "%Y-%m-%d %H:%M:%S")
                    - datetime.strptime(o["ts"], "%Y-%m-%d %H:%M:%S")).total_seconds()
        except ValueError:
            hold = 0

        ts = row["timestamp"]
        sig = row.get("signal", "FLAT")
        is_long = o["dir"] == "LONG"
        tp_thresh = o["opt_entry"] * mult * self.p.get("option_tp_pct", 50) / 100
        sl_thresh = o["opt_entry"] * mult * self.p.get("option_sl_pct", 35) / 100

        reason = ""
        if pnl_per_contract >= tp_thresh:
            reason = "TAKE_PROFIT"
        elif pnl_per_contract <= -sl_thresh:
            reason = "STOP_LOSS"
        elif (self._peak_pnl > o["opt_entry"] * mult * 0.20
              and pnl_per_contract < self._peak_pnl * (1 - self.p.get("option_trail_pct", 25) / 100)):
            reason = "TRAIL_STOP"
        elif hold >= self.p.get("max_hold_minutes", 15) * 60:
            reason = "TIME_EXIT"
        elif (is_long and sig == "SHORT") or (not is_long and sig == "LONG"):
            reason = "SIGNAL_REVERSAL"

        if not reason:
            return

        comm = 2 * self.p["commission_per_side"] * contracts
        gross_dollar = round(pnl_per_contract * contracts, 2)
        net_dollar = round(gross_dollar - comm, 2)

        self.trades.append({
            "entry_ts": o["ts"], "exit_ts": ts, "dir": o["dir"],
            "opt_type": o["opt_type"], "strike": o["strike"],
            "spot_entry": o["spot_entry"], "spot_exit": self._opt_spot(price),
            "opt_entry": o["opt_entry"], "opt_exit": round(current, 2),
            "contracts": contracts,
            "gross_dollar": gross_dollar, "net_dollar": net_dollar,
            "commission": round(comm, 2), "reason": reason,
            "regime": o["regime"], "hold_sec": int(hold), "mode": "0dte",
        })
        self._daily_pnl += net_dollar
        self._total_pnl += net_dollar
        if self._total_pnl <= -self.p["max_drawdown"]:
            self._paused = True
        self._last_exit_ts = ts
        self.open = None

    def force_close(self, row):
        if not self.open:
            return
        price = float(row["spx_price"])
        vix = float(row.get("vix", 20))
        atr = float(row.get("atr14", 0)) or price * 0.00035
        # Trigger a time exit
        fake = dict(row)
        fake["signal"] = "FLAT"
        if self.open["mode"] == "spot":
            # Set hold time past max
            self.open["_force"] = True
            o = self.open
            is_long = o["dir"] == "LONG"
            slip = self.p["slippage_pct"]
            if is_long:
                exit_px = price * (1 - slip / 100)
            else:
                exit_px = price * (1 + slip / 100)
            exit_px = round(exit_px, 2)
            gross = (exit_px - o["entry"]) if is_long else (o["entry"] - exit_px)
            comm = 2 * self.p["commission_per_side"]
            shares = o.get("shares", 1)
            net = round((gross - comm) * shares, 2)
            self.trades.append({
                "entry_ts": o["ts"], "exit_ts": row["timestamp"], "dir": o["dir"],
                "entry": o["entry"], "exit": exit_px, "shares": shares,
                "gross_pts": round(gross, 2), "net_pts": round(gross - comm, 2),
                "net_dollar": net, "reason": "EOD_CLOSE",
                "regime": o["regime"], "hold_sec": 0, "mode": "spot",
            })
            self._total_pnl += net
            self.open = None
        else:
            self._check_exit_option(fake, price, vix)
            if self.open:
                # Force option close
                T = tte_years(row["timestamp"])
                sigma = vix / 100.0
                o = self.open
                opt_spot = self._opt_spot(price)
                if o["opt_type"] == "CALL":
                    current = bs_call(opt_spot, o["strike"], T, sigma)
                else:
                    current = bs_put(opt_spot, o["strike"], T, sigma)
                mult = self.p.get("multiplier", 100)
                pnl = (current - o["opt_entry"]) * mult * o["contracts"]
                comm = 2 * self.p["commission_per_side"] * o["contracts"]
                net = round(pnl - comm, 2)
                self.trades.append({
                    "entry_ts": o["ts"], "exit_ts": row["timestamp"], "dir": o["dir"],
                    "opt_type": o["opt_type"], "strike": o["strike"],
                    "spot_entry": o["spot_entry"], "spot_exit": self._opt_spot(price),
                    "opt_entry": o["opt_entry"], "opt_exit": round(current, 2),
                    "contracts": o["contracts"],
                    "gross_dollar": round(pnl, 2), "net_dollar": net,
                    "commission": round(comm, 2), "reason": "EOD_CLOSE",
                    "regime": o["regime"], "hold_sec": 0, "mode": "0dte",
                })
                self._total_pnl += net
                self.open = None


# ─── Print results ───────────────────────────────────────────────────────────

def print_account(name: str, profile: dict, sim: AccountSim):
    trades = sim.trades
    mode = profile["mode"]

    print(f"\n{'=' * 65}")
    print(f"  {name} ({profile['mode'].upper()} mode)")
    print(f"  Account: ${profile['account_size']:,.0f} | "
          f"{'PDT limited' if profile.get('pdt_limited') else 'No PDT'} | "
          f"Max DD: ${profile['max_drawdown']:,.0f}")
    print(f"{'=' * 65}")

    if not trades:
        print("  No trades taken.")
        if sim._paused:
            print("  ** PAUSED — max drawdown hit **")
        return

    if mode == "spot":
        nets = [t["net_dollar"] for t in trades]
        wins = [n for n in nets if n > 0]
        losses = [n for n in nets if n <= 0]

        cum = 0; peak = 0; dd = 0
        for n in nets:
            cum += n; peak = max(peak, cum); dd = min(dd, cum - peak)

        print(f"  Trades: {len(trades)}  |  Win rate: {len(wins)/len(trades)*100:.0f}%  |  "
              f"Shares/trade: {trades[0].get('shares', 1)}")
        print(f"  Net PnL: ${sum(nets):+,.2f}  |  Max DD: ${dd:+,.2f}")
        if wins:
            print(f"  Avg win: ${sum(wins)/len(wins):+,.2f}  |  Avg loss: ${sum(losses)/len(losses) if losses else 0:+,.2f}")
        print(f"  Account after: ${profile['account_size'] + sum(nets):,.2f}")
        print()
        print(f"  {'#':>3}  {'Time':>5}  {'Dir':>5}  {'Entry':>8}  {'Exit':>8}  "
              f"{'PnL/sh':>7}  {'Net$':>7}  {'Reason'}")
        print(f"  " + "-" * 65)
        for i, t in enumerate(trades, 1):
            tm = t["entry_ts"].split(" ")[1][:5]
            print(f"  {i:>3}  {tm:>5}  {t['dir']:>5}  ${t['entry']:>7.2f}  ${t['exit']:>7.2f}  "
                  f"{t['net_pts']:>+7.2f}  ${t['net_dollar']:>+6.2f}  {t['reason']}")

    elif mode == "0dte":
        nets = [t["net_dollar"] for t in trades]
        wins = [n for n in nets if n > 0]
        losses = [n for n in nets if n <= 0]

        cum = 0; peak = 0; dd = 0
        for n in nets:
            cum += n; peak = max(peak, cum); dd = min(dd, cum - peak)

        scale = profile.get("scale_on_trail", False)
        print(f"  Trades: {len(trades)}  |  Win rate: {len(wins)/len(trades)*100:.0f}%  |  "
              f"Scale-on-trail: {'ON' if scale else 'OFF'}")
        print(f"  Net PnL: ${sum(nets):+,.2f}  |  Max DD: ${dd:+,.2f}")
        if wins:
            print(f"  Avg win: ${sum(wins)/len(wins):+,.2f}  |  Avg loss: ${sum(losses)/len(losses) if losses else 0:+,.2f}")
        print()
        print(f"  {'#':>3}  {'Time':>5}  {'Type':>4}  {'K':>6}  {'Cts':>3}  {'SPX':>6}->{'':>6}  "
              f"{'Prem':>6}->{'':>6}  {'Net$':>9}  {'Reason'}")
        print(f"  " + "-" * 80)
        for i, t in enumerate(trades, 1):
            tm = t["entry_ts"].split(" ")[1][:5]
            print(f"  {i:>3}  {tm:>5}  {t['opt_type']:>4}  {t['strike']:>6.0f}  "
                  f"{t['contracts']:>3}  "
                  f"${t['spot_entry']:>5.0f}->${t['spot_exit']:>5.0f}  "
                  f"${t['opt_entry']:>5.2f}->${t['opt_exit']:>5.2f}  "
                  f"${t['net_dollar']:>+8.2f}  {t['reason']}")


# ─── Main ────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Simulate scalp bot across account profiles")
    parser.add_argument("--date", type=str, default=None)
    parser.add_argument("--log-dir", type=str, default="logs/scalp_bot")
    parser.add_argument("--accounts", type=str, default="robinhood,pm,etrade",
                        help="Comma-separated account names")
    args = parser.parse_args()

    log_dir = ROOT / args.log_dir
    date_str = args.date or datetime.now(ET).strftime("%Y%m%d")
    sig_path = log_dir / f"signals_{date_str}.csv"

    if not sig_path.exists():
        print(f"Error: {sig_path} not found")
        sys.exit(1)

    with open(sig_path) as f:
        rows = list(csv.DictReader(f))

    account_names = [a.strip() for a in args.accounts.split(",")]
    sims = {}

    print("=" * 65)
    print(f"ACCOUNT SIMULATION — {date_str} ({len(rows)} signals)")
    print(f"  Source: {sig_path.name}")
    print("=" * 65)

    # Separate signals by price range (log may contain interleaved tickers)
    spx_rows = [r for r in rows if float(r["spx_price"]) > 1000]
    other_rows = [r for r in rows if float(r["spx_price"]) <= 1000]
    print(f"  SPX signals: {len(spx_rows)}  |  Other ticker signals: {len(other_rows)}")

    for name in account_names:
        if name not in ACCOUNTS:
            print(f"  Unknown account: {name}")
            continue
        profile = ACCOUNTS[name]
        sim = AccountSim(profile)
        ticker = profile.get("ticker", "SPY")

        # SPX accounts use SPX rows directly; SPY accounts use SPX rows scaled
        source_rows = spx_rows

        for row in source_rows:
            r = dict(row)  # copy so we don't mutate shared data
            if ticker == "SPY":
                # Scale SPX price to SPY (~1/10)
                r["spx_price"] = str(round(float(r["spx_price"]) / 10, 2))
                if r.get("atr14") and float(r["atr14"]) > 0:
                    r["atr14"] = str(round(float(r["atr14"]) / 10, 4))
            sim.tick(r)

        if source_rows:
            last = dict(source_rows[-1])
            if ticker == "SPY":
                last["spx_price"] = str(round(float(last["spx_price"]) / 10, 2))
                if last.get("atr14") and float(last["atr14"]) > 0:
                    last["atr14"] = str(round(float(last["atr14"]) / 10, 4))
            sim.force_close(last)

        sims[name] = sim
        print_account(profile["name"], profile, sim)

    # Summary comparison
    print(f"\n{'=' * 65}")
    print("DAILY P&L SUMMARY")
    print(f"{'=' * 65}")
    print(f"  {'Account':<20} {'Trades':>6} {'Win%':>6} {'Net P&L':>12} {'of Account':>10}")
    print(f"  {'-' * 56}")
    for name in account_names:
        if name not in sims:
            continue
        sim = sims[name]
        p = ACCOUNTS[name]
        trades = sim.trades
        if not trades:
            print(f"  {p['name']:<20} {'0':>6} {'--':>6} {'$0.00':>12} {'--':>10}")
            continue
        nets = [t["net_dollar"] for t in trades]
        total = sum(nets)
        wins = len([n for n in nets if n > 0])
        wr = wins / len(trades) * 100
        pct = total / p["account_size"] * 100
        print(f"  {p['name']:<20} {len(trades):>6} {wr:>5.0f}% ${total:>+10,.2f} {pct:>+9.2f}%")
    print(f"{'=' * 65}")


if __name__ == "__main__":
    main()
