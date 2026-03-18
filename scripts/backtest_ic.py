#!/usr/bin/env python3
"""
Backtest all 3 strategies against signal data using the StrategyRouter.

Replays ticks through: Directional, Iron Condor, and Gamma Scalp.
Since GEX was UNKNOWN today, runs what-if scenarios for POSITIVE and NEGATIVE.

Usage:
    python scripts/backtest_ic.py [--signals logs/scalp_bot/signals_20260311.csv]
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from scripts.spx_scalp_bot import (
    ScalpSignal, TradeManager, IronCondorManager, GammaScalpManager,
    StrategyRouter,
)


def load_signals(path: str) -> list[ScalpSignal]:
    signals = []
    with open(path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            signals.append(ScalpSignal(
                timestamp=row["timestamp"],
                ticker=row["ticker"],
                spx_price=float(row["spx_price"]),
                vix=float(row["vix"]),
                gex_regime=row["gex_regime"],
                flip_line=float(row["flip_line"]),
                distance_to_flip=float(row["distance_to_flip"]),
                net_gex=float(row["net_gex"]),
                call_wall=float(row["call_wall"]),
                put_wall=float(row["put_wall"]),
                ema9=float(row["ema9"]),
                ema21=float(row["ema21"]),
                rsi14=float(row["rsi14"]),
                atr14=float(row["atr14"]),
                price_vs_vwap=float(row["price_vs_vwap"]),
                signal=row["signal"],
                reason=row["reason"],
                confidence=float(row["confidence"]),
            ))
    return signals


def override_signal(sig: ScalpSignal, gex_regime: str = None,
                    vix_override: float = None) -> ScalpSignal:
    """Return a copy of sig with overridden GEX regime and/or VIX."""
    return ScalpSignal(
        timestamp=sig.timestamp, ticker=sig.ticker,
        spx_price=sig.spx_price,
        vix=vix_override if (vix_override and sig.vix <= 0) else sig.vix,
        gex_regime=gex_regime if gex_regime else sig.gex_regime,
        flip_line=sig.flip_line, distance_to_flip=sig.distance_to_flip,
        net_gex=sig.net_gex, call_wall=sig.call_wall, put_wall=sig.put_wall,
        ema9=sig.ema9, ema21=sig.ema21, rsi14=sig.rsi14, atr14=sig.atr14,
        price_vs_vwap=sig.price_vs_vwap, signal=sig.signal,
        reason=sig.reason, confidence=sig.confidence,
    )


def run_scenario(signals: list[ScalpSignal], gex_override: str = None,
                 vix_default: float = None, label: str = ""):
    """Run the full StrategyRouter against signals and print results."""
    log_dir = ROOT / "logs" / "scalp_bot" / "backtest"
    log_dir.mkdir(parents=True, exist_ok=True)

    # Clean previous backtest logs
    for f in log_dir.glob("*.csv"):
        f.unlink()

    profile = {"ticker": "XSP", "mode": "0dte", "min_confidence": 0.60,
               "multiplier": 100, "contracts": 5, "max_premium": 300,
               "min_premium": 0.20, "option_tp_pct": 50, "option_sl_pct": 35,
               "option_trail_pct": 25, "max_hold_minutes": 30,
               "min_tte_minutes": 30, "commission_per_side": 0.65,
               "cooldown_sec": 300, "max_trades_per_day": 6,
               "scale_on_trail": False}

    trade_mgr = TradeManager(log_dir, profile=profile, ibkr=None)
    ic_mgr = IronCondorManager(log_dir, ibkr=None, profile=profile)
    gs_mgr = GammaScalpManager(log_dir, ibkr=None, profile=profile)
    router = StrategyRouter(trade_mgr, ic_mgr, gs_mgr)

    for sig in signals:
        if gex_override or vix_default:
            sig = override_signal(sig, gex_regime=gex_override, vix_override=vix_default)
        router.process_tick(sig)

    # Force close all at end
    if signals:
        last = signals[-1]
        if gex_override or vix_default:
            last = override_signal(last, gex_regime=gex_override, vix_override=vix_default)
        router.force_close_all(last)
        # Also force-close directional trade
        if trade_mgr.open_trade:
            trade_mgr.MAX_HOLD_MINUTES = 0
            if trade_mgr.mode == "0dte":
                trade_mgr._check_exit_option(last)
            else:
                trade_mgr._check_exit(last)

    # Print results
    print(f"\n{'='*60}")
    print(f"  {label}")
    print(f"{'='*60}")
    if signals:
        prices = [s.spx_price for s in signals]
        print(f"  Period: {signals[0].timestamp} → {signals[-1].timestamp}")
        print(f"  Ticks: {len(signals)}  |  XSP: {min(prices):.2f}–{max(prices):.2f} "
              f"(${max(prices)-min(prices):.2f} range)")

    # Directional
    ds = trade_mgr.stats
    if ds["total"]:
        wr = ds["wins"] / ds["total"] * 100
        print(f"\n  DIRECTIONAL: {ds['total']} trades, {ds['wins']}W/{ds['losses']}L "
              f"({wr:.0f}% WR)  PnL: ${ds['total_pnl']:+,.2f}")
    else:
        print(f"\n  DIRECTIONAL: 0 trades")

    # Iron Condor
    ics = ic_mgr.stats
    if ics["total"]:
        wr = ics["wins"] / ics["total"] * 100
        print(f"  IRON CONDOR: {ics['total']} trades, {ics['wins']}W/{ics['losses']}L "
              f"({wr:.0f}% WR)  PnL: ${ics['total_pnl']:+,.2f}")
        # Show IC details
        ic_log = log_dir / ic_mgr.ic_log_path.name
        if ic_log.exists():
            with open(ic_log) as f:
                for row in csv.DictReader(f):
                    print(f"    {row['entry_time'][:16]} → {row['exit_time'][:16]}  "
                          f"P{row['put_short']}/P{row['put_long']} "
                          f"C{row['call_short']}/C{row['call_long']}  "
                          f"credit=${row['credit']} → debit=${row['exit_debit']}  "
                          f"net=${row['net_pnl']}  [{row['exit_reason']}]")
    else:
        print(f"  IRON CONDOR: 0 trades")

    # Gamma Scalp
    gss = gs_mgr.stats
    if gss["total"]:
        wr = gss["wins"] / gss["total"] * 100
        print(f"  GAMMA SCALP: {gss['total']} trades, {gss['wins']}W/{gss['losses']}L "
              f"({wr:.0f}% WR)  Rebalances: {gss['rebalances']}  "
              f"PnL: ${gss['total_pnl']:+,.2f}")
        gs_log = log_dir / gs_mgr.gs_log_path.name
        if gs_log.exists():
            with open(gs_log) as f:
                for row in csv.DictReader(f):
                    print(f"    {row['entry_time'][:16]} → {row['exit_time'][:16]}  "
                          f"K={row['strike']}  cost=${row['total_cost']}  "
                          f"hedges={row['hedge_count']}  "
                          f"net=${row['net_pnl']}  [{row['exit_reason']}]")
    else:
        print(f"  GAMMA SCALP: 0 trades")

    combined = ds["total_pnl"] + ics["total_pnl"] + gss["total_pnl"]
    print(f"\n  COMBINED PnL: ${combined:+,.2f}")
    print(f"{'='*60}")

    return combined


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Backtest all strategies")
    parser.add_argument("--signals", default="logs/scalp_bot/signals_20260311.csv")
    args = parser.parse_args()

    sig_path = ROOT / args.signals
    if not sig_path.exists():
        print(f"Signal file not found: {sig_path}")
        sys.exit(1)

    signals = load_signals(str(sig_path))
    print(f"Loaded {len(signals)} signals from {sig_path.name}")

    # Scenario 1: Actual data (GEX=UNKNOWN, no strategies fire)
    run_scenario(signals, label="SCENARIO 1: Actual (GEX=UNKNOWN)")

    # Scenario 2: Positive GEX → IC should fire
    run_scenario(signals, gex_override="POSITIVE", vix_default=24.5,
                 label="SCENARIO 2: What-if POSITIVE GEX → Iron Condor")

    # Scenario 3: Negative GEX → Gamma Scalp should fire
    run_scenario(signals, gex_override="NEGATIVE", vix_default=24.5,
                 label="SCENARIO 3: What-if NEGATIVE GEX → Gamma Scalp")

    # Scenario 4: Mixed — first half positive, second half negative
    mid = len(signals) // 2
    mixed = []
    for i, sig in enumerate(signals):
        if i < mid:
            mixed.append(override_signal(sig, gex_regime="POSITIVE", vix_override=24.5))
        else:
            mixed.append(override_signal(sig, gex_regime="NEGATIVE", vix_override=24.5))

    # Run mixed manually
    from scripts.spx_scalp_bot import TradeManager as TM, IronCondorManager as ICM, GammaScalpManager as GSM
    log_dir = ROOT / "logs" / "scalp_bot" / "backtest"
    for f in log_dir.glob("*.csv"):
        f.unlink()
    profile = {"ticker": "XSP", "mode": "0dte", "min_confidence": 0.60,
               "multiplier": 100, "contracts": 5, "max_premium": 300,
               "min_premium": 0.20, "option_tp_pct": 50, "option_sl_pct": 35,
               "option_trail_pct": 25, "max_hold_minutes": 30,
               "min_tte_minutes": 30, "commission_per_side": 0.65,
               "cooldown_sec": 300, "max_trades_per_day": 6,
               "scale_on_trail": False}
    tm = TM(log_dir, profile=profile)
    im = ICM(log_dir, profile=profile)
    gm = GSM(log_dir, profile=profile)
    rt = StrategyRouter(tm, im, gm)
    for sig in mixed:
        rt.process_tick(sig)
    if mixed:
        rt.force_close_all(mixed[-1])

    print(f"\n{'='*60}")
    print(f"  SCENARIO 4: Mixed (POSITIVE 1st half → NEGATIVE 2nd half)")
    print(f"{'='*60}")
    print(f"  Pivot at tick {mid}: {signals[mid].timestamp}")
    ds, ics, gss = tm.stats, im.stats, gm.stats
    for name, s in [("DIRECTIONAL", ds), ("IRON CONDOR", ics), ("GAMMA SCALP", gss)]:
        if s["total"]:
            wr = s["wins"] / s["total"] * 100
            extra = f"  Rebalances: {s['rebalances']}" if "rebalances" in s else ""
            print(f"  {name}: {s['total']} trades, {s['wins']}W/{s['losses']}L "
                  f"({wr:.0f}% WR){extra}  PnL: ${s['total_pnl']:+,.2f}")
        else:
            print(f"  {name}: 0 trades")
    combined = ds["total_pnl"] + ics["total_pnl"] + gss["total_pnl"]
    print(f"\n  COMBINED PnL: ${combined:+,.2f}")
    print(f"{'='*60}")
