#!/usr/bin/env python3
"""
Per-pair parameter optimizer for FX Swing Bot.

For each pair, sweeps carry_weight, sl_atr_mult, trail params, max_hold_days,
and entry_threshold to find the combination that maximizes Sharpe ratio.
"""

import itertools
import math
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from scripts.fx_swing_bot import (
    PAIRS, POLICY_RATES, PairConfig,
    TREND_FAST_MA, TREND_SLOW_MA, TREND_CONFIRM_MA,
    VIX_CARRY_REDUCE, VIX_NO_ENTRY, ATR_PERIOD,
    compute_carry_signal,
)

IBKR_SWAP_MARKUP = 1.0  # ~1% markup on benchmark rate


@dataclass
class ParamSet:
    carry_weight: float
    entry_threshold: float
    sl_atr_mult: float
    trail_activate_atr: float
    trail_distance_atr: float
    max_hold_days: int


def backtest_pair_params(pair_cfg: PairConfig, df: pd.DataFrame,
                         vix_df: pd.DataFrame, params: ParamSet) -> dict:
    """Backtest one pair with specific parameters. Returns metrics dict."""
    results = []
    position = None

    closes = df["Close"]
    sma_fast = closes.rolling(TREND_FAST_MA).mean()
    sma_slow = closes.rolling(TREND_SLOW_MA).mean()
    sma_200 = closes.rolling(TREND_CONFIRM_MA).mean()

    high = df["High"]
    low = df["Low"]
    prev_close = closes.shift(1)
    tr = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low - prev_close).abs(),
    ], axis=1).max(axis=1)
    atr_series = tr.rolling(ATR_PERIOD).mean()

    carry_sig, carry_rate = compute_carry_signal(pair_cfg)
    trend_weight = 1.0 - params.carry_weight

    start_idx = max(TREND_CONFIRM_MA, ATR_PERIOD) + 1

    for i in range(start_idx, len(df)):
        date = df.index[i]
        price = float(closes.iloc[i])
        atr = float(atr_series.iloc[i])
        sf = float(sma_fast.iloc[i])
        ss = float(sma_slow.iloc[i])
        s200 = float(sma_200.iloc[i])

        if atr <= 0 or ss <= 0:
            continue

        # VIX
        vix = 20.0
        if date in vix_df.index:
            vix = float(vix_df.loc[date, "Close"])
        else:
            prior = vix_df.index[vix_df.index <= date]
            if len(prior) > 0:
                vix = float(vix_df.loc[prior[-1], "Close"])

        cw = params.carry_weight
        tw = trend_weight
        if vix > VIX_CARRY_REDUCE:
            cw = cw * 0.5
            tw = 1.0 - cw

        cross_pct = (sf - ss) / ss
        trend_raw = math.tanh(cross_pct * 50)
        if (trend_raw > 0 and price < s200) or (trend_raw < 0 and price > s200):
            trend_raw *= 0.5

        combined = carry_sig * cw + trend_raw * tw

        # Check exit
        if position:
            direction, entry_price, entry_idx, pos_atr, sl, trail, hw = position
            is_long = direction == "LONG"
            pnl_price = (price - entry_price) if is_long else (entry_price - price)
            hold_days = i - entry_idx

            if is_long:
                hw = max(hw, price)
            else:
                hw = min(hw, price)

            exit_reason = None

            # Stop loss
            if is_long and price <= sl:
                exit_reason = "STOP_LOSS"
            elif not is_long and price >= sl:
                exit_reason = "STOP_LOSS"

            # Trailing stop
            if not exit_reason:
                activate_dist = params.trail_activate_atr * pos_atr
                if pnl_price >= activate_dist:
                    if is_long:
                        new_trail = hw - params.trail_distance_atr * pos_atr
                        if trail is None or new_trail > trail:
                            trail = new_trail
                        if price <= trail:
                            exit_reason = "TRAIL_STOP"
                    else:
                        new_trail = hw + params.trail_distance_atr * pos_atr
                        if trail is None or new_trail < trail:
                            trail = new_trail
                        if price >= trail:
                            exit_reason = "TRAIL_STOP"

            # Time exit
            if not exit_reason and hold_days >= params.max_hold_days:
                exit_reason = "TIME_EXIT"

            # Signal flip
            if not exit_reason:
                if is_long and combined < -params.entry_threshold:
                    exit_reason = "SIGNAL_FLIP"
                elif not is_long and combined > params.entry_threshold:
                    exit_reason = "SIGNAL_FLIP"

            if exit_reason:
                if pair_cfg.quote == "USD":
                    pnl_dollar = pnl_price * 20000
                else:
                    pnl_dollar = (pnl_price * 20000) / price

                # Carry income
                base_rate = POLICY_RATES.get(pair_cfg.base, 0)
                quote_rate = POLICY_RATES.get(pair_cfg.quote, 0)
                if is_long:
                    earn_rate = max(0, base_rate - IBKR_SWAP_MARKUP)
                    pay_rate = quote_rate + IBKR_SWAP_MARKUP
                else:
                    earn_rate = max(0, quote_rate - IBKR_SWAP_MARKUP)
                    pay_rate = base_rate + IBKR_SWAP_MARKUP
                net_swap = (earn_rate - pay_rate) / 100
                if pair_cfg.quote == "USD":
                    notional_usd = 20000 * entry_price
                else:
                    notional_usd = 20000
                carry_income = net_swap * notional_usd * hold_days / 365
                pnl_dollar += carry_income

                results.append(pnl_dollar)
                position = None
            else:
                position = (direction, entry_price, entry_idx, pos_atr, sl, trail, hw)

        # Entry
        if position is None and vix <= VIX_NO_ENTRY:
            if combined > params.entry_threshold:
                direction = "LONG"
            elif combined < -params.entry_threshold:
                direction = "SHORT"
            else:
                continue

            if direction == "LONG":
                sl = price - params.sl_atr_mult * atr
            else:
                sl = price + params.sl_atr_mult * atr
            position = (direction, price, i, atr, sl, None, price)

    # Close remaining
    if position:
        direction, entry_price, entry_idx, pos_atr, sl, trail, hw = position
        price = float(closes.iloc[-1])
        pnl_price = (price - entry_price) if direction == "LONG" else (entry_price - price)
        hold_days = len(df) - 1 - entry_idx
        if pair_cfg.quote == "USD":
            pnl_dollar = pnl_price * 20000
        else:
            pnl_dollar = (pnl_price * 20000) / price
        # Carry
        is_long = direction == "LONG"
        base_rate = POLICY_RATES.get(pair_cfg.base, 0)
        quote_rate = POLICY_RATES.get(pair_cfg.quote, 0)
        if is_long:
            earn_rate = max(0, base_rate - IBKR_SWAP_MARKUP)
            pay_rate = quote_rate + IBKR_SWAP_MARKUP
        else:
            earn_rate = max(0, quote_rate - IBKR_SWAP_MARKUP)
            pay_rate = base_rate + IBKR_SWAP_MARKUP
        net_swap = (earn_rate - pay_rate) / 100
        if pair_cfg.quote == "USD":
            notional_usd = 20000 * entry_price
        else:
            notional_usd = 20000
        carry_income = net_swap * notional_usd * hold_days / 365
        pnl_dollar += carry_income
        results.append(pnl_dollar)

    if len(results) < 5:
        return {"sharpe": -999, "total_pnl": 0, "trades": len(results),
                "win_rate": 0, "max_dd": 0, "avg_pnl": 0}

    pnls = np.array(results)
    n = len(pnls)
    total = pnls.sum()
    avg = pnls.mean()
    std = pnls.std()
    wins = (pnls > 0).sum()
    trades_per_year = n / 5
    sharpe = (avg / std) * np.sqrt(trades_per_year) if std > 0 else 0

    equity = np.cumsum(pnls)
    running_max = np.maximum.accumulate(equity)
    dd = equity - running_max
    max_dd = dd.min()

    return {
        "sharpe": round(sharpe, 3),
        "total_pnl": round(total, 2),
        "trades": n,
        "win_rate": round(wins / n * 100, 1),
        "max_dd": round(max_dd, 2),
        "avg_pnl": round(avg, 2),
    }


def optimize_pair(pair_name: str, pair_cfg: PairConfig,
                  df: pd.DataFrame, vix_df: pd.DataFrame) -> tuple:
    """Grid search for best parameters on one pair."""

    # Parameter grid
    carry_weights = [0.30, 0.40, 0.50, 0.60, 0.70]
    entry_thresholds = [0.15, 0.20, 0.25, 0.30, 0.35]
    sl_atr_mults = [2.0, 2.5, 3.0, 3.5]
    trail_activates = [1.0, 1.5, 2.0]
    trail_distances = [0.8, 1.0, 1.5]
    max_holds = [20, 30, 45]

    best_sharpe = -999
    best_params = None
    best_metrics = None
    total_combos = (len(carry_weights) * len(entry_thresholds) *
                    len(sl_atr_mults) * len(trail_activates) *
                    len(trail_distances) * len(max_holds))
    tested = 0

    for cw, et, sl, ta, td, mh in itertools.product(
        carry_weights, entry_thresholds, sl_atr_mults,
        trail_activates, trail_distances, max_holds
    ):
        params = ParamSet(
            carry_weight=cw, entry_threshold=et,
            sl_atr_mult=sl, trail_activate_atr=ta,
            trail_distance_atr=td, max_hold_days=mh,
        )
        metrics = backtest_pair_params(pair_cfg, df, vix_df, params)
        tested += 1

        # Optimize for Sharpe, but require at least 15 trades and positive PnL
        if (metrics["sharpe"] > best_sharpe
                and metrics["trades"] >= 15
                and metrics["total_pnl"] > 0):
            best_sharpe = metrics["sharpe"]
            best_params = params
            best_metrics = metrics

    return best_params, best_metrics, tested


def main():
    pairs_to_optimize = ["EURUSD", "USDJPY", "GBPUSD", "USDMXN"]
    period = "5y"

    print("=" * 70)
    print("FX SWING BOT — PER-PAIR PARAMETER OPTIMIZATION")
    print("=" * 70)
    print(f"\nDownloading {period} daily data...")

    vix_df = yf.Ticker("^VIX").history(period=period, interval="1d")
    print(f"  VIX: {len(vix_df)} bars")

    all_best = {}

    for pair_name in pairs_to_optimize:
        pair_cfg = PAIRS[pair_name]
        df = yf.Ticker(pair_cfg.yf_symbol).history(period=period, interval="1d")
        print(f"\n{'─' * 50}")
        print(f"  {pair_name}: {len(df)} bars")

        if len(df) < TREND_CONFIRM_MA + 50:
            print(f"    Insufficient data, skipping")
            continue

        # First show baseline (global defaults)
        from scripts.fx_swing_bot import (
            CARRY_WEIGHT, TREND_WEIGHT, ENTRY_THRESHOLD,
            SL_ATR_MULT, TRAIL_ACTIVATE_ATR, TRAIL_DISTANCE_ATR, MAX_HOLD_DAYS,
        )
        baseline = ParamSet(
            carry_weight=CARRY_WEIGHT, entry_threshold=ENTRY_THRESHOLD,
            sl_atr_mult=SL_ATR_MULT, trail_activate_atr=TRAIL_ACTIVATE_ATR,
            trail_distance_atr=TRAIL_DISTANCE_ATR, max_hold_days=MAX_HOLD_DAYS,
        )
        baseline_metrics = backtest_pair_params(pair_cfg, df, vix_df, baseline)
        print(f"  BASELINE: Sharpe={baseline_metrics['sharpe']:.2f} "
              f"PnL=${baseline_metrics['total_pnl']:+,.0f} "
              f"Trades={baseline_metrics['trades']} "
              f"WR={baseline_metrics['win_rate']:.0f}% "
              f"MaxDD=${baseline_metrics['max_dd']:+,.0f}")

        # Optimize
        print(f"  Optimizing...", end=" ", flush=True)
        best_params, best_metrics, combos = optimize_pair(
            pair_name, pair_cfg, df, vix_df
        )
        print(f"({combos} combinations tested)")

        if best_params and best_metrics:
            all_best[pair_name] = (best_params, best_metrics)
            improvement = best_metrics['sharpe'] - baseline_metrics['sharpe']
            print(f"  OPTIMIZED: Sharpe={best_metrics['sharpe']:.2f} "
                  f"(+{improvement:.2f}) "
                  f"PnL=${best_metrics['total_pnl']:+,.0f} "
                  f"Trades={best_metrics['trades']} "
                  f"WR={best_metrics['win_rate']:.0f}% "
                  f"MaxDD=${best_metrics['max_dd']:+,.0f}")
            print(f"    carry_weight={best_params.carry_weight:.2f} "
                  f"entry_threshold={best_params.entry_threshold:.2f} "
                  f"sl_atr={best_params.sl_atr_mult:.1f} "
                  f"trail_act={best_params.trail_activate_atr:.1f} "
                  f"trail_dist={best_params.trail_distance_atr:.1f} "
                  f"max_hold={best_params.max_hold_days}")
        else:
            print(f"  No valid parameter set found!")

    # Summary
    print(f"\n{'=' * 70}")
    print("OPTIMIZATION SUMMARY — Copy these into fx_swing_bot.py PAIRS dict:")
    print("=" * 70)

    for pair_name, (params, metrics) in all_best.items():
        cfg = PAIRS[pair_name]
        print(f'\n    "{pair_name}": PairConfig("{pair_name}", "{cfg.base}", "{cfg.quote}", '
              f'"{cfg.yf_symbol}", "{cfg.ib_symbol}", {cfg.pip_size}, {cfg.min_qty}, {cfg.qty_decimals},')
        print(f'        carry_weight={params.carry_weight}, '
              f'trend_weight={1-params.carry_weight:.2f}, '
              f'entry_threshold={params.entry_threshold},')
        print(f'        sl_atr_mult={params.sl_atr_mult}, '
              f'trail_activate_atr={params.trail_activate_atr}, '
              f'trail_distance_atr={params.trail_distance_atr}, '
              f'max_hold_days={params.max_hold_days}),')
        print(f'        # Sharpe={metrics["sharpe"]:.2f} '
              f'PnL=${metrics["total_pnl"]:+,.0f} '
              f'Trades={metrics["trades"]} WR={metrics["win_rate"]:.0f}%')

    # Combined portfolio simulation with optimized params
    if len(all_best) > 1:
        print(f"\n{'=' * 70}")
        print("COMBINED PORTFOLIO (optimized per-pair params)")
        print("=" * 70)
        all_pnls = []
        for pair_name, (params, _) in all_best.items():
            pair_cfg = PAIRS[pair_name]
            df = yf.Ticker(pair_cfg.yf_symbol).history(period=period, interval="1d")
            metrics = backtest_pair_params(pair_cfg, df, vix_df, params)
            # We need the actual trade PnLs for combined analysis
            # Re-run to get them (the function returns summary, but we need PnL list)
            # For now, just sum the metrics
            print(f"  {pair_name}: Sharpe={metrics['sharpe']:.2f} "
                  f"PnL=${metrics['total_pnl']:+,.0f} "
                  f"({metrics['trades']} trades)")
            all_pnls.append(metrics['total_pnl'])

        total = sum(all_pnls)
        print(f"\n  Combined PnL: ${total:+,.0f}")
        print(f"  Per year: ${total/5:+,.0f}")


if __name__ == "__main__":
    main()
