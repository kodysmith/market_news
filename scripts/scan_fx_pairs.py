#!/usr/bin/env python3
"""
Comprehensive FX Pair Scanner — find best carry+trend opportunities.

Scans 30 FX pairs (G10 + EM + crosses) with parameter optimization.
Reports ranked results by Sharpe ratio.
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
    TREND_FAST_MA, TREND_SLOW_MA, TREND_CONFIRM_MA,
    VIX_CARRY_REDUCE, VIX_NO_ENTRY, ATR_PERIOD,
)

# ─── Central Bank Rates (March 2026) ────────────────────────────────────────
RATES = {
    "USD": 3.625, "EUR": 2.00, "JPY": 0.75, "GBP": 4.50,
    "AUD": 4.10, "NZD": 3.75, "CAD": 2.75, "CHF": 0.50,
    "NOK": 3.25, "SEK": 2.50,
    # EM
    "MXN": 7.00, "ZAR": 7.25, "BRL": 12.25, "PLN": 5.25,
    "CZK": 3.50, "HUF": 6.50, "SGD": 3.00, "CNY": 3.10,
    "INR": 6.00, "TRY": 42.50,
}

# IBKR swap markup: 1% for G10, 2% for EM exotics
G10 = {"USD", "EUR", "JPY", "GBP", "AUD", "NZD", "CAD", "CHF", "NOK", "SEK"}

def swap_markup(ccy1, ccy2):
    if ccy1 in G10 and ccy2 in G10:
        return 1.0
    return 2.0

# ─── All pairs to scan ──────────────────────────────────────────────────────
SCAN_PAIRS = [
    # G10 majors
    ("EURUSD", "EUR", "USD", "EURUSD=X"),
    ("USDJPY", "USD", "JPY", "USDJPY=X"),
    ("GBPUSD", "GBP", "USD", "GBPUSD=X"),
    ("AUDUSD", "AUD", "USD", "AUDUSD=X"),
    ("NZDUSD", "NZD", "USD", "NZDUSD=X"),
    ("USDCAD", "USD", "CAD", "USDCAD=X"),
    ("USDCHF", "USD", "CHF", "USDCHF=X"),
    # G10 crosses
    ("EURGBP", "EUR", "GBP", "EURGBP=X"),
    ("EURJPY", "EUR", "JPY", "EURJPY=X"),
    ("GBPJPY", "GBP", "JPY", "GBPJPY=X"),
    ("AUDJPY", "AUD", "JPY", "AUDJPY=X"),
    ("NZDJPY", "NZD", "JPY", "NZDJPY=X"),
    ("EURCHF", "EUR", "CHF", "EURCHF=X"),
    ("GBPCHF", "GBP", "CHF", "GBPCHF=X"),
    ("AUDNZD", "AUD", "NZD", "AUDNZD=X"),
    # Scandies
    ("EURNOK", "EUR", "NOK", "EURNOK=X"),
    ("EURSEK", "EUR", "SEK", "EURSEK=X"),
    ("USDNOK", "USD", "NOK", "USDNOK=X"),
    ("USDSEK", "USD", "SEK", "USDSEK=X"),
    # EM
    ("USDMXN", "USD", "MXN", "USDMXN=X"),
    ("USDZAR", "USD", "ZAR", "USDZAR=X"),
    ("USDBRL", "USD", "BRL", "USDBRL=X"),
    ("USDPLN", "USD", "PLN", "USDPLN=X"),
    ("USDCZK", "USD", "CZK", "USDCZK=X"),
    ("USDHUF", "USD", "HUF", "USDHUF=X"),
    ("USDSGD", "USD", "SGD", "USDSGD=X"),
    ("USDCNY", "USD", "CNY", "USDCNY=X"),
    ("USDINR", "USD", "INR", "USDINR=X"),
    ("USDTRY", "USD", "TRY", "USDTRY=X"),
]


@dataclass
class ParamSet:
    carry_weight: float
    entry_threshold: float
    sl_atr_mult: float
    trail_activate_atr: float
    trail_distance_atr: float
    max_hold_days: int


def compute_carry_signal(base, quote):
    base_rate = RATES.get(base, 0)
    quote_rate = RATES.get(quote, 0)
    diff = base_rate - quote_rate
    signal = math.tanh(diff / 2.0)
    return signal, diff


def backtest(base, quote, df, vix_df, params, notional=20000):
    """Backtest carry+trend on one pair with given params. Returns PnL list."""
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
        high - low, (high - prev_close).abs(), (low - prev_close).abs(),
    ], axis=1).max(axis=1)
    atr_series = tr.rolling(ATR_PERIOD).mean()

    carry_sig, carry_rate = compute_carry_signal(base, quote)
    trend_weight = 1.0 - params.carry_weight
    markup = swap_markup(base, quote)

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

            if is_long and price <= sl:
                exit_reason = "SL"
            elif not is_long and price >= sl:
                exit_reason = "SL"

            if not exit_reason:
                act_dist = params.trail_activate_atr * pos_atr
                if pnl_price >= act_dist:
                    if is_long:
                        new_trail = hw - params.trail_distance_atr * pos_atr
                        if trail is None or new_trail > trail:
                            trail = new_trail
                        if price <= trail:
                            exit_reason = "TRAIL"
                    else:
                        new_trail = hw + params.trail_distance_atr * pos_atr
                        if trail is None or new_trail < trail:
                            trail = new_trail
                        if price >= trail:
                            exit_reason = "TRAIL"

            if not exit_reason and hold_days >= params.max_hold_days:
                exit_reason = "TIME"

            if not exit_reason:
                if is_long and combined < -params.entry_threshold:
                    exit_reason = "FLIP"
                elif not is_long and combined > params.entry_threshold:
                    exit_reason = "FLIP"

            if exit_reason:
                # PnL calculation
                if quote == "USD":
                    pnl_dollar = pnl_price * notional
                else:
                    pnl_dollar = (pnl_price * notional) / price

                # Carry/swap income with IBKR markup
                base_rate = RATES.get(base, 0)
                quote_rate = RATES.get(quote, 0)
                if is_long:
                    earn = max(0, base_rate - markup)
                    pay = quote_rate + markup
                else:
                    earn = max(0, quote_rate - markup)
                    pay = base_rate + markup
                net_swap = (earn - pay) / 100
                if quote == "USD":
                    notional_usd = notional * entry_price
                else:
                    notional_usd = notional
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
        if quote == "USD":
            pnl_dollar = pnl_price * notional
        else:
            pnl_dollar = (pnl_price * notional) / price
        is_long = direction == "LONG"
        base_rate = RATES.get(base, 0)
        quote_rate = RATES.get(quote, 0)
        if is_long:
            earn = max(0, base_rate - markup)
            pay = quote_rate + markup
        else:
            earn = max(0, quote_rate - markup)
            pay = base_rate + markup
        net_swap = (earn - pay) / 100
        if quote == "USD":
            notional_usd = notional * entry_price
        else:
            notional_usd = notional
        carry_income = net_swap * notional_usd * hold_days / 365
        pnl_dollar += carry_income
        results.append(pnl_dollar)

    return results


def optimize_pair(name, base, quote, df, vix_df):
    """Grid search best params for one pair."""
    carry_weights = [0.30, 0.40, 0.50, 0.60]
    entry_thresholds = [0.20, 0.25, 0.35]
    sl_atr_mults = [2.0, 2.5, 3.5]
    trail_activates = [1.0, 2.0]
    trail_distances = [0.8, 1.5]
    max_holds = [20, 30, 45]

    best_sharpe = -999
    best_params = None
    best_metrics = None

    for cw, et, sl, ta, td, mh in itertools.product(
        carry_weights, entry_thresholds, sl_atr_mults,
        trail_activates, trail_distances, max_holds
    ):
        params = ParamSet(cw, et, sl, ta, td, mh)
        pnls = backtest(base, quote, df, vix_df, params)

        if len(pnls) < 8:
            continue

        arr = np.array(pnls)
        total = arr.sum()
        avg = arr.mean()
        std = arr.std()
        if std <= 0 or total <= 0:
            continue

        n = len(arr)
        trades_per_year = n / 5
        sharpe = (avg / std) * np.sqrt(trades_per_year)

        equity = np.cumsum(arr)
        max_dd = (equity - np.maximum.accumulate(equity)).min()

        if sharpe > best_sharpe:
            best_sharpe = sharpe
            best_params = params
            best_metrics = {
                "sharpe": round(sharpe, 3),
                "total_pnl": round(total, 2),
                "trades": n,
                "win_rate": round((arr > 0).sum() / n * 100, 1),
                "max_dd": round(max_dd, 2),
                "avg_pnl": round(avg, 2),
                "pnl_per_year": round(total / 5, 2),
            }

    return best_params, best_metrics


def main():
    print("=" * 70)
    print("COMPREHENSIVE FX PAIR SCAN — Carry + Trend Strategy")
    print("=" * 70)

    print("\nDownloading VIX data...")
    vix_df = yf.Ticker("^VIX").history(period="5y", interval="1d")
    print(f"  VIX: {len(vix_df)} bars")

    current_pairs = {"EURUSD", "USDJPY", "GBPUSD", "USDMXN"}
    results_list = []

    for name, base, quote, yf_sym in SCAN_PAIRS:
        print(f"\n  Scanning {name}...", end=" ", flush=True)

        # Check if rates available
        if base not in RATES or quote not in RATES:
            print(f"SKIP (no rate for {base} or {quote})")
            continue

        try:
            df = yf.Ticker(yf_sym).history(period="5y", interval="1d")
        except Exception as e:
            print(f"SKIP (download failed: {e})")
            continue

        if len(df) < TREND_CONFIRM_MA + 50:
            print(f"SKIP ({len(df)} bars, need {TREND_CONFIRM_MA + 50})")
            continue

        carry_sig, carry_diff = compute_carry_signal(base, quote)
        best_params, best_metrics = optimize_pair(name, base, quote, df, vix_df)

        if best_params and best_metrics:
            tag = "CURRENT" if name in current_pairs else "NEW"
            print(f"Sharpe={best_metrics['sharpe']:.2f} "
                  f"PnL=${best_metrics['total_pnl']:+,.0f} "
                  f"({best_metrics['trades']} trades, "
                  f"WR={best_metrics['win_rate']:.0f}%) [{tag}]")

            results_list.append({
                "pair": name,
                "base": base,
                "quote": quote,
                "carry_diff": round(carry_diff, 2),
                "markup": swap_markup(base, quote),
                **best_metrics,
                "cw": best_params.carry_weight,
                "et": best_params.entry_threshold,
                "sl": best_params.sl_atr_mult,
                "ta": best_params.trail_activate_atr,
                "td": best_params.trail_distance_atr,
                "mh": best_params.max_hold_days,
                "in_portfolio": name in current_pairs,
            })
        else:
            print(f"NO VIABLE PARAMS (insufficient trades or negative PnL)")

    if not results_list:
        print("\nNo viable pairs found!")
        return

    # Rank by Sharpe
    results_df = pd.DataFrame(results_list).sort_values("sharpe", ascending=False)

    print("\n" + "=" * 70)
    print("RANKED RESULTS (by Sharpe ratio)")
    print("=" * 70)
    print(f"\n{'Rank':<5} {'Pair':<8} {'Sharpe':>7} {'PnL/yr':>10} {'Trades':>7} "
          f"{'WR%':>5} {'MaxDD':>10} {'Carry%':>7} {'Markup':>7} {'Status':<10}")
    print("-" * 80)

    for rank, (_, row) in enumerate(results_df.iterrows(), 1):
        status = "CURRENT" if row["in_portfolio"] else "NEW"
        print(f"{rank:<5} {row['pair']:<8} {row['sharpe']:>7.2f} "
              f"${row['pnl_per_year']:>+9,.0f} {row['trades']:>7} "
              f"{row['win_rate']:>5.0f} ${row['max_dd']:>+9,.0f} "
              f"{row['carry_diff']:>+6.1f}% {row['markup']:>5.0f}% {status:<10}")

    # Show new opportunities with Sharpe > 0.50
    new_good = results_df[(~results_df["in_portfolio"]) & (results_df["sharpe"] > 0.50)]
    if not new_good.empty:
        print(f"\n{'=' * 70}")
        print("NEW OPPORTUNITIES (Sharpe > 0.50, not in current portfolio)")
        print("=" * 70)
        for _, row in new_good.iterrows():
            print(f"\n  {row['pair']}: Sharpe={row['sharpe']:.2f} "
                  f"PnL/yr=${row['pnl_per_year']:+,.0f} "
                  f"WR={row['win_rate']:.0f}% "
                  f"MaxDD=${row['max_dd']:+,.0f}")
            print(f"    Carry diff: {row['carry_diff']:+.2f}% "
                  f"(IBKR markup: {row['markup']:.0f}%)")
            print(f"    Optimal params: cw={row['cw']:.2f} et={row['et']:.2f} "
                  f"sl={row['sl']:.1f} trail_act={row['ta']:.1f} "
                  f"trail_dist={row['td']:.1f} max_hold={row['mh']}")
            print(f"    PairConfig snippet:")
            print(f'    "{row["pair"]}": PairConfig("{row["pair"]}", '
                  f'"{row["base"]}", "{row["quote"]}", '
                  f'"{row["pair"]}=X", "{row["pair"]}", '
                  f'0.0001, 20000, 0,')
            print(f'        carry_weight={row["cw"]}, '
                  f'trend_weight={1-row["cw"]:.2f}, '
                  f'entry_threshold={row["et"]},')
            print(f'        sl_atr_mult={row["sl"]}, '
                  f'trail_activate_atr={row["ta"]}, '
                  f'trail_distance_atr={row["td"]}, '
                  f'max_hold_days={int(row["mh"])}),')

    # Save results
    out = Path("data/fx_pair_scan_results.csv")
    out.parent.mkdir(exist_ok=True)
    results_df.to_csv(out, index=False)
    print(f"\nResults saved to {out}")


if __name__ == "__main__":
    main()
