#!/usr/bin/env python3
"""
Backtest FX Swing Bot — Carry + Trend strategy over historical data.

Downloads daily bars from yfinance, simulates the carry+trend strategy,
and reports performance metrics.
"""

import math
import sys
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf

# Import strategy constants from the live bot
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from scripts.fx_swing_bot import (
    PAIRS, POLICY_RATES, PairConfig, PairSignal,
    CARRY_WEIGHT, TREND_WEIGHT, ENTRY_THRESHOLD,
    TREND_FAST_MA, TREND_SLOW_MA, TREND_CONFIRM_MA,
    VIX_CARRY_REDUCE, VIX_NO_ENTRY,
    SL_ATR_MULT, TRAIL_ACTIVATE_ATR, TRAIL_DISTANCE_ATR,
    MAX_HOLD_DAYS, ATR_PERIOD,
    compute_carry_signal, compute_atr,
)


IBKR_SWAP_MARKUP = 1.0  # IBKR charges ~1% markup on benchmark rate for swaps


def backtest_pair(pair_cfg: PairConfig, df: pd.DataFrame, vix_df: pd.DataFrame,
                  account: float = 100_000) -> pd.DataFrame:
    """Simulate carry+trend strategy on one pair over historical daily bars."""
    results = []
    position = None  # (direction, entry_price, entry_idx, atr, stop, trail, hw)

    closes = df["Close"]
    sma_fast = closes.rolling(TREND_FAST_MA).mean()
    sma_slow = closes.rolling(TREND_SLOW_MA).mean()
    sma_200 = closes.rolling(TREND_CONFIRM_MA).mean()

    # ATR
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

        # Get VIX for this date
        vix = 20.0
        if date in vix_df.index:
            vix = float(vix_df.loc[date, "Close"])
        else:
            # Find closest prior date
            prior = vix_df.index[vix_df.index <= date]
            if len(prior) > 0:
                vix = float(vix_df.loc[prior[-1], "Close"])

        # Carry weight adjustment (per-pair)
        cw = pair_cfg.get_carry_weight()
        tw = pair_cfg.get_trend_weight()
        if vix > VIX_CARRY_REDUCE:
            cw = cw * 0.5
            tw = 1.0 - cw

        # Trend signal
        cross_pct = (sf - ss) / ss
        trend_raw = math.tanh(cross_pct * 50)
        if (trend_raw > 0 and price < s200) or (trend_raw < 0 and price > s200):
            trend_raw *= 0.5

        combined = carry_sig * cw + trend_raw * tw

        # Check exit if positioned
        if position:
            direction, entry_price, entry_idx, pos_atr, sl, trail, hw = position
            is_long = direction == "LONG"
            pnl_price = (price - entry_price) if is_long else (entry_price - price)
            hold_days = i - entry_idx

            # Update high water
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

            # Trailing stop (per-pair)
            if not exit_reason:
                activate_dist = pair_cfg.get_trail_activate_atr() * pos_atr
                if pnl_price >= activate_dist:
                    if is_long:
                        new_trail = hw - pair_cfg.get_trail_distance_atr() * pos_atr
                        if trail is None or new_trail > trail:
                            trail = new_trail
                        if price <= trail:
                            exit_reason = "TRAIL_STOP"
                    else:
                        new_trail = hw + pair_cfg.get_trail_distance_atr() * pos_atr
                        if trail is None or new_trail < trail:
                            trail = new_trail
                        if price >= trail:
                            exit_reason = "TRAIL_STOP"

            # Time exit (per-pair)
            if not exit_reason and hold_days >= pair_cfg.get_max_hold_days():
                exit_reason = "TIME_EXIT"

            # Signal flip (per-pair threshold)
            if not exit_reason:
                if is_long and combined < -pair_cfg.get_entry_threshold():
                    exit_reason = "SIGNAL_FLIP"
                elif not is_long and combined > pair_cfg.get_entry_threshold():
                    exit_reason = "SIGNAL_FLIP"

            if exit_reason:
                if pair_cfg.quote == "USD":
                    pnl_dollar = pnl_price * 20000  # fixed sizing for backtest
                else:
                    pnl_dollar = (pnl_price * 20000) / price

                # Swap/carry income: IBKR pays benchmark rate minus ~1% markup
                base_rate = POLICY_RATES.get(pair_cfg.base, 0)
                quote_rate = POLICY_RATES.get(pair_cfg.quote, 0)
                # Long earns base, pays quote; short earns quote, pays base
                if is_long:
                    earn_rate = max(0, base_rate - IBKR_SWAP_MARKUP)
                    pay_rate = quote_rate + IBKR_SWAP_MARKUP
                else:
                    earn_rate = max(0, quote_rate - IBKR_SWAP_MARKUP)
                    pay_rate = base_rate + IBKR_SWAP_MARKUP
                net_swap_rate = (earn_rate - pay_rate) / 100  # annualized decimal
                # Notional in USD: 20K for XXX/USD pairs, 20K/price for others
                if pair_cfg.quote == "USD":
                    notional_usd = 20000 * entry_price
                else:
                    notional_usd = 20000  # base is USD for USDJPY, USDMXN
                carry_income = net_swap_rate * notional_usd * hold_days / 365
                pnl_dollar += carry_income

                results.append({
                    "entry_date": df.index[entry_idx],
                    "exit_date": date,
                    "direction": direction,
                    "entry_price": entry_price,
                    "exit_price": price,
                    "pnl_dollar": round(pnl_dollar, 2),
                    "carry_income": round(carry_income, 2),
                    "hold_days": hold_days,
                    "exit_reason": exit_reason,
                    "vix_at_entry": vix,
                })
                position = None
            else:
                position = (direction, entry_price, entry_idx, pos_atr, sl, trail, hw)

        # Check entry if flat (per-pair threshold and stop)
        if position is None and vix <= VIX_NO_ENTRY:
            threshold = pair_cfg.get_entry_threshold()
            if combined > threshold:
                direction = "LONG"
            elif combined < -threshold:
                direction = "SHORT"
            else:
                continue

            sl_mult = pair_cfg.get_sl_atr_mult()
            if direction == "LONG":
                sl = price - sl_mult * atr
            else:
                sl = price + sl_mult * atr
            position = (direction, price, i, atr, sl, None, price)

    # Close any remaining position at last bar
    if position:
        direction, entry_price, entry_idx, pos_atr, sl, trail, hw = position
        price = float(closes.iloc[-1])
        pnl_price = (price - entry_price) if direction == "LONG" else (entry_price - price)
        hold_days = len(df) - 1 - entry_idx
        if pair_cfg.quote == "USD":
            pnl_dollar = pnl_price * 20000
        else:
            pnl_dollar = (pnl_price * 20000) / price
        # Swap income
        is_long = direction == "LONG"
        base_rate = POLICY_RATES.get(pair_cfg.base, 0)
        quote_rate = POLICY_RATES.get(pair_cfg.quote, 0)
        if is_long:
            earn_rate = max(0, base_rate - IBKR_SWAP_MARKUP)
            pay_rate = quote_rate + IBKR_SWAP_MARKUP
        else:
            earn_rate = max(0, quote_rate - IBKR_SWAP_MARKUP)
            pay_rate = base_rate + IBKR_SWAP_MARKUP
        net_swap_rate = (earn_rate - pay_rate) / 100
        if pair_cfg.quote == "USD":
            notional_usd = 20000 * entry_price
        else:
            notional_usd = 20000
        carry_income = net_swap_rate * notional_usd * hold_days / 365
        pnl_dollar += carry_income
        results.append({
            "entry_date": df.index[entry_idx],
            "exit_date": df.index[-1],
            "direction": direction,
            "entry_price": entry_price,
            "exit_price": price,
            "pnl_dollar": round(pnl_dollar, 2),
            "carry_income": round(carry_income, 2),
            "hold_days": hold_days,
            "exit_reason": "OPEN",
            "vix_at_entry": 0,
        })

    return pd.DataFrame(results)


def main():
    print("=" * 70)
    print("FX SWING BOT BACKTEST — Carry + Trend")
    print("=" * 70)

    # Download data
    pairs_to_test = ["EURUSD", "USDJPY", "GBPUSD", "USDMXN"]
    period = "5y"

    print(f"\nDownloading {period} daily data...")
    vix_df = yf.Ticker("^VIX").history(period=period, interval="1d")
    print(f"  VIX: {len(vix_df)} bars")

    all_results = []

    for pair_name in pairs_to_test:
        pair_cfg = PAIRS[pair_name]
        df = yf.Ticker(pair_cfg.yf_symbol).history(period=period, interval="1d")
        print(f"  {pair_name}: {len(df)} bars ({df.index[0].date()} to {df.index[-1].date()})")

        if len(df) < TREND_CONFIRM_MA + 50:
            print(f"    Insufficient data, skipping")
            continue

        results = backtest_pair(pair_cfg, df, vix_df)
        if results.empty:
            print(f"    No trades generated")
            continue

        results["pair"] = pair_name
        all_results.append(results)

        # Per-pair summary
        n = len(results)
        wins = (results["pnl_dollar"] > 0).sum()
        total_pnl = results["pnl_dollar"].sum()
        avg_pnl = results["pnl_dollar"].mean()
        avg_hold = results["hold_days"].mean()
        max_win = results["pnl_dollar"].max()
        max_loss = results["pnl_dollar"].min()

        total_carry = results["carry_income"].sum() if "carry_income" in results.columns else 0
        price_pnl = total_pnl - total_carry

        print(f"\n  {pair_name} Results:")
        print(f"    Trades: {n} | Wins: {wins} ({wins/n*100:.0f}%)")
        print(f"    Total PnL: ${total_pnl:+,.2f} (price: ${price_pnl:+,.2f} + carry: ${total_carry:+,.2f})")
        print(f"    Avg PnL: ${avg_pnl:+,.2f}")
        print(f"    Max win: ${max_win:+,.2f} | Max loss: ${max_loss:+,.2f}")
        print(f"    Avg hold: {avg_hold:.1f} days")

        # By exit reason
        print(f"    Exit reasons:")
        for reason, group in results.groupby("exit_reason"):
            print(f"      {reason}: {len(group)} trades, ${group['pnl_dollar'].sum():+,.2f}")

    if not all_results:
        print("\nNo trades across any pair!")
        return

    # Combined results
    combined = pd.concat(all_results, ignore_index=True)
    combined = combined.sort_values("entry_date").reset_index(drop=True)

    print("\n" + "=" * 70)
    print("COMBINED RESULTS (all pairs)")
    print("=" * 70)

    n = len(combined)
    wins = (combined["pnl_dollar"] > 0).sum()
    total_pnl = combined["pnl_dollar"].sum()
    avg_pnl = combined["pnl_dollar"].mean()

    total_carry = combined["carry_income"].sum() if "carry_income" in combined.columns else 0
    price_pnl = total_pnl - total_carry

    print(f"  Total trades: {n}")
    print(f"  Win rate: {wins}/{n} ({wins/n*100:.0f}%)")
    print(f"  Total PnL: ${total_pnl:+,.2f} (price: ${price_pnl:+,.2f} + carry: ${total_carry:+,.2f})")
    print(f"  Avg PnL per trade: ${avg_pnl:+,.2f}")

    # Equity curve and drawdown
    equity = combined["pnl_dollar"].cumsum()
    running_max = equity.cummax()
    drawdown = equity - running_max
    max_dd = drawdown.min()

    print(f"  Max drawdown: ${max_dd:+,.2f}")
    if max_dd < 0:
        print(f"  Return/MaxDD: {total_pnl / abs(max_dd):.2f}x")

    # Sharpe (annualized, assuming ~50 trades/year)
    if len(combined) > 1:
        trades_per_year = n / 5  # 5 years
        sharpe = (avg_pnl / combined["pnl_dollar"].std()) * np.sqrt(trades_per_year)
        print(f"  Sharpe (approx): {sharpe:.2f}")

    # Monthly breakdown
    combined["month"] = pd.to_datetime(combined["exit_date"]).dt.to_period("M")
    print("\n  Monthly PnL:")
    for month, group in combined.groupby("month"):
        m_pnl = group["pnl_dollar"].sum()
        m_n = len(group)
        print(f"    {month}: ${m_pnl:+,.2f} ({m_n} trades)")

    # Save to CSV
    out = Path("data/fx_swing_backtest.csv")
    out.parent.mkdir(exist_ok=True)
    combined.to_csv(out, index=False)
    print(f"\n  Results saved to {out}")


if __name__ == "__main__":
    main()
