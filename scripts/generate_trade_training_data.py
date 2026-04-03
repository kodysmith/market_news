#!/usr/bin/env python3
"""Generate balanced training data from backtest: top winners vs top losers.

Runs the regime-aware daily cashflow backtest, enriches each trade with market
context, then selects equal numbers of the biggest winners and biggest losers,
preferring pairs that occur near the same time period for apples-to-apples
comparison.

Output: CSV with full trade details + market context for ML training.
"""
from __future__ import annotations

import math
import sys
from datetime import date, datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.optimize_spx_verticals_historical import (
    load_spx_vix, get_trading_days, friday_expirations,
    bs_put_price, bs_call_price, strike_for_delta, MULT,
)
from bot.regime_engine import compute_regime_score, update_regime, RegimeState

SLIPPAGE_PER_LEG = 0.10
COMMISSION_PER_LEG = 0.65

OUT_DIR = ROOT / "data" / "training"


def simulate_spread_with_context(
    config_id: str,
    strategy: str,
    dte: int,
    short_delta: float,
    width: float,
    profit_target_pct: float,
    stop_mult: float,
    time_stop_dte: int | None,
    spx_close: pd.Series,
    vix_close: pd.Series,
    trading_days: list,
    trading_idx: dict,
    expirations: list,
    start: date,
    end: date,
    sma50: pd.Series,
    sma200: pd.Series,
    max_hold_days: int | None = None,
) -> list[dict]:
    """Simulate a spread config and attach rich market context to each trade."""
    trades = []

    for exp_date in expirations:
        if exp_date < start or exp_date > end:
            continue
        exp_i = trading_idx.get(exp_date)
        if exp_i is None:
            continue

        # Find entry date
        entry_date = None
        for check_i in range(max(0, exp_i - dte - 5), exp_i):
            d = trading_days[check_i]
            if d < start:
                continue
            cal_dte = (exp_date - d).days
            if abs(cal_dte - dte) <= 2:
                entry_date = d
                break

        if entry_date is None or entry_date not in spx_close.index:
            continue

        entry_idx = trading_idx[entry_date]
        dte_trading = exp_i - entry_idx
        if dte_trading <= 0:
            continue

        S_entry = float(spx_close[entry_date])
        vix_val = float(vix_close.get(entry_date, 20.0))
        sigma = vix_val / 100.0 if vix_val > 3 else vix_val
        T_entry = dte_trading / 252.0

        # Compute strikes and credit
        credit = 0.0
        K_put_short = K_put_long = K_call_short = K_call_long = None
        n_legs = 4

        if strategy in ("iron_condor", "put_vertical"):
            K_put_short = strike_for_delta(S_entry, T_entry, short_delta, sigma, is_put=True)
            if K_put_short is None:
                continue
            K_put_short = round(K_put_short / 5) * 5
            K_put_long = K_put_short - width
            if K_put_long <= 0:
                continue
            credit += bs_put_price(S_entry, K_put_short, T_entry, sigma) - \
                      bs_put_price(S_entry, K_put_long, T_entry, sigma)

        if strategy in ("iron_condor", "call_vertical"):
            K_call_short = strike_for_delta(S_entry, T_entry, short_delta, sigma, is_put=False)
            if K_call_short is None:
                continue
            K_call_short = round(K_call_short / 5) * 5
            K_call_long = K_call_short + width
            credit += bs_call_price(S_entry, K_call_short, T_entry, sigma) - \
                      bs_call_price(S_entry, K_call_long, T_entry, sigma)

        if strategy == "iron_condor":
            n_legs = 8

        entry_slippage = n_legs * SLIPPAGE_PER_LEG / 2.0
        credit_adj = credit - entry_slippage
        if credit_adj <= 0:
            continue

        max_loss = (width - credit_adj) * MULT

        # Simulate daily
        close_threshold = (1.0 - profit_target_pct) * credit_adj
        stop_threshold = stop_mult * credit_adj

        exit_date = exp_date
        exit_reason = "expiry"
        exit_debit = None
        max_adverse = 0.0  # track worst intra-trade drawdown
        max_favorable = 0.0  # track best intra-trade move

        for i in range(entry_idx + 1, exp_i):
            if i >= len(trading_days):
                break
            d = trading_days[i]
            if d not in spx_close.index:
                continue

            S_d = float(spx_close[d])
            dte_left = exp_i - i
            T_d = dte_left / 252.0

            sigma_d = float(vix_close.get(d, vix_val))
            sigma_d = sigma_d / 100.0 if sigma_d > 3 else sigma_d

            if max_hold_days is not None and (i - entry_idx) >= max_hold_days:
                spread_val = 0.0
                if K_put_short is not None and T_d > 0:
                    spread_val += bs_put_price(S_d, K_put_short, T_d, sigma_d) - \
                                  bs_put_price(S_d, K_put_long, T_d, sigma_d)
                if K_call_short is not None and T_d > 0:
                    spread_val += bs_call_price(S_d, K_call_short, T_d, sigma_d) - \
                                  bs_call_price(S_d, K_call_long, T_d, sigma_d)
                exit_date = d
                exit_reason = "max_hold"
                exit_debit = spread_val
                break

            if T_d <= 0:
                break

            spread_val = 0.0
            if K_put_short is not None:
                spread_val += bs_put_price(S_d, K_put_short, T_d, sigma_d) - \
                              bs_put_price(S_d, K_put_long, T_d, sigma_d)
            if K_call_short is not None:
                spread_val += bs_call_price(S_d, K_call_short, T_d, sigma_d) - \
                              bs_call_price(S_d, K_call_long, T_d, sigma_d)

            # Track intra-trade extremes (unrealized P&L)
            unrealized = (credit_adj - spread_val) * MULT
            max_favorable = max(max_favorable, unrealized)
            max_adverse = min(max_adverse, unrealized)

            if spread_val <= close_threshold:
                exit_date = d
                exit_reason = "profit_target"
                exit_debit = spread_val
                break
            if spread_val >= stop_threshold:
                exit_date = d
                exit_reason = "stop_loss"
                exit_debit = spread_val
                break
            if time_stop_dte is not None and dte_left <= time_stop_dte:
                exit_date = d
                exit_reason = "time_stop"
                exit_debit = spread_val
                break

        # P&L
        if exit_reason == "expiry":
            S_exp = float(spx_close.get(exp_date, S_entry))
            settlement = 0.0
            if K_put_short is not None:
                settlement += max(0, K_put_short - S_exp) - max(0, K_put_long - S_exp)
            if K_call_short is not None:
                settlement += max(0, S_exp - K_call_short) - max(0, S_exp - K_call_long)
            commission = (n_legs / 2) * COMMISSION_PER_LEG
            pnl = (credit_adj - settlement) * MULT - commission
        else:
            exit_slippage = n_legs * SLIPPAGE_PER_LEG / 2.0
            debit_adj = (exit_debit or 0) + exit_slippage
            commission = n_legs * COMMISSION_PER_LEG
            pnl = (credit_adj - debit_adj) * MULT - commission

        hold_days = (exit_date - entry_date).days

        # --- Market context at entry ---
        sma50_val = float(sma50.get(entry_date, np.nan))
        sma200_val = float(sma200.get(entry_date, np.nan))

        # SPX move during trade
        S_exit = float(spx_close.get(exit_date, S_entry))
        spx_move_pct = (S_exit - S_entry) / S_entry * 100

        # Distance from strikes to SPX at entry
        put_distance_pct = ((S_entry - K_put_short) / S_entry * 100) if K_put_short else None
        call_distance_pct = ((K_call_short - S_entry) / S_entry * 100) if K_call_short else None

        # VIX at exit
        vix_exit = float(vix_close.get(exit_date, vix_val))
        vix_change = vix_exit - vix_val

        # Day of week at entry
        dow = entry_date.weekday()

        trades.append({
            # Trade identifiers
            "config": config_id,
            "strategy": strategy,
            "dte": dte,
            "short_delta": short_delta,
            "width": width,
            "entry_date": entry_date,
            "exit_date": exit_date,
            "expiration": exp_date,
            "exit_reason": exit_reason,
            "hold_days": hold_days,

            # Strikes
            "spx_at_entry": S_entry,
            "put_short_strike": K_put_short,
            "put_long_strike": K_put_long,
            "call_short_strike": K_call_short,
            "call_long_strike": K_call_long,

            # P&L
            "credit": round(credit_adj * MULT, 2),
            "max_loss": round(max_loss, 2),
            "pnl": round(pnl, 2),
            "pnl_pct_of_credit": round(pnl / (credit_adj * MULT) * 100, 2) if credit_adj > 0 else 0,
            "pnl_pct_of_max_loss": round(pnl / max_loss * 100, 2) if max_loss > 0 else 0,
            "is_win": int(pnl > 0),

            # Intra-trade dynamics
            "max_favorable_excursion": round(max_favorable, 2),
            "max_adverse_excursion": round(max_adverse, 2),

            # Market context at entry
            "vix_at_entry": round(vix_val, 2),
            "vix_at_exit": round(vix_exit, 2),
            "vix_change": round(vix_change, 2),
            "sma50": round(sma50_val, 2) if not np.isnan(sma50_val) else None,
            "sma200": round(sma200_val, 2) if not np.isnan(sma200_val) else None,
            "spx_above_sma50": int(S_entry > sma50_val) if not np.isnan(sma50_val) else None,
            "spx_above_sma200": int(S_entry > sma200_val) if not np.isnan(sma200_val) else None,

            # SPX dynamics
            "spx_at_exit": round(S_exit, 2),
            "spx_move_pct": round(spx_move_pct, 4),
            "put_distance_pct": round(put_distance_pct, 4) if put_distance_pct else None,
            "call_distance_pct": round(call_distance_pct, 4) if call_distance_pct else None,

            # Calendar
            "entry_dow": dow,
            "entry_month": entry_date.month,
            "entry_year": entry_date.year,
        })

    return trades


def compute_regime_for_trades(trades_df: pd.DataFrame, spx_close: pd.Series,
                               vix_close: pd.Series, sma50: pd.Series,
                               sma200: pd.Series) -> pd.DataFrame:
    """Add regime score and tier to each trade based on entry date."""
    # Pre-compute regime for all trading days
    dates = sorted(spx_close.index)
    regime_state = RegimeState()
    regime_by_date = {}

    for d in dates:
        spx_price = float(spx_close[d])
        vix_val = float(vix_close.get(d, 20.0))
        s50 = float(sma50.get(d, spx_price))
        s200 = float(sma200.get(d, spx_price))

        # SMA200 from 20 trading days ago
        d_idx = dates.index(d)
        s200_prev = float(sma200.get(dates[max(0, d_idx - 20)], s200))

        # VIX rolling highs
        vix_dates = [dd for dd in dates if dd <= d]
        vix_20d = [float(vix_close.get(dd, 20.0)) for dd in vix_dates[-20:]]
        vix_40d = [float(vix_close.get(dd, 20.0)) for dd in vix_dates[-40:]]
        vix_20d_high = max(vix_20d) if vix_20d else None
        vix_40d_high = max(vix_40d) if vix_40d else None

        score = compute_regime_score(
            spx_price=spx_price,
            sma50=s50,
            sma200=s200,
            sma200_prev=s200_prev,
            vix=vix_val,
            vix_20d_high=vix_20d_high,
            vix_40d_high=vix_40d_high,
        )
        regime_state = update_regime(regime_state, score)
        regime_by_date[d] = {
            "regime_score": score,
            "regime_tier": regime_state.tier,
            "regime_position_scale": regime_state.position_scale,
            "regime_days_in_tier": regime_state.days_in_tier,
        }

    # Map regime to trades
    for col in ["regime_score", "regime_tier", "regime_position_scale", "regime_days_in_tier"]:
        trades_df[col] = trades_df["entry_date"].map(
            lambda d, c=col: regime_by_date.get(d, {}).get(c)
        )

    return trades_df


def select_balanced_pairs(trades_df: pd.DataFrame, n_samples: int = 100) -> pd.DataFrame:
    """Select top N winners and top N losers, preferring time-matched pairs.

    Strategy: sort by P&L, take top N winners and bottom N losers,
    then for each loser find the nearest-in-time winner (and vice versa)
    to ensure temporal coverage overlap.
    """
    winners = trades_df[trades_df["pnl"] > 0].sort_values("pnl", ascending=False)
    losers = trades_df[trades_df["pnl"] < 0].sort_values("pnl", ascending=True)  # worst first

    if len(winners) < n_samples or len(losers) < n_samples:
        n_samples = min(len(winners), len(losers))
        print(f"  Adjusted to {n_samples} samples per side (limited by available trades)")

    # Take worst losers
    worst_losers = losers.head(n_samples).copy()

    # For each loser, find closest-in-time winner from the top winners pool
    # Use a larger pool of winners to allow time matching
    winner_pool = winners.head(n_samples * 3).copy()  # 3x pool for matching

    matched_winner_idxs = set()
    for _, loser in worst_losers.iterrows():
        loser_date = loser["entry_date"]
        # Find closest winner by date that hasn't been matched yet
        available = winner_pool[~winner_pool.index.isin(matched_winner_idxs)]
        if available.empty:
            break
        date_diffs = available["entry_date"].apply(lambda d: abs((d - loser_date).days))
        best_match_idx = date_diffs.idxmin()
        matched_winner_idxs.add(best_match_idx)

    # If we didn't get enough matches, fill with remaining top winners
    if len(matched_winner_idxs) < n_samples:
        remaining = winner_pool[~winner_pool.index.isin(matched_winner_idxs)]
        needed = n_samples - len(matched_winner_idxs)
        for idx in remaining.head(needed).index:
            matched_winner_idxs.add(idx)

    best_winners = winners.loc[list(matched_winner_idxs)[:n_samples]].copy()

    # Add label column
    best_winners["label"] = "WIN"
    worst_losers["label"] = "LOSS"

    balanced = pd.concat([best_winners, worst_losers], ignore_index=True)
    balanced = balanced.sort_values("entry_date").reset_index(drop=True)

    return balanced


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Generate balanced win/loss training data")
    parser.add_argument("--start", default="2015-01-01")
    parser.add_argument("--end", default="2025-12-31")
    parser.add_argument("--samples", type=int, default=150,
                        help="Number of samples per side (winners/losers)")
    parser.add_argument("--config", default=None,
                        help="Only use specific config (e.g. IC3_1DTE_D10_TP50)")
    args = parser.parse_args()

    start = datetime.strptime(args.start, "%Y-%m-%d").date()
    end = datetime.strptime(args.end, "%Y-%m-%d").date()

    print("Loading SPX + VIX data...", file=sys.stderr)
    spx_df, vix_df = load_spx_vix(start, end)
    if spx_df.empty:
        print("ERROR: No SPX data loaded", file=sys.stderr)
        return

    spx_close = spx_df["Close"].copy()
    spx_close.index = spx_close.index.date
    vix_close = vix_df["Close"].copy()
    if not vix_close.empty:
        vix_close.index = vix_close.index.date

    # Compute SMAs
    print("Computing SMAs...", file=sys.stderr)
    spx_series = spx_close.sort_index()
    sma50 = spx_series.rolling(50, min_periods=50).mean()
    sma200 = spx_series.rolling(200, min_periods=200).mean()

    trading_days = get_trading_days(start, end)
    trading_idx = {d: i for i, d in enumerate(trading_days)}
    fri_exps = friday_expirations(start, end)
    daily_exps = [d for d in trading_days if d >= start and d <= end]

    # Active strategy configs (regime strategy: IC3 + IC6 + butterflies)
    configs = [
        # (id, strategy, dte, delta, width, tp%, stop_mult, time_stop, exps, max_hold)
        ("IC3_1DTE_D10_TP50", "iron_condor", 1, 0.10, 30, 0.50, 2.0, None, daily_exps, None),
        ("IC6_3DTE_D12_TP40", "iron_condor", 3, 0.12, 30, 0.40, 2.0, None, daily_exps, None),
        ("IC9_7DTE_D12_TP30", "iron_condor", 7, 0.12, 50, 0.30, 2.0, 1, fri_exps, None),
    ]

    if args.config:
        configs = [c for c in configs if c[0] == args.config]
        if not configs:
            print(f"ERROR: Config '{args.config}' not found", file=sys.stderr)
            return

    print(f"Running backtest ({len(configs)} configs, {start} to {end})...", file=sys.stderr)
    all_trades = []
    for cfg in configs:
        cid, strat, dte, delta, w, tp, sl, ts, exps, mh = cfg
        result = simulate_spread_with_context(
            cid, strat, dte, delta, w, tp, sl, ts,
            spx_close, vix_close, trading_days, trading_idx,
            exps, start, end, sma50, sma200,
            max_hold_days=mh,
        )
        all_trades.extend(result)
        wins = sum(1 for t in result if t["pnl"] > 0)
        losses = sum(1 for t in result if t["pnl"] < 0)
        print(f"  {cid}: {len(result)} trades ({wins}W / {losses}L)", file=sys.stderr)

    trades_df = pd.DataFrame(all_trades)
    if trades_df.empty:
        print("ERROR: No trades generated", file=sys.stderr)
        return

    # Add regime context
    print("Computing regime scores...", file=sys.stderr)
    trades_df = compute_regime_for_trades(trades_df, spx_close, vix_close, sma50, sma200)

    # Summary stats
    total_trades = len(trades_df)
    total_wins = (trades_df["pnl"] > 0).sum()
    total_losses = (trades_df["pnl"] < 0).sum()
    print(f"\nTotal trades: {total_trades} ({total_wins}W / {total_losses}L)", file=sys.stderr)
    print(f"Total P&L: ${trades_df['pnl'].sum():,.0f}", file=sys.stderr)
    print(f"Avg win: ${trades_df.loc[trades_df['pnl'] > 0, 'pnl'].mean():,.0f}", file=sys.stderr)
    print(f"Avg loss: ${trades_df.loc[trades_df['pnl'] < 0, 'pnl'].mean():,.0f}", file=sys.stderr)

    # Save full trade log
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    full_path = OUT_DIR / "all_trades_with_context.csv"
    trades_df.to_csv(full_path, index=False)
    print(f"\nFull trade log: {full_path} ({len(trades_df)} trades)", file=sys.stderr)

    # Select balanced training data
    print(f"\nSelecting {args.samples} balanced pairs...", file=sys.stderr)
    balanced = select_balanced_pairs(trades_df, n_samples=args.samples)

    balanced_path = OUT_DIR / "balanced_winners_losers.csv"
    balanced.to_csv(balanced_path, index=False)

    # Print summary
    wins = balanced[balanced["label"] == "WIN"]
    losses = balanced[balanced["label"] == "LOSS"]
    print(f"\n{'='*80}", file=sys.stderr)
    print(f"BALANCED TRAINING DATA SUMMARY", file=sys.stderr)
    print(f"{'='*80}", file=sys.stderr)
    print(f"  Winners: {len(wins)} trades", file=sys.stderr)
    print(f"    Avg P&L: ${wins['pnl'].mean():,.0f}", file=sys.stderr)
    print(f"    Avg P&L range: ${wins['pnl'].min():,.0f} to ${wins['pnl'].max():,.0f}", file=sys.stderr)
    print(f"    Date range: {wins['entry_date'].min()} to {wins['entry_date'].max()}", file=sys.stderr)
    print(f"  Losers: {len(losses)} trades", file=sys.stderr)
    print(f"    Avg P&L: ${losses['pnl'].mean():,.0f}", file=sys.stderr)
    print(f"    Avg P&L range: ${losses['pnl'].min():,.0f} to ${losses['pnl'].max():,.0f}", file=sys.stderr)
    print(f"    Date range: {losses['entry_date'].min()} to {losses['entry_date'].max()}", file=sys.stderr)
    print(f"\n  Saved to: {balanced_path}", file=sys.stderr)

    # Time overlap check
    win_dates = set(pd.to_datetime(wins["entry_date"]).dt.to_period("M"))
    loss_dates = set(pd.to_datetime(losses["entry_date"]).dt.to_period("M"))
    overlap = win_dates & loss_dates
    print(f"  Month overlap: {len(overlap)}/{max(len(win_dates), len(loss_dates))} "
          f"({len(overlap)/max(len(win_dates), len(loss_dates))*100:.0f}%)", file=sys.stderr)

    # Top 10 worst losers
    print(f"\n{'='*80}", file=sys.stderr)
    print(f"TOP 10 WORST LOSERS", file=sys.stderr)
    print(f"{'='*80}", file=sys.stderr)
    worst = losses.nsmallest(10, "pnl")
    for _, t in worst.iterrows():
        print(f"  {t['entry_date']} → {t['exit_date']}  {t['config']:<25} "
              f"P&L=${t['pnl']:>8,.0f}  VIX={t['vix_at_entry']:>5.1f}  "
              f"SPX move={t['spx_move_pct']:>+.2f}%  Regime={t['regime_tier']}  "
              f"Exit={t['exit_reason']}", file=sys.stderr)

    # Top 10 best winners
    print(f"\n{'='*80}", file=sys.stderr)
    print(f"TOP 10 BEST WINNERS", file=sys.stderr)
    print(f"{'='*80}", file=sys.stderr)
    best = wins.nlargest(10, "pnl")
    for _, t in best.iterrows():
        print(f"  {t['entry_date']} → {t['exit_date']}  {t['config']:<25} "
              f"P&L=${t['pnl']:>8,.0f}  VIX={t['vix_at_entry']:>5.1f}  "
              f"SPX move={t['spx_move_pct']:>+.2f}%  Regime={t['regime_tier']}  "
              f"Exit={t['exit_reason']}", file=sys.stderr)

    # Regime breakdown in training data
    print(f"\n{'='*80}", file=sys.stderr)
    print(f"REGIME BREAKDOWN IN TRAINING DATA", file=sys.stderr)
    print(f"{'='*80}", file=sys.stderr)
    for tier in ["BEAR", "CAUTION", "NEUTRAL", "BULL"]:
        tier_data = balanced[balanced["regime_tier"] == tier]
        if len(tier_data) == 0:
            continue
        tier_wins = (tier_data["label"] == "WIN").sum()
        tier_losses = (tier_data["label"] == "LOSS").sum()
        print(f"  {tier:<10} {len(tier_data):>4} trades  "
              f"({tier_wins}W / {tier_losses}L)  "
              f"Avg P&L=${tier_data['pnl'].mean():>+8,.0f}", file=sys.stderr)

    print(f"\n{'='*80}", file=sys.stderr)


if __name__ == "__main__":
    main()
