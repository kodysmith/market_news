#!/usr/bin/env python3
"""Analyze SPX price moves over 7-day (and other) periods to find best day to sell premium and best day to close.

Loads SPX daily close (yfinance) and optionally options Parquet for expiration dates.
Computes move stats by expiration, by DTE at entry, and by day-of-week.
Recommends: (1) best day/DTE to sell premium, (2) best close timing (e.g. 3 DTE or Friday).
"""
from __future__ import annotations

import argparse
import sys
from datetime import date, datetime, timedelta
from pathlib import Path

import pandas as pd
import numpy as np

ROOT = Path(__file__).resolve().parent.parent

R = 0.05  # risk-free for BS


def _d1(S: float, K: float, T: float, sigma: float) -> float:
    if T <= 0 or sigma <= 0 or S <= 0 or K <= 0:
        return float("nan")
    return (np.log(S / K) + (R + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))


def _d2(S: float, K: float, T: float, sigma: float) -> float:
    d1_val = _d1(S, K, T, sigma)
    return d1_val - sigma * np.sqrt(T) if not np.isnan(d1_val) else float("nan")


def bs_put_price(S: float, K: float, T: float, sigma: float) -> float:
    """Black-Scholes put price (per share)."""
    if T <= 0 or sigma <= 0:
        return 0.0
    from scipy.stats import norm
    d1_val = _d1(S, K, T, sigma)
    d2_val = _d2(S, K, T, sigma)
    if np.isnan(d1_val) or np.isnan(d2_val):
        return 0.0
    return float(K * np.exp(-R * T) * norm.cdf(-d2_val) - S * norm.cdf(-d1_val))


def load_spx_close(start: date, end: date) -> pd.Series:
    """SPX daily close from yfinance ^GSPC; index = date (normalized)."""
    try:
        import yfinance as yf
    except ImportError:
        raise RuntimeError("yfinance required: pip install yfinance")
    t = yf.Ticker("^GSPC")
    hist = t.history(start=start.isoformat(), end=(end + timedelta(days=10)).isoformat())
    if hist.empty:
        return pd.Series(dtype=float)
    hist.index = pd.to_datetime(hist.index).tz_localize(None).normalize()
    close = hist["Close"]
    close.index = close.index.date
    return close


def get_spot(series: pd.Series, d: date) -> float | None:
    """Value on date d, or next available, or last available before."""
    if d in series.index:
        return float(series.loc[d])
    idx = series.index
    for i in range(len(series)):
        di = idx[i] if isinstance(idx[i], date) else (idx[i].date() if hasattr(idx[i], "date") else pd.Timestamp(idx[i]).date())
        if di >= d:
            return float(series.iloc[i])
    return float(series.iloc[-1]) if len(series) else None


def get_expirations_from_options(data_dir: Path, days_limit: int | None) -> list[date]:
    """Discover expiration dates from SPX options Parquet (daily or combined)."""
    data_dir = Path(data_dir).resolve()
    if not data_dir.exists():
        return []
    # Daily files: each file is one snapshot; we need expirations from the chain
    parquet_error: str | None = None
    daily = list(data_dir.glob("spx_options_[0-9][0-9][0-9][0-9]-[0-9][0-9]-[0-9][0-9].parquet"))
    if daily:
        all_exp = set()
        for p in daily:
            try:
                df = pd.read_parquet(p)
                if "expiration_date" not in df.columns:
                    continue
                exp = pd.to_datetime(df["expiration_date"], errors="coerce").dt.date
                all_exp.update(exp.dropna().unique())
            except ImportError as e:
                parquet_error = "parquet engine missing (pip install pyarrow)?"
                break
            except Exception as e:
                parquet_error = str(e)
                continue
        out = sorted(all_exp)
        if days_limit:
            cutoff = date.today() - timedelta(days=days_limit)
            out = [d for d in out if d >= cutoff]
        if parquet_error and not out:
            sys.stderr.write(f"Note: {parquet_error}\n")
        return out
    combined = list(data_dir.glob("spx_options_expirations_*.parquet"))
    if combined:
        try:
            df = pd.read_parquet(combined[0])
            if "expiration_date" not in df.columns:
                return []
            exp = pd.to_datetime(df["expiration_date"], errors="coerce").dt.date
            out = sorted(exp.dropna().unique())
            if days_limit:
                cutoff = date.today() - timedelta(days=days_limit)
                out = [d for d in out if d >= cutoff]
            return out
        except ImportError as e:
            if not parquet_error:
                sys.stderr.write(f"Note: parquet engine missing (pip install pyarrow)?\n")
        except Exception:
            pass
    return []


def generate_friday_expirations(start: date, end: date) -> list[date]:
    """Generate Fridays in [start, end] as proxy expirations (weekly)."""
    out = []
    d = start
    while d <= end:
        if d.weekday() == 4:
            out.append(d)
        d += timedelta(days=1)
    return out


def add_profit_target_columns(
    df: pd.DataFrame,
    spot: pd.Series,
    iv: float,
    profit_target_pct: float,
) -> pd.DataFrame:
    """Add hit_75pct, close_at_profit_date, days_to_profit. Close when option value <= (1 - profit_target_pct/100) * premium_entry."""
    if profit_target_pct <= 0 or profit_target_pct >= 100:
        df["hit_profit_target"] = False
        df["close_at_profit_date"] = None
        df["days_to_profit"] = np.nan
        df["premium_entry"] = np.nan
        return df
    threshold_pct = 1 - profit_target_pct / 100  # e.g. 75% profit -> close when value <= 0.25 * premium
    hit = []
    close_date = []
    days_to = []
    prem_entry = []
    for _, row in df.iterrows():
        entry_d = row["entry_date"]
        exp_d = row["expiration_date"]
        spot_e = row["spot_entry"]
        strike = row["strike_approx"]
        T_entry = (exp_d - entry_d).days / 365.0
        if T_entry <= 0 or spot_e <= 0 or strike <= 0:
            hit.append(False)
            close_date.append(None)
            days_to.append(np.nan)
            prem_entry.append(np.nan)
            continue
        prem = bs_put_price(spot_e, strike, T_entry, iv)
        prem_entry.append(prem)
        if prem <= 0:
            hit.append(False)
            close_date.append(None)
            days_to.append(np.nan)
            continue
        close_threshold = threshold_pct * prem
        found = False
        close_d = None
        days = None
        for day_offset in range(1, (exp_d - entry_d).days + 1):
            d = entry_d + timedelta(days=day_offset)
            if d > exp_d:
                break
            spot_d = get_spot(spot, d)
            if spot_d is None or spot_d <= 0:
                continue
            T_d = (exp_d - d).days / 365.0
            if T_d <= 0:
                break
            val = bs_put_price(spot_d, strike, T_d, iv)
            if val <= close_threshold:
                found = True
                close_d = d
                days = day_offset
                break
        hit.append(found)
        close_date.append(close_d)
        days_to.append(days if found else np.nan)
    df["premium_entry"] = prem_entry
    df["hit_profit_target"] = hit
    df["close_at_profit_date"] = close_date
    df["days_to_profit"] = days_to
    return df


def build_periods(
    spot: pd.Series,
    expirations: list[date],
    dte_at_entry: int = 7,
) -> pd.DataFrame:
    """Build one row per period: entry_date, expiration_date, spot_entry, spot_expiry, spot_at_3dte (if applicable), moves."""
    rows = []
    for exp in expirations:
        entry = exp - timedelta(days=dte_at_entry)
        spot_e = get_spot(spot, entry)
        spot_exp = get_spot(spot, exp)
        if spot_e is None or spot_exp is None or spot_e <= 0:
            continue
        move_pct = (spot_exp - spot_e) / spot_e
        move_pts = spot_exp - spot_e
        # Close at 3 DTE = hold 4 days (from 7 DTE to 3 DTE)
        close_3dte_date = exp - timedelta(days=3)
        spot_3dte = get_spot(spot, close_3dte_date) if close_3dte_date >= entry else None
        move_to_3dte_pct = (spot_3dte - spot_e) / spot_e if spot_3dte is not None and spot_e else None
        move_to_3dte_pts = (spot_3dte - spot_e) if spot_3dte is not None else None
        # Last 3 days move (from 3 DTE to expiry)
        move_last_3d_pct = (spot_exp - spot_3dte) / spot_3dte if spot_3dte and spot_3dte > 0 else None
        rows.append({
            "entry_date": entry,
            "expiration_date": exp,
            "dte_at_entry": dte_at_entry,
            "spot_entry": spot_e,
            "spot_expiry": spot_exp,
            "spot_at_3dte": spot_3dte,
            "move_pct": move_pct,
            "move_pts": move_pts,
            "move_to_3dte_pct": move_to_3dte_pct,
            "move_to_3dte_pts": move_to_3dte_pts,
            "move_last_3d_pct": move_last_3d_pct,
            "entry_weekday": entry.strftime("%A"),
            "exp_weekday": exp.strftime("%A"),
        })
    return pd.DataFrame(rows)


def main() -> int:
    parser = argparse.ArgumentParser(description="SPX premium timing: best day to sell, best day to close.")
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data" / "spx_options", help="Options Parquet dir (for expirations); if empty use Fridays")
    parser.add_argument("--start", type=str, default=None, help="Start date YYYY-MM-DD (default: 1 year ago)")
    parser.add_argument("--end", type=str, default=None, help="End date YYYY-MM-DD (default: today)")
    parser.add_argument("--dte", type=int, default=7, help="Primary DTE at entry (default 7)")
    parser.add_argument("--dte-range", type=str, default=None, help="Compare multiple DTE: e.g. 5,6,7,8,9")
    parser.add_argument("--strike-otm-pct", type=float, default=2.0, help="Short put strike approx: spot * (1 - this/100); 2 = ~2%% OTM (default)")
    parser.add_argument("--profit-target-pct", type=float, default=75.0, help="Close when option value <= (100 - this)%% of premium (e.g. 75 = 75%% profit to close)")
    parser.add_argument("--iv", type=float, default=0.15, help="Implied vol for BS option value (default 0.15); used for profit-target simulation")
    parser.add_argument("--output-csv", type=Path, default=None, help="Write periods to CSV")
    args = parser.parse_args()
    end = date.today() if not args.end else datetime.strptime(args.end, "%Y-%m-%d").date()
    start = end - timedelta(days=365) if not args.start else datetime.strptime(args.start, "%Y-%m-%d").date()
    # Expirations
    expirations = get_expirations_from_options(args.data_dir, days_limit=None)
    if not expirations:
        expirations = generate_friday_expirations(start, end)
        print("No options data for expirations (install pyarrow to read Parquet, or use --start/--end to match option dates); using weekly Fridays.", file=sys.stderr)
    else:
        in_range = [e for e in expirations if start <= e <= end]
        if not in_range:
            print("Options expirations outside date range; using weekly Fridays.", file=sys.stderr)
            expirations = generate_friday_expirations(start, end)
        else:
            expirations = in_range
    if not expirations:
        print("No expirations in range.", file=sys.stderr)
        return 1
    # SPX close
    spot = load_spx_close(start - timedelta(days=args.dte + 5), end + timedelta(days=5))
    if spot.empty:
        print("Could not load SPX close (yfinance ^GSPC).", file=sys.stderr)
        return 1
    # Build periods (7-day by default)
    df = build_periods(spot, expirations, dte_at_entry=args.dte)
    if df.empty:
        print("No periods with valid spot.", file=sys.stderr)
        return 1
    # Strike approx for "win" (short put: win if spot_expiry >= strike)
    strike_pct = 1 - args.strike_otm_pct / 100
    df["strike_approx"] = df["spot_entry"] * strike_pct
    df["win_hold_to_expiry"] = df["spot_expiry"] >= df["strike_approx"]
    # Close at 3 DTE: "win" if spot at 3 DTE >= strike (we'd close for profit)
    df["win_close_at_3dte"] = np.where(
        df["spot_at_3dte"].notna(),
        df["spot_at_3dte"] >= df["strike_approx"],
        np.nan,
    )
    df["exp_is_friday"] = df["expiration_date"].apply(lambda d: d.weekday() == 4)
    # Profit-target close (e.g. 75% profit = close when option value <= 25% of premium)
    df = add_profit_target_columns(df, spot, args.iv, args.profit_target_pct)
    n = len(df)
    # Optional: compare multiple DTE
    dte_comparison = []
    if args.dte_range:
        dte_list = [int(x.strip()) for x in args.dte_range.split(",") if x.strip()]
        for d in dte_list:
            if d < 1:
                continue
            df_d = build_periods(spot, expirations, dte_at_entry=d)
            if df_d.empty:
                continue
            df_d["strike_approx"] = df_d["spot_entry"] * strike_pct
            df_d["win"] = df_d["spot_expiry"] >= df_d["strike_approx"]
            wr = df_d["win"].mean()
            dte_comparison.append((d, wr, len(df_d)))
    win_exp = df["win_hold_to_expiry"].sum()
    win_3dte = df["win_close_at_3dte"].dropna()
    n_3 = len(win_3dte)
    win_3dte_sum = win_3dte.sum() if n_3 else 0
    hit_profit = df["hit_profit_target"].sum()
    days_to_profit = df["days_to_profit"].dropna()
    avg_days_to_profit = float(days_to_profit.mean()) if len(days_to_profit) else None
    # By entry weekday
    by_dow = df.groupby("entry_weekday").agg(
        count=("entry_date", "count"),
        win_rate_exp=("win_hold_to_expiry", "mean"),
        win_rate_3dte=("win_close_at_3dte", "mean"),
        win_rate_profit_target=("hit_profit_target", "mean"),
        avg_move_pct=("move_pct", "mean"),
    ).round(4)
    # By exp weekday (close on Friday?)
    by_exp_dow = df.groupby("exp_weekday").agg(
        count=("expiration_date", "count"),
        win_rate_exp=("win_hold_to_expiry", "mean"),
        avg_move_pct=("move_pct", "mean"),
    ).round(4)
    # Last 3 days volatility (if we close at 3 DTE we avoid this)
    last3 = df["move_last_3d_pct"].dropna()
    avg_last3 = last3.mean() * 100 if len(last3) else None
    std_last3 = last3.std() * 100 if len(last3) else None
    # ----- Output -----
    print("=" * 60)
    print("SPX PREMIUM TIMING (by expiration)")
    print("=" * 60)
    print(f"Periods: {n}  (entry {args.dte} DTE, strike ~{args.strike_otm_pct}% OTM)")
    print(f"Date range: {df['entry_date'].min()} to {df['expiration_date'].max()}")
    if dte_comparison:
        print()
        print("0) BEST DTE TO SELL (win rate hold to expiry)")
        print("-" * 50)
        for d, wr, cnt in sorted(dte_comparison, key=lambda x: -x[1]):
            print(f"   DTE {d}:  win rate = {wr:.1%}  n={cnt}")
        best_dte = max(dte_comparison, key=lambda x: x[1])
        print(f"   -> Best DTE to sell: {best_dte[0]}  (win rate {best_dte[1]:.1%})")
    print()
    print("1) BEST DAY TO SELL PREMIUM (by entry day of week)")
    print("-" * 50)
    by_dow_sorted = by_dow.sort_values("win_rate_exp", ascending=False)
    for day, row in by_dow_sorted.iterrows():
        pt = f"  hit_{args.profit_target_pct:.0f}pct={row['win_rate_profit_target']:.1%}" if "win_rate_profit_target" in row and args.profit_target_pct > 0 and args.profit_target_pct < 100 else ""
        print(f"   {day:10}  n={int(row['count']):3}  win_rate(hold to exp)={row['win_rate_exp']:.1%}  win_rate(close at 3 DTE)={row['win_rate_3dte']:.1%}{pt}  avg_move={row['avg_move_pct']:.2%}")
    best_entry_dow = by_dow_sorted.index[0] if len(by_dow_sorted) else None
    print(f"   -> Best entry day (hold to expiry): {best_entry_dow}")
    print()
    print("2) BEST DAY TO CLOSE")
    print("-" * 50)
    print(f"   Hold to expiry:  win rate = {win_exp/n:.1%}  ({int(win_exp)}/{n})")
    if n_3:
        print(f"   Close at 3 DTE: win rate = {win_3dte_sum/n_3:.1%}  ({int(win_3dte_sum)}/{n_3})  (common rule for ~50% profit capture)")
    if args.profit_target_pct > 0 and args.profit_target_pct < 100:
        print(f"   Close at {args.profit_target_pct:.0f}% profit: hit rate = {hit_profit/n:.1%}  ({int(hit_profit)}/{n})  (close when option value <= {100-args.profit_target_pct:.0f}% of premium)")
        if avg_days_to_profit is not None:
            print(f"      Avg days to hit {args.profit_target_pct:.0f}% profit: {avg_days_to_profit:.1f}")
    if avg_last3 is not None:
        print(f"   Move in last 3 days to expiry: avg={avg_last3:.2f}%  std={std_last3:.2f}%  (closing at 3 DTE avoids this)")
    friday_exp = df[df["exp_is_friday"]]
    if len(friday_exp) > 0:
        wf = friday_exp["win_hold_to_expiry"].mean()
        print(f"   Expiration on Friday (close Friday before close): win rate = {wf:.1%}  n={len(friday_exp)}")
    print(f"   By expiration weekday:")
    for day, row in by_exp_dow.sort_values("win_rate_exp", ascending=False).iterrows():
        print(f"      {day:10}  n={int(row['count']):3}  win_rate={row['win_rate_exp']:.1%}  avg_move={row['avg_move_pct']:.2%}")
    print()
    print("3) MOVE STATS OVER FULL PERIOD (7 days)")
    print("-" * 50)
    print(f"   Avg move (entry -> expiry): {df['move_pct'].mean()*100:.2f}%  std={df['move_pct'].std()*100:.2f}%")
    if df["move_to_3dte_pct"].notna().any():
        print(f"   Avg move (entry -> 3 DTE):  {df['move_to_3dte_pct'].mean()*100:.2f}%  (if you close at 3 DTE)")
    print()
    if args.output_csv:
        df.to_csv(args.output_csv, index=False)
        print(f"Periods written to {args.output_csv}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
