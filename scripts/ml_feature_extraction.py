#!/usr/bin/env python3
"""Extract ML features from SPX + VIX daily data (2015-2025)."""
from __future__ import annotations

import sys
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.optimize_spx_verticals_historical import load_spx_vix, get_trading_days


def third_friday(year: int, month: int) -> date:
    """Return the third Friday of a given month."""
    d = date(year, month, 1)
    # Find first Friday
    offset = (4 - d.weekday()) % 7
    first_fri = d + timedelta(days=offset)
    return first_fri + timedelta(weeks=2)


def compute_features(spx_df: pd.DataFrame, vix_df: pd.DataFrame) -> pd.DataFrame:
    """Compute all features for each trading day."""
    # Align on common dates
    spx = spx_df[["Open", "High", "Low", "Close"]].copy()
    spx.columns = ["spx_open", "spx_high", "spx_low", "spx_close"]
    vix = vix_df[["Close"]].rename(columns={"Close": "vix_close"})
    df = spx.join(vix, how="inner").sort_index()
    df = df.dropna(subset=["spx_close", "vix_close"])

    c = df["spx_close"]

    # --- Price features ---
    for n in [1, 3, 5, 10, 20]:
        df[f"ret_{n}d"] = c.pct_change(n)

    for w in [20, 50, 200]:
        sma = c.rolling(w).mean()
        df[f"dist_sma{w}_pct"] = (c - sma) / sma * 100
        df[f"sma{w}"] = sma

    # Intraday range proxy
    df["range_pct"] = (df["spx_high"] - df["spx_low"]) / c * 100
    # Gap %
    df["gap_pct"] = (df["spx_open"] - c.shift(1)) / c.shift(1) * 100

    # --- Volatility features ---
    df["vix"] = df["vix_close"]
    df["vix_zscore_252"] = (df["vix"] - df["vix"].rolling(252).mean()) / df["vix"].rolling(252).std()
    df["vix_chg_1d"] = df["vix"].diff(1)
    df["vix_chg_5d"] = df["vix"].diff(5)

    log_ret = np.log(c / c.shift(1))
    for w in [5, 10, 20]:
        df[f"rv_{w}d"] = log_ret.rolling(w).std() * np.sqrt(252) * 100

    df["rv_iv_ratio"] = df["rv_20d"] / df["vix"].replace(0, np.nan)
    df["iv_rank_252"] = df["vix"].rolling(252).apply(
        lambda x: pd.Series(x).rank(pct=True).iloc[-1], raw=False
    )

    # --- Regime features ---
    df["above_sma50"] = (c > df["sma50"]).astype(int)
    df["above_sma200"] = (c > df["sma200"]).astype(int)
    df["vix_zone"] = pd.cut(
        df["vix"], bins=[-np.inf, 15, 24, 35, np.inf], labels=[0, 1, 2, 3]
    ).astype(float)
    df["momentum"] = (df["sma20"] - df["sma50"]) / df["sma50"] * 100

    # Consecutive up/down days (clamped ±5)
    daily_dir = np.sign(c.diff())
    consec = pd.Series(0.0, index=df.index)
    prev = 0.0
    for i in range(len(daily_dir)):
        d = daily_dir.iloc[i]
        if d == 0:
            prev = 0.0
        elif np.sign(prev) == d:
            prev += d
        else:
            prev = d
        consec.iloc[i] = max(-5, min(5, prev))
    df["consec_days"] = consec

    # --- Calendar features ---
    dow = df.index.dayofweek
    df["dow_sin"] = np.sin(2 * np.pi * dow / 5)
    df["dow_cos"] = np.cos(2 * np.pi * dow / 5)
    df["is_friday"] = (dow == 4).astype(int)
    df["is_monday"] = (dow == 0).astype(int)

    # Month of year (cyclical) — seasonality matters (Jan effect, Sep weakness, Dec rally)
    month = df.index.month
    df["month_sin"] = np.sin(2 * np.pi * month / 12)
    df["month_cos"] = np.cos(2 * np.pi * month / 12)

    # Week of month (1-5, normalized)
    df["week_of_month"] = (df.index.day - 1) // 7 / 4.0  # 0.0 to 1.0

    # Quarter (cyclical)
    quarter = (month - 1) // 3
    df["quarter_sin"] = np.sin(2 * np.pi * quarter / 4)
    df["quarter_cos"] = np.cos(2 * np.pi * quarter / 4)

    # Year progress (0.0 = Jan 1, 1.0 = Dec 31)
    day_of_year = df.index.dayofyear
    df["year_progress"] = day_of_year / 365.0

    # Days to monthly opex (third Friday)
    opex_dates = []
    for yr in range(df.index.min().year, df.index.max().year + 2):
        for mo in range(1, 13):
            try:
                opex_dates.append(third_friday(yr, mo))
            except Exception:
                pass
    opex_dates = sorted(opex_dates)
    opex_arr = np.array(opex_dates)

    days_to_opex = []
    is_opex_week = []
    for ts in df.index:
        d = ts.date() if hasattr(ts, "date") else ts
        future = opex_arr[opex_arr >= d]
        if len(future) > 0:
            delta = (future[0] - d).days
            days_to_opex.append(delta)
            is_opex_week.append(1 if delta <= 5 else 0)
        else:
            days_to_opex.append(0)
            is_opex_week.append(0)
    df["days_to_opex"] = days_to_opex
    df["is_opex_week"] = is_opex_week
    df["is_opex_day"] = [1 if dto == 0 else 0 for dto in days_to_opex]

    # --- Microstructure / volume-proxy features ---
    # True range (captures volatility expansion better than range_pct)
    prev_close = c.shift(1)
    true_range = pd.concat([
        df["spx_high"] - df["spx_low"],
        (df["spx_high"] - prev_close).abs(),
        (df["spx_low"] - prev_close).abs(),
    ], axis=1).max(axis=1)
    df["atr_5"] = true_range.rolling(5).mean() / c * 100
    df["atr_10"] = true_range.rolling(10).mean() / c * 100
    df["atr_ratio"] = df["atr_5"] / df["atr_10"].replace(0, np.nan)

    # Bollinger band position (where is price relative to 2σ bands?)
    bb_sma = c.rolling(20).mean()
    bb_std = c.rolling(20).std()
    df["bb_position"] = (c - bb_sma) / (2 * bb_std).replace(0, np.nan)  # -1 to +1 normally

    # RSI-like momentum (14-day)
    delta_price = c.diff()
    gain = delta_price.clip(lower=0).rolling(14).mean()
    loss = (-delta_price.clip(upper=0)).rolling(14).mean()
    rs = gain / loss.replace(0, np.nan)
    df["rsi_14"] = 100 - (100 / (1 + rs))
    df["rsi_14_norm"] = (df["rsi_14"] - 50) / 50  # normalize to -1 to +1

    # VIX term structure proxy: VIX 5d change / VIX 20d change (acceleration)
    df["vix_acceleration"] = df["vix_chg_5d"] / df["vix_chg_1d"].rolling(5).sum().replace(0, np.nan)

    # Put/call-like proxy: skew from high-low bias
    df["hl_skew"] = (df["spx_high"] - c) / (c - df["spx_low"]).replace(0, np.nan) - 1

    # Volume of movement: |return| * range (captures conviction)
    df["move_conviction"] = df["ret_1d"].abs() * df["range_pct"]

    # Distance from recent high/low (drawdown proxy)
    df["dist_from_20d_high"] = (c - c.rolling(20).max()) / c.rolling(20).max() * 100
    df["dist_from_20d_low"] = (c - c.rolling(20).min()) / c.rolling(20).min() * 100

    # --- VWAP proxy (typical price as daily VWAP estimate) ---
    # True VWAP needs intraday volume; daily typical price is a reasonable proxy
    typical_price = (df["spx_high"] + df["spx_low"] + df["spx_close"]) / 3
    df["vwap_proxy"] = typical_price.rolling(5).mean()  # 5-day rolling typical price
    df["dist_from_vwap"] = (c - df["vwap_proxy"]) / df["vwap_proxy"] * 100

    # --- GEX/Options structure proxies (computed from vol & price behavior) ---
    # We don't have historical GEX/wall data, but we can proxy the EFFECTS:

    # Pin proxy: distance to nearest $25 round strike (where GEX pinning occurs)
    nearest_25 = (c / 25).round() * 25
    df["dist_to_pin_25"] = (c - nearest_25) / c * 100
    df["dist_to_pin_25_abs"] = df["dist_to_pin_25"].abs()

    # Pin proxy: distance to nearest $50 round strike
    nearest_50 = (c / 50).round() * 50
    df["dist_to_pin_50"] = (c - nearest_50) / c * 100

    # Gamma proxy: when realized vol is LOW relative to implied (VIX),
    # dealers are likely long gamma → market pins. When RV > IV, short gamma → trends.
    df["gamma_regime"] = np.where(
        df["rv_iv_ratio"] < 0.8, 1,   # long gamma (pinning)
        np.where(df["rv_iv_ratio"] > 1.2, -1, 0)  # short gamma (trending)
    ).astype(float)

    # Flip line proxy: SMA20 often acts as the gamma flip zone
    # When price is near SMA20, dealer hedging creates mean-reversion
    df["dist_to_flip_proxy"] = df["dist_sma20_pct"]
    df["near_flip_zone"] = (df["dist_sma20_pct"].abs() < 0.5).astype(float)

    # Put wall proxy: recent 20-day low acts as support (where put OI accumulates)
    df["dist_to_put_wall_proxy"] = df["dist_from_20d_low"]

    # Call wall proxy: recent 20-day high acts as resistance (where call OI accumulates)
    df["dist_to_call_wall_proxy"] = df["dist_from_20d_high"]

    # Skew proxy: VIX vs 10d RV spread indicates put skew demand
    df["skew_proxy"] = (df["vix"] - df["rv_10d"]) / df["vix"].replace(0, np.nan)

    # --- Normalize ---
    # z-score (252d rolling) for continuous features
    zscore_cols = [
        "ret_1d", "ret_3d", "ret_5d", "ret_10d", "ret_20d",
        "dist_sma20_pct", "dist_sma50_pct", "dist_sma200_pct",
        "range_pct", "gap_pct",
        "vix", "vix_chg_1d", "vix_chg_5d",
        "rv_5d", "rv_10d", "rv_20d", "rv_iv_ratio",
        "momentum",
        "atr_5", "atr_10", "atr_ratio",
        "bb_position", "rsi_14_norm",
        "vix_acceleration", "hl_skew", "move_conviction",
        "dist_from_20d_high", "dist_from_20d_low",
        "dist_from_vwap", "dist_to_pin_25", "dist_to_pin_50",
        "dist_to_flip_proxy", "dist_to_put_wall_proxy", "dist_to_call_wall_proxy",
        "skew_proxy",
    ]
    for col in zscore_cols:
        if col in df.columns:
            roll_mean = df[col].rolling(252, min_periods=60).mean()
            roll_std = df[col].rolling(252, min_periods=60).std().replace(0, np.nan)
            df[f"{col}_z"] = (df[col] - roll_mean) / roll_std

    # Select final feature columns
    feature_cols = (
        [f"{c}_z" for c in zscore_cols if f"{c}_z" in df.columns]
        + ["vix_zscore_252", "iv_rank_252"]
        + ["above_sma50", "above_sma200", "vix_zone", "consec_days"]
        + ["dow_sin", "dow_cos", "is_friday", "is_monday",
           "month_sin", "month_cos", "week_of_month",
           "quarter_sin", "quarter_cos", "year_progress",
           "days_to_opex", "is_opex_week", "is_opex_day"]
        + ["gamma_regime", "near_flip_zone", "dist_to_pin_25_abs"]
    )

    # Also keep raw values needed for trade simulation
    raw_cols = ["spx_close", "spx_open", "spx_high", "spx_low", "vix_close"]

    keep = [c for c in feature_cols + raw_cols if c in df.columns]
    out = df[keep].copy()

    # Handle NaN/inf
    out = out.replace([np.inf, -np.inf], np.nan)
    out = out.ffill().fillna(0)

    return out


def main():
    print("Loading SPX + VIX data 2014-2025 (extra year for rolling warmup)...")
    spx_df, vix_df = load_spx_vix(date(2014, 1, 1), date(2025, 12, 31))
    print(f"  SPX rows: {len(spx_df)}, VIX rows: {len(vix_df)}")

    print("Computing features...")
    features = compute_features(spx_df, vix_df)

    # Trim to 2015+ (after rolling warmup)
    features = features[features.index >= "2015-01-01"]
    print(f"  Feature matrix: {features.shape[0]} days x {features.shape[1]} columns")
    print(f"  Date range: {features.index.min().date()} to {features.index.max().date()}")
    print(f"  NaN count: {features.isna().sum().sum()}")

    out_path = ROOT / "data" / "ml_features.parquet"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    features.to_parquet(out_path)
    print(f"  Saved to {out_path}")

    # Summary stats
    print("\nFeature summary:")
    print(features.describe().T[["mean", "std", "min", "max"]].to_string())


if __name__ == "__main__":
    main()
