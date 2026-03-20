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


def compute_features(spx_df: pd.DataFrame, vix_df: pd.DataFrame,
                     extra_dfs: dict | None = None) -> pd.DataFrame:
    """Compute all features for each trading day.

    CRITICAL TIMING RULE:
      Decision is at 9:35 AM on day T.
      All features must use data from day T-1 close or earlier.
      The ONLY same-day data allowed is: day-of-week, calendar info (known in advance).

      Implementation: we shift all price/vol features by 1 day so that
      row T contains features computed from T-1 and earlier.
      This means: feature "ret_1d" on row T = return from T-2 close to T-1 close.

    Args:
        extra_dfs: optional dict of additional DataFrames keyed by name
                   (e.g. 'vix3m', 'spy_volume', 'tnx', 'dxy')
    """
    # Align on common dates
    spx = spx_df[["Open", "High", "Low", "Close"]].copy()
    spx.columns = ["spx_open", "spx_high", "spx_low", "spx_close"]
    vix = vix_df[["Close"]].rename(columns={"Close": "vix_close"})
    df = spx.join(vix, how="inner").sort_index()
    df = df.dropna(subset=["spx_close", "vix_close"])

    # Join extra data sources
    if extra_dfs:
        for name, edf in extra_dfs.items():
            if edf is not None and not edf.empty:
                for col in edf.columns:
                    df = df.join(edf[[col]], how="left")

    # === USE PREVIOUS DAY'S CLOSE for all price-based features ===
    # c = yesterday's close (available at decision time)
    c = df["spx_close"].shift(1)
    h = df["spx_high"].shift(1)
    l = df["spx_low"].shift(1)
    o = df["spx_open"].shift(1)
    v = df["vix_close"].shift(1)

    # =====================================================================
    # ALL features below use c/h/l/o/v which are SHIFTED by 1 day.
    # Row T contains features from T-1 close. This prevents look-ahead bias.
    # Calendar features (dow, month, opex) use day T directly since they're
    # known in advance — you know what day of the week tomorrow is.
    # =====================================================================

    # --- Price features (all from T-1 close and earlier) ---
    for n in [1, 3, 5, 10, 20]:
        df[f"ret_{n}d"] = c.pct_change(n)  # c is already shifted, so this is T-2 to T-1

    for w in [20, 50, 200]:
        sma = c.rolling(w).mean()
        df[f"dist_sma{w}_pct"] = (c - sma) / sma * 100
        df[f"sma{w}"] = sma

    # Intraday range (yesterday's high-low)
    df["range_pct"] = (h - l) / c * 100
    # Gap % (yesterday's open vs day-before-yesterday's close)
    df["gap_pct"] = (o - c.shift(1)) / c.shift(1) * 100

    # --- Volatility features (from T-1 VIX close and earlier) ---
    df["vix"] = v  # T-1 VIX close
    df["vix_zscore_252"] = (v - v.rolling(252).mean()) / v.rolling(252).std()
    df["vix_chg_1d"] = v.diff(1)
    df["vix_chg_5d"] = v.diff(5)

    log_ret = np.log(c / c.shift(1))
    for w in [5, 10, 20]:
        df[f"rv_{w}d"] = log_ret.rolling(w).std() * np.sqrt(252) * 100

    df["rv_iv_ratio"] = df["rv_20d"] / v.replace(0, np.nan)
    df["iv_rank_252"] = v.rolling(252).apply(
        lambda x: pd.Series(x).rank(pct=True).iloc[-1], raw=False
    )

    # --- NEW: VIX term structure (VIX vs VIX3M) ---
    if "vix3m_close" in df.columns:
        vix3m = df["vix3m_close"].shift(1)  # T-1 close
        df["vix_term_structure"] = v / vix3m.replace(0, np.nan)  # >1 = backwardation (danger)
        df["vix_contango"] = (vix3m - v) / vix3m.replace(0, np.nan) * 100  # negative = backwardation
    else:
        df["vix_term_structure"] = 1.0
        df["vix_contango"] = 0.0

    # --- NEW: SPY volume (T-1) ---
    if "spy_volume" in df.columns:
        spy_vol = df["spy_volume"].shift(1)  # T-1 volume
        df["volume_raw"] = spy_vol
        df["volume_vs_20d"] = spy_vol / spy_vol.rolling(20).mean().replace(0, np.nan)
        df["low_volume"] = (df["volume_vs_20d"] < 0.7).astype(float)
    else:
        df["volume_vs_20d"] = 1.0
        df["low_volume"] = 0.0

    # --- NEW: 10Y Treasury yield (T-1) ---
    if "tnx_close" in df.columns:
        tnx = df["tnx_close"].shift(1)
        df["yield_10y"] = tnx
        df["yield_chg_5d"] = tnx.diff(5)
    else:
        df["yield_10y"] = 0.0
        df["yield_chg_5d"] = 0.0

    # --- NEW: Dollar index DXY (T-1) ---
    if "dxy_close" in df.columns:
        dxy = df["dxy_close"].shift(1)
        df["dxy"] = dxy
        df["dxy_chg_5d"] = dxy.pct_change(5) * 100
    else:
        df["dxy"] = 0.0
        df["dxy_chg_5d"] = 0.0

    # --- Regime features (from T-1 close) ---
    df["above_sma50"] = (c > df["sma50"]).astype(int)
    df["above_sma200"] = (c > df["sma200"]).astype(int)
    df["vix_zone"] = pd.cut(
        v, bins=[-np.inf, 15, 24, 35, np.inf], labels=[0, 1, 2, 3]
    ).astype(float)
    df["momentum"] = (df["sma20"] - df["sma50"]) / df["sma50"].replace(0, np.nan) * 100

    # Consecutive up/down days (from T-1 and earlier)
    daily_dir = np.sign(c.diff())
    consec = pd.Series(0.0, index=df.index)
    prev_val = 0.0
    for i in range(len(daily_dir)):
        d = daily_dir.iloc[i]
        if pd.isna(d) or d == 0:
            prev_val = 0.0
        elif np.sign(prev_val) == d:
            prev_val += d
        else:
            prev_val = d
        consec.iloc[i] = max(-5, min(5, prev_val))
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

    # --- Microstructure features (all from T-1 and earlier) ---
    # True range using shifted h/l/c
    prev_c = c.shift(1)  # T-2 close
    true_range = pd.concat([
        h - l,
        (h - prev_c).abs(),
        (l - prev_c).abs(),
    ], axis=1).max(axis=1)
    df["atr_5"] = true_range.rolling(5).mean() / c * 100
    df["atr_10"] = true_range.rolling(10).mean() / c * 100
    df["atr_ratio"] = df["atr_5"] / df["atr_10"].replace(0, np.nan)

    # Bollinger band position (T-1 close relative to T-1 bands)
    bb_sma = c.rolling(20).mean()
    bb_std = c.rolling(20).std()
    df["bb_position"] = (c - bb_sma) / (2 * bb_std).replace(0, np.nan)

    # RSI (14-day, from T-1 close series)
    delta_price = c.diff()
    gain = delta_price.clip(lower=0).rolling(14).mean()
    loss_rs = (-delta_price.clip(upper=0)).rolling(14).mean()
    rs = gain / loss_rs.replace(0, np.nan)
    df["rsi_14"] = 100 - (100 / (1 + rs))
    df["rsi_14_norm"] = (df["rsi_14"] - 50) / 50

    # VIX acceleration (from T-1 VIX)
    df["vix_acceleration"] = df["vix_chg_5d"] / df["vix_chg_1d"].rolling(5).sum().replace(0, np.nan)

    # High-low skew (T-1 bar)
    df["hl_skew"] = (h - c) / (c - l).replace(0, np.nan) - 1

    # Move conviction (T-1)
    df["move_conviction"] = df["ret_1d"].abs() * df["range_pct"]

    # Distance from recent high/low (T-1 close vs rolling high/low of T-1 series)
    df["dist_from_20d_high"] = (c - c.rolling(20).max()) / c.rolling(20).max() * 100
    df["dist_from_20d_low"] = (c - c.rolling(20).min()) / c.rolling(20).min() * 100

    # VWAP proxy (5-day rolling typical price from T-1 bars)
    typical_price = (h + l + c) / 3
    df["vwap_proxy"] = typical_price.rolling(5).mean()
    df["dist_from_vwap"] = (c - df["vwap_proxy"]) / df["vwap_proxy"].replace(0, np.nan) * 100

    # GEX/pin proxies (from T-1 close)
    nearest_25 = (c / 25).round() * 25
    df["dist_to_pin_25"] = (c - nearest_25) / c * 100
    df["dist_to_pin_25_abs"] = df["dist_to_pin_25"].abs()

    nearest_50 = (c / 50).round() * 50
    df["dist_to_pin_50"] = (c - nearest_50) / c * 100

    df["gamma_regime"] = np.where(
        df["rv_iv_ratio"] < 0.8, 1,
        np.where(df["rv_iv_ratio"] > 1.2, -1, 0)
    ).astype(float)

    df["dist_to_flip_proxy"] = df["dist_sma20_pct"]
    df["near_flip_zone"] = (df["dist_sma20_pct"].abs() < 0.5).astype(float)
    df["dist_to_put_wall_proxy"] = df["dist_from_20d_low"]
    df["dist_to_call_wall_proxy"] = df["dist_from_20d_high"]
    df["skew_proxy"] = (v - df["rv_10d"]) / v.replace(0, np.nan)

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
        "vix_term_structure", "vix_contango",
        "volume_vs_20d", "yield_10y", "yield_chg_5d",
        "dxy", "dxy_chg_5d",
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
        + ["gamma_regime", "near_flip_zone", "dist_to_pin_25_abs", "low_volume"]
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
    import yfinance as yf

    print("Loading SPX + VIX data 2014-2025 (extra year for rolling warmup)...")
    spx_df, vix_df = load_spx_vix(date(2014, 1, 1), date(2025, 12, 31))
    print(f"  SPX rows: {len(spx_df)}, VIX rows: {len(vix_df)}")

    # Fetch additional data sources (all free from yfinance)
    extra_dfs = {}
    print("Fetching VIX3M (term structure)...")
    try:
        vix3m = yf.download("^VIX3M", start="2014-01-01", end="2025-12-31", progress=False)
        if not vix3m.empty:
            vix3m_col = vix3m["Close"].copy()
            if isinstance(vix3m_col, pd.DataFrame):
                vix3m_col = vix3m_col.iloc[:, 0]
            extra_dfs["vix3m"] = pd.DataFrame({"vix3m_close": vix3m_col})
            print(f"  VIX3M: {len(extra_dfs['vix3m'])} days")
    except Exception as e:
        print(f"  VIX3M failed: {e}")

    print("Fetching SPY (volume)...")
    try:
        spy = yf.download("SPY", start="2014-01-01", end="2025-12-31", progress=False)
        if not spy.empty:
            spy_vol = spy["Volume"].copy()
            if isinstance(spy_vol, pd.DataFrame):
                spy_vol = spy_vol.iloc[:, 0]
            extra_dfs["spy"] = pd.DataFrame({"spy_volume": spy_vol})
            print(f"  SPY volume: {len(extra_dfs['spy'])} days")
    except Exception as e:
        print(f"  SPY volume failed: {e}")

    print("Fetching 10Y Treasury (^TNX)...")
    try:
        tnx = yf.download("^TNX", start="2014-01-01", end="2025-12-31", progress=False)
        if not tnx.empty:
            tnx_col = tnx["Close"].copy()
            if isinstance(tnx_col, pd.DataFrame):
                tnx_col = tnx_col.iloc[:, 0]
            extra_dfs["tnx"] = pd.DataFrame({"tnx_close": tnx_col})
            print(f"  10Y yield: {len(extra_dfs['tnx'])} days")
    except Exception as e:
        print(f"  10Y yield failed: {e}")

    print("Fetching DXY (Dollar Index)...")
    try:
        dxy = yf.download("DX-Y.NYB", start="2014-01-01", end="2025-12-31", progress=False)
        if not dxy.empty:
            dxy_col = dxy["Close"].copy()
            if isinstance(dxy_col, pd.DataFrame):
                dxy_col = dxy_col.iloc[:, 0]
            extra_dfs["dxy"] = pd.DataFrame({"dxy_close": dxy_col})
            print(f"  DXY: {len(extra_dfs['dxy'])} days")
    except Exception as e:
        print(f"  DXY failed: {e}")

    print("Computing features (with look-ahead bias protection)...")
    features = compute_features(spx_df, vix_df, extra_dfs=extra_dfs)

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
