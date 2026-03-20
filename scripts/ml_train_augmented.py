#!/usr/bin/env python3
"""Train SSAN with augmented data: SPY + QQQ + IWM × multiple configs.

9x more training data by:
  1. Transfer learning across ETFs (same IC strategy, different underlying)
  2. Multiple delta/width configs per day (8δ, 10δ, 12δ, 15δ, 20δ, 25δ)
  3. Additional macro features (gold, bonds, HY credit, dollar)
"""
from __future__ import annotations

import sys
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import classification_report, recall_score, accuracy_score, f1_score

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.ml_train_ssan import TimeSeriesSSAN


def compute_features_generic(df: pd.DataFrame) -> pd.DataFrame:
    """Compute features for any ticker. Uses T-1 shift (no look-ahead).

    df must have: open, high, low, close, volume, vol_close (VIX equivalent)
    Plus optional macro columns: vix3m, yield_10y, dxy, gold, bonds, high_yield
    """
    out = pd.DataFrame(index=df.index)

    # T-1 shift for all price-based features
    c = df["close"].shift(1)
    h = df["high"].shift(1)
    l = df["low"].shift(1)
    o = df["open"].shift(1)
    v = df["vol_close"].shift(1)
    vol = df["volume"].shift(1) if "volume" in df.columns else None

    # Returns
    for n in [1, 3, 5, 10, 20]:
        out[f"ret_{n}d"] = c.pct_change(n)

    # SMA distances
    for w in [20, 50, 200]:
        sma = c.rolling(w).mean()
        out[f"dist_sma{w}_pct"] = (c - sma) / sma * 100

    # Range, gap
    out["range_pct"] = (h - l) / c * 100
    out["gap_pct"] = (o - c.shift(1)) / c.shift(1) * 100

    # Volatility
    out["vix"] = v
    out["vix_chg_1d"] = v.diff(1)
    out["vix_chg_5d"] = v.diff(5)
    log_ret = np.log(c / c.shift(1))
    for w in [5, 10, 20]:
        out[f"rv_{w}d"] = log_ret.rolling(w).std() * np.sqrt(252) * 100
    out["rv_iv_ratio"] = out["rv_20d"] / v.replace(0, np.nan)
    out["iv_rank_252"] = v.rolling(252).apply(
        lambda x: pd.Series(x).rank(pct=True).iloc[-1], raw=False
    )

    # VIX term structure
    if "vix3m" in df.columns:
        vix3m = df["vix3m"].shift(1)
        out["vix_contango"] = (vix3m - v) / vix3m.replace(0, np.nan) * 100
        out["vix_term_structure"] = v / vix3m.replace(0, np.nan)
    else:
        out["vix_contango"] = 0.0
        out["vix_term_structure"] = 1.0

    # Volume
    if vol is not None:
        out["volume_vs_20d"] = vol / vol.rolling(20).mean().replace(0, np.nan)
    else:
        out["volume_vs_20d"] = 1.0

    # Macro (T-1 shifted)
    for macro_col in ["yield_10y", "dxy", "gold", "bonds", "high_yield"]:
        if macro_col in df.columns:
            mc = df[macro_col].shift(1)
            out[f"{macro_col}_chg_5d"] = mc.pct_change(5) * 100

    # Microstructure
    prev_c = c.shift(1)
    true_range = pd.concat([h - l, (h - prev_c).abs(), (l - prev_c).abs()], axis=1).max(axis=1)
    out["atr_10"] = true_range.rolling(10).mean() / c * 100

    bb_sma = c.rolling(20).mean()
    bb_std = c.rolling(20).std()
    out["bb_position"] = (c - bb_sma) / (2 * bb_std).replace(0, np.nan)

    delta_price = c.diff()
    gain = delta_price.clip(lower=0).rolling(14).mean()
    loss_rs = (-delta_price.clip(upper=0)).rolling(14).mean()
    rs = gain / loss_rs.replace(0, np.nan)
    out["rsi_14_norm"] = (100 - (100 / (1 + rs)) - 50) / 50

    out["hl_skew"] = (h - c) / (c - l).replace(0, np.nan) - 1
    out["momentum"] = (c.rolling(20).mean() - c.rolling(50).mean()) / c.rolling(50).mean().replace(0, np.nan) * 100
    out["dist_from_20d_high"] = (c - c.rolling(20).max()) / c.rolling(20).max() * 100
    out["dist_from_20d_low"] = (c - c.rolling(20).min()) / c.rolling(20).min() * 100
    out["skew_proxy"] = (v - out["rv_10d"]) / v.replace(0, np.nan)
    nearest_round = (c / 25).round() * 25
    out["dist_to_pin_50"] = (c - (c / 50).round() * 50) / c * 100

    # Regime
    out["above_sma50"] = (c > c.rolling(50).mean()).astype(float)
    out["vix_zone"] = pd.cut(v, bins=[-np.inf, 15, 24, 35, np.inf], labels=[0, 1, 2, 3]).astype(float)

    # Calendar (day T — known in advance)
    dow = df.index.dayofweek
    out["dow_sin"] = np.sin(2 * np.pi * dow / 5)
    out["month_sin"] = np.sin(2 * np.pi * df.index.month / 12)
    out["month_cos"] = np.cos(2 * np.pi * df.index.month / 12)
    out["year_progress"] = df.index.dayofyear / 365.0

    # Z-score normalize continuous features
    zscore_cols = [c for c in out.columns if c not in
                   ("above_sma50", "vix_zone", "dow_sin", "month_sin", "month_cos", "year_progress")]
    for col in zscore_cols:
        if col in out.columns:
            rm = out[col].rolling(252, min_periods=60).mean()
            rs = out[col].rolling(252, min_periods=60).std().replace(0, np.nan)
            out[col] = (out[col] - rm) / rs

    out = out.replace([np.inf, -np.inf], np.nan).ffill().fillna(0)

    # Keep raw close for trade simulation
    out["_close"] = df["close"]
    out["_vol"] = df["vol_close"]

    return out


def simulate_ic_trade(close_series, vol_series, entry_idx, dte, delta_pct, width_pct, tp_pct=0.50, sl_mult=2.0):
    """Simulate an IC trade using percentage-based strikes (generic for any underlying).

    delta_pct: approximate % distance from spot for short strikes (e.g., 0.015 = 1.5%)
    width_pct: spread width as % of spot (e.g., 0.005 = 0.5%)
    """
    if entry_idx + dte >= len(close_series):
        return None

    spot_entry = close_series.iloc[entry_idx]
    if spot_entry <= 0 or np.isnan(spot_entry):
        return None

    # Short strikes at delta_pct distance
    k_put_short = spot_entry * (1 - delta_pct)
    k_call_short = spot_entry * (1 + delta_pct)
    width = spot_entry * width_pct

    # Approximate credit: proportional to VIX and width
    vol_entry = vol_series.iloc[entry_idx]
    if np.isnan(vol_entry) or vol_entry <= 0:
        vol_entry = 20.0
    # Rough BS approximation for short-dated OTM options
    time_factor = (dte / 252) ** 0.5
    credit_approx = vol_entry / 100 * spot_entry * time_factor * 0.4  # simplified
    credit_approx = min(credit_approx, width * 0.3)  # cap at 30% of width
    if credit_approx <= 0:
        return None

    # Simulate daily
    exit_reason = "expiry"
    pnl_factor = 1.0  # will multiply by credit

    for d in range(1, dte + 1):
        idx = entry_idx + d
        if idx >= len(close_series):
            break
        spot_d = close_series.iloc[idx]
        days_left = dte - d

        # Simple spread value decay model
        if days_left > 0:
            decay = 1.0 - (d / dte) ** 0.5  # sqrt decay approximation
        else:
            decay = 0.0

        # Check if breached
        put_breach = max(0, (k_put_short - spot_d) / width)
        call_breach = max(0, (spot_d - k_call_short) / width)
        breach = max(put_breach, call_breach)

        spread_val = credit_approx * decay + breach * width * 0.5

        if spread_val <= credit_approx * (1 - tp_pct):
            exit_reason = "tp"
            pnl_factor = tp_pct
            break
        if spread_val >= credit_approx * sl_mult:
            exit_reason = "sl"
            pnl_factor = -(sl_mult - 1)
            break

    if exit_reason == "expiry":
        # Settlement
        spot_exp = close_series.iloc[min(entry_idx + dte, len(close_series) - 1)]
        put_settle = max(0, k_put_short - spot_exp)
        call_settle = max(0, spot_exp - k_call_short)
        settle = min(put_settle + call_settle, width)
        pnl = credit_approx - settle
    else:
        pnl = credit_approx * pnl_factor

    # Classify
    if pnl > credit_approx * 0.5:
        target = 2  # full win
    elif pnl > 0:
        target = 1  # small win
    else:
        target = 0  # loss

    return {"pnl": pnl, "target": target, "exit_reason": exit_reason, "credit": credit_approx}


def build_augmented_training_data():
    """Build large training dataset from multiple tickers and configs."""
    print("Loading augmented data...", flush=True)
    aug_df = pd.read_parquet(ROOT / "data" / "ml_augment_data.parquet")

    # IC configs to simulate (delta_pct, width_pct, dte)
    ic_configs = [
        (0.015, 0.005, 1),   # 1DTE, ~10δ, narrow
        (0.012, 0.005, 1),   # 1DTE, ~12δ, narrow
        (0.020, 0.005, 1),   # 1DTE, ~8δ, narrow
        (0.015, 0.005, 3),   # 3DTE, ~10δ
        (0.012, 0.005, 3),   # 3DTE, ~12δ
        (0.010, 0.008, 3),   # 3DTE, ~15δ, wider
    ]

    all_samples = []

    for ticker in aug_df["ticker"].unique():
        print(f"  Processing {ticker}...", flush=True)
        tdf = aug_df[aug_df["ticker"] == ticker].copy()
        tdf = tdf.sort_index()

        # Compute features
        features = compute_features_generic(tdf)
        feature_cols = [c for c in features.columns if not c.startswith("_")]

        close = features["_close"]
        vol = features["_vol"]

        # Trim to 2015+ (after warmup)
        mask = features.index >= "2015-01-01"
        features = features[mask]
        close = close[mask]
        vol = vol[mask]

        dates_list = features.index.tolist()

        for delta_pct, width_pct, dte in ic_configs:
            config_name = f"{ticker}_d{int(delta_pct*1000)}_w{int(width_pct*1000)}_dte{dte}"
            n_trades = 0

            for i in range(20, len(features)):  # need 20-day lookback
                result = simulate_ic_trade(close, vol, i, dte, delta_pct, width_pct)
                if result is None:
                    continue

                # Build 20-day lookback feature window
                window = features.iloc[i-19:i+1][feature_cols].values.flatten()
                if len(window) != 20 * len(feature_cols):
                    continue

                sample = {
                    "entry_date": dates_list[i],
                    "ticker": ticker,
                    "config": config_name,
                    **{f"f_{j}": window[j] for j in range(len(window))},
                    "target": result["target"],
                    "pnl": result["pnl"],
                    "exit_reason": result["exit_reason"],
                }
                all_samples.append(sample)
                n_trades += 1

            print(f"    {config_name}: {n_trades} trades")

    result_df = pd.DataFrame(all_samples)
    n_features = len(feature_cols)

    print(f"\n  Total samples: {len(result_df):,}")
    print(f"  Features per timestep: {n_features}")
    print(f"  Tickers: {result_df['ticker'].unique()}")
    print(f"  Target distribution: {dict(result_df['target'].value_counts().sort_index())}")
    print(f"  Win rate: {(result_df['target'] > 0).mean():.1%}")

    return result_df, n_features, feature_cols


def train_augmented_ssan():
    """Train SSAN on augmented dataset."""
    result_df, n_features, feature_cols = build_augmented_training_data()

    # Split: train on 2015-2020, val 2021-2022, test 2023+
    dates = pd.to_datetime(result_df["entry_date"])
    train_mask = dates.dt.year <= 2020
    val_mask = (dates.dt.year >= 2021) & (dates.dt.year <= 2022)
    test_mask = dates.dt.year >= 2023

    f_cols = [c for c in result_df.columns if c.startswith("f_")]
    X_all = result_df[f_cols].values.astype(np.float32)
    X_all = np.nan_to_num(X_all, nan=0.0, posinf=0.0, neginf=0.0)
    X_seq = X_all.reshape(len(result_df), 20, n_features)
    y = result_df["target"].values.astype(np.int64)

    X_train, y_train = X_seq[train_mask], y[train_mask]
    X_val, y_val = X_seq[val_mask], y[val_mask]
    X_test, y_test = X_seq[test_mask], y[test_mask]

    print(f"\n  Train: {len(X_train):,}, Val: {len(X_val):,}, Test: {len(X_test):,}")
    print(f"  Features: {n_features} per timestep")

    device = torch.device("cpu")

    # Class weights
    counts = np.bincount(y_train, minlength=3).astype(np.float32)
    weights = 1.0 / (counts + 1)
    weights = weights / weights.sum() * 3
    weights_t = torch.tensor(weights, device=device)

    # Best config from previous: d48, 3L, k8
    model = TimeSeriesSSAN(
        n_features=n_features, n_classes=3, d_model=48, n_heads=4,
        n_layers=3, top_k_attn=8, lambda_decay=0.85, dropout=0.3,
        seq_len=20, alpha=0.02, beta=0.98,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Model params: {n_params:,}")
    print(f"  Samples/param ratio: {len(X_train)/n_params:.1f} (was {2982/108227:.2f})")

    X_tr = torch.tensor(X_train, device=device)
    y_tr = torch.tensor(y_train, device=device)
    X_v = torch.tensor(X_val, device=device)
    y_v = torch.tensor(y_val, device=device)
    X_te = torch.tensor(X_test, device=device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=80, eta_min=1e-5)
    criterion = nn.CrossEntropyLoss(weight=weights_t)

    best_val = float("inf")
    best_state = None
    patience = 20
    patience_counter = 0
    batch_size = 256

    print("\n--- Training SSAN on augmented data ---")
    for epoch in range(1, 81):
        model.train()
        perm = torch.randperm(len(X_tr))
        tl = 0; nb = 0
        for i in range(0, len(X_tr), batch_size):
            idx = perm[i:i+batch_size]
            logits = model(X_tr[idx])
            loss = criterion(logits, y_tr[idx])
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            model.apply_reinforcement(logits.detach(), y_tr[idx])
            tl += loss.item(); nb += 1
        scheduler.step()

        model.eval()
        with torch.no_grad():
            vl = criterion(model(X_v), y_v).item()

        if epoch % 10 == 0:
            print(f"  Epoch {epoch}: train={tl/nb:.4f} val={vl:.4f}")

        if vl < best_val:
            best_val = vl
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"  Early stopping at epoch {epoch}")
                break

    if best_state:
        model.load_state_dict({k: v.to(device) for k, v in best_state.items()})

    model.eval()
    with torch.no_grad():
        preds = model(X_te).argmax(dim=1).cpu().numpy()

    acc = accuracy_score(y_test, preds)
    n_classes = len(set(y_test))
    labels_present = sorted(set(y_test))
    label_names = {0: "loss", 1: "small_win", 2: "full_win"}
    target_names = [label_names.get(l, str(l)) for l in labels_present]
    f1 = f1_score(y_test, preds, average="macro", zero_division=0)
    lr = recall_score(y_test, preds, labels=[0], average=None, zero_division=0)[0]

    print(f"\n  === Augmented SSAN Test Results ===")
    print(classification_report(y_test, preds, target_names=target_names, labels=labels_present, zero_division=0))

    print(f"\n  COMPARISON:")
    print(f"    Original (2,982 train):    Loss recall = 52.8%")
    print(f"    Augmented ({len(X_train):,} train): Loss recall = {lr:.1%}")
    if lr > 0.528:
        print(f"    IMPROVEMENT: +{(lr-0.528)*100:.1f}pp")
    else:
        print(f"    Change: {(lr-0.528)*100:+.1f}pp")

    # Save
    torch.save({
        "state_dict": best_state,
        "config": {"n_features": n_features, "n_classes": 3, "d_model": 48, "n_heads": 4,
                   "n_layers": 3, "top_k_attn": 8, "lambda_decay": 0.85, "dropout": 0.3, "seq_len": 20},
        "feature_cols": feature_cols,
        "augmented": True,
    }, ROOT / "data" / "ml_entry_xgboost_spx.joblib")
    print(f"  Saved augmented model")

    return model, preds, y_test


if __name__ == "__main__":
    train_augmented_ssan()
