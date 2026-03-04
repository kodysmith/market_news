#!/usr/bin/env python3
"""GEX-based 7DTE broken-wing butterfly (BWB) put backtest.

Uses SPX options Parquet from collect_spx_options. Trading days only (NYSE calendar).
When GEX is solidly positive, sell a 7-trading-session DTE ~20-delta BWB put:
  - Buy 1 put at K_upper  (closer to ATM, protective wing)
  - Sell 2 puts at K_body  (~20-delta, the profit center)
  - Buy 1 put at K_lower  (far OTM, crash wing — wider than upper)
Broken wing: lower_width > upper_width → net credit entry with capped downside risk.
Max profit at K_body = credit + upper_width.  Max loss = lower_width - upper_width - credit.
P&L uses actual chain mids, fixed-dollar slippage, and commission per leg.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from datetime import date, datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import QuantEngine.gex_calculator as gex_calculator
from scipy.stats import norm

from scripts.trading_calendar import (
    entry_date_for_dte,
    get_trading_days_index,
    get_trading_days_in_range,
    trading_days_between,
)

# Optional: use VectorBT for standard portfolio stats (pip install vectorbt)
try:
    import vectorbt as vbt
    _VBT_AVAILABLE = True
except ImportError:
    vbt = None  # type: ignore[assignment]
    _VBT_AVAILABLE = False

# ---------- Black-Scholes ----------
def _d1(S: float, K: float, T: float, r: float, sigma: float) -> float:
    if T <= 0 or sigma <= 0 or S <= 0 or K <= 0:
        return float("nan")
    return (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))


def _d2(S: float, K: float, T: float, sigma: float) -> float:
    d1_val = _d1(S, K, T, 0.0, sigma)
    return d1_val - sigma * np.sqrt(T) if not np.isnan(d1_val) else float("nan")


def bs_put_delta(S: float, K: float, T: float, r: float, sigma: float) -> float:
    d1_val = _d1(S, K, T, r, sigma)
    if np.isnan(d1_val):
        return float("nan")
    return float(norm.cdf(d1_val) - 1.0)


def bs_put_price(S: float, K: float, T: float, r: float, sigma: float) -> float:
    if T <= 0 or sigma <= 0:
        return 0.0
    d1_val = _d1(S, K, T, r, sigma)
    d2_val = _d2(S, K, T, sigma)
    if np.isnan(d1_val) or np.isnan(d2_val):
        return 0.0
    return float(K * np.exp(-r * T) * norm.cdf(-d2_val) - S * norm.cdf(-d1_val))


def years_to_expiry(entry_date: date, exp_date: date) -> float:
    return (exp_date - entry_date).days / 365.0 if exp_date > entry_date else 0.0


# ---------- Adapter: DataFrame -> GEX snap ----------
def _get_contract_type(r: pd.Series) -> str | None:
    ct = (r.get("contract_type") or r.get("option_type") or "").lower()
    if ct in ("call", "put"):
        return ct
    if ct in ("c", "p"):
        return "call" if ct == "c" else "put"
    return None


def _get_iv(r: pd.Series, default_iv: float | None) -> float | None:
    iv = r.get("implied_volatility")
    if iv is None or (isinstance(iv, float) and np.isnan(iv)):
        iv = r.get("iv")
    if pd.isna(iv) or (isinstance(iv, (int, float)) and iv <= 0):
        return default_iv
    return float(iv)


def _sigma_from_iv(iv: float) -> float:
    return float(iv) / 100.0 if float(iv) > 3 else float(iv)


def df_to_snap(df: pd.DataFrame, default_iv: float | None = None) -> tuple[dict, dict]:
    rows = []
    dropped_oi = dropped_iv = dropped_exp = dropped_ct = dropped_strike = 0
    for _, r in df.iterrows():
        oi = r.get("open_interest")
        if pd.isna(oi) or (isinstance(oi, (int, float)) and oi <= 0):
            dropped_oi += 1
            continue
        iv = _get_iv(r, default_iv)
        if iv is None:
            dropped_iv += 1
            continue
        exp = r.get("expiration_date")
        if pd.isna(exp):
            dropped_exp += 1
            continue
        exp_str = exp.strftime("%Y-%m-%d") if hasattr(exp, "strftime") else str(exp)[:10]
        ct = _get_contract_type(r)
        if ct is None:
            dropped_ct += 1
            continue
        strike = r.get("strike")
        if pd.isna(strike):
            dropped_strike += 1
            continue
        mult = r.get("shares_per_contract", 100)
        if pd.isna(mult):
            mult = 100
        mult = int(mult) if isinstance(mult, (int, float)) else 100
        rows.append({
            "details": {
                "strike_price": float(strike),
                "expiration_date": exp_str,
                "contract_type": ct,
                "shares_per_contract": mult,
            },
            "open_interest": int(oi),
            "implied_volatility": float(iv),
            "symbol": r.get("option_symbol"),
        })
    stats = {
        "adapter_rows": len(rows),
        "df_rows": len(df),
        "dropped_oi": dropped_oi,
        "dropped_iv": dropped_iv,
        "dropped_exp": dropped_exp,
        "dropped_ct": dropped_ct,
        "dropped_strike": dropped_strike,
    }
    return {"results": rows}, stats


def infer_strike_increment(df: pd.DataFrame, spot: float) -> float:
    strikes = sorted(df["strike"].dropna().unique())
    if len(strikes) < 2:
        return 5.0
    diffs = [strikes[i + 1] - strikes[i] for i in range(len(strikes) - 1) if strikes[i + 1] > strikes[i]]
    return float(min(diffs)) if diffs else 5.0


def load_spot_series(start_date: date, end_date: date) -> pd.Series:
    try:
        import yfinance as yf
    except ImportError:
        raise RuntimeError("yfinance is required; pip install yfinance")
    ticker = yf.Ticker("^GSPC")
    start = start_date.isoformat()
    end = (end_date + timedelta(days=5)).isoformat()
    hist = ticker.history(start=start, end=end)
    if hist.empty:
        return pd.Series(dtype=float)
    hist.index = hist.index.tz_localize(None).normalize()
    return hist["Close"]


def load_vix_series(start_date: date, end_date: date) -> pd.Series:
    """Load VIX close for regime filter. Returns empty series on failure."""
    try:
        import yfinance as yf
        ticker = yf.Ticker("^VIX")
        hist = ticker.history(start=start_date.isoformat(), end=(end_date + timedelta(days=5)).isoformat())
        if hist.empty:
            return pd.Series(dtype=float)
        hist.index = hist.index.tz_localize(None).normalize()
        return hist["Close"]
    except Exception:
        return pd.Series(dtype=float)


def vix_on_date(vix_series: pd.Series, d: date) -> float | None:
    """VIX close on date d if available; no lookahead."""
    if vix_series.empty:
        return None
    ts = pd.Timestamp(d)
    if ts not in vix_series.index:
        return None
    val = vix_series.loc[ts]
    if pd.isna(val) or val <= 0:
        return None
    return float(val)


def spot_on_trading_day(spot_series: pd.Series, trading_dates: set[date], d: date) -> float | None:
    """Return SPX close for date d only if d is a trading day and we have data. No lookahead."""
    if d not in trading_dates:
        return None
    ts = pd.Timestamp(d)
    if ts not in spot_series.index:
        return None
    val = spot_series.loc[ts]
    if pd.isna(val) or val <= 0:
        return None
    return float(val)


def get_mid_from_chain(df: pd.DataFrame, strike: float, exp_date: date, contract_type: str) -> float | None:
    """Per-share mid for (strike, exp_date, contract_type). Uses mid column or (bid+ask)/2; else None."""
    exp_d = pd.to_datetime(df["expiration_date"], errors="coerce").dt.date
    mask = (df["strike"] == strike) & (exp_d == exp_date)
    ct_col = "contract_type" if "contract_type" in df.columns else "option_type"
    ct_ser = df[ct_col].astype(str).str.lower()
    if contract_type == "put":
        mask &= (ct_ser == "put") | (ct_ser == "p")
    else:
        mask &= (ct_ser == "call") | (ct_ser == "c")
    sub = df.loc[mask]
    if sub.empty:
        return None
    row = sub.iloc[0]
    mid = row.get("mid")
    if mid is not None and not (isinstance(mid, float) and np.isnan(mid)):
        return float(mid)
    bid, ask = row.get("bid"), row.get("ask")
    if bid is not None and ask is not None and not (np.isnan(bid) or np.isnan(ask)):
        return (float(bid) + float(ask)) / 2.0
    return None


def discover_snapshot_dates(data_dir: Path, days_limit: int | None) -> list[date]:
    data_dir = data_dir.resolve()
    if not data_dir.exists():
        return []
    daily = list(data_dir.glob("spx_options_[0-9][0-9][0-9][0-9]-[0-9][0-9]-[0-9][0-9].parquet"))
    if daily:
        out = []
        for p in daily:
            try:
                stem = p.stem
                if stem.startswith("spx_options_"):
                    s = stem.replace("spx_options_", "")
                    out.append(datetime.strptime(s, "%Y-%m-%d").date())
            except ValueError:
                continue
        out = sorted(set(out))
        if days_limit:
            cutoff = date.today() - timedelta(days=days_limit)
            out = [d for d in out if d >= cutoff]
        return out
    combined = list(data_dir.glob("spx_options_expirations_*.parquet"))
    if combined:
        try:
            df = pd.read_parquet(combined[0])
        except (ImportError, Exception):
            return []
        if "snapshot_date" in df.columns:
            sd = df["snapshot_date"]
            if hasattr(sd.iloc[0], "date"):
                out = sorted(sd.apply(lambda x: x.date() if hasattr(x, "date") else x).unique())
            else:
                out = sorted(pd.to_datetime(sd).dt.date.unique())
            if days_limit:
                cutoff = date.today() - timedelta(days=days_limit)
                out = [d for d in out if d >= cutoff]
            return out
    return []


def _read_parquet(path: Path, warn_once: list[bool]) -> pd.DataFrame | None:
    try:
        return pd.read_parquet(path)
    except ImportError:
        if warn_once and not warn_once[0]:
            warn_once[0] = True
            print("Parquet engine missing (pip install pyarrow). Cannot load options data.", file=sys.stderr)
        return None
    except Exception:
        return None


def load_options_for_date(data_dir: Path, snapshot_date: date, parquet_warn: list[bool] | None = None) -> pd.DataFrame | None:
    data_dir = data_dir.resolve()
    warn = parquet_warn if parquet_warn is not None else [False]
    p = data_dir / f"spx_options_{snapshot_date.isoformat()}.parquet"
    if p.exists():
        df = _read_parquet(p, warn)
        if df is not None:
            return df
        return None
    for path in data_dir.glob("spx_options_expirations_*.parquet"):
        df = _read_parquet(path, warn)
        if df is None:
            continue
        if "snapshot_date" not in df.columns:
            continue
        sd = df["snapshot_date"]
        if len(sd) == 0:
            continue
        if hasattr(sd.iloc[0], "date"):
            mask = sd.apply(lambda x: (x.date() if hasattr(x, "date") else x) == snapshot_date)
        else:
            mask = pd.to_datetime(sd).dt.date == snapshot_date
        sub = df.loc[mask]
        if not sub.empty:
            return sub
    return None


# ---------- Performance stats ----------
def compute_stats(trades: list[dict], initial_capital: float | None) -> dict:
    if not trades:
        return {}
    pnls = [t["pnl"] for t in trades if t.get("pnl") is not None]
    if not pnls:
        return {}
    pnls_arr = np.array(pnls)
    n = len(pnls_arr)
    expectancy = float(np.mean(pnls_arr))
    wins = pnls_arr[pnls_arr > 0]
    losses = pnls_arr[pnls_arr < 0]
    avg_win = float(np.mean(wins)) if len(wins) else 0.0
    avg_loss = float(np.mean(losses)) if len(losses) else 0.0
    equity = np.cumsum(pnls_arr)
    peak = np.maximum.accumulate(equity)
    drawdown = peak - equity
    max_dd = float(np.max(drawdown)) if len(drawdown) else 0.0
    std = float(np.std(pnls_arr))
    total_pnl = float(np.sum(pnls_arr))
    years = 1.0
    if trades and trades[0].get("entry_date") and trades[-1].get("entry_date"):
        try:
            d0 = datetime.strptime(trades[0]["entry_date"], "%Y-%m-%d").date()
            d1 = datetime.strptime(trades[-1]["entry_date"], "%Y-%m-%d").date()
            years = max(0.08, (d1 - d0).days / 365.0)
        except Exception:
            pass
    n_per_year = (n / years) if years and n else 1.0
    if std > 0:
        sharpe = (expectancy / std) * np.sqrt(n_per_year) if std else 0.0
        downside = pnls_arr[pnls_arr < 0]
        down_std = float(np.std(downside)) if len(downside) > 0 else 0.0
        sortino = (expectancy / down_std * np.sqrt(n_per_year)) if down_std else 0.0
    else:
        sharpe = sortino = 0.0
    worst_1pct = float(np.percentile(pnls_arr, 1)) if n >= 100 else (float(np.min(pnls_arr)) if n else 0)
    by_month: dict[str, list[float]] = {}
    for t in trades:
        ed = t.get("entry_date")
        if not ed:
            continue
        try:
            ym = ed[:7]
        except Exception:
            ym = ""
        if ym not in by_month:
            by_month[ym] = []
        if t.get("pnl") is not None:
            by_month[ym].append(t["pnl"])
    worst_month_tup = min(((sum(v), k) for k, v in by_month.items() if v), default=(0.0, ""))
    worst_month_pnl = worst_month_tup[0]
    capital = initial_capital if initial_capital and initial_capital > 0 else abs(avg_loss) * 20
    cagr = (1 + total_pnl / capital) ** (1 / years) - 1 if capital and years else 0.0
    by_regime: dict[str, list[float]] = {}
    for t in trades:
        r = t.get("regime") or "unknown"
        if r not in by_regime:
            by_regime[r] = []
        if t.get("pnl") is not None:
            by_regime[r].append(t["pnl"])
    regime_stats = {
        r: {"n": len(v), "total_pnl": sum(v), "avg_pnl": sum(v) / len(v) if v else 0}
        for r, v in by_regime.items()
    }
    return {
        "n_trades": n,
        "expectancy": expectancy,
        "avg_win": avg_win,
        "avg_loss": avg_loss,
        "total_pnl": total_pnl,
        "max_drawdown": max_dd,
        "sharpe_ratio": sharpe,
        "sortino_ratio": sortino,
        "cagr": cagr,
        "worst_1pct_trade": worst_1pct,
        "worst_month_pnl": worst_month_pnl,
        "by_regime": regime_stats,
    }


def build_vectorbt_portfolio(
    trades: list[dict],
    trading_index: pd.DatetimeIndex,
    initial_capital: float = 0.0,
):
    """Build VectorBT-backed stats from GEX 7DTE backtest trades (standard backtesting library).

    Converts trade PnLs to a daily PnL series, then to an equity curve and daily returns.
    Uses vbt.returns accessor for standard metrics (Sharpe, drawdown, win rate, etc.) so
    the GEX 7DTE strategy is evaluated the same way as the rest of the app (see
    market_news_app/backtesting_v2). Returns an object with a .stats() method for display.
    """
    if not _VBT_AVAILABLE or not trades or trading_index is None or len(trading_index) == 0:
        return None
    daily_pnl: dict[str, float] = defaultdict(float)
    for t in trades:
        if t.get("pnl") is None:
            continue
        ed = t.get("exit_date") or ""
        if ed:
            daily_pnl[ed] += float(t["pnl"])
    # Daily PnL series aligned to trading_index
    pnl_series = pd.Series(0.0, index=trading_index)
    for date_str, pnl in daily_pnl.items():
        try:
            ts = pd.Timestamp(date_str)
            if ts in pnl_series.index:
                pnl_series.loc[ts] = pnl
            else:
                idx = trading_index[trading_index.normalize() == pd.Timestamp(date_str).normalize()]
                if len(idx):
                    pnl_series.loc[idx[0]] = pnl
        except Exception:
            continue
    # Use provided capital or a notional (e.g. one SPX spread) so returns are in a sensible range
    init = float(initial_capital) if initial_capital and initial_capital > 0 else 10000.0
    equity = init + pnl_series.cumsum()
    # Avoid div-by-zero: no return on first day if no prior value
    returns = equity.pct_change().fillna(0.0)
    # VectorBT returns accessor (standard stats)
    ret_acc = returns.vbt.returns(freq="d")
    return ret_acc


def _get_bwb_mid_from_snapshot(
    data_dir: Path,
    d: date,
    K_upper: float,
    K_body: float,
    K_lower: float,
    exp_date: date,
    parquet_warn: list[bool],
) -> float | None:
    """Load that day's option snapshot and return the BWB close-debit (cost to buy back).

    BWB value = 2*mid(body) - mid(upper) - mid(lower).
    Positive means it still costs money to close; as theta decays, this shrinks toward zero.
    """
    df = load_options_for_date(data_dir, d, parquet_warn)
    if df is None or df.empty:
        return None
    mid_u = get_mid_from_chain(df, K_upper, exp_date, "put")
    mid_b = get_mid_from_chain(df, K_body, exp_date, "put")
    mid_l = get_mid_from_chain(df, K_lower, exp_date, "put")
    if mid_u is not None and mid_b is not None and mid_l is not None:
        val = 2 * mid_b - mid_u - mid_l
        return val
    return None


def run_backtest(
    data_dir: Path,
    days_limit: int | None,
    start_date: date | None,
    end_date: date | None,
    delta_target: float,
    dte_target: int,
    upper_width: float,
    lower_width: float,
    positive_threshold: float | None,
    strike_increment: float | None,
    output_csv: Path | None,
    output_json: Path | None,
    verbose: bool,
    default_iv: float | None,
    profit_target_pct: float = 50.0,
    relax_symmetric: bool = False,
    slippage_per_leg: float = 0.10,
    commission_per_leg: float = 0.65,
    stop_loss_mult: float | None = None,
    stop_loss_pct_max: float | None = None,
    vix_max: float | None = None,
    initial_capital: float | None = 300_000.0,
    ignore_regime: bool = False,
    min_credit: float = 0.50,
    max_concurrent: int | None = 1,
) -> int:
    snapshot_dates = discover_snapshot_dates(data_dir, days_limit)
    if not snapshot_dates:
        print("No snapshot dates found. Ensure data_dir contains spx_options_YYYY-MM-DD.parquet or spx_options_expirations_*.parquet.", file=sys.stderr)
        return (1, [], None)
    if start_date is not None:
        snapshot_dates = [d for d in snapshot_dates if d >= start_date]
    if end_date is not None:
        snapshot_dates = [d for d in snapshot_dates if d <= end_date]
    if not snapshot_dates:
        print("No snapshot dates in range (after --start/--end filter).", file=sys.stderr)
        return (1, [], None)

    cal_start = min(snapshot_dates)
    cal_end = max(snapshot_dates) + timedelta(days=45)
    trading_index = get_trading_days_index(cal_start, cal_end)
    trading_dates_set = set(pd.Timestamp(ts).date() for ts in trading_index)
    snapshot_dates = [d for d in snapshot_dates if d in trading_dates_set]

    spot_series = load_spot_series(cal_start, cal_end)
    if spot_series.empty:
        print("Could not load SPX spot series (yfinance ^GSPC).", file=sys.stderr)
        return (1, [], None)
    reindex_dates = pd.DatetimeIndex([pd.Timestamp(d) for d in trading_dates_set])
    spot_series = spot_series.reindex(reindex_dates).dropna()
    spot_dates = set(pd.Timestamp(ts).date() for ts in spot_series.index)

    vix_series = load_vix_series(cal_start, cal_end) if vix_max is not None else pd.Series(dtype=float)

    if verbose:
        print(f"Found {len(snapshot_dates)} snapshot date(s) on trading calendar", file=sys.stderr)

    orig_now = gex_calculator.get_eastern_now
    orig_today = gex_calculator.get_eastern_today
    r = 0.05
    SPX_MULT = 100
    trades: list[dict] = []
    skip_reasons: dict[str, int] = {}
    parquet_warn = [False]

    for snapshot_date in snapshot_dates:
        spot_entry = spot_on_trading_day(spot_series, spot_dates, snapshot_date)
        if spot_entry is None:
            skip_reasons["no_spot"] = skip_reasons.get("no_spot", 0) + 1
            if verbose:
                print(f"  {snapshot_date}: skip (no spot on trading day)", file=sys.stderr)
            continue
        df = load_options_for_date(data_dir, snapshot_date, parquet_warn)
        if df is None or df.empty:
            skip_reasons["no_df"] = skip_reasons.get("no_df", 0) + 1
            continue
        snap, adapter_stats = df_to_snap(df, default_iv=default_iv)
        if not snap.get("results"):
            skip_reasons["adapter_empty"] = skip_reasons.get("adapter_empty", 0) + 1
            continue
        inc = strike_increment if strike_increment is not None and strike_increment > 0 else infer_strike_increment(df, spot_entry)
        snapshot_dt = datetime(snapshot_date.year, snapshot_date.month, snapshot_date.day, 16, 0, 0)
        try:
            tz = gex_calculator.get_eastern_timezone()
            snapshot_dt = snapshot_dt.replace(tzinfo=tz)
        except Exception:
            pass
        gex_calculator.get_eastern_now = lambda _dt=snapshot_dt: _dt
        gex_calculator.get_eastern_today = lambda _d=snapshot_date: _d
        try:
            df_gex, agg, metrics, skipped, _ = gex_calculator.calculate_gex(
                snap, spot_entry, num_strikes_each_side=20, strike_increment=inc,
            )
        finally:
            gex_calculator.get_eastern_now = orig_now
            gex_calculator.get_eastern_today = orig_today
        regime = (metrics.get("regime") or "").lower().replace("_gamma", "")
        total_gex = metrics.get("total_gex") or 0
        is_symmetric = (metrics.get("diagnostics") or {}).get("is_symmetric", True)
        if positive_threshold is not None and total_gex <= positive_threshold:
            skip_reasons["gex_below_threshold"] = skip_reasons.get("gex_below_threshold", 0) + 1
            continue
        if not ignore_regime and regime != "positive" and not (relax_symmetric and regime == "unknown" and total_gex > 0):
            skip_reasons["regime_not_positive"] = skip_reasons.get("regime_not_positive", 0) + 1
            if verbose:
                print(f"  {snapshot_date}: skip (regime={regime})", file=sys.stderr)
            continue
        if vix_max is not None:
            vix_val = vix_on_date(vix_series, snapshot_date)
            if vix_val is not None and vix_val > vix_max:
                skip_reasons["vix_above_max"] = skip_reasons.get("vix_above_max", 0) + 1
                continue

        exp_dates = set()
        for _, row in df.iterrows():
            exp = row.get("expiration_date")
            if pd.isna(exp):
                continue
            if hasattr(exp, "date"):
                exp_dates.add(exp)
            else:
                exp_dates.add(datetime.strptime(str(exp)[:10], "%Y-%m-%d").date())
        exp_dates = sorted([d for d in exp_dates if d > snapshot_date and d in trading_dates_set])
        if not exp_dates:
            skip_reasons["no_future_exp"] = skip_reasons.get("no_future_exp", 0) + 1
            continue

        # Find expiration closest to dte_target trading days away (within +-2 tolerance)
        best_exp = None
        best_dte_diff = float("inf")
        for e in exp_dates:
            dte_actual = trading_days_between(trading_index, snapshot_date, e)
            diff = abs(dte_actual - dte_target)
            if diff <= 2 and diff < best_dte_diff:
                best_dte_diff = diff
                best_exp = e
        if best_exp is None:
            skip_reasons["no_entry_match"] = skip_reasons.get("no_entry_match", 0) + 1
            continue
        exp_str = best_exp.strftime("%Y-%m-%d")
        T_entry = years_to_expiry(snapshot_date, best_exp)
        if T_entry <= 0 or T_entry > 0.1:
            skip_reasons["dte_out_of_range"] = skip_reasons.get("dte_out_of_range", 0) + 1
            continue

        exp_dates_ser = pd.to_datetime(df["expiration_date"], errors="coerce").dt.date
        put_mask = (exp_dates_ser == best_exp)
        ct_col = "contract_type" if "contract_type" in df.columns else "option_type"
        ct_ser = df[ct_col].astype(str).str.lower()
        put_mask &= (ct_ser == "put") | (ct_ser == "p")
        puts = df.loc[put_mask]
        if puts.empty:
            skip_reasons["no_puts_for_exp"] = skip_reasons.get("no_puts_for_exp", 0) + 1
            continue

        # --- BWB strike selection ---
        # K_body = ~20-delta put (short 2x), K_upper = body + upper_width, K_lower = body - lower_width
        best_put = None
        best_delta_diff = float("inf")
        for _, row in puts.iterrows():
            iv = _get_iv(row, default_iv)
            if iv is None or iv <= 0:
                continue
            sigma = _sigma_from_iv(float(iv))
            k = float(row["strike"])
            delta = bs_put_delta(spot_entry, k, T_entry, r, sigma)
            if np.isnan(delta):
                continue
            if abs(delta - (-delta_target)) < best_delta_diff:
                best_delta_diff = abs(delta - (-delta_target))
                best_put = row
        if best_put is None:
            skip_reasons["no_20d_put"] = skip_reasons.get("no_20d_put", 0) + 1
            continue
        K_body = float(best_put["strike"])
        K_upper = K_body + upper_width
        K_lower = K_body - lower_width
        if K_lower <= 0:
            skip_reasons["strike_below_zero"] = skip_reasons.get("strike_below_zero", 0) + 1
            continue

        # Verify all 3 strikes exist in the chain
        has_upper = not puts[(puts["strike"] == K_upper)].empty or not df.loc[
            (df["strike"] == K_upper) & ((ct_ser == "put") | (ct_ser == "p"))
        ].empty
        has_lower = not puts[(puts["strike"] == K_lower)].empty
        if not has_upper or not has_lower:
            skip_reasons["missing_bwb_leg"] = skip_reasons.get("missing_bwb_leg", 0) + 1
            if verbose:
                print(f"  {snapshot_date}: skip (missing leg: upper={K_upper} found={has_upper}, lower={K_lower} found={has_lower})", file=sys.stderr)
            continue

        # --- BWB entry pricing ---
        # Net credit = 2*mid(body) - mid(upper) - mid(lower)
        mid_body = get_mid_from_chain(df, K_body, best_exp, "put")
        mid_upper = get_mid_from_chain(df, K_upper, best_exp, "put")
        mid_lower = get_mid_from_chain(df, K_lower, best_exp, "put")
        iv1 = _get_iv(best_put, default_iv) or 0.20
        sigma1 = _sigma_from_iv(float(iv1))
        if mid_body is None:
            mid_body = bs_put_price(spot_entry, K_body, T_entry, r, sigma1)
        if mid_upper is None:
            mid_upper = bs_put_price(spot_entry, K_upper, T_entry, r, sigma1)
        if mid_lower is None:
            mid_lower = bs_put_price(spot_entry, K_lower, T_entry, r, sigma1)
        entry_credit = 2 * mid_body - mid_upper - mid_lower
        if entry_credit <= 0:
            skip_reasons["bad_credit"] = skip_reasons.get("bad_credit", 0) + 1
            if verbose:
                print(f"  {snapshot_date}: skip (BWB credit {entry_credit:.2f} <= 0, body={mid_body:.2f} upper={mid_upper:.2f} lower={mid_lower:.2f})", file=sys.stderr)
            continue
        if min_credit > 0 and entry_credit < min_credit:
            skip_reasons["credit_below_min"] = skip_reasons.get("credit_below_min", 0) + 1
            if verbose:
                print(f"  {snapshot_date}: skip (credit {entry_credit:.2f} < min {min_credit:.2f})", file=sys.stderr)
            continue

        # Enforce max concurrent open positions
        if max_concurrent is not None and max_concurrent > 0:
            open_count = sum(
                1 for t in trades
                if t.get("exit_date") and t["exit_date"] >= snapshot_date.isoformat()
            )
            if open_count >= max_concurrent:
                skip_reasons["max_concurrent"] = skip_reasons.get("max_concurrent", 0) + 1
                continue

        # Slippage: 4 legs at entry (buy upper, sell 2x body, buy lower)
        entry_slippage = 4 * slippage_per_leg
        entry_credit_adj = entry_credit - entry_slippage
        if entry_credit_adj <= 0:
            skip_reasons["credit_after_slippage"] = skip_reasons.get("credit_after_slippage", 0) + 1
            continue
        # BWB max loss = lower_width - upper_width - entry_credit_adj (capped at K_lower)
        bwb_max_loss = lower_width - upper_width - entry_credit_adj
        # BWB max profit = entry_credit_adj + upper_width (at K_body)
        bwb_max_profit = entry_credit_adj + upper_width
        close_threshold = (1 - profit_target_pct / 100.0) * entry_credit_adj if profit_target_pct and profit_target_pct < 100 else None
        stop_mult = stop_loss_mult
        stop_pct = stop_loss_pct_max

        # --- Daily exit check ---
        sessions = get_trading_days_in_range(trading_index, snapshot_date, best_exp)
        exit_date = best_exp
        exit_reason = "expiry"
        exit_debit = None
        for d in sessions:
            if d <= snapshot_date:
                continue
            if d >= best_exp:
                break
            # Try actual option chain data (4-leg BWB value)
            bwb_val = _get_bwb_mid_from_snapshot(data_dir, d, K_upper, K_body, K_lower, best_exp, parquet_warn)
            if bwb_val is None:
                spot_d = spot_on_trading_day(spot_series, spot_dates, d)
                if spot_d is None:
                    continue
                T_d = years_to_expiry(d, best_exp)
                if T_d <= 0:
                    break
                mid_b_d = bs_put_price(spot_d, K_body, T_d, r, sigma1)
                mid_u_d = bs_put_price(spot_d, K_upper, T_d, r, sigma1)
                mid_l_d = bs_put_price(spot_d, K_lower, T_d, r, sigma1)
                bwb_val = 2 * mid_b_d - mid_u_d - mid_l_d
            # bwb_val = cost to close; profit target: close when bwb_val is small enough
            if close_threshold is not None and bwb_val <= close_threshold:
                exit_date = d
                exit_reason = f"closed_{profit_target_pct:.0f}pct"
                exit_debit = bwb_val
                break
            if stop_mult is not None and bwb_val >= stop_mult * entry_credit:
                exit_date = d
                exit_reason = "stop_loss"
                exit_debit = bwb_val
                break
            if stop_pct is not None and bwb_max_loss > 0:
                loss_pct = (bwb_val - entry_credit_adj) / bwb_max_loss * 100
                if loss_pct >= stop_pct:
                    exit_date = d
                    exit_reason = "stop_loss_pct"
                    exit_debit = bwb_val
                    break

        # Commission: 4 legs each way = 8 legs total if closing early, 4 at entry if expiry
        if exit_reason == "expiry":
            total_commission = 4 * commission_per_leg
            spot_settle = spot_on_trading_day(spot_series, spot_dates, best_exp)
            if spot_settle is None:
                skip_reasons["no_settlement"] = skip_reasons.get("no_settlement", 0) + 1
                continue
            # BWB settlement: net payoff to us
            upper_payoff = max(0.0, K_upper - spot_settle)
            body_payoff = max(0.0, K_body - spot_settle)
            lower_payoff = max(0.0, K_lower - spot_settle)
            net_settlement = upper_payoff - 2 * body_payoff + lower_payoff
            pnl = (entry_credit_adj + net_settlement) * SPX_MULT - total_commission
        else:
            total_commission = 8 * commission_per_leg
            exit_slippage = 4 * slippage_per_leg
            exit_debit_adj = (exit_debit or 0) + exit_slippage
            pnl = (entry_credit_adj - exit_debit_adj) * SPX_MULT - total_commission

        trades.append({
            "entry_date": snapshot_date.isoformat(),
            "expiration_date": exp_str,
            "exit_date": exit_date.isoformat() if hasattr(exit_date, "isoformat") else str(exit_date),
            "K_upper": K_upper,
            "K_body": K_body,
            "K_lower": K_lower,
            "spot_entry": spot_entry,
            "entry_credit": round(entry_credit_adj, 4),
            "max_profit": round(bwb_max_profit, 4),
            "max_loss": round(bwb_max_loss, 4),
            "result": exit_reason,
            "pnl": pnl,
            "regime": regime,
        })

    gex_calculator.get_eastern_now = orig_now
    gex_calculator.get_eastern_today = orig_today
    if verbose and skip_reasons:
        print("Skip reasons:", skip_reasons, file=sys.stderr)
    if not trades:
        print("No trades. Run with --verbose to see skip reasons.")
        if skip_reasons.get("no_df"):
            print("Tip: pip install pyarrow", file=sys.stderr)
        return (0, [], trading_index)

    # Compute max concurrent positions and margin-at-risk
    max_open = 0
    for t in trades:
        open_count = sum(
            1 for t2 in trades
            if t2["entry_date"] <= t["entry_date"] and t2["exit_date"] >= t["entry_date"]
        )
        max_open = max(max_open, open_count)
    avg_max_loss = np.mean([t["max_loss"] for t in trades]) * SPX_MULT if trades else 0
    avg_max_profit = np.mean([t["max_profit"] for t in trades]) * SPX_MULT if trades else 0
    margin_per = avg_max_loss if avg_max_loss > 0 else lower_width * SPX_MULT
    max_margin = max_open * margin_per

    stats = compute_stats(trades, initial_capital)
    n_wins = sum(1 for t in trades if (t.get("pnl") or 0) > 0)
    n_losses = sum(1 for t in trades if (t.get("pnl") or 0) < 0)
    n_flat = sum(1 for t in trades if (t.get("pnl") or 0) == 0)
    avg_credit = np.mean([t["entry_credit"] for t in trades]) if trades else 0

    print(f"=== BWB Put Backtest (upper_width={upper_width}, lower_width={lower_width}) ===")
    print("Total trades:", stats.get("n_trades", len(trades)))
    print(f"Win/Loss/Flat: {n_wins}/{n_losses}/{n_flat}  ({n_wins/len(trades)*100:.0f}% win rate)")
    print("Expectancy (avg P&L per trade):", f"${stats.get('expectancy', 0):.2f}")
    print("Avg win:", f"${stats.get('avg_win', 0):.2f}")
    print("Avg loss:", f"${stats.get('avg_loss', 0):.2f}")
    print("Total P&L:", f"${stats.get('total_pnl', 0):.2f}")
    print("Max drawdown:", f"${stats.get('max_drawdown', 0):.2f}")
    print("Sharpe (ann.):", f"{stats.get('sharpe_ratio', 0):.2f}")
    print("Sortino (ann.):", f"{stats.get('sortino_ratio', 0):.2f}")
    print("CAGR:", f"{stats.get('cagr', 0):.1%}")
    print("Worst 1% trade:", f"${stats.get('worst_1pct_trade', 0):.2f}")
    print("Worst month P&L:", f"${stats.get('worst_month_pnl', 0):.2f}")
    print(f"Max concurrent open: {max_open} (margin at risk: ${max_margin:,.0f})")
    print(f"Avg credit: ${avg_credit:.2f}/share  Avg max profit/contract: ${avg_max_profit:,.0f}")
    print(f"Avg max loss/contract: ${avg_max_loss:,.0f}  Risk/reward: 1:{avg_max_profit/avg_max_loss:.1f}" if avg_max_loss > 0 else "")
    if n_wins > 0 and n_losses == 0 and avg_max_loss > 0:
        worst_concurrent = max_open * avg_max_loss
        wins_wiped = worst_concurrent / stats.get("expectancy", 1) if stats.get("expectancy") else 0
        print(f"Worst-case concurrent loss: ${worst_concurrent:,.0f}  (wipes {wins_wiped:.0f} avg wins)")
    if initial_capital and initial_capital > 0:
        pnl_pct = stats.get("total_pnl", 0) / initial_capital * 100
        worst_pct = avg_max_loss / initial_capital * 100 if avg_max_loss > 0 else 0
        print(f"Return on ${initial_capital:,.0f} capital: {pnl_pct:.2f}%  Worst single loss: {worst_pct:.2f}%")
    if stats.get("by_regime"):
        print("By regime:", stats["by_regime"])

    if output_csv:
        pd.DataFrame(trades).to_csv(output_csv, index=False)
        print(f"Trades written to {output_csv}")
    if output_json:
        with open(output_json, "w") as f:
            json.dump({"stats": stats, "n_trades": len(trades)}, f, indent=2)
        print(f"Stats written to {output_json}")
    return (0, trades, trading_index)


def main() -> int:
    parser = argparse.ArgumentParser(description="GEX 7DTE broken-wing butterfly (BWB) put backtest.")
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data" / "spx_options", help="Directory with spx_options Parquet files")
    parser.add_argument("--start", type=str, default=None, metavar="YYYY-MM-DD")
    parser.add_argument("--end", type=str, default=None, metavar="YYYY-MM-DD")
    parser.add_argument("--days", type=int, default=None, help="Use only last N calendar days of data")
    parser.add_argument("--delta-target", type=float, default=0.20, help="Target put delta for body strikes (e.g. 0.20)")
    parser.add_argument("--dte-target", type=int, default=7, help="DTE in trading sessions")
    parser.add_argument("--upper-width", type=float, default=25.0, help="BWB upper wing width: K_upper - K_body (default 25 SPX points)")
    parser.add_argument("--lower-width", type=float, default=50.0, help="BWB lower wing width: K_body - K_lower (default 50 SPX points, wider = more broken)")
    parser.add_argument("--positive-threshold", type=float, default=None)
    parser.add_argument("--strike-increment", type=float, default=None)
    parser.add_argument("--default-iv", type=float, default=None)
    parser.add_argument("--profit-target-pct", type=float, default=50.0, help="Close when BWB can be bought back for <= (1 - this/100) * credit (default 50%%)")
    parser.add_argument("--relax-symmetric", action="store_true")
    parser.add_argument("--slippage-per-leg", type=float, default=0.10, help="Slippage (bid-ask half-spread) per option leg in $ (default $0.10)")
    parser.add_argument("--commission-per-leg", type=float, default=0.65, help="Commission per option leg in $ (default $0.65, typical IBKR)")
    parser.add_argument("--min-credit", type=float, default=0.50, help="Minimum entry credit per share to accept trade (default $0.50)")
    parser.add_argument("--max-concurrent", type=int, default=1, help="Max simultaneous open positions (default 1)")
    parser.add_argument("--stop-loss-mult", type=float, default=None, help="Close when BWB debit >= this * entry credit")
    parser.add_argument("--stop-loss-pct-max", type=float, default=None, help="Close when loss >= this %% of max loss")
    parser.add_argument("--vix-max", type=float, default=None, help="Only enter when VIX <= this (optional)")
    parser.add_argument("--initial-capital", type=float, default=300_000.0, help="Account capital (default $300,000)")
    parser.add_argument("--ignore-regime", action="store_true", help="Ignore GEX regime (take every day); for testing pipeline")
    parser.add_argument("--output-csv", type=Path, default=None)
    parser.add_argument("--output-json", type=Path, default=None, help="Write stats summary JSON")
    parser.add_argument("--vectorbt", action="store_true", help="Report stats via VectorBT (standard backtesting library)")
    parser.add_argument("--verbose", "-v", action="store_true")
    args = parser.parse_args()
    start_date = datetime.strptime(args.start, "%Y-%m-%d").date() if args.start else None
    end_date = datetime.strptime(args.end, "%Y-%m-%d").date() if args.end else None
    if args.vectorbt and not _VBT_AVAILABLE:
        print("--vectorbt requires vectorbt. Install with: pip install vectorbt", file=sys.stderr)
        return 1
    exit_code, trades, trading_index = run_backtest(
        data_dir=args.data_dir,
        days_limit=args.days,
        start_date=start_date,
        end_date=end_date,
        delta_target=args.delta_target,
        dte_target=args.dte_target,
        upper_width=args.upper_width,
        lower_width=args.lower_width,
        positive_threshold=args.positive_threshold,
        strike_increment=args.strike_increment,
        output_csv=args.output_csv,
        output_json=args.output_json,
        verbose=args.verbose,
        default_iv=args.default_iv,
        profit_target_pct=args.profit_target_pct,
        relax_symmetric=args.relax_symmetric,
        slippage_per_leg=args.slippage_per_leg,
        commission_per_leg=args.commission_per_leg,
        stop_loss_mult=args.stop_loss_mult,
        stop_loss_pct_max=args.stop_loss_pct_max,
        vix_max=args.vix_max,
        initial_capital=args.initial_capital,
        ignore_regime=args.ignore_regime,
        min_credit=args.min_credit,
        max_concurrent=args.max_concurrent,
    )
    if exit_code != 0:
        return exit_code
    if args.vectorbt and trades and trading_index is not None:
        vbt_stats = build_vectorbt_portfolio(trades, trading_index, initial_capital=args.initial_capital)
        if vbt_stats is not None:
            print("\n--- VectorBT stats (standard backtesting library, returns-based) ---")
            print(vbt_stats.stats())
    return exit_code


if __name__ == "__main__":
    sys.exit(main())
