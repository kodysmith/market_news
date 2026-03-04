#!/usr/bin/env python3
"""BWB Trade Chart and Replay Simulator — Streamlit app with two tabs.

Tab 1: Static trade chart from backtest (SPX + trade markers + equity curve).
Tab 2: Day-by-day replay simulator with auto-play.
"""
from __future__ import annotations

import io
import sys
import time
from contextlib import redirect_stdout, redirect_stderr
from datetime import date, datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Backtest and calendar imports (after path setup)
from scripts.backtest_gex_7dte import (
    run_backtest,
    load_spot_series,
    load_options_for_date,
    get_mid_from_chain,
    _get_bwb_mid_from_snapshot,
    discover_snapshot_dates,
    get_trading_days_index,
    get_trading_days_in_range,
    trading_days_between,
    spot_on_trading_day,
    years_to_expiry,
    bs_put_delta,
    bs_put_price,
    df_to_snap,
    _get_iv,
    _sigma_from_iv,
    infer_strike_increment,
)
import QuantEngine.gex_calculator as gex_calculator

SPX_MULT = 100
r = 0.05


def _default_data_dir() -> Path:
    return ROOT / "data" / "spx_options"


# ---------- Tab 1: Trade Chart ----------
def _run_backtest_silent(data_dir: Path, start_date: date | None, end_date: date | None, **kwargs):
    """Run backtest and return (exit_code, trades, trading_index), suppressing stdout/stderr."""
    fout, ferr = io.StringIO(), io.StringIO()
    with redirect_stdout(fout), redirect_stderr(ferr):
        result = run_backtest(
            data_dir=data_dir,
            days_limit=None,
            start_date=start_date,
            end_date=end_date,
            output_csv=None,
            output_json=None,
            verbose=False,
            default_iv=kwargs.get("default_iv"),
            **{k: v for k, v in kwargs.items() if k != "default_iv"},
        )
    return result


def _load_spot_for_trades(trades: list, padding_days: int = 30) -> pd.Series:
    if not trades:
        return pd.Series(dtype=float)
    dates = []
    for t in trades:
        for key in ("entry_date", "exit_date"):
            s = t.get(key)
            if s:
                try:
                    d = datetime.strptime(s[:10], "%Y-%m-%d").date()
                    dates.append(d)
                except ValueError:
                    pass
    if not dates:
        return pd.Series(dtype=float)
    cal_start = min(dates) - timedelta(days=padding_days)
    cal_end = max(dates) + timedelta(days=padding_days)
    return load_spot_series(cal_start, cal_end)


def _build_trade_chart(spot_series: pd.Series, trades: list) -> go.Figure:
    """Plotly figure: SPX line, entry/exit markers, K_upper/K_body/K_lower bands, equity curve subplot."""
    if spot_series.empty:
        fig = go.Figure()
        fig.add_annotation(text="No SPX data", xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False)
        return fig

    spot_dates = spot_series.index
    spot_vals = spot_series.values

    n_plots = 2  # SPX + equity
    fig = make_subplots(
        rows=n_plots, cols=1, shared_xaxes=True, vertical_spacing=0.06,
        row_heights=[0.65, 0.35], subplot_titles=("SPX & BWB trades", "Cumulative P&L"),
    )

    # SPX line
    fig.add_trace(
        go.Scatter(x=spot_dates, y=spot_vals, name="SPX", mode="lines", line=dict(color="rgb(31,119,180)", width=2)),
        row=1, col=1,
    )

    # Trade bands and markers
    entry_dates, entry_spots, exit_dates, exit_spots = [], [], [], []
    for t in trades:
        try:
            ed = t.get("entry_date")
            exd = t.get("exit_date")
            spot_e = t.get("spot_entry")
            ku = t.get("K_upper")
            kb = t.get("K_body")
            kl = t.get("K_lower")
            if not all([ed, exd, spot_e is not None, ku is not None, kb is not None, kl is not None]):
                continue
            entry_dt = pd.Timestamp(ed)
            exit_dt = pd.Timestamp(exd)
            entry_dates.append(entry_dt)
            entry_spots.append(spot_e)
            exit_dates.append(exit_dt)
            # Exit spot: use spot_series if available, else approximate
            if exit_dt in spot_series.index:
                exit_spots.append(spot_series.loc[exit_dt])
            else:
                exit_spots.append(spot_e)  # fallback

            # Horizontal band for this trade (span entry to exit)
            fig.add_shape(
                type="rect", x0=entry_dt, x1=exit_dt, y0=kl, y1=ku,
                fillcolor="rgba(200,200,200,0.2)", line=dict(width=0), row=1, col=1,
            )
            fig.add_hline(y=kb, line_dash="dot", line_color="gray", opacity=0.5, row=1, col=1)
        except Exception:
            continue

    if entry_dates:
        fig.add_trace(
            go.Scatter(x=entry_dates, y=entry_spots, name="Entry", mode="markers",
                       marker=dict(symbol="triangle-up", size=12, color="green")),
            row=1, col=1,
        )
        fig.add_trace(
            go.Scatter(x=exit_dates, y=exit_spots, name="Exit", mode="markers",
                       marker=dict(symbol="triangle-down", size=12, color="red")),
            row=1, col=1,
        )

    # Cumulative P&L
    pnls = [t.get("pnl") for t in trades if t.get("pnl") is not None]
    if pnls:
        cum_pnl = np.cumsum(pnls)
        exit_dates_pnl = [t.get("exit_date") for t in trades if t.get("pnl") is not None]
        try:
            x_pnl = [pd.Timestamp(d) for d in exit_dates_pnl]
        except Exception:
            x_pnl = list(range(len(cum_pnl)))
        fig.add_trace(
            go.Scatter(x=x_pnl, y=cum_pnl, name="Cumulative P&L", mode="lines+markers",
                       line=dict(color="darkgreen", width=2), marker=dict(size=6)),
            row=2, col=1,
        )
        fig.add_hline(y=0, line_dash="dash", line_color="gray", row=2, col=1)
    else:
        fig.add_trace(
            go.Scatter(x=[], y=[], name="Cumulative P&L", mode="lines"),
            row=2, col=1,
        )

    fig.update_layout(height=600, showlegend=True, legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
    fig.update_yaxes(title_text="SPX", row=1, col=1)
    fig.update_yaxes(title_text="P&L ($)", row=2, col=1)
    return fig


def render_tab1():
    st.header("Trade Chart (backtest results)")
    data_dir = Path(st.sidebar.text_input("Data dir", value=str(_default_data_dir())))
    use_csv = st.sidebar.checkbox("Load from CSV instead of running backtest", value=False)
    csv_path = None
    if use_csv:
        csv_path = st.sidebar.text_input("Path to backtest CSV", value="")

    start_str = st.sidebar.text_input("Start date (YYYY-MM-DD)", value="2025-12-01")
    end_str = st.sidebar.text_input("End date (YYYY-MM-DD)", value="2025-12-31")
    delta_target = st.sidebar.number_input("Delta target (put)", value=0.20, min_value=0.05, max_value=0.50, step=0.01)
    dte_target = st.sidebar.number_input("DTE target", value=7, min_value=1, max_value=14)
    upper_width = st.sidebar.number_input("Upper width", value=25.0, min_value=5.0, max_value=100.0, step=5.0)
    lower_width = st.sidebar.number_input("Lower width", value=50.0, min_value=10.0, max_value=150.0, step=5.0)
    profit_target_pct = st.sidebar.number_input("Profit target %", value=50.0, min_value=10.0, max_value=90.0, step=5.0)
    min_credit = st.sidebar.number_input("Min credit", value=0.50, min_value=0.0, step=0.10)
    ignore_regime = st.sidebar.checkbox("Ignore GEX regime (take every day)", value=False)

    trades = []
    trading_index = None
    spot_series = pd.Series(dtype=float)

    if use_csv and csv_path and Path(csv_path).exists():
        try:
            df = pd.read_csv(csv_path)
            trades = df.to_dict("records")
            spot_series = _load_spot_for_trades(trades)
        except Exception as e:
            st.error(f"Failed to load CSV: {e}")
    else:
        if not use_csv or not csv_path:
            try:
                start_date = datetime.strptime(start_str, "%Y-%m-%d").date() if start_str else None
                end_date = datetime.strptime(end_str, "%Y-%m-%d").date() if end_str else None
            except ValueError:
                start_date = end_date = None
            if start_date and end_date:
                with st.spinner("Running backtest…"):
                    exit_code, trades, trading_index = _run_backtest_silent(
                        data_dir=data_dir,
                        start_date=start_date,
                        end_date=end_date,
                        delta_target=delta_target,
                        dte_target=dte_target,
                        upper_width=upper_width,
                        lower_width=lower_width,
                        positive_threshold=None,
                        strike_increment=None,
                        profit_target_pct=profit_target_pct,
                        relax_symmetric=False,
                        slippage_per_leg=0.10,
                        commission_per_leg=0.65,
                        stop_loss_mult=None,
                        stop_loss_pct_max=None,
                        vix_max=None,
                        initial_capital=300_000.0,
                        ignore_regime=ignore_regime,
                        min_credit=min_credit,
                        max_concurrent=1,
                    )
                if exit_code != 0:
                    st.warning("Backtest returned no data (check data dir and date range).")
                else:
                    spot_series = _load_spot_for_trades(trades)
            else:
                st.info("Set start and end date to run backtest.")

    if trades:
        st.sidebar.subheader("Trades")
        rows = []
        for t in trades:
            rows.append({
                "Entry": t.get("entry_date"),
                "Exit": t.get("exit_date"),
                "K_upper": t.get("K_upper"),
                "K_body": t.get("K_body"),
                "K_lower": t.get("K_lower"),
                "Credit": t.get("entry_credit"),
                "P&L": t.get("pnl"),
                "Result": t.get("result"),
            })
        st.sidebar.dataframe(pd.DataFrame(rows), use_container_width=True, height=300)

    fig = _build_trade_chart(spot_series, trades)
    st.plotly_chart(fig, use_container_width=True)


# ---------- Tab 2: Replay Simulator ----------
def _try_bwb_entry(
    data_dir: Path,
    snapshot_date: date,
    spot_entry: float,
    trading_index: pd.DatetimeIndex,
    trading_dates_set: set,
    spot_series: pd.Series,
    spot_dates: set,
    delta_target: float,
    dte_target: int,
    upper_width: float,
    lower_width: float,
    min_credit: float,
    default_iv: float | None,
    parquet_warn: list,
    relax_symmetric: bool = False,
    positive_threshold: float | None = None,
) -> dict | None:
    """Evaluate BWB entry for one day. Returns position dict or None. Reuses backtest logic."""
    df = load_options_for_date(data_dir, snapshot_date, parquet_warn)
    if df is None or df.empty:
        return None
    snap, _ = df_to_snap(df, default_iv=default_iv)
    if not snap.get("results"):
        return None
    inc = infer_strike_increment(df, spot_entry)
    snapshot_dt = datetime(snapshot_date.year, snapshot_date.month, snapshot_date.day, 16, 0, 0)
    try:
        tz = gex_calculator.get_eastern_timezone()
        snapshot_dt = snapshot_dt.replace(tzinfo=tz)
    except Exception:
        pass
    orig_now = gex_calculator.get_eastern_now
    orig_today = gex_calculator.get_eastern_today
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
    if positive_threshold is not None and total_gex <= positive_threshold:
        return None
    if regime != "positive" and not (relax_symmetric and regime == "unknown" and total_gex > 0):
        return None

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
        return None
    best_exp = None
    best_dte_diff = float("inf")
    for e in exp_dates:
        dte_actual = trading_days_between(trading_index, snapshot_date, e)
        diff = abs(dte_actual - dte_target)
        if diff <= 2 and diff < best_dte_diff:
            best_dte_diff = diff
            best_exp = e
    if best_exp is None:
        return None
    exp_str = best_exp.strftime("%Y-%m-%d")
    T_entry = years_to_expiry(snapshot_date, best_exp)
    if T_entry <= 0 or T_entry > 0.1:
        return None

    ct_col = "contract_type" if "contract_type" in df.columns else "option_type"
    ct_ser = df[ct_col].astype(str).str.lower()
    exp_dates_ser = pd.to_datetime(df["expiration_date"], errors="coerce").dt.date
    put_mask = (exp_dates_ser == best_exp) & ((ct_ser == "put") | (ct_ser == "p"))
    puts = df.loc[put_mask]
    if puts.empty:
        return None
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
        return None
    K_body = float(best_put["strike"])
    K_upper = K_body + upper_width
    K_lower = K_body - lower_width
    if K_lower <= 0:
        return None
    has_upper = not puts[(puts["strike"] == K_upper)].empty
    has_lower = not puts[(puts["strike"] == K_lower)].empty
    if not has_upper or not has_lower:
        return None

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
    if entry_credit <= 0 or (min_credit > 0 and entry_credit < min_credit):
        return None
    entry_slippage = 4 * 0.10
    entry_credit_adj = entry_credit - entry_slippage
    if entry_credit_adj <= 0:
        return None
    bwb_max_loss = lower_width - upper_width - entry_credit_adj
    bwb_max_profit = entry_credit_adj + upper_width
    return {
        "entry_date": snapshot_date.isoformat(),
        "expiration_date": exp_str,
        "K_upper": K_upper,
        "K_body": K_body,
        "K_lower": K_lower,
        "spot_entry": spot_entry,
        "entry_credit": round(entry_credit_adj, 4),
        "max_profit": round(bwb_max_profit, 4),
        "max_loss": round(bwb_max_loss, 4),
        "exp_date": best_exp,
        "sigma1": sigma1,
        "regime": regime,
    }


def _get_bwb_value_today(
    data_dir: Path,
    d: date,
    K_upper: float,
    K_body: float,
    K_lower: float,
    exp_date: date,
    spot_d: float | None,
    sigma1: float,
    parquet_warn: list,
) -> float | None:
    """Current BWB mark-to-market (cost to close)."""
    val = _get_bwb_mid_from_snapshot(data_dir, d, K_upper, K_body, K_lower, exp_date, parquet_warn)
    if val is not None:
        return val
    if spot_d is None:
        return None
    T_d = years_to_expiry(d, exp_date)
    if T_d <= 0:
        return None
    mid_b = bs_put_price(spot_d, K_body, T_d, r, sigma1)
    mid_u = bs_put_price(spot_d, K_upper, T_d, r, sigma1)
    mid_l = bs_put_price(spot_d, K_lower, T_d, r, sigma1)
    return 2 * mid_b - mid_u - mid_l


def render_tab2():
    st.header("Replay Simulator")
    data_dir = Path(st.sidebar.text_input("Data dir (Replay)", value=str(_default_data_dir()), key="replay_data_dir"))
    start_str = st.sidebar.text_input("Start (Replay)", value="2025-12-01", key="replay_start")
    end_str = st.sidebar.text_input("End (Replay)", value="2025-12-31", key="replay_end")
    speed = st.sidebar.slider("Speed (sec per day)", 0.5, 3.0, 1.0, 0.25)
    delta_target = st.sidebar.number_input("Delta target (Replay)", value=0.20, min_value=0.05, max_value=0.50, step=0.01, key="rdelta")
    dte_target = st.sidebar.number_input("DTE (Replay)", value=7, min_value=1, max_value=14, key="rdte")
    upper_width = st.sidebar.number_input("Upper width (Replay)", value=25.0, min_value=5.0, max_value=100.0, step=5.0, key="ruw")
    lower_width = st.sidebar.number_input("Lower width (Replay)", value=50.0, min_value=10.0, max_value=150.0, step=5.0, key="rlw")
    profit_target_pct = st.sidebar.number_input("Profit target % (Replay)", value=50.0, min_value=10.0, max_value=90.0, step=5.0, key="rpt")
    min_credit = st.sidebar.number_input("Min credit (Replay)", value=0.50, min_value=0.0, step=0.10, key="rmin")

    if "playing" not in st.session_state:
        st.session_state.playing = False
    if "replay_trades" not in st.session_state:
        st.session_state.replay_trades = []
    if "replay_position" not in st.session_state:
        st.session_state.replay_position = None
    if "replay_equity" not in st.session_state:
        st.session_state.replay_equity = []
    if "replay_spx_dates" not in st.session_state:
        st.session_state.replay_spx_dates = []
    if "replay_spx_vals" not in st.session_state:
        st.session_state.replay_spx_vals = []

    col1, col2, col3 = st.columns([1, 1, 2])
    with col1:
        if st.button("Play", key="play_btn"):
            st.session_state.playing = True
            st.session_state.replay_trades = []
            st.session_state.replay_position = None
            st.session_state.replay_equity = []
            st.session_state.replay_spx_dates = []
            st.session_state.replay_spx_vals = []
    with col2:
        if st.button("Pause", key="pause_btn"):
            st.session_state.playing = False

    try:
        start_date = datetime.strptime(start_str, "%Y-%m-%d").date()
        end_date = datetime.strptime(end_str, "%Y-%m-%d").date()
    except ValueError:
        st.error("Invalid start/end date (use YYYY-MM-DD).")
        return
    if start_date > end_date:
        st.error("Start must be before end.")
        return

    cal_end = end_date + timedelta(days=45)
    trading_index = get_trading_days_index(start_date, cal_end)
    trading_days = get_trading_days_in_range(trading_index, start_date, end_date)
    if not trading_days:
        st.warning("No trading days in range.")
        return

    spot_series = load_spot_series(start_date, cal_end)
    if spot_series.empty:
        st.warning("Could not load SPX (yfinance).")
        return
    reindex_dates = pd.DatetimeIndex([pd.Timestamp(d) for d in set(trading_days)])
    spot_series = spot_series.reindex(reindex_dates).dropna()
    spot_dates = set(pd.Timestamp(ts).date() for ts in spot_series.index)
    trading_dates_set = set(trading_days)
    snapshot_dates = set(discover_snapshot_dates(data_dir, None))
    parquet_warn = [False]

    placeholder = st.empty()
    for i, d in enumerate(trading_days):
        if not st.session_state.playing:
            break
        spot_today = spot_on_trading_day(spot_series, spot_dates, d)
        if spot_today is not None:
            st.session_state.replay_spx_dates.append(pd.Timestamp(d))
            st.session_state.replay_spx_vals.append(spot_today)

        pos = st.session_state.replay_position
        # --- Exit check ---
        if pos is not None:
            exp_date = pos["exp_date"]
            if d >= exp_date:
                # Settle at expiry
                spot_settle = spot_on_trading_day(spot_series, spot_dates, exp_date)
                if spot_settle is not None:
                    upper_payoff = max(0.0, pos["K_upper"] - spot_settle)
                    body_payoff = max(0.0, pos["K_body"] - spot_settle)
                    lower_payoff = max(0.0, pos["K_lower"] - spot_settle)
                    net_settlement = upper_payoff - 2 * body_payoff + lower_payoff
                    pnl = (pos["entry_credit"] + net_settlement) * SPX_MULT - 4 * 0.65
                else:
                    pnl = 0.0
                st.session_state.replay_trades.append({
                    "entry_date": pos["entry_date"],
                    "exit_date": exp_date.isoformat(),
                    "K_upper": pos["K_upper"], "K_body": pos["K_body"], "K_lower": pos["K_lower"],
                    "entry_credit": pos["entry_credit"], "pnl": pnl, "result": "expiry", "regime": pos.get("regime", ""),
                })
                st.session_state.replay_equity.append(pnl)
                st.session_state.replay_position = None
            else:
                bwb_val = _get_bwb_value_today(
                    data_dir, d, pos["K_upper"], pos["K_body"], pos["K_lower"],
                    exp_date, spot_today, pos["sigma1"], parquet_warn,
                )
                if bwb_val is not None:
                    close_threshold = (1 - profit_target_pct / 100.0) * pos["entry_credit"]
                    if bwb_val <= close_threshold:
                        exit_slippage = 4 * 0.10
                        exit_debit_adj = bwb_val + exit_slippage
                        pnl = (pos["entry_credit"] - exit_debit_adj) * SPX_MULT - 8 * 0.65
                        st.session_state.replay_trades.append({
                            "entry_date": pos["entry_date"],
                            "exit_date": d.isoformat(),
                            "K_upper": pos["K_upper"], "K_body": pos["K_body"], "K_lower": pos["K_lower"],
                            "entry_credit": pos["entry_credit"], "pnl": pnl, "result": f"closed_{profit_target_pct:.0f}pct", "regime": pos.get("regime", ""),
                        })
                        st.session_state.replay_equity.append(pnl)
                        st.session_state.replay_position = None

        # --- Entry (only if no position and this day has parquet and is in snapshot_dates) ---
        if st.session_state.replay_position is None and d in snapshot_dates and spot_today is not None:
            new_pos = _try_bwb_entry(
                data_dir, d, spot_today, trading_index, trading_dates_set,
                spot_series, spot_dates, delta_target, dte_target, upper_width, lower_width,
                min_credit, None, parquet_warn, relax_symmetric=False, positive_threshold=None,
            )
            if new_pos is not None:
                st.session_state.replay_position = new_pos

        with placeholder.container():
            # SPX chart so far
            if st.session_state.replay_spx_dates:
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=st.session_state.replay_spx_dates,
                    y=st.session_state.replay_spx_vals,
                    name="SPX", mode="lines", line=dict(color="rgb(31,119,180)", width=2),
                ))
                fig.update_layout(title=f"SPX (through {d})", height=350, xaxis_title="Date", yaxis_title="SPX")
                st.plotly_chart(fig, use_container_width=True)

            # Position card
            pos = st.session_state.replay_position
            if pos is not None:
                spot_today = spot_on_trading_day(spot_series, spot_dates, d)
                bwb_val = _get_bwb_value_today(
                    data_dir, d, pos["K_upper"], pos["K_body"], pos["K_lower"],
                    pos["exp_date"], spot_today, pos["sigma1"], parquet_warn,
                ) if spot_today is not None else None
                dte_left = trading_days_between(trading_index, d, pos["exp_date"]) if d < pos["exp_date"] else 0
                unrealized = (pos["entry_credit"] - (bwb_val or 0)) * SPX_MULT if bwb_val is not None else None
                st.subheader("Current position")
                st.write(f"**K_upper** {pos['K_upper']} | **K_body** {pos['K_body']} | **K_lower** {pos['K_lower']}")
                st.write(f"Entry credit: ${pos['entry_credit']:.2f} | DTE left: {dte_left}")
                if bwb_val is not None:
                    st.write(f"Current BWB value: ${bwb_val:.2f}")
                if unrealized is not None:
                    color = "green" if unrealized >= 0 else "red"
                    st.write(f"Unrealized P&L: <span style='color:{color}'>${unrealized:,.0f}</span>", unsafe_allow_html=True)
                st.write(f"GEX regime: {pos.get('regime', '—')}")
            else:
                st.info("No open position.")

            # Trade log
            if st.session_state.replay_trades:
                st.subheader("Closed trades")
                st.dataframe(pd.DataFrame(st.session_state.replay_trades), use_container_width=True, height=200)
            if st.session_state.replay_equity:
                cum = np.cumsum(st.session_state.replay_equity)
                st.write(f"**Cumulative P&L:** ${cum[-1]:,.0f}")
                st.line_chart(pd.Series(cum, index=range(len(cum))))

        time.sleep(speed)

    if not st.session_state.playing and not st.session_state.replay_spx_dates:
        st.info("Click Play to start replay.")


# ---------- Main ----------
def main():
    st.set_page_config(page_title="BWB Simulator", page_icon="📈", layout="wide")
    st.title("BWB Trade Chart & Replay Simulator")
    tab1, tab2 = st.tabs(["Trade Chart", "Replay Simulator"])
    with tab1:
        render_tab1()
    with tab2:
        render_tab2()


if __name__ == "__main__":
    main()
