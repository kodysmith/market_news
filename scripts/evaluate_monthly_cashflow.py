#!/usr/bin/env python3
"""SPX Monthly Cash Flow Strategy Evaluator.

Backtests 6 strategies x 2 management styles (12 configs) over 2015-2025,
collects trade-level results, aggregates to monthly P&L, and produces
an interactive Plotly HTML report with equity curves, monthly bars,
heatmap, drawdown, win rate by year, and summary table.
"""
from __future__ import annotations

import sys
from datetime import date, datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.optimize_spx_verticals_historical import (
    load_spx_vix,
    friday_expirations,
    get_trading_days,
    simulate_one,
)

# 6 strategies x 2 management styles = 12 configs
# Management A: TP 50%, SL 2x credit
# Management B: TP 60%, SL 1.5x credit
STRATEGY_CONFIGS = [
    # (config_id, strategy, dte, short_delta, width, profit_target, stop_mult)
    ("IC_14_15_50_A", "iron_condor", 14, 0.15, 50.0, 0.50, 2.0),
    ("IC_14_15_50_B", "iron_condor", 14, 0.15, 50.0, 0.60, 1.5),
    ("IC_14_10_30_A", "iron_condor", 14, 0.10, 30.0, 0.50, 2.0),
    ("IC_14_10_30_B", "iron_condor", 14, 0.10, 30.0, 0.60, 1.5),
    ("IC_21_20_50_A", "iron_condor", 21, 0.20, 50.0, 0.50, 2.0),
    ("IC_21_20_50_B", "iron_condor", 21, 0.20, 50.0, 0.60, 1.5),
    ("IC_30_15_50_A", "iron_condor", 30, 0.15, 50.0, 0.50, 2.0),
    ("IC_30_15_50_B", "iron_condor", 30, 0.15, 50.0, 0.60, 1.5),
    ("PCS_14_15_50_A", "put_vertical", 14, 0.15, 50.0, 0.50, 2.0),
    ("PCS_14_15_50_B", "put_vertical", 14, 0.15, 50.0, 0.60, 1.5),
    ("PCS_21_20_30_A", "put_vertical", 21, 0.20, 30.0, 0.50, 2.0),
    ("PCS_21_20_30_B", "put_vertical", 21, 0.20, 30.0, 0.60, 1.5),
]

SLIPPAGE_PER_LEG = 0.10
COMMISSION_PER_LEG = 0.65
DEFAULT_START = date(2015, 1, 1)
DEFAULT_END = date(2025, 12, 31)


def run_backtest(
    start: date,
    end: date,
    spx_close: pd.Series,
    vix_close: pd.Series,
    trading: list[date],
    expirations: list[date],
) -> pd.DataFrame:
    """Run all 12 configs and return a trade-level DataFrame."""
    trading_set = set(trading)
    exp_list = [e for e in expirations if e in trading_set]
    rows: list[dict] = []

    for config_id, strategy, dte_target, short_delta, width, pt, sl in STRATEGY_CONFIGS:
        for exp in exp_list:
            try:
                exp_pos = trading.index(exp)
            except ValueError:
                continue
            if exp_pos < dte_target:
                continue
            entry_date = trading[exp_pos - dte_target]
            if entry_date not in spx_close.index or exp not in spx_close.index:
                continue

            iv_entry = 0.20
            if entry_date in vix_close.index:
                iv_entry = float(vix_close[entry_date])  # raw VIX level; simulate_one converts

            result = simulate_one(
                strategy=strategy,
                spx_close=spx_close,
                entry_date=entry_date,
                exp_date=exp,
                iv_entry=iv_entry,
                short_put_delta=short_delta,
                short_call_delta=short_delta,
                width=width,
                profit_target_pct=pt,
                stop_mult=sl,
                slippage_per_leg=SLIPPAGE_PER_LEG,
                commission_per_leg=COMMISSION_PER_LEG,
                trading_days=trading,
            )
            if result is None:
                continue
            rows.append({
                "config_id": config_id,
                "strategy": strategy,
                "dte": dte_target,
                "short_delta": short_delta,
                "width": width,
                "profit_target": pt,
                "stop_mult": sl,
                "entry_date": result["entry_date"],
                "exit_date": result["exit_date"],
                "exp_date": result["exp_date"],
                "pnl": result["pnl"],
                "win": result["win"],
                "exit_reason": result["exit_reason"],
            })

    return pd.DataFrame(rows)


def monthly_pnl_and_metrics(trades: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Aggregate trades to monthly P&L per config; compute equity, drawdown, and summary stats."""
    if trades.empty:
        empty_monthly = pd.DataFrame(columns=["config_id", "year_month", "pnl", "year", "month"])
        empty_equity = pd.DataFrame(columns=["config_id", "exit_date", "cum_pnl", "peak", "drawdown"])
        return empty_monthly, empty_equity, {}

    trades = trades.copy()
    trades["year_month"] = pd.to_datetime(trades["exit_date"]).dt.to_period("M").astype(str)
    trades["year"] = pd.to_datetime(trades["exit_date"]).dt.year
    trades["month"] = pd.to_datetime(trades["exit_date"]).dt.month

    monthly = trades.groupby(["config_id", "year_month"], as_index=False).agg(
        pnl=("pnl", "sum"),
        year=("year", "first"),
        month=("month", "first"),
    )

    # Equity curve and drawdown per config (by exit_date order)
    trades_sorted = trades.sort_values(["config_id", "exit_date"])
    equity_rows: list[dict] = []
    for config_id, grp in trades_sorted.groupby("config_id"):
        grp = grp.sort_values("exit_date")
        cum = grp["pnl"].cumsum()
        peak = cum.cummax()
        dd = peak - cum
        for i, (_, row) in enumerate(grp.iterrows()):
            equity_rows.append({
                "config_id": config_id,
                "exit_date": row["exit_date"],
                "cum_pnl": cum.iloc[i],
                "peak": peak.iloc[i],
                "drawdown": dd.iloc[i],
            })
    equity_df = pd.DataFrame(equity_rows)

    # Summary stats per config
    summary = {}
    for config_id in trades["config_id"].unique():
        pnls = trades.loc[trades["config_id"] == config_id, "pnl"].values
        if len(pnls) == 0:
            summary[config_id] = {
                "total_pnl": 0.0, "avg_monthly_pnl": 0.0, "worst_month": 0.0,
                "pct_profitable_months": 0.0, "sharpe_like": 0.0, "max_drawdown": 0.0,
                "n_trades": 0, "win_rate": 0.0, "profit_factor": 0.0,
            }
            continue
        monthly_cfg = monthly[monthly["config_id"] == config_id]
        monthly_pnls = monthly_cfg["pnl"].values
        n_months = len(monthly_pnls)
        profitable_months = (monthly_pnls > 0).sum()
        total_pnl = float(np.sum(pnls))
        avg_monthly = total_pnl / n_months if n_months else 0.0
        worst_month = float(np.min(monthly_pnls)) if n_months else 0.0
        std_pnl = float(np.std(pnls)) if len(pnls) > 1 else 0.0
        sharpe = (float(np.mean(pnls)) / std_pnl) if std_pnl > 0 else 0.0
        equity_cfg = equity_df[equity_df["config_id"] == config_id]
        max_dd = float(equity_cfg["drawdown"].max()) if not equity_cfg.empty else 0.0
        wins = pnls[pnls > 0]
        losses = pnls[pnls < 0]
        gross_win = float(np.sum(wins)) if len(wins) else 0.0
        gross_loss = abs(float(np.sum(losses))) if len(losses) else 1e-9
        pf = gross_win / gross_loss if gross_loss > 0 else 0.0
        summary[config_id] = {
            "total_pnl": total_pnl,
            "avg_monthly_pnl": avg_monthly,
            "worst_month": worst_month,
            "pct_profitable_months": 100.0 * profitable_months / n_months if n_months else 0.0,
            "sharpe_like": sharpe,
            "max_drawdown": max_dd,
            "n_trades": len(pnls),
            "win_rate": 100.0 * (pnls > 0).sum() / len(pnls),
            "profit_factor": pf,
        }

    return monthly, equity_df, summary


def build_report_html(
    trades: pd.DataFrame,
    monthly: pd.DataFrame,
    equity_df: pd.DataFrame,
    summary: dict,
    output_path: Path,
) -> None:
    """Generate a single Plotly HTML file with all charts and summary table."""
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    config_ids = sorted(trades["config_id"].unique()) if not trades.empty else []
    colors = [
        "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
        "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf",
        "#aec7e8", "#ffbb78",
    ]
    config_color = {c: colors[i % len(colors)] for i, c in enumerate(config_ids)}

    figs: list[go.Figure] = []

    # 1. Equity curves
    fig_equity = go.Figure()
    for config_id in config_ids:
        eq = equity_df[equity_df["config_id"] == config_id].sort_values("exit_date")
        if eq.empty:
            continue
        fig_equity.add_trace(go.Scatter(
            x=eq["exit_date"],
            y=eq["cum_pnl"],
            mode="lines",
            name=config_id,
            line=dict(color=config_color.get(config_id, "#333"), width=2),
        ))
    fig_equity.update_layout(
        title="Equity Curves (Cumulative P&L)",
        xaxis_title="Date",
        yaxis_title="Cumulative P&L ($)",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        height=450,
    )
    figs.append(fig_equity)

    # 2. Monthly P&L bar chart (grouped by strategy)
    if not monthly.empty:
        fig_bars = go.Figure()
        for config_id in config_ids:
            m = monthly[monthly["config_id"] == config_id].sort_values("year_month")
            if m.empty:
                continue
            color = ["#2ca02c" if x > 0 else "#d62728" for x in m["pnl"]]
            fig_bars.add_trace(go.Bar(
                x=m["year_month"],
                y=m["pnl"],
                name=config_id,
                marker_color=color,
            ))
        fig_bars.update_layout(
            barmode="group",
            title="Monthly P&L by Strategy",
            xaxis_title="Year-Month",
            yaxis_title="P&L ($)",
            height=450,
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        )
        figs.append(fig_bars)
    else:
        figs.append(go.Figure().add_annotation(text="No monthly data", x=0.5, y=0.5, showarrow=False))

    # 3. Monthly P&L heatmap (one subplot per config or single heatmap for first config as example)
    if not monthly.empty and config_ids:
        # One heatmap per config in a grid
        n_configs = len(config_ids)
        n_cols = 3
        n_rows = (n_configs + n_cols - 1) // n_cols
        fig_heat = make_subplots(
            rows=n_rows,
            cols=n_cols,
            subplot_titles=config_ids,
            vertical_spacing=0.08,
            horizontal_spacing=0.06,
        )
        for idx, config_id in enumerate(config_ids):
            m = monthly[monthly["config_id"] == config_id]
            if m.empty:
                continue
            row, col = idx // n_cols + 1, idx % n_cols + 1
            pivot = m.pivot_table(index="year", columns="month", values="pnl", aggfunc="sum")
            pivot = pivot.reindex(columns=range(1, 13), fill_value=np.nan)
            fig_heat.add_trace(
                go.Heatmap(
                    z=pivot.values,
                    x=pivot.columns,
                    y=pivot.index,
                    colorscale="RdYlGn",
                    zmid=0,
                    colorbar=dict(len=1 / n_configs, y=(1 - (row - 0.5) / n_rows)),
                ),
                row=row,
                col=col,
            )
        fig_heat.update_layout(
            title="Monthly P&L Heatmap (Year x Month) by Strategy",
            height=300 * n_rows,
        )
        figs.append(fig_heat)
    else:
        figs.append(go.Figure().add_annotation(text="No monthly data", x=0.5, y=0.5, showarrow=False))

    # 4. Drawdown chart
    fig_dd = go.Figure()
    for config_id in config_ids:
        eq = equity_df[equity_df["config_id"] == config_id].sort_values("exit_date")
        if eq.empty:
            continue
        fig_dd.add_trace(go.Scatter(
            x=eq["exit_date"],
            y=eq["drawdown"],
            mode="lines",
            name=config_id,
            line=dict(color=config_color.get(config_id, "#333"), width=2),
        ))
    fig_dd.update_layout(
        title="Drawdown (Peak-to-Trough)",
        xaxis_title="Date",
        yaxis_title="Drawdown ($)",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        height=450,
    )
    figs.append(fig_dd)

    # 5. Win rate by year
    if not trades.empty:
        trades_copy = trades.copy()
        trades_copy["year"] = pd.to_datetime(trades_copy["exit_date"]).dt.year
        wr_by_year = trades_copy.groupby(["config_id", "year"]).agg(
            wins=("win", "sum"),
            n=("win", "count"),
        ).reset_index()
        wr_by_year["win_rate_pct"] = 100.0 * wr_by_year["wins"] / wr_by_year["n"]

        fig_wr = go.Figure()
        for config_id in config_ids:
            w = wr_by_year[wr_by_year["config_id"] == config_id]
            if w.empty:
                continue
            fig_wr.add_trace(go.Bar(
                x=w["year"],
                y=w["win_rate_pct"],
                name=config_id,
            ))
        fig_wr.update_layout(
            barmode="group",
            title="Win Rate by Year (%)",
            xaxis_title="Year",
            yaxis_title="Win Rate (%)",
            height=450,
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        )
        figs.append(fig_wr)
    else:
        figs.append(go.Figure().add_annotation(text="No trade data", x=0.5, y=0.5, showarrow=False))

    # 6. Summary table
    if summary:
        table_configs = list(summary.keys())
        table_data = [
            [
                summary[c]["total_pnl"],
                summary[c]["avg_monthly_pnl"],
                summary[c]["worst_month"],
                summary[c]["pct_profitable_months"],
                summary[c]["sharpe_like"],
                summary[c]["max_drawdown"],
                summary[c]["n_trades"],
                summary[c]["win_rate"],
                summary[c]["profit_factor"],
            ]
            for c in table_configs
        ]
        fig_table = go.Figure(data=[go.Table(
            header=dict(
                values=["Config", "Total P&L", "Avg Monthly", "Worst Month", "% Prof Months",
                        "Sharpe-like", "Max DD", "Trades", "Win %", "Profit Factor"],
                fill_color="paleturquoise",
                align="left",
            ),
            cells=dict(
                values=[
                    table_configs,
                    [round(x[0], 2) for x in table_data],
                    [round(x[1], 2) for x in table_data],
                    [round(x[2], 2) for x in table_data],
                    [round(x[3], 1) for x in table_data],
                    [round(x[4], 4) for x in table_data],
                    [round(x[5], 2) for x in table_data],
                    [int(x[6]) for x in table_data],
                    [round(x[7], 1) for x in table_data],
                    [round(x[8], 2) for x in table_data],
                ],
                align="left",
            ),
        )])
        fig_table.update_layout(
            title="Summary Table",
            height=400,
        )
        figs.append(fig_table)
    else:
        figs.append(go.Figure().add_annotation(text="No summary data", x=0.5, y=0.5, showarrow=False))

    # Write single HTML with all figures
    with open(output_path, "w") as f:
        f.write("<!DOCTYPE html><html><head><meta charset='utf-8'><title>SPX Monthly Cash Flow Report</title></head><body>\n")
        for i, fig in enumerate(figs):
            f.write(fig.to_html(full_html=False, include_plotlyjs=("cdn" if i == 0 else False)))
        f.write("</body></html>\n")


def main() -> int:
    import argparse
    parser = argparse.ArgumentParser(description="SPX Monthly Cash Flow Strategy Evaluator")
    parser.add_argument("--start", type=str, default=DEFAULT_START.isoformat())
    parser.add_argument("--end", type=str, default=DEFAULT_END.isoformat())
    parser.add_argument("--trades-csv", type=Path, default=ROOT / "data" / "monthly_cashflow_trades.csv")
    parser.add_argument("--report-html", type=Path, default=ROOT / "data" / "monthly_cashflow_report.html")
    args = parser.parse_args()

    start = datetime.strptime(args.start, "%Y-%m-%d").date()
    end = datetime.strptime(args.end, "%Y-%m-%d").date()

    print("Loading SPX + VIX...", file=sys.stderr)
    spx_df, vix_df = load_spx_vix(start, end)
    if spx_df.empty:
        print("No SPX data.", file=sys.stderr)
        return 1

    spx_close = spx_df["Close"].copy()
    spx_close.index = pd.to_datetime(spx_close.index).tz_localize(None).normalize()
    spx_close.index = spx_close.index.date
    vix_close = vix_df["Close"].copy() if not vix_df.empty else pd.Series(dtype=float)
    if not vix_close.empty:
        vix_close.index = pd.to_datetime(vix_close.index).tz_localize(None).normalize()
        vix_close.index = vix_close.index.date

    trading = get_trading_days(start, end + timedelta(days=60))
    expirations = [e for e in friday_expirations(start, end) if e in set(trading)]
    print(f"Expirations: {len(expirations)}, Configs: {len(STRATEGY_CONFIGS)}", file=sys.stderr)

    trades = run_backtest(start, end, spx_close, vix_close, trading, expirations)
    print(f"Total trades: {len(trades)}", file=sys.stderr)

    args.trades_csv.parent.mkdir(parents=True, exist_ok=True)
    trades.to_csv(args.trades_csv, index=False)
    print(f"Wrote {args.trades_csv}", file=sys.stderr)

    monthly, equity_df, summary = monthly_pnl_and_metrics(trades)
    args.report_html.parent.mkdir(parents=True, exist_ok=True)
    build_report_html(trades, monthly, equity_df, summary, args.report_html)
    print(f"Wrote {args.report_html}", file=sys.stderr)

    return 0


if __name__ == "__main__":
    sys.exit(main())
