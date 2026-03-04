#!/usr/bin/env python3
"""Scan SPX option chains for high-quality BWB entries (theta collection).

Focus:
- Expirations in a DTE window (default 14-30 calendar days)
- Weekly and standard monthly OPEX expirations
- Broken-wing put butterfly candidates (net credit entries)
"""
from __future__ import annotations

import argparse
import math
import sys
from datetime import date, datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.backtest_gex_7dte import (  # noqa: E402
    _get_iv,
    _sigma_from_iv,
    bs_put_delta,
    bs_put_price,
    discover_snapshot_dates,
    get_mid_from_chain,
    load_options_for_date,
    load_spot_series,
    years_to_expiry,
)


def _is_standard_monthly_opex(exp: date) -> bool:
    return exp.weekday() == 4 and 15 <= exp.day <= 21


def _expiration_label(exp: date) -> str:
    return "monthly_opex" if _is_standard_monthly_opex(exp) else "weekly"


def _safe_float(val: object) -> float | None:
    if val is None:
        return None
    try:
        f = float(val)
    except (TypeError, ValueError):
        return None
    if math.isnan(f):
        return None
    return f


def _risk_neutral_pop(spot: float, be_down: float, T: float, sigma: float, r: float = 0.05) -> float:
    """Approximate probability underlying settles above lower break-even."""
    if spot <= 0 or be_down <= 0 or T <= 0 or sigma <= 0:
        return 0.0
    d2 = (math.log(spot / be_down) + (r - 0.5 * sigma * sigma) * T) / (sigma * math.sqrt(T))
    return float(norm.cdf(d2))


def _leg_oi(puts: pd.DataFrame, strike: float) -> int:
    rows = puts.loc[puts["strike"] == strike]
    if rows.empty:
        return 0
    oi_series = rows["open_interest"].apply(_safe_float).dropna()
    if oi_series.empty:
        return 0
    return int(max(v for v in oi_series if v is not None and v > 0) if any(v and v > 0 for v in oi_series) else 0)


def _row_mid(row: pd.Series) -> float | None:
    mid = _safe_float(row.get("mid"))
    if mid is not None:
        return mid
    bid = _safe_float(row.get("bid"))
    ask = _safe_float(row.get("ask"))
    if bid is None or ask is None:
        return None
    return (bid + ask) / 2.0


def _pick_leg_row(puts: pd.DataFrame, strike: float) -> pd.Series | None:
    rows = puts.loc[puts["strike"] == strike].copy()
    if rows.empty:
        return None
    # Prefer rows with both bid/ask and highest OI.
    rows["bid_f"] = rows["bid"].apply(_safe_float)
    rows["ask_f"] = rows["ask"].apply(_safe_float)
    rows["has_quote"] = rows["bid_f"].notna() & rows["ask_f"].notna()
    rows["oi_f"] = rows["open_interest"].apply(_safe_float).fillna(0.0)
    rows = rows.sort_values(by=["has_quote", "oi_f"], ascending=[False, False])
    return rows.iloc[0]


def _first_positive(v: float) -> float:
    return v if v > 1e-9 else 1e-9


def _build_candidate(
    puts: pd.DataFrame,
    snapshot_date: date,
    exp: date,
    spot: float,
    body_row: pd.Series,
    upper_width: float,
    lower_width: float,
    min_credit: float,
    min_leg_oi: int,
    fill_fraction: float,
    allow_theoretical: bool,
) -> dict | None:
    K_body = float(body_row["strike"])
    K_upper = K_body + upper_width
    K_lower = K_body - lower_width
    if K_lower <= 0:
        return None

    if puts.loc[puts["strike"] == K_upper].empty or puts.loc[puts["strike"] == K_lower].empty:
        return None

    oi_upper = _leg_oi(puts, K_upper)
    oi_body = _leg_oi(puts, K_body)
    oi_lower = _leg_oi(puts, K_lower)
    if min(oi_upper, oi_body, oi_lower) < min_leg_oi:
        return None

    upper_row = _pick_leg_row(puts, K_upper)
    body_sel_row = _pick_leg_row(puts, K_body)
    lower_row = _pick_leg_row(puts, K_lower)
    if upper_row is None or body_sel_row is None or lower_row is None:
        return None

    T = years_to_expiry(snapshot_date, exp)
    if T <= 0:
        return None

    iv = _get_iv(body_sel_row, None)
    if iv is None or iv <= 0:
        return None
    sigma = _sigma_from_iv(float(iv))

    bid_body = _safe_float(body_sel_row.get("bid"))
    bid_upper = _safe_float(upper_row.get("bid"))
    bid_lower = _safe_float(lower_row.get("bid"))
    ask_body = _safe_float(body_sel_row.get("ask"))
    ask_upper = _safe_float(upper_row.get("ask"))
    ask_lower = _safe_float(lower_row.get("ask"))
    mid_body = _row_mid(body_sel_row)
    mid_upper = _row_mid(upper_row)
    mid_lower = _row_mid(lower_row)

    natural_credit = None
    if bid_body is not None and ask_upper is not None and ask_lower is not None:
        natural_credit = 2 * bid_body - ask_upper - ask_lower
    combo_mid_credit = None
    if mid_body is not None and mid_upper is not None and mid_lower is not None:
        combo_mid_credit = 2 * mid_body - mid_upper - mid_lower

    if natural_credit is None and not allow_theoretical:
        return None

    if natural_credit is None and allow_theoretical:
        theo_body = bs_put_price(spot, K_body, T, 0.05, sigma)
        theo_upper = bs_put_price(spot, K_upper, T, 0.05, sigma)
        theo_lower = bs_put_price(spot, K_lower, T, 0.05, sigma)
        entry_credit = 2 * theo_body - theo_upper - theo_lower
        pricing_mode = "theoretical_fallback"
    else:
        lo = float(natural_credit)
        hi = float(combo_mid_credit) if combo_mid_credit is not None else lo
        if hi < lo:
            lo, hi = hi, lo
        entry_credit = lo + max(0.0, min(1.0, fill_fraction)) * (hi - lo)
        pricing_mode = "quoted"

    if entry_credit < min_credit:
        return None

    max_profit = entry_credit + upper_width
    max_loss = lower_width - upper_width - entry_credit
    if max_loss <= 0:
        return None

    # Approximate 1-day theta edge using static spot/vol decay.
    T_next = max(T - 1.0 / 365.0, 1.0 / 3650.0)
    value_next = (
        2 * bs_put_price(spot, K_body, T_next, 0.05, sigma)
        - bs_put_price(spot, K_upper, T_next, 0.05, sigma)
        - bs_put_price(spot, K_lower, T_next, 0.05, sigma)
    )
    theta_edge = max(entry_credit - value_next, 0.0)

    break_even_down = K_body - (entry_credit + upper_width)
    pop = _risk_neutral_pop(spot, break_even_down, T, sigma)

    # Blend quality dimensions into one scanner score.
    theta_yield = theta_edge / _first_positive(max_loss)
    rr_ratio = max_profit / _first_positive(max_loss)
    score = 100.0 * (0.45 * theta_yield + 0.25 * rr_ratio + 0.30 * pop)

    return {
        "snapshot_date": snapshot_date.isoformat(),
        "expiration": exp.isoformat(),
        "expiration_kind": _expiration_label(exp),
        "dte": (exp - snapshot_date).days,
        "spot": round(spot, 2),
        "K_upper": K_upper,
        "K_body": K_body,
        "K_lower": K_lower,
        "upper_width": upper_width,
        "lower_width": lower_width,
        "entry_credit": round(entry_credit, 3),
        "natural_credit": round(natural_credit, 3) if natural_credit is not None else None,
        "combo_mid_credit": round(combo_mid_credit, 3) if combo_mid_credit is not None else None,
        "max_profit": round(max_profit, 3),
        "max_loss": round(max_loss, 3),
        "risk_reward": round(rr_ratio, 3),
        "theta_edge_1d": round(theta_edge, 4),
        "theta_yield_1d": round(theta_yield, 5),
        "break_even_down": round(break_even_down, 2),
        "pop_above_be": round(pop, 4),
        "leg_oi_min": int(min(oi_upper, oi_body, oi_lower)),
        "pricing_mode": pricing_mode,
        "body_bid": bid_body,
        "body_ask": ask_body,
        "upper_bid": bid_upper,
        "upper_ask": ask_upper,
        "lower_bid": bid_lower,
        "lower_ask": ask_lower,
        "score": round(score, 3),
    }


def _pick_body_for_delta(puts: pd.DataFrame, spot: float, T: float, delta_target: float) -> pd.Series | None:
    best_row = None
    best_diff = float("inf")
    for _, row in puts.iterrows():
        iv = _get_iv(row, None)
        if iv is None or iv <= 0:
            continue
        sigma = _sigma_from_iv(float(iv))
        delta = bs_put_delta(spot, float(row["strike"]), T, 0.05, sigma)
        if np.isnan(delta):
            continue
        diff = abs(delta + delta_target)
        if diff < best_diff:
            best_diff = diff
            best_row = row
    return best_row


def run_scan(
    data_dir: Path,
    min_dte: int,
    max_dte: int,
    min_credit: float,
    min_leg_oi: int,
    delta_targets: list[float],
    upper_widths: list[float],
    lower_widths: list[float],
    fill_fraction: float,
    allow_theoretical: bool,
) -> pd.DataFrame:
    snapshot_dates = discover_snapshot_dates(data_dir, days_limit=None)
    if not snapshot_dates:
        raise RuntimeError("No SPX snapshot dates found. Run scripts/collect_spx_options.py first.")
    snapshot_date = max(snapshot_dates)

    parquet_warn = [False]
    df = load_options_for_date(data_dir, snapshot_date, parquet_warn)
    if df is None or df.empty:
        raise RuntimeError(f"No options chain found for snapshot date {snapshot_date.isoformat()}.")

    spot_series = load_spot_series(snapshot_date, snapshot_date + timedelta(days=2))
    spot = float(spot_series.iloc[-1]) if not spot_series.empty else None
    if spot is None or spot <= 0:
        raise RuntimeError("Could not fetch SPX spot (^GSPC). Try again when market data is available.")

    exp_values = pd.to_datetime(df["expiration_date"], errors="coerce").dt.date.dropna().unique()
    all_future_dtes = sorted((d - snapshot_date).days for d in exp_values if d > snapshot_date)
    expirations = sorted(
        d
        for d in exp_values
        if d > snapshot_date and min_dte <= (d - snapshot_date).days <= max_dte
    )
    if not expirations:
        if not all_future_dtes:
            raise RuntimeError("No future expirations found in this snapshot.")
        dte_preview = ", ".join(str(v) for v in all_future_dtes[:12])
        raise RuntimeError(
            f"No expirations found in DTE range {min_dte}-{max_dte}. "
            f"Available DTEs in snapshot: [{dte_preview}]"
        )

    ct_col = "contract_type" if "contract_type" in df.columns else "option_type"
    ct_ser = df[ct_col].astype(str).str.lower()
    exp_ser = pd.to_datetime(df["expiration_date"], errors="coerce").dt.date

    out: list[dict] = []
    for exp in expirations:
        put_mask = (exp_ser == exp) & ((ct_ser == "put") | (ct_ser == "p"))
        puts = df.loc[put_mask].copy()
        if puts.empty:
            continue

        T = years_to_expiry(snapshot_date, exp)
        if T <= 0:
            continue

        for delta_target in delta_targets:
            body_row = _pick_body_for_delta(puts, spot, T, delta_target)
            if body_row is None:
                continue
            for upper_width in upper_widths:
                for lower_width in lower_widths:
                    if lower_width <= upper_width:
                        continue
                    candidate = _build_candidate(
                        puts=puts,
                        snapshot_date=snapshot_date,
                        exp=exp,
                        spot=spot,
                        body_row=body_row,
                        upper_width=float(upper_width),
                        lower_width=float(lower_width),
                        min_credit=min_credit,
                        min_leg_oi=min_leg_oi,
                        fill_fraction=fill_fraction,
                        allow_theoretical=allow_theoretical,
                    )
                    if candidate is not None:
                        candidate["delta_target"] = delta_target
                        out.append(candidate)

    if not out:
        return pd.DataFrame()

    ranked = pd.DataFrame(out).sort_values(
        by=["score", "theta_edge_1d", "entry_credit", "pop_above_be"],
        ascending=[False, False, False, False],
    )
    return ranked.reset_index(drop=True)


def _parse_float_list(value: str) -> list[float]:
    return [float(x.strip()) for x in value.split(",") if x.strip()]


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Scan SPX 14-30 DTE chains for best broken-wing butterfly (theta collection) setup."
    )
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data" / "spx_options")
    parser.add_argument("--min-dte", type=int, default=14, help="Minimum calendar DTE (default 14)")
    parser.add_argument("--max-dte", type=int, default=30, help="Maximum calendar DTE (default 30)")
    parser.add_argument("--min-credit", type=float, default=0.40, help="Minimum net credit per share")
    parser.add_argument("--min-leg-oi", type=int, default=100, help="Minimum OI on each BWB leg")
    parser.add_argument("--delta-targets", type=str, default="0.12,0.15,0.20,0.25")
    parser.add_argument("--upper-widths", type=str, default="20,25,30")
    parser.add_argument("--lower-widths", type=str, default="40,50,60")
    parser.add_argument(
        "--fill-fraction",
        type=float,
        default=0.25,
        help="Where to target fills between natural(0.0) and combo-mid(1.0). Default 0.25",
    )
    parser.add_argument(
        "--allow-theoretical",
        action="store_true",
        help="Allow Black-Scholes fallback when bid/ask are missing (off by default for realism).",
    )
    parser.add_argument("--top", type=int, default=20, help="Top N setups to print")
    parser.add_argument("--output-csv", type=Path, default=None, help="Optional file to write full ranked results")
    args = parser.parse_args()

    try:
        ranked = run_scan(
            data_dir=args.data_dir.resolve(),
            min_dte=args.min_dte,
            max_dte=args.max_dte,
            min_credit=args.min_credit,
            min_leg_oi=args.min_leg_oi,
            delta_targets=_parse_float_list(args.delta_targets),
            upper_widths=_parse_float_list(args.upper_widths),
            lower_widths=_parse_float_list(args.lower_widths),
            fill_fraction=args.fill_fraction,
            allow_theoretical=args.allow_theoretical,
        )
    except Exception as e:
        print(f"Scanner error: {e}", file=sys.stderr)
        return 1

    if ranked.empty:
        print("No valid BWB setups found for the selected constraints.")
        return 0

    best = ranked.iloc[0]
    print("=== SPX BWB Scanner (Theta Collection) ===")
    print(
        "Best setup:",
        f"exp={best['expiration']} ({best['expiration_kind']})",
        f"DTE={int(best['dte'])}",
        f"strikes={best['K_upper']}/{best['K_body']}/{best['K_lower']}",
        f"credit={best['entry_credit']}",
        f"maxP={best['max_profit']}",
        f"maxL={best['max_loss']}",
        f"POP={best['pop_above_be']:.2%}",
        f"score={best['score']}",
    )

    cols = [
        "expiration",
        "expiration_kind",
        "dte",
        "K_upper",
        "K_body",
        "K_lower",
        "entry_credit",
        "natural_credit",
        "combo_mid_credit",
        "max_profit",
        "max_loss",
        "risk_reward",
        "pricing_mode",
        "theta_edge_1d",
        "pop_above_be",
        "score",
    ]
    print()
    print(ranked.loc[:, cols].head(args.top).to_string(index=False))

    if args.output_csv is not None:
        out_path = args.output_csv.resolve()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        ranked.to_csv(out_path, index=False)
        print(f"\nWrote {len(ranked)} setups to {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
