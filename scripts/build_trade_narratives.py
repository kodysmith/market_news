#!/usr/bin/env python3
"""Build structured narratives for AI training from balanced trade data.

For each trade (winner or loser), generates a narrative that includes:
  1. What we KNEW at entry (signals we had)
  2. What HAPPENED during the trade
  3. What we were MISSING (signals we didn't have)
  4. The LESSON (pattern for AI to learn)

Also produces a gap analysis showing which missing data sources would have
prevented the most losses.
"""
from __future__ import annotations

import json
import sys
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

OUT_DIR = ROOT / "data" / "training"

# Known market events (curated from historical records)
# These represent the type of information a news reader would have caught
KNOWN_EVENTS = {
    # 2015
    "2015-08-17": "China yuan devaluation shock (Aug 11), fears of global slowdown spreading",
    "2015-08-21": "China yuan devaluation + global selloff accelerating, Black Monday eve",
    "2015-08-24": "Black Monday — Dow dropped 1,000+ pts at open, VIX spiked to 53",
    # 2018
    "2018-02-02": "Jobs report showed wage growth → inflation fears → 'Volmageddon' weekend ahead",
    "2018-02-05": "Volmageddon — XIV blown up, VIX 37→50 intraday, -4.1% SPX",
    "2018-10-17": "US-China trade war escalation, Fed hawkish, treasury yields spiking",
    "2018-10-22": "Continued trade war fears, Italy budget crisis, Saudi tensions",
    "2018-12-03": "Trade war 'ceasefire' unwinding, yield curve inverting (2s10s)",
    # 2019
    "2019-05-10": "US-China trade talks collapsed, Trump raised tariffs to 25%",
    "2019-05-13": "China retaliation tariffs, full trade war escalation",
    "2019-07-31": "Fed cut rates but hawkish guidance ('mid-cycle adjustment')",
    "2019-08-01": "Trump announced new tariffs on remaining $300B Chinese goods",
    "2019-08-02": "Markets digesting surprise tariff announcement, flight to safety",
    "2019-09-30": "US-China trade deal uncertainty ahead of Oct talks",
    # 2020
    "2020-02-19": "COVID-19 spreading beyond China (South Korea, Italy outbreaks emerging)",
    "2020-02-21": "COVID-19 Italy lockdowns began, pandemic fears rising sharply",
    "2020-02-24": "COVID crash begins — SPX -3.4%, Italy quarantine zones",
    "2020-02-28": "COVID panic week, fastest correction from ATH in history",
    "2020-03-06": "COVID spreading rapidly in US, conference cancellations, oil war brewing",
    "2020-03-09": "Oil crash (Saudi-Russia price war) + COVID = circuit breaker day",
    "2020-03-23": "COVID bottom day — Fed announced unlimited QE",
    "2020-04-03": "Massive unemployment claims (6.6M), markets choppy in COVID recovery",
    "2020-04-06": "COVID deaths plateau hopes, but economy in freefall",
    # 2021
    "2021-05-10": "Inflation fears building (used cars, lumber), crypto selloff starting",
    "2021-05-12": "CPI print 4.2% — highest since 2008, 'transitory' debate ignites",
    # 2022
    "2022-05-06": "Fed 50bp hike, recession fears, tech earnings weak",
    "2022-06-10": "CPI 8.6% (higher than expected), Fed 75bp hike expected",
    "2022-06-13": "Market pricing in aggressive Fed tightening, bear market confirmed",
    "2022-09-12": "Market rally ahead of CPI, hoping for peak inflation",
    "2022-09-13": "CPI 8.3% — hotter than expected, SPX -4.3%, worst day since June 2020",
    # 2023
    "2023-01-30": "Pre-FOMC positioning, tech earnings mixed (MSFT guidance weak)",
    "2023-03-06": "SVB Financial stress emerging (stock dropped 60%)",
    "2023-03-09": "SVB bank run, FDIC seized bank next day, banking crisis fears",
    "2023-09-18": "Pre-FOMC week, 'higher for longer' narrative, UAW strike began",
    "2023-10-30": "Bond market stress (10Y hit 5%), Middle East tensions (Israel-Hamas)",
    "2023-11-13": "Post-CPI rally continuation, market pricing in rate cuts",
    # 2024
    "2024-08-02": "Jobs report missed badly (114K vs 175K expected), recession fears spiked",
    "2024-08-05": "Japanese carry trade unwind, global selloff, VIX spiked to 65 intraday",
    "2024-10-30": "Pre-election uncertainty, mega-cap earnings (MSFT, META)",
    "2024-12-16": "Pre-FOMC, market expects hawkish hold, year-end rebalancing",
    "2024-12-18": "Fed cut 25bp but hawkish dot plot (only 2 cuts in 2025), SPX -3%",
    # 2025
    "2025-03-26": "Tariff uncertainty, trade policy fears",
    "2025-04-23": "Tariff escalation fears, global trade tensions",
    "2025-05-09": "US-China trade deal announcement surprise, gap up",
}


def classify_loss_pattern(trade: pd.Series) -> str:
    """Classify the loss into a pattern category."""
    spx_move = abs(trade["spx_move_pct"])
    vix_change = trade["vix_change"]
    vix_entry = trade["vix_at_entry"]

    if spx_move > 3.0 and vix_change > 5:
        return "SHOCK_EVENT"  # sudden crash, VIX spike
    elif spx_move > 2.5 and vix_entry < 18:
        return "LOW_VOL_SURPRISE"  # calm before storm
    elif spx_move > 2.0 and trade["exit_reason"] == "expiry":
        return "OVERNIGHT_GAP"  # gap through strikes, no stop triggered
    elif trade["exit_reason"] == "stop_loss":
        return "TREND_MOVE"  # sustained move hit stop
    elif vix_change > 8:
        return "VOL_EXPLOSION"  # VIX expansion crushed the position
    elif spx_move > 1.5:
        return "MODERATE_MOVE"  # moderate adverse move
    else:
        return "THETA_DECAY_FAIL"  # didn't decay fast enough or wrong direction


def classify_win_pattern(trade: pd.Series) -> str:
    """Classify the win into a pattern category."""
    vix_entry = trade["vix_at_entry"]
    exit_reason = trade["exit_reason"]

    if vix_entry > 35 and exit_reason == "expiry":
        return "HIGH_VOL_PREMIUM"  # fat premium in crisis, held to expiry
    elif exit_reason == "profit_target":
        return "CLEAN_THETA_DECAY"  # textbook decay, hit target early
    elif vix_entry > 25:
        return "ELEVATED_VOL_WIN"  # good premium, market cooperated
    elif abs(trade["spx_move_pct"]) < 0.5:
        return "RANGE_BOUND"  # market went nowhere, perfect for IC
    else:
        return "FAVORABLE_MOVE"  # moved but stayed in range


def build_narrative(trade: pd.Series) -> dict:
    """Build a structured narrative for one trade."""
    is_win = trade["label"] == "WIN"
    entry_str = str(trade["entry_date"])
    exit_str = str(trade["exit_date"])

    # Pattern classification
    if is_win:
        pattern = classify_win_pattern(trade)
    else:
        pattern = classify_loss_pattern(trade)

    # What we knew at entry
    known_signals = []
    known_signals.append(f"VIX at {trade['vix_at_entry']:.1f}")
    if trade["spx_above_sma50"] == 1:
        known_signals.append("SPX above SMA50 (short-term uptrend)")
    else:
        known_signals.append("SPX below SMA50 (short-term downtrend)")
    if pd.notna(trade.get("spx_above_sma200")):
        if trade["spx_above_sma200"] == 1:
            known_signals.append("SPX above SMA200 (long-term uptrend)")
        else:
            known_signals.append("SPX below SMA200 (long-term downtrend)")
    known_signals.append(f"Regime: {trade['regime_tier']} (score {trade['regime_score']})")
    known_signals.append(f"Put cushion: {trade['put_distance_pct']:.2f}% OTM")
    if trade.get("call_distance_pct"):
        known_signals.append(f"Call cushion: {trade['call_distance_pct']:.2f}% OTM")

    # What happened
    what_happened = []
    what_happened.append(f"SPX moved {trade['spx_move_pct']:+.2f}% over {trade['hold_days']} days")
    what_happened.append(f"VIX changed {trade['vix_change']:+.1f} ({trade['vix_at_entry']:.1f} → {trade['vix_at_exit']:.1f})")
    what_happened.append(f"Exit: {trade['exit_reason']}")
    what_happened.append(f"P&L: ${trade['pnl']:+,.0f} ({trade['pnl_pct_of_credit']:+.0f}% of credit)")
    if trade["max_adverse_excursion"] < 0:
        what_happened.append(f"Max adverse excursion: ${trade['max_adverse_excursion']:,.0f}")

    # News context (what a news reader would have known)
    news_context = KNOWN_EVENTS.get(entry_str, None)
    # Also check day before (news from previous day affects next day)
    entry_dt = datetime.strptime(entry_str, "%Y-%m-%d")
    prev_day = (entry_dt - timedelta(days=1)).strftime("%Y-%m-%d")
    next_day = (entry_dt + timedelta(days=1)).strftime("%Y-%m-%d")
    prev_news = KNOWN_EVENTS.get(prev_day, None)
    next_news = KNOWN_EVENTS.get(next_day, None)

    # What was missing (gap analysis)
    missing_signals = []

    # 1. News/sentiment
    if news_context or prev_news:
        missing_signals.append({
            "signal": "NEWS_SENTIMENT",
            "detail": f"News headline awareness: {news_context or prev_news}",
            "would_have_helped": not is_win,
            "action": "SKIP or REDUCE_SIZE" if not is_win else "CONFIRMED_ENTRY",
        })
    else:
        missing_signals.append({
            "signal": "NEWS_SENTIMENT",
            "detail": "No major headline identified — but a real-time news feed might have caught sector/macro signals",
            "would_have_helped": "unknown",
            "action": "MONITOR",
        })

    # 2. Economic calendar
    is_fomc_week = any(k in (news_context or "") for k in ["FOMC", "Fed"]) if news_context else False
    is_cpi = any(k in (news_context or "") for k in ["CPI", "inflation"]) if news_context else False
    is_jobs = any(k in (news_context or "") for k in ["Jobs", "unemployment", "jobs"]) if news_context else False
    if is_fomc_week or is_cpi or is_jobs:
        event_type = "FOMC" if is_fomc_week else ("CPI" if is_cpi else "JOBS")
        missing_signals.append({
            "signal": "ECONOMIC_CALENDAR",
            "detail": f"High-impact event: {event_type} — should have been flagged",
            "would_have_helped": not is_win,
            "action": "SKIP_OR_WIDEN_STRIKES",
        })

    # 3. Overnight gap risk
    if not is_win and trade["exit_reason"] == "expiry" and abs(trade["spx_move_pct"]) > 2:
        missing_signals.append({
            "signal": "OVERNIGHT_GAP_PROTECTION",
            "detail": f"SPX gapped {trade['spx_move_pct']:+.2f}% — stop loss couldn't trigger overnight",
            "would_have_helped": True,
            "action": "USE_HARD_STOP_ON_UNDERLYING or CLOSE_BEFORE_OVERNIGHT",
        })

    # 4. VIX term structure / regime speed
    if not is_win and trade["vix_at_entry"] < 18 and trade["vix_change"] > 5:
        missing_signals.append({
            "signal": "VOL_REGIME_SPEED",
            "detail": f"VIX was low ({trade['vix_at_entry']:.1f}) but exploded {trade['vix_change']:+.1f} — regime engine too slow to react",
            "would_have_helped": True,
            "action": "FASTER_REGIME_DOWNGRADE or INTRADAY_VIX_MONITORING",
        })

    # 5. Cross-asset signals
    if not is_win and trade["vix_change"] > 8:
        missing_signals.append({
            "signal": "CROSS_ASSET_STRESS",
            "detail": "Large VIX spike suggests credit/rates/FX stress — not captured by current signals",
            "would_have_helped": True,
            "action": "MONITOR_TLT_HYG_DXY for early warning",
        })

    # 6. Intraday data
    if not is_win and trade["max_adverse_excursion"] == 0 and abs(trade["spx_move_pct"]) > 2:
        missing_signals.append({
            "signal": "INTRADAY_PRICE_ACTION",
            "detail": "No intraday tracking — position went from entry to expiry with no intermediate monitoring",
            "would_have_helped": True,
            "action": "INTRADAY_STOP_MONITORING or CLOSING_BEFORE_EXPIRY",
        })

    # 7. Options flow / dealer positioning
    if not is_win:
        missing_signals.append({
            "signal": "OPTIONS_FLOW",
            "detail": "Large put buying or call selling ahead of the move would be visible in options flow data",
            "would_have_helped": "likely",
            "action": "MONITOR_PUT_CALL_RATIO and UNUSUAL_OPTIONS_ACTIVITY",
        })

    # 8. Earnings proximity
    # (we don't track this in backtest but it matters)
    missing_signals.append({
        "signal": "EARNINGS_CALENDAR",
        "detail": "Proximity to major earnings releases (AAPL/MSFT/NVDA/AMZN) affects index moves",
        "would_have_helped": "unknown",
        "action": "CHECK_MEGA_CAP_EARNINGS_WITHIN_DTE",
    })

    # Build the lesson
    if is_win:
        lesson = _build_win_lesson(trade, pattern, news_context)
    else:
        lesson = _build_loss_lesson(trade, pattern, news_context, missing_signals)

    return {
        "trade_id": f"{trade['config']}_{entry_str}",
        "label": trade["label"],
        "pattern": pattern,
        "entry_date": entry_str,
        "exit_date": exit_str,
        "config": trade["config"],
        "pnl": float(trade["pnl"]),
        "pnl_pct_of_credit": float(trade["pnl_pct_of_credit"]),

        # Structured sections
        "known_signals_at_entry": known_signals,
        "what_happened": what_happened,
        "news_context": {
            "entry_day": news_context,
            "prev_day": prev_news,
            "next_day": next_news,
        },
        "missing_signals": missing_signals,
        "lesson": lesson,

        # Raw features for ML
        "features": {
            "vix_at_entry": float(trade["vix_at_entry"]),
            "vix_change": float(trade["vix_change"]),
            "spx_move_pct": float(trade["spx_move_pct"]),
            "regime_tier": trade["regime_tier"],
            "regime_score": int(trade["regime_score"]),
            "put_distance_pct": float(trade["put_distance_pct"]) if pd.notna(trade.get("put_distance_pct")) else None,
            "call_distance_pct": float(trade["call_distance_pct"]) if pd.notna(trade.get("call_distance_pct")) else None,
            "spx_above_sma50": int(trade["spx_above_sma50"]) if pd.notna(trade.get("spx_above_sma50")) else None,
            "spx_above_sma200": int(trade["spx_above_sma200"]) if pd.notna(trade.get("spx_above_sma200")) else None,
            "entry_dow": int(trade["entry_dow"]),
            "entry_month": int(trade["entry_month"]),
            "hold_days": int(trade["hold_days"]),
            "exit_reason": trade["exit_reason"],
            "credit": float(trade["credit"]),
            "max_loss": float(trade["max_loss"]),
        },
    }


def _build_loss_lesson(trade, pattern, news_context, missing_signals):
    """Generate a lesson narrative for a losing trade."""
    lines = []

    if pattern == "SHOCK_EVENT":
        lines.append("PATTERN: Shock event — sudden market crash with VIX explosion.")
        lines.append(f"The market moved {trade['spx_move_pct']:+.2f}% while VIX spiked {trade['vix_change']:+.1f} points.")
        if news_context:
            lines.append(f"CATALYST: {news_context}")
        lines.append("LESSON: A news-aware system would have flagged elevated risk BEFORE entry.")
        lines.append("ACTION: Skip entries when breaking macro news is negative, or at minimum halve position size.")

    elif pattern == "LOW_VOL_SURPRISE":
        lines.append("PATTERN: Low-vol surprise — VIX was calm, giving false sense of safety.")
        lines.append(f"VIX was only {trade['vix_at_entry']:.1f} at entry but the market dropped {trade['spx_move_pct']:+.2f}%.")
        lines.append("LESSON: Low VIX doesn't mean low risk. It means options are CHEAP, so the cushion is thin.")
        lines.append("ACTION: When VIX < 15, either widen strikes or reduce size. The premium isn't worth the tail risk.")

    elif pattern == "OVERNIGHT_GAP":
        lines.append("PATTERN: Overnight gap — SPX gapped through strikes before stops could trigger.")
        lines.append(f"1DTE position held overnight, SPX gapped {trade['spx_move_pct']:+.2f}%.")
        lines.append("LESSON: 1DTE iron condors are exposed to overnight risk with no stop protection.")
        lines.append("ACTION: Close 1DTE positions before market close if news risk is elevated, or use 0DTE instead.")

    elif pattern == "TREND_MOVE":
        lines.append("PATTERN: Sustained directional move — stop loss triggered but damage was done.")
        lines.append(f"The 7DTE position hit its stop after SPX moved {trade['spx_move_pct']:+.2f}%.")
        lines.append("LESSON: Longer DTE gives more time for adverse moves to develop. Stops help but don't eliminate loss.")
        lines.append("ACTION: Tighter stops in uncertain regimes, or skip longer DTE when momentum is against you.")

    elif pattern == "VOL_EXPLOSION":
        lines.append("PATTERN: Volatility explosion — VIX spiked, inflating the cost to close.")
        lines.append(f"VIX jumped {trade['vix_change']:+.1f} points ({trade['vix_at_entry']:.1f} → {trade['vix_at_exit']:.1f}).")
        lines.append("LESSON: Short premium positions are short vega. A VIX spike alone can cause large losses even without much SPX movement.")
        lines.append("ACTION: Monitor VIX term structure and VVIX for early signs of vol expansion.")

    elif pattern == "MODERATE_MOVE":
        lines.append("PATTERN: Moderate adverse move — not a crash, but enough to breach strikes.")
        lines.append(f"SPX moved {trade['spx_move_pct']:+.2f}% with only {trade['put_distance_pct']:.1f}% put cushion.")
        lines.append("LESSON: Strike distance was too tight for the realized volatility.")
        lines.append("ACTION: Use wider delta (0.08 instead of 0.10-0.12) when regime is CAUTION or when VIX is trending up.")

    else:
        lines.append(f"PATTERN: {pattern}")
        lines.append(f"SPX moved {trade['spx_move_pct']:+.2f}%, VIX changed {trade['vix_change']:+.1f}.")

    # What would have saved this trade
    actionable = [s for s in missing_signals if s.get("would_have_helped") is True]
    if actionable:
        lines.append("")
        lines.append("SIGNALS THAT WOULD HAVE PREVENTED THIS LOSS:")
        for sig in actionable:
            lines.append(f"  - {sig['signal']}: {sig['detail']}")
            lines.append(f"    → {sig['action']}")

    return "\n".join(lines)


def _build_win_lesson(trade, pattern, news_context):
    """Generate a lesson narrative for a winning trade."""
    lines = []

    if pattern == "HIGH_VOL_PREMIUM":
        lines.append("PATTERN: High-vol premium collection — entered during crisis with massive premium.")
        lines.append(f"VIX was {trade['vix_at_entry']:.1f}, providing huge cushion ({trade['put_distance_pct']:.1f}% OTM).")
        lines.append("LESSON: Crisis environments offer the BEST risk/reward for short premium IF you size correctly.")
        lines.append("ACTION: When VIX > 40, consider small positions with very wide strikes. The premium pays for the risk.")

    elif pattern == "CLEAN_THETA_DECAY":
        lines.append("PATTERN: Clean theta decay — hit profit target before expiration.")
        lines.append(f"Closed at profit target, SPX only moved {trade['spx_move_pct']:+.2f}%.")
        lines.append("LESSON: Taking profits early (30-50% of max) is the highest-EV exit strategy.")
        lines.append("ACTION: Continue using profit targets. This is the bread-and-butter of consistent income.")

    elif pattern == "ELEVATED_VOL_WIN":
        lines.append("PATTERN: Elevated vol win — good premium in stressed market, held successfully.")
        lines.append(f"VIX at {trade['vix_at_entry']:.1f} gave better premium and wider strikes.")
        lines.append("LESSON: Moderately elevated VIX (20-30) is the sweet spot for premium selling.")

    elif pattern == "RANGE_BOUND":
        lines.append("PATTERN: Range-bound market — SPX barely moved, perfect for iron condors.")
        lines.append(f"SPX moved only {trade['spx_move_pct']:+.2f}%, both sides decayed to profit.")
        lines.append("LESSON: Iron condors thrive in range-bound, low-momentum environments.")

    else:
        lines.append(f"PATTERN: {pattern} — market cooperated with the position.")

    if news_context:
        lines.append(f"CONTEXT: {news_context}")
        lines.append("NOTE: This won DESPITE news — meaning the position was sized/placed correctly for the environment.")

    return "\n".join(lines)


def gap_analysis(narratives: list[dict]) -> dict:
    """Analyze which missing signals would have prevented the most losses."""
    signal_counts = {}
    signal_savings = {}

    for n in narratives:
        if n["label"] != "LOSS":
            continue
        for sig in n["missing_signals"]:
            name = sig["signal"]
            if sig.get("would_have_helped") is True:
                signal_counts[name] = signal_counts.get(name, 0) + 1
                signal_savings[name] = signal_savings.get(name, 0) + abs(n["pnl"])

    # Sort by total savings
    ranked = sorted(signal_savings.items(), key=lambda x: -x[1])

    return {
        "ranked_by_savings": [
            {
                "signal": name,
                "trades_prevented": signal_counts.get(name, 0),
                "total_savings": round(savings, 2),
                "avg_savings_per_trade": round(savings / max(signal_counts.get(name, 0), 1), 2),
            }
            for name, savings in ranked
        ],
    }


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default=str(OUT_DIR / "balanced_winners_losers.csv"))
    parser.add_argument("--top-n", type=int, default=None,
                        help="Only narrate top N winners + bottom N losers (default: all)")
    args = parser.parse_args()

    df = pd.read_csv(args.input)
    print(f"Loaded {len(df)} trades ({(df['label']=='WIN').sum()}W / {(df['label']=='LOSS').sum()}L)",
          file=sys.stderr)

    if args.top_n:
        wins = df[df["label"] == "WIN"].nlargest(args.top_n, "pnl")
        losses = df[df["label"] == "LOSS"].nsmallest(args.top_n, "pnl")
        df = pd.concat([wins, losses]).sort_values("entry_date")
        print(f"  Filtered to top {args.top_n} each side = {len(df)} trades", file=sys.stderr)

    # Build narratives
    narratives = []
    for _, trade in df.iterrows():
        narratives.append(build_narrative(trade))

    # Gap analysis
    gaps = gap_analysis(narratives)

    # Save narratives as JSONL (one per line, easy for LLM training)
    jsonl_path = OUT_DIR / "trade_narratives.jsonl"
    with open(jsonl_path, "w") as f:
        for n in narratives:
            f.write(json.dumps(n, default=str) + "\n")
    print(f"\nNarratives: {jsonl_path} ({len(narratives)} trades)", file=sys.stderr)

    # Save gap analysis
    gap_path = OUT_DIR / "gap_analysis.json"
    with open(gap_path, "w") as f:
        json.dump(gaps, f, indent=2)
    print(f"Gap analysis: {gap_path}", file=sys.stderr)

    # Print gap analysis summary
    print(f"\n{'='*80}", file=sys.stderr)
    print("MISSING SIGNAL GAP ANALYSIS", file=sys.stderr)
    print("Which signals would have prevented the most losses?", file=sys.stderr)
    print(f"{'='*80}", file=sys.stderr)
    print(f"{'Signal':<30} {'Trades Saved':>13} {'Total $':>12} {'Avg $/Trade':>12}",
          file=sys.stderr)
    print("-" * 70, file=sys.stderr)
    for item in gaps["ranked_by_savings"]:
        print(f"  {item['signal']:<28} {item['trades_prevented']:>10} "
              f"${item['total_savings']:>10,.0f} ${item['avg_savings_per_trade']:>10,.0f}",
              file=sys.stderr)

    # Print a few example narratives
    print(f"\n{'='*80}", file=sys.stderr)
    print("EXAMPLE LOSS NARRATIVE", file=sys.stderr)
    print(f"{'='*80}", file=sys.stderr)
    worst = [n for n in narratives if n["label"] == "LOSS"]
    worst.sort(key=lambda x: x["pnl"])
    if worst:
        ex = worst[0]
        print(f"\nTrade: {ex['trade_id']}", file=sys.stderr)
        print(f"P&L: ${ex['pnl']:+,.0f}  Pattern: {ex['pattern']}", file=sys.stderr)
        print(f"\nKNOWN AT ENTRY:", file=sys.stderr)
        for s in ex["known_signals_at_entry"]:
            print(f"  • {s}", file=sys.stderr)
        print(f"\nWHAT HAPPENED:", file=sys.stderr)
        for s in ex["what_happened"]:
            print(f"  • {s}", file=sys.stderr)
        if ex["news_context"]["entry_day"]:
            print(f"\nNEWS: {ex['news_context']['entry_day']}", file=sys.stderr)
        print(f"\nLESSON:\n{ex['lesson']}", file=sys.stderr)

    print(f"\n{'='*80}", file=sys.stderr)
    print("EXAMPLE WIN NARRATIVE", file=sys.stderr)
    print(f"{'='*80}", file=sys.stderr)
    best = [n for n in narratives if n["label"] == "WIN"]
    best.sort(key=lambda x: -x["pnl"])
    if best:
        ex = best[0]
        print(f"\nTrade: {ex['trade_id']}", file=sys.stderr)
        print(f"P&L: ${ex['pnl']:+,.0f}  Pattern: {ex['pattern']}", file=sys.stderr)
        print(f"\nKNOWN AT ENTRY:", file=sys.stderr)
        for s in ex["known_signals_at_entry"]:
            print(f"  • {s}", file=sys.stderr)
        print(f"\nWHAT HAPPENED:", file=sys.stderr)
        for s in ex["what_happened"]:
            print(f"  • {s}", file=sys.stderr)
        if ex["news_context"]["entry_day"]:
            print(f"\nNEWS: {ex['news_context']['entry_day']}", file=sys.stderr)
        print(f"\nLESSON:\n{ex['lesson']}", file=sys.stderr)

    # Pattern distribution
    print(f"\n{'='*80}", file=sys.stderr)
    print("LOSS PATTERN DISTRIBUTION", file=sys.stderr)
    print(f"{'='*80}", file=sys.stderr)
    loss_patterns = {}
    for n in narratives:
        if n["label"] == "LOSS":
            p = n["pattern"]
            loss_patterns[p] = loss_patterns.get(p, {"count": 0, "total_loss": 0})
            loss_patterns[p]["count"] += 1
            loss_patterns[p]["total_loss"] += abs(n["pnl"])
    for p, v in sorted(loss_patterns.items(), key=lambda x: -x[1]["total_loss"]):
        print(f"  {p:<25} {v['count']:>4} trades  ${v['total_loss']:>10,.0f} total loss",
              file=sys.stderr)

    print(f"\n{'='*80}", file=sys.stderr)


if __name__ == "__main__":
    main()
