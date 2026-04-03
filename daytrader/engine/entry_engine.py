"""Layer 3: Entry Engine — detect first pullback, VWAP reclaim, and flat top patterns.

Consumes 1-min bars from BarManager and emits entry signals with defined
stop, target, and position size.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime
from typing import Optional

import pandas as pd
import numpy as np

from daytrader.engine.indicators import (
    compute_vwap, compute_atr, compute_ema, opening_range,
    is_green, is_red, candle_body_position,
)
from daytrader.config import (
    ENTRY_OR_MINUTES, ENTRY_BREAKOUT_VOL_MULT, ENTRY_PULLBACK_MAX_RETRACE,
    ENTRY_MAX_ENTRY_TIME_MIN, ENTRY_REQUIRE_ABOVE_VWAP, ENTRY_MIN_STOP_ATR_MULT,
    MAX_RISK_DOLLARS, SLIPPAGE_PER_SHARE,
)

logger = logging.getLogger("daytrader.entry")


@dataclass
class EntrySignal:
    """A trade entry signal from the entry engine."""
    symbol: str
    pattern: str            # "first_pullback" | "vwap_reclaim" | "flat_top"
    direction: str          # "long" | "short"
    entry_price: float
    stop_price: float
    target_price: float
    shares: int
    risk_per_share: float
    r_target: float         # target R-multiple
    vwap: float
    atr: float
    bar_index: int          # index in the bar DataFrame
    timestamp: str
    confidence: float       # 0-1 quality score
    reason: str


def compute_position_size(entry_price: float, stop_price: float,
                          max_risk: float = MAX_RISK_DOLLARS) -> tuple[int, float]:
    """Calculate shares and risk per share from entry and stop."""
    risk_per_share = abs(entry_price - stop_price)
    if risk_per_share < 0.01:
        return 0, 0.0
    shares = int(max_risk / risk_per_share)
    # Cap position value at a reasonable level
    max_shares = int(200_000 * 0.20 / entry_price)  # 20% of account max
    shares = min(shares, max_shares, 10_000)
    return max(shares, 0), risk_per_share


def detect_first_pullback(symbol: str, bars: pd.DataFrame,
                          catalyst_score: int = 6) -> Optional[EntrySignal]:
    """
    Detect first pullback pattern after opening range breakout.

    Steps:
      1. Opening range = first 15 bars (15 min of 1-min bars)
      2. Breakout: first bar after OR that closes above OR high with volume
      3. Pullback: first red candle after breakout
      4. Entry: next green candle closes above pullback low → buy next bar open
    """
    if len(bars) < 25:
        return None

    or_high, or_low = opening_range(bars, ENTRY_OR_MINUTES)
    or_range = or_high - or_low
    if or_range < 0.05:
        return None

    vwap = compute_vwap(bars)
    atr_series = compute_atr(bars, 14)

    # Average volume during opening range
    or_avg_vol = bars.iloc[:ENTRY_OR_MINUTES]["Volume"].mean()

    # Find breakout bar (after OR, closes above OR high, volume > 2x OR avg)
    breakout_idx = None
    max_search = min(ENTRY_OR_MINUTES + ENTRY_MAX_ENTRY_TIME_MIN, len(bars))

    for i in range(ENTRY_OR_MINUTES, max_search):
        bar = bars.iloc[i]
        if (bar["Close"] > or_high and
                bar["Volume"] > or_avg_vol * ENTRY_BREAKOUT_VOL_MULT):
            breakout_idx = i
            break

    if breakout_idx is None:
        return None

    breakout_high = bars.iloc[breakout_idx]["High"]
    breakout_close = bars.iloc[breakout_idx]["Close"]

    # Find first red candle after breakout (pullback)
    pullback_idx = None
    for j in range(breakout_idx + 1, min(breakout_idx + 15, len(bars))):
        if is_red(bars.iloc[j]):
            pullback_idx = j
            break

    if pullback_idx is None:
        return None

    pullback_low = bars.iloc[pullback_idx]["Low"]

    # Check pullback depth: must retrace < 50% of breakout move
    breakout_move = breakout_high - or_high
    retrace = breakout_high - pullback_low
    if breakout_move > 0 and retrace / breakout_move > ENTRY_PULLBACK_MAX_RETRACE:
        return None

    # Check pullback volume < breakout volume (sellers exhausting)
    pb_vol = bars.iloc[pullback_idx]["Volume"]
    bo_vol = bars.iloc[breakout_idx]["Volume"]
    if pb_vol > bo_vol:
        return None  # sellers more active than buyers — not a good pullback

    # Wait for green candle after pullback (confirmation)
    green_idx = None
    for k in range(pullback_idx + 1, min(pullback_idx + 10, len(bars))):
        if is_green(bars.iloc[k]) and bars.iloc[k]["Close"] > pullback_low:
            green_idx = k
            break

    if green_idx is None or green_idx + 1 >= len(bars):
        return None

    # Entry at next bar open
    entry_idx = green_idx + 1
    if entry_idx >= len(bars):
        return None

    entry_bar = bars.iloc[entry_idx]
    entry_price = float(entry_bar["Open"])
    cur_vwap = float(vwap.iloc[entry_idx])
    cur_atr = float(atr_series.iloc[entry_idx]) if not np.isnan(atr_series.iloc[entry_idx]) else or_range

    # VWAP filter
    if ENTRY_REQUIRE_ABOVE_VWAP and entry_price < cur_vwap:
        return None

    # Stop below pullback low, but at least 1x ATR
    raw_stop = pullback_low - 0.02
    min_stop = entry_price - max(cur_atr * ENTRY_MIN_STOP_ATR_MULT, 0.10)
    stop_price = min(raw_stop, min_stop)  # take the wider (more protective) stop

    # Target: 2R for scale-out, 3R for runner
    risk_per_share = entry_price - stop_price
    if risk_per_share <= 0.02:
        return None

    target_price = entry_price + 2.0 * risk_per_share
    shares, rps = compute_position_size(entry_price, stop_price)
    if shares < 10:
        return None

    confidence = min(1.0, (catalyst_score / 10) * 0.5 + 0.5)

    return EntrySignal(
        symbol=symbol,
        pattern="first_pullback",
        direction="long",
        entry_price=round(entry_price, 2),
        stop_price=round(stop_price, 2),
        target_price=round(target_price, 2),
        shares=shares,
        risk_per_share=round(rps, 2),
        r_target=2.0,
        vwap=round(cur_vwap, 2),
        atr=round(cur_atr, 2),
        bar_index=entry_idx,
        timestamp=str(bars.index[entry_idx]),
        confidence=round(confidence, 2),
        reason=f"OR break at bar {breakout_idx}, pullback at {pullback_idx}, green confirm at {green_idx}",
    )


def detect_vwap_reclaim(symbol: str, bars: pd.DataFrame,
                         catalyst_score: int = 6) -> Optional[EntrySignal]:
    """
    Detect VWAP reclaim: stock below VWAP, then crosses above with volume surge.

    Only valid if reclaim happens before 10:15 AM (bar 45 of 1-min bars).
    """
    if len(bars) < 30:
        return None

    vwap = compute_vwap(bars)
    atr_series = compute_atr(bars, 14)

    # Look for reclaim between bar 20 and bar 45 (9:50-10:15 AM)
    for i in range(20, min(45, len(bars))):
        bar = bars.iloc[i]
        cur_vwap = float(vwap.iloc[i])

        # Must have been below VWAP for at least 10 consecutive bars
        recent_below = all(
            float(bars.iloc[j]["Close"]) < float(vwap.iloc[j])
            for j in range(max(0, i - 10), i)
        )
        if not recent_below:
            continue

        # Reclaim: current bar closes above VWAP with volume surge
        avg_vol = bars.iloc[max(0, i-10):i]["Volume"].mean()
        if (bar["Close"] > cur_vwap and
                bar["Volume"] > avg_vol * 2.0 and
                is_green(bar)):

            entry_price = float(bar["Close"])
            cur_atr = float(atr_series.iloc[i]) if not np.isnan(atr_series.iloc[i]) else 0.30
            stop_price = cur_vwap - max(0.15, cur_atr * ENTRY_MIN_STOP_ATR_MULT * 0.5)

            risk_per_share = entry_price - stop_price
            if risk_per_share <= 0.02:
                continue

            target_price = entry_price + 3.0 * risk_per_share
            shares, rps = compute_position_size(entry_price, stop_price)
            if shares < 10:
                continue

            confidence = min(1.0, (catalyst_score / 10) * 0.4 + 0.4)

            return EntrySignal(
                symbol=symbol,
                pattern="vwap_reclaim",
                direction="long",
                entry_price=round(entry_price, 2),
                stop_price=round(stop_price, 2),
                target_price=round(target_price, 2),
                shares=shares,
                risk_per_share=round(rps, 2),
                r_target=3.0,
                vwap=round(cur_vwap, 2),
                atr=round(cur_atr, 2),
                bar_index=i,
                timestamp=str(bars.index[i]),
                confidence=round(confidence, 2),
                reason=f"VWAP reclaim at bar {i} after {10}+ bars below",
            )

    return None


def detect_flat_top(symbol: str, bars: pd.DataFrame,
                     catalyst_score: int = 6) -> Optional[EntrySignal]:
    """
    Detect flat top breakout: multiple tests of same resistance with higher lows.

    Looks for 2+ tests of the same high (within $0.05) with ascending lows.
    """
    if len(bars) < 30:
        return None

    vwap = compute_vwap(bars)
    atr_series = compute_atr(bars, 14)

    # Scan for flat top formation between bar 15 and bar 60
    for start in range(15, min(50, len(bars) - 10)):
        window = bars.iloc[start:start + 20]
        if len(window) < 10:
            continue

        # Find the resistance level (most frequent high within $0.05 bands)
        highs = window["High"].values
        # Simple: find max high that was tested 2+ times
        max_high = float(window["High"].max())
        tolerance = 0.05
        tests = [j for j, h in enumerate(highs) if abs(h - max_high) <= tolerance]

        if len(tests) < 2:
            continue

        # Check higher lows between tests
        lows_between = [float(window.iloc[j]["Low"]) for j in range(tests[0], tests[-1] + 1)]
        if len(lows_between) < 3:
            continue

        # Simple higher-low check: last low > first low
        first_low = min(lows_between[:len(lows_between)//2])
        last_low = min(lows_between[len(lows_between)//2:])
        if last_low <= first_low:
            continue  # not making higher lows

        # Look for breakout candle after the flat top
        for k in range(start + tests[-1] + 1, min(start + tests[-1] + 10, len(bars))):
            bar = bars.iloc[k]
            cur_vwap = float(vwap.iloc[k])
            cur_atr = float(atr_series.iloc[k]) if not np.isnan(atr_series.iloc[k]) else 0.30

            if bar["Close"] > max_high + 0.02 and bar["Volume"] > window["Volume"].mean() * 1.5:
                entry_price = float(bar["Close"])

                if ENTRY_REQUIRE_ABOVE_VWAP and entry_price < cur_vwap:
                    continue

                stop_price = last_low - 0.02
                stop_price = min(stop_price, entry_price - cur_atr * ENTRY_MIN_STOP_ATR_MULT)

                risk_per_share = entry_price - stop_price
                if risk_per_share <= 0.02:
                    continue

                target_price = entry_price + 2.0 * risk_per_share
                shares, rps = compute_position_size(entry_price, stop_price)
                if shares < 10:
                    continue

                confidence = min(1.0, (catalyst_score / 10) * 0.3 + 0.3 + len(tests) * 0.1)

                return EntrySignal(
                    symbol=symbol,
                    pattern="flat_top",
                    direction="long",
                    entry_price=round(entry_price, 2),
                    stop_price=round(stop_price, 2),
                    target_price=round(target_price, 2),
                    shares=shares,
                    risk_per_share=round(rps, 2),
                    r_target=2.0,
                    vwap=round(cur_vwap, 2),
                    atr=round(cur_atr, 2),
                    bar_index=k,
                    timestamp=str(bars.index[k]),
                    confidence=round(confidence, 2),
                    reason=f"Flat top at ${max_high:.2f} tested {len(tests)}x, higher lows confirmed",
                )
            break  # only try one bar after formation

    return None


def scan_for_entries(symbol: str, bars: pd.DataFrame,
                     catalyst_score: int = 6) -> list[EntrySignal]:
    """Run all pattern detectors on a symbol's bars. Returns list of signals."""
    signals = []

    for detector in [detect_first_pullback, detect_vwap_reclaim, detect_flat_top]:
        try:
            signal = detector(symbol, bars, catalyst_score)
            if signal is not None:
                signals.append(signal)
                logger.info("SIGNAL: %s %s on %s at $%.2f (stop $%.2f, %d shares)",
                           signal.pattern, signal.direction, signal.symbol,
                           signal.entry_price, signal.stop_price, signal.shares)
        except Exception as e:
            logger.warning("Error in %s for %s: %s", detector.__name__, symbol, e)

    # Sort by confidence (best first)
    signals.sort(key=lambda s: s.confidence, reverse=True)
    return signals
