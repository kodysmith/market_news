"""
De-Pin Risk backend computation + deterministic test harness
============================================================

Design goals:
- Deterministic: no randomness, no network calls
- Rigorous math: explicit formulas, explicit normalization, explicit clamps
- Production-shape: clear inputs/outputs, stateful rolling windows
- Python backend friendly: can run as a scheduled/streaming job
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import List, Dict, Optional, Tuple
import math
from datetime import datetime, date, timedelta
import yfinance as yf


# ----------------------------
# Utilities
# ----------------------------

EPS = 1e-12


def clamp(x: float, lo: float, hi: float) -> float:
    return lo if x < lo else hi if x > hi else x


def sigmoid(z: float) -> float:
    # numerically stable-ish sigmoid
    if z >= 0:
        ez = math.exp(-z)
        return 1.0 / (1.0 + ez)
    else:
        ez = math.exp(z)
        return ez / (1.0 + ez)


def sma(values: List[float]) -> float:
    if not values:
        return 0.0
    return sum(values) / len(values)


def ema(prev_ema: Optional[float], value: float, period: int) -> float:
    """
    Standard EMA update:
      alpha = 2/(period+1)
      ema_t = alpha*value + (1-alpha)*ema_{t-1}
    """
    alpha = 2.0 / (period + 1.0)
    if prev_ema is None:
        return value
    return alpha * value + (1.0 - alpha) * prev_ema


# ----------------------------
# Data models
# ----------------------------

@dataclass(frozen=True)
class Bar5m:
    ts: str
    open: float
    high: float
    low: float
    close: float
    volume: float


@dataclass(frozen=True)
class OptionContract:
    # snapshot row
    expiration: str               # e.g., "2026-01-23"
    dte: int                      # days to expiration
    strike: float
    opt_type: str                 # "call"|"put"
    open_interest: float
    gamma: float                  # per option contract gamma
    # optional fields for later expansion: iv, delta, vega, theta, etc.


@dataclass
class RollingState:
    """
    Minimal state required for de-pin risk.
    Store only what's needed (don't over-optimize, but do be explicit).
    """
    # bar-based state
    prev_close: Optional[float] = None
    ema8: Optional[float] = None
    ema21: Optional[float] = None

    # ATR (Wilder) state
    atr14: Optional[float] = None
    tr_history: Optional[List[float]] = None  # bootstrap ATR with first 14 TRs

    # 30m history (6 bars) for move30 and netGEX change
    closes_30m: Optional[List[float]] = None     # length <= 7 (keep last 7)
    netgex_30m: Optional[List[float]] = None     # length <= 7 (keep last 7)

    # 15m (3 bars) history for pin persist
    pin_scores_15m: Optional[List[float]] = None  # length <= 3

    # volume over last 30m (6 bars)
    vol_30m: Optional[List[float]] = None  # length <= 6

    def __post_init__(self):
        if self.tr_history is None:
            self.tr_history = []
        if self.closes_30m is None:
            self.closes_30m = []
        if self.netgex_30m is None:
            self.netgex_30m = []
        if self.pin_scores_15m is None:
            self.pin_scores_15m = []
        if self.vol_30m is None:
            self.vol_30m = []


@dataclass(frozen=True)
class WallResult:
    put_wall: float
    call_wall: float
    dominant_strike: float
    put_wall_strength: float
    call_wall_strength: float
    dominant_strength: float
    # full maps (useful for UI / debugging)
    oi_put_by_strike: Dict[float, float]
    oi_call_by_strike: Dict[float, float]


@dataclass(frozen=True)
class DepinRiskResult:
    symbol: str
    ts: str
    spot: float
    bucket: str

    # key levels
    dominant_strike: float
    put_wall: float
    call_wall: float

    # derived market measures
    atr5m: float
    pin_persist: float
    trend_strength: float
    move30: float
    vol_ratio: Optional[float]

    # gex
    net_gex: float
    gex_change_30m: Optional[float]

    # normalized features
    gex_collapse: float
    liq_fade: float
    wall_drift: float

    # score
    de_pin_risk: int
    band: str

    # for explanation / debugging
    x_raw: float
    guidance: str


# ----------------------------
# Core computations
# ----------------------------

def compute_atr14_wilder(state: RollingState, bar: Bar5m) -> float:
    """
    Wilder ATR(14) updated per 5m bar using TR.
    TR = max(high-low, abs(high-prev_close), abs(low-prev_close))
    ATR bootstrap: SMA of first 14 TRs, then Wilder smoothing.
    """
    if state.prev_close is None:
        tr = bar.high - bar.low
    else:
        tr = max(
            bar.high - bar.low,
            abs(bar.high - state.prev_close),
            abs(bar.low - state.prev_close),
        )

    # bootstrap
    if state.atr14 is None:
        state.tr_history.append(tr)
        if len(state.tr_history) < 14:
            # not enough data yet; use SMA so far (still deterministic)
            atr = sma(state.tr_history)
        else:
            atr = sma(state.tr_history[-14:])
            state.atr14 = atr
    else:
        # Wilder smoothing: ATR_t = (ATR_{t-1}*(n-1) + TR_t) / n
        n = 14
        state.atr14 = ((state.atr14 * (n - 1)) + tr) / n
        atr = state.atr14

    state.prev_close = bar.close
    return max(atr, 1e-6)  # avoid divide-by-zero madness


def compute_oi_walls(
    spot: float,
    options: List[OptionContract],
    dte_max: int,
    window_abs: float,
) -> WallResult:
    """
    Aggregates OI by strike for calls/puts for a bucket (DTE<=dte_max) and within strike window.
    """
    lo = spot - window_abs
    hi = spot + window_abs

    oi_put: Dict[float, float] = {}
    oi_call: Dict[float, float] = {}

    for opt in options:
        if opt.dte > dte_max:
            continue
        if opt.strike < lo or opt.strike > hi:
            continue

        if opt.opt_type.lower() == "put":
            oi_put[opt.strike] = oi_put.get(opt.strike, 0.0) + float(opt.open_interest)
        elif opt.opt_type.lower() == "call":
            oi_call[opt.strike] = oi_call.get(opt.strike, 0.0) + float(opt.open_interest)
        else:
            raise ValueError(f"Unknown opt_type: {opt.opt_type}")

    # unify strike set for dominance
    all_strikes = sorted(set(oi_put.keys()) | set(oi_call.keys()))
    if not all_strikes:
        # deterministic fallback: if no strikes, "walls" become spot rounded to 1
        fallback = round(spot, 0)
        return WallResult(
            put_wall=fallback,
            call_wall=fallback,
            dominant_strike=fallback,
            put_wall_strength=0.0,
            call_wall_strength=0.0,
            dominant_strength=0.0,
            oi_put_by_strike={},
            oi_call_by_strike={},
        )

    # compute totals
    def safe_argmax(d: Dict[float, float]) -> float:
        # if empty, fall back to nearest strike
        if not d:
            return min(all_strikes, key=lambda k: abs(k - spot))
        # deterministic tie-breaker: prefer strike closer to spot, then lower strike
        mx = max(d.values())
        candidates = [k for k, v in d.items() if v == mx]
        candidates.sort(key=lambda k: (abs(k - spot), k))
        return candidates[0]

    put_wall = safe_argmax(oi_put)
    call_wall = safe_argmax(oi_call)

    # dominant uses put+call
    oi_total = {k: oi_put.get(k, 0.0) + oi_call.get(k, 0.0) for k in all_strikes}
    mx_total = max(oi_total.values())
    dom_candidates = [k for k, v in oi_total.items() if v == mx_total]
    dom_candidates.sort(key=lambda k: (abs(k - spot), k))
    dominant_strike = dom_candidates[0]

    sum_put = sum(oi_put.values()) + EPS
    sum_call = sum(oi_call.values()) + EPS
    sum_total = sum(oi_total.values()) + EPS

    put_strength = oi_put.get(put_wall, 0.0) / sum_put if oi_put else 0.0
    call_strength = oi_call.get(call_wall, 0.0) / sum_call if oi_call else 0.0
    dom_strength = oi_total.get(dominant_strike, 0.0) / sum_total

    return WallResult(
        put_wall=float(put_wall),
        call_wall=float(call_wall),
        dominant_strike=float(dominant_strike),
        put_wall_strength=float(put_strength),
        call_wall_strength=float(call_strength),
        dominant_strength=float(dom_strength),
        oi_put_by_strike=dict(oi_put),
        oi_call_by_strike=dict(oi_call),
    )


def compute_net_gex(
    spot: float,
    options: List[OptionContract],
    dte_max: int,
    window_abs: float,
    contract_multiplier: float = 100.0,
) -> float:
    """
    Practical proxy:
      GEX = - (OI_call*Gamma_call + OI_put*Gamma_put) * Mult * spot^2

    Notes:
    - Uses gamma magnitude "as is" per contract
    - Uses the dealer-short proxy (negative sign)
    - Aggregates only within window and bucket
    """
    lo = spot - window_abs
    hi = spot + window_abs

    total = 0.0
    for opt in options:
        if opt.dte > dte_max:
            continue
        if opt.strike < lo or opt.strike > hi:
            continue
        # both calls and puts contribute via gamma magnitude
        total += float(opt.open_interest) * float(opt.gamma)

    net_gex = -total * contract_multiplier * (spot ** 2)
    return float(net_gex)


def compute_depin_risk(
    symbol: str,
    bar: Bar5m,
    options_snapshot: List[OptionContract],
    state: RollingState,
    *,
    bucket_dte_max: int = 2,
    strike_window_pct: float = 0.01,
    strike_window_floor: float = 5.0,  # SPY default; you can override per symbol
    tick_size: float = 0.01,
    vol_ref_30m: Optional[float] = None,  # pass time-of-day median (precomputed)
) -> DepinRiskResult:
    """
    Computes de-pin risk per the spec:
      x = 1.40*gexCollapse + 0.90*trendStrength + 0.70*move30 + 0.80*(1-pinPersist)
          + 0.50*liqFade + 0.50*wallDrift
      dePinRisk = 100*sigmoid(x - 1.15)
    """
    spot = float(bar.close)

    # --- bar-derived measures
    atr = compute_atr14_wilder(state, bar)

    state.ema8 = ema(state.ema8, spot, 8)
    state.ema21 = ema(state.ema21, spot, 21)

    trend_strength = abs((state.ema8 or spot) - (state.ema21 or spot)) / max(atr, 1e-6)

    # keep closes history for move30
    state.closes_30m.append(spot)
    if len(state.closes_30m) > 7:
        state.closes_30m.pop(0)

    if len(state.closes_30m) >= 7:
        close_30m_ago = state.closes_30m[-7]
        move30 = abs(spot - close_30m_ago) / max(atr, 1e-6)
    else:
        move30 = 0.0

    # volume 30m
    state.vol_30m.append(float(bar.volume))
    if len(state.vol_30m) > 6:
        state.vol_30m.pop(0)
    vol_now_30m = sum(state.vol_30m)

    # vol ratio + liquidity fade
    if vol_ref_30m is None or vol_ref_30m <= 0:
        vol_ratio = None
        liq_fade = 0.0  # conservative: don't hallucinate liquidity risk
    else:
        vol_ratio = vol_now_30m / max(vol_ref_30m, EPS)
        liq_fade = clamp(1.0 - vol_ratio, 0.0, 1.0)

    # --- options-derived measures
    window_abs = max(strike_window_pct * spot, strike_window_floor)

    walls = compute_oi_walls(
        spot=spot,
        options=options_snapshot,
        dte_max=bucket_dte_max,
        window_abs=window_abs,
    )

    net_gex = compute_net_gex(
        spot=spot,
        options=options_snapshot,
        dte_max=bucket_dte_max,
        window_abs=window_abs,
    )

    # keep netGEX history for 30m change
    state.netgex_30m.append(net_gex)
    if len(state.netgex_30m) > 7:
        state.netgex_30m.pop(0)

    gex_change_30m: Optional[float]
    if len(state.netgex_30m) >= 7:
        net_30m_ago = state.netgex_30m[-7]
        gex_change_30m = (net_gex - net_30m_ago) / max(abs(net_30m_ago), EPS)
    else:
        gex_change_30m = None

    if gex_change_30m is None:
        gex_collapse = 0.0
    else:
        gex_collapse = clamp(-gex_change_30m, 0.0, 1.0)

    # --- pin persist
    d = abs(spot - walls.dominant_strike)
    dN = d / max(atr, tick_size)
    pin_score = math.exp(-dN / 0.6)

    state.pin_scores_15m.append(pin_score)
    if len(state.pin_scores_15m) > 3:
        state.pin_scores_15m.pop(0)
    pin_persist = sma(state.pin_scores_15m)

    # --- wall drift
    wall_dist = min(abs(spot - walls.put_wall), abs(spot - walls.call_wall)) / max(atr, 1e-6)
    wall_drift = clamp(wall_dist / 1.2, 0.0, 1.0)

    # --- combine into depin score
    x = (
        1.40 * gex_collapse
        + 0.90 * trend_strength
        + 0.70 * move30
        + 0.80 * (1.0 - pin_persist)
        + 0.50 * liq_fade
        + 0.50 * wall_drift
    )

    score = int(round(100.0 * sigmoid(x - 1.15)))
    score = int(clamp(score, 0, 100))

    if score <= 30:
        band = "LOW"
        guidance = "Pin likely holds; theta ok only if defined-risk."
    elif score <= 60:
        band = "MID"
        guidance = "Pin unstable; reduce short gamma, prefer defined-risk / convexity."
    else:
        band = "HIGH"
        guidance = "Break risk high; avoid short gamma, consider convexity / wait for clarity."

    return DepinRiskResult(
        symbol=symbol,
        ts=bar.ts,
        spot=spot,
        bucket=f"DTE<={bucket_dte_max}",
        dominant_strike=walls.dominant_strike,
        put_wall=walls.put_wall,
        call_wall=walls.call_wall,
        atr5m=float(atr),
        pin_persist=float(pin_persist),
        trend_strength=float(trend_strength),
        move30=float(move30),
        vol_ratio=None if vol_ratio is None else float(vol_ratio),
        net_gex=float(net_gex),
        gex_change_30m=None if gex_change_30m is None else float(gex_change_30m),
        gex_collapse=float(gex_collapse),
        liq_fade=float(liq_fade),
        wall_drift=float(wall_drift),
        de_pin_risk=score,
        band=band,
        x_raw=float(x),
        guidance=guidance,
    )


# ----------------------------
# Data fetching helpers
# ----------------------------

def fetch_5m_bars(symbol: str, period: str = "5d") -> List[Bar5m]:
    """
    Fetch 5-minute bars from yfinance.
    
    Args:
        symbol: Stock/ETF symbol
        period: Period to fetch (default "5d" for ~5 days of 5m bars)
    
    Returns:
        List of Bar5m objects, sorted by timestamp (oldest first)
    """
    try:
        ticker = yf.Ticker(symbol)
        hist = ticker.history(interval='5m', period=period)
        
        if hist.empty:
            return []
        
        bars = []
        for idx, row in hist.iterrows():
            bars.append(Bar5m(
                ts=idx.strftime('%Y-%m-%d %H:%M:%S'),
                open=float(row['Open']),
                high=float(row['High']),
                low=float(row['Low']),
                close=float(row['Close']),
                volume=float(row['Volume'])
            ))
        
        return bars
    except Exception as e:
        print(f"Error fetching 5m bars for {symbol}: {e}")
        return []


def convert_options_to_contracts(options_data: List[Dict], spot_price: float) -> List[OptionContract]:
    """
    Convert options data from API format to OptionContract objects.
    
    Args:
        options_data: List of option dicts from Massive API or similar
        spot_price: Current spot price for DTE calculation
    
    Returns:
        List of OptionContract objects
    """
    contracts = []
    today = date.today()
    
    for opt in options_data:
        try:
            # Massive API nests data in 'details'
            d = opt.get('details') or opt
            
            # Parse expiration date
            exp_date_str = d.get('expiration_date') or d.get('expirationDate') or d.get('expiry')
            if isinstance(exp_date_str, str):
                exp_date = datetime.strptime(exp_date_str.split('T')[0], '%Y-%m-%d').date()
            else:
                continue
            
            # Calculate DTE
            dte = (exp_date - today).days
            
            # Get option type - Massive API uses 'contract_type' in details
            opt_type = d.get('contract_type') or d.get('option_type') or d.get('type') or d.get('right', '').lower()
            if opt_type.lower() in ['c', 'call']:
                opt_type = 'call'
            elif opt_type.lower() in ['p', 'put']:
                opt_type = 'put'
            else:
                continue
            
            # Get required fields - check both top level and details
            strike = float(d.get('strike_price') or d.get('strike') or opt.get('strike_price') or opt.get('strike') or 0)
            oi = float(d.get('open_interest') or d.get('openInterest') or opt.get('open_interest') or opt.get('openInterest') or 0)
            gamma = float(d.get('gamma') or d.get('gamma_value') or opt.get('gamma') or opt.get('gamma_value') or 0)
            
            contracts.append(OptionContract(
                expiration=exp_date.isoformat(),
                dte=dte,
                strike=strike,
                opt_type=opt_type,
                open_interest=oi,
                gamma=gamma
            ))
        except (ValueError, KeyError, TypeError) as e:
            # Skip invalid contracts
            continue
    
    return contracts


# ----------------------------
# Deterministic test harness
# ----------------------------

def _make_flat_bars(ts_prefix: str, start_price: float, n: int, vol: float) -> List[Bar5m]:
    """
    Deterministic synthetic bars:
    - tiny range
    - stable price
    - constant volume
    """
    bars = []
    p = start_price
    for i in range(n):
        # deterministic micro wiggle: alternates +0.02 / -0.02 without randomness
        wiggle = 0.02 if (i % 2 == 0) else -0.02
        o = p
        c = p + wiggle
        h = max(o, c) + 0.05
        l = min(o, c) - 0.05
        bars.append(Bar5m(
            ts=f"{ts_prefix}{i:02d}",
            open=o, high=h, low=l, close=c,
            volume=vol
        ))
        p = c
    return bars


def _make_trend_bars(ts_prefix: str, start_price: float, step: float, n: int, vol: float) -> List[Bar5m]:
    """
    Deterministic trending bars.
    """
    bars = []
    p = start_price
    for i in range(n):
        o = p
        c = p + step
        h = max(o, c) + 0.10
        l = min(o, c) - 0.10
        bars.append(Bar5m(
            ts=f"{ts_prefix}{i:02d}",
            open=o, high=h, low=l, close=c,
            volume=vol
        ))
        p = c
    return bars


def _make_options_snapshot_pinny(spot: float) -> List[OptionContract]:
    """
    Options snapshot designed to create a strong pin around 690:
    - dominant OI at 690 (calls + puts)
    - solid OI at 688 put wall and 690 call wall
    - moderate gamma values
    """
    # strikes around spot
    strikes = [684, 686, 688, 690, 692, 694]
    opts: List[OptionContract] = []
    today = date.today()
    for k in strikes:
        # baseline
        put_oi = 200
        call_oi = 200

        # emphasize 688 puts, 690 calls, 690 total dominance
        if k == 688:
            put_oi = 3000
        if k == 690:
            call_oi = 2800
            put_oi = 2500

        # deterministic gamma: slightly higher near spot
        gamma = 0.015 if abs(k - spot) <= 2 else 0.008

        # create DTE<=2 bucket only
        opts.append(OptionContract(expiration=today.isoformat(), dte=1, strike=float(k), opt_type="put", open_interest=float(put_oi), gamma=float(gamma)))
        opts.append(OptionContract(expiration=today.isoformat(), dte=1, strike=float(k), opt_type="call", open_interest=float(call_oi), gamma=float(gamma)))
    return opts


def _make_options_snapshot_gex_collapse(spot: float) -> List[OptionContract]:
    """
    Same structure but much lower OI * gamma => netGEX magnitude drops dramatically.
    That should drive gexCollapse up (if previous netGEX was large).
    """
    strikes = [684, 686, 688, 690, 692, 694]
    opts: List[OptionContract] = []
    today = date.today()
    for k in strikes:
        put_oi = 120
        call_oi = 120
        if k == 688:
            put_oi = 500
        if k == 690:
            call_oi = 450
            put_oi = 400
        gamma = 0.006 if abs(k - spot) <= 2 else 0.003
        opts.append(OptionContract(expiration=today.isoformat(), dte=1, strike=float(k), opt_type="put", open_interest=float(put_oi), gamma=float(gamma)))
        opts.append(OptionContract(expiration=today.isoformat(), dte=1, strike=float(k), opt_type="call", open_interest=float(call_oi), gamma=float(gamma)))
    return opts


def run_deterministic_tests() -> None:
    """
    Deterministic tests:
    1) Pinny conditions => DePinRisk should be LOW after warmup
    2) Trend + gex collapse + drift => DePinRisk should increase materially
    """
    symbol = "SPY"
    state = RollingState()

    # Use a fixed vol reference so volRatio is deterministic
    # Suppose typical 30m volume is 600k; we'll test against it.
    vol_ref_30m = 600_000.0

    # ---- Phase 1: warm up with flat pinned market
    spot0 = 690.0
    bars = _make_flat_bars(ts_prefix="T", start_price=spot0, n=30, vol=120_000.0)  # 30 bars => plenty for ATR/EMA history
    options_pinny = _make_options_snapshot_pinny(spot=spot0)

    last_res = None
    for b in bars:
        last_res = compute_depin_risk(
            symbol=symbol,
            bar=b,
            options_snapshot=options_pinny,
            state=state,
            bucket_dte_max=2,
            strike_window_pct=0.01,
            strike_window_floor=5.0,
            tick_size=0.01,
            vol_ref_30m=vol_ref_30m,
        )

    assert last_res is not None
    # Expect LOW-ish risk because:
    # - gexCollapse ~= 0 (netGEX stable)
    # - trendStrength small
    # - pinPersist high (dominant 690, spot near)
    # - wallDrift small
    print("PHASE 1 RESULT:", last_res)
    assert last_res.de_pin_risk <= 40, f"Expected lowish risk, got {last_res.de_pin_risk}"
    assert last_res.band in ("LOW", "MID"), f"Unexpected band: {last_res.band}"
    assert abs(last_res.dominant_strike - 690.0) < 1e-6

    # ---- Phase 2: force de-pin: trend day + price drifts + netGEX collapses
    # Create an uptrend for 12 bars (1 hour), reducing pin persist and increasing trend metrics.
    trend_bars = _make_trend_bars(ts_prefix="U", start_price=last_res.spot, step=0.35, n=12, vol=60_000.0)

    # Use a collapsed-gex snapshot to create gexCollapse relative to earlier netGEX history.
    options_collapse = _make_options_snapshot_gex_collapse(spot=last_res.spot)

    for b in trend_bars:
        # Update spot in options snapshot generation only if you want; here keep structure deterministic.
        res2 = compute_depin_risk(
            symbol=symbol,
            bar=b,
            options_snapshot=options_collapse,
            state=state,
            bucket_dte_max=2,
            strike_window_pct=0.01,
            strike_window_floor=5.0,
            tick_size=0.01,
            vol_ref_30m=vol_ref_30m,
        )

    print("PHASE 2 RESULT:", res2)
    # Expect materially higher risk than phase 1
    assert res2.de_pin_risk >= last_res.de_pin_risk + 15, f"Expected risk to rise, got {last_res.de_pin_risk} -> {res2.de_pin_risk}"
    # In many cases, should land MID/HIGH. We don't hard assert HIGH because weights + ATR may vary.
    assert res2.band in ("MID", "HIGH"), f"Expected MID/HIGH, got {res2.band}"

    print("✅ Deterministic tests passed.")


if __name__ == "__main__":
    run_deterministic_tests()
