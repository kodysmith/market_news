"""
Trade Ideas Engine - Regime-Aware Allowed-Only Trade Generation

Generates only actionable trade ideas that are permitted by current market regime,
ranked by risk-adjusted ROI using GEX walls and options chain data.

Design goals:
- Deterministic, rigorous math
- Production-shape design
- Clear inputs/outputs
- Regime-aware permissioning
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import List, Dict, Optional, Tuple, Any
from datetime import datetime, date, timedelta
from collections import defaultdict
import json
import os
import math
import logging

import numpy as np
from scipy.stats import norm

logger = logging.getLogger(__name__)

# Risk-free rate (approximate, can be made configurable)
RISK_FREE_RATE = 0.05


# ===============================
# Data Models
# ===============================

@dataclass
class MarketContext:
    """Market context extracted from cockpit state"""
    symbol: str
    asOf: str  # ISO timestamp
    spot: float
    
    # Regime
    volatilityRegime: str  # "LOW" | "NORMAL" | "ELEVATED" | "EXTREME"
    trendRegime: str  # "UP" | "RANGE" | "DOWN" | "CHAOTIC"
    liquidityState: str  # "NORMAL" | "THIN" | "STRESSED"
    
    # GEX / structure
    putWall: Optional[float] = None
    callWall: Optional[float] = None
    flipLine: Optional[float] = None
    rangePts: Optional[float] = None
    depinRisk: Optional[float] = None  # 0-100
    pinConfidence: str = "LOW"  # "LOW" | "MEDIUM" | "HIGH"
    
    gexState: str = "NEUTRAL"  # "POSITIVE" | "NEUTRAL" | "NEGATIVE"
    gexSlope: Optional[str] = None  # "SUPPORTIVE" | "NEUTRAL" | "UNSTABLE"
    
    # Events
    earningsClusterScore: int = 0
    macroEventNext24h: bool = False
    megaCapEarningsNext24h: bool = False
    
    # Action filter
    actionFilter: str = "REAL_OK"  # "REAL_OK" | "WAIT_FOR_CLARITY" | "PAPER_ONLY" | "BLOCKED"


@dataclass
class OptionQuote:
    """Enhanced option contract with pricing and Greeks"""
    expiry: str  # YYYY-MM-DD
    dte: int
    type: str  # "CALL" | "PUT"
    strike: float
    
    bid: float
    ask: float
    mid: float  # (bid+ask)/2
    
    delta: Optional[float] = None
    theta: Optional[float] = None
    iv: Optional[float] = None
    
    openInterest: int = 0
    volume: int = 0
    bidAskSpread: float = 0.0  # ask-bid


@dataclass
class ExecutionLeg:
    """Single leg of a spread"""
    action: str  # "SELL" | "BUY"
    type: str  # "PUT" | "CALL"
    strike: float
    priceMid: float
    bid: float
    ask: float
    delta: Optional[float] = None
    oi: int = 0
    volume: int = 0


@dataclass
class BestMetrics:
    """Summary metrics for collapsed view"""
    expiry: str
    dte: int
    netCredit: float
    maxProfit: float  # $
    maxLoss: float  # $
    roc: float  # 0.18 = 18%
    pop: float  # 0.88
    ev: float  # $
    wallBufferPct: float
    liquidityGrade: str  # "A" | "B" | "C" | "D"
    score: float


@dataclass
class ReferenceExecution:
    """Individual spread structure"""
    expiry: str
    dte: int
    legs: List[ExecutionLeg]
    
    # Metrics
    netCredit: float
    width: float
    maxProfit: float
    maxLoss: float
    roc: float
    breakeven: float
    pop: float
    ev: float
    wallBufferPct: float
    liquidityScore: float
    score: float


@dataclass
class ContextSnapshot:
    """Context at generation time"""
    spot: float
    putWall: Optional[float] = None
    callWall: Optional[float] = None
    flipLine: Optional[float] = None
    depinRisk: Optional[float] = None
    gexState: str = "NEUTRAL"
    pinConfidence: str = "LOW"
    earningsClusterScore: int = 0
    macroEventNext24h: bool = False


@dataclass
class TradeIdea:
    """Complete trade idea"""
    id: str
    symbol: str
    asOf: str
    
    strategy: str  # "PUT_CREDIT_SPREAD" | "CALL_CREDIT_SPREAD" | "IRON_CONDOR"
    label: str  # e.g. "SPY Put Credit Spread — Intraday"
    
    # Permissioning
    allowed: bool = True
    mode: str = "OVERNIGHT_ALLOWED"  # "INTRADAY_ONLY" | "OVERNIGHT_ALLOWED" | "HEDGE_REQUIRED"
    confidence: str = "MEDIUM"  # "HIGH" | "MEDIUM" | "LOW"
    reasons: List[str] = None  # Why allowed
    
    # Market status
    preMarket: bool = False  # True if market is closed (weekend/after hours)
    marketStatusNote: Optional[str] = None  # Warning message for pre-market ideas
    
    # Reference executions
    executions: List[ReferenceExecution] = None
    
    # Summary metrics
    best: Optional[BestMetrics] = None
    
    # Context snapshot
    contextSnapshot: Optional[ContextSnapshot] = None
    
    def __post_init__(self):
        if self.reasons is None:
            self.reasons = []
        if self.executions is None:
            self.executions = []


# ===============================
# Black-Scholes Delta
# ===============================

def bs_delta(S: float, K: float, T: float, sigma: float, option_type: str, r: float = RISK_FREE_RATE) -> float:
    """
    Calculate Black-Scholes delta.
    
    Args:
        S: Spot price
        K: Strike price
        T: Time to expiry in years
        sigma: Implied volatility (as decimal)
        option_type: "call" or "put"
        r: Risk-free rate (default 0.05)
    
    Returns:
        Delta (call: 0-1, put: -1-0)
    """
    if S <= 0 or K <= 0 or T <= 0 or sigma <= 0:
        return 0.0
    
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    
    if option_type.lower() == "call":
        return float(norm.cdf(d1))
    else:  # put
        return float(norm.cdf(d1) - 1.0)


# ===============================
# Config Loading
# ===============================

def load_trade_ideas_config(symbol: str = "SPY") -> Dict[str, Any]:
    """
    Load trade ideas configuration.
    
    Args:
        symbol: Symbol to get symbol-specific config
    
    Returns:
        Config dict with defaults
    """
    config_path = os.path.join(
        os.path.dirname(os.path.dirname(__file__)),
        'data',
        'trade_ideas_config.json'
    )
    
    defaults = {
        "supportedSymbols": ["SPY", "QQQ", "IWM"],
        "baseStrategySet": ["PUT_CREDIT_SPREAD", "CALL_CREDIT_SPREAD"],
        "timeframes": {
            "thisWeek": {
                "label": "This Week",
                "expiryCandidates": [{"minDTE": 0, "maxDTE": 5}],
                "earningsWeekExpiryCandidates": [{"minDTE": 0, "maxDTE": 3}]
            },
            "thisMonth": {
                "label": "This Month",
                "expiryCandidates": [{"minDTE": 7, "maxDTE": 30}],
                "earningsWeekExpiryCandidates": [{"minDTE": 5, "maxDTE": 14}]
            },
            "thisYear": {
                "label": "This Year",
                "expiryCandidates": [{"minDTE": 30, "maxDTE": 365}],
                "earningsWeekExpiryCandidates": [{"minDTE": 14, "maxDTE": 90}]
            }
        },
        "expiryCandidates": [{"minDTE": 7, "maxDTE": 10}],
        "earningsWeekExpiryCandidates": [{"minDTE": 5, "maxDTE": 7}],
        "deltaTarget": {"min": 0.08, "max": 0.12},
        "deltaConservative": {"min": 0.05, "max": 0.08},
        "widthCandidates": [3, 5, 10],
        "wallBufferPct": {
            "lowRisk": 0.007,
            "midRisk": 0.010,
            "highRisk": 0.014
        },
        "liquidityFilters": {
            "minOI": 500,
            "minVolume": 50,
            "maxSpreadPct": 0.15
        },
        "riskEnvelope": {
            "maxLossPerIdea": 1000.0,
            "minCredit": 0.20
        },
        "scoring": {
            "wROC": 0.30,
            "wPOP": 0.25,
            "wWallDist": 0.20,
            "wEV": 0.15,
            "wLiq": 0.10
        },
        "multipliers": {
            "positiveGexHighPin": 1.15,
            "neutral": 1.00
        },
        "blocks": {
            "blockShortPremiumIfGexNegative": True,
            "blockIfActionFilterBlocked": True,
            "blockOvernightIfEarningsOrMacro": True
        },
        "maxIdeasToShow": 3
    }
    
    if os.path.exists(config_path):
        try:
            with open(config_path, 'r') as f:
                file_config = json.load(f)
                # Merge with defaults
                defaults.update(file_config)
                # Check for symbol-specific overrides
                if symbol in file_config.get("symbolOverrides", {}):
                    symbol_config = file_config["symbolOverrides"][symbol]
                    defaults.update(symbol_config)
        except Exception as e:
            logger.warning(f"Failed to load config from {config_path}: {e}")
    
    return defaults


# ===============================
# Options Chain Enhancement
# ===============================

def calculate_dte(expiry_str: str) -> int:
    """Calculate days to expiry from YYYY-MM-DD string"""
    try:
        exp_date = datetime.strptime(expiry_str, "%Y-%m-%d").date()
        today = date.today()
        dte = (exp_date - today).days
        return max(0, dte)
    except:
        return 0


def is_market_open() -> Tuple[bool, str]:
    """
    Check if US stock market is currently open.
    
    Returns:
        (is_open, status_note)
        is_open: True if market is open
        status_note: Human-readable status message
    """
    from pytz import timezone
    
    try:
        # Get current time in US/Eastern
        eastern = timezone('US/Eastern')
        now_et = datetime.now(eastern)
        
        # Check if it's a weekday (Monday=0, Friday=4)
        day_of_week = now_et.weekday()
        if day_of_week >= 5:  # Saturday or Sunday
            if day_of_week == 5:  # Saturday
                return False, "Market closed (Saturday). Ideas pending Monday open."
            else:  # Sunday
                return False, "Market closed (Sunday). Ideas pending Monday open."
        
        # Check market hours (9:30 AM - 4:00 PM ET)
        market_open = now_et.replace(hour=9, minute=30, second=0, microsecond=0)
        market_close = now_et.replace(hour=16, minute=0, second=0, microsecond=0)
        
        if now_et < market_open:
            return False, "Pre-market. Ideas pending 9:30 AM ET open."
        elif now_et > market_close:
            return False, "After hours. Ideas pending next market open."
        else:
            return True, "Market open"
    
    except Exception as e:
        # If timezone check fails, assume market might be open
        logger.warning(f"Could not determine market status: {e}")
        return True, "Market status unknown"


def enhance_contract_with_pricing(contract: Dict, spot: float, api_item: Dict) -> Optional[OptionQuote]:
    """
    Enhance a parsed contract with bid/ask/delta from API response.
    
    Args:
        contract: Parsed contract from parse_contract()
        spot: Current spot price
        api_item: Raw API response item
    
    Returns:
        OptionQuote or None if invalid
    """
    # Extract bid/ask from API
    day_data = api_item.get("day", {})
    bid = day_data.get("bid")
    ask = day_data.get("ask")
    
    # Fallback to last price if bid/ask missing
    if bid is None or ask is None:
        last_price = day_data.get("close")
        if last_price:
            # Use last price with small spread estimate
            mid = float(last_price)
            spread_est = mid * 0.01  # 1% estimate
            bid = mid - spread_est / 2
            ask = mid + spread_est / 2
        else:
            return None
    
    bid = float(bid) if bid is not None else 0.0
    ask = float(ask) if ask is not None else 0.0
    mid = (bid + ask) / 2.0 if (bid > 0 and ask > 0) else 0.0
    
    if mid <= 0:
        return None
    
    # Extract delta from greeks or calculate
    greeks = api_item.get("greeks", {})
    delta = greeks.get("delta")
    
    if delta is None:
        # Calculate delta using Black-Scholes
        iv = contract.get("iv", 0.0)
        T = contract.get("T", 0.0)
        strike = contract.get("strike", 0.0)
        option_type = contract.get("type", "call")
        
        if iv > 0 and T > 0 and strike > 0:
            delta = bs_delta(spot, strike, T, iv, option_type)
        else:
            return None
    
    delta = float(delta)
    
    # Extract other fields
    volume = day_data.get("volume", 0) or 0
    oi = api_item.get("open_interest", 0) or 0
    
    # Calculate DTE
    expiry_str = contract.get("expiry_str", "")
    if not expiry_str:
        return None
    dte = calculate_dte(expiry_str)
    
    return OptionQuote(
        expiry=expiry_str,
        dte=dte,
        type=contract["type"].upper(),
        strike=contract["strike"],
        bid=bid,
        ask=ask,
        mid=mid,
        delta=delta,
        iv=contract.get("iv"),
        openInterest=int(oi),
        volume=int(volume),
        bidAskSpread=ask - bid
    )


def build_strike_index(quotes: List[OptionQuote]) -> Dict[Tuple[str, float], OptionQuote]:
    """
    Build O(1) lookup index: (expiry, strike) -> OptionQuote
    
    Args:
        quotes: List of OptionQuote
    
    Returns:
        Dict mapping (expiry, strike) -> OptionQuote
    """
    index = {}
    for quote in quotes:
        key = (quote.expiry, quote.strike)
        index[key] = quote
    return index


# ===============================
# Permissioning Logic
# ===============================

def check_global_blocks(context: MarketContext, config: Dict[str, Any]) -> Tuple[bool, Optional[str]]:
    """
    Enhanced global blocks with flip line and DePin logic.
    
    Returns:
        (is_blocked, reason)
    """
    blocks = config.get("blocks", {})
    
    # Check action filter
    if blocks.get("blockIfActionFilterBlocked", True):
        if context.actionFilter in ["BLOCKED", "WAIT_FOR_CLARITY"]:
            return True, f"Action filter is {context.actionFilter}"
    
    # Check flip line availability
    spot_above_flip = is_spot_above_flip_line(context, config)
    if spot_above_flip is None and context.flipLine is None:
        # Flip line unavailable - conservative check
        if context.gexState != "POSITIVE":
            return True, "Flip line unavailable and GEX not positive"
        if is_depin_risk_elevated(context, config):
            return True, "Flip line unavailable and DePin risk elevated"
    
    # Check DePin risk confirmation
    if is_depin_risk_confirmed(context, config):
        return True, f"DePin risk confirmed (risk={context.depinRisk})"
    
    # Check negative GEX + flip line position
    if blocks.get("blockShortPremiumIfGexNegative", True):
        if context.gexState == "NEGATIVE":
            # If spot is below flip line, negative GEX is expected, but still block blind short premium
            # If spot is above or at flip line, block short premium
            if spot_above_flip is not False:  # Not below flip line
                return True, "GEX is negative - short premium blocked"
    
    # Check depinRisk availability
    if context.depinRisk is None:
        # Conservative: block if missing
        return True, "De-pin risk data unavailable"
    
    return False, None


def determine_mode(context: MarketContext, config: Dict[str, Any]) -> str:
    """
    Determine if ideas should be INTRADAY_ONLY, OVERNIGHT_ALLOWED, or HEDGE_REQUIRED.
    
    Returns:
        "INTRADAY_ONLY" | "OVERNIGHT_ALLOWED" | "HEDGE_REQUIRED"
    """
    blocks = config.get("blocks", {})
    
    # Event risk: Hard constraint - intraday only
    if blocks.get("blockOvernightIfEarningsOrMacro", True):
        if context.macroEventNext24h or context.megaCapEarningsNext24h:
            return "INTRADAY_ONLY"
        if context.earningsClusterScore >= 3:  # Threshold
            return "INTRADAY_ONLY"
    
    # DePin risk: Elevated but not confirmed - require hedge for overnight
    if is_depin_risk_elevated(context, config):
        if not is_depin_risk_confirmed(context, config):
            return "HEDGE_REQUIRED"
        else:
            # Confirmed should be blocked, but return INTRADAY_ONLY as safety
            return "INTRADAY_ONLY"
    
    # Below flip line: Require hedge for overnight
    spot_above_flip = is_spot_above_flip_line(context, config)
    if spot_above_flip is False:
        return "HEDGE_REQUIRED"
    
    return "OVERNIGHT_ALLOWED"


# ===============================
# Flip Line and DePin Helpers
# ===============================

def is_spot_above_flip_line(context: MarketContext, config: Optional[Dict[str, Any]] = None) -> Optional[bool]:
    """
    Determine if spot is above flip line.
    
    Uses tolerance from config to handle cases where spot is very close to flip line.
    
    Args:
        context: MarketContext with spot and flipLine
        config: Optional config dict (for tolerance override)
    
    Returns:
        True if spot > flipLine (with tolerance), False if below, None if unavailable
    """
    if context.flipLine is None:
        return None
    
    # Get tolerance from config if provided
    if config:
        tolerance_pct = config.get("flipLineTolerance", 0.002)
    else:
        tolerance_pct = 0.002  # Default 0.2% tolerance
    
    tolerance_abs = context.spot * tolerance_pct
    
    if context.spot > (context.flipLine + tolerance_abs):
        return True
    elif context.spot < (context.flipLine - tolerance_abs):
        return False
    else:
        # Very close to flip line - treat as "at" flip line, return None
        return None


def is_depin_risk_confirmed(context: MarketContext, config: Dict[str, Any]) -> bool:
    """
    Check if DePin risk is high and confirmed.
    
    Returns:
        True if depinRisk >= confirmed threshold (default 70)
    """
    if context.depinRisk is None:
        return False
    
    threshold_config = config.get("depinRiskThreshold", {})
    threshold = threshold_config.get("confirmed", 70)
    
    return context.depinRisk >= threshold


def is_depin_risk_elevated(context: MarketContext, config: Dict[str, Any]) -> bool:
    """
    Check if DePin risk is elevated but not necessarily confirmed.
    
    Returns:
        True if depinRisk >= elevated threshold (default 50)
    """
    if context.depinRisk is None:
        return False
    
    threshold_config = config.get("depinRiskThreshold", {})
    threshold = threshold_config.get("elevated", 50)
    
    return context.depinRisk >= threshold


# ===============================
# Strategy Filtering
# ===============================

def get_allowed_strategies(context: MarketContext, config: Dict[str, Any]) -> List[str]:
    """
    Determine which strategies are allowed based on flip line and regime.
    
    Philosophy:
    - Above flip line: Favor time-decay (credit spreads) if no event risk
    - Below flip line: Disable blind short premium, allow only hedged structures
    - Negative GEX: Disable time-decay unless hedged
    - High DePin: Conservative intraday-only time-decay
    
    Returns:
        List of allowed strategy types
    """
    base_strategies = config.get("baseStrategySet", ["PUT_CREDIT_SPREAD", "CALL_CREDIT_SPREAD"])
    allowed = []
    
    # Check flip line position
    spot_above_flip = is_spot_above_flip_line(context, config)
    depin_confirmed = is_depin_risk_confirmed(context, config)
    depin_elevated = is_depin_risk_elevated(context, config)
    
    # Strategy preference logic
    if spot_above_flip is True:
        # Above flip line: Favor time-decay strategies
        if context.gexState == "NEGATIVE":
            # Negative GEX: Block time-decay unless hedged
            # For now, we don't have hedged structures, so block
            logger.info("Above flip line but GEX negative - blocking time-decay strategies")
            return []
        
        if depin_confirmed:
            # High DePin risk: Block time-decay
            logger.info("Above flip line but DePin risk confirmed - blocking time-decay strategies")
            return []
        
        # Above flip + positive/neutral GEX + no major DePin: Allow time-decay
        if "PUT_CREDIT_SPREAD" in base_strategies:
            allowed.append("PUT_CREDIT_SPREAD")
        if "CALL_CREDIT_SPREAD" in base_strategies:
            allowed.append("CALL_CREDIT_SPREAD")
    
    elif spot_above_flip is False:
        # Below flip line: Disable blind short premium
        # Only allow hedged structures (iron condors, butterflies)
        hedged_strategies = []
        
        # Iron Condor: Always allowed below flip line
        hedged_strategies.append("IRON_CONDOR")
        
        # Butterfly: Only if pin confidence is HIGH
        if context.pinConfidence == "HIGH":
            hedged_strategies.append("BUTTERFLY")
        
        logger.info(f"Below flip line - allowing hedged structures: {hedged_strategies}")
        return hedged_strategies
    
    else:
        # Flip line unavailable or very close: Conservative approach
        # Only allow if GEX is positive and DePin is low
        if context.gexState == "POSITIVE" and not depin_elevated:
            if "PUT_CREDIT_SPREAD" in base_strategies:
                allowed.append("PUT_CREDIT_SPREAD")
            if "CALL_CREDIT_SPREAD" in base_strategies:
                allowed.append("CALL_CREDIT_SPREAD")
        else:
            logger.info(f"Flip line unavailable/close - GEX: {context.gexState}, DePin elevated: {depin_elevated} - blocking strategies")
    
    return allowed


# ===============================
# Candidate Generation
# ===============================

def filter_expiry_candidates(
    quotes: List[OptionQuote],
    context: MarketContext,
    config: Dict[str, Any]
) -> List[OptionQuote]:
    """
    Filter quotes by expiry based on earnings/macro events and day of week.
    
    Handles end-of-week cases:
    - Friday (day 5): Prefer 0DTE (today)
    - Saturday/Sunday (day 6-7): Prefer next week's options (1-5 DTE from Monday)
    - If preferred range has no options, step back/forward to find available expiries
    
    NOTE: Weekend logic only applies if expiryCandidates are narrow (default 7-10 DTE).
    If a specific wide timeframe is requested (e.g., 7-30 DTE for "thisMonth"), 
    that range is respected regardless of day of week.
    
    DEBUG: Set config["debug"]["bypassExpiryFilter"] = true to return all quotes.
    
    Returns:
        Filtered quotes
    """
    from datetime import datetime, timedelta
    
    # Debug toggle: bypass all expiry filtering
    debug_config = config.get("debug", {})
    if debug_config.get("bypassExpiryFilter", False):
        logger.warning("⚠️ DEBUG MODE: Bypassing expiry filter - returning all quotes")
        return quotes
    
    # Get current day of week (0=Monday, 4=Friday, 5=Saturday, 6=Sunday)
    now = datetime.now()
    day_of_week = now.weekday()
    
    # Choose base expiry range based on events
    if context.macroEventNext24h or context.megaCapEarningsNext24h or context.earningsClusterScore >= 3:
        base_ranges = config.get("earningsWeekExpiryCandidates", [{"minDTE": 5, "maxDTE": 7}])
    else:
        base_ranges = config.get("expiryCandidates", [{"minDTE": 7, "maxDTE": 10}])
    
    # Check if the requested range is a "wide" timeframe (e.g., 7-30, 30-365)
    # If so, don't apply weekend logic - respect the requested range
    is_wide_timeframe = False
    if base_ranges:
        first_range = base_ranges[0]
        range_width = first_range.get("maxDTE", 0) - first_range.get("minDTE", 0)
        # If range is wider than 5 days, treat it as a specific timeframe request
        # This means "thisMonth" (7-30, width=23) or "thisYear" (30-365, width=335) should be respected
        if range_width > 5:
            is_wide_timeframe = True
            logger.info(f"Wide timeframe detected (width={range_width} days). Respecting requested range: {first_range}")
    
    # Adjust for end-of-week/weekend (only if NOT a wide timeframe)
    adjusted_ranges = []
    
    if not is_wide_timeframe and day_of_week >= 4:  # Friday (4), Saturday (5), or Sunday (6)
        if day_of_week == 4:  # Friday
            # Prefer 0DTE (today) for portfolio protection
            adjusted_ranges.append({"minDTE": 0, "maxDTE": 0})
            # Fallback to next week if 0DTE not available
            adjusted_ranges.append({"minDTE": 1, "maxDTE": 5})
        else:  # Saturday (5) or Sunday (6)
            # Next week's options (Monday-Friday)
            # From Monday, that's 1-5 trading days
            adjusted_ranges.append({"minDTE": 1, "maxDTE": 5})
            # Also allow 0DTE if market is open (unlikely but possible)
            adjusted_ranges.append({"minDTE": 0, "maxDTE": 0})
        
        # Add original ranges as fallback
        adjusted_ranges.extend(base_ranges)
    else:
        # Normal weekday (Monday-Thursday) OR wide timeframe requested - use original ranges
        adjusted_ranges = base_ranges
    
    # Filter quotes by the ranges
    filtered = []
    
    # Try each range in order
    for range_def in adjusted_ranges:
        min_dte = range_def.get("minDTE", 0)
        max_dte = range_def.get("maxDTE", 365)
        for quote in quotes:
            if min_dte <= quote.dte <= max_dte:
                filtered.append(quote)
        
        # If this is a wide timeframe, return immediately when we find matches
        # Otherwise, continue to try other ranges
        if filtered and is_wide_timeframe:
            logger.info(f"Found {len(filtered)} quotes in wide timeframe range {min_dte}-{max_dte} DTE")
            return filtered
    
    # If we found matches in any range, return them
    if filtered:
        return filtered
    
    # If no quotes found in preferred ranges, find available expiries and use closest
    if not filtered and quotes:
        # Get unique DTE values from available quotes
        available_dtes = sorted(set(q.dte for q in quotes if q.dte >= 0))
        
        if available_dtes:
            # Find the closest DTE to our preferred range
            preferred_min = base_ranges[0].get("minDTE", 7) if base_ranges else 7
            preferred_max = base_ranges[0].get("maxDTE", 10) if base_ranges else 10
            preferred_center = (preferred_min + preferred_max) / 2
            
            # Find closest available DTE
            closest_dte = min(available_dtes, key=lambda x: abs(x - preferred_center))
            
            # Use a small window around the closest DTE (±2 days)
            window_min = max(0, closest_dte - 2)
            window_max = closest_dte + 2
            
            logger.info(f"No quotes in preferred DTE range. Using closest available: {closest_dte} DTE (window: {window_min}-{window_max})")
            
            # Filter with the adjusted window
            for quote in quotes:
                if window_min <= quote.dte <= window_max:
                    filtered.append(quote)
    
    return filtered


def get_wall_buffer(context: MarketContext, config: Dict[str, Any]) -> float:
    """
    Get wall buffer percentage based on depinRisk.
    
    Returns:
        Buffer as percentage (e.g., 0.007 for 0.7%)
    """
    depin_risk = context.depinRisk or 50
    wall_buffer_pct = config.get("wallBufferPct", {})
    
    if depin_risk <= 40:
        return wall_buffer_pct.get("lowRisk", 0.007)
    elif depin_risk <= 60:
        return wall_buffer_pct.get("midRisk", 0.010)
    else:
        return wall_buffer_pct.get("highRisk", 0.014)


def generate_put_credit_candidates(
    quotes: List[OptionQuote],
    context: MarketContext,
    config: Dict[str, Any]
) -> List[OptionQuote]:
    """
    Generate put credit spread short strike candidates.
    
    Filters by:
    - Delta range
    - Liquidity (OI, volume, spread)
    - Wall buffer (shortStrike <= putWall - buffer)
    
    Returns:
        List of candidate OptionQuote for short leg
    """
    # Debug toggle: bypass wall requirement
    debug_config = config.get("debug", {})
    bypass_wall_buffer = debug_config.get("bypassWallBufferFilter", False)
    
    if context.putWall is None and not bypass_wall_buffer:
        logger.warning("⚠️ No putWall available. Set bypassWallBufferFilter=true to generate candidates anyway.")
        return []
    
    if context.putWall is None and bypass_wall_buffer:
        logger.warning("⚠️ DEBUG: Bypassing putWall requirement - generating candidates without wall constraint")
    
    delta_target = config.get("deltaTarget", {"min": 0.08, "max": 0.12})
    delta_min = delta_target.get("min", 0.08)
    delta_max = delta_target.get("max", 0.12)
    
    liquidity = config.get("liquidityFilters", {})
    min_oi = liquidity.get("minOI", 500)
    min_volume = liquidity.get("minVolume", 50)
    max_spread_pct = liquidity.get("maxSpreadPct", 0.15)
    
    # Debug toggles
    debug_config = config.get("debug", {})
    bypass_liquidity = debug_config.get("bypassLiquidityFilter", False)
    bypass_wall_buffer = debug_config.get("bypassWallBufferFilter", False)
    
    # Calculate max strike (putWall - buffer, or infinity if bypassed or no wall)
    if bypass_wall_buffer or context.putWall is None:
        max_strike = float('inf')
    else:
        buffer_pct = get_wall_buffer(context, config)
        buffer_abs = context.spot * buffer_pct
        max_strike = context.putWall - buffer_abs
    
    candidates = []
    
    # Debug toggle: bypass delta filter
    bypass_delta = debug_config.get("bypassDeltaFilter", False)
    
    for quote in quotes:
        if quote.type != "PUT":
            continue
        
        # Delta filter (unless bypassed)
        if not bypass_delta:
            if quote.delta is None or not (delta_min <= abs(quote.delta) <= delta_max):
                continue
        
        # Strike must be below putWall - buffer (unless bypassed)
        if not bypass_wall_buffer and quote.strike > max_strike:
            continue
        
        # Liquidity filters (unless bypassed)
        if not bypass_liquidity:
            spread_pct = quote.bidAskSpread / quote.mid if quote.mid > 0 else 1.0
            if spread_pct > max_spread_pct:
                continue
            
            if quote.openInterest < min_oi and quote.volume < min_volume:
                continue
        
        candidates.append(quote)
    
    return candidates


def generate_call_credit_candidates(
    quotes: List[OptionQuote],
    context: MarketContext,
    config: Dict[str, Any]
) -> List[OptionQuote]:
    """
    Generate call credit spread short strike candidates.
    
    Filters by:
    - Delta range
    - Liquidity
    - Wall buffer (shortStrike >= callWall + buffer)
    
    Returns:
        List of candidate OptionQuote for short leg
    """
    # Debug toggle: bypass wall requirement
    debug_config = config.get("debug", {})
    bypass_wall_buffer = debug_config.get("bypassWallBufferFilter", False)
    
    if context.callWall is None and not bypass_wall_buffer:
        logger.warning("⚠️ No callWall available. Set bypassWallBufferFilter=true to generate candidates anyway.")
        return []
    
    if context.callWall is None and bypass_wall_buffer:
        logger.warning("⚠️ DEBUG: Bypassing callWall requirement - generating candidates without wall constraint")
    
    delta_target = config.get("deltaTarget", {"min": 0.08, "max": 0.12})
    delta_min = delta_target.get("min", 0.08)
    delta_max = delta_target.get("max", 0.12)
    
    liquidity = config.get("liquidityFilters", {})
    min_oi = liquidity.get("minOI", 500)
    min_volume = liquidity.get("minVolume", 50)
    max_spread_pct = liquidity.get("maxSpreadPct", 0.15)
    
    # Debug toggles
    debug_config = config.get("debug", {})
    bypass_liquidity = debug_config.get("bypassLiquidityFilter", False)
    bypass_wall_buffer = debug_config.get("bypassWallBufferFilter", False)
    
    # Calculate min strike (callWall + buffer, or 0 if bypassed or no wall)
    if bypass_wall_buffer or context.callWall is None:
        min_strike = 0.0
    else:
        buffer_pct = get_wall_buffer(context, config)
        buffer_abs = context.spot * buffer_pct
        min_strike = context.callWall + buffer_abs
    
    candidates = []
    
    # Debug toggle: bypass delta filter
    bypass_delta = debug_config.get("bypassDeltaFilter", False)
    
    for quote in quotes:
        if quote.type != "CALL":
            continue
        
        # Delta filter (for calls, delta is positive) (unless bypassed)
        if not bypass_delta:
            if quote.delta is None or not (delta_min <= quote.delta <= delta_max):
                continue
        
        # Strike must be above callWall + buffer (unless bypassed)
        if not bypass_wall_buffer and quote.strike < min_strike:
            continue
        
        # Liquidity filters (unless bypassed)
        if not bypass_liquidity:
            spread_pct = quote.bidAskSpread / quote.mid if quote.mid > 0 else 1.0
            if spread_pct > max_spread_pct:
                continue
            
            if quote.openInterest < min_oi and quote.volume < min_volume:
                continue
        
        candidates.append(quote)
    
    return candidates


# ===============================
# Metrics Calculation
# ===============================

def calculate_spread_metrics(
    short_quote: OptionQuote,
    long_quote: OptionQuote,
    width: float
) -> Dict[str, float]:
    """
    Calculate spread metrics.
    
    Returns:
        Dict with netCredit, maxProfit, maxLoss, roc, breakeven
    """
    net_credit = short_quote.mid - long_quote.mid
    max_profit = net_credit * 100.0  # Per contract
    max_loss = (width - net_credit) * 100.0
    roc = max_profit / max_loss if max_loss > 0 else 0.0
    
    # Breakeven
    if short_quote.type == "PUT":
        breakeven = short_quote.strike - net_credit
    else:  # CALL
        breakeven = short_quote.strike + net_credit
    
    return {
        "netCredit": net_credit,
        "maxProfit": max_profit,
        "maxLoss": max_loss,
        "roc": roc,
        "breakeven": breakeven,
        "width": width
    }


def calculate_pop_proxy(delta_short: float) -> float:
    """
    Proxy for Probability of Profit: 1 - abs(deltaShort)
    
    Args:
        delta_short: Delta of short leg
    
    Returns:
        POP estimate (0-1)
    """
    return max(0.0, min(1.0, 1.0 - abs(delta_short)))


def calculate_ev_proxy(pop: float, max_profit: float, max_loss: float) -> float:
    """
    Proxy for Expected Value: pop*maxProfit - (1-pop)*maxLoss
    
    Returns:
        EV in dollars
    """
    return pop * max_profit - (1.0 - pop) * max_loss


def calculate_iron_condor_metrics(
    put_short: OptionQuote,
    put_long: OptionQuote,
    call_short: OptionQuote,
    call_long: OptionQuote
) -> Dict[str, float]:
    """
    Calculate Iron Condor metrics.
    
    Returns:
        Dict with netCredit, maxProfit, maxLoss, roc, breakevenLower, breakevenUpper
    """
    # Calculate put spread credit
    put_credit = put_short.mid - put_long.mid
    
    # Calculate call spread credit
    call_credit = call_short.mid - call_long.mid
    
    # Total net credit
    net_credit = put_credit + call_credit
    
    # Widths
    put_width = put_short.strike - put_long.strike
    call_width = call_short.strike - call_long.strike
    
    # Max loss is width of wider wing minus net credit
    max_width = max(put_width, call_width)
    max_loss = (max_width - net_credit) * 100.0
    
    # Max profit is net credit (if price stays between short strikes)
    max_profit = net_credit * 100.0
    
    # ROC
    roc = max_profit / max_loss if max_loss > 0 else 0.0
    
    # Breakevens
    breakeven_lower = put_short.strike - net_credit
    breakeven_upper = call_short.strike + net_credit
    
    return {
        "netCredit": net_credit,
        "maxProfit": max_profit,
        "maxLoss": max_loss,
        "roc": roc,
        "breakevenLower": breakeven_lower,
        "breakevenUpper": breakeven_upper,
        "putWidth": put_width,
        "callWidth": call_width,
        "profitZone": breakeven_upper - breakeven_lower
    }


def calculate_butterfly_metrics(
    lower: OptionQuote,
    middle1: OptionQuote,
    middle2: OptionQuote,
    upper: OptionQuote,
    butterfly_type: str
) -> Dict[str, float]:
    """
    Calculate Butterfly metrics.
    
    Args:
        butterfly_type: "PUT" or "CALL"
        middle1 and middle2 are same strike (2 contracts)
    
    Returns:
        Dict with netDebit, maxProfit, maxLoss, roc, breakevenLower, breakevenUpper
    """
    # Butterfly: Buy lower, Sell 2 middle, Buy upper
    # Net debit = (lower.mid + upper.mid) - (2 * middle1.mid)
    net_debit = (lower.mid + upper.mid) - (2 * middle1.mid)
    
    # Wing width
    wing_width = middle1.strike - lower.strike  # Should equal upper - middle
    
    # Max profit at middle strike
    # Profit = (2 * middle.mid) - (lower.mid + upper.mid) = -net_debit
    # But we need to account for intrinsic value at expiration
    if butterfly_type == "PUT":
        # Max profit if price = middle strike at expiration
        # Lower and upper expire worthless, middle expires ITM
        # But we sold 2 and bought 1 each side, so max profit is wing_width - net_debit
        max_profit = (wing_width - net_debit) * 100.0
    else:  # CALL
        max_profit = (wing_width - net_debit) * 100.0
    
    # Max loss is net debit (if price moves outside wings)
    max_loss = abs(net_debit) * 100.0 if net_debit > 0 else 0.0
    
    # ROC
    roc = max_profit / max_loss if max_loss > 0 else 0.0
    
    # Breakevens
    if butterfly_type == "PUT":
        breakeven_lower = lower.strike + net_debit
        breakeven_upper = upper.strike - net_debit
    else:  # CALL
        breakeven_lower = lower.strike + net_debit
        breakeven_upper = upper.strike - net_debit
    
    return {
        "netDebit": net_debit,
        "maxProfit": max_profit,
        "maxLoss": max_loss,
        "roc": roc,
        "breakevenLower": breakeven_lower,
        "breakevenUpper": breakeven_upper,
        "wingWidth": wing_width,
        "middleStrike": middle1.strike
    }


def calculate_iron_condor_pop(
    put_short_delta: Optional[float],
    call_short_delta: Optional[float],
    spot: float,
    breakeven_lower: float,
    breakeven_upper: float
) -> float:
    """
    Calculate Probability of Profit for Iron Condor.
    
    Probability that price stays between breakevens at expiration.
    """
    if put_short_delta is None or call_short_delta is None:
        # Fallback: use breakeven range
        range_pct = (breakeven_upper - breakeven_lower) / spot
        # Rough estimate: wider range = higher POP
        return min(0.8, max(0.3, 1.0 - range_pct * 2))
    
    # Use delta-based proxy
    # POP is roughly 1 - (abs(put_delta) + abs(call_delta))
    # But this is conservative, so adjust upward
    delta_sum = abs(put_short_delta) + abs(call_short_delta)
    pop = max(0.0, min(1.0, 1.0 - delta_sum * 0.8))
    
    return pop


def calculate_butterfly_pop(
    middle_strike: float,
    spot: float,
    breakeven_lower: float,
    breakeven_upper: float
) -> float:
    """
    Calculate Probability of Profit for Butterfly.
    
    Probability that price is near middle strike at expiration.
    """
    # Butterflies profit if price is at middle strike
    # Use normal distribution approximation
    # Rough estimate: tighter range = lower POP but higher max profit
    
    # Distance from spot to middle
    distance_to_middle = abs(spot - middle_strike)
    
    # Range between breakevens
    profit_range = breakeven_upper - breakeven_lower
    
    # POP is roughly proportional to profit range / current volatility
    # Conservative estimate: 30-50% for butterflies
    if profit_range > 0:
        pop = min(0.5, max(0.2, profit_range / (spot * 0.1)))  # Assume ~10% vol
    else:
        pop = 0.3
    
    return pop


def calculate_liquidity_score(quote: OptionQuote, config: Dict[str, Any]) -> float:
    """
    Calculate liquidity score (0-1).
    
    Composite of spread, OI, volume.
    """
    liquidity = config.get("liquidityFilters", {})
    max_spread_pct = liquidity.get("maxSpreadPct", 0.15)
    target_oi = liquidity.get("minOI", 500) * 2  # Target is 2x minimum
    target_vol = liquidity.get("minVolume", 50) * 2
    
    # Spread score (lower spread = higher score)
    spread_pct = quote.bidAskSpread / quote.mid if quote.mid > 0 else 1.0
    spread_score = max(0.0, min(1.0, 1.0 - (spread_pct / max_spread_pct)))
    
    # OI score (log scale)
    oi_score = min(1.0, math.log(1 + quote.openInterest) / math.log(1 + target_oi))
    
    # Volume score (log scale)
    vol_score = min(1.0, math.log(1 + quote.volume) / math.log(1 + target_vol))
    
    # Weighted composite
    return 0.5 * spread_score + 0.3 * oi_score + 0.2 * vol_score


def calculate_wall_distance_score(
    short_strike: float,
    wall: float,
    spot: float,
    buffer_pct: float
) -> float:
    """
    Calculate wall distance score (0-1).
    
    Higher score = farther from wall (better).
    """
    if wall is None:
        return 0.5  # Neutral if no wall
    
    if short_strike <= spot:  # Put
        distance = wall - short_strike
    else:  # Call
        distance = short_strike - wall
    
    target_distance = spot * buffer_pct
    score = min(1.0, distance / target_distance) if target_distance > 0 else 0.5
    
    return score


# ===============================
# Hedge Structure Generation
# ===============================

def generate_iron_condor_candidates(
    quotes: List[OptionQuote],
    context: MarketContext,
    config: Dict[str, Any],
    strike_index: Dict[Tuple[str, float], OptionQuote]
) -> List[Tuple[OptionQuote, OptionQuote, OptionQuote, OptionQuote]]:
    """
    Generate Iron Condor candidates by combining put and call credit spreads.
    
    Iron Condor: Sell OTM put spread + Sell OTM call spread
    - Put spread below spot
    - Call spread above spot
    - Both spreads OTM
    - Same expiry
    
    Returns:
        List of (put_short, put_long, call_short, call_long) tuples
    """
    # Get put credit spread candidates
    put_candidates = generate_put_credit_candidates(quotes, context, config)
    
    # Get call credit spread candidates
    call_candidates = generate_call_credit_candidates(quotes, context, config)
    
    if not put_candidates or not call_candidates:
        return []
    
    iron_condors = []
    width_candidates = config.get("widthCandidates", [3, 5, 10])
    iron_condor_config = config.get("ironCondor", {})
    symmetric_wings = iron_condor_config.get("symmetricWings", True)
    
    for put_short in put_candidates:
        for call_short in call_candidates:
            # Must be same expiry
            if put_short.expiry != call_short.expiry:
                continue
            
            # Both must be OTM
            if put_short.strike >= context.spot or call_short.strike <= context.spot:
                continue
            
            # Find long legs for put spread
            for put_width in width_candidates:
                put_long_strike = put_short.strike - put_width
                put_long_key = (put_short.expiry, put_long_strike)
                put_long = strike_index.get(put_long_key)
                
                if put_long is None or put_long.type != "PUT":
                    continue
                
                # Find long legs for call spread
                for call_width in width_candidates:
                    # Prefer symmetric wings
                    if symmetric_wings and put_width != call_width:
                        continue
                    
                    call_long_strike = call_short.strike + call_width
                    call_long_key = (call_short.expiry, call_long_strike)
                    call_long = strike_index.get(call_long_key)
                    
                    if call_long is None or call_long.type != "CALL":
                        continue
                    
                    # Check liquidity on all legs
                    min_oi = config.get("liquidityFilters", {}).get("minOI", 500)
                    min_vol = config.get("liquidityFilters", {}).get("minVolume", 50)
                    
                    if all([
                        put_short.openInterest >= min_oi,
                        put_long.openInterest >= min_oi,
                        call_short.openInterest >= min_oi,
                        call_long.openInterest >= min_oi,
                        put_short.volume >= min_vol,
                        call_short.volume >= min_vol,
                    ]):
                        iron_condors.append((put_short, put_long, call_short, call_long))
    
    return iron_condors


def generate_butterfly_candidates(
    quotes: List[OptionQuote],
    context: MarketContext,
    config: Dict[str, Any],
    butterfly_type: str,
    strike_index: Dict[Tuple[str, float], OptionQuote]
) -> List[Tuple[OptionQuote, OptionQuote, OptionQuote, OptionQuote]]:
    """
    Generate Butterfly candidates.
    
    Butterfly: Buy 1 OTM, Sell 2 ATM, Buy 1 OTM
    - Middle strike near spot (ATM)
    - Symmetric wings
    - Same expiry
    
    Args:
        butterfly_type: "PUT" or "CALL"
    
    Returns:
        List of (lower, middle1, middle2, upper) tuples
        Note: middle1 and middle2 are the same strike (2 contracts)
    """
    # Butterflies require high pin confidence
    if context.pinConfidence != "HIGH":
        return []
    
    butterfly_config = config.get("butterfly", {})
    wing_widths = butterfly_config.get("wingWidth", [5, 10])
    
    # Middle strike should be near spot (ATM)
    middle_strike = round(context.spot)
    
    butterflies = []
    
    # Group quotes by expiry
    quotes_by_expiry = defaultdict(list)
    for quote in quotes:
        if quote.type == butterfly_type:
            quotes_by_expiry[quote.expiry].append(quote)
    
    for expiry, expiry_quotes in quotes_by_expiry.items():
        for wing_width in wing_widths:
            lower_strike = middle_strike - wing_width
            upper_strike = middle_strike + wing_width
            
            # Find all 4 legs (middle is 2 contracts at same strike)
            lower_key = (expiry, lower_strike)
            middle_key = (expiry, middle_strike)
            upper_key = (expiry, upper_strike)
            
            lower_leg = strike_index.get(lower_key)
            middle_leg1 = strike_index.get(middle_key)
            middle_leg2 = strike_index.get(middle_key)  # Same strike, will use same quote
            upper_leg = strike_index.get(upper_key)
            
            if not all([lower_leg, middle_leg1, upper_leg]):
                continue
            
            # Check types match
            if (lower_leg.type != butterfly_type or 
                middle_leg1.type != butterfly_type or 
                upper_leg.type != butterfly_type):
                continue
            
            # Check liquidity
            min_oi = config.get("liquidityFilters", {}).get("minOI", 500)
            min_vol = config.get("liquidityFilters", {}).get("minVolume", 50)
            
            # Middle leg needs 2x liquidity (we're selling 2)
            if all([
                lower_leg.openInterest >= min_oi,
                middle_leg1.openInterest >= min_oi * 2,  # Need 2 contracts
                upper_leg.openInterest >= min_oi,
                middle_leg1.volume >= min_vol * 2,
            ]):
                butterflies.append((lower_leg, middle_leg1, middle_leg1, upper_leg))
    
    return butterflies


# ===============================
# Scoring & Ranking
# ===============================

def score_execution(
    execution: ReferenceExecution,
    context: MarketContext,
    config: Dict[str, Any],
    all_executions: List[ReferenceExecution]
) -> float:
    """
    Score an execution using normalized metrics.
    
    Normalizes within the group, then applies weights and regime multiplier.
    """
    if not all_executions:
        return 0.0
    
    # Normalize metrics within group (0-1)
    rocs = [e.roc for e in all_executions]
    pops = [e.pop for e in all_executions]
    evs = [e.ev for e in all_executions]
    wall_dists = [e.wallBufferPct for e in all_executions]
    liqs = [e.liquidityScore for e in all_executions]
    
    def normalize(value, values):
        if not values or max(values) == min(values):
            return 0.5
        return (value - min(values)) / (max(values) - min(values))
    
    roc_norm = normalize(execution.roc, rocs)
    pop_norm = normalize(execution.pop, pops)
    ev_norm = normalize(execution.ev, evs)
    wall_dist_norm = normalize(execution.wallBufferPct, wall_dists)
    liq_norm = normalize(execution.liquidityScore, liqs)
    
    # Apply weights
    scoring = config.get("scoring", {})
    base_score = (
        scoring.get("wROC", 0.30) * roc_norm +
        scoring.get("wPOP", 0.25) * pop_norm +
        scoring.get("wWallDist", 0.20) * wall_dist_norm +
        scoring.get("wEV", 0.15) * ev_norm +
        scoring.get("wLiq", 0.10) * liq_norm
    )
    
    # Apply regime multiplier
    multipliers = config.get("multipliers", {})
    if context.gexState == "POSITIVE" and context.pinConfidence == "HIGH" and (context.depinRisk or 100) <= 40:
        multiplier = multipliers.get("positiveGexHighPin", 1.15)
    else:
        multiplier = multipliers.get("neutral", 1.00)
    
    return base_score * multiplier


def rank_and_filter(
    ideas: List[TradeIdea],
    config: Dict[str, Any]
) -> List[TradeIdea]:
    """
    Rank ideas and filter to top N.
    
    Returns:
        Top-ranked ideas
    """
    max_ideas = config.get("maxIdeasToShow", 3)
    
    # Sort by best execution score (descending)
    ideas.sort(key=lambda i: i.best.score if i.best else 0.0, reverse=True)
    
    return ideas[:max_ideas]


# ===============================
# Main Entry Point
# ===============================

def generate_trade_ideas(
    context: MarketContext,
    quotes: List[OptionQuote],
    config: Optional[Dict[str, Any]] = None
) -> List[TradeIdea]:
    """
    Generate trade ideas based on context and options chain.
    
    Args:
        context: Market context from cockpit
        quotes: Enhanced option quotes with pricing
        config: Optional config (loads from file if None)
    
    Returns:
        List of TradeIdea (empty if blocked)
    """
    if config is None:
        config = load_trade_ideas_config(context.symbol)
    
    # Check market status
    market_is_open, market_status_note = is_market_open()
    
    # Check global blocks (but allow ideas even if market is closed - just mark them)
    is_blocked, reason = check_global_blocks(context, config)
    
    # If market is closed, we still generate ideas but mark them as pre-market
    # Only block if there's a real safety issue (not just market closed)
    if is_blocked and market_is_open:
        logger.info(f"Trade ideas blocked: {reason}")
        return []
    
    # If market is closed, we'll generate ideas but mark them with warnings
    if not market_is_open:
        logger.info(f"Market is closed. Generating pre-market ideas: {market_status_note}")
    
    # Determine mode
    mode = determine_mode(context, config)
    
    # Get allowed strategies based on flip line and regime
    allowed_strategies = get_allowed_strategies(context, config)
    
    if not allowed_strategies:
        spot_above_flip = is_spot_above_flip_line(context, config)
        depin_confirmed = is_depin_risk_confirmed(context, config)
        logger.info(f"No strategies allowed. Spot above flip: {spot_above_flip}, GEX: {context.gexState}, DePin confirmed: {depin_confirmed}")
        return []
    
    # Filter by expiry
    filtered_quotes = filter_expiry_candidates(quotes, context, config)
    
    # Log diagnostics
    debug_config = config.get("debug", {})
    if debug_config.get("bypassExpiryFilter", False):
        logger.info(f"⚠️ DEBUG: Expiry filter bypassed. Using {len(filtered_quotes)} quotes (from {len(quotes)} total)")
    else:
        logger.info(f"Filtered to {len(filtered_quotes)} quotes (from {len(quotes)} total) by expiry")
    
    if len(filtered_quotes) == 0:
        logger.warning(f"⚠️ No quotes after expiry filtering. Available DTE values: {sorted(set(q.dte for q in quotes))[:10]}")
        return []
    
    # Build strike index for O(1) lookup
    strike_index = build_strike_index(filtered_quotes)
    
    ideas = []
    strategies = allowed_strategies  # Use filtered strategies, not baseStrategySet
    width_candidates = config.get("widthCandidates", [3, 5, 10])
    risk_env = config.get("riskEnvelope", {})
    min_credit = risk_env.get("minCredit", 0.20)
    max_loss = risk_env.get("maxLossPerIdea", 1000.0)
    
    # Generate PUT_CREDIT_SPREAD ideas
    if "PUT_CREDIT_SPREAD" in strategies:
        put_candidates = generate_put_credit_candidates(filtered_quotes, context, config)
        put_executions = []
        
        for short_quote in put_candidates:
            for width in width_candidates:
                long_strike = short_quote.strike - width
                long_key = (short_quote.expiry, long_strike)
                
                long_quote = strike_index.get(long_key)
                if long_quote is None or long_quote.type != "PUT":
                    continue
                
                # Calculate metrics
                try:
                    metrics = calculate_spread_metrics(short_quote, long_quote, width)
                except Exception as e:
                    logger.warning(f"Error calculating metrics for put spread {short_quote.strike}/{long_quote.strike}: {e}")
                    continue
                
                # Check risk envelope (unless bypassed)
                debug_config = config.get("debug", {})
                bypass_risk = debug_config.get("bypassRiskEnvelope", False)
                if not bypass_risk:
                    if metrics["netCredit"] < min_credit:
                        continue
                    if metrics["maxLoss"] > max_loss:
                        continue
                
                # Calculate additional metrics
                pop = calculate_pop_proxy(short_quote.delta or 0.0)
                ev = calculate_ev_proxy(pop, metrics["maxProfit"], metrics["maxLoss"])
                
                # Filter out negative EV trades
                if ev < 0:
                    continue
                
                liq_score = calculate_liquidity_score(short_quote, config)
                buffer_pct = get_wall_buffer(context, config)
                wall_dist_score = calculate_wall_distance_score(
                    short_quote.strike, context.putWall, context.spot, buffer_pct
                )
                
                # Create execution
                execution = ReferenceExecution(
                    expiry=short_quote.expiry,
                    dte=short_quote.dte,
                    legs=[
                        ExecutionLeg(
                            action="SELL",
                            type="PUT",
                            strike=short_quote.strike,
                            priceMid=short_quote.mid,
                            bid=short_quote.bid,
                            ask=short_quote.ask,
                            delta=short_quote.delta,
                            oi=short_quote.openInterest,
                            volume=short_quote.volume
                        ),
                        ExecutionLeg(
                            action="BUY",
                            type="PUT",
                            strike=long_quote.strike,
                            priceMid=long_quote.mid,
                            bid=long_quote.bid,
                            ask=long_quote.ask,
                            delta=long_quote.delta,
                            oi=long_quote.openInterest,
                            volume=long_quote.volume
                        )
                    ],
                    netCredit=metrics["netCredit"],
                    width=width,
                    maxProfit=metrics["maxProfit"],
                    maxLoss=metrics["maxLoss"],
                    roc=metrics["roc"],
                    breakeven=metrics["breakeven"],
                    pop=pop,
                    ev=ev,
                    wallBufferPct=wall_dist_score,
                    liquidityScore=liq_score,
                    score=0.0  # Will be set after scoring
                )
                
                put_executions.append(execution)
        
        # Score all executions
        for execution in put_executions:
            execution.score = score_execution(execution, context, config, put_executions)
        
        # Sort and take top 1-3
        put_executions.sort(key=lambda e: e.score, reverse=True)
        top_executions = put_executions[:3]
        
        if top_executions:
            best_exec = top_executions[0]
            best_metrics = BestMetrics(
                expiry=best_exec.expiry,
                dte=best_exec.dte,
                netCredit=best_exec.netCredit,
                maxProfit=best_exec.maxProfit,
                maxLoss=best_exec.maxLoss,
                roc=best_exec.roc,
                pop=best_exec.pop,
                ev=best_exec.ev,
                wallBufferPct=best_exec.wallBufferPct,
                liquidityGrade="A" if best_exec.liquidityScore > 0.8 else "B" if best_exec.liquidityScore > 0.6 else "C" if best_exec.liquidityScore > 0.4 else "D",
                score=best_exec.score
            )
            
            # Build reasons
            reasons = []
            if context.gexState == "POSITIVE":
                reasons.append("Positive GEX regime supports premium selling")
            if context.pinConfidence == "HIGH":
                reasons.append("High pin confidence - price likely to stay in range")
            if best_exec.wallBufferPct > 0.7:
                reasons.append(f"Short strike is {best_exec.wallBufferPct*100:.1f}% beyond put wall buffer")
            
            idea = TradeIdea(
                id=f"{context.symbol}_PUT_CREDIT_{datetime.now().isoformat()}",
                symbol=context.symbol,
                asOf=context.asOf,
                strategy="PUT_CREDIT_SPREAD",
                label=f"{context.symbol} Put Credit Spread — {mode.replace('_', ' ').title()}",
                mode=mode,
                confidence="HIGH" if best_exec.score > 0.7 else "MEDIUM" if best_exec.score > 0.5 else "LOW",
                reasons=reasons[:3],
                preMarket=not market_is_open,
                marketStatusNote=market_status_note if not market_is_open else None,
                executions=top_executions,
                best=best_metrics,
                contextSnapshot=ContextSnapshot(
                    spot=context.spot,
                    putWall=context.putWall,
                    callWall=context.callWall,
                    flipLine=context.flipLine,
                    depinRisk=context.depinRisk,
                    gexState=context.gexState,
                    pinConfidence=context.pinConfidence,
                    earningsClusterScore=context.earningsClusterScore,
                    macroEventNext24h=context.macroEventNext24h
                )
            )
            ideas.append(idea)
    
    # Generate CALL_CREDIT_SPREAD ideas (similar logic)
    if "CALL_CREDIT_SPREAD" in strategies:
        call_candidates = generate_call_credit_candidates(filtered_quotes, context, config)
        logger.info(f"Generated {len(call_candidates)} call credit spread candidates (from {len(filtered_quotes)} filtered quotes)")
        call_executions = []
        
        if len(call_candidates) == 0:
            logger.warning(f"⚠️ No call candidates generated. Context: callWall={context.callWall}, spot={context.spot}, quotes={len(filtered_quotes)}")
        
        for short_quote in call_candidates:
            for width in width_candidates:
                long_strike = short_quote.strike + width
                long_key = (short_quote.expiry, long_strike)
                
                long_quote = strike_index.get(long_key)
                if long_quote is None or long_quote.type != "CALL":
                    continue
                
                metrics = calculate_spread_metrics(short_quote, long_quote, width)
                
                # Check risk envelope (unless bypassed)
                debug_config = config.get("debug", {})
                bypass_risk = debug_config.get("bypassRiskEnvelope", False)
                if not bypass_risk:
                    if metrics["netCredit"] < min_credit:
                        continue
                    if metrics["maxLoss"] > max_loss:
                        continue
                
                # Calculate additional metrics
                pop = calculate_pop_proxy(short_quote.delta or 0.0)
                ev = calculate_ev_proxy(pop, metrics["maxProfit"], metrics["maxLoss"])
                
                # Filter out negative EV trades (always applied, no bypass)
                if ev < 0:
                    continue
                
                liq_score = calculate_liquidity_score(short_quote, config)
                buffer_pct = get_wall_buffer(context, config)
                wall_dist_score = calculate_wall_distance_score(
                    short_quote.strike, context.callWall, context.spot, buffer_pct
                )
                
                execution = ReferenceExecution(
                    expiry=short_quote.expiry,
                    dte=short_quote.dte,
                    legs=[
                        ExecutionLeg(
                            action="SELL",
                            type="CALL",
                            strike=short_quote.strike,
                            priceMid=short_quote.mid,
                            bid=short_quote.bid,
                            ask=short_quote.ask,
                            delta=short_quote.delta,
                            oi=short_quote.openInterest,
                            volume=short_quote.volume
                        ),
                        ExecutionLeg(
                            action="BUY",
                            type="CALL",
                            strike=long_quote.strike,
                            priceMid=long_quote.mid,
                            bid=long_quote.bid,
                            ask=long_quote.ask,
                            delta=long_quote.delta,
                            oi=long_quote.openInterest,
                            volume=long_quote.volume
                        )
                    ],
                    netCredit=metrics["netCredit"],
                    width=width,
                    maxProfit=metrics["maxProfit"],
                    maxLoss=metrics["maxLoss"],
                    roc=metrics["roc"],
                    breakeven=metrics["breakeven"],
                    pop=pop,
                    ev=ev,
                    wallBufferPct=wall_dist_score,
                    liquidityScore=liq_score,
                    score=0.0
                )
                
                call_executions.append(execution)
        
        # Score and rank
        for execution in call_executions:
            execution.score = score_execution(execution, context, config, call_executions)
        
        call_executions.sort(key=lambda e: e.score, reverse=True)
        top_executions = call_executions[:3]
        
        if top_executions:
            best_exec = top_executions[0]
            best_metrics = BestMetrics(
                expiry=best_exec.expiry,
                dte=best_exec.dte,
                netCredit=best_exec.netCredit,
                maxProfit=best_exec.maxProfit,
                maxLoss=best_exec.maxLoss,
                roc=best_exec.roc,
                pop=best_exec.pop,
                ev=best_exec.ev,
                wallBufferPct=best_exec.wallBufferPct,
                liquidityGrade="A" if best_exec.liquidityScore > 0.8 else "B" if best_exec.liquidityScore > 0.6 else "C" if best_exec.liquidityScore > 0.4 else "D",
                score=best_exec.score
            )
            
            reasons = []
            if context.gexState == "POSITIVE":
                reasons.append("Positive GEX regime supports premium selling")
            if context.pinConfidence == "HIGH":
                reasons.append("High pin confidence - price likely to stay in range")
            if best_exec.wallBufferPct > 0.7:
                reasons.append(f"Short strike is {best_exec.wallBufferPct*100:.1f}% beyond call wall buffer")
            
            idea = TradeIdea(
                id=f"{context.symbol}_CALL_CREDIT_{datetime.now().isoformat()}",
                symbol=context.symbol,
                asOf=context.asOf,
                strategy="CALL_CREDIT_SPREAD",
                label=f"{context.symbol} Call Credit Spread — {mode.replace('_', ' ').title()}",
                mode=mode,
                confidence="HIGH" if best_exec.score > 0.7 else "MEDIUM" if best_exec.score > 0.5 else "LOW",
                reasons=reasons[:3],
                preMarket=not market_is_open,
                marketStatusNote=market_status_note if not market_is_open else None,
                executions=top_executions,
                best=best_metrics,
                contextSnapshot=ContextSnapshot(
                    spot=context.spot,
                    putWall=context.putWall,
                    callWall=context.callWall,
                    flipLine=context.flipLine,
                    depinRisk=context.depinRisk,
                    gexState=context.gexState,
                    pinConfidence=context.pinConfidence,
                    earningsClusterScore=context.earningsClusterScore,
                    macroEventNext24h=context.macroEventNext24h
                )
            )
            ideas.append(idea)
    
    # Generate IRON_CONDOR ideas
    if "IRON_CONDOR" in strategies:
        iron_condor_config = config.get("ironCondor", {})
        min_credit_ic = iron_condor_config.get("minCredit", 0.30)
        
        iron_condor_candidates = generate_iron_condor_candidates(
            filtered_quotes, context, config, strike_index
        )
        logger.info(f"Generated {len(iron_condor_candidates)} Iron Condor candidates")
        
        iron_condor_executions = []
        
        for put_short, put_long, call_short, call_long in iron_condor_candidates:
            try:
                metrics = calculate_iron_condor_metrics(put_short, put_long, call_short, call_long)
            except Exception as e:
                logger.warning(f"Error calculating Iron Condor metrics: {e}")
                continue
            
            # Check risk envelope
            debug_config = config.get("debug", {})
            bypass_risk = debug_config.get("bypassRiskEnvelope", False)
            if not bypass_risk:
                if metrics["netCredit"] < min_credit_ic:
                    continue
                if metrics["maxLoss"] > max_loss:
                    continue
            
            # Calculate POP and EV
            pop = calculate_iron_condor_pop(
                put_short.delta, call_short.delta, context.spot,
                metrics["breakevenLower"], metrics["breakevenUpper"]
            )
            ev = calculate_ev_proxy(pop, metrics["maxProfit"], metrics["maxLoss"])
            
            # Filter out negative EV
            if ev < 0:
                continue
            
            # Calculate liquidity score (average of all legs)
            liq_scores = [
                calculate_liquidity_score(put_short, config),
                calculate_liquidity_score(put_long, config),
                calculate_liquidity_score(call_short, config),
                calculate_liquidity_score(call_long, config)
            ]
            avg_liq_score = sum(liq_scores) / len(liq_scores)
            
            # Wall distance score (average of put and call)
            buffer_pct = get_wall_buffer(context, config)
            put_wall_dist = calculate_wall_distance_score(
                put_short.strike, context.putWall, context.spot, buffer_pct
            )
            call_wall_dist = calculate_wall_distance_score(
                call_short.strike, context.callWall, context.spot, buffer_pct
            )
            avg_wall_dist = (put_wall_dist + call_wall_dist) / 2.0
            
            # Create execution
            execution = ReferenceExecution(
                expiry=put_short.expiry,
                dte=put_short.dte,
                legs=[
                    ExecutionLeg(
                        action="SELL",
                        type="PUT",
                        strike=put_short.strike,
                        priceMid=put_short.mid,
                        bid=put_short.bid,
                        ask=put_short.ask,
                        delta=put_short.delta,
                        oi=put_short.openInterest,
                        volume=put_short.volume
                    ),
                    ExecutionLeg(
                        action="BUY",
                        type="PUT",
                        strike=put_long.strike,
                        priceMid=put_long.mid,
                        bid=put_long.bid,
                        ask=put_long.ask,
                        delta=put_long.delta,
                        oi=put_long.openInterest,
                        volume=put_long.volume
                    ),
                    ExecutionLeg(
                        action="SELL",
                        type="CALL",
                        strike=call_short.strike,
                        priceMid=call_short.mid,
                        bid=call_short.bid,
                        ask=call_short.ask,
                        delta=call_short.delta,
                        oi=call_short.openInterest,
                        volume=call_short.volume
                    ),
                    ExecutionLeg(
                        action="BUY",
                        type="CALL",
                        strike=call_long.strike,
                        priceMid=call_long.mid,
                        bid=call_long.bid,
                        ask=call_long.ask,
                        delta=call_long.delta,
                        oi=call_long.openInterest,
                        volume=call_long.volume
                    )
                ],
                netCredit=metrics["netCredit"],
                width=metrics["putWidth"] + metrics["callWidth"],  # Total width
                maxProfit=metrics["maxProfit"],
                maxLoss=metrics["maxLoss"],
                roc=metrics["roc"],
                breakeven=metrics["breakevenLower"],  # Lower breakeven
                pop=pop,
                ev=ev,
                wallBufferPct=avg_wall_dist,
                liquidityScore=avg_liq_score,
                score=0.0
            )
            
            iron_condor_executions.append(execution)
        
        # Score and rank
        for execution in iron_condor_executions:
            execution.score = score_execution(execution, context, config, iron_condor_executions)
        
        iron_condor_executions.sort(key=lambda e: e.score, reverse=True)
        top_executions = iron_condor_executions[:3]
        
        if top_executions:
            best_exec = top_executions[0]
            best_metrics = BestMetrics(
                expiry=best_exec.expiry,
                dte=best_exec.dte,
                netCredit=best_exec.netCredit,
                maxProfit=best_exec.maxProfit,
                maxLoss=best_exec.maxLoss,
                roc=best_exec.roc,
                pop=best_exec.pop,
                ev=best_exec.ev,
                wallBufferPct=best_exec.wallBufferPct,
                liquidityGrade="A" if best_exec.liquidityScore > 0.8 else "B" if best_exec.liquidityScore > 0.6 else "C" if best_exec.liquidityScore > 0.4 else "D",
                score=best_exec.score
            )
            
            reasons = ["Hedged structure for below flip line"]
            if context.pinConfidence == "HIGH":
                reasons.append("High pin confidence supports range-bound trade")
            
            trade_idea = TradeIdea(
                id=f"iron_condor_{context.symbol}_{best_exec.expiry}_{int(best_exec.legs[0].strike)}",
                symbol=context.symbol,
                asOf=context.asOf,
                strategy="IRON_CONDOR",
                label=f"{context.symbol} Iron Condor — {mode.replace('_', ' ').title()}",
                allowed=True,
                mode=mode,
                confidence="HIGH" if best_exec.score > 0.7 else "MEDIUM" if best_exec.score > 0.5 else "LOW",
                reasons=reasons[:3],
                preMarket=not market_is_open,
                marketStatusNote=market_status_note if not market_is_open else None,
                executions=top_executions,
                best=best_metrics,
                contextSnapshot=ContextSnapshot(
                    spot=context.spot,
                    putWall=context.putWall,
                    callWall=context.callWall,
                    flipLine=context.flipLine,
                    depinRisk=context.depinRisk,
                    gexState=context.gexState,
                    pinConfidence=context.pinConfidence,
                    earningsClusterScore=context.earningsClusterScore,
                    macroEventNext24h=context.macroEventNext24h
                )
            )
            
            ideas.append(trade_idea)
    
    # Generate BUTTERFLY ideas
    if "BUTTERFLY" in strategies:
        butterfly_config = config.get("butterfly", {})
        max_debit = butterfly_config.get("maxDebit", 0.50)
        
        # Try both PUT and CALL butterflies
        for butterfly_type in ["PUT", "CALL"]:
            butterfly_candidates = generate_butterfly_candidates(
                filtered_quotes, context, config, butterfly_type, strike_index
            )
            logger.info(f"Generated {len(butterfly_candidates)} {butterfly_type} Butterfly candidates")
            
            butterfly_executions = []
            
            for lower, middle1, middle2, upper in butterfly_candidates:
                try:
                    metrics = calculate_butterfly_metrics(lower, middle1, middle2, upper, butterfly_type)
                except Exception as e:
                    logger.warning(f"Error calculating Butterfly metrics: {e}")
                    continue
                
                # Check risk envelope
                debug_config = config.get("debug", {})
                bypass_risk = debug_config.get("bypassRiskEnvelope", False)
                if not bypass_risk:
                    if metrics["netDebit"] > max_debit:
                        continue
                    if metrics["maxLoss"] > max_loss:
                        continue
                
                # Calculate POP and EV
                pop = calculate_butterfly_pop(
                    metrics["middleStrike"], context.spot,
                    metrics["breakevenLower"], metrics["breakevenUpper"]
                )
                ev = calculate_ev_proxy(pop, metrics["maxProfit"], metrics["maxLoss"])
                
                # Filter out negative EV
                if ev < 0:
                    continue
                
                # Calculate liquidity score (average of all legs)
                liq_scores = [
                    calculate_liquidity_score(lower, config),
                    calculate_liquidity_score(middle1, config),
                    calculate_liquidity_score(upper, config)
                ]
                avg_liq_score = sum(liq_scores) / len(liq_scores)
                
                # Wall distance score (from middle strike)
                buffer_pct = get_wall_buffer(context, config)
                if butterfly_type == "PUT":
                    wall_dist = calculate_wall_distance_score(
                        metrics["middleStrike"], context.putWall, context.spot, buffer_pct
                    )
                else:
                    wall_dist = calculate_wall_distance_score(
                        metrics["middleStrike"], context.callWall, context.spot, buffer_pct
                    )
                
                # Create execution
                # Note: For butterfly, we need to handle the fact that middle is 2 contracts
                execution = ReferenceExecution(
                    expiry=lower.expiry,
                    dte=lower.dte,
                    legs=[
                        ExecutionLeg(
                            action="BUY",
                            type=butterfly_type,
                            strike=lower.strike,
                            priceMid=lower.mid,
                            bid=lower.bid,
                            ask=lower.ask,
                            delta=lower.delta,
                            oi=lower.openInterest,
                            volume=lower.volume
                        ),
                        ExecutionLeg(
                            action="SELL",
                            type=butterfly_type,
                            strike=middle1.strike,
                            priceMid=middle1.mid,
                            bid=middle1.bid,
                            ask=middle1.ask,
                            delta=middle1.delta,
                            oi=middle1.openInterest,
                            volume=middle1.volume
                        ),
                        ExecutionLeg(
                            action="SELL",
                            type=butterfly_type,
                            strike=middle2.strike,
                            priceMid=middle2.mid,
                            bid=middle2.bid,
                            ask=middle2.ask,
                            delta=middle2.delta,
                            oi=middle2.openInterest,
                            volume=middle2.volume
                        ),
                        ExecutionLeg(
                            action="BUY",
                            type=butterfly_type,
                            strike=upper.strike,
                            priceMid=upper.mid,
                            bid=upper.bid,
                            ask=upper.ask,
                            delta=upper.delta,
                            oi=upper.openInterest,
                            volume=upper.volume
                        )
                    ],
                    netCredit=-metrics["netDebit"] if metrics["netDebit"] > 0 else 0.0,  # Negative if debit
                    width=metrics["wingWidth"] * 2,  # Total width
                    maxProfit=metrics["maxProfit"],
                    maxLoss=metrics["maxLoss"],
                    roc=metrics["roc"],
                    breakeven=metrics["breakevenLower"],
                    pop=pop,
                    ev=ev,
                    wallBufferPct=wall_dist,
                    liquidityScore=avg_liq_score,
                    score=0.0
                )
                
                butterfly_executions.append(execution)
            
            # Score and rank
            for execution in butterfly_executions:
                execution.score = score_execution(execution, context, config, butterfly_executions)
            
            butterfly_executions.sort(key=lambda e: e.score, reverse=True)
            top_executions = butterfly_executions[:3]
            
            if top_executions:
                best_exec = top_executions[0]
                best_metrics = BestMetrics(
                    expiry=best_exec.expiry,
                    dte=best_exec.dte,
                    netCredit=best_exec.netCredit,
                    maxProfit=best_exec.maxProfit,
                    maxLoss=best_exec.maxLoss,
                    roc=best_exec.roc,
                    pop=best_exec.pop,
                    ev=best_exec.ev,
                    wallBufferPct=best_exec.wallBufferPct,
                    liquidityGrade="A" if best_exec.liquidityScore > 0.8 else "B" if best_exec.liquidityScore > 0.6 else "C" if best_exec.liquidityScore > 0.4 else "D",
                    score=best_exec.score
                )
                
                reasons = ["Hedged structure for high pin confidence"]
                if context.pinConfidence == "HIGH":
                    reasons.append("High pin confidence - price likely to stay at middle strike")
                
                trade_idea = TradeIdea(
                    id=f"butterfly_{butterfly_type.lower()}_{context.symbol}_{best_exec.expiry}_{int(best_exec.legs[1].strike)}",
                    symbol=context.symbol,
                    asOf=context.asOf,
                    strategy="BUTTERFLY",
                    label=f"{context.symbol} {butterfly_type} Butterfly — {mode.replace('_', ' ').title()}",
                    allowed=True,
                    mode=mode,
                    confidence="HIGH" if best_exec.score > 0.7 else "MEDIUM" if best_exec.score > 0.5 else "LOW",
                    reasons=reasons[:3],
                    preMarket=not market_is_open,
                    marketStatusNote=market_status_note if not market_is_open else None,
                    executions=top_executions,
                    best=best_metrics,
                    contextSnapshot=ContextSnapshot(
                        spot=context.spot,
                        putWall=context.putWall,
                        callWall=context.callWall,
                        flipLine=context.flipLine,
                        depinRisk=context.depinRisk,
                        gexState=context.gexState,
                        pinConfidence=context.pinConfidence,
                        earningsClusterScore=context.earningsClusterScore,
                        macroEventNext24h=context.macroEventNext24h
                    )
                )
                
                ideas.append(trade_idea)
    
    # Rank and filter final ideas
    return rank_and_filter(ideas, config)
