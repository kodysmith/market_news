"""
Gamma Exposure (GEX) Calculator Module - OPEX-Aware Multi-Lens Analysis

Features:
- Trading calendar utilities (OPEX detection, trading days)
- OPEX-aware expiry bucketing and weighting
- Dealer-centric sign convention
- Multi-lens analysis: Today (0-2 DTE), Tactical (0-14 DTE), Regime (0-60 DTE)
- Price grid flip line detection with interpolation
- Transition flag detection (near flip, flip moving)
- Separate call/put wall detection

Based on industry-standard GEX computation spec.
"""

import json
import math
from pathlib import Path
from datetime import datetime, timezone, timedelta, date
from typing import Optional, Dict, Tuple, List, Callable, Any
from collections import defaultdict

import pandas as pd
import numpy as np
from scipy.stats import norm
import requests
import yfinance as yf


# ===============================
# Constants
# ===============================
REGIME_DTE_MAX = 60       # For regime / flip aggregation
TACTICAL_DTE_MAX = 14     # For tactical walls
TODAY_DTE_MAX = 2         # For "today walls" / pin

STRIKE_GRID_STEP = 1.0    # SPY strikes are typically $1 increments
TRANSITION_DISTANCE_PTS = 2.0  # Points from flip to trigger transition
TRANSITION_FLIP_MOVE_PTS_1H = 1.5  # Flip movement threshold

MULT = 100  # Standard equity options multiplier


# ===============================
# Trading Calendar Utilities
# ===============================
def get_eastern_timezone():
    """Get US Eastern timezone object"""
    try:
        from zoneinfo import ZoneInfo
        return ZoneInfo("America/New_York")
    except ImportError:
        # Fallback for Python < 3.9
        return timezone(timedelta(hours=-5))


def get_eastern_now() -> datetime:
    """Get current datetime in US Eastern time"""
    return datetime.now(get_eastern_timezone())


def get_eastern_today() -> date:
    """Get current date in US Eastern time"""
    return get_eastern_now().date()


def third_friday_of_month(year: int, month: int) -> date:
    """
    Find the 3rd Friday of a given month (monthly OPEX).
    
    Args:
        year: Year
        month: Month (1-12)
    
    Returns:
        Date of the 3rd Friday
    """
    # Find first day of the month
    first_day = date(year, month, 1)
    
    # Find first Friday (weekday 4 = Friday)
    days_until_friday = (4 - first_day.weekday()) % 7
    first_friday = first_day + timedelta(days=days_until_friday)
    
    # 3rd Friday is 2 weeks after first Friday
    third_friday = first_friday + timedelta(weeks=2)
    
    return third_friday


def next_monthly_opex(now: date) -> date:
    """
    Find the next monthly OPEX date (3rd Friday of the month).
    
    If today is on or after this month's OPEX, returns next month's OPEX.
    
    Args:
        now: Current date
    
    Returns:
        Next monthly OPEX date
    """
    # Get this month's OPEX
    this_month_opex = third_friday_of_month(now.year, now.month)
    
    # If we haven't passed it yet (including today), return it
    if now <= this_month_opex:
        return this_month_opex
    
    # Otherwise, get next month's OPEX
    if now.month == 12:
        return third_friday_of_month(now.year + 1, 1)
    else:
        return third_friday_of_month(now.year, now.month + 1)


def is_trading_day(d: date) -> bool:
    """
    Check if a date is a trading day (weekday, not a holiday).
    
    Note: This is a simplified version that only checks weekdays.
    For production, integrate with a proper market calendar (e.g., pandas_market_calendars).
    """
    # Weekday: 0=Monday, 6=Sunday
    # Markets are open Monday-Friday (0-4)
    return d.weekday() < 5


def trading_days_between(start: date, end: date) -> int:
    """
    Count trading days between two dates (exclusive of start, inclusive of end).
    
    Args:
        start: Start date (not counted)
        end: End date (counted if trading day)
    
    Returns:
        Number of trading days
    """
    if end <= start:
        return 0
    
    count = 0
    current = start + timedelta(days=1)
    
    while current <= end:
        if is_trading_day(current):
            count += 1
        current += timedelta(days=1)
    
    return count


def dte_in_trading_days(now: date, expiry: date) -> int:
    """
    Calculate DTE in trading days.
    
    Args:
        now: Current date
        expiry: Expiration date
    
    Returns:
        Trading days to expiration
    """
    return trading_days_between(now, expiry)


def bucket_weight(expiry: date, now: date, opex: date) -> float:
    """
    Calculate OPEX-aware weight for an expiration date.
    
    Weights (intentionally simple and explainable):
    - 1.0: Front expiry (0-2 DTE) - dominates "today" hedging
    - 0.8: OPEX date - extra weight for institutional positioning
    - 0.6: Weeklies leading into OPEX
    - 0.3: Background near-term gamma (up to 60 DTE)
    - 0.0: Exclude far-dated by default
    
    Args:
        expiry: Option expiration date
        now: Current date
        opex: Next monthly OPEX date
    
    Returns:
        Weight between 0.0 and 1.0
    """
    dte = dte_in_trading_days(now, expiry)
    opex_dte = dte_in_trading_days(now, opex)
    
    # Front expiry (0-2 DTE) - dominates "today" hedging
    if dte <= 2:
        return 1.0
    
    # OPEX date deserves extra weight (institutional positioning)
    if expiry == opex:
        return 0.8
    
    # Weeklies leading into OPEX
    if dte <= opex_dte:
        return 0.6
    
    # Background near-term gamma (up to 60 DTE)
    if dte <= 60:
        return 0.3
    
    # Exclude far-dated
    return 0.0


# ===============================
# Black-Scholes Gamma
# ===============================
def bs_gamma(S: float, K: float, T: float, sigma: float) -> float:
    """
    Calculate Black-Scholes gamma.
    
    Args:
        S: Spot price
        K: Strike price
        T: Time to expiry in years
        sigma: Implied volatility (as decimal, e.g., 0.20 for 20%)
    
    Returns:
        Gamma per share per $1 move
    """
    if S <= 0 or K <= 0 or T <= 0 or sigma <= 0:
        return 0.0
    
    d1 = (np.log(S / K) + 0.5 * sigma**2 * T) / (sigma * np.sqrt(T))
    gamma = norm.pdf(d1) / (S * sigma * np.sqrt(T))
    
    return float(gamma)


def years_to_expiry(exp_str: str) -> float:
    """
    Convert expiration date string to years to expiry.
    
    For 0DTE options (expiring today), returns a small positive value (1/365)
    to allow gamma calculation since 0DTE options still have significant gamma.
    
    Uses US Eastern time since US options expire at 4:00 PM ET.
    
    Args:
        exp_str: Expiration date in YYYY-MM-DD format
    
    Returns:
        Years to expiry (minimum 1/365 for 0DTE)
    """
    exp_date = datetime.strptime(exp_str, "%Y-%m-%d").date()
    today = get_eastern_today()
    
    # 0DTE (same day) - use 1 day as minimum for gamma calculation
    if exp_date == today:
        return 1.0 / 365.0
    
    # Expired
    if exp_date < today:
        return 0.0
    
    # Calculate calendar days to expiry
    days_to_expiry = (exp_date - today).days
    return days_to_expiry / 365.0


def is_valid_number(x) -> bool:
    """Check if value is a valid finite number"""
    if not isinstance(x, (int, float)):
        return False
    x = float(x)
    return math.isfinite(x) and not math.isnan(x)


# ===============================
# API Functions
# ===============================
def get_option_chain_snapshot(
    api_key: str, 
    underlying: str, 
    limit: int = 250, 
    days_out: int = 65,
    spot_price: Optional[float] = None,
    strike_range: Optional[float] = None
) -> dict:
    """
    Fetch option chain snapshot from Massive API.
    
    For index options (SPX, NDX, etc.), uses I: prefix.
    Massive API requires I: prefix for index options chains (e.g., I:SPX).
    
    Args:
        api_key: Massive API key
        underlying: Stock/ETF/Index ticker
        limit: Max contracts per request (250 max)
        days_out: Number of days to look ahead for expiration dates (default 65 for regime lens)
        spot_price: Current spot price (if provided, used to center strike range)
        strike_range: Range around spot for strikes (e.g., 50 means ±50 from spot)
    
    Returns:
        API response dict with 'results' list of option contracts
    """
    # List of indices that need I: prefix for Massive API
    index_options = ["SPX", "NDX", "DJX", "RUT", "XSP"]
    # VIX might use different format, keep separate
    futures_indices = ["VIX"]
    
    underlying_upper = underlying.upper()
    
    # Add I: prefix for index options (Massive API format)
    if underlying_upper in index_options:
        if underlying_upper.startswith("I:"):
            underlying_for_api = underlying_upper
        else:
            underlying_for_api = f"I:{underlying_upper}"
    elif underlying_upper in futures_indices and not underlying_upper.startswith("$"):
        # Some futures might still use $ prefix
        underlying_for_api = f"${underlying_upper}"
    else:
        underlying_for_api = underlying_upper
    
    url = f"https://api.massive.com/v3/snapshot/options/{underlying_for_api}"
    
    # Date range for expirations
    today = datetime.now().strftime("%Y-%m-%d")
    end_date = (datetime.now() + timedelta(days=days_out)).strftime("%Y-%m-%d")
    
    params = {
        "apiKey": api_key,
        "limit": limit,
        "expiration_date.gte": today,
        "expiration_date.lte": end_date,
    }
    
    # Add strike range filtering if spot price is provided
    if spot_price and strike_range:
        params["strike_price.gte"] = spot_price - strike_range
        params["strike_price.lte"] = spot_price + strike_range
    
    r = requests.get(url, params=params, timeout=30)
    r.raise_for_status()
    return r.json()


def get_option_chain_centered(
    api_key: str,
    underlying: str,
    spot_price: float,
    strike_range: float = 50,
    days_out: int = 65,
    limit: int = 250
) -> dict:
    """
    Fetch option chain centered around the current spot price.
    
    This ensures we get both calls and puts near ATM for proper flip line calculation.
    
    Args:
        api_key: Massive API key
        underlying: Stock/ETF/Index ticker
        spot_price: Current spot price to center the strike range
        strike_range: Points above/below spot to include (default 50)
        days_out: Number of days for expiration range
        limit: Max contracts per request
    
    Returns:
        API response dict with 'results' list of option contracts
    """
    return get_option_chain_snapshot(
        api_key=api_key,
        underlying=underlying,
        limit=limit,
        days_out=days_out,
        spot_price=spot_price,
        strike_range=strike_range
    )


def build_symmetric_strike_set(
    spot: float, 
    num_strikes_each_side: int = 20, 
    strike_increment: float = 1.0
) -> List[float]:
    """
    Build a symmetric set of strikes centered around ATM.
    
    Args:
        spot: Current spot price
        num_strikes_each_side: Number of strikes on each side of ATM (default 20)
        strike_increment: Strike price increment (default $1 for SPY)
    
    Returns:
        List of strikes from (ATM - N) to (ATM + N)
    """
    # Round spot to nearest strike increment to get ATM
    atm = round(spot / strike_increment) * strike_increment
    
    # Build symmetric strike set
    strikes = []
    for k in range(-num_strikes_each_side, num_strikes_each_side + 1):
        strikes.append(atm + k * strike_increment)
    
    return sorted(strikes)


def process_chain_symmetric(
    snap: dict, 
    spot: float, 
    num_strikes_each_side: int = 20,
    strike_increment: float = 1.0
) -> Tuple[List[Dict], Dict]:
    """
    Process option chain with symmetric strike selection.
    
    Filters API response to include ONLY strikes within ±N of ATM.
    This ensures we include both calls AND puts for every strike in our target set.
    Missing contracts are noted but don't distort the selection.
    
    Args:
        snap: API response with 'results' list
        spot: Current spot price
        num_strikes_each_side: Number of strikes each side of ATM (default 20)
        strike_increment: Strike price increment (default $1 for SPY)
    
    Returns:
        Tuple of (contracts_list, diagnostics_dict)
        - contracts_list: List of parsed contracts
        - diagnostics: Dict with counts and missing info
    """
    # Build the target strike set
    target_strikes = set(build_symmetric_strike_set(spot, num_strikes_each_side, strike_increment))
    
    # Index API results by (strike, expiry, right)
    api_contracts = {}
    all_expiries = set()
    
    for item in snap.get("results", []):
        d = item.get("details") or {}
        
        contract_type = d.get("contract_type")
        strike = d.get("strike_price")
        exp_str = d.get("expiration_date")
        
        if contract_type not in ("call", "put") or strike is None or exp_str is None:
            continue
        
        try:
            strike = float(strike)
        except (ValueError, TypeError):
            continue
        
        right = "C" if contract_type == "call" else "P"
        key = (strike, exp_str, right)
        
        api_contracts[key] = item
        all_expiries.add(exp_str)
    
    # Parse contracts, tracking what's in our target set
    contracts = []
    diagnostics = {
        "target_strikes": len(target_strikes),
        "target_strikes_list": sorted(target_strikes),
        "expiries_found": sorted(all_expiries),
        "num_expiries": len(all_expiries),
        "calls_found": 0,
        "puts_found": 0,
        "calls_in_target": 0,
        "puts_in_target": 0,
        "missing_from_api": 0,
        "returned_but_unusable": 0,
        "contracts_processed": 0
    }
    
    # Process all contracts from API that are in our target strike set
    for key, item in api_contracts.items():
        strike, exp_str, right = key
        
        # Track all found
        if right == "C":
            diagnostics["calls_found"] += 1
        else:
            diagnostics["puts_found"] += 1
        
        # Only process if in our target strike set
        if strike not in target_strikes:
            continue
        
        # Track in-target
        if right == "C":
            diagnostics["calls_in_target"] += 1
        else:
            diagnostics["puts_in_target"] += 1
        
        # Parse the contract
        contract = parse_contract(item, spot)
        
        if contract:
            contracts.append(contract)
            diagnostics["contracts_processed"] += 1
        else:
            diagnostics["returned_but_unusable"] += 1
    
    # Count missing contracts (in target set but not in API)
    expected_contracts = len(target_strikes) * len(all_expiries) * 2  # strikes × expiries × (call+put)
    actually_found_in_target = diagnostics["calls_in_target"] + diagnostics["puts_in_target"]
    diagnostics["missing_from_api"] = expected_contracts - actually_found_in_target
    diagnostics["expected_contracts"] = expected_contracts
    
    # Symmetry check - allow up to 30% difference or 15 contracts absolute
    # This accommodates real-world data where puts may have less OI/liquidity
    calls = diagnostics["calls_in_target"]
    puts = diagnostics["puts_in_target"]
    abs_diff = abs(calls - puts)
    max_count = max(calls, puts, 1)  # Avoid division by zero
    pct_diff = abs_diff / max_count
    
    # Consider symmetric if difference is < 30% AND < 15 absolute
    diagnostics["is_symmetric"] = (pct_diff < 0.30) and (abs_diff <= 15)
    diagnostics["symmetry_pct_diff"] = round(pct_diff * 100, 1)
    diagnostics["symmetry_abs_diff"] = abs_diff
    
    return contracts, diagnostics


def compute_gex_symmetric(
    snap: dict,
    spot: float,
    num_strikes_each_side: int = 20,
    strike_increment: float = 1.0
) -> Tuple[Dict[float, float], Dict[float, float], Dict]:
    """
    Compute GEX with symmetric strike selection.
    
    Args:
        snap: API response
        spot: Current spot price
        num_strikes_each_side: Strikes each side of ATM
        strike_increment: Strike increment
    
    Returns:
        Tuple of (call_gex_by_strike, put_gex_by_strike, diagnostics)
    """
    contracts, diagnostics = process_chain_symmetric(
        snap, spot, num_strikes_each_side, strike_increment
    )
    
    # Separate by call/put
    call_gex = defaultdict(float)
    put_gex = defaultdict(float)
    
    today = get_eastern_today()
    opex = next_monthly_opex(today)
    
    for c in contracts:
        # Apply expiry filter and weight
        dte = dte_in_trading_days(today, c["expiry"])
        if dte > REGIME_DTE_MAX:
            continue
        
        weight = bucket_weight(c["expiry"], today, opex)
        if weight == 0:
            continue
        
        gex = contract_dealer_gex(c["gamma"], c["oi"], spot, c["type"], c["mult"])
        weighted_gex = gex * weight
        
        if c["type"] == "call":
            call_gex[c["strike"]] += weighted_gex
        else:
            put_gex[c["strike"]] += weighted_gex
    
    # Ensure all target strikes have entries (0 if missing)
    target_strikes = build_symmetric_strike_set(spot, num_strikes_each_side, strike_increment)
    for strike in target_strikes:
        if strike not in call_gex:
            call_gex[strike] = 0.0
        if strike not in put_gex:
            put_gex[strike] = 0.0
    
    return dict(call_gex), dict(put_gex), diagnostics


def get_spot_from_alpha_vantage(api_key: str, ticker: str) -> Optional[float]:
    """Fetch spot price via Alpha Vantage GLOBAL_QUOTE"""
    try:
        url = "https://www.alphavantage.co/query"
        params = {
            "function": "GLOBAL_QUOTE",
            "symbol": ticker,
            "apikey": api_key,
        }
        r = requests.get(url, params=params, timeout=10)
        r.raise_for_status()
        data = r.json()

        quote = data.get("Global Quote", {})
        price = quote.get("05. price")

        if price is not None:
            price = float(price)
            if price > 0:
                return price
    except Exception as e:
        print(f"[spot] Alpha Vantage failed for {ticker}: {e}")
    return None


def get_spot_from_yfinance(ticker: str) -> Optional[float]:
    """Fetch spot price from Yahoo Finance (works for indices like SPX)"""
    try:
        # For indices, use ^ prefix
        yf_symbol = f"^{ticker}" if ticker in ["SPX", "NDX", "DJI", "RUT", "XSP"] else ticker
        ticker_obj = yf.Ticker(yf_symbol)
        info = ticker_obj.info
        
        current_price = info.get('regularMarketPrice') or info.get('previousClose')
        if current_price:
            return float(current_price)
        
        # Fallback: get from history
        hist = ticker_obj.history(period="1d", interval="1m")
        if not hist.empty:
            return float(hist['Close'].iloc[-1])
    except Exception as e:
        print(f"[spot] yfinance failed for {ticker}: {e}")
    return None


def get_underlying_spot_price_v2(api_key: str, ticker: str) -> Optional[float]:
    """Uses Massive v2 single ticker snapshot for spot price"""
    try:
        url = f"https://api.massive.com/v2/snapshot/locale/us/markets/stocks/tickers/{ticker}"
        params = {"apiKey": api_key}
        r = requests.get(url, params=params, timeout=10)
        r.raise_for_status()
        data = r.json()

        tkr = data.get("ticker", {})
        if not isinstance(tkr, dict):
            return None

        # Priority: lastTrade -> lastQuote midpoint -> min.c -> day.c -> prevDay.c
        last_trade = tkr.get("lastTrade", {})
        p = last_trade.get("p")
        if isinstance(p, (int, float)) and p > 0:
            return float(p)

        last_quote = tkr.get("lastQuote", {})
        bid = last_quote.get("p")
        ask = last_quote.get("P")
        if isinstance(bid, (int, float)) and isinstance(ask, (int, float)) and bid > 0 and ask > 0:
            return float((bid + ask) / 2.0)

        minute = tkr.get("min", {})
        mc = minute.get("c")
        if isinstance(mc, (int, float)) and mc > 0:
            return float(mc)

        day = tkr.get("day", {})
        dc = day.get("c")
        if isinstance(dc, (int, float)) and dc > 0:
            return float(dc)

        prev = tkr.get("prevDay", {})
        pc = prev.get("c")
        if isinstance(pc, (int, float)) and pc > 0:
            return float(pc)
    except Exception as e:
        print(f"[spot] v2 snapshot failed for {ticker}: {e}")
    return None


def get_spot_price(
    ticker: str,
    massive_api_key: Optional[str] = None,
    alphavantage_api_key: Optional[str] = None
) -> Optional[float]:
    """
    Get spot price with smart routing.
    
    Priority:
    1. Massive API v2 (same source as options chain - reduces discrepancies)
    2. For indices: yfinance
    3. For stocks: Alpha Vantage → yfinance
    """
    if massive_api_key:
        spot = get_underlying_spot_price_v2(massive_api_key, ticker)
        if spot:
            return spot
    
    is_index = ticker in ["SPX", "NDX", "DJI", "RUT", "VIX", "XSP"]
    
    if is_index:
        spot = get_spot_from_yfinance(ticker)
        if spot:
            return spot
    else:
        if alphavantage_api_key:
            spot = get_spot_from_alpha_vantage(alphavantage_api_key, ticker)
            if spot:
                return spot
        spot = get_spot_from_yfinance(ticker)
        if spot:
            return spot
    
    return None


# ===============================
# Core GEX Calculation - Dealer Centric
# ===============================
def contract_dealer_gex(gamma: float, oi: int, spot: float, option_type: str, mult: int = MULT) -> float:
    """
    Calculate dealer GEX for a single contract.
    
    Sign convention (industry standard):
    - Dealers are assumed net short customer option positions
    - Call GEX = +gamma × OI × mult × spot² (dealers buy when price rises)
    - Put GEX = -gamma × OI × mult × spot² (dealers sell when price rises)
    
    This creates a flip line where call GEX balances put GEX.
    
    Args:
        gamma: Black-Scholes gamma per share (always positive)
        oi: Open interest (number of contracts)
        spot: Current spot price
        option_type: "call" or "put" (or "C" / "P")
        mult: Contract multiplier (default 100 for equity options)
    
    Returns:
        Dealer GEX with appropriate sign for option type
    """
    raw_gex = gamma * oi * mult * (spot ** 2)
    
    # Calls contribute positive GEX (stabilizing hedging flow)
    # Puts contribute negative GEX (destabilizing hedging flow)
    if option_type.lower() in ("call", "c"):
        return raw_gex
    else:
        return -raw_gex


def parse_contract(item: dict, spot: float) -> Optional[Dict]:
    """
    Parse a single option contract from API response.
    
    Args:
        item: Contract dict from API
        spot: Current spot price
    
    Returns:
        Parsed contract dict or None if invalid
    """
    d = item.get("details") or {}
    
    # Required fields
    contract_type = d.get("contract_type")
    strike = d.get("strike_price")
    exp_str = d.get("expiration_date")
    
    if contract_type not in ("call", "put") or strike is None or exp_str is None:
        return None
    
    try:
        strike = float(strike)
    except (ValueError, TypeError):
        return None
    
    # Open interest
    oi = item.get("open_interest")
    oi = float(oi) if isinstance(oi, (int, float)) else 0.0
    if oi <= 0:
        return None
    
    # Time to expiry
    T = years_to_expiry(exp_str)
    if T <= 0:
        return None
    
    # Implied volatility
    iv = item.get("implied_volatility")
    if not is_valid_number(iv) or float(iv) <= 0:
        return None
    
    # Convert IV to decimal
    sigma = float(iv)
    sigma = sigma / 100.0 if sigma > 3 else sigma
    
    # Calculate gamma
    gamma = bs_gamma(spot, strike, T, sigma)
    if not is_valid_number(gamma) or gamma <= 0:
        return None
    
    # Contract multiplier
    mult = d.get("shares_per_contract", MULT)
    mult = float(mult) if isinstance(mult, (int, float)) else float(MULT)
    
    # Parse expiration date
    exp_date = datetime.strptime(exp_str, "%Y-%m-%d").date()
    
    return {
        "strike": strike,
        "type": contract_type,  # "call" or "put"
        "right": "C" if contract_type == "call" else "P",
        "expiry": exp_date,
        "expiry_str": exp_str,
        "oi": oi,
        "gamma": gamma,
        "iv": sigma,
        "mult": mult,
        "T": T
    }


def parse_option_chain(snap: dict, spot: float) -> List[Dict]:
    """
    Parse all valid contracts from API snapshot.
    
    Args:
        snap: API response with 'results' list
        spot: Current spot price
    
    Returns:
        List of parsed contract dicts
    """
    contracts = []
    
    for item in snap.get("results", []):
        contract = parse_contract(item, spot)
        if contract:
            contracts.append(contract)
    
    return contracts


# ===============================
# Multi-Lens GEX Aggregation
# ===============================
def aggregate_gex_by_strike(
    contracts: List[Dict],
    spot: float,
    expiry_filter: Callable[[date], bool],
    weight_fn: Callable[[date], float]
) -> Dict[float, float]:
    """
    Aggregate GEX by strike with filtering and weighting.
    
    Args:
        contracts: List of parsed contracts
        spot: Current spot price
        expiry_filter: Function that returns True if expiry should be included
        weight_fn: Function that returns weight for an expiry date
    
    Returns:
        Dict mapping strike -> total weighted GEX
    """
    gex_by_strike = defaultdict(float)
    
    for c in contracts:
        if not expiry_filter(c["expiry"]):
            continue
        
        weight = weight_fn(c["expiry"])
        if weight == 0:
            continue
        
        gex = contract_dealer_gex(c["gamma"], c["oi"], spot, c["type"], c["mult"])
        weighted_gex = gex * weight
        
        gex_by_strike[c["strike"]] += weighted_gex
    
    return dict(gex_by_strike)


def aggregate_gex_by_strike_and_right(
    contracts: List[Dict],
    spot: float,
    expiry_filter: Callable[[date], bool],
    weight_fn: Callable[[date], float]
) -> Tuple[Dict[float, float], Dict[float, float]]:
    """
    Aggregate GEX by strike, separately for calls and puts.
    
    Args:
        contracts: List of parsed contracts
        spot: Current spot price
        expiry_filter: Function that returns True if expiry should be included
        weight_fn: Function that returns weight for an expiry date
    
    Returns:
        Tuple of (call_gex_by_strike, put_gex_by_strike)
    """
    call_gex = defaultdict(float)
    put_gex = defaultdict(float)
    
    for c in contracts:
        if not expiry_filter(c["expiry"]):
            continue
        
        weight = weight_fn(c["expiry"])
        if weight == 0:
            continue
        
        gex = contract_dealer_gex(c["gamma"], c["oi"], spot, c["type"], c["mult"])
        weighted_gex = gex * weight
        
        if c["right"] == "C":
            call_gex[c["strike"]] += weighted_gex
        else:
            put_gex[c["strike"]] += weighted_gex
    
    return dict(call_gex), dict(put_gex)


# ===============================
# Wall Detection (Multi-candidate with OI and GEX separation)
# ===============================

def aggregate_oi_by_strike_and_right(
    contracts: List[Dict],
    expiry_filter: Callable[[date], bool]
) -> Tuple[Dict[float, int], Dict[float, int]]:
    """
    Aggregate Open Interest by strike, separately for calls and puts.
    
    Args:
        contracts: List of parsed contracts
        expiry_filter: Function that returns True if expiry should be included
    
    Returns:
        Tuple of (call_oi_by_strike, put_oi_by_strike)
    """
    call_oi = defaultdict(int)
    put_oi = defaultdict(int)
    
    for c in contracts:
        if not expiry_filter(c["expiry"]):
            continue
        
        oi = c.get("oi", 0)
        if c["right"] == "C":
            call_oi[c["strike"]] += oi
        else:
            put_oi[c["strike"]] += oi
    
    return dict(call_oi), dict(put_oi)


def find_top_k_walls(
    value_map: Dict[float, float],
    spot: float,
    side: str,  # "call" or "put"
    k: int = 3,
    use_abs: bool = True
) -> List[Dict[str, Any]]:
    """
    Find top K wall candidates on the correct side of spot.
    
    Args:
        value_map: Dict mapping strike -> value (GEX or OI)
        spot: Current spot price
        side: "call" (above spot) or "put" (below spot)
        k: Number of candidates to return
        use_abs: Whether to use absolute value for ranking (True for GEX)
    
    Returns:
        List of dicts with 'strike', 'value', 'distance_from_spot'
    """
    if not value_map:
        return []
    
    # Filter to correct side
    if side == "call":
        # Call walls are ABOVE spot (resistance)
        filtered = {s: v for s, v in value_map.items() if s > spot}
        if not filtered:
            # Include ATM if nothing above
            filtered = {s: v for s, v in value_map.items() if s >= spot}
    else:
        # Put walls are BELOW spot (support)
        filtered = {s: v for s, v in value_map.items() if s < spot}
        if not filtered:
            # Include ATM if nothing below
            filtered = {s: v for s, v in value_map.items() if s <= spot}
    
    if not filtered:
        return []
    
    # Sort by value (absolute for GEX, raw for OI)
    if use_abs:
        sorted_strikes = sorted(filtered.keys(), key=lambda k: abs(filtered[k]), reverse=True)
    else:
        sorted_strikes = sorted(filtered.keys(), key=lambda k: filtered[k], reverse=True)
    
    # Return top K
    result = []
    for strike in sorted_strikes[:k]:
        result.append({
            'strike': strike,
            'value': filtered[strike],
            'distance_from_spot': round(strike - spot, 2)
        })
    
    return result


def compute_walls_multi(
    call_gex: Dict[float, float],
    put_gex: Dict[float, float],
    call_oi: Dict[float, int],
    put_oi: Dict[float, int],
    spot: float,
    k: int = 3
) -> Dict[str, Any]:
    """
    Compute comprehensive wall data with both GEX and OI walls.
    
    Returns primary + secondary walls for both GEX and OI.
    
    Args:
        call_gex: Call GEX by strike
        put_gex: Put GEX by strike
        call_oi: Call OI by strike
        put_oi: Put OI by strike
        spot: Current spot price
        k: Number of candidates per side
    
    Returns:
        Dict with gex_walls and oi_walls, each having call and put candidates
    """
    # GEX walls (use absolute value for ranking)
    gex_call_candidates = find_top_k_walls(call_gex, spot, "call", k, use_abs=True)
    gex_put_candidates = find_top_k_walls(put_gex, spot, "put", k, use_abs=True)
    
    # OI walls (use raw value for ranking - higher OI = stronger wall)
    oi_call_candidates = find_top_k_walls(call_oi, spot, "call", k, use_abs=False)
    oi_put_candidates = find_top_k_walls(put_oi, spot, "put", k, use_abs=False)
    
    # Extract primary walls (first candidate)
    gex_call_primary = gex_call_candidates[0]['strike'] if gex_call_candidates else None
    gex_put_primary = gex_put_candidates[0]['strike'] if gex_put_candidates else None
    oi_call_primary = oi_call_candidates[0]['strike'] if oi_call_candidates else None
    oi_put_primary = oi_put_candidates[0]['strike'] if oi_put_candidates else None
    
    return {
        'gex_walls': {
            'call': {
                'primary': gex_call_primary,
                'candidates': gex_call_candidates
            },
            'put': {
                'primary': gex_put_primary,
                'candidates': gex_put_candidates
            }
        },
        'oi_walls': {
            'call': {
                'primary': oi_call_primary,
                'candidates': oi_call_candidates
            },
            'put': {
                'primary': oi_put_primary,
                'candidates': oi_put_candidates
            }
        },
        # Legacy format for backward compatibility
        'call': gex_call_primary,
        'put': gex_put_primary
    }


# Legacy single-wall functions for backward compatibility
def find_call_wall(call_map: Dict[float, float], spot: Optional[float] = None) -> Optional[float]:
    """Find primary call wall (highest GEX magnitude above spot)"""
    candidates = find_top_k_walls(call_map, spot or 0, "call", k=1, use_abs=True)
    return candidates[0]['strike'] if candidates else None


def find_put_wall(put_map: Dict[float, float], spot: Optional[float] = None) -> Optional[float]:
    """Find primary put wall (highest GEX magnitude below spot)"""
    candidates = find_top_k_walls(put_map, spot or 0, "put", k=1, use_abs=True)
    return candidates[0]['strike'] if candidates else None


def compute_walls(
    call_map: Dict[float, float],
    put_map: Dict[float, float],
    spot: Optional[float] = None
) -> Dict[str, Optional[float]]:
    """Legacy function - returns only primary GEX walls"""
    return {
        "call": find_call_wall(call_map, spot),
        "put": find_put_wall(put_map, spot)
    }


# ===============================
# Flip Line Detection
# ===============================
def rescale_gex_by_spot(
    gex_by_strike: Dict[float, float],
    new_spot: float,
    base_spot: float
) -> Dict[float, float]:
    """
    Rescale GEX values for a different spot price.
    
    Since GEX ~ spot^2, rescale by (new_spot^2 / base_spot^2).
    
    Args:
        gex_by_strike: Original GEX by strike at base_spot
        new_spot: New spot price to rescale to
        base_spot: Original spot price
    
    Returns:
        Rescaled GEX by strike
    """
    if base_spot <= 0:
        return gex_by_strike
    
    factor = (new_spot ** 2) / (base_spot ** 2)
    return {strike: gex * factor for strike, gex in gex_by_strike.items()}


def find_zero_crossing_linear(series: List[Tuple[float, float]]) -> Optional[float]:
    """
    Find zero crossing in a series using linear interpolation.
    
    Args:
        series: List of (x, y) tuples in ascending x order
    
    Returns:
        X value where y crosses zero, or None if no crossing
    """
    for i in range(len(series) - 1):
        x1, y1 = series[i]
        x2, y2 = series[i + 1]
        
        if y1 == 0:
            return x1
        if y2 == 0:
            return x2
        
        # Check for sign change
        if y1 * y2 < 0:
            # Linear interpolation
            if abs(y2 - y1) < 1e-10:
                return x1
            return x1 + (0 - y1) * (x2 - x1) / (y2 - y1)
    
    return None


def compute_flip_line(
    gex_by_strike: Dict[float, float],
    spot: float,
    grid_range: float = 20.0,
    grid_step: float = STRIKE_GRID_STEP
) -> Dict:
    """
    Compute flip line using cumulative GEX method.
    
    The flip line is the strike where cumulative GEX (summing from lowest to 
    highest strike) crosses zero. This represents the price level where 
    dealer gamma exposure shifts from negative to positive (or vice versa).
    
    Also generates a net_series for charting that shows cumulative GEX at each strike.
    
    Args:
        gex_by_strike: Aggregated GEX by strike
        spot: Current spot price
        grid_range: Not used in cumulative method (kept for compatibility)
        grid_step: Not used in cumulative method (kept for compatibility)
    
    Returns:
        Dict with flip_price, net_series, net_gex_at_spot
    """
    if not gex_by_strike:
        return {
            "flip_price": None,
            "net_series": [],
            "net_gex_at_spot": 0
        }
    
    # Sort strikes and compute cumulative GEX
    sorted_strikes = sorted(gex_by_strike.keys())
    
    cumulative = 0
    cumulative_series = []
    
    for strike in sorted_strikes:
        gex = gex_by_strike[strike]
        cumulative += gex
        cumulative_series.append((strike, cumulative))
    
    # Find zero crossings in cumulative GEX
    flip_price = None
    crossings = []
    
    for i in range(len(cumulative_series) - 1):
        strike1, cum1 = cumulative_series[i]
        strike2, cum2 = cumulative_series[i + 1]
        
        # Check for sign change (zero crossing)
        if cum1 * cum2 < 0:
            # Linear interpolation to find exact crossing point
            if abs(cum2 - cum1) > 1e-10:
                crossing = strike1 + (0 - cum1) * (strike2 - strike1) / (cum2 - cum1)
            else:
                crossing = strike1
            crossings.append(crossing)
    
    # Select the flip line closest to spot price
    if crossings:
        # Filter crossings within reasonable range of spot (±15%)
        valid_crossings = [c for c in crossings if abs(c - spot) / spot < 0.15]
        
        if valid_crossings:
            # Pick the one closest to spot
            flip_price = min(valid_crossings, key=lambda x: abs(x - spot))
        else:
            # If none within range, pick closest overall
            flip_price = min(crossings, key=lambda x: abs(x - spot))
    
    # Compute cumulative GEX at spot price (interpolated)
    net_gex_at_spot = 0
    for i in range(len(cumulative_series) - 1):
        strike1, cum1 = cumulative_series[i]
        strike2, cum2 = cumulative_series[i + 1]
        
        if strike1 <= spot <= strike2:
            # Interpolate
            t = (spot - strike1) / (strike2 - strike1) if strike2 != strike1 else 0
            net_gex_at_spot = cum1 + t * (cum2 - cum1)
            break
    else:
        # Spot outside range - use nearest bound
        if spot < sorted_strikes[0]:
            net_gex_at_spot = cumulative_series[0][1]
        else:
            net_gex_at_spot = cumulative_series[-1][1]
    
    return {
        "flip_price": flip_price,
        "net_series": cumulative_series,  # Now shows cumulative GEX by strike
        "net_gex_at_spot": net_gex_at_spot
    }


def compute_cumulative_gex(gex_by_strike: Dict[float, float]) -> List[Tuple[float, float]]:
    """
    Compute cumulative gamma exposure by strike.
    
    Returns list of (strike, cumulative_gex) tuples sorted by strike.
    This is the same computation as in compute_flip_line, extracted for reuse.
    
    Args:
        gex_by_strike: Dictionary mapping strike -> GEX value
    
    Returns:
        List of (strike, cumulative_gex) tuples sorted by strike
    """
    if not gex_by_strike:
        return []
    
    strikes = sorted(gex_by_strike.keys())
    cumulative = []
    running_sum = 0.0
    
    for strike in strikes:
        running_sum += gex_by_strike[strike]
        cumulative.append((strike, running_sum))
    
    return cumulative


def compute_gamma_slope(
    cumulative_gex: List[Tuple[float, float]], 
    spot: float, 
    flip_line: Optional[float] = None
) -> Dict[str, Any]:
    """
    Compute gamma slope (first derivative of cumulative gamma).
    
    The slope indicates how hedging pressure changes with price movement:
    - Positive slope: Moves stabilize (pinning effect)
    - Negative slope: Moves accelerate (destabilizing)
    - Near zero: Neutral
    
    Args:
        cumulative_gex: List of (strike, cumulative_gex) tuples
        spot: Current spot price
        flip_line: Optional flip line price for slope calculation at that level
    
    Returns:
        Dict with slope_at_spot, slope_at_flip, slope_bucket, and interpretation
    """
    if len(cumulative_gex) < 2:
        return {
            "slope_at_spot": 0.0,
            "slope_at_flip": None,
            "slope_bucket": "NEUTRAL",
            "interpretation": "Neutral"
        }
    
    # Find points around spot price for slope calculation
    slope_at_spot = 0.0
    slope_at_flip = None
    
    # Calculate slope at spot using nearby points
    for i in range(len(cumulative_gex) - 1):
        strike1, cum1 = cumulative_gex[i]
        strike2, cum2 = cumulative_gex[i + 1]
        
        if strike1 <= spot <= strike2:
            # Linear interpolation for slope
            if strike2 != strike1:
                slope_at_spot = (cum2 - cum1) / (strike2 - strike1)
            else:
                slope_at_spot = 0.0
            break
    else:
        # Spot outside range - use nearest segment
        if spot < cumulative_gex[0][0]:
            # Use first two points
            strike1, cum1 = cumulative_gex[0]
            strike2, cum2 = cumulative_gex[1]
            if strike2 != strike1:
                slope_at_spot = (cum2 - cum1) / (strike2 - strike1)
        else:
            # Use last two points
            strike1, cum1 = cumulative_gex[-2]
            strike2, cum2 = cumulative_gex[-1]
            if strike2 != strike1:
                slope_at_spot = (cum2 - cum1) / (strike2 - strike1)
    
    # Calculate slope at flip line if provided
    if flip_line is not None:
        for i in range(len(cumulative_gex) - 1):
            strike1, cum1 = cumulative_gex[i]
            strike2, cum2 = cumulative_gex[i + 1]
            
            if strike1 <= flip_line <= strike2:
                if strike2 != strike1:
                    slope_at_flip = (cum2 - cum1) / (strike2 - strike1)
                else:
                    slope_at_flip = 0.0
                break
    
    # Bucket classification
    SLOPE_THRESHOLD = 0.1  # Threshold for significant slope
    if slope_at_spot > SLOPE_THRESHOLD:
        bucket = "STABILIZING"
        interpretation = "Moves stabilize"
    elif slope_at_spot < -SLOPE_THRESHOLD:
        bucket = "ACCELERATIVE"
        interpretation = "Moves accelerate"
    else:
        bucket = "NEUTRAL"
        interpretation = "Neutral"
    
    return {
        "slope_at_spot": slope_at_spot,
        "slope_at_flip": slope_at_flip,
        "slope_bucket": bucket,
        "interpretation": interpretation
    }


# ===============================
# Transition Detection
# ===============================
# Store last flip price for movement detection
_last_flip_cache = {}


def detect_transition(
    spot: float,
    flip: Optional[float],
    ticker: str = "SPY",
    distance_threshold: float = TRANSITION_DISTANCE_PTS
) -> Dict:
    """
    Detect if market is in a regime transition state.
    
    Transition is flagged when:
    1. Spot is within threshold distance of flip line
    2. Flip line has moved significantly in the last hour
    
    Args:
        spot: Current spot price
        flip: Current flip line price
        ticker: Ticker symbol (for caching)
        distance_threshold: Points from flip to trigger transition
    
    Returns:
        Dict with transition flag and reason
    """
    global _last_flip_cache
    
    transition = False
    reason = None
    
    if flip is not None:
        # Check distance to flip
        distance_to_flip = abs(spot - flip)
        if distance_to_flip <= distance_threshold:
            transition = True
            reason = f"Spot within {distance_threshold} pts of flip"
        
        # Check flip movement (would need historical tracking for production)
        # For now, just track if flip changed significantly from last call
        cache_key = f"{ticker}_flip"
        if cache_key in _last_flip_cache:
            last_flip, last_time = _last_flip_cache[cache_key]
            flip_move = abs(flip - last_flip)
            
            # Check if within last hour and moved significantly
            time_diff = (datetime.now() - last_time).total_seconds()
            if time_diff < 3600 and flip_move >= TRANSITION_FLIP_MOVE_PTS_1H:
                transition = True
                reason = f"Flip moved {flip_move:.1f} pts in {time_diff/60:.0f}m"
        
        # Update cache
        _last_flip_cache[cache_key] = (flip, datetime.now())
    
    return {
        "transition": transition,
        "reason": reason,
        "distance_to_flip": abs(spot - flip) if flip else None
    }


# ===============================
# Main Cockpit State Computation
# ===============================
def compute_cockpit_state(
    snap: dict,
    spot: float,
    now: Optional[datetime] = None,
    ticker: str = "SPY",
    num_strikes_each_side: int = 20,
    strike_increment: float = 1.0
) -> Dict:
    """
    Compute complete cockpit state with multi-lens analysis.
    
    Uses SYMMETRIC strike selection: ±N strikes around ATM.
    This ensures balanced call/put coverage for accurate flip line calculation.
    
    Args:
        snap: API response with options chain
        spot: Current spot price
        now: Current datetime (default: now in Eastern time)
        ticker: Ticker symbol
        num_strikes_each_side: Number of strikes each side of ATM (default 40)
        strike_increment: Strike price increment (default $1 for SPY)
    
    Returns:
        Dict with regime, flip, walls, net_series, and diagnostics
    """
    if now is None:
        now = get_eastern_now()
    
    today = now.date()
    opex = next_monthly_opex(today)
    
    # === SYMMETRIC STRIKE SELECTION ===
    # Process contracts with symmetric strike selection
    contracts, diagnostics = process_chain_symmetric(
        snap, spot, num_strikes_each_side, strike_increment
    )
    
    if not contracts:
        return {
            "spot": spot,
            "flip": None,
            "net_gex": 0,
            "regime": "UNKNOWN",
            "transition": {"transition": False, "reason": None, "distance_to_flip": None},
            "walls_regime": {"put": None, "call": None},
            "walls_tactical": {"put": None, "call": None},
            "walls_today": {"put": None, "call": None},
            "net_series": [],
            "contracts_processed": 0,
            "contracts_total": len(snap.get("results", [])),
            "diagnostics": diagnostics
        }
    
    # === REGIME LENS (0-60 DTE) ===
    regime_filter = lambda exp: dte_in_trading_days(today, exp) <= REGIME_DTE_MAX
    regime_weight = lambda exp: bucket_weight(exp, today, opex)
    
    gex_regime = aggregate_gex_by_strike(contracts, spot, regime_filter, regime_weight)
    
    # Ensure all target strikes have entries
    target_strikes = build_symmetric_strike_set(spot, num_strikes_each_side, strike_increment)
    for strike in target_strikes:
        if strike not in gex_regime:
            gex_regime[strike] = 0.0
    
    flip_info = compute_flip_line(gex_regime, spot)
    
    flip = flip_info["flip_price"]
    net_gex = flip_info["net_gex_at_spot"]
    net_series = flip_info["net_series"]
    
    # Determine regime - but check for data quality first
    # If selection is too asymmetric, regime is suspect
    if not diagnostics.get("is_symmetric", True):
        regime = "UNKNOWN"
    elif net_gex >= 0:
        regime = "POSITIVE_GAMMA"
    else:
        regime = "NEGATIVE_GAMMA"
    
    # === REGIME LENS (0-60 DTE) - GEX + OI walls ===
    regime_call_gex, regime_put_gex = aggregate_gex_by_strike_and_right(
        contracts, spot, regime_filter, regime_weight
    )
    regime_call_oi, regime_put_oi = aggregate_oi_by_strike_and_right(
        contracts, regime_filter
    )
    walls_regime = compute_walls_multi(
        regime_call_gex, regime_put_gex, 
        regime_call_oi, regime_put_oi, 
        spot, k=3
    )
    
    # === TACTICAL LENS (0-14 DTE) - GEX + OI walls ===
    tactical_filter = lambda exp: dte_in_trading_days(today, exp) <= TACTICAL_DTE_MAX
    tactical_weight = lambda exp: bucket_weight(exp, today, opex)
    
    tactical_call_gex, tactical_put_gex = aggregate_gex_by_strike_and_right(
        contracts, spot, tactical_filter, tactical_weight
    )
    tactical_call_oi, tactical_put_oi = aggregate_oi_by_strike_and_right(
        contracts, tactical_filter
    )
    walls_tactical = compute_walls_multi(
        tactical_call_gex, tactical_put_gex,
        tactical_call_oi, tactical_put_oi,
        spot, k=3
    )
    
    # === TODAY LENS (0-2 DTE) - GEX + OI walls ===
    today_filter = lambda exp: dte_in_trading_days(today, exp) <= TODAY_DTE_MAX
    today_weight = lambda exp: 1.0  # Full weight for today's expiries
    
    today_call_gex, today_put_gex = aggregate_gex_by_strike_and_right(
        contracts, spot, today_filter, today_weight
    )
    today_call_oi, today_put_oi = aggregate_oi_by_strike_and_right(
        contracts, today_filter
    )
    walls_today = compute_walls_multi(
        today_call_gex, today_put_gex,
        today_call_oi, today_put_oi,
        spot, k=3
    )
    
    # === TRANSITION DETECTION ===
    transition = detect_transition(spot, flip, ticker)
    
    # Calculate total GEX by side for transparency
    total_call_gex = sum(regime_call_gex.values())
    total_put_gex = sum(regime_put_gex.values())
    
    return {
        "spot": spot,
        "flip": flip,
        "net_gex": net_gex,
        "call_gex_total": total_call_gex,
        "put_gex_total": total_put_gex,
        "regime": regime,
        "transition": transition,
        "walls_regime": walls_regime,
        "walls_tactical": walls_tactical,
        "walls_today": walls_today,
        "net_series": net_series,
        "contracts_processed": diagnostics.get("contracts_processed", len(contracts)),
        "contracts_total": len(snap.get("results", [])),
        "opex": opex.isoformat() if opex else None,
        "diagnostics": {
            "calls_in_target": diagnostics.get("calls_in_target", 0),
            "puts_in_target": diagnostics.get("puts_in_target", 0),
            "is_symmetric": diagnostics.get("is_symmetric", False),
            "target_strikes": diagnostics.get("target_strikes", 0),
            "missing_from_api": diagnostics.get("missing_from_api", 0),
            "returned_but_unusable": diagnostics.get("returned_but_unusable", 0),
            "num_expiries": diagnostics.get("num_expiries", 0)
        }
    }


# ===============================
# Legacy Compatibility Functions
# ===============================
def calculate_gex(
    snap: dict,
    spot_price: float,
    massive_api_key: Optional[str] = None,
    alphavantage_api_key: Optional[str] = None,
    num_strikes_each_side: int = 20,
    strike_increment: float = 1.0
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict, int]:
    """
    Calculate GEX with symmetric strike selection.
    
    Args:
        snap: API response with options chain
        spot_price: Current spot price
        massive_api_key: Not used (kept for compatibility)
        alphavantage_api_key: Not used (kept for compatibility)
        num_strikes_each_side: Number of strikes each side of ATM (default 40)
        strike_increment: Strike price increment (default $1 for SPY)
    
    Returns:
        - df: DataFrame with individual contract data
        - agg: DataFrame with aggregated GEX by strike
        - metrics: Dictionary with total_gex, put_wall, call_wall, flip_line, regime
        - skipped: Number of contracts skipped
    """
    # Use symmetric strike selection
    contracts, diagnostics = process_chain_symmetric(
        snap, spot_price, num_strikes_each_side, strike_increment
    )
    skipped = diagnostics.get("returned_but_unusable", 0) + diagnostics.get("missing_from_api", 0)
    
    if not contracts:
        return pd.DataFrame(), pd.DataFrame(), {"diagnostics": diagnostics}, skipped
    
    # Build DataFrames
    df_rows = []
    for c in contracts:
        gex = contract_dealer_gex(c["gamma"], c["oi"], spot_price, c["type"], c["mult"])
        df_rows.append({
            "strike": c["strike"],
            "type": c["type"],
            "gex": gex,
            "oi": c["oi"],
            "gamma": c["gamma"],
            "expiry": c["expiry_str"]
        })
    
    df = pd.DataFrame(df_rows)
    
    # Aggregate by strike - ensure all target strikes are included
    target_strikes = build_symmetric_strike_set(spot_price, num_strikes_each_side, strike_increment)
    
    agg = df.groupby("strike", as_index=False)["gex"].sum().sort_values("strike")
    
    # Add missing strikes with 0 GEX
    existing_strikes = set(agg["strike"].values)
    missing_strikes = [s for s in target_strikes if s not in existing_strikes]
    if missing_strikes:
        missing_df = pd.DataFrame({"strike": missing_strikes, "gex": [0.0] * len(missing_strikes)})
        agg = pd.concat([agg, missing_df], ignore_index=True).sort_values("strike")
    
    agg["cumulative_gex"] = agg["gex"].cumsum()
    
    # Compute cockpit state for metrics with same parameters
    state = compute_cockpit_state(
        snap, spot_price, 
        num_strikes_each_side=num_strikes_each_side, 
        strike_increment=strike_increment
    )
    
    metrics = {
        "total_gex": state["net_gex"],
        "put_wall": state["walls_regime"]["put"],
        "call_wall": state["walls_regime"]["call"],
        "flip_line": state["flip"],
        "spot_price": spot_price,
        "regime": state["regime"].lower().replace("_gamma", ""),
        "transition": state["transition"]["transition"],
        "walls_tactical": state["walls_tactical"],
        "walls_today": state["walls_today"],
        "diagnostics": {
            "calls_in_target": diagnostics.get("calls_in_target", 0),
            "puts_in_target": diagnostics.get("puts_in_target", 0),
            "is_symmetric": diagnostics.get("is_symmetric", False),
            "target_strikes": len(target_strikes),
            "missing_from_api": diagnostics.get("missing_from_api", 0),
            "returned_but_unusable": diagnostics.get("returned_but_unusable", 0)
        }
    }
    
    return df, agg, metrics, skipped


def calculate_flip_line(agg_df: pd.DataFrame, spot_price: Optional[float] = None) -> Optional[float]:
    """
    Legacy flip line calculation for backward compatibility.
    """
    if len(agg_df) == 0:
        return None
    
    # Convert DataFrame to dict
    gex_by_strike = dict(zip(agg_df["strike"], agg_df["gex"]))
    
    if spot_price is None:
        spot_price = (agg_df["strike"].min() + agg_df["strike"].max()) / 2
    
    flip_info = compute_flip_line(gex_by_strike, spot_price)
    return flip_info["flip_price"]


def get_spot_price_comparison(
    ticker: str,
    massive_api_key: Optional[str] = None,
    alphavantage_api_key: Optional[str] = None
) -> Dict[str, Optional[float]]:
    """Get spot price from all sources for comparison"""
    result = {}
    
    if massive_api_key:
        result['massive'] = get_underlying_spot_price_v2(massive_api_key, ticker)
    
    if alphavantage_api_key:
        result['alphavantage'] = get_spot_from_alpha_vantage(alphavantage_api_key, ticker)
    
    result['yfinance'] = get_spot_from_yfinance(ticker)
    
    return result
