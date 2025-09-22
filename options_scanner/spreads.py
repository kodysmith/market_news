"""
Bull put spread analysis and scoring
"""

import numpy as np
from typing import List, Dict, Any, Tuple
from dataclasses import dataclass
from datetime import datetime, timedelta
import math

from fetch_alpha_vantage import OptionQuote
from iv_solver import IVSolver
from probability import ProbabilityCalculator

@dataclass
class BullPutSpread:
    """Represents a bull put spread opportunity"""
    ticker: str
    expiry: str
    short_strike: float
    long_strike: float
    width: float
    credit: float
    max_loss: float
    pop: float
    ev: float
    dte: int
    short_iv: float
    long_iv: float
    bid_ask_width: float
    oi_short: int
    oi_long: int
    vol_short: int
    vol_long: int
    fill_score: float
    spread_id: str
    timestamp: str

class SpreadAnalyzer:
    """Analyze and score bull put spreads"""
    
    def __init__(self, risk_free_rate: float = 0.02):
        self.risk_free_rate = risk_free_rate
        self.iv_solver = IVSolver(risk_free_rate)
        self.prob_calc = ProbabilityCalculator(risk_free_rate)
    
    def build_bull_put_candidates(self, put_options: List[OptionQuote], spot_price: float, 
                                 min_dte: int = 20, max_dte: int = 45) -> List[BullPutSpread]:
        """
        Build candidate bull put spreads from put options
        
        Args:
            put_options: List of put options
            spot_price: Current stock price
            min_dte: Minimum days to expiration
            max_dte: Maximum days to expiration
        
        Returns:
            List of candidate bull put spreads
        """
        candidates = []
        
        # Filter by DTE
        today = datetime.now().date()
        filtered_options = []
        
        for option in put_options:
            try:
                expiry_date = datetime.strptime(option.expiry, '%Y-%m-%d').date()
                dte = (expiry_date - today).days
                
                if min_dte <= dte <= max_dte and option.strike < spot_price:
                    filtered_options.append(option)
            except ValueError:
                continue
        
        # Sort by strike price (descending for bull put spreads)
        filtered_options.sort(key=lambda x: x.strike, reverse=True)
        
        # Group by expiration date
        by_expiry = {}
        for option in filtered_options:
            if option.expiry not in by_expiry:
                by_expiry[option.expiry] = []
            by_expiry[option.expiry].append(option)
        
        # Build spreads for each expiration
        for expiry, options in by_expiry.items():
            if len(options) < 2:
                continue
            
            # Try different width spreads
            for width in [1.0, 2.0, 5.0, 10.0]:
                spreads = self._build_spreads_for_width(options, spot_price, expiry, width)
                candidates.extend(spreads)
        
        return candidates
    
    def _build_spreads_for_width(self, options: List[OptionQuote], spot_price: float, 
                                expiry: str, target_width: float) -> List[BullPutSpread]:
        """Build spreads for a specific width"""
        spreads = []
        
        for i in range(len(options) - 1):
            short_put = options[i]  # Higher strike
            long_put = options[i + 1]  # Lower strike
            
            actual_width = short_put.strike - long_put.strike
            
            # Check if width is close to target (within 20%)
            if abs(actual_width - target_width) / target_width > 0.2:
                continue
            
            # Check liquidity requirements
            if (short_put.open_interest < 100 or long_put.open_interest < 100 or
                short_put.volume < 1):
                continue
            
            # Calculate spread metrics
            spread = self._calculate_spread_metrics(short_put, long_put, spot_price, expiry)
            if spread:
                spreads.append(spread)
        
        return spreads
    
    def _calculate_spread_metrics(self, short_put: OptionQuote, long_put: OptionQuote, 
                                 spot_price: float, expiry: str) -> BullPutSpread:
        """Calculate all metrics for a bull put spread"""
        try:
            # Basic spread parameters
            short_strike = short_put.strike
            long_strike = long_put.strike
            width = short_strike - long_strike
            
            # Calculate credit (use smart mid with NBBO cap)
            short_mid = (short_put.bid + short_put.ask) / 2
            long_mid = (long_put.bid + long_put.ask) / 2
            credit = max(0, short_mid - long_mid)
            
            if credit <= 0:
                return None
            
            # Calculate time to expiration
            expiry_date = datetime.strptime(expiry, '%Y-%m-%d').date()
            today = datetime.now().date()
            dte = (expiry_date - today).days
            T = dte / 365.0
            
            if T <= 0:
                return None
            
            # Get implied volatilities
            short_iv = self.iv_solver.get_iv_for_option(short_put, spot_price, T)
            long_iv = self.iv_solver.get_iv_for_option(long_put, spot_price, T)
            
            # Calculate probability of profit
            pop = self.prob_calc.pop_bull_put_spread(
                spot_price, short_strike, long_strike, T, self.risk_free_rate, short_iv
            )
            
            # Calculate expected value
            max_loss = width - credit
            ev = (pop * credit) - ((1 - pop) * max_loss)
            
            # Calculate bid-ask width
            bid_ask_width = (short_put.ask - short_put.bid) + (long_put.ask - long_put.bid)
            
            # Calculate fill score
            fill_score = self._calculate_fill_score(short_put, long_put, bid_ask_width, spot_price)
            
            # Generate spread ID
            spread_id = f"{short_put.symbol}_{expiry}_{short_strike}_{long_strike}_BULL_PUT"
            
            return BullPutSpread(
                ticker=short_put.symbol,
                expiry=expiry,
                short_strike=short_strike,
                long_strike=long_strike,
                width=width,
                credit=credit,
                max_loss=max_loss,
                pop=pop,
                ev=ev,
                dte=dte,
                short_iv=short_iv,
                long_iv=long_iv,
                bid_ask_width=bid_ask_width,
                oi_short=short_put.open_interest,
                oi_long=long_put.open_interest,
                vol_short=short_put.volume,
                vol_long=long_put.volume,
                fill_score=fill_score,
                spread_id=spread_id,
                timestamp=datetime.now().isoformat()
            )
            
        except Exception as e:
            print(f"Error calculating spread metrics: {e}")
            return None
    
    def _calculate_fill_score(self, short_put: OptionQuote, long_put: OptionQuote, 
                             bid_ask_width: float, spot_price: float) -> float:
        """Calculate fill score based on liquidity and bid-ask spread"""
        # Normalize bid-ask width as percentage of underlying
        width_pct = bid_ask_width / spot_price
        
        # Sigmoid function for bid-ask width (lower is better)
        width_score = 1 / (1 + np.exp(10 * (width_pct - 0.02)))  # Penalty above 2%
        
        # Open interest score (higher is better)
        oi_score = min(1.0, (short_put.open_interest + long_put.open_interest) / 2000)
        
        # Volume score (higher is better)
        vol_score = min(1.0, (short_put.volume + long_put.volume) / 100)
        
        # Combine scores
        fill_score = 0.4 * width_score + 0.4 * oi_score + 0.2 * vol_score
        
        return max(0.0, min(1.0, fill_score))
    
    def apply_filters(self, spreads: List[BullPutSpread], 
                     min_pop: float = 0.50, 
                     min_ev_per_100: float = 0.10,
                     max_bid_ask_pct: float = 0.02) -> List[BullPutSpread]:
        """
        Apply filtering criteria to spreads
        
        Args:
            spreads: List of bull put spreads
            min_pop: Minimum probability of profit
            min_ev_per_100: Minimum EV per $100 collateral
            max_bid_ask_pct: Maximum bid-ask width as % of underlying
        
        Returns:
            Filtered list of spreads
        """
        filtered = []
        
        for spread in spreads:
            # Check probability of profit
            if spread.pop < min_pop:
                continue
            
            # Check EV per $100 collateral
            ev_per_100 = spread.ev / (spread.width * 100)
            if ev_per_100 < min_ev_per_100:
                continue
            
            # Check bid-ask width (assuming we have spot price)
            # This would need to be passed in or calculated differently
            # For now, use absolute bid-ask width
            if spread.bid_ask_width > 0.10:  # $0.10 max width
                continue
            
            # Check fill score
            if spread.fill_score < 0.3:
                continue
            
            filtered.append(spread)
        
        return filtered
    
    def rank_spreads(self, spreads: List[BullPutSpread]) -> List[BullPutSpread]:
        """Rank spreads by expected value and liquidity"""
        def ranking_key(spread):
            # Primary: Expected value
            # Secondary: Fill score
            # Tertiary: Probability of profit
            return (spread.ev, spread.fill_score, spread.pop)
        
        return sorted(spreads, key=ranking_key, reverse=True)
    
    def top_n_spreads(self, spreads: List[BullPutSpread], n: int = 5) -> List[BullPutSpread]:
        """Get top N spreads by ranking"""
        ranked = self.rank_spreads(spreads)
        return ranked[:n]
    
    def spread_to_dict(self, spread: BullPutSpread) -> Dict[str, Any]:
        """Convert spread to dictionary for JSON serialization"""
        return {
            'ticker': spread.ticker,
            'strategy': 'BULL_PUT',
            'expiry': spread.expiry,
            'shortK': spread.short_strike,
            'longK': spread.long_strike,
            'width': spread.width,
            'credit': spread.credit,
            'maxLoss': spread.max_loss,
            'dte': spread.dte,
            'pop': spread.pop,
            'ev': spread.ev,
            'ivShort': spread.short_iv,
            'ivLong': spread.long_iv,
            'bidAskW': spread.bid_ask_width,
            'oiShort': spread.oi_short,
            'oiLong': spread.oi_long,
            'volShort': spread.vol_short,
            'volLong': spread.vol_long,
            'fillScore': spread.fill_score,
            'id': spread.spread_id
        }

