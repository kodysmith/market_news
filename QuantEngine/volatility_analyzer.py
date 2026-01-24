"""
Volatility Analyzer - IV calculation and term structure analysis
Provides volatility state for the Decision Cockpit.
"""

import logging
from typing import Dict, Any, Optional, List, Tuple
from datetime import datetime, timedelta
import yfinance as yf
import numpy as np
from scipy.stats import norm
import math

logger = logging.getLogger(__name__)


class VolatilityAnalyzer:
    """
    Analyzes options implied volatility and VIX to determine volatility state.
    
    Outputs:
    - Front IV (nearest expiration ATM IV)
    - Back IV (next expiration ATM IV)
    - Term structure (contango/flat/inverted)
    - Direction (rising/falling/flat)
    - State (expanding/contracting/normal/elevated/compressed)
    """
    
    # VIX thresholds
    VIX_LOW = 12
    VIX_NORMAL_LOW = 15
    VIX_NORMAL_HIGH = 20
    VIX_ELEVATED = 25
    VIX_HIGH = 30
    
    def __init__(self, ticker: str = 'SPY'):
        self.ticker = ticker.upper()
        self.stock = None
        self.vix_data = None
        self._cache = {}
        self._cache_time = None
        self._cache_duration = 300  # 5 minutes
    
    def analyze(self) -> Dict[str, Any]:
        """
        Perform full volatility analysis.
        
        Returns:
            Dict with front_iv, back_iv, direction, term_structure, state, vix
        """
        try:
            # Get VIX data
            vix_info = self._get_vix_data()
            
            # Get options IV data
            iv_info = self._get_options_iv()
            
            # Combine and determine state
            result = {
                'front_iv': iv_info.get('front_iv'),
                'back_iv': iv_info.get('back_iv'),
                'front_iv_change_1h': iv_info.get('front_iv_change_1h', 0),
                'front_iv_change_1d': iv_info.get('front_iv_change_1d', 0),
                'term_structure': self._determine_term_structure(
                    iv_info.get('front_iv'),
                    iv_info.get('back_iv')
                ),
                'direction': self._determine_direction(
                    iv_info.get('front_iv_change_1h', 0),
                    vix_info.get('vix_change', 0)
                ),
                'vix': vix_info.get('vix'),
                'vix_change': vix_info.get('vix_change'),
                'vix_change_pct': vix_info.get('vix_change_pct'),
                'timestamp': datetime.now().isoformat()
            }
            
            # Determine overall state
            result['state'] = self._determine_state(result)
            
            return result
            
        except Exception as e:
            logger.error(f"Volatility analysis failed: {e}")
            return {
                'front_iv': None,
                'back_iv': None,
                'front_iv_change_1h': 0,
                'front_iv_change_1d': 0,
                'term_structure': 'UNKNOWN',
                'direction': 'FLAT',
                'state': 'UNKNOWN',
                'vix': None,
                'vix_change': None,
                'error': str(e)
            }
    
    def _get_vix_data(self) -> Dict[str, Any]:
        """Get VIX current value and change."""
        try:
            vix = yf.Ticker('^VIX')
            hist = vix.history(period='2d')
            
            if hist.empty:
                return {'vix': None, 'vix_change': None, 'vix_change_pct': None}
            
            current_vix = float(hist['Close'].iloc[-1])
            
            if len(hist) > 1:
                prev_vix = float(hist['Close'].iloc[-2])
                vix_change = current_vix - prev_vix
                vix_change_pct = (vix_change / prev_vix) * 100 if prev_vix else 0
            else:
                vix_change = 0
                vix_change_pct = 0
            
            return {
                'vix': current_vix,
                'vix_change': vix_change,
                'vix_change_pct': vix_change_pct
            }
            
        except Exception as e:
            logger.error(f"Failed to get VIX data: {e}")
            return {'vix': None, 'vix_change': None, 'vix_change_pct': None}
    
    def _get_options_iv(self) -> Dict[str, Any]:
        """
        Get ATM implied volatility from options chain.
        Calculates front-month and back-month IV.
        """
        try:
            stock = yf.Ticker(self.ticker)
            
            # Get current price
            info = stock.info
            spot = info.get('currentPrice') or info.get('regularMarketPrice', 0)
            
            if not spot:
                hist = stock.history(period='1d')
                if not hist.empty:
                    spot = float(hist['Close'].iloc[-1])
            
            if not spot:
                return {'front_iv': None, 'back_iv': None}
            
            # Get expiration dates
            expirations = stock.options
            if not expirations or len(expirations) < 2:
                return {'front_iv': None, 'back_iv': None}
            
            # Get front month (nearest expiration)
            front_exp = expirations[0]
            front_iv = self._get_atm_iv(stock, front_exp, spot)
            
            # Get back month (next expiration, at least 2 weeks out)
            back_iv = None
            for exp in expirations[1:]:
                exp_date = datetime.strptime(exp, '%Y-%m-%d')
                if (exp_date - datetime.now()).days >= 14:
                    back_iv = self._get_atm_iv(stock, exp, spot)
                    break
            
            # If no back month found, use second expiration
            if back_iv is None and len(expirations) > 1:
                back_iv = self._get_atm_iv(stock, expirations[1], spot)
            
            # Estimate IV change (using VIX as proxy since we don't store historical IV)
            vix_info = self._get_vix_data()
            front_iv_change_1h = vix_info.get('vix_change', 0) * 0.8 if front_iv else 0
            front_iv_change_1d = vix_info.get('vix_change', 0) if front_iv else 0
            
            return {
                'front_iv': front_iv,
                'back_iv': back_iv,
                'front_iv_change_1h': front_iv_change_1h,
                'front_iv_change_1d': front_iv_change_1d,
                'spot': spot
            }
            
        except Exception as e:
            logger.error(f"Failed to get options IV: {e}")
            return {'front_iv': None, 'back_iv': None}
    
    def _get_atm_iv(self, stock, expiration: str, spot: float) -> Optional[float]:
        """
        Get ATM implied volatility for a specific expiration.
        Uses the average of ATM call and put IV.
        """
        try:
            chain = stock.option_chain(expiration)
            calls = chain.calls
            puts = chain.puts
            
            if calls.empty and puts.empty:
                return None
            
            # Find ATM strike (closest to spot)
            if not calls.empty:
                strikes = calls['strike'].values
                atm_idx = np.argmin(np.abs(strikes - spot))
                atm_strike = strikes[atm_idx]
                
                # Get call IV
                call_row = calls[calls['strike'] == atm_strike]
                call_iv = float(call_row['impliedVolatility'].iloc[0]) * 100 if not call_row.empty else None
            else:
                call_iv = None
            
            if not puts.empty:
                strikes = puts['strike'].values
                atm_idx = np.argmin(np.abs(strikes - spot))
                atm_strike = strikes[atm_idx]
                
                # Get put IV
                put_row = puts[puts['strike'] == atm_strike]
                put_iv = float(put_row['impliedVolatility'].iloc[0]) * 100 if not put_row.empty else None
            else:
                put_iv = None
            
            # Average call and put IV
            if call_iv and put_iv:
                return (call_iv + put_iv) / 2
            elif call_iv:
                return call_iv
            elif put_iv:
                return put_iv
            else:
                return None
                
        except Exception as e:
            logger.error(f"Failed to get ATM IV for {expiration}: {e}")
            return None
    
    def _determine_term_structure(self, front_iv: Optional[float], back_iv: Optional[float]) -> str:
        """
        Determine term structure shape.
        
        Returns:
            CONTANGO (normal - back > front)
            FLAT
            INVERTED (fear - front > back)
        """
        if front_iv is None or back_iv is None:
            return 'UNKNOWN'
        
        diff = front_iv - back_iv
        
        if diff > 2:
            return 'INVERTED'  # Front > Back (fear, event risk)
        elif diff < -2:
            return 'CONTANGO'  # Back > Front (normal)
        else:
            return 'FLAT'
    
    def _determine_direction(self, iv_change: float, vix_change: float) -> str:
        """
        Determine volatility direction.
        
        Returns:
            RISING, FALLING, or FLAT
        """
        # Use the larger absolute change
        change = iv_change if abs(iv_change) > abs(vix_change) else vix_change
        
        if change > 0.5:
            return 'RISING'
        elif change < -0.5:
            return 'FALLING'
        else:
            return 'FLAT'
    
    def _determine_state(self, data: Dict) -> str:
        """
        Determine overall volatility state.
        
        States:
            EXPANDING - Vol rising, term inverted, or VIX > 25
            CONTRACTING - Vol falling and VIX < 15
            ELEVATED - VIX > 30
            COMPRESSED - VIX < 12
            NORMAL - Everything else
        """
        vix = data.get('vix') or 0
        direction = data.get('direction', 'FLAT')
        term_structure = data.get('term_structure', 'FLAT')
        
        # Check extreme states first
        if vix > self.VIX_HIGH:
            return 'ELEVATED'
        
        if vix < self.VIX_LOW:
            return 'COMPRESSED'
        
        # Check directional states
        if direction == 'RISING':
            if term_structure == 'INVERTED' or vix > self.VIX_ELEVATED:
                return 'EXPANDING'
        
        if direction == 'FALLING':
            if vix < self.VIX_NORMAL_LOW:
                return 'CONTRACTING'
        
        return 'NORMAL'


def analyze_volatility(ticker: str = 'SPY') -> Dict[str, Any]:
    """
    Convenience function to analyze volatility for a ticker.
    
    Args:
        ticker: Stock ticker symbol
    
    Returns:
        Volatility analysis dict
    """
    analyzer = VolatilityAnalyzer(ticker)
    return analyzer.analyze()
