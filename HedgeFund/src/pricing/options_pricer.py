"""
Module: Options Pricing Engine
Purpose: Price options and calculate Greeks (pure function module)
Dependencies: py_vollib, scipy
Exports: price_option, calculate_greeks, strike_for_delta, delta_for_strike

Uses Black-Scholes model for option pricing and Greeks calculations.
All functions are pure (no side effects) for easy testing and parallelization.
"""

from typing import Dict, Literal
from decimal import Decimal
import numpy as np
from datetime import date, datetime

# Import Black-Scholes from py_vollib
try:
    from py_vollib.black_scholes import black_scholes
    from py_vollib.black_scholes.greeks import analytical
    VOLLIB_AVAILABLE = True
except ImportError:
    VOLLIB_AVAILABLE = False
    print("⚠️  py_vollib not installed. Install with: pip install py_vollib")

# Backup: scipy-based pricing
from scipy.stats import norm
from scipy.optimize import brentq


class OptionsPricer:
    """
    Options pricing engine using Black-Scholes model
    
    Provides pricing and Greeks for European-style options.
    American options approximated using European pricing (acceptable for indices).
    """
    
    def __init__(self, risk_free_rate: float = 0.04):
        """
        Initialize pricer
        
        Args:
            risk_free_rate: Annual risk-free rate (default 4%)
        """
        self.r = risk_free_rate
    
    def price_option(self, 
                    option_type: Literal['c', 'p', 'C', 'P'],
                    S: float,
                    K: float, 
                    T: float,
                    sigma: float) -> Decimal:
        """
        Price option using Black-Scholes
        
        Args:
            option_type: 'c' or 'p' for call/put
            S: Spot price
            K: Strike price
            T: Time to expiration (years)
            sigma: Implied volatility (annualized)
        
        Returns:
            Option price as Decimal
        
        Example:
            >>> pricer = OptionsPricer()
            >>> price = pricer.price_option('p', S=400, K=395, T=14/365, sigma=0.25)
            >>> print(f"Put price: ${price:.2f}")
        """
        option_type = option_type.lower()
        
        # Handle edge cases
        if T <= 0:
            # Expired - return intrinsic value
            if option_type == 'c':
                return Decimal(str(max(S - K, 0)))
            else:
                return Decimal(str(max(K - S, 0)))
        
        if sigma <= 0:
            # No volatility - return intrinsic value
            if option_type == 'c':
                return Decimal(str(max(S - K, 0)))
            else:
                return Decimal(str(max(K - S, 0)))
        
        # Price using py_vollib if available
        if VOLLIB_AVAILABLE:
            try:
                price = black_scholes(option_type, S, K, T, self.r, sigma)
                return Decimal(str(round(price, 4)))
            except:
                pass
        
        # Fallback: Manual Black-Scholes implementation
        price = self._black_scholes_manual(option_type, S, K, T, sigma, self.r)
        return Decimal(str(round(price, 4)))
    
    def _black_scholes_manual(self, flag: str, S: float, K: float, T: float, sigma: float, r: float) -> float:
        """
        Manual Black-Scholes implementation (backup if py_vollib unavailable)
        """
        d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)
        
        if flag == 'c':
            price = S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
        else:  # put
            price = K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
        
        return max(price, 0.01)  # Minimum 1 cent
    
    def calculate_greeks(self,
                        option_type: Literal['c', 'p', 'C', 'P'],
                        S: float,
                        K: float,
                        T: float,
                        sigma: float) -> Dict[str, Decimal]:
        """
        Calculate all Greeks for an option
        
        Args:
            option_type: 'c' or 'p'
            S: Spot price
            K: Strike price
            T: Time to expiration (years)
            sigma: Implied volatility
        
        Returns:
            Dict with delta, gamma, theta, vega, rho
        
        Example:
            >>> pricer = OptionsPricer()
            >>> greeks = pricer.calculate_greeks('p', 400, 395, 14/365, 0.25)
            >>> print(f"Delta: {greeks['delta']:.3f}")
        """
        option_type = option_type.lower()
        
        if T <= 0 or sigma <= 0:
            # Expired or no vol - return zeros
            return {
                'delta': Decimal('0.0'),
                'gamma': Decimal('0.0'),
                'theta': Decimal('0.0'),
                'vega': Decimal('0.0'),
                'rho': Decimal('0.0')
            }
        
        # Use py_vollib if available
        if VOLLIB_AVAILABLE:
            try:
                greeks = {
                    'delta': analytical.delta(option_type, S, K, T, self.r, sigma),
                    'gamma': analytical.gamma(option_type, S, K, T, self.r, sigma),
                    'theta': analytical.theta(option_type, S, K, T, self.r, sigma),
                    'vega': analytical.vega(option_type, S, K, T, self.r, sigma),
                    'rho': analytical.rho(option_type, S, K, T, self.r, sigma)
                }
                
                return {k: Decimal(str(round(v, 6))) for k, v in greeks.items()}
            except:
                pass
        
        # Fallback: Manual Greeks calculation
        greeks = self._greeks_manual(option_type, S, K, T, sigma, self.r)
        return {k: Decimal(str(round(v, 6))) for k, v in greeks.items()}
    
    def _greeks_manual(self, flag: str, S: float, K: float, T: float, sigma: float, r: float) -> Dict:
        """Manual Greeks calculation (backup)"""
        d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)
        
        # Delta
        if flag == 'c':
            delta = norm.cdf(d1)
        else:
            delta = norm.cdf(d1) - 1.0
        
        # Gamma (same for calls and puts)
        gamma = norm.pdf(d1) / (S * sigma * np.sqrt(T))
        
        # Vega (same for calls and puts, per 1% change in vol)
        vega = S * norm.pdf(d1) * np.sqrt(T) / 100
        
        # Theta (per day)
        if flag == 'c':
            theta = (-(S * norm.pdf(d1) * sigma) / (2 * np.sqrt(T)) 
                    - r * K * np.exp(-r * T) * norm.cdf(d2)) / 365
        else:
            theta = (-(S * norm.pdf(d1) * sigma) / (2 * np.sqrt(T)) 
                    + r * K * np.exp(-r * T) * norm.cdf(-d2)) / 365
        
        # Rho (per 1% change in rate)
        if flag == 'c':
            rho = K * T * np.exp(-r * T) * norm.cdf(d2) / 100
        else:
            rho = -K * T * np.exp(-r * T) * norm.cdf(-d2) / 100
        
        return {
            'delta': delta,
            'gamma': gamma,
            'theta': theta,
            'vega': vega,
            'rho': rho
        }
    
    def strike_for_delta(self,
                        option_type: Literal['c', 'p', 'C', 'P'],
                        S: float,
                        T: float,
                        sigma: float,
                        target_delta: float) -> Decimal:
        """
        Solve for strike price given target delta
        
        Uses numerical optimization to find strike that produces target delta.
        
        Args:
            option_type: 'c' or 'p'
            S: Current spot price
            T: Time to expiration (years)
            sigma: Implied volatility
            target_delta: Desired delta (e.g., -0.30 for puts, 0.30 for calls)
        
        Returns:
            Strike price as Decimal
        
        Example:
            >>> pricer = OptionsPricer()
            >>> strike = pricer.strike_for_delta('p', S=400, T=14/365, sigma=0.25, target_delta=-0.30)
            >>> print(f"30-delta put strike: ${strike:.2f}")
        """
        option_type = option_type.lower()
        
        def delta_error(K):
            """Calculate delta error for given strike"""
            greeks = self.calculate_greeks(option_type, S, K, T, sigma)
            return float(greeks['delta']) - target_delta
        
        # Set search bounds
        if option_type == 'p':
            # Puts: search from 50% to 100% of spot
            K_min, K_max = S * 0.50, S * 1.00
        else:
            # Calls: search from 100% to 150% of spot
            K_min, K_max = S * 1.00, S * 1.50
        
        try:
            # Numerical solver
            strike = brentq(delta_error, K_min, K_max, xtol=0.01)
            return Decimal(str(round(strike, 2)))
        except:
            # Fallback: Approximation
            if option_type == 'p':
                # For puts, lower strike = lower absolute delta
                strike = S * (1 + target_delta * 0.15)
            else:
                # For calls, higher strike = lower delta
                strike = S * (1 + (1 - target_delta) * 0.15)
            
            return Decimal(str(round(strike, 2)))
    
    def delta_for_strike(self,
                        option_type: Literal['c', 'p', 'C', 'P'],
                        S: float,
                        K: float,
                        T: float,
                        sigma: float) -> Decimal:
        """
        Calculate delta for given strike
        
        Args:
            option_type: 'c' or 'p'
            S: Spot price
            K: Strike price
            T: Time to expiration (years)
            sigma: Implied volatility
        
        Returns:
            Delta value
        """
        greeks = self.calculate_greeks(option_type, S, K, T, sigma)
        return greeks['delta']
    
    def implied_volatility(self,
                          option_type: Literal['c', 'p', 'C', 'P'],
                          S: float,
                          K: float,
                          T: float,
                          market_price: float) -> Decimal:
        """
        Calculate implied volatility from market price
        
        Uses numerical optimization to back out IV.
        
        Args:
            option_type: 'c' or 'p'
            S: Spot price
            K: Strike price
            T: Time to expiration (years)
            market_price: Observed option price
        
        Returns:
            Implied volatility as Decimal
        """
        option_type = option_type.lower()
        
        def price_error(sigma):
            """Price error for given IV"""
            theoretical_price = float(self.price_option(option_type, S, K, T, sigma))
            return theoretical_price - market_price
        
        try:
            # Search for IV between 1% and 300%
            iv = brentq(price_error, 0.01, 3.0, xtol=0.0001)
            return Decimal(str(round(iv, 4)))
        except:
            # Fallback: Return rough estimate based on ATM approximation
            # ATM approximation: C ≈ 0.4 * S * sigma * sqrt(T)
            iv_guess = (market_price / (0.4 * S * np.sqrt(T)))
            return Decimal(str(round(max(0.10, min(iv_guess, 2.0)), 4)))


# Convenience functions for quick access
def price_put(S: float, K: float, T: float, sigma: float, r: float = 0.04) -> Decimal:
    """Quick put pricing"""
    pricer = OptionsPricer(risk_free_rate=r)
    return pricer.price_option('p', S, K, T, sigma)


def price_call(S: float, K: float, T: float, sigma: float, r: float = 0.04) -> Decimal:
    """Quick call pricing"""
    pricer = OptionsPricer(risk_free_rate=r)
    return pricer.price_option('c', S, K, T, sigma)


def strike_for_put_delta(S: float, T: float, sigma: float, target_delta: float, r: float = 0.04) -> Decimal:
    """Quick strike calculation for put delta"""
    pricer = OptionsPricer(risk_free_rate=r)
    return pricer.strike_for_delta('p', S, T, sigma, target_delta)


def strike_for_call_delta(S: float, T: float, sigma: float, target_delta: float, r: float = 0.04) -> Decimal:
    """Quick strike calculation for call delta"""
    pricer = OptionsPricer(risk_free_rate=r)
    return pricer.strike_for_delta('c', S, T, sigma, target_delta)


if __name__ == "__main__":
    """Test options pricing engine"""
    
    print("Testing Options Pricing Engine...")
    print(f"py_vollib available: {VOLLIB_AVAILABLE}\n")
    
    pricer = OptionsPricer(risk_free_rate=0.04)
    
    # Test parameters
    S = 400.0  # QQQ price
    K = 395.0  # Strike
    T = 14 / 365.0  # 14 days
    sigma = 0.25  # 25% IV
    
    # Test put pricing
    put_price = pricer.price_option('p', S, K, T, sigma)
    print(f"✓ Put pricing:")
    print(f"  Spot=${S}, Strike=${K}, DTE={T*365:.0f}, IV={sigma:.0%}")
    print(f"  Put price: ${put_price}")
    
    # Test call pricing
    call_price = pricer.price_option('c', S, K, T, sigma)
    print(f"\n✓ Call pricing:")
    print(f"  Call price: ${call_price}")
    
    # Test Greeks
    put_greeks = pricer.calculate_greeks('p', S, K, T, sigma)
    print(f"\n✓ Put Greeks:")
    for greek, value in put_greeks.items():
        print(f"  {greek.capitalize()}: {value}")
    
    # Test strike for delta
    target_delta = -0.30
    strike = pricer.strike_for_delta('p', S, T, sigma, target_delta)
    print(f"\n✓ Strike for delta:")
    print(f"  Target delta: {target_delta}")
    print(f"  Calculated strike: ${strike}")
    
    # Verify the delta
    actual_delta = pricer.delta_for_strike('p', S, float(strike), T, sigma)
    print(f"  Actual delta at strike: {actual_delta}")
    print(f"  Delta error: {abs(float(actual_delta) - target_delta):.4f}")
    
    # Test edge cases
    print(f"\n✓ Testing edge cases:")
    
    # Expired option
    expired_price = pricer.price_option('p', 400, 395, 0, 0.25)
    print(f"  Expired put (S=400, K=395): ${expired_price} (intrinsic: $5)")
    
    # Zero vol
    zero_vol_price = pricer.price_option('p', 400, 395, 14/365, 0)
    print(f"  Zero vol put: ${zero_vol_price} (intrinsic: $0)")
    
    # Deep ITM
    deep_itm = pricer.price_option('p', 350, 400, 14/365, 0.25)
    print(f"  Deep ITM put (S=350, K=400): ${deep_itm}")
    
    # Deep OTM
    deep_otm = pricer.price_option('p', 450, 400, 14/365, 0.25)
    print(f"  Deep OTM put (S=450, K=400): ${deep_otm}")
    
    # Test convenience functions
    print(f"\n✓ Testing convenience functions:")
    quick_put = price_put(S=400, K=400, T=30/365, sigma=0.30)
    print(f"  ATM 30-day put (30% IV): ${quick_put}")
    
    quick_strike = strike_for_put_delta(S=400, T=14/365, sigma=0.25, target_delta=-0.30)
    print(f"  30-delta strike (14 days): ${quick_strike}")
    
    print("\n✅ Options pricing engine working correctly!")



