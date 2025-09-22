"""
Implied volatility solver using Black-Scholes-Merton model
"""

import numpy as np
from scipy.optimize import brentq
from scipy.stats import norm
from typing import Optional
import math

class IVSolver:
    """Black-Scholes implied volatility solver for options"""
    
    def __init__(self, risk_free_rate: float = 0.02):
        self.risk_free_rate = risk_free_rate
    
    def black_scholes_put(self, S: float, K: float, T: float, r: float, sigma: float) -> float:
        """Calculate Black-Scholes put price"""
        if T <= 0:
            return max(0, K - S)
        
        if sigma <= 0:
            return max(0, K - S)
        
        d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)
        
        put_price = K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
        return max(0, put_price)
    
    def implied_vol_put(self, mid_price: float, S: float, K: float, T: float, r: Optional[float] = None) -> float:
        """
        Solve for implied volatility of a put option using Brent's method
        
        Args:
            mid_price: Market price of the option
            S: Current stock price
            K: Strike price
            T: Time to expiration in years
            r: Risk-free rate (uses default if None)
        
        Returns:
            Implied volatility (sigma)
        """
        if r is None:
            r = self.risk_free_rate
        
        if T <= 0 or mid_price <= 0 or S <= 0 or K <= 0:
            return 0.2  # Default IV
        
        # Handle edge cases
        intrinsic_value = max(0, K - S)
        if mid_price <= intrinsic_value:
            return 0.01  # Minimum IV
        
        if mid_price >= K:
            return 5.0  # Maximum IV
        
        def objective(sigma):
            """Objective function: difference between model and market price"""
            try:
                model_price = self.black_scholes_put(S, K, T, r, sigma)
                return model_price - mid_price
            except (OverflowError, ValueError, ZeroDivisionError):
                return float('inf')
        
        try:
            # Use Brent's method to find root
            # Search in reasonable IV range [0.01, 5.0]
            iv = brentq(objective, 0.01, 5.0, xtol=1e-6, maxiter=100)
            
            # Clamp to reasonable range
            return max(0.01, min(5.0, iv))
            
        except (ValueError, RuntimeError):
            # If root finding fails, try binary search
            return self._binary_search_iv(objective, 0.01, 5.0)
    
    def _binary_search_iv(self, objective_func, min_iv: float, max_iv: float, tolerance: float = 1e-6) -> float:
        """Binary search for implied volatility"""
        for _ in range(50):  # Max 50 iterations
            mid_iv = (min_iv + max_iv) / 2
            try:
                error = objective_func(mid_iv)
                if abs(error) < tolerance:
                    return mid_iv
                elif error > 0:
                    max_iv = mid_iv
                else:
                    min_iv = mid_iv
            except (OverflowError, ValueError, ZeroDivisionError):
                # If function fails, return middle of range
                return (min_iv + max_iv) / 2
        
        return (min_iv + max_iv) / 2
    
    def estimate_iv_from_delta(self, delta: float, S: float, K: float, T: float, r: Optional[float] = None) -> float:
        """
        Estimate implied volatility from delta using approximation
        
        For puts: delta = -N(-d1) where d1 = (ln(S/K) + (r + 0.5*sigma^2)*T) / (sigma*sqrt(T))
        """
        if r is None:
            r = self.risk_free_rate
        
        if T <= 0 or S <= 0 or K <= 0:
            return 0.2
        
        # Convert put delta to d1
        # For puts: delta = -N(-d1), so -delta = N(-d1), so -d1 = N^(-1)(-delta)
        try:
            d1_neg = norm.ppf(-delta)  # -d1
            d1 = -d1_neg
            
            # Solve for sigma: d1 = (ln(S/K) + (r + 0.5*sigma^2)*T) / (sigma*sqrt(T))
            # Rearranging: sigma^2*T - 2*d1*sigma*sqrt(T) + 2*ln(S/K) + 2*r*T = 0
            
            a = T
            b = -2 * d1 * np.sqrt(T)
            c = 2 * np.log(S / K) + 2 * r * T
            
            # Quadratic formula
            discriminant = b**2 - 4 * a * c
            if discriminant >= 0:
                sigma1 = (-b + np.sqrt(discriminant)) / (2 * a)
                sigma2 = (-b - np.sqrt(discriminant)) / (2 * a)
                
                # Choose positive solution
                sigma = max(sigma1, sigma2)
                return max(0.01, min(5.0, sigma))
            else:
                return 0.2  # Default if no real solution
                
        except (ValueError, OverflowError):
            return 0.2  # Default if calculation fails
    
    def get_iv_for_option(self, option_quote, spot_price: float, time_to_expiry: float) -> float:
        """
        Get implied volatility for an option, using provided IV or solving if missing
        
        Args:
            option_quote: OptionQuote object
            spot_price: Current stock price
            time_to_expiry: Time to expiration in years
        
        Returns:
            Implied volatility
        """
        # Use provided IV if available and reasonable
        if (option_quote.implied_volatility is not None and 
            0.01 <= option_quote.implied_volatility <= 5.0):
            return option_quote.implied_volatility
        
        # Try to estimate from delta if available
        if (option_quote.delta is not None and 
            -1.0 <= option_quote.delta <= 0.0):  # Put delta should be negative
            try:
                return self.estimate_iv_from_delta(
                    option_quote.delta, spot_price, option_quote.strike, time_to_expiry
                )
            except:
                pass
        
        # Fall back to solving from mid price
        return self.implied_vol_put(
            option_quote.mid, spot_price, option_quote.strike, time_to_expiry
        )

