"""
Probability calculations for options strategies
"""

import numpy as np
from scipy.stats import norm
from typing import Tuple

class ProbabilityCalculator:
    """Calculate probability of profit and other option probabilities"""
    
    def __init__(self, risk_free_rate: float = 0.02):
        self.risk_free_rate = risk_free_rate
    
    def pop_short_put(self, S: float, K_short: float, T: float, r: float, sigma: float) -> float:
        """
        Calculate probability of profit for short put (bull put spread)
        
        This is the probability that the stock price stays above the short strike
        at expiration, meaning the short put expires worthless.
        
        Args:
            S: Current stock price
            K_short: Short put strike price
            T: Time to expiration in years
            r: Risk-free rate
            sigma: Implied volatility
        
        Returns:
            Probability of profit (0.0 to 1.0)
        """
        if T <= 0 or sigma <= 0:
            return 0.5  # Default to 50% if invalid inputs
        
        # For a short put, we profit if S_T >= K_short
        # P(S_T >= K_short) = P(ln(S_T) >= ln(K_short))
        # Under risk-neutral measure: ln(S_T) ~ N(ln(S) + (r - 0.5*sigma^2)*T, sigma^2*T)
        
        d = (np.log(S / K_short) + (r - 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
        
        # P(S_T >= K_short) = P(Z >= -d) = 1 - P(Z <= -d) = 1 - N(-d) = N(d)
        return norm.cdf(d)
    
    def pop_bull_put_spread(self, S: float, K_short: float, K_long: float, T: float, 
                           r: float, sigma: float) -> float:
        """
        Calculate probability of profit for bull put spread
        
        For a bull put spread:
        - We profit if S_T >= K_short (both puts expire worthless)
        - We lose max if S_T <= K_long (both puts are in the money)
        - Partial loss if K_long < S_T < K_short
        
        Args:
            S: Current stock price
            K_short: Short put strike (higher strike)
            K_long: Long put strike (lower strike)
            T: Time to expiration in years
            r: Risk-free rate
            sigma: Implied volatility
        
        Returns:
            Probability of profit (0.0 to 1.0)
        """
        if K_short <= K_long:
            return 0.0  # Invalid spread
        
        # For bull put spread, we profit if S_T >= K_short
        return self.pop_short_put(S, K_short, T, r, sigma)
    
    def calculate_delta_put(self, S: float, K: float, T: float, r: float, sigma: float) -> float:
        """
        Calculate delta for a put option using Black-Scholes
        
        Args:
            S: Current stock price
            K: Strike price
            T: Time to expiration in years
            r: Risk-free rate
            sigma: Implied volatility
        
        Returns:
            Put delta (negative value)
        """
        if T <= 0 or sigma <= 0:
            return -0.5  # Default delta
        
        d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
        return norm.cdf(d1) - 1.0  # Put delta is negative
    
    def calculate_gamma_put(self, S: float, K: float, T: float, r: float, sigma: float) -> float:
        """
        Calculate gamma for a put option
        
        Args:
            S: Current stock price
            K: Strike price
            T: Time to expiration in years
            r: Risk-free rate
            sigma: Implied volatility
        
        Returns:
            Put gamma (positive value)
        """
        if T <= 0 or sigma <= 0 or S <= 0:
            return 0.0
        
        d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
        return norm.pdf(d1) / (S * sigma * np.sqrt(T))
    
    def calculate_theta_put(self, S: float, K: float, T: float, r: float, sigma: float) -> float:
        """
        Calculate theta for a put option
        
        Args:
            S: Current stock price
            K: Strike price
            T: Time to expiration in years
            r: Risk-free rate
            sigma: Implied volatility
        
        Returns:
            Put theta (negative value, time decay)
        """
        if T <= 0 or sigma <= 0 or S <= 0:
            return 0.0
        
        d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)
        
        theta = (-S * norm.pdf(d1) * sigma / (2 * np.sqrt(T)) - 
                 r * K * np.exp(-r * T) * norm.cdf(-d2))
        
        return theta
    
    def calculate_vega_put(self, S: float, K: float, T: float, r: float, sigma: float) -> float:
        """
        Calculate vega for a put option
        
        Args:
            S: Current stock price
            K: Strike price
            T: Time to expiration in years
            r: Risk-free rate
            sigma: Implied volatility
        
        Returns:
            Put vega (positive value)
        """
        if T <= 0 or sigma <= 0 or S <= 0:
            return 0.0
        
        d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
        return S * norm.pdf(d1) * np.sqrt(T)
    
    def calculate_rho_put(self, S: float, K: float, T: float, r: float, sigma: float) -> float:
        """
        Calculate rho for a put option
        
        Args:
            S: Current stock price
            K: Strike price
            T: Time to expiration in years
            r: Risk-free rate
            sigma: Implied volatility
        
        Returns:
            Put rho (negative value)
        """
        if T <= 0 or sigma <= 0 or S <= 0:
            return 0.0
        
        d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)
        
        return -K * T * np.exp(-r * T) * norm.cdf(-d2)
    
    def calculate_all_greeks(self, S: float, K: float, T: float, r: float, sigma: float) -> Tuple[float, float, float, float, float]:
        """
        Calculate all Greeks for a put option
        
        Returns:
            Tuple of (delta, gamma, theta, vega, rho)
        """
        delta = self.calculate_delta_put(S, K, T, r, sigma)
        gamma = self.calculate_gamma_put(S, K, T, r, sigma)
        theta = self.calculate_theta_put(S, K, T, r, sigma)
        vega = self.calculate_vega_put(S, K, T, r, sigma)
        rho = self.calculate_rho_put(S, K, T, r, sigma)
        
        return delta, gamma, theta, vega, rho

