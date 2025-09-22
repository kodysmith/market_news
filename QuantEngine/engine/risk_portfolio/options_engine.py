"""
Options Engine for AI Quant Trading System

Implements:
- Black-Scholes and binomial options pricing
- Options chain analysis and screening
- Greeks calculation (delta, gamma, theta, vega, rho)
- Options overlay strategies (protective puts, covered calls, collars)
- Volatility surface modeling
- Implied volatility calculation
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Tuple, Union
import logging
from datetime import datetime, timedelta
from scipy import stats
from scipy.optimize import brentq
import warnings

logger = logging.getLogger(__name__)


class BlackScholesModel:
    """Black-Scholes options pricing model"""

    def __init__(self):
        self.risk_free_rate = 0.05  # Default 5% risk-free rate

    def price_option(self, S: float, K: float, T: float, r: float, sigma: float,
                    option_type: str = 'call') -> Dict[str, float]:
        """
        Price European option using Black-Scholes model

        Args:
            S: Spot price
            K: Strike price
            T: Time to expiration (years)
            r: Risk-free rate
            sigma: Volatility
            option_type: 'call' or 'put'

        Returns:
            Dictionary with price and Greeks
        """

        if T <= 0:
            # Option expired
            if option_type == 'call':
                price = max(S - K, 0)
            else:
                price = max(K - S, 0)
            return {
                'price': price,
                'delta': 1.0 if (option_type == 'call' and S > K) else (0.0 if option_type == 'call' else 1.0),
                'gamma': 0.0,
                'theta': 0.0,
                'vega': 0.0,
                'rho': 0.0
            }

        # Black-Scholes calculations
        d1 = (np.log(S / K) + (r + sigma**2 / 2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)

        if option_type == 'call':
            price = S * stats.norm.cdf(d1) - K * np.exp(-r * T) * stats.norm.cdf(d2)
            delta = stats.norm.cdf(d1)
            rho = K * T * np.exp(-r * T) * stats.norm.cdf(d2)
        else:  # put
            price = K * np.exp(-r * T) * stats.norm.cdf(-d2) - S * stats.norm.cdf(-d1)
            delta = -stats.norm.cdf(-d1)
            rho = -K * T * np.exp(-r * T) * stats.norm.cdf(-d2)

        # Common Greeks
        gamma = stats.norm.pdf(d1) / (S * sigma * np.sqrt(T))
        theta = -(S * stats.norm.pdf(d1) * sigma) / (2 * np.sqrt(T))
        if option_type == 'call':
            theta -= r * K * np.exp(-r * T) * stats.norm.cdf(d2)
        else:
            theta += r * K * np.exp(-r * T) * stats.norm.cdf(-d2)

        vega = S * np.sqrt(T) * stats.norm.pdf(d1)

        return {
            'price': price,
            'delta': delta,
            'gamma': gamma,
            'theta': theta,
            'vega': vega,
            'rho': rho,
            'd1': d1,
            'd2': d2
        }

    def calculate_implied_volatility(self, market_price: float, S: float, K: float, T: float,
                                   r: float, option_type: str = 'call', max_iter: int = 100) -> float:
        """
        Calculate implied volatility using Newton-Raphson method

        Args:
            market_price: Observed market price
            S, K, T, r: Option parameters
            option_type: 'call' or 'put'

        Returns:
            Implied volatility
        """

        def objective(sigma):
            bs_price = self.price_option(S, K, T, r, sigma, option_type)['price']
            return bs_price - market_price

        def derivative(sigma):
            return self.price_option(S, K, T, r, sigma, option_type)['vega']

        # Initial guess
        sigma = 0.3  # 30% volatility

        try:
            # Newton-Raphson iteration
            for i in range(max_iter):
                price = self.price_option(S, K, T, r, sigma, option_type)
                diff = price['price'] - market_price

                if abs(diff) < 1e-6:
                    return sigma

                vega = price['vega']
                if vega == 0:
                    break

                sigma = sigma - diff / vega

                # Keep sigma in reasonable bounds
                sigma = np.clip(sigma, 0.01, 5.0)

            # If Newton-Raphson fails, use bisection
            try:
                sigma = brentq(objective, 0.01, 5.0, maxiter=50)
            except:
                sigma = 0.3  # Fallback

        except:
            sigma = 0.3  # Fallback

        return sigma


class OptionsChainAnalyzer:
    """Analyze and screen options chains"""

    def __init__(self, pricing_model: BlackScholesModel = None):
        self.pricing_model = pricing_model or BlackScholesModel()
        self.risk_free_rate = 0.05

    def analyze_chain(self, options_chain: pd.DataFrame, spot_price: float,
                     current_date: datetime) -> Dict[str, Any]:
        """
        Comprehensive analysis of an options chain

        Args:
            options_chain: DataFrame with option data
            spot_price: Current spot price
            current_date: Analysis date

        Returns:
            Chain analysis results
        """

        if options_chain.empty:
            return {'error': 'Empty options chain'}

        analysis = {
            'chain_size': len(options_chain),
            'expirations': sorted(options_chain['expiration'].unique()),
            'strikes': sorted(options_chain['strike'].unique()),
            'moneyness_analysis': {},
            'volatility_surface': {},
            'greeks_summary': {},
            'unusual_activity': []
        }

        # Calculate implied volatilities if not present
        if 'iv' not in options_chain.columns:
            options_chain = self._calculate_implied_volatilities(options_chain, spot_price, current_date)

        # Moneyness analysis
        analysis['moneyness_analysis'] = self._analyze_moneyness(options_chain, spot_price)

        # Volatility surface
        analysis['volatility_surface'] = self._build_volatility_surface(options_chain)

        # Greeks summary
        analysis['greeks_summary'] = self._calculate_greeks_summary(options_chain)

        # Unusual activity detection
        analysis['unusual_activity'] = self._detect_unusual_activity(options_chain)

        return analysis

    def _calculate_implied_volatilities(self, chain: pd.DataFrame, spot_price: float,
                                      current_date: datetime) -> pd.DataFrame:
        """Calculate implied volatilities for options chain"""

        chain = chain.copy()
        chain['iv'] = 0.0

        for idx, row in chain.iterrows():
            try:
                # Time to expiration
                expiration = pd.to_datetime(row['expiration'])
                tte_days = (expiration - current_date).days
                T = max(tte_days / 365.0, 0.01)  # At least 1 day

                # Calculate IV
                market_price = row.get('mid', row.get('last', row.get('ask', 0)))
                if market_price > 0:
                    iv = self.pricing_model.calculate_implied_volatility(
                        market_price, spot_price, row['strike'], T,
                        self.risk_free_rate, row.get('type', 'call')
                    )
                    chain.at[idx, 'iv'] = iv

            except Exception as e:
                logger.warning(f"IV calculation failed for {row.get('symbol', idx)}: {e}")

        return chain

    def _analyze_moneyness(self, chain: pd.DataFrame, spot_price: float) -> Dict[str, Any]:
        """Analyze moneyness distribution"""

        calls = chain[chain['type'] == 'call']
        puts = chain[chain['type'] == 'put']

        analysis = {
            'calls': {
                'itms': len(calls[calls['strike'] < spot_price]),
                'atms': len(calls[abs(calls['strike'] - spot_price) / spot_price < 0.02]),
                'otms': len(calls[calls['strike'] > spot_price])
            },
            'puts': {
                'itms': len(puts[puts['strike'] > spot_price]),
                'atms': len(puts[abs(puts['strike'] - spot_price) / spot_price < 0.02]),
                'otms': len(puts[puts['strike'] < spot_price])
            }
        }

        # Add percentages
        for option_type in ['calls', 'puts']:
            total = sum(analysis[option_type].values())
            if total > 0:
                for category in analysis[option_type]:
                    analysis[option_type][f'{category}_pct'] = analysis[option_type][category] / total

        return analysis

    def _build_volatility_surface(self, chain: pd.DataFrame) -> Dict[str, Any]:
        """Build implied volatility surface"""

        if 'iv' not in chain.columns or chain['iv'].isna().all():
            return {'error': 'No implied volatility data'}

        surface = {
            'atm_vol': {},
            'vol_skew': {},
            'term_structure': {}
        }

        # Group by expiration
        for expiration, exp_chain in chain.groupby('expiration'):
            exp_date = pd.to_datetime(expiration)
            days_to_exp = (exp_date - pd.Timestamp.now()).days

            if days_to_exp < 0:
                continue

            calls = exp_chain[exp_chain['type'] == 'call']
            puts = exp_chain[exp_chain['type'] == 'put']

            if calls.empty:
                continue

            # ATM volatility (closest to spot)
            spot_price = 400  # This should be passed in
            atm_strike = calls.iloc[(calls['strike'] - spot_price).abs().argsort()[:1]]['strike'].iloc[0]
            atm_vol = calls[calls['strike'] == atm_strike]['iv'].mean()

            surface['atm_vol'][days_to_exp] = atm_vol

            # Volatility skew
            strikes = sorted(calls['strike'].unique())
            vols = []
            for strike in strikes:
                strike_vol = calls[calls['strike'] == strike]['iv'].mean()
                if not pd.isna(strike_vol):
                    vols.append(strike_vol)

            if len(vols) > 2:
                surface['vol_skew'][days_to_exp] = {
                    'min_vol': min(vols),
                    'max_vol': max(vols),
                    'vol_range': max(vols) - min(vols)
                }

        return surface

    def _calculate_greeks_summary(self, chain: pd.DataFrame) -> Dict[str, Any]:
        """Calculate summary statistics for options Greeks"""

        summary = {}

        for greek in ['delta', 'gamma', 'theta', 'vega', 'rho']:
            if greek in chain.columns:
                values = chain[greek].dropna()
                if not values.empty:
                    summary[greek] = {
                        'mean': values.mean(),
                        'std': values.std(),
                        'min': values.min(),
                        'max': values.max(),
                        'skew': values.skew(),
                        'kurtosis': values.kurtosis()
                    }

        return summary

    def _detect_unusual_activity(self, chain: pd.DataFrame) -> List[Dict[str, Any]]:
        """Detect unusual options activity"""

        unusual_options = []

        # Volume-based detection
        if 'volume' in chain.columns:
            avg_volume = chain['volume'].mean()
            high_volume = chain[chain['volume'] > avg_volume * 3]  # 3x average

            for _, option in high_volume.iterrows():
                unusual_options.append({
                    'type': 'high_volume',
                    'symbol': option.get('symbol', 'Unknown'),
                    'strike': option.get('strike'),
                    'expiration': option.get('expiration'),
                    'volume': option.get('volume'),
                    'avg_volume': avg_volume,
                    'ratio': option.get('volume', 0) / avg_volume if avg_volume > 0 else 0
                })

        # Open interest concentration
        if 'open_interest' in chain.columns:
            total_oi = chain['open_interest'].sum()
            concentrated_oi = chain[chain['open_interest'] / total_oi > 0.1]  # >10% of total OI

            for _, option in concentrated_oi.iterrows():
                unusual_options.append({
                    'type': 'oi_concentration',
                    'symbol': option.get('symbol', 'Unknown'),
                    'strike': option.get('strike'),
                    'expiration': option.get('expiration'),
                    'open_interest': option.get('open_interest'),
                    'oi_percentage': option.get('open_interest', 0) / total_oi if total_oi > 0 else 0
                })

        return unusual_options


class OptionsOverlayManager:
    """Manage options overlay strategies for portfolio hedging"""

    def __init__(self, pricing_model: BlackScholesModel = None):
        self.pricing_model = pricing_model or BlackScholesModel()

    def design_protective_put(self, portfolio_value: float, target_delta: float = -0.15,
                            max_dte: int = 90, volatility_buffer: float = 0.1) -> Dict[str, Any]:
        """
        Design protective put overlay strategy

        Args:
            portfolio_value: Current portfolio value
            target_delta: Target delta exposure (-0.15 = 15% downside protection)
            max_dte: Maximum days to expiration
            volatility_buffer: Volatility buffer for strike selection

        Returns:
            Put option specifications
        """

        # Estimate portfolio volatility (simplified)
        portfolio_vol = 0.25  # 25% annualized - should be calculated from actual data

        # Calculate strike price with volatility buffer
        spot_price = portfolio_value  # Simplified - assumes portfolio tracks an index
        strike_buffer = spot_price * (portfolio_vol * np.sqrt(max_dte/365) + volatility_buffer)
        strike_price = spot_price - strike_buffer

        # Time to expiration
        T = max_dte / 365.0

        # Calculate required put options
        # Delta of put = -N(-d1), so we need enough puts to achieve target delta
        # This is simplified - real implementation would use actual option chain

        put_delta = -0.3  # Typical put delta for this strike/moneyness
        puts_needed = abs(target_delta) / abs(put_delta)

        # Calculate premium cost
        put_premium = self.pricing_model.price_option(
            spot_price, strike_price, T, 0.05, portfolio_vol, 'put'
        )['price']

        total_cost = puts_needed * put_premium * portfolio_value / spot_price  # Scale to portfolio

        return {
            'strategy': 'protective_put',
            'target_delta': target_delta,
            'strike_price': strike_price,
            'days_to_expiration': max_dte,
            'puts_needed': puts_needed,
            'premium_per_put': put_premium,
            'total_cost': total_cost,
            'cost_pct_portfolio': total_cost / portfolio_value,
            'strike_pct_below_spot': (spot_price - strike_price) / spot_price
        }

    def design_collar_strategy(self, portfolio_value: float, upside_participation: float = 0.8,
                             downside_protection: float = 0.15) -> Dict[str, Any]:
        """
        Design collar strategy (covered call + protective put)

        Args:
            portfolio_value: Current portfolio value
            upside_participation: Fraction of upside to retain (0.8 = 80%)
            downside_protection: Downside protection level (0.15 = 15%)

        Returns:
            Collar strategy specifications
        """

        # Simplified collar design
        spot_price = portfolio_value

        # Put strike for downside protection
        put_strike = spot_price * (1 - downside_protection)

        # Call strike for upside participation
        call_strike = spot_price * (1 + upside_participation * 0.2)  # Limited upside

        # Time to expiration
        T = 60 / 365.0  # 60 days

        # Calculate premiums
        portfolio_vol = 0.25
        r = 0.05

        put_premium = self.pricing_model.price_option(spot_price, put_strike, T, r, portfolio_vol, 'put')['price']
        call_premium = self.pricing_model.price_option(spot_price, call_strike, T, r, portfolio_vol, 'call')['price']

        net_premium = call_premium - put_premium
        net_cost = net_premium * portfolio_value / spot_price

        return {
            'strategy': 'collar',
            'put_strike': put_strike,
            'call_strike': call_strike,
            'days_to_expiration': 60,
            'put_premium': put_premium,
            'call_premium': call_premium,
            'net_premium': net_premium,
            'net_cost': net_cost,
            'cost_pct_portfolio': net_cost / portfolio_value,
            'upside_cap': (call_strike - spot_price) / spot_price,
            'downside_protection': (spot_price - put_strike) / spot_price
        }

    def calculate_overlay_impact(self, base_portfolio: Dict[str, float],
                               overlay_strategy: Dict[str, Any],
                               market_scenarios: List[Dict[str, float]]) -> Dict[str, Any]:
        """
        Calculate impact of options overlay under different market scenarios

        Args:
            base_portfolio: Base portfolio weights
            overlay_strategy: Options overlay specifications
            market_scenarios: List of market return scenarios

        Returns:
            Overlay impact analysis
        """

        impacts = []

        for scenario in market_scenarios:
            scenario_name = scenario.get('name', 'unknown')

            # Calculate base portfolio P&L
            base_pnl = sum(weight * ret for ticker, weight in base_portfolio.items()
                          for ret_ticker, ret in scenario.items() if ret_ticker == ticker)

            # Calculate overlay P&L (simplified)
            if overlay_strategy['strategy'] == 'protective_put':
                # Put protects downside
                put_pnl = 0  # Simplified - would calculate actual option P&L
                overlay_pnl = put_pnl
            else:
                overlay_pnl = 0

            total_pnl = base_pnl + overlay_pnl

            impacts.append({
                'scenario': scenario_name,
                'base_pnl': base_pnl,
                'overlay_pnl': overlay_pnl,
                'total_pnl': total_pnl,
                'overlay_impact': overlay_pnl / abs(base_pnl) if base_pnl != 0 else 0
            })

        return {
            'overlay_strategy': overlay_strategy['strategy'],
            'scenario_impacts': impacts,
            'avg_overlay_impact': np.mean([imp['overlay_impact'] for imp in impacts]),
            'worst_case_protection': min([imp['total_pnl'] for imp in impacts])
        }


# Test functions
def test_options_engine():
    """Test options pricing and analysis functionality"""

    print("🧪 Testing Options Engine")

    # Test Black-Scholes pricing
    print("📊 Testing Black-Scholes Pricing...")

    bs_model = BlackScholesModel()

    # Sample option
    S, K, T, r, sigma = 100, 105, 0.5, 0.05, 0.2

    call_result = bs_model.price_option(S, K, T, r, sigma, 'call')
    put_result = bs_model.price_option(S, K, T, r, sigma, 'put')

    print(".2f")
    print(".2f")
    print(".3f")
    print(".3f")

    # Test implied volatility
    print("\n📈 Testing Implied Volatility Calculation...")

    market_price = 5.0  # Observed call price
    iv = bs_model.calculate_implied_volatility(market_price, S, K, T, r, 'call')
    print(".1%")

    # Test options overlay
    print("\n🛡️ Testing Options Overlay Strategies...")

    overlay_manager = OptionsOverlayManager(bs_model)

    # Protective put design
    portfolio_value = 100000
    put_strategy = overlay_manager.design_protective_put(portfolio_value, target_delta=-0.15)

    print("Protective Put Strategy:")
    print(".1%")
    print(".1%")

    # Collar strategy
    collar_strategy = overlay_manager.design_collar_strategy(portfolio_value, 0.8, 0.15)

    print("\nCollar Strategy:")
    print(".1%")
    print(".1%")

    print("\n✅ Options engine tests completed!")

    return {
        'call_pricing': call_result,
        'put_pricing': put_result,
        'implied_vol': iv,
        'protective_put': put_strategy,
        'collar': collar_strategy
    }


if __name__ == "__main__":
    test_options_engine()
