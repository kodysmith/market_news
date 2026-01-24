"""
Intrinsic Value Calculator Module
Implements 6 standard valuation methods to compute intrinsic value of stocks
"""

import math
import logging
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, List, Tuple
import yfinance as yf
import numpy as np

logger = logging.getLogger(__name__)


class IntrinsicValueCalculator:
    """
    Comprehensive intrinsic value calculator using 6 valuation methods:
    1. DCF (Discounted Cash Flow)
    2. Graham Number
    3. Dividend Discount Model (DDM)
    4. Earnings Power Value (EPV)
    5. Peer Comparable Analysis
    6. Net Asset Value (NAV)
    """
    
    # Default assumptions
    DEFAULT_DISCOUNT_RATE = 0.10  # 10% WACC
    DEFAULT_TERMINAL_GROWTH = 0.03  # 3% perpetual growth
    DEFAULT_RISK_FREE_RATE = 0.045  # 4.5% risk-free rate
    
    def __init__(self, ticker: str):
        self.ticker = ticker.upper()
        self.stock = None
        self.info = {}
        self.financials = None
        self.balance_sheet = None
        self.cashflow = None
        self.history = None
        self.current_price = 0.0
        self._loaded = False
        
    def _load_data(self) -> bool:
        """Load all necessary financial data from yfinance"""
        if self._loaded:
            return True
            
        try:
            logger.info(f"Loading data for {self.ticker}...")
            self.stock = yf.Ticker(self.ticker)
            self.info = self.stock.info or {}
            
            # Get current price
            self.current_price = self.info.get('currentPrice') or self.info.get('regularMarketPrice', 0)
            if not self.current_price:
                hist = self.stock.history(period='1d')
                if not hist.empty:
                    self.current_price = float(hist['Close'].iloc[-1])
            
            if self.current_price <= 0:
                logger.error(f"Could not get current price for {self.ticker}")
                return False
            
            # Load financial statements
            try:
                self.financials = self.stock.financials
                self.balance_sheet = self.stock.balance_sheet
                self.cashflow = self.stock.cashflow
            except Exception as e:
                logger.warning(f"Could not load financials: {e}")
            
            # Load price history for peer comparisons
            self.history = self.stock.history(period='1y')
            
            self._loaded = True
            logger.info(f"Successfully loaded data for {self.ticker}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load data for {self.ticker}: {e}")
            return False
    
    def calculate_all(self) -> Dict[str, Any]:
        """
        Calculate intrinsic value using all 6 methods and return composite result.
        
        Returns:
            Dict with all valuations, composite value, and divergence percentage
        """
        if not self._load_data():
            return {
                'ticker': self.ticker,
                'error': 'Failed to load financial data',
                'current_price': 0
            }
        
        # Calculate each method
        valuations = {
            'dcf': self.dcf_valuation(),
            'graham': self.graham_number(),
            'ddm': self.dividend_discount_model(),
            'epv': self.earnings_power_value(),
            'peer_comps': self.peer_comparable_analysis(),
            'nav': self.net_asset_value()
        }
        
        # Calculate composite (weighted average of applicable methods)
        composite = self._calculate_composite(valuations)
        
        return {
            'ticker': self.ticker,
            'company_name': self.info.get('longName', self.ticker),
            'sector': self.info.get('sector', 'Unknown'),
            'industry': self.info.get('industry', 'Unknown'),
            'current_price': self.current_price,
            'valuations': valuations,
            'composite': composite,
            'timestamp': datetime.now().isoformat()
        }
    
    def _calculate_composite(self, valuations: Dict[str, Dict]) -> Dict[str, Any]:
        """Calculate weighted average of applicable valuation methods"""
        applicable_values = []
        total_weight = 0.0
        weighted_sum = 0.0
        
        for method, result in valuations.items():
            if result.get('applicable', False) and result.get('value', 0) > 0:
                value = result['value']
                confidence = result.get('confidence', 0.5)
                applicable_values.append({
                    'method': method,
                    'value': value,
                    'confidence': confidence
                })
                weighted_sum += value * confidence
                total_weight += confidence
        
        if total_weight == 0:
            return {
                'value': None,
                'method_count': 0,
                'divergence_pct': None,
                'verdict': 'Insufficient Data'
            }
        
        composite_value = weighted_sum / total_weight
        divergence_pct = ((self.current_price - composite_value) / composite_value) * 100
        
        # Determine verdict
        if divergence_pct < -30:
            verdict = 'Significantly Undervalued'
        elif divergence_pct < -20:
            verdict = 'Undervalued'
        elif divergence_pct < -10:
            verdict = 'Slightly Undervalued'
        elif divergence_pct <= 10:
            verdict = 'Fair Value'
        elif divergence_pct <= 20:
            verdict = 'Slightly Overvalued'
        elif divergence_pct <= 30:
            verdict = 'Overvalued'
        else:
            verdict = 'Significantly Overvalued'
        
        return {
            'value': round(composite_value, 2),
            'method_count': len(applicable_values),
            'methods_used': [v['method'] for v in applicable_values],
            'divergence_pct': round(divergence_pct, 2),
            'verdict': verdict
        }
    
    # =========================================================================
    # Method 1: Discounted Cash Flow (DCF)
    # =========================================================================
    def dcf_valuation(self) -> Dict[str, Any]:
        """
        DCF Valuation: Present value of projected free cash flows + terminal value
        Best for: Stable, cash-generating companies
        """
        try:
            # Get required data
            fcf = self.info.get('freeCashflow', 0)
            shares_outstanding = self.info.get('sharesOutstanding', 0)
            total_debt = self.info.get('totalDebt', 0)
            revenue_growth = self.info.get('revenueGrowth', 0) or 0
            
            if fcf <= 0 or shares_outstanding <= 0:
                return {
                    'value': None,
                    'confidence': 0,
                    'applicable': False,
                    'reason': 'Negative or zero free cash flow'
                }
            
            # Conservative growth assumptions
            growth_rate_1_5 = min(max(revenue_growth, 0.02), 0.15)  # Cap at 15%
            growth_rate_6_10 = min(growth_rate_1_5 * 0.5, 0.08)  # Slower growth later
            terminal_growth = self.DEFAULT_TERMINAL_GROWTH
            discount_rate = self.DEFAULT_DISCOUNT_RATE
            
            # Project 10 years of cash flows
            projected_fcf = []
            current_fcf = fcf
            for year in range(1, 11):
                growth = growth_rate_1_5 if year <= 5 else growth_rate_6_10
                current_fcf = current_fcf * (1 + growth)
                projected_fcf.append(current_fcf)
            
            # Calculate present value of cash flows
            pv_cash_flows = sum(
                cf / ((1 + discount_rate) ** (i + 1))
                for i, cf in enumerate(projected_fcf)
            )
            
            # Calculate terminal value (Gordon Growth Model)
            terminal_fcf = projected_fcf[-1] * (1 + terminal_growth)
            terminal_value = terminal_fcf / (discount_rate - terminal_growth)
            pv_terminal = terminal_value / ((1 + discount_rate) ** 10)
            
            # Calculate intrinsic value per share
            enterprise_value = pv_cash_flows + pv_terminal
            equity_value = max(enterprise_value - total_debt, 0)
            intrinsic_value = equity_value / shares_outstanding
            
            # Confidence based on data quality and FCF consistency
            confidence = 0.85 if fcf > 0 and revenue_growth > 0 else 0.60
            
            return {
                'value': round(intrinsic_value, 2),
                'confidence': confidence,
                'applicable': True,
                'details': {
                    'free_cash_flow': fcf,
                    'growth_rate_1_5': round(growth_rate_1_5 * 100, 1),
                    'growth_rate_6_10': round(growth_rate_6_10 * 100, 1),
                    'discount_rate': round(discount_rate * 100, 1),
                    'terminal_growth': round(terminal_growth * 100, 1),
                    'enterprise_value': round(enterprise_value, 0),
                    'equity_value': round(equity_value, 0)
                }
            }
            
        except Exception as e:
            logger.error(f"DCF calculation failed: {e}")
            return {
                'value': None,
                'confidence': 0,
                'applicable': False,
                'reason': str(e)
            }
    
    # =========================================================================
    # Method 2: Graham Number
    # =========================================================================
    def graham_number(self) -> Dict[str, Any]:
        """
        Graham Number: sqrt(22.5 × EPS × Book Value per Share)
        Best for: Defensive value stocks
        Formula derived from P/E < 15 and P/B < 1.5 (15 × 1.5 = 22.5)
        """
        try:
            eps = self.info.get('trailingEps', 0)
            book_value_per_share = self.info.get('bookValue', 0)
            
            if eps <= 0 or book_value_per_share <= 0:
                return {
                    'value': None,
                    'confidence': 0,
                    'applicable': False,
                    'reason': 'Negative EPS or book value'
                }
            
            # Graham Number formula
            graham_value = math.sqrt(22.5 * eps * book_value_per_share)
            
            # Confidence based on earnings stability
            pe_ratio = self.info.get('trailingPE', 0)
            confidence = 0.75
            if pe_ratio and 5 < pe_ratio < 20:
                confidence = 0.85  # Higher confidence for reasonable P/E
            elif pe_ratio and pe_ratio > 30:
                confidence = 0.50  # Lower confidence for high P/E growth stocks
            
            return {
                'value': round(graham_value, 2),
                'confidence': confidence,
                'applicable': True,
                'details': {
                    'eps': round(eps, 2),
                    'book_value_per_share': round(book_value_per_share, 2),
                    'pe_ratio': round(pe_ratio, 2) if pe_ratio else None,
                    'pb_ratio': round(self.current_price / book_value_per_share, 2) if book_value_per_share else None
                }
            }
            
        except Exception as e:
            logger.error(f"Graham Number calculation failed: {e}")
            return {
                'value': None,
                'confidence': 0,
                'applicable': False,
                'reason': str(e)
            }
    
    # =========================================================================
    # Method 3: Dividend Discount Model (DDM) - Gordon Growth Model
    # =========================================================================
    def dividend_discount_model(self) -> Dict[str, Any]:
        """
        DDM (Gordon Growth Model): Intrinsic Value = D1 / (r - g)
        Where D1 = next year's dividend, r = required return, g = dividend growth rate
        Best for: Stable dividend-paying companies
        """
        try:
            dividend_rate = self.info.get('dividendRate', 0)
            dividend_yield = self.info.get('dividendYield', 0)
            five_year_avg_yield = self.info.get('fiveYearAvgDividendYield', 0)
            
            if dividend_rate <= 0:
                return {
                    'value': None,
                    'confidence': 0,
                    'applicable': False,
                    'reason': 'No dividend payments'
                }
            
            # Estimate dividend growth rate
            # Use payout ratio and earnings growth to estimate sustainable dividend growth
            payout_ratio = self.info.get('payoutRatio', 0.5)
            earnings_growth = self.info.get('earningsGrowth', 0) or 0
            
            # Conservative dividend growth estimate
            if five_year_avg_yield and dividend_yield:
                # If current yield is lower than avg, dividends have been growing
                implied_growth = max(0, (five_year_avg_yield - dividend_yield) / 5 / 100)
            else:
                implied_growth = 0
            
            dividend_growth = min(
                max(earnings_growth * (1 - payout_ratio), implied_growth, 0.02),
                0.08  # Cap at 8%
            )
            
            # Required return (cost of equity)
            beta = self.info.get('beta', 1.0) or 1.0
            market_risk_premium = 0.055  # 5.5%
            required_return = self.DEFAULT_RISK_FREE_RATE + (beta * market_risk_premium)
            
            # Ensure required return > dividend growth (model constraint)
            if required_return <= dividend_growth:
                dividend_growth = required_return * 0.6  # Adjust growth down
            
            # Next year's dividend
            d1 = dividend_rate * (1 + dividend_growth)
            
            # Gordon Growth Model
            intrinsic_value = d1 / (required_return - dividend_growth)
            
            # Confidence based on dividend history
            confidence = 0.70
            if five_year_avg_yield and five_year_avg_yield > 0:
                confidence = 0.80  # Higher if consistent dividend history
            if payout_ratio and 0.3 <= payout_ratio <= 0.7:
                confidence += 0.05  # Sustainable payout ratio
            
            return {
                'value': round(intrinsic_value, 2),
                'confidence': min(confidence, 0.90),
                'applicable': True,
                'details': {
                    'current_dividend': round(dividend_rate, 2),
                    'dividend_yield_pct': round((dividend_yield or 0) * 100, 2),
                    'dividend_growth_pct': round(dividend_growth * 100, 2),
                    'required_return_pct': round(required_return * 100, 2),
                    'payout_ratio': round((payout_ratio or 0) * 100, 1)
                }
            }
            
        except Exception as e:
            logger.error(f"DDM calculation failed: {e}")
            return {
                'value': None,
                'confidence': 0,
                'applicable': False,
                'reason': str(e)
            }
    
    # =========================================================================
    # Method 4: Earnings Power Value (EPV)
    # =========================================================================
    def earnings_power_value(self) -> Dict[str, Any]:
        """
        EPV: Normalized Earnings / Cost of Capital
        Assumes zero growth - values only current earning power
        Best for: Mature companies with stable earnings
        """
        try:
            # Get earnings data
            net_income = self.info.get('netIncomeToCommon', 0)
            shares_outstanding = self.info.get('sharesOutstanding', 0)
            total_debt = self.info.get('totalDebt', 0)
            
            if net_income <= 0 or shares_outstanding <= 0:
                return {
                    'value': None,
                    'confidence': 0,
                    'applicable': False,
                    'reason': 'Negative or zero net income'
                }
            
            # Normalize earnings (average of recent years if available)
            # Use operating income for more stable base
            operating_income = self.info.get('operatingIncome', net_income)
            
            # Apply corporate tax rate to get after-tax operating income
            tax_rate = 0.21  # US corporate tax rate
            normalized_earnings = operating_income * (1 - tax_rate)
            
            # Cost of capital (WACC simplified)
            beta = self.info.get('beta', 1.0) or 1.0
            market_cap = self.info.get('marketCap', 0)
            
            # Cost of equity
            cost_of_equity = self.DEFAULT_RISK_FREE_RATE + (beta * 0.055)
            
            # Cost of debt (approximate)
            interest_expense = self.info.get('interestExpense', 0)
            cost_of_debt = (abs(interest_expense) / total_debt * (1 - tax_rate)) if total_debt > 0 else 0.05
            
            # WACC
            if market_cap and total_debt:
                total_capital = market_cap + total_debt
                wacc = (market_cap / total_capital * cost_of_equity) + \
                       (total_debt / total_capital * cost_of_debt)
            else:
                wacc = cost_of_equity
            
            wacc = max(wacc, 0.08)  # Floor at 8%
            
            # EPV calculation
            enterprise_epv = normalized_earnings / wacc
            equity_epv = enterprise_epv - total_debt
            epv_per_share = equity_epv / shares_outstanding
            
            # Confidence based on earnings stability
            profit_margin = self.info.get('profitMargins', 0) or 0
            confidence = 0.70
            if profit_margin > 0.10:
                confidence = 0.80
            if profit_margin > 0.20:
                confidence = 0.85
            
            return {
                'value': round(max(epv_per_share, 0), 2),
                'confidence': confidence,
                'applicable': True,
                'details': {
                    'normalized_earnings': round(normalized_earnings, 0),
                    'wacc_pct': round(wacc * 100, 2),
                    'enterprise_epv': round(enterprise_epv, 0),
                    'equity_epv': round(equity_epv, 0),
                    'profit_margin_pct': round(profit_margin * 100, 2)
                }
            }
            
        except Exception as e:
            logger.error(f"EPV calculation failed: {e}")
            return {
                'value': None,
                'confidence': 0,
                'applicable': False,
                'reason': str(e)
            }
    
    # =========================================================================
    # Method 5: Peer Comparable Analysis
    # =========================================================================
    def peer_comparable_analysis(self) -> Dict[str, Any]:
        """
        Peer Comps: Apply sector average multiples to company's metrics
        Uses P/E, EV/EBITDA, and P/S ratios
        Best for: Relative valuation within industry
        """
        try:
            # Get company metrics
            eps = self.info.get('trailingEps', 0)
            revenue = self.info.get('totalRevenue', 0)
            ebitda = self.info.get('ebitda', 0)
            shares_outstanding = self.info.get('sharesOutstanding', 0)
            total_debt = self.info.get('totalDebt', 0)
            cash = self.info.get('totalCash', 0)
            
            if shares_outstanding <= 0:
                return {
                    'value': None,
                    'confidence': 0,
                    'applicable': False,
                    'reason': 'Missing shares outstanding'
                }
            
            # Sector average multiples (typical values by sector)
            sector = self.info.get('sector', 'Unknown')
            sector_multiples = self._get_sector_multiples(sector)
            
            valuations = []
            
            # P/E based valuation
            if eps > 0:
                pe_value = eps * sector_multiples['pe']
                valuations.append(('PE', pe_value, 0.40))
            
            # EV/EBITDA based valuation
            if ebitda > 0:
                enterprise_value = ebitda * sector_multiples['ev_ebitda']
                equity_value = enterprise_value - total_debt + cash
                ev_ebitda_value = equity_value / shares_outstanding
                if ev_ebitda_value > 0:
                    valuations.append(('EV/EBITDA', ev_ebitda_value, 0.35))
            
            # P/S based valuation
            if revenue > 0:
                revenue_per_share = revenue / shares_outstanding
                ps_value = revenue_per_share * sector_multiples['ps']
                valuations.append(('PS', ps_value, 0.25))
            
            if not valuations:
                return {
                    'value': None,
                    'confidence': 0,
                    'applicable': False,
                    'reason': 'Insufficient data for peer comparison'
                }
            
            # Weighted average of valuations
            total_weight = sum(v[2] for v in valuations)
            weighted_value = sum(v[1] * v[2] for v in valuations) / total_weight
            
            # Confidence based on how many metrics we used
            base_confidence = 0.60 + (len(valuations) * 0.10)
            
            return {
                'value': round(weighted_value, 2),
                'confidence': min(base_confidence, 0.85),
                'applicable': True,
                'details': {
                    'sector': sector,
                    'sector_avg_pe': sector_multiples['pe'],
                    'sector_avg_ev_ebitda': sector_multiples['ev_ebitda'],
                    'sector_avg_ps': sector_multiples['ps'],
                    'methods_used': [v[0] for v in valuations],
                    'individual_values': {v[0]: round(v[1], 2) for v in valuations}
                }
            }
            
        except Exception as e:
            logger.error(f"Peer comparable analysis failed: {e}")
            return {
                'value': None,
                'confidence': 0,
                'applicable': False,
                'reason': str(e)
            }
    
    def _get_sector_multiples(self, sector: str) -> Dict[str, float]:
        """Get typical valuation multiples by sector"""
        # Industry average multiples (approximate)
        sector_data = {
            'Technology': {'pe': 25, 'ev_ebitda': 15, 'ps': 5},
            'Healthcare': {'pe': 20, 'ev_ebitda': 12, 'ps': 3},
            'Financial Services': {'pe': 12, 'ev_ebitda': 8, 'ps': 2},
            'Consumer Cyclical': {'pe': 18, 'ev_ebitda': 10, 'ps': 1.5},
            'Consumer Defensive': {'pe': 20, 'ev_ebitda': 12, 'ps': 1.2},
            'Industrials': {'pe': 18, 'ev_ebitda': 10, 'ps': 1.5},
            'Energy': {'pe': 12, 'ev_ebitda': 6, 'ps': 1},
            'Utilities': {'pe': 16, 'ev_ebitda': 10, 'ps': 2},
            'Real Estate': {'pe': 35, 'ev_ebitda': 18, 'ps': 6},
            'Basic Materials': {'pe': 14, 'ev_ebitda': 8, 'ps': 1.2},
            'Communication Services': {'pe': 18, 'ev_ebitda': 10, 'ps': 2.5},
        }
        
        return sector_data.get(sector, {'pe': 18, 'ev_ebitda': 10, 'ps': 2})
    
    # =========================================================================
    # Method 6: Net Asset Value (NAV)
    # =========================================================================
    def net_asset_value(self) -> Dict[str, Any]:
        """
        NAV: (Total Assets - Total Liabilities) / Shares Outstanding
        Also known as Book Value
        Best for: Asset-heavy companies, banks, REITs
        """
        try:
            total_assets = self.info.get('totalAssets', 0)
            total_liabilities = self.info.get('totalLiab', 0)
            shares_outstanding = self.info.get('sharesOutstanding', 0)
            book_value = self.info.get('bookValue', 0)
            
            # Prefer direct book value if available
            if book_value and book_value > 0:
                nav_per_share = book_value
            elif total_assets > 0 and shares_outstanding > 0:
                net_assets = total_assets - total_liabilities
                nav_per_share = net_assets / shares_outstanding
            else:
                return {
                    'value': None,
                    'confidence': 0,
                    'applicable': False,
                    'reason': 'Missing balance sheet data'
                }
            
            if nav_per_share <= 0:
                return {
                    'value': None,
                    'confidence': 0,
                    'applicable': False,
                    'reason': 'Negative book value'
                }
            
            # Confidence depends on sector (NAV is more relevant for some)
            sector = self.info.get('sector', 'Unknown')
            confidence = 0.50  # Base confidence
            
            # Higher confidence for asset-heavy sectors
            if sector in ['Financial Services', 'Real Estate', 'Utilities', 'Basic Materials']:
                confidence = 0.70
            elif sector in ['Technology', 'Healthcare']:
                confidence = 0.40  # Less relevant for intangible-heavy companies
            
            # Calculate tangible book value
            goodwill = self.info.get('goodwill', 0) or 0
            intangibles = self.info.get('intangibleAssets', 0) or 0
            tangible_book = nav_per_share - (goodwill + intangibles) / shares_outstanding if shares_outstanding else nav_per_share
            
            return {
                'value': round(nav_per_share, 2),
                'confidence': confidence,
                'applicable': True,
                'details': {
                    'total_assets': round(total_assets, 0) if total_assets else None,
                    'total_liabilities': round(total_liabilities, 0) if total_liabilities else None,
                    'book_value_per_share': round(nav_per_share, 2),
                    'tangible_book_value': round(tangible_book, 2) if tangible_book > 0 else None,
                    'price_to_book': round(self.current_price / nav_per_share, 2) if nav_per_share > 0 else None
                }
            }
            
        except Exception as e:
            logger.error(f"NAV calculation failed: {e}")
            return {
                'value': None,
                'confidence': 0,
                'applicable': False,
                'reason': str(e)
            }


def calculate_intrinsic_value(ticker: str) -> Dict[str, Any]:
    """
    Convenience function to calculate intrinsic value for a ticker.
    
    Args:
        ticker: Stock ticker symbol (e.g., 'AAPL')
        
    Returns:
        Dict with all valuations, composite value, and divergence percentage
    """
    calculator = IntrinsicValueCalculator(ticker)
    return calculator.calculate_all()


def batch_calculate(tickers: List[str]) -> List[Dict[str, Any]]:
    """
    Calculate intrinsic values for multiple tickers.
    
    Args:
        tickers: List of stock ticker symbols
        
    Returns:
        List of valuation results
    """
    results = []
    for ticker in tickers:
        try:
            result = calculate_intrinsic_value(ticker)
            results.append(result)
        except Exception as e:
            results.append({
                'ticker': ticker,
                'error': str(e)
            })
    return results
