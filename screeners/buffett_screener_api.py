#!/usr/bin/env python3
"""
Buffett Screener API - Real-time company valuation analysis
Integrates with existing MarketNews architecture
"""

import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from decimal import Decimal
import json
import logging
from typing import List, Dict, Any, Optional
import time
import requests
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class BuffettScreener:
    """Warren Buffett-style company screener with real data"""
    
    def __init__(self):
        self.last_scan_time = None
        self.cached_results = None
        self.cache_duration = 3600  # 1 hour cache
        
        # API keys
        self.alpha_vantage_key = os.getenv('ALPHAVANTAGE_API_KEY')
        self.fmp_key = os.getenv('FMP_API_KEY')
        
        # Rate limiting
        self.last_alpha_vantage_call = 0
        self.last_fmp_call = 0
        self.min_call_interval = 1.0  # 1 second between calls
        
    def get_sp500_symbols(self) -> List[str]:
        """Get S&P 500 symbols for screening"""
        # In production, you'd fetch this from a reliable source
        # For now, using a curated list of major companies
        return [
            'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'META', 'NVDA', 'BRK-B',
            'JNJ', 'V', 'PG', 'UNH', 'HD', 'MA', 'DIS', 'PYPL', 'ADBE', 'NFLX',
            'CRM', 'INTC', 'CMCSA', 'PFE', 'T', 'ABT', 'PEP', 'KO', 'WMT', 'MRK',
            'BAC', 'XOM', 'JPM', 'CVX', 'LLY', 'AVGO', 'ACN', 'COST', 'DHR', 'VZ',
            'NKE', 'MCD', 'SBUX', 'LMT', 'RTX', 'BA', 'CAT', 'GE', 'MMM', 'HON',
            'IBM', 'CSCO', 'ORCL', 'QCOM', 'TXN', 'AMAT', 'AMD', 'INTC', 'MU',
            'BKNG', 'CHTR', 'CME', 'COF', 'EL', 'FIS', 'FISV', 'GPN', 'ICE',
            'ISRG', 'LRCX', 'MCO', 'MDT', 'NOC', 'NOW', 'PLD', 'REGN', 'SYK',
            'TMO', 'VRTX', 'WBA', 'ZTS', 'AON', 'APD', 'BDX', 'BIIB', 'BSX',
            'CERN', 'CL', 'CTAS', 'CTSH', 'D', 'EA', 'EW', 'FANG', 'FDX',
            'GILD', 'GM', 'HCA', 'HUM', 'IDXX', 'ILMN', 'ITW', 'JCI', 'KMB',
            'LHX', 'LIN', 'LMT', 'LOW', 'MCHP', 'MDLZ', 'MRNA', 'NEE', 'NSC',
            'NTRS', 'NUE', 'OXY', 'PAYX', 'PCAR', 'PGR', 'PNC', 'PPG', 'PRU',
            'PSA', 'QCOM', 'RMD', 'ROK', 'ROP', 'ROST', 'SRE', 'STZ', 'SWK',
            'TEL', 'TGT', 'TJX', 'TROW', 'TRV', 'TXN', 'USB', 'VRSK', 'VRTX',
            'WEC', 'WY', 'XEL', 'ZBRA', 'ZBH'
        ]
    
    def _rate_limit(self, api_type: str = 'alpha_vantage'):
        """Implement rate limiting for API calls"""
        current_time = time.time()
        
        if api_type == 'alpha_vantage':
            time_since_last = current_time - self.last_alpha_vantage_call
            if time_since_last < self.min_call_interval:
                sleep_time = self.min_call_interval - time_since_last
                time.sleep(sleep_time)
            self.last_alpha_vantage_call = time.time()
        elif api_type == 'fmp':
            time_since_last = current_time - self.last_fmp_call
            if time_since_last < self.min_call_interval:
                sleep_time = self.min_call_interval - time_since_last
                time.sleep(sleep_time)
            self.last_fmp_call = time.time()
    
    def get_alpha_vantage_data(self, symbol: str) -> Optional[Dict[str, Any]]:
        """Get company data from Alpha Vantage"""
        if not self.alpha_vantage_key:
            return None
        
        try:
            self._rate_limit('alpha_vantage')
            
            # Get company overview
            url = "https://www.alphavantage.co/query"
            params = {
                'function': 'OVERVIEW',
                'symbol': symbol,
                'apikey': self.alpha_vantage_key
            }
            
            response = requests.get(url, params=params, timeout=10)
            if response.status_code == 200:
                data = response.json()
                
                # Check for API limit message
                if 'Note' in data or 'Information' in data:
                    logger.warning(f"Alpha Vantage API limit reached for {symbol}")
                    return None
                
                return data
            else:
                logger.warning(f"Alpha Vantage API error for {symbol}: {response.status_code}")
                return None
                
        except Exception as e:
            logger.error(f"Error fetching Alpha Vantage data for {symbol}: {e}")
            return None
    
    def get_fmp_data(self, symbol: str) -> Optional[Dict[str, Any]]:
        """Get company data from Financial Modeling Prep"""
        if not self.fmp_key:
            return None
        
        try:
            self._rate_limit('fmp')
            
            # Get company profile
            url = f"https://financialmodelingprep.com/api/v3/profile/{symbol}"
            params = {'apikey': self.fmp_key}
            
            response = requests.get(url, params=params, timeout=10)
            if response.status_code == 200:
                data = response.json()
                if data and len(data) > 0:
                    return data[0]  # Return first result
                else:
                    return None
            else:
                logger.warning(f"FMP API error for {symbol}: {response.status_code}")
                return None
                
        except Exception as e:
            logger.error(f"Error fetching FMP data for {symbol}: {e}")
            return None
    
    def fetch_company_data(self, symbol: str) -> Optional[Dict[str, Any]]:
        """Fetch comprehensive company data from multiple sources"""
        try:
            # Get data from multiple sources
            ticker = yf.Ticker(symbol)
            yf_info = ticker.info
            alpha_vantage_data = self.get_alpha_vantage_data(symbol)
            fmp_data = self.get_fmp_data(symbol)
            
            # Use yfinance as primary source, supplement with others
            info = yf_info
            
            if not info or 'marketCap' not in info:
                logger.warning(f"No data available for {symbol}")
                return None
            
            # Supplement with Alpha Vantage data if available
            if alpha_vantage_data:
                # Alpha Vantage provides some additional metrics
                info.update({
                    'sector': alpha_vantage_data.get('Sector', info.get('sector', 'Unknown')),
                    'industry': alpha_vantage_data.get('Industry', info.get('industry', 'Unknown')),
                    'description': alpha_vantage_data.get('Description', info.get('longBusinessSummary', '')),
                    'dividend_yield': float(alpha_vantage_data.get('DividendYield', 0)) / 100 if alpha_vantage_data.get('DividendYield') else info.get('dividendYield', 0),
                    'pe_ratio': float(alpha_vantage_data.get('PERatio', 0)) if alpha_vantage_data.get('PERatio') else info.get('trailingPE', 0),
                    'pb_ratio': float(alpha_vantage_data.get('PriceToBookRatio', 0)) if alpha_vantage_data.get('PriceToBookRatio') else info.get('priceToBook', 0),
                    'ps_ratio': float(alpha_vantage_data.get('PriceToSalesRatioTTM', 0)) if alpha_vantage_data.get('PriceToSalesRatioTTM') else info.get('priceToSalesTrailing12Months', 0),
                    'ev_ebitda': float(alpha_vantage_data.get('EVToEBITDA', 0)) if alpha_vantage_data.get('EVToEBITDA') else info.get('enterpriseToEbitda', 0),
                    'roe': float(alpha_vantage_data.get('ReturnOnEquityTTM', 0)) / 100 if alpha_vantage_data.get('ReturnOnEquityTTM') else 0,
                    'roa': float(alpha_vantage_data.get('ReturnOnAssetsTTM', 0)) / 100 if alpha_vantage_data.get('ReturnOnAssetsTTM') else 0,
                    'debt_to_equity': float(alpha_vantage_data.get('DebtToEquity', 0)) if alpha_vantage_data.get('DebtToEquity') else 0,
                    'current_ratio': float(alpha_vantage_data.get('CurrentRatio', 0)) if alpha_vantage_data.get('CurrentRatio') else 0,
                    'interest_coverage': float(alpha_vantage_data.get('InterestCoverage', 0)) if alpha_vantage_data.get('InterestCoverage') else 999.0,
                    'revenue_growth_5y': float(alpha_vantage_data.get('RevenueGrowth5Y', 0)) / 100 if alpha_vantage_data.get('RevenueGrowth5Y') else 0,
                    'earnings_growth_5y': float(alpha_vantage_data.get('EarningsGrowth5Y', 0)) / 100 if alpha_vantage_data.get('EarningsGrowth5Y') else 0,
                })
            
            # Supplement with FMP data if available
            if fmp_data:
                info.update({
                    'beta': fmp_data.get('beta', info.get('beta', 1.0)),
                    'volatility': fmp_data.get('volatility', 0),
                    'last_div': fmp_data.get('lastDiv', 0),
                    'range': fmp_data.get('range', ''),
                    'changes': fmp_data.get('changes', 0),
                    'company_name': fmp_data.get('companyName', info.get('longName', symbol)),
                    'exchange': fmp_data.get('exchange', info.get('exchange', 'NASDAQ')),
                    'website': fmp_data.get('website', info.get('website', '')),
                    'ceo': fmp_data.get('ceo', ''),
                    'sector': fmp_data.get('sector', info.get('sector', 'Unknown')),
                    'industry': fmp_data.get('industry', info.get('industry', 'Unknown')),
                    'description': fmp_data.get('description', info.get('longBusinessSummary', '')),
                })
            
            # Get financial statements
            financials = ticker.financials
            balance_sheet = ticker.balance_sheet
            cash_flow = ticker.cashflow
            
            # Calculate key metrics
            market_cap = info.get('marketCap', 0)
            enterprise_value = info.get('enterpriseValue', 0)
            total_revenue = info.get('totalRevenue', 0)
            net_income = info.get('netIncomeToCommon', 0)
            total_debt = info.get('totalDebt', 0)
            total_equity = info.get('totalStockholderEquity', 0)
            current_assets = info.get('totalCurrentAssets', 0)
            current_liabilities = info.get('totalCurrentLiabilities', 0)
            operating_cash_flow = info.get('operatingCashflow', 0)
            capital_expenditures = info.get('capitalExpenditures', 0)
            shares_outstanding = info.get('sharesOutstanding', 0)
            
            # Try to get missing data from financial statements
            if not total_equity and not balance_sheet.empty:
                try:
                    # Look for total stockholder equity in balance sheet
                    for idx in balance_sheet.index:
                        if 'stockholder' in str(idx).lower() and 'equity' in str(idx).lower():
                            total_equity = float(balance_sheet.loc[idx].iloc[0])
                            break
                except:
                    pass
            
            if not operating_cash_flow and not cash_flow.empty:
                try:
                    # Look for operating cash flow
                    for idx in cash_flow.index:
                        if 'operating' in str(idx).lower() and 'cash' in str(idx).lower():
                            operating_cash_flow = float(cash_flow.loc[idx].iloc[0])
                            break
                except:
                    pass
            
            if not capital_expenditures and not cash_flow.empty:
                try:
                    # Look for capital expenditures
                    for idx in cash_flow.index:
                        if 'capital' in str(idx).lower() and 'expenditure' in str(idx).lower():
                            capital_expenditures = float(cash_flow.loc[idx].iloc[0])
                            break
                except:
                    pass
            
            # Calculate derived metrics
            free_cash_flow = operating_cash_flow - capital_expenditures if operating_cash_flow and capital_expenditures else 0
            
            # Growth rates (simplified - would need historical data for accuracy)
            revenue_growth_5y = self._calculate_growth_rate(financials, 'totalRevenue', 5)
            net_income_growth_5y = self._calculate_growth_rate(financials, 'netIncome', 5)
            
            # Profitability ratios
            net_margin = (net_income / total_revenue) if total_revenue > 0 else 0
            roe = (net_income / total_equity) if total_equity > 0 else 0
            roa = (net_income / info.get('totalAssets', 1)) if info.get('totalAssets', 0) > 0 else 0
            roic = self._calculate_roic(net_income, total_debt, total_equity)
            
            # Debt ratios
            debt_to_equity = (total_debt / total_equity) if total_equity > 0 else 0
            current_ratio = (current_assets / current_liabilities) if current_liabilities > 0 else 0
            interest_coverage = self._calculate_interest_coverage(net_income, info.get('interestExpense', 0))
            
            # Cash flow ratios
            fcf_margin = (free_cash_flow / total_revenue) if total_revenue > 0 else 0
            
            # Valuation ratios
            pe_ratio = info.get('trailingPE', 0)
            pb_ratio = info.get('priceToBook', 0)
            ps_ratio = info.get('priceToSalesTrailing12Months', 0)
            ev_ebitda = info.get('enterpriseToEbitda', 0)
            price_to_fcf = (market_cap / free_cash_flow) if free_cash_flow > 0 else 0
            
            # Dividend info
            dividend_yield = info.get('dividendYield', 0) or 0
            dividend_payout_ratio = info.get('payoutRatio', 0) or 0
            
            # Moat analysis (simplified)
            moat_score = self._calculate_moat_score(info, net_margin, roe, roic)
            
            # Management quality (simplified)
            management_score = self._calculate_management_score(info, roe, roic, debt_to_equity)
            
            return {
                'symbol': symbol,
                'name': info.get('longName', symbol),
                'sector': info.get('sector', 'Unknown'),
                'industry': info.get('industry', 'Unknown'),
                'market_cap': market_cap,
                'enterprise_value': enterprise_value,
                'total_revenue': total_revenue,
                'net_income': net_income,
                'total_debt': total_debt,
                'total_equity': total_equity,
                'shares_outstanding': shares_outstanding,
                'current_price': info.get('currentPrice', 0),
                
                # Growth
                'revenue_growth_5y': revenue_growth_5y,
                'net_income_growth_5y': net_income_growth_5y,
                
                # Profitability
                'net_margin': net_margin,
                'roe': roe,
                'roa': roa,
                'roic': roic,
                
                # Debt & Leverage
                'debt_to_equity': debt_to_equity,
                'current_ratio': current_ratio,
                'interest_coverage': interest_coverage,
                
                # Cash Flow
                'operating_cash_flow': operating_cash_flow,
                'free_cash_flow': free_cash_flow,
                'fcf_margin': fcf_margin,
                'capital_expenditures': capital_expenditures,
                
                # Valuation
                'pe_ratio': pe_ratio,
                'pb_ratio': pb_ratio,
                'ps_ratio': ps_ratio,
                'ev_ebitda': ev_ebitda,
                'price_to_fcf': price_to_fcf,
                
                # Dividends
                'dividend_yield': dividend_yield,
                'dividend_payout_ratio': dividend_payout_ratio,
                
                # Quality Scores
                'moat_score': moat_score,
                'management_score': management_score,
                
                # Timestamps
                'last_updated': datetime.now().isoformat(),
                'data_source': 'yfinance'
            }
            
        except Exception as e:
            logger.error(f"Error fetching data for {symbol}: {e}")
            return None
    
    def _calculate_growth_rate(self, financials: pd.DataFrame, metric: str, years: int) -> float:
        """Calculate growth rate over specified years"""
        try:
            if financials.empty or len(financials.columns) < years + 1:
                return 0.0
            
            # Find the metric row (simplified - would need better matching)
            metric_data = None
            for idx, row in financials.iterrows():
                if metric.lower() in str(idx).lower():
                    metric_data = row
                    break
            
            if metric_data is None or len(metric_data) < years + 1:
                return 0.0
            
            current_value = float(metric_data.iloc[0])
            past_value = float(metric_data.iloc[years])
            
            if past_value > 0:
                return (current_value - past_value) / past_value
            else:
                return 0.0
                
        except Exception as e:
            logger.error(f"Error calculating growth rate: {e}")
            return 0.0
    
    def _calculate_roic(self, net_income: float, total_debt: float, total_equity: float) -> float:
        """Calculate Return on Invested Capital"""
        try:
            invested_capital = total_debt + total_equity
            if invested_capital > 0:
                return net_income / invested_capital
            else:
                return 0.0
        except:
            return 0.0
    
    def _calculate_interest_coverage(self, net_income: float, interest_expense: float) -> float:
        """Calculate interest coverage ratio"""
        try:
            if interest_expense > 0:
                return (net_income + interest_expense) / interest_expense
            else:
                return 999.0  # No interest expense
        except:
            return 999.0
    
    def _calculate_moat_score(self, info: Dict, net_margin: float, roe: float, roic: float) -> int:
        """Calculate economic moat score (1-10)"""
        try:
            score = 5  # Base score
            
            # Brand strength based on margins
            if net_margin > 0.20:
                score += 2
            elif net_margin > 0.15:
                score += 1
            
            # Return metrics
            if roe > 0.20:
                score += 1
            if roic > 0.15:
                score += 1
            
            # Industry-specific adjustments
            industry = info.get('industry', '').lower()
            if 'software' in industry or 'technology' in industry:
                score += 1  # High switching costs
            elif 'pharmaceutical' in industry:
                score += 1  # Regulatory barriers
            
            return min(10, max(1, score))
        except:
            return 5
    
    def _calculate_management_score(self, info: Dict, roe: float, roic: float, debt_to_equity: float) -> int:
        """Calculate management quality score (1-10)"""
        try:
            score = 5  # Base score
            
            # Capital allocation
            if roe > 0.15:
                score += 1
            if roic > 0.12:
                score += 1
            
            # Debt management
            if debt_to_equity < 0.30:
                score += 1
            elif debt_to_equity > 0.70:
                score -= 1
            
            # Profitability consistency
            if info.get('trailingPE', 0) > 0 and info.get('trailingPE', 0) < 25:
                score += 1
            
            return min(10, max(1, score))
        except:
            return 5
    
    def apply_buffett_criteria(self, company_data: Dict[str, Any]) -> Dict[str, Any]:
        """Apply Warren Buffett screening criteria"""
        criteria = {
            'min_revenue_growth_5y': 0.03,  # 3% (more realistic)
            'min_net_margin': 0.08,  # 8% (more realistic)
            'min_roe': 0.12,  # 12% (more realistic)
            'min_roic': 0.10,  # 10% (more realistic)
            'max_debt_to_equity': 0.60,  # 60% (more realistic)
            'min_interest_coverage': 3.0,  # 3x (more realistic)
            'min_current_ratio': 1.2,  # 1.2x (more realistic)
            'min_fcf_margin': 0.05,  # 5% (more realistic)
            'max_pe_ratio': 30.0,  # 30x (more realistic)
            'max_pb_ratio': 4.0,  # 4x (more realistic)
            'max_ev_ebitda': 20.0,  # 20x (more realistic)
            'max_price_to_fcf': 25.0,  # 25x (more realistic)
            'min_moat_score': 5,  # 5/10 (more realistic)
            'min_management_score': 5,  # 5/10 (more realistic)
        }
        
        passes = True
        reasons = []
        score = 0
        max_score = 100
        
        # Revenue growth
        if company_data['revenue_growth_5y'] < criteria['min_revenue_growth_5y']:
            passes = False
            reasons.append(f"Revenue growth too low: {company_data['revenue_growth_5y']:.1%}")
        else:
            score += 10
        
        # Profitability
        if company_data['net_margin'] < criteria['min_net_margin']:
            passes = False
            reasons.append(f"Net margin too low: {company_data['net_margin']:.1%}")
        else:
            score += 10
        
        if company_data['roe'] < criteria['min_roe']:
            passes = False
            reasons.append(f"ROE too low: {company_data['roe']:.1%}")
        else:
            score += 10
        
        if company_data['roic'] < criteria['min_roic']:
            passes = False
            reasons.append(f"ROIC too low: {company_data['roic']:.1%}")
        else:
            score += 10
        
        # Debt
        if company_data['debt_to_equity'] > criteria['max_debt_to_equity']:
            passes = False
            reasons.append(f"Debt-to-equity too high: {company_data['debt_to_equity']:.2f}")
        else:
            score += 10
        
        if company_data['interest_coverage'] < criteria['min_interest_coverage']:
            passes = False
            reasons.append(f"Interest coverage too low: {company_data['interest_coverage']:.1f}x")
        else:
            score += 5
        
        if company_data['current_ratio'] < criteria['min_current_ratio']:
            passes = False
            reasons.append(f"Current ratio too low: {company_data['current_ratio']:.2f}")
        else:
            score += 5
        
        # Cash flow
        if company_data['fcf_margin'] < criteria['min_fcf_margin']:
            passes = False
            reasons.append(f"FCF margin too low: {company_data['fcf_margin']:.1%}")
        else:
            score += 10
        
        # Valuation
        if company_data['pe_ratio'] > criteria['max_pe_ratio']:
            passes = False
            reasons.append(f"P/E ratio too high: {company_data['pe_ratio']:.1f}")
        else:
            score += 5
        
        if company_data['pb_ratio'] > criteria['max_pb_ratio']:
            passes = False
            reasons.append(f"P/B ratio too high: {company_data['pb_ratio']:.1f}")
        else:
            score += 5
        
        if company_data['ev_ebitda'] > criteria['max_ev_ebitda']:
            passes = False
            reasons.append(f"EV/EBITDA too high: {company_data['ev_ebitda']:.1f}")
        else:
            score += 5
        
        if company_data['price_to_fcf'] > criteria['max_price_to_fcf']:
            passes = False
            reasons.append(f"Price-to-FCF too high: {company_data['price_to_fcf']:.1f}")
        else:
            score += 5
        
        # Quality scores
        if company_data['moat_score'] < criteria['min_moat_score']:
            passes = False
            reasons.append(f"Moat score too low: {company_data['moat_score']}/10")
        else:
            score += 10
        
        if company_data['management_score'] < criteria['min_management_score']:
            passes = False
            reasons.append(f"Management score too low: {company_data['management_score']}/10")
        else:
            score += 10
        
        return {
            'passes': passes,
            'score': score,
            'reasons': reasons,
            'criteria': criteria
        }
    
    def perform_dcf_analysis(self, company_data: Dict[str, Any]) -> Dict[str, Any]:
        """Perform simplified DCF analysis"""
        try:
            current_fcf = company_data['free_cash_flow']
            current_price = company_data['current_price']
            shares_outstanding = company_data['shares_outstanding']
            
            if current_fcf <= 0 or shares_outstanding <= 0:
                return None
            
            # Conservative assumptions
            growth_rate_1_3_years = min(company_data['revenue_growth_5y'], 0.15)
            growth_rate_4_10_years = min(growth_rate_1_3_years * 0.5, 0.08)
            terminal_growth_rate = 0.03
            discount_rate = 0.10  # 10% WACC
            
            # Project 10 years of cash flows
            projected_cash_flows = []
            for year in range(1, 11):
                if year <= 3:
                    growth = growth_rate_1_3_years
                else:
                    growth = growth_rate_4_10_years
                
                fcf = current_fcf * ((1 + growth) ** year)
                projected_cash_flows.append(fcf)
            
            # Calculate present value of cash flows
            pv_cash_flows = sum(
                cf / ((1 + discount_rate) ** (i + 1))
                for i, cf in enumerate(projected_cash_flows)
            )
            
            # Calculate terminal value
            terminal_fcf = projected_cash_flows[-1] * (1 + terminal_growth_rate)
            terminal_value = terminal_fcf / (discount_rate - terminal_growth_rate)
            pv_terminal_value = terminal_value / ((1 + discount_rate) ** 10)
            
            # Calculate intrinsic value
            enterprise_value = pv_cash_flows + pv_terminal_value
            equity_value = enterprise_value - company_data['total_debt']
            intrinsic_value_per_share = equity_value / shares_outstanding
            
            # Calculate margin of safety
            margin_of_safety = (intrinsic_value_per_share - current_price) / intrinsic_value_per_share
            upside_potential = (intrinsic_value_per_share - current_price) / current_price
            
            return {
                'intrinsic_value_per_share': intrinsic_value_per_share,
                'current_price': current_price,
                'margin_of_safety': margin_of_safety,
                'upside_potential': upside_potential,
                'enterprise_value': enterprise_value,
                'equity_value': equity_value,
                'growth_rate_1_3_years': growth_rate_1_3_years,
                'growth_rate_4_10_years': growth_rate_4_10_years,
                'terminal_growth_rate': terminal_growth_rate,
                'discount_rate': discount_rate
            }
            
        except Exception as e:
            logger.error(f"Error performing DCF analysis: {e}")
            return None
    
    def run_screening(self, symbols: List[str] = None) -> Dict[str, Any]:
        """Run the complete Buffett screening process"""
        if symbols is None:
            symbols = self.get_sp500_symbols()
        
        logger.info(f"Starting Buffett screening for {len(symbols)} companies")
        start_time = time.time()
        
        companies = []
        companies_passing_screen = []
        companies_with_mos_30_plus = []
        
        for i, symbol in enumerate(symbols):
            try:
                logger.info(f"Processing {symbol} ({i+1}/{len(symbols)})")
                
                # Fetch company data
                company_data = self.fetch_company_data(symbol)
                if not company_data:
                    continue
                
                # Apply screening criteria
                screening_result = self.apply_buffett_criteria(company_data)
                company_data.update(screening_result)
                
                companies.append(company_data)
                
                if company_data['passes']:
                    companies_passing_screen.append(company_data)
                    logger.info(f"✅ {symbol} passed screening (score: {company_data['score']})")
                    
                    # Perform DCF analysis for shortlisted companies
                    dcf_analysis = self.perform_dcf_analysis(company_data)
                    if dcf_analysis:
                        company_data['dcf'] = dcf_analysis
                        
                        if dcf_analysis['margin_of_safety'] >= 0.30:
                            companies_with_mos_30_plus.append(company_data)
                            logger.info(f"🎯 {symbol} has MOS ≥30%: {dcf_analysis['margin_of_safety']:.1%}")
                
                # Rate limiting
                time.sleep(0.1)
                
            except Exception as e:
                logger.error(f"Error processing {symbol}: {e}")
                continue
        
        # Calculate summary statistics
        screening_duration = time.time() - start_time
        
        avg_moat_score = np.mean([c['moat_score'] for c in companies_passing_screen]) if companies_passing_screen else 0
        avg_management_score = np.mean([c['management_score'] for c in companies_passing_screen]) if companies_passing_screen else 0
        avg_margin_of_safety = np.mean([c['dcf']['margin_of_safety'] for c in companies_with_mos_30_plus if c.get('dcf')]) if companies_with_mos_30_plus else 0
        avg_upside_potential = np.mean([c['dcf']['upside_potential'] for c in companies_with_mos_30_plus if c.get('dcf')]) if companies_with_mos_30_plus else 0
        
        results = {
            'timestamp': datetime.now().isoformat(),
            'total_companies_screened': len(companies),
            'companies_passing_screen': len(companies_passing_screen),
            'companies_with_mos_30_plus': len(companies_with_mos_30_plus),
            'screening_duration_seconds': screening_duration,
            'avg_moat_score': float(avg_moat_score),
            'avg_management_score': float(avg_management_score),
            'avg_margin_of_safety': float(avg_margin_of_safety),
            'avg_upside_potential': float(avg_upside_potential),
            'top_candidates': companies_with_mos_30_plus[:10],  # Top 10
            'all_companies': companies
        }
        
        logger.info(f"Screening complete: {len(companies_passing_screen)} passed, {len(companies_with_mos_30_plus)} with MOS ≥30%")
        
        return results
    
    def get_cached_results(self) -> Optional[Dict[str, Any]]:
        """Get cached results if still fresh"""
        if (self.cached_results and self.last_scan_time and 
            time.time() - self.last_scan_time < self.cache_duration):
            return self.cached_results
        return None
    
    def update_cache(self, results: Dict[str, Any]):
        """Update cache with new results"""
        self.cached_results = results
        self.last_scan_time = time.time()

# Global screener instance
screener = BuffettScreener()

def get_buffett_screener_results() -> Dict[str, Any]:
    """Get Buffett screener results (cached or fresh)"""
    # Check cache first
    cached_results = screener.get_cached_results()
    if cached_results:
        logger.info("Returning cached Buffett screener results")
        return cached_results
    
    # Run fresh screening
    logger.info("Running fresh Buffett screening")
    results = screener.run_screening()
    screener.update_cache(results)
    
    return results

if __name__ == "__main__":
    # Test the screener
    results = get_buffett_screener_results()
    print(f"Found {results['companies_with_mos_30_plus']} companies with MOS ≥30%")
    
    # Save results to JSON
    with open('buffett_screener_results.json', 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    print("Results saved to buffett_screener_results.json")
