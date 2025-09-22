import yfinance as yf
import requests
import time
from typing import Optional, Dict, Any, List
from decimal import Decimal
from datetime import datetime, timedelta
import logging

from ..models.company import Company, FinancialMetrics, MoatAnalysis, ManagementQuality

logger = logging.getLogger(__name__)

class FinancialDataFetcher:
    """Fetches financial data for Buffett-style analysis"""
    
    def __init__(self, api_key: str = None):
        self.api_key = api_key
        self.session = requests.Session()
        self.last_request_time = 0
        self.min_request_interval = 1.0  # 1 second between requests
        
    def _rate_limit(self):
        """Implement rate limiting"""
        current_time = time.time()
        time_since_last = current_time - self.last_request_time
        
        if time_since_last < self.min_request_interval:
            sleep_time = self.min_request_interval - time_since_last
            time.sleep(sleep_time)
        
        self.last_request_time = time.time()
    
    def get_company_data(self, symbol: str) -> Optional[Company]:
        """Get complete company data for Buffett analysis"""
        try:
            self._rate_limit()
            
            # Get basic info and financials from yfinance
            ticker = yf.Ticker(symbol)
            info = ticker.info
            financials = ticker.financials
            balance_sheet = ticker.balance_sheet
            cash_flow = ticker.cashflow
            quarterly_financials = ticker.quarterly_financials
            
            if not info or not financials.empty:
                logger.warning(f"No data available for {symbol}")
                return None
            
            # Extract financial metrics
            financial_metrics = self._extract_financial_metrics(info, financials, balance_sheet, cash_flow)
            if not financial_metrics:
                return None
            
            # Analyze moat (simplified for now)
            moat = self._analyze_moat(symbol, info, financial_metrics)
            
            # Analyze management quality (simplified for now)
            management = self._analyze_management(symbol, info, financial_metrics)
            
            # Create company object
            company = Company(
                symbol=symbol,
                name=info.get('longName', symbol),
                sector=info.get('sector', 'Unknown'),
                industry=info.get('industry', 'Unknown'),
                market_cap_category=self._get_market_cap_category(info.get('marketCap', 0)),
                financials=financial_metrics,
                moat=moat,
                management=management,
                last_updated=datetime.now(),
                data_source='yfinance'
            )
            
            return company
            
        except Exception as e:
            logger.error(f"Error fetching data for {symbol}: {e}")
            return None
    
    def _extract_financial_metrics(self, info: Dict, financials, balance_sheet, cash_flow) -> Optional[FinancialMetrics]:
        """Extract financial metrics from yfinance data"""
        try:
            # Revenue
            revenue_ttm = Decimal(str(info.get('totalRevenue', 0)))
            
            # Calculate growth rates (simplified)
            revenue_growth_3y = self._calculate_revenue_growth(financials, 3)
            revenue_growth_5y = self._calculate_revenue_growth(financials, 5)
            
            # Net income and margins
            net_income_ttm = Decimal(str(info.get('netIncomeToCommon', 0)))
            net_margin = net_income_ttm / revenue_ttm if revenue_ttm > 0 else Decimal('0')
            
            # ROE, ROA, ROIC
            total_equity = Decimal(str(info.get('totalStockholderEquity', 0)))
            total_assets = Decimal(str(info.get('totalAssets', 0)))
            
            roe = net_income_ttm / total_equity if total_equity > 0 else Decimal('0')
            roa = net_income_ttm / total_assets if total_assets > 0 else Decimal('0')
            roic = self._calculate_roic(info, financials)
            
            # Debt metrics
            total_debt = Decimal(str(info.get('totalDebt', 0)))
            debt_to_equity = total_debt / total_equity if total_equity > 0 else Decimal('0')
            
            # Interest coverage (simplified)
            interest_expense = Decimal(str(info.get('interestExpense', 0)))
            interest_coverage = (net_income_ttm + interest_expense) / interest_expense if interest_expense > 0 else Decimal('999')
            
            # Current ratio
            current_assets = Decimal(str(info.get('totalCurrentAssets', 0)))
            current_liabilities = Decimal(str(info.get('totalCurrentLiabilities', 0)))
            current_ratio = current_assets / current_liabilities if current_liabilities > 0 else Decimal('0')
            
            # Cash flow
            operating_cash_flow_ttm = Decimal(str(info.get('operatingCashflow', 0)))
            capex_ttm = Decimal(str(info.get('capitalExpenditures', 0)))
            free_cash_flow_ttm = operating_cash_flow_ttm - capex_ttm
            fcf_margin = free_cash_flow_ttm / revenue_ttm if revenue_ttm > 0 else Decimal('0')
            
            # Valuation metrics
            market_cap = Decimal(str(info.get('marketCap', 0)))
            enterprise_value = Decimal(str(info.get('enterpriseValue', 0)))
            pe_ratio = Decimal(str(info.get('trailingPE', 0)))
            pb_ratio = Decimal(str(info.get('priceToBook', 0)))
            ps_ratio = Decimal(str(info.get('priceToSalesTrailing12Months', 0)))
            ev_ebitda = Decimal(str(info.get('enterpriseToEbitda', 0)))
            price_to_fcf = market_cap / free_cash_flow_ttm if free_cash_flow_ttm > 0 else Decimal('0')
            
            # Dividends
            dividend_yield = Decimal(str(info.get('dividendYield', 0)))
            dividend_payout_ratio = Decimal(str(info.get('payoutRatio', 0)))
            dividend_growth_5y = self._calculate_dividend_growth(info)
            
            # Share count
            shares_outstanding = Decimal(str(info.get('sharesOutstanding', 0)))
            shares_outstanding_5y_ago = self._get_shares_5y_ago(info)
            share_buyback_rate = self._calculate_share_buyback_rate(shares_outstanding, shares_outstanding_5y_ago)
            
            return FinancialMetrics(
                revenue_ttm=revenue_ttm,
                revenue_growth_3y=revenue_growth_3y,
                revenue_growth_5y=revenue_growth_5y,
                net_income_ttm=net_income_ttm,
                net_margin=net_margin,
                roe=roe,
                roa=roa,
                roic=roic,
                total_debt=total_debt,
                debt_to_equity=debt_to_equity,
                interest_coverage=interest_coverage,
                current_ratio=current_ratio,
                operating_cash_flow_ttm=operating_cash_flow_ttm,
                free_cash_flow_ttm=free_cash_flow_ttm,
                fcf_margin=fcf_margin,
                capex_ttm=capex_ttm,
                market_cap=market_cap,
                enterprise_value=enterprise_value,
                pe_ratio=pe_ratio,
                pb_ratio=pb_ratio,
                ps_ratio=ps_ratio,
                ev_ebitda=ev_ebitda,
                price_to_fcf=price_to_fcf,
                dividend_yield=dividend_yield,
                dividend_payout_ratio=dividend_payout_ratio,
                dividend_growth_5y=dividend_growth_5y,
                shares_outstanding=shares_outstanding,
                shares_outstanding_5y_ago=shares_outstanding_5y_ago,
                share_buyback_rate=share_buyback_rate
            )
            
        except Exception as e:
            logger.error(f"Error extracting financial metrics: {e}")
            return None
    
    def _calculate_revenue_growth(self, financials, years: int) -> Decimal:
        """Calculate revenue growth over specified years"""
        try:
            if financials.empty or len(financials.columns) < years:
                return Decimal('0')
            
            # Get revenue data (assuming first row is revenue)
            revenue_data = financials.iloc[0]  # First row is usually revenue
            
            if len(revenue_data) < years + 1:
                return Decimal('0')
            
            current_revenue = Decimal(str(revenue_data.iloc[0]))
            past_revenue = Decimal(str(revenue_data.iloc[years]))
            
            if past_revenue > 0:
                growth_rate = (current_revenue - past_revenue) / past_revenue
                return growth_rate
            else:
                return Decimal('0')
                
        except Exception as e:
            logger.error(f"Error calculating revenue growth: {e}")
            return Decimal('0')
    
    def _calculate_roic(self, info: Dict, financials) -> Decimal:
        """Calculate Return on Invested Capital"""
        try:
            net_income = Decimal(str(info.get('netIncomeToCommon', 0)))
            total_debt = Decimal(str(info.get('totalDebt', 0)))
            total_equity = Decimal(str(info.get('totalStockholderEquity', 0)))
            
            invested_capital = total_debt + total_equity
            
            if invested_capital > 0:
                return net_income / invested_capital
            else:
                return Decimal('0')
                
        except Exception as e:
            logger.error(f"Error calculating ROIC: {e}")
            return Decimal('0')
    
    def _calculate_dividend_growth(self, info: Dict) -> Decimal:
        """Calculate 5-year dividend growth rate"""
        try:
            # This is simplified - in reality you'd need historical dividend data
            current_dividend = Decimal(str(info.get('dividendRate', 0)))
            # For now, return 0 - would need historical data for accurate calculation
            return Decimal('0')
        except:
            return Decimal('0')
    
    def _get_shares_5y_ago(self, info: Dict) -> Decimal:
        """Get shares outstanding 5 years ago"""
        try:
            # This is simplified - would need historical data
            current_shares = Decimal(str(info.get('sharesOutstanding', 0)))
            # Assume 5% annual dilution for now
            return current_shares * Decimal('1.25')  # Rough estimate
        except:
            return Decimal('0')
    
    def _calculate_share_buyback_rate(self, current_shares: Decimal, past_shares: Decimal) -> Decimal:
        """Calculate share buyback rate"""
        try:
            if past_shares > 0:
                return (past_shares - current_shares) / past_shares
            else:
                return Decimal('0')
        except:
            return Decimal('0')
    
    def _analyze_moat(self, symbol: str, info: Dict, financials: FinancialMetrics) -> MoatAnalysis:
        """Analyze economic moat (simplified)"""
        try:
            # This is a simplified moat analysis
            # In reality, this would require much more sophisticated analysis
            
            # Brand strength based on market cap and margins
            brand_strength = min(10, max(1, int(financials.net_margin * 50)))
            
            # Switching costs based on industry
            industry = info.get('industry', '').lower()
            if 'software' in industry or 'technology' in industry:
                switching_costs = 8
            elif 'banking' in industry or 'financial' in industry:
                switching_costs = 7
            else:
                switching_costs = 5
            
            # Network effects (simplified)
            network_effects = 5  # Default
            
            # Cost advantages based on margins
            cost_advantages = min(10, max(1, int(financials.net_margin * 30)))
            
            # Regulatory barriers
            if 'pharmaceutical' in industry or 'utilities' in industry:
                regulatory_barriers = 8
            else:
                regulatory_barriers = 3
            
            # Overall moat score
            overall_moat_score = (brand_strength + switching_costs + network_effects + 
                                cost_advantages + regulatory_barriers) // 5
            
            moat_description = f"Brand: {brand_strength}/10, Switching: {switching_costs}/10, " \
                             f"Network: {network_effects}/10, Cost: {cost_advantages}/10, " \
                             f"Regulatory: {regulatory_barriers}/10"
            
            return MoatAnalysis(
                brand_strength=brand_strength,
                switching_costs=switching_costs,
                network_effects=network_effects,
                cost_advantages=cost_advantages,
                regulatory_barriers=regulatory_barriers,
                overall_moat_score=overall_moat_score,
                moat_description=moat_description
            )
            
        except Exception as e:
            logger.error(f"Error analyzing moat for {symbol}: {e}")
            return MoatAnalysis(5, 5, 5, 5, 5, 5, "Analysis failed")
    
    def _analyze_management(self, symbol: str, info: Dict, financials: FinancialMetrics) -> ManagementQuality:
        """Analyze management quality (simplified)"""
        try:
            # CEO tenure (simplified - would need historical data)
            ceo_tenure = 5  # Default assumption
            
            # Insider ownership (simplified)
            insider_ownership = Decimal('0.05')  # 5% default
            
            # Share repurchases
            share_repurchases = financials.share_buyback_rate > 0
            
            # Debt management based on debt ratios
            debt_management = min(10, max(1, int(10 - financials.debt_to_equity * 10)))
            
            # Capital allocation based on ROIC
            capital_allocation = min(10, max(1, int(financials.roic * 50)))
            
            # Transparency (simplified)
            transparency = 7  # Default assumption
            
            # Overall score
            overall_score = (debt_management + capital_allocation + transparency) // 3
            
            return ManagementQuality(
                ceo_tenure=ceo_tenure,
                insider_ownership=insider_ownership,
                share_repurchases=share_repurchases,
                debt_management=debt_management,
                capital_allocation=capital_allocation,
                transparency=transparency,
                overall_score=overall_score
            )
            
        except Exception as e:
            logger.error(f"Error analyzing management for {symbol}: {e}")
            return ManagementQuality(5, Decimal('0.05'), False, 5, 5, 5, 5)
    
    def _get_market_cap_category(self, market_cap: int) -> str:
        """Categorize market cap"""
        if market_cap >= 200_000_000_000:  # $200B+
            return "Mega Cap"
        elif market_cap >= 10_000_000_000:  # $10B+
            return "Large Cap"
        elif market_cap >= 2_000_000_000:  # $2B+
            return "Mid Cap"
        else:
            return "Small Cap"
    
    def get_sp500_symbols(self) -> List[str]:
        """Get S&P 500 symbols for screening"""
        try:
            # This would typically fetch from a reliable source
            # For now, return a sample of well-known companies
            return [
                'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'META', 'NVDA', 'BRK-B',
                'JNJ', 'V', 'PG', 'UNH', 'HD', 'MA', 'DIS', 'PYPL', 'ADBE', 'NFLX',
                'CRM', 'INTC', 'CMCSA', 'PFE', 'T', 'ABT', 'PEP', 'KO', 'WMT', 'MRK',
                'BAC', 'XOM', 'JPM', 'CVX', 'LLY', 'AVGO', 'ACN', 'COST', 'DHR', 'VZ'
            ]
        except Exception as e:
            logger.error(f"Error getting S&P 500 symbols: {e}")
            return []

