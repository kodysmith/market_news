import logging
from typing import List, Optional
from decimal import Decimal
from datetime import datetime

from ..models.company import Company, ScreeningCriteria, ScreeningResult
from ..data.financial_fetcher import FinancialDataFetcher

logger = logging.getLogger(__name__)

class BuffettScreener:
    """Warren Buffett-style company screener"""
    
    def __init__(self, fetcher: FinancialDataFetcher):
        self.fetcher = fetcher
        
    def screen_companies(self, symbols: List[str], criteria: ScreeningCriteria) -> ScreeningResult:
        """Screen companies using Buffett criteria"""
        start_time = datetime.now()
        
        logger.info(f"Starting Buffett screening for {len(symbols)} companies")
        
        companies = []
        companies_passing_screen = []
        companies_for_dcf = []
        companies_with_mos_30_plus = []
        
        for i, symbol in enumerate(symbols):
            try:
                logger.info(f"Processing {symbol} ({i+1}/{len(symbols)})")
                
                # Fetch company data
                company = self.fetcher.get_company_data(symbol)
                if not company:
                    logger.warning(f"No data available for {symbol}")
                    continue
                
                companies.append(company)
                
                # Apply initial screening
                passes_screen, score, reasons = self._apply_initial_screen(company, criteria)
                company.passes_initial_screen = passes_screen
                company.screening_score = score
                company.screening_reasons = reasons
                
                if passes_screen:
                    companies_passing_screen.append(company)
                    logger.info(f"✅ {symbol} passed initial screen (score: {score})")
                    
                    # Apply DCF analysis for shortlisted companies
                    dcf_start = datetime.now()
                    dcf_analysis = self._perform_dcf_analysis(company)
                    dcf_duration = (datetime.now() - dcf_start).total_seconds()
                    
                    if dcf_analysis and dcf_analysis.margin_of_safety >= Decimal('0.30'):
                        companies_with_mos_30_plus.append(company)
                        logger.info(f"🎯 {symbol} has MOS ≥30%: {dcf_analysis.margin_of_safety:.1%}")
                    
                    if dcf_analysis:
                        companies_for_dcf.append(company)
                
            except Exception as e:
                logger.error(f"Error processing {symbol}: {e}")
                continue
        
        # Calculate summary statistics
        screening_duration = (datetime.now() - start_time).total_seconds()
        
        avg_moat_score = self._calculate_avg_score(companies_passing_screen, 'moat')
        avg_management_score = self._calculate_avg_score(companies_passing_screen, 'management')
        avg_margin_of_safety = self._calculate_avg_margin_of_safety(companies_with_mos_30_plus)
        avg_upside_potential = self._calculate_avg_upside_potential(companies_with_mos_30_plus)
        
        result = ScreeningResult(
            total_companies_screened=len(companies),
            companies_passing_screen=len(companies_passing_screen),
            companies_for_dcf=len(companies_for_dcf),
            companies_with_mos_30_plus=len(companies_with_mos_30_plus),
            top_candidates=companies_with_mos_30_plus[:10],  # Top 10
            avg_moat_score=avg_moat_score,
            avg_management_score=avg_management_score,
            avg_margin_of_safety=avg_margin_of_safety,
            avg_upside_potential=avg_upside_potential,
            screening_duration_seconds=screening_duration,
            dcf_duration_seconds=0,  # Will be calculated separately
            total_duration_seconds=screening_duration
        )
        
        logger.info(f"Screening complete: {len(companies_passing_screen)} passed, "
                   f"{len(companies_with_mos_30_plus)} with MOS ≥30%")
        
        return result
    
    def _apply_initial_screen(self, company: Company, criteria: ScreeningCriteria) -> tuple[bool, int, List[str]]:
        """Apply initial screening criteria"""
        reasons = []
        score = 0
        max_score = 100
        
        # Market cap check
        if not (criteria.min_market_cap <= company.financials.market_cap <= criteria.max_market_cap):
            reasons.append(f"Market cap outside range: ${company.financials.market_cap:,.0f}")
            return False, 0, reasons
        score += 10
        
        # Revenue growth
        if company.financials.revenue_growth_5y < criteria.min_revenue_growth_5y:
            reasons.append(f"Revenue growth too low: {company.financials.revenue_growth_5y:.1%}")
            return False, 0, reasons
        score += 10
        
        # Profitability checks
        if company.financials.net_margin < criteria.min_net_margin:
            reasons.append(f"Net margin too low: {company.financials.net_margin:.1%}")
            return False, 0, reasons
        score += 10
        
        if company.financials.roe < criteria.min_roe:
            reasons.append(f"ROE too low: {company.financials.roe:.1%}")
            return False, 0, reasons
        score += 10
        
        if company.financials.roic < criteria.min_roic:
            reasons.append(f"ROIC too low: {company.financials.roic:.1%}")
            return False, 0, reasons
        score += 10
        
        # Debt checks
        if company.financials.debt_to_equity > criteria.max_debt_to_equity:
            reasons.append(f"Debt-to-equity too high: {company.financials.debt_to_equity:.2f}")
            return False, 0, reasons
        score += 10
        
        if company.financials.interest_coverage < criteria.min_interest_coverage:
            reasons.append(f"Interest coverage too low: {company.financials.interest_coverage:.1f}x")
            return False, 0, reasons
        score += 5
        
        if company.financials.current_ratio < criteria.min_current_ratio:
            reasons.append(f"Current ratio too low: {company.financials.current_ratio:.2f}")
            return False, 0, reasons
        score += 5
        
        # Cash flow checks
        if company.financials.fcf_margin < criteria.min_fcf_margin:
            reasons.append(f"FCF margin too low: {company.financials.fcf_margin:.1%}")
            return False, 0, reasons
        score += 10
        
        # Valuation checks
        if company.financials.pe_ratio > criteria.max_pe_ratio:
            reasons.append(f"P/E ratio too high: {company.financials.pe_ratio:.1f}")
            return False, 0, reasons
        score += 5
        
        if company.financials.pb_ratio > criteria.max_pb_ratio:
            reasons.append(f"P/B ratio too high: {company.financials.pb_ratio:.1f}")
            return False, 0, reasons
        score += 5
        
        if company.financials.ev_ebitda > criteria.max_ev_ebitda:
            reasons.append(f"EV/EBITDA too high: {company.financials.ev_ebitda:.1f}")
            return False, 0, reasons
        score += 5
        
        if company.financials.price_to_fcf > criteria.max_price_to_fcf:
            reasons.append(f"Price-to-FCF too high: {company.financials.price_to_fcf:.1f}")
            return False, 0, reasons
        score += 5
        
        # Moat and management checks
        if company.moat.overall_moat_score < criteria.min_moat_score:
            reasons.append(f"Moat score too low: {company.moat.overall_moat_score}/10")
            return False, 0, reasons
        score += 10
        
        if company.management.overall_score < criteria.min_management_score:
            reasons.append(f"Management score too low: {company.management.overall_score}/10")
            return False, 0, reasons
        score += 10
        
        # Industry exclusions
        if criteria.exclude_financials and 'financial' in company.sector.lower():
            reasons.append("Excluded: Financial sector")
            return False, 0, reasons
        
        if criteria.exclude_utilities and 'utilities' in company.sector.lower():
            reasons.append("Excluded: Utilities sector")
            return False, 0, reasons
        
        if criteria.exclude_tech and 'technology' in company.sector.lower():
            reasons.append("Excluded: Technology sector")
            return False, 0, reasons
        
        if criteria.exclude_biotech and 'biotechnology' in company.industry.lower():
            reasons.append("Excluded: Biotechnology industry")
            return False, 0, reasons
        
        # All checks passed
        reasons.append("Passed all screening criteria")
        return True, score, reasons
    
    def _perform_dcf_analysis(self, company: Company) -> Optional[object]:
        """Perform DCF analysis (simplified)"""
        try:
            # This is a simplified DCF - in reality would be much more sophisticated
            from ..models.company import DCFAnalysis
            
            # Get current price
            current_price = company.financials.market_cap / company.financials.shares_outstanding
            
            # Make conservative assumptions
            growth_rate_1_3_years = min(company.financials.revenue_growth_5y, Decimal('0.15'))
            growth_rate_4_10_years = min(growth_rate_1_3_years * Decimal('0.5'), Decimal('0.08'))
            terminal_growth_rate = Decimal('0.03')
            discount_rate = Decimal('0.10')  # 10% WACC
            
            # Calculate projected cash flows (simplified)
            current_fcf = company.financials.free_cash_flow_ttm
            
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
            equity_value = enterprise_value - company.financials.total_debt
            intrinsic_value_per_share = equity_value / company.financials.shares_outstanding
            
            # Calculate margin of safety
            margin_of_safety = (intrinsic_value_per_share - current_price) / intrinsic_value_per_share
            upside_potential = (intrinsic_value_per_share - current_price) / current_price
            
            dcf_analysis = DCFAnalysis(
                growth_rate_1_3_years=growth_rate_1_3_years,
                growth_rate_4_10_years=growth_rate_4_10_years,
                terminal_growth_rate=terminal_growth_rate,
                discount_rate=discount_rate,
                present_value_cash_flows=pv_cash_flows,
                terminal_value=terminal_value,
                enterprise_value=enterprise_value,
                equity_value=equity_value,
                intrinsic_value_per_share=intrinsic_value_per_share,
                current_price=current_price,
                margin_of_safety=margin_of_safety,
                upside_potential=upside_potential
            )
            
            company.dcf = dcf_analysis
            return dcf_analysis
            
        except Exception as e:
            logger.error(f"Error performing DCF analysis for {company.symbol}: {e}")
            return None
    
    def _calculate_avg_score(self, companies: List[Company], score_type: str) -> Decimal:
        """Calculate average score for companies"""
        if not companies:
            return Decimal('0')
        
        if score_type == 'moat':
            scores = [c.moat.overall_moat_score for c in companies]
        elif score_type == 'management':
            scores = [c.management.overall_score for c in companies]
        else:
            return Decimal('0')
        
        return Decimal(sum(scores)) / Decimal(len(scores))
    
    def _calculate_avg_margin_of_safety(self, companies: List[Company]) -> Decimal:
        """Calculate average margin of safety"""
        if not companies:
            return Decimal('0')
        
        mos_values = [c.dcf.margin_of_safety for c in companies if c.dcf]
        if not mos_values:
            return Decimal('0')
        
        return Decimal(sum(mos_values)) / Decimal(len(mos_values))
    
    def _calculate_avg_upside_potential(self, companies: List[Company]) -> Decimal:
        """Calculate average upside potential"""
        if not companies:
            return Decimal('0')
        
        upside_values = [c.dcf.upside_potential for c in companies if c.dcf]
        if not upside_values:
            return Decimal('0')
        
        return Decimal(sum(upside_values)) / Decimal(len(upside_values))

