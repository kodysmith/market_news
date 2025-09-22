from dataclasses import dataclass
from typing import Optional, List, Dict, Any
from datetime import datetime
from decimal import Decimal

@dataclass
class FinancialMetrics:
    """Core financial metrics for Buffett-style analysis"""
    # Revenue & Growth
    revenue_ttm: Decimal
    revenue_growth_3y: Decimal
    revenue_growth_5y: Decimal
    
    # Profitability
    net_income_ttm: Decimal
    net_margin: Decimal
    roe: Decimal  # Return on Equity
    roa: Decimal  # Return on Assets
    roic: Decimal  # Return on Invested Capital
    
    # Debt & Leverage
    total_debt: Decimal
    debt_to_equity: Decimal
    interest_coverage: Decimal
    current_ratio: Decimal
    
    # Cash Flow
    operating_cash_flow_ttm: Decimal
    free_cash_flow_ttm: Decimal
    fcf_margin: Decimal
    capex_ttm: Decimal
    
    # Valuation
    market_cap: Decimal
    enterprise_value: Decimal
    pe_ratio: Decimal
    pb_ratio: Decimal
    ps_ratio: Decimal
    ev_ebitda: Decimal
    price_to_fcf: Decimal
    
    # Dividends
    dividend_yield: Decimal
    dividend_payout_ratio: Decimal
    dividend_growth_5y: Decimal
    
    # Share Count
    shares_outstanding: Decimal
    shares_outstanding_5y_ago: Decimal
    share_buyback_rate: Decimal

@dataclass
class MoatAnalysis:
    """Economic moat analysis"""
    brand_strength: int  # 1-10 scale
    switching_costs: int  # 1-10 scale
    network_effects: int  # 1-10 scale
    cost_advantages: int  # 1-10 scale
    regulatory_barriers: int  # 1-10 scale
    overall_moat_score: int  # 1-10 scale
    moat_description: str

@dataclass
class ManagementQuality:
    """Management quality assessment"""
    ceo_tenure: int  # years
    insider_ownership: Decimal  # percentage
    share_repurchases: bool
    debt_management: int  # 1-10 scale
    capital_allocation: int  # 1-10 scale
    transparency: int  # 1-10 scale
    overall_score: int  # 1-10 scale

@dataclass
class DCFAnalysis:
    """Discounted Cash Flow analysis"""
    # Assumptions
    growth_rate_1_3_years: Decimal
    growth_rate_4_10_years: Decimal
    terminal_growth_rate: Decimal
    discount_rate: Decimal
    
    # Calculations
    present_value_cash_flows: Decimal
    terminal_value: Decimal
    enterprise_value: Decimal
    equity_value: Decimal
    intrinsic_value_per_share: Decimal
    
    # Current vs Intrinsic
    current_price: Decimal
    margin_of_safety: Decimal  # percentage
    upside_potential: Decimal  # percentage

@dataclass
class Company:
    """Complete company profile for Buffett analysis"""
    # Basic Info
    symbol: str
    name: str
    sector: str
    industry: str
    market_cap_category: str  # Large, Mid, Small
    
    # Financial Data
    financials: FinancialMetrics
    moat: MoatAnalysis
    management: ManagementQuality
    dcf: Optional[DCFAnalysis] = None
    
    # Screening Results
    passes_initial_screen: bool = False
    screening_score: int = 0  # 0-100
    screening_reasons: List[str] = None
    
    # Timestamps
    last_updated: datetime = None
    data_source: str = "unknown"
    
    def __post_init__(self):
        if self.screening_reasons is None:
            self.screening_reasons = []

@dataclass
class ScreeningCriteria:
    """Buffett-style screening criteria"""
    # Market Cap
    min_market_cap: Decimal = Decimal('1000000000')  # $1B
    max_market_cap: Decimal = Decimal('1000000000000')  # $1T
    
    # Financial Health
    min_revenue_growth_5y: Decimal = Decimal('0.05')  # 5%
    min_net_margin: Decimal = Decimal('0.10')  # 10%
    min_roe: Decimal = Decimal('0.15')  # 15%
    min_roic: Decimal = Decimal('0.12')  # 12%
    max_debt_to_equity: Decimal = Decimal('0.50')  # 50%
    min_interest_coverage: Decimal = Decimal('5.0')  # 5x
    min_current_ratio: Decimal = Decimal('1.5')  # 1.5x
    
    # Cash Flow
    min_fcf_margin: Decimal = Decimal('0.08')  # 8%
    min_fcf_growth_5y: Decimal = Decimal('0.05')  # 5%
    
    # Valuation
    max_pe_ratio: Decimal = Decimal('25.0')  # 25x
    max_pb_ratio: Decimal = Decimal('3.0')  # 3x
    max_ev_ebitda: Decimal = Decimal('15.0')  # 15x
    max_price_to_fcf: Decimal = Decimal('20.0')  # 20x
    
    # Moat & Management
    min_moat_score: int = 6  # 6/10
    min_management_score: int = 6  # 6/10
    
    # Dividends (optional)
    min_dividend_yield: Decimal = Decimal('0.0')  # 0% (optional)
    min_dividend_growth_5y: Decimal = Decimal('0.0')  # 0% (optional)
    
    # Exclusions
    exclude_financials: bool = False
    exclude_utilities: bool = False
    exclude_tech: bool = False
    exclude_biotech: bool = False

@dataclass
class ScreeningResult:
    """Result of screening process"""
    total_companies_screened: int
    companies_passing_screen: int
    companies_for_dcf: int
    companies_with_mos_30_plus: int
    
    # Top candidates
    top_candidates: List[Company]
    
    # Summary stats
    avg_moat_score: Decimal
    avg_management_score: Decimal
    avg_margin_of_safety: Decimal
    avg_upside_potential: Decimal
    
    # Screening time
    screening_duration_seconds: float
    dcf_duration_seconds: float
    total_duration_seconds: float

