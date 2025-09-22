#!/usr/bin/env python3
"""
Buffett Screener - Warren Buffett-style value investment analysis
"""

import logging
import json
from datetime import datetime
from decimal import Decimal
from typing import List

from models.company import ScreeningCriteria
from data.financial_fetcher import FinancialDataFetcher
from screening.screener import BuffettScreener

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def create_default_criteria() -> ScreeningCriteria:
    """Create default Buffett-style screening criteria"""
    return ScreeningCriteria(
        min_market_cap=Decimal('10000000000'),  # $10B
        max_market_cap=Decimal('1000000000000'),  # $1T
        min_revenue_growth_5y=Decimal('0.05'),  # 5%
        min_net_margin=Decimal('0.10'),  # 10%
        min_roe=Decimal('0.15'),  # 15%
        min_roic=Decimal('0.12'),  # 12%
        max_debt_to_equity=Decimal('0.50'),  # 50%
        min_interest_coverage=Decimal('5.0'),  # 5x
        min_current_ratio=Decimal('1.5'),  # 1.5x
        min_fcf_margin=Decimal('0.08'),  # 8%
        min_fcf_growth_5y=Decimal('0.05'),  # 5%
        max_pe_ratio=Decimal('25.0'),  # 25x
        max_pb_ratio=Decimal('3.0'),  # 3x
        max_ev_ebitda=Decimal('15.0'),  # 15x
        max_price_to_fcf=Decimal('20.0'),  # 20x
        min_moat_score=6,  # 6/10
        min_management_score=6,  # 6/10
        min_dividend_yield=Decimal('0.0'),  # 0% (optional)
        min_dividend_growth_5y=Decimal('0.0'),  # 0% (optional)
        exclude_financials=False,
        exclude_utilities=False,
        exclude_tech=False,
        exclude_biotech=False
    )

def get_sample_symbols() -> List[str]:
    """Get sample symbols for screening"""
    return [
        'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'META', 'NVDA', 'BRK-B',
        'JNJ', 'V', 'PG', 'UNH', 'HD', 'MA', 'DIS', 'PYPL', 'ADBE', 'NFLX',
        'CRM', 'INTC', 'CMCSA', 'PFE', 'T', 'ABT', 'PEP', 'KO', 'WMT', 'MRK',
        'BAC', 'XOM', 'JPM', 'CVX', 'LLY', 'AVGO', 'ACN', 'COST', 'DHR', 'VZ',
        'NKE', 'MCD', 'SBUX', 'LMT', 'RTX', 'BA', 'CAT', 'GE', 'MMM', 'HON'
    ]

def save_results_to_json(results, filename: str = 'buffett_screener_results.json'):
    """Save screening results to JSON file"""
    try:
        # Convert Decimal to float for JSON serialization
        def convert_decimals(obj):
            if isinstance(obj, Decimal):
                return float(obj)
            elif isinstance(obj, dict):
                return {k: convert_decimals(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_decimals(item) for item in obj]
            return obj
        
        results_dict = convert_decimals(results.__dict__)
        
        with open(filename, 'w') as f:
            json.dump(results_dict, f, indent=2, default=str)
        
        logger.info(f"Results saved to {filename}")
        
    except Exception as e:
        logger.error(f"Error saving results: {e}")

def main():
    """Main screening function"""
    logger.info("🎯 Starting Buffett Screener")
    
    # Initialize components
    fetcher = FinancialDataFetcher()
    screener = BuffettScreener(fetcher)
    
    # Get symbols to screen
    symbols = get_sample_symbols()
    logger.info(f"📊 Screening {len(symbols)} companies")
    
    # Create screening criteria
    criteria = create_default_criteria()
    logger.info("📋 Using default Buffett criteria")
    
    # Run screening
    try:
        results = screener.screen_companies(symbols, criteria)
        
        # Display results
        logger.info("=" * 60)
        logger.info("📈 SCREENING RESULTS")
        logger.info("=" * 60)
        logger.info(f"Total companies screened: {results.total_companies_screened}")
        logger.info(f"Companies passing screen: {results.companies_passing_screen}")
        logger.info(f"Companies with DCF analysis: {results.companies_for_dcf}")
        logger.info(f"Companies with MOS ≥30%: {results.companies_with_mos_30_plus}")
        logger.info(f"Average moat score: {results.avg_moat_score:.1f}/10")
        logger.info(f"Average management score: {results.avg_management_score:.1f}/10")
        logger.info(f"Average margin of safety: {results.avg_margin_of_safety:.1%}")
        logger.info(f"Average upside potential: {results.avg_upside_potential:.1%}")
        logger.info(f"Screening duration: {results.screening_duration_seconds:.1f}s")
        
        # Display top candidates
        if results.top_candidates:
            logger.info("\n🏆 TOP CANDIDATES (MOS ≥30%)")
            logger.info("-" * 60)
            for i, company in enumerate(results.top_candidates[:5], 1):
                logger.info(f"{i}. {company.symbol} - {company.name}")
                logger.info(f"   Sector: {company.sector}")
                logger.info(f"   Market Cap: ${company.financials.market_cap:,.0f}")
                logger.info(f"   Moat Score: {company.moat.overall_moat_score}/10")
                logger.info(f"   Management Score: {company.management.overall_score}/10")
                if company.dcf:
                    logger.info(f"   Intrinsic Value: ${company.dcf.intrinsic_value_per_share:.2f}")
                    logger.info(f"   Current Price: ${company.dcf.current_price:.2f}")
                    logger.info(f"   Margin of Safety: {company.dcf.margin_of_safety:.1%}")
                    logger.info(f"   Upside Potential: {company.dcf.upside_potential:.1%}")
                logger.info("")
        
        # Save results
        save_results_to_json(results)
        
        logger.info("✅ Buffett screening completed successfully!")
        
    except Exception as e:
        logger.error(f"❌ Error during screening: {e}")
        raise

if __name__ == "__main__":
    main()

