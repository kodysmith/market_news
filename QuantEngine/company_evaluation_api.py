#!/usr/bin/env python3
"""
Company Evaluation API - Hedge Fund Level Stock Analysis

Provides comprehensive fundamental, technical, and competitive analysis for any stock.

Framework:
1. Fundamentals (The Balance Sheet and Income Engine)
2. Industry Momentum (The Macro Winds)
3. Company Leadership (The Competitive Edge)
4. Technical Sentiment (Overbought vs Oversold)
5. Support & Resistance (The Battlegrounds)
"""

import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json
from typing import Dict, Any, List, Optional
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class CompanyEvaluator:
    """Comprehensive stock evaluation engine"""
    
    def __init__(self, ticker: str):
        self.ticker = ticker.upper()
        self.stock = yf.Ticker(self.ticker)
        self.info = None
        self.history = None
        self.financials = None
        self.balance_sheet = None
        self.cashflow = None
        
    def evaluate(self) -> Dict[str, Any]:
        """
        Run complete evaluation and return comprehensive analysis
        """
        logger.info(f"🔍 Evaluating {self.ticker}...")
        
        try:
            # Load data
            self._load_data()
            
            # Run all evaluation components
            evaluation = {
                'ticker': self.ticker,
                'timestamp': datetime.now().isoformat(),
                'fundamentals': self._evaluate_fundamentals(),
                'industry_momentum': self._evaluate_industry_momentum(),
                'company_leadership': self._evaluate_company_leadership(),
                'technical_sentiment': self._evaluate_technical_sentiment(),
                'support_resistance': self._evaluate_support_resistance(),
                'overall_score': None,
                'recommendation': None
            }
            
            # Calculate overall score and recommendation
            evaluation['overall_score'] = self._calculate_overall_score(evaluation)
            evaluation['recommendation'] = self._generate_recommendation(evaluation)
            
            logger.info(f"✅ {self.ticker} evaluation complete")
            return evaluation
            
        except Exception as e:
            logger.error(f"❌ Failed to evaluate {self.ticker}: {e}")
            return {'error': str(e), 'ticker': self.ticker}
    
    def _load_data(self):
        """Load all necessary data"""
        logger.info(f"📊 Loading data for {self.ticker}...")
        
        # Basic info
        self.info = self.stock.info
        
        # Historical price data (2 years for comprehensive analysis)
        self.history = self.stock.history(period='2y')
        
        # Financial statements
        try:
            self.financials = self.stock.financials
            self.balance_sheet = self.stock.balance_sheet
            self.cashflow = self.stock.cashflow
        except Exception as e:
            logger.warning(f"⚠️ Could not load financials: {e}")
    
    # ==========================================
    # 1. FUNDAMENTALS (The Balance Sheet and Income Engine)
    # ==========================================
    
    def _evaluate_fundamentals(self) -> Dict[str, Any]:
        """
        Evaluate financial fundamentals:
        - Revenue & Earnings Growth
        - Margins
        - Debt Levels
        - Cash Flow
        - Valuation
        """
        logger.info("💰 Analyzing fundamentals...")
        
        fundamentals = {
            'revenue_growth': self._get_revenue_growth(),
            'earnings_growth': self._get_earnings_growth(),
            'margins': self._get_margins(),
            'debt_metrics': self._get_debt_metrics(),
            'cash_flow': self._get_cash_flow_metrics(),
            'valuation': self._get_valuation_metrics(),
            'score': 0,
            'verdict': ''
        }
        
        # Calculate fundamentals score (0-100)
        fundamentals['score'] = self._score_fundamentals(fundamentals)
        fundamentals['verdict'] = self._fundamentals_verdict(fundamentals['score'])
        
        return fundamentals
    
    def _get_revenue_growth(self) -> Dict[str, Any]:
        """Calculate revenue growth YoY"""
        try:
            if self.financials is None or self.financials.empty:
                return {'error': 'No financial data available'}
            
            # Get total revenue (top row)
            revenues = self.financials.loc['Total Revenue'] if 'Total Revenue' in self.financials.index else None
            
            if revenues is None or len(revenues) < 2:
                return {'error': 'Insufficient revenue data'}
            
            # Calculate YoY growth
            latest_revenue = float(revenues.iloc[0])
            previous_revenue = float(revenues.iloc[1])
            yoy_growth = ((latest_revenue - previous_revenue) / previous_revenue) * 100
            
            # Calculate 3-year CAGR if available
            cagr = None
            if len(revenues) >= 4:
                oldest_revenue = float(revenues.iloc[3])
                years = 3
                cagr = (((latest_revenue / oldest_revenue) ** (1/years)) - 1) * 100
            
            return {
                'latest_revenue': latest_revenue,
                'yoy_growth_pct': round(yoy_growth, 2),
                'cagr_3y_pct': round(cagr, 2) if cagr else None,
                'trend': 'Growing' if yoy_growth > 0 else 'Declining'
            }
            
        except Exception as e:
            logger.warning(f"Could not calculate revenue growth: {e}")
            return {'error': str(e)}
    
    def _get_earnings_growth(self) -> Dict[str, Any]:
        """Calculate earnings growth YoY"""
        try:
            eps_current = self.info.get('trailingEps')
            eps_forward = self.info.get('forwardEps')
            earnings_growth = self.info.get('earningsGrowth')
            
            return {
                'trailing_eps': eps_current,
                'forward_eps': eps_forward,
                'earnings_growth_pct': round(earnings_growth * 100, 2) if earnings_growth else None,
                'trend': 'Growing' if earnings_growth and earnings_growth > 0 else 'Declining'
            }
            
        except Exception as e:
            logger.warning(f"Could not calculate earnings growth: {e}")
            return {'error': str(e)}
    
    def _get_margins(self) -> Dict[str, Any]:
        """Calculate profit margins"""
        try:
            return {
                'gross_margin_pct': round(self.info.get('grossMargins', 0) * 100, 2),
                'operating_margin_pct': round(self.info.get('operatingMargins', 0) * 100, 2),
                'profit_margin_pct': round(self.info.get('profitMargins', 0) * 100, 2),
                'ebitda_margin_pct': round(self.info.get('ebitdaMargins', 0) * 100, 2)
            }
        except Exception as e:
            logger.warning(f"Could not calculate margins: {e}")
            return {'error': str(e)}
    
    def _get_debt_metrics(self) -> Dict[str, Any]:
        """Calculate debt levels and coverage"""
        try:
            debt_to_equity = self.info.get('debtToEquity', 0) / 100 if self.info.get('debtToEquity') else None
            interest_coverage = self.info.get('interestCoverage')
            total_debt = self.info.get('totalDebt', 0)
            total_cash = self.info.get('totalCash', 0)
            
            return {
                'debt_to_equity': round(debt_to_equity, 2) if debt_to_equity else None,
                'interest_coverage': round(interest_coverage, 2) if interest_coverage else None,
                'total_debt': total_debt,
                'total_cash': total_cash,
                'net_debt': total_debt - total_cash,
                'debt_health': self._assess_debt_health(debt_to_equity, interest_coverage)
            }
        except Exception as e:
            logger.warning(f"Could not calculate debt metrics: {e}")
            return {'error': str(e)}
    
    def _assess_debt_health(self, debt_to_equity: Optional[float], interest_coverage: Optional[float]) -> str:
        """Assess overall debt health"""
        if debt_to_equity is None:
            return 'Unknown'
        
        if debt_to_equity < 0.5 and (interest_coverage is None or interest_coverage > 5):
            return 'Excellent - Low debt, strong coverage'
        elif debt_to_equity < 1.0 and (interest_coverage is None or interest_coverage > 3):
            return 'Good - Manageable debt'
        elif debt_to_equity < 2.0:
            return 'Moderate - Watch closely'
        else:
            return 'High Risk - Heavy debt burden'
    
    def _get_cash_flow_metrics(self) -> Dict[str, Any]:
        """Calculate cash flow metrics"""
        try:
            free_cash_flow = self.info.get('freeCashflow', 0)
            operating_cash_flow = self.info.get('operatingCashflow', 0)
            
            return {
                'free_cash_flow': free_cash_flow,
                'operating_cash_flow': operating_cash_flow,
                'fcf_trend': 'Positive' if free_cash_flow > 0 else 'Negative',
                'fcf_margin_pct': round((free_cash_flow / self.info.get('totalRevenue', 1)) * 100, 2) if self.info.get('totalRevenue') else None
            }
        except Exception as e:
            logger.warning(f"Could not calculate cash flow: {e}")
            return {'error': str(e)}
    
    def _get_valuation_metrics(self) -> Dict[str, Any]:
        """Calculate valuation metrics"""
        try:
            return {
                'pe_ratio': round(self.info.get('trailingPE', 0), 2),
                'forward_pe': round(self.info.get('forwardPE', 0), 2),
                'peg_ratio': round(self.info.get('pegRatio', 0), 2),
                'price_to_book': round(self.info.get('priceToBook', 0), 2),
                'price_to_sales': round(self.info.get('priceToSalesTrailing12Months', 0), 2),
                'ev_to_ebitda': round(self.info.get('enterpriseToEbitda', 0), 2),
                'valuation_verdict': self._valuation_verdict()
            }
        except Exception as e:
            logger.warning(f"Could not calculate valuation: {e}")
            return {'error': str(e)}
    
    def _valuation_verdict(self) -> str:
        """Determine if stock is cheap, fair, or expensive"""
        pe = self.info.get('trailingPE', 0)
        peg = self.info.get('pegRatio', 0)
        
        if pe == 0 or pe > 100:
            return 'Unknown - No earnings or extreme P/E'
        elif peg < 1 and pe < 15:
            return 'Undervalued - Strong buy opportunity'
        elif peg < 1.5 and pe < 25:
            return 'Fair Value - Reasonably priced'
        elif peg < 2 and pe < 35:
            return 'Slight Premium - Growth priced in'
        else:
            return 'Overvalued - Expensive relative to growth'
    
    def _score_fundamentals(self, fundamentals: Dict[str, Any]) -> int:
        """Score fundamentals 0-100"""
        score = 0
        
        # Revenue growth (20 points)
        rev_growth = fundamentals['revenue_growth'].get('yoy_growth_pct', 0)
        if rev_growth > 20:
            score += 20
        elif rev_growth > 10:
            score += 15
        elif rev_growth > 5:
            score += 10
        elif rev_growth > 0:
            score += 5
        
        # Earnings growth (20 points)
        earn_growth = fundamentals['earnings_growth'].get('earnings_growth_pct', 0)
        if earn_growth and earn_growth > 20:
            score += 20
        elif earn_growth and earn_growth > 10:
            score += 15
        elif earn_growth and earn_growth > 5:
            score += 10
        
        # Margins (20 points)
        profit_margin = fundamentals['margins'].get('profit_margin_pct', 0)
        if profit_margin > 20:
            score += 20
        elif profit_margin > 10:
            score += 15
        elif profit_margin > 5:
            score += 10
        
        # Debt health (20 points)
        debt_health = fundamentals['debt_metrics'].get('debt_health', '')
        if 'Excellent' in debt_health:
            score += 20
        elif 'Good' in debt_health:
            score += 15
        elif 'Moderate' in debt_health:
            score += 10
        
        # Valuation (20 points)
        valuation = fundamentals['valuation'].get('valuation_verdict', '')
        if 'Undervalued' in valuation:
            score += 20
        elif 'Fair Value' in valuation:
            score += 15
        elif 'Premium' in valuation:
            score += 10
        
        return score
    
    def _fundamentals_verdict(self, score: int) -> str:
        """Generate verdict based on score"""
        if score >= 80:
            return 'STRONG - Financially robust, well-positioned'
        elif score >= 60:
            return 'GOOD - Solid fundamentals, some areas of concern'
        elif score >= 40:
            return 'MIXED - Strengths and weaknesses balanced'
        else:
            return 'WEAK - Fundamental concerns, high risk'
    
    # ==========================================
    # 2. INDUSTRY MOMENTUM (The Macro Winds)
    # ==========================================
    
    def _evaluate_industry_momentum(self) -> Dict[str, Any]:
        """
        Evaluate industry and sector dynamics:
        - Sector growth
        - Tailwinds/Headwinds
        - Peer comparison
        """
        logger.info("🌊 Analyzing industry momentum...")
        
        industry_momentum = {
            'sector': self.info.get('sector', 'Unknown'),
            'industry': self.info.get('industry', 'Unknown'),
            'sector_performance': self._get_sector_performance(),
            'relative_strength': self._get_relative_strength(),
            'market_cap_position': self._get_market_cap_position(),
            'score': 0,
            'verdict': ''
        }
        
        industry_momentum['score'] = self._score_industry_momentum(industry_momentum)
        industry_momentum['verdict'] = self._industry_verdict(industry_momentum['score'])
        
        return industry_momentum
    
    def _get_sector_performance(self) -> Dict[str, Any]:
        """Get sector performance metrics"""
        try:
            sector = self.info.get('sector', 'Unknown')
            
            # Map sector to ETF ticker
            sector_etfs = {
                'Technology': 'XLK',
                'Healthcare': 'XLV',
                'Financials': 'XLF',
                'Consumer Cyclical': 'XLY',
                'Consumer Defensive': 'XLP',
                'Energy': 'XLE',
                'Utilities': 'XLU',
                'Real Estate': 'XLRE',
                'Materials': 'XLB',
                'Industrials': 'XLI',
                'Communication Services': 'XLC'
            }
            
            etf_ticker = sector_etfs.get(sector, 'SPY')
            
            # Get sector ETF performance
            sector_etf = yf.Ticker(etf_ticker)
            sector_hist = sector_etf.history(period='1y')
            
            if not sector_hist.empty:
                sector_ytd_return = ((sector_hist['Close'].iloc[-1] / sector_hist['Close'].iloc[0]) - 1) * 100
            else:
                sector_ytd_return = 0
            
            return {
                'sector': sector,
                'sector_etf': etf_ticker,
                'sector_ytd_return_pct': round(sector_ytd_return, 2),
                'sector_trend': 'Rising' if sector_ytd_return > 0 else 'Falling'
            }
            
        except Exception as e:
            logger.warning(f"Could not get sector performance: {e}")
            return {'error': str(e)}
    
    def _get_relative_strength(self) -> Dict[str, Any]:
        """Calculate relative strength vs S&P 500"""
        try:
            # Get SPY performance
            spy = yf.Ticker('SPY')
            spy_hist = spy.history(period='1y')
            
            if self.history.empty or spy_hist.empty:
                return {'error': 'Insufficient data'}
            
            # Calculate returns
            stock_return = ((self.history['Close'].iloc[-1] / self.history['Close'].iloc[0]) - 1) * 100
            spy_return = ((spy_hist['Close'].iloc[-1] / spy_hist['Close'].iloc[0]) - 1) * 100
            
            relative_strength = stock_return - spy_return
            
            return {
                'stock_1y_return_pct': round(stock_return, 2),
                'spy_1y_return_pct': round(spy_return, 2),
                'relative_strength_pct': round(relative_strength, 2),
                'outperforming': relative_strength > 0
            }
            
        except Exception as e:
            logger.warning(f"Could not calculate relative strength: {e}")
            return {'error': str(e)}
    
    def _get_market_cap_position(self) -> Dict[str, Any]:
        """Determine market cap position"""
        try:
            market_cap = self.info.get('marketCap', 0)
            
            if market_cap > 200e9:
                category = 'Mega Cap (>$200B)'
            elif market_cap > 10e9:
                category = 'Large Cap ($10B-$200B)'
            elif market_cap > 2e9:
                category = 'Mid Cap ($2B-$10B)'
            elif market_cap > 300e6:
                category = 'Small Cap ($300M-$2B)'
            else:
                category = 'Micro Cap (<$300M)'
            
            return {
                'market_cap': market_cap,
                'category': category
            }
            
        except Exception as e:
            logger.warning(f"Could not get market cap: {e}")
            return {'error': str(e)}
    
    def _score_industry_momentum(self, momentum: Dict[str, Any]) -> int:
        """Score industry momentum 0-100"""
        score = 0
        
        # Sector performance (40 points)
        sector_return = momentum['sector_performance'].get('sector_ytd_return_pct', 0)
        if sector_return > 20:
            score += 40
        elif sector_return > 10:
            score += 30
        elif sector_return > 0:
            score += 20
        
        # Relative strength (60 points)
        rel_strength = momentum['relative_strength'].get('relative_strength_pct', 0)
        if rel_strength > 20:
            score += 60
        elif rel_strength > 10:
            score += 45
        elif rel_strength > 0:
            score += 30
        elif rel_strength > -10:
            score += 15
        
        return score
    
    def _industry_verdict(self, score: int) -> str:
        """Generate industry verdict"""
        if score >= 80:
            return 'TAILWINDS - Sector momentum strong, rising tide'
        elif score >= 60:
            return 'POSITIVE - Sector doing well, good environment'
        elif score >= 40:
            return 'NEUTRAL - Mixed sector signals'
        else:
            return 'HEADWINDS - Sector weakness, fighting the tide'
    
    # ==========================================
    # 3. COMPANY LEADERSHIP (The Competitive Edge)
    # ==========================================
    
    def _evaluate_company_leadership(self) -> Dict[str, Any]:
        """
        Evaluate competitive position:
        - Market share
        - Moat strength
        - Execution track record
        - Innovation
        """
        logger.info("👑 Analyzing company leadership...")
        
        leadership = {
            'market_position': self._get_market_position(),
            'competitive_moat': self._assess_competitive_moat(),
            'execution_track_record': self._assess_execution(),
            'growth_initiatives': self._assess_growth_initiatives(),
            'score': 0,
            'verdict': ''
        }
        
        leadership['score'] = self._score_leadership(leadership)
        leadership['verdict'] = self._leadership_verdict(leadership['score'])
        
        return leadership
    
    def _get_market_position(self) -> Dict[str, Any]:
        """Assess market position"""
        try:
            market_cap = self.info.get('marketCap', 0)
            industry = self.info.get('industry', 'Unknown')
            
            # Determine position based on market cap within industry
            if market_cap > 100e9:
                position = 'Industry Leader'
            elif market_cap > 10e9:
                position = 'Major Player'
            elif market_cap > 1e9:
                position = 'Emerging Competitor'
            else:
                position = 'Small Player'
            
            return {
                'industry': industry,
                'market_cap': market_cap,
                'position': position,
                'revenue_rank': self._estimate_revenue_rank()
            }
            
        except Exception as e:
            logger.warning(f"Could not assess market position: {e}")
            return {'error': str(e)}
    
    def _estimate_revenue_rank(self) -> str:
        """Estimate revenue ranking"""
        revenue_growth = self.info.get('revenueGrowth', 0)
        
        if revenue_growth > 0.3:
            return 'Fast Grower - Gaining Share'
        elif revenue_growth > 0.1:
            return 'Steady Grower - Maintaining Share'
        elif revenue_growth > 0:
            return 'Slow Grower - At Risk'
        else:
            return 'Declining - Losing Share'
    
    def _assess_competitive_moat(self) -> Dict[str, Any]:
        """Assess competitive advantages"""
        try:
            # Indicators of moat strength
            gross_margin = self.info.get('grossMargins', 0)
            return_on_equity = self.info.get('returnOnEquity', 0)
            return_on_assets = self.info.get('returnOnAssets', 0)
            
            # High margins + high ROE = strong moat
            moat_score = 0
            if gross_margin > 0.5:
                moat_score += 30
            elif gross_margin > 0.3:
                moat_score += 20
            
            if return_on_equity and return_on_equity > 0.2:
                moat_score += 40
            elif return_on_equity and return_on_equity > 0.15:
                moat_score += 25
            
            if return_on_assets and return_on_assets > 0.1:
                moat_score += 30
            elif return_on_assets and return_on_assets > 0.05:
                moat_score += 15
            
            if moat_score >= 75:
                moat_strength = 'WIDE MOAT - Defensible advantages'
            elif moat_score >= 50:
                moat_strength = 'MODERATE MOAT - Some advantages'
            else:
                moat_strength = 'NARROW MOAT - Vulnerable to competition'
            
            return {
                'gross_margin_pct': round(gross_margin * 100, 2),
                'return_on_equity_pct': round(return_on_equity * 100, 2) if return_on_equity else None,
                'return_on_assets_pct': round(return_on_assets * 100, 2) if return_on_assets else None,
                'moat_score': moat_score,
                'moat_strength': moat_strength
            }
            
        except Exception as e:
            logger.warning(f"Could not assess moat: {e}")
            return {'error': str(e)}
    
    def _assess_execution(self) -> Dict[str, Any]:
        """Assess management execution"""
        try:
            # Track record indicators
            earnings_surprise = self.info.get('earningsSurprisePct')
            recommendation = self.info.get('recommendationKey', 'hold')
            target_price = self.info.get('targetMeanPrice', 0)
            current_price = self.info.get('currentPrice', 0)
            
            upside_potential = ((target_price - current_price) / current_price * 100) if current_price > 0 else 0
            
            return {
                'earnings_surprise_pct': round(earnings_surprise, 2) if earnings_surprise else None,
                'analyst_recommendation': recommendation,
                'target_price': target_price,
                'current_price': current_price,
                'upside_potential_pct': round(upside_potential, 2),
                'execution_verdict': 'Strong' if earnings_surprise and earnings_surprise > 0 else 'Mixed'
            }
            
        except Exception as e:
            logger.warning(f"Could not assess execution: {e}")
            return {'error': str(e)}
    
    def _assess_growth_initiatives(self) -> Dict[str, Any]:
        """Assess growth and innovation"""
        try:
            revenue_growth = self.info.get('revenueGrowth', 0)
            earnings_growth = self.info.get('earningsGrowth', 0)
            
            return {
                'revenue_growth_pct': round(revenue_growth * 100, 2) if revenue_growth else None,
                'earnings_growth_pct': round(earnings_growth * 100, 2) if earnings_growth else None,
                'growth_verdict': self._growth_verdict(revenue_growth, earnings_growth)
            }
            
        except Exception as e:
            logger.warning(f"Could not assess growth: {e}")
            return {'error': str(e)}
    
    def _growth_verdict(self, rev_growth: float, earn_growth: float) -> str:
        """Determine growth verdict"""
        if rev_growth and earn_growth:
            avg_growth = (rev_growth + earn_growth) / 2
            if avg_growth > 0.2:
                return 'HIGH GROWTH - Leading innovation'
            elif avg_growth > 0.1:
                return 'MODERATE GROWTH - Steady expansion'
            elif avg_growth > 0:
                return 'SLOW GROWTH - Mature business'
            else:
                return 'DECLINING - Innovation needed'
        return 'Unknown'
    
    def _score_leadership(self, leadership: Dict[str, Any]) -> int:
        """Score leadership 0-100"""
        score = 0
        
        # Market position (25 points)
        position = leadership['market_position'].get('position', '')
        if 'Leader' in position:
            score += 25
        elif 'Major' in position:
            score += 20
        elif 'Emerging' in position:
            score += 15
        
        # Competitive moat (35 points)
        moat_score = leadership['competitive_moat'].get('moat_score', 0)
        score += int(moat_score * 0.35)
        
        # Execution (20 points)
        earnings_surprise = leadership['execution_track_record'].get('earnings_surprise_pct', 0)
        if earnings_surprise and earnings_surprise > 5:
            score += 20
        elif earnings_surprise and earnings_surprise > 0:
            score += 15
        elif earnings_surprise and earnings_surprise > -5:
            score += 10
        
        # Growth (20 points)
        growth = leadership['growth_initiatives'].get('revenue_growth_pct', 0)
        if growth and growth > 20:
            score += 20
        elif growth and growth > 10:
            score += 15
        elif growth and growth > 0:
            score += 10
        
        return score
    
    def _leadership_verdict(self, score: int) -> str:
        """Generate leadership verdict"""
        if score >= 80:
            return 'PREDATOR - Market leader, strong execution'
        elif score >= 60:
            return 'CONTENDER - Competitive, well-positioned'
        elif score >= 40:
            return 'SURVIVOR - Holding ground, needs improvement'
        else:
            return 'PREY - Weak position, at risk'
    
    # ==========================================
    # 4. TECHNICAL SENTIMENT (Overbought vs Oversold)
    # ==========================================
    
    def _evaluate_technical_sentiment(self) -> Dict[str, Any]:
        """
        Evaluate technical indicators:
        - RSI
        - MACD
        - Moving Averages
        - Volume Analysis
        """
        logger.info("📈 Analyzing technical sentiment...")
        
        if self.history.empty:
            return {'error': 'No price history available'}
        
        technical = {
            'rsi': self._calculate_rsi(),
            'macd': self._calculate_macd(),
            'moving_averages': self._analyze_moving_averages(),
            'volume_analysis': self._analyze_volume(),
            'price_momentum': self._analyze_price_momentum(),
            'score': 0,
            'verdict': ''
        }
        
        technical['score'] = self._score_technical(technical)
        technical['verdict'] = self._technical_verdict(technical['score'])
        
        return technical
    
    def _calculate_rsi(self, period: int = 14) -> Dict[str, Any]:
        """Calculate RSI"""
        try:
            closes = self.history['Close']
            delta = closes.diff()
            
            gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
            
            rs = gain / loss
            rsi = 100 - (100 / (1 + rs))
            
            current_rsi = float(rsi.iloc[-1])
            
            if current_rsi > 70:
                signal = 'OVERBOUGHT - Risk of pullback'
            elif current_rsi < 30:
                signal = 'OVERSOLD - Potential bounce'
            elif current_rsi > 60:
                signal = 'STRONG - Bullish momentum'
            elif current_rsi < 40:
                signal = 'WEAK - Bearish pressure'
            else:
                signal = 'NEUTRAL - No extreme'
            
            return {
                'current': round(current_rsi, 2),
                'signal': signal,
                'is_overbought': current_rsi > 70,
                'is_oversold': current_rsi < 30
            }
            
        except Exception as e:
            logger.warning(f"Could not calculate RSI: {e}")
            return {'error': str(e)}
    
    def _calculate_macd(self) -> Dict[str, Any]:
        """Calculate MACD"""
        try:
            closes = self.history['Close']
            
            ema_12 = closes.ewm(span=12, adjust=False).mean()
            ema_26 = closes.ewm(span=26, adjust=False).mean()
            
            macd_line = ema_12 - ema_26
            signal_line = macd_line.ewm(span=9, adjust=False).mean()
            histogram = macd_line - signal_line
            
            current_macd = float(macd_line.iloc[-1])
            current_signal = float(signal_line.iloc[-1])
            current_histogram = float(histogram.iloc[-1])
            
            if current_histogram > 0 and histogram.iloc[-2] <= 0:
                signal = 'BULLISH CROSSOVER - Buy signal'
            elif current_histogram < 0 and histogram.iloc[-2] >= 0:
                signal = 'BEARISH CROSSOVER - Sell signal'
            elif current_histogram > 0:
                signal = 'BULLISH - Positive momentum'
            else:
                signal = 'BEARISH - Negative momentum'
            
            return {
                'macd': round(current_macd, 2),
                'signal': round(current_signal, 2),
                'histogram': round(current_histogram, 2),
                'trend_signal': signal
            }
            
        except Exception as e:
            logger.warning(f"Could not calculate MACD: {e}")
            return {'error': str(e)}
    
    def _analyze_moving_averages(self) -> Dict[str, Any]:
        """Analyze moving averages"""
        try:
            closes = self.history['Close']
            current_price = float(closes.iloc[-1])
            
            sma_20 = float(closes.rolling(20).mean().iloc[-1])
            sma_50 = float(closes.rolling(50).mean().iloc[-1])
            sma_200 = float(closes.rolling(200).mean().iloc[-1]) if len(closes) >= 200 else None
            
            # Determine trend
            if sma_200 and current_price > sma_20 > sma_50 > sma_200:
                trend = 'STRONG UPTREND - All systems go'
            elif current_price > sma_20 > sma_50:
                trend = 'UPTREND - Bullish structure'
            elif sma_200 and current_price < sma_20 < sma_50 < sma_200:
                trend = 'STRONG DOWNTREND - Stay away'
            elif current_price < sma_20 < sma_50:
                trend = 'DOWNTREND - Bearish structure'
            else:
                trend = 'MIXED - Unclear direction'
            
            return {
                'current_price': round(current_price, 2),
                'sma_20': round(sma_20, 2),
                'sma_50': round(sma_50, 2),
                'sma_200': round(sma_200, 2) if sma_200 else None,
                'trend': trend,
                'above_sma_20': current_price > sma_20,
                'above_sma_50': current_price > sma_50,
                'above_sma_200': current_price > sma_200 if sma_200 else None
            }
            
        except Exception as e:
            logger.warning(f"Could not analyze moving averages: {e}")
            return {'error': str(e)}
    
    def _analyze_volume(self) -> Dict[str, Any]:
        """Analyze volume trends"""
        try:
            volumes = self.history['Volume']
            closes = self.history['Close']
            
            avg_volume = float(volumes.rolling(50).mean().iloc[-1])
            current_volume = float(volumes.iloc[-1])
            
            # Price and volume relationship
            price_change = ((closes.iloc[-1] / closes.iloc[-2]) - 1) * 100
            volume_ratio = current_volume / avg_volume
            
            if price_change > 0 and volume_ratio > 1.5:
                signal = 'BULLISH CONVICTION - Rising price on high volume'
            elif price_change > 0 and volume_ratio < 0.7:
                signal = 'WEAK RALLY - Rising price on low volume'
            elif price_change < 0 and volume_ratio > 1.5:
                signal = 'BEARISH CONVICTION - Falling price on high volume'
            elif price_change < 0 and volume_ratio < 0.7:
                signal = 'WEAK SELLOFF - Low volume decline'
            else:
                signal = 'NEUTRAL - Normal activity'
            
            return {
                'current_volume': int(current_volume),
                'avg_volume_50d': int(avg_volume),
                'volume_ratio': round(volume_ratio, 2),
                'signal': signal
            }
            
        except Exception as e:
            logger.warning(f"Could not analyze volume: {e}")
            return {'error': str(e)}
    
    def _analyze_price_momentum(self) -> Dict[str, Any]:
        """Analyze price momentum"""
        try:
            closes = self.history['Close']
            
            # Calculate returns over different periods
            returns_1w = ((closes.iloc[-1] / closes.iloc[-5]) - 1) * 100 if len(closes) >= 5 else 0
            returns_1m = ((closes.iloc[-1] / closes.iloc[-20]) - 1) * 100 if len(closes) >= 20 else 0
            returns_3m = ((closes.iloc[-1] / closes.iloc[-60]) - 1) * 100 if len(closes) >= 60 else 0
            returns_1y = ((closes.iloc[-1] / closes.iloc[-252]) - 1) * 100 if len(closes) >= 252 else 0
            
            return {
                'returns_1w_pct': round(returns_1w, 2),
                'returns_1m_pct': round(returns_1m, 2),
                'returns_3m_pct': round(returns_3m, 2),
                'returns_1y_pct': round(returns_1y, 2),
                'momentum_verdict': self._momentum_verdict(returns_1m, returns_3m)
            }
            
        except Exception as e:
            logger.warning(f"Could not analyze momentum: {e}")
            return {'error': str(e)}
    
    def _momentum_verdict(self, returns_1m: float, returns_3m: float) -> str:
        """Determine momentum verdict"""
        if returns_1m > 10 and returns_3m > 20:
            return 'HOT MOMENTUM - Strong buying'
        elif returns_1m > 5 and returns_3m > 10:
            return 'POSITIVE MOMENTUM - Trending up'
        elif returns_1m > 0 and returns_3m > 0:
            return 'MILD MOMENTUM - Slow grind higher'
        elif returns_1m < -10 and returns_3m < -20:
            return 'COLD MOMENTUM - Heavy selling'
        elif returns_1m < 0 and returns_3m < 0:
            return 'NEGATIVE MOMENTUM - Trending down'
        else:
            return 'MIXED MOMENTUM - Choppy action'
    
    def _score_technical(self, technical: Dict[str, Any]) -> int:
        """Score technical indicators 0-100"""
        score = 0
        
        # RSI (20 points)
        rsi = technical['rsi'].get('current', 50)
        if 40 <= rsi <= 60:
            score += 20  # Neutral zone
        elif 30 <= rsi < 40:
            score += 15  # Slightly oversold (buying opportunity)
        elif 60 < rsi <= 70:
            score += 15  # Slightly overbought (caution)
        elif rsi < 30:
            score += 10  # Oversold (contrarian buy)
        elif rsi > 70:
            score += 5   # Overbought (risk)
        
        # MACD (20 points)
        histogram = technical['macd'].get('histogram', 0)
        if histogram > 0:
            score += 20
        elif histogram > -1:
            score += 10
        
        # Moving Averages (30 points)
        ma = technical['moving_averages']
        if ma.get('above_sma_20') and ma.get('above_sma_50'):
            score += 30
        elif ma.get('above_sma_20'):
            score += 20
        elif ma.get('above_sma_50'):
            score += 15
        
        # Volume (15 points)
        volume_signal = technical['volume_analysis'].get('signal', '')
        if 'BULLISH CONVICTION' in volume_signal:
            score += 15
        elif 'NEUTRAL' in volume_signal:
            score += 10
        
        # Momentum (15 points)
        returns_3m = technical['price_momentum'].get('returns_3m_pct', 0)
        if returns_3m > 20:
            score += 15
        elif returns_3m > 10:
            score += 12
        elif returns_3m > 0:
            score += 8
        
        return score
    
    def _technical_verdict(self, score: int) -> str:
        """Generate technical verdict"""
        if score >= 80:
            return 'STRONG BUY - Technical breakout confirmed'
        elif score >= 60:
            return 'BUY - Positive technical setup'
        elif score >= 40:
            return 'HOLD - Mixed technical signals'
        elif score >= 20:
            return 'SELL - Weak technical setup'
        else:
            return 'STRONG SELL - Technical breakdown'
    
    # ==========================================
    # 5. SUPPORT & RESISTANCE (The Battlegrounds)
    # ==========================================
    
    def _evaluate_support_resistance(self) -> Dict[str, Any]:
        """
        Evaluate support and resistance levels:
        - Key support/resistance
        - Moving average levels
        - Breakout/breakdown analysis
        """
        logger.info("🎯 Analyzing support & resistance...")
        
        if self.history.empty:
            return {'error': 'No price history available'}
        
        support_resistance = {
            'current_price': float(self.history['Close'].iloc[-1]),
            'support_levels': self._find_support_levels(),
            'resistance_levels': self._find_resistance_levels(),
            'ma_levels': self._get_ma_support_resistance(),
            'nearest_support': None,
            'nearest_resistance': None,
            'trading_range': None,
            'breakout_analysis': self._analyze_breakout_potential(),
            'verdict': ''
        }
        
        # Identify nearest levels
        support_resistance['nearest_support'] = self._get_nearest_support(
            support_resistance['current_price'],
            support_resistance['support_levels']
        )
        support_resistance['nearest_resistance'] = self._get_nearest_resistance(
            support_resistance['current_price'],
            support_resistance['resistance_levels']
        )
        
        # Calculate trading range
        if support_resistance['nearest_support'] and support_resistance['nearest_resistance']:
            support_resistance['trading_range'] = {
                'lower': support_resistance['nearest_support'],
                'upper': support_resistance['nearest_resistance'],
                'range_pct': round(
                    ((support_resistance['nearest_resistance'] - support_resistance['nearest_support']) / 
                     support_resistance['nearest_support']) * 100, 2
                )
            }
        
        support_resistance['verdict'] = self._support_resistance_verdict(support_resistance)
        
        return support_resistance
    
    def _find_support_levels(self) -> List[float]:
        """Find key support levels"""
        try:
            closes = self.history['Close']
            lows = self.history['Low']
            
            # Recent lows (last 60 days)
            recent_lows = lows.rolling(20).min().dropna()
            
            # Find local minima
            support_levels = []
            for i in range(2, len(recent_lows) - 2):
                if (recent_lows.iloc[i] < recent_lows.iloc[i-1] and 
                    recent_lows.iloc[i] < recent_lows.iloc[i-2] and
                    recent_lows.iloc[i] < recent_lows.iloc[i+1] and
                    recent_lows.iloc[i] < recent_lows.iloc[i+2]):
                    support_levels.append(float(recent_lows.iloc[i]))
            
            # Add 52-week low
            low_52w = float(lows.min())
            if low_52w not in support_levels:
                support_levels.append(low_52w)
            
            # Sort and deduplicate (within 2% threshold)
            support_levels = sorted(set(support_levels), reverse=True)
            filtered_levels = []
            for level in support_levels:
                if not filtered_levels or abs(level - filtered_levels[-1]) / filtered_levels[-1] > 0.02:
                    filtered_levels.append(level)
            
            return filtered_levels[:5]  # Top 5 support levels
            
        except Exception as e:
            logger.warning(f"Could not find support levels: {e}")
            return []
    
    def _find_resistance_levels(self) -> List[float]:
        """Find key resistance levels"""
        try:
            closes = self.history['Close']
            highs = self.history['High']
            
            # Recent highs (last 60 days)
            recent_highs = highs.rolling(20).max().dropna()
            
            # Find local maxima
            resistance_levels = []
            for i in range(2, len(recent_highs) - 2):
                if (recent_highs.iloc[i] > recent_highs.iloc[i-1] and 
                    recent_highs.iloc[i] > recent_highs.iloc[i-2] and
                    recent_highs.iloc[i] > recent_highs.iloc[i+1] and
                    recent_highs.iloc[i] > recent_highs.iloc[i+2]):
                    resistance_levels.append(float(recent_highs.iloc[i]))
            
            # Add 52-week high
            high_52w = float(highs.max())
            if high_52w not in resistance_levels:
                resistance_levels.append(high_52w)
            
            # Sort and deduplicate (within 2% threshold)
            resistance_levels = sorted(set(resistance_levels))
            filtered_levels = []
            for level in resistance_levels:
                if not filtered_levels or abs(level - filtered_levels[-1]) / filtered_levels[-1] > 0.02:
                    filtered_levels.append(level)
            
            return filtered_levels[:5]  # Top 5 resistance levels
            
        except Exception as e:
            logger.warning(f"Could not find resistance levels: {e}")
            return []
    
    def _get_ma_support_resistance(self) -> Dict[str, float]:
        """Get moving averages as support/resistance"""
        try:
            closes = self.history['Close']
            
            return {
                'sma_20': round(float(closes.rolling(20).mean().iloc[-1]), 2),
                'sma_50': round(float(closes.rolling(50).mean().iloc[-1]), 2),
                'sma_200': round(float(closes.rolling(200).mean().iloc[-1]), 2) if len(closes) >= 200 else None
            }
            
        except Exception as e:
            logger.warning(f"Could not get MA levels: {e}")
            return {}
    
    def _get_nearest_support(self, current_price: float, support_levels: List[float]) -> Optional[float]:
        """Find nearest support below current price"""
        below_supports = [s for s in support_levels if s < current_price]
        return max(below_supports) if below_supports else None
    
    def _get_nearest_resistance(self, current_price: float, resistance_levels: List[float]) -> Optional[float]:
        """Find nearest resistance above current price"""
        above_resistances = [r for r in resistance_levels if r > current_price]
        return min(above_resistances) if above_resistances else None
    
    def _analyze_breakout_potential(self) -> Dict[str, Any]:
        """Analyze breakout/breakdown potential"""
        try:
            closes = self.history['Close']
            volumes = self.history['Volume']
            
            current_price = float(closes.iloc[-1])
            high_52w = float(self.history['High'].max())
            low_52w = float(self.history['Low'].min())
            
            # Check if near 52-week high/low
            near_high = (current_price / high_52w) > 0.98
            near_low = (current_price / low_52w) < 1.02
            
            # Volume trend
            avg_volume = float(volumes.rolling(20).mean().iloc[-1])
            current_volume = float(volumes.iloc[-1])
            volume_surge = current_volume > (avg_volume * 1.5)
            
            if near_high and volume_surge:
                potential = 'HIGH BREAKOUT POTENTIAL - At resistance with volume'
            elif near_high:
                potential = 'NEAR RESISTANCE - Watch for breakout or rejection'
            elif near_low and volume_surge:
                potential = 'HIGH BREAKDOWN RISK - At support with volume'
            elif near_low:
                potential = 'NEAR SUPPORT - Watch for bounce or breakdown'
            else:
                potential = 'MID-RANGE - No immediate breakout/breakdown'
            
            return {
                'high_52w': round(high_52w, 2),
                'low_52w': round(low_52w, 2),
                'distance_from_high_pct': round(((current_price / high_52w) - 1) * 100, 2),
                'distance_from_low_pct': round(((current_price / low_52w) - 1) * 100, 2),
                'volume_surge': volume_surge,
                'potential': potential
            }
            
        except Exception as e:
            logger.warning(f"Could not analyze breakout: {e}")
            return {}
    
    def _support_resistance_verdict(self, sr: Dict[str, Any]) -> str:
        """Generate support/resistance verdict"""
        current = sr['current_price']
        nearest_support = sr['nearest_support']
        nearest_resistance = sr['nearest_resistance']
        
        if nearest_support and nearest_resistance:
            distance_to_support = ((current - nearest_support) / current) * 100
            distance_to_resistance = ((nearest_resistance - current) / current) * 100
            
            if distance_to_support < 2:
                return f'AT SUPPORT (${nearest_support:.2f}) - Buy the dip zone'
            elif distance_to_resistance < 2:
                return f'AT RESISTANCE (${nearest_resistance:.2f}) - Sell the rip zone'
            elif distance_to_support < distance_to_resistance:
                return f'LOWER RANGE - Closer to support ${nearest_support:.2f}'
            else:
                return f'UPPER RANGE - Closer to resistance ${nearest_resistance:.2f}'
        
        return 'UNKNOWN - Insufficient level data'
    
    # ==========================================
    # OVERALL SCORING & RECOMMENDATION
    # ==========================================
    
    def _calculate_overall_score(self, evaluation: Dict[str, Any]) -> int:
        """Calculate weighted overall score"""
        weights = {
            'fundamentals': 0.30,
            'industry_momentum': 0.15,
            'company_leadership': 0.25,
            'technical_sentiment': 0.30
        }
        
        total_score = 0
        for category, weight in weights.items():
            category_score = evaluation[category].get('score', 0)
            total_score += category_score * weight
        
        return int(total_score)
    
    def _generate_recommendation(self, evaluation: Dict[str, Any]) -> Dict[str, Any]:
        """Generate final investment recommendation"""
        overall_score = evaluation['overall_score']
        
        # Determine action
        if overall_score >= 80:
            action = 'STRONG BUY'
            confidence = 'HIGH'
            reasoning = 'Strong fundamentals + positive technicals + competitive advantage'
        elif overall_score >= 65:
            action = 'BUY'
            confidence = 'MODERATE'
            reasoning = 'Solid fundamentals with decent technical setup'
        elif overall_score >= 50:
            action = 'HOLD'
            confidence = 'LOW'
            reasoning = 'Mixed signals - wait for better entry'
        elif overall_score >= 35:
            action = 'SELL'
            confidence = 'MODERATE'
            reasoning = 'Weak fundamentals or poor technicals'
        else:
            action = 'STRONG SELL'
            confidence = 'HIGH'
            reasoning = 'Multiple red flags across categories'
        
        # Entry/Exit suggestions
        sr = evaluation['support_resistance']
        entry_zone = sr.get('nearest_support', sr.get('current_price', 0))
        exit_zone = sr.get('nearest_resistance', sr.get('current_price', 0) * 1.1)
        
        return {
            'action': action,
            'confidence': confidence,
            'overall_score': overall_score,
            'reasoning': reasoning,
            'entry_zone': round(entry_zone, 2) if entry_zone else None,
            'exit_zone': round(exit_zone, 2) if exit_zone else None,
            'stop_loss': round(entry_zone * 0.95, 2) if entry_zone else None,
            'risk_reward_ratio': round((exit_zone - entry_zone) / (entry_zone * 0.05), 2) if entry_zone and exit_zone else None
        }


def evaluate_company(ticker: str, output_format: str = 'json') -> str:
    """
    Main API function to evaluate a company
    
    Args:
        ticker: Stock ticker symbol
        output_format: 'json' or 'markdown'
    
    Returns:
        Formatted evaluation report
    """
    evaluator = CompanyEvaluator(ticker)
    evaluation = evaluator.evaluate()
    
    if output_format == 'markdown':
        return format_as_markdown(evaluation)
    else:
        return json.dumps(evaluation, indent=2)


def format_as_markdown(evaluation: Dict[str, Any]) -> str:
    """Format evaluation as markdown report"""
    ticker = evaluation['ticker']
    timestamp = evaluation['timestamp']
    
    md = f"""# 📊 {ticker} - Comprehensive Company Evaluation

**Generated:** {timestamp}

---

## 🎯 OVERALL RECOMMENDATION

**Action:** {evaluation['recommendation']['action']}  
**Confidence:** {evaluation['recommendation']['confidence']}  
**Overall Score:** {evaluation['recommendation']['overall_score']}/100

**Reasoning:** {evaluation['recommendation']['reasoning']}

**Trade Setup:**
- **Entry Zone:** ${evaluation['recommendation']['entry_zone']}
- **Exit Zone:** ${evaluation['recommendation']['exit_zone']}
- **Stop Loss:** ${evaluation['recommendation']['stop_loss']}
- **Risk/Reward:** {evaluation['recommendation']['risk_reward_ratio']}:1

---

## 1️⃣ FUNDAMENTALS (The Balance Sheet and Income Engine)

**Score:** {evaluation['fundamentals']['score']}/100  
**Verdict:** {evaluation['fundamentals']['verdict']}

### Revenue & Earnings Growth
- **Revenue Growth YoY:** {evaluation['fundamentals']['revenue_growth'].get('yoy_growth_pct', 'N/A')}%
- **Revenue Trend:** {evaluation['fundamentals']['revenue_growth'].get('trend', 'N/A')}
- **Earnings Growth:** {evaluation['fundamentals']['earnings_growth'].get('earnings_growth_pct', 'N/A')}%
- **EPS Trend:** {evaluation['fundamentals']['earnings_growth'].get('trend', 'N/A')}

### Margins
- **Gross Margin:** {evaluation['fundamentals']['margins'].get('gross_margin_pct', 'N/A')}%
- **Operating Margin:** {evaluation['fundamentals']['margins'].get('operating_margin_pct', 'N/A')}%
- **Profit Margin:** {evaluation['fundamentals']['margins'].get('profit_margin_pct', 'N/A')}%

### Debt & Cash Flow
- **Debt/Equity:** {evaluation['fundamentals']['debt_metrics'].get('debt_to_equity', 'N/A')}
- **Debt Health:** {evaluation['fundamentals']['debt_metrics'].get('debt_health', 'N/A')}
- **Free Cash Flow:** ${evaluation['fundamentals']['cash_flow'].get('free_cash_flow', 0):,.0f}
- **FCF Trend:** {evaluation['fundamentals']['cash_flow'].get('fcf_trend', 'N/A')}

### Valuation
- **P/E Ratio:** {evaluation['fundamentals']['valuation'].get('pe_ratio', 'N/A')}
- **PEG Ratio:** {evaluation['fundamentals']['valuation'].get('peg_ratio', 'N/A')}
- **P/B Ratio:** {evaluation['fundamentals']['valuation'].get('price_to_book', 'N/A')}
- **Verdict:** {evaluation['fundamentals']['valuation'].get('valuation_verdict', 'N/A')}

**💡 Trader's Question:** *Is this company financially strong, or is it just dressed up for Wall Street?*  
**Answer:** {evaluation['fundamentals']['verdict']}

---

## 2️⃣ INDUSTRY MOMENTUM (The Macro Winds)

**Score:** {evaluation['industry_momentum']['score']}/100  
**Verdict:** {evaluation['industry_momentum']['verdict']}

### Sector & Position
- **Sector:** {evaluation['industry_momentum']['sector']}
- **Industry:** {evaluation['industry_momentum']['industry']}
- **Sector YTD Return:** {evaluation['industry_momentum']['sector_performance'].get('sector_ytd_return_pct', 'N/A')}%

### Relative Strength
- **Stock 1Y Return:** {evaluation['industry_momentum']['relative_strength'].get('stock_1y_return_pct', 'N/A')}%
- **S&P 500 1Y Return:** {evaluation['industry_momentum']['relative_strength'].get('spy_1y_return_pct', 'N/A')}%
- **Relative Strength:** {evaluation['industry_momentum']['relative_strength'].get('relative_strength_pct', 'N/A')}%
- **Outperforming:** {evaluation['industry_momentum']['relative_strength'].get('outperforming', 'N/A')}

**💡 Trader's Question:** *Is the tide lifting all boats, or is this sector a dead weight?*  
**Answer:** {evaluation['industry_momentum']['verdict']}

---

## 3️⃣ COMPANY LEADERSHIP (The Competitive Edge)

**Score:** {evaluation['company_leadership']['score']}/100  
**Verdict:** {evaluation['company_leadership']['verdict']}

### Market Position
- **Position:** {evaluation['company_leadership']['market_position'].get('position', 'N/A')}
- **Market Cap:** ${evaluation['company_leadership']['market_position'].get('market_cap', 0):,.0f}
- **Revenue Rank:** {evaluation['company_leadership']['market_position'].get('revenue_rank', 'N/A')}

### Competitive Moat
- **Moat Strength:** {evaluation['company_leadership']['competitive_moat'].get('moat_strength', 'N/A')}
- **ROE:** {evaluation['company_leadership']['competitive_moat'].get('return_on_equity_pct', 'N/A')}%
- **ROA:** {evaluation['company_leadership']['competitive_moat'].get('return_on_assets_pct', 'N/A')}%

### Execution & Growth
- **Execution Verdict:** {evaluation['company_leadership']['execution_track_record'].get('execution_verdict', 'N/A')}
- **Analyst Rating:** {evaluation['company_leadership']['execution_track_record'].get('analyst_recommendation', 'N/A')}
- **Upside Potential:** {evaluation['company_leadership']['execution_track_record'].get('upside_potential_pct', 'N/A')}%
- **Growth Verdict:** {evaluation['company_leadership']['growth_initiatives'].get('growth_verdict', 'N/A')}

**💡 Trader's Question:** *Is this company the predator or the prey?*  
**Answer:** {evaluation['company_leadership']['verdict']}

---

## 4️⃣ TECHNICAL SENTIMENT (Overbought vs Oversold)

**Score:** {evaluation['technical_sentiment']['score']}/100  
**Verdict:** {evaluation['technical_sentiment']['verdict']}

### RSI (Relative Strength Index)
- **Current RSI:** {evaluation['technical_sentiment']['rsi'].get('current', 'N/A')}
- **Signal:** {evaluation['technical_sentiment']['rsi'].get('signal', 'N/A')}

### MACD
- **MACD Line:** {evaluation['technical_sentiment']['macd'].get('macd', 'N/A')}
- **Signal Line:** {evaluation['technical_sentiment']['macd'].get('signal', 'N/A')}
- **Histogram:** {evaluation['technical_sentiment']['macd'].get('histogram', 'N/A')}
- **Trend Signal:** {evaluation['technical_sentiment']['macd'].get('trend_signal', 'N/A')}

### Moving Averages
- **Current Price:** ${evaluation['technical_sentiment']['moving_averages'].get('current_price', 'N/A')}
- **20-day SMA:** ${evaluation['technical_sentiment']['moving_averages'].get('sma_20', 'N/A')}
- **50-day SMA:** ${evaluation['technical_sentiment']['moving_averages'].get('sma_50', 'N/A')}
- **200-day SMA:** ${evaluation['technical_sentiment']['moving_averages'].get('sma_200', 'N/A')}
- **Trend:** {evaluation['technical_sentiment']['moving_averages'].get('trend', 'N/A')}

### Volume Analysis
- **Signal:** {evaluation['technical_sentiment']['volume_analysis'].get('signal', 'N/A')}
- **Volume Ratio:** {evaluation['technical_sentiment']['volume_analysis'].get('volume_ratio', 'N/A')}x

### Price Momentum
- **1 Week:** {evaluation['technical_sentiment']['price_momentum'].get('returns_1w_pct', 'N/A')}%
- **1 Month:** {evaluation['technical_sentiment']['price_momentum'].get('returns_1m_pct', 'N/A')}%
- **3 Months:** {evaluation['technical_sentiment']['price_momentum'].get('returns_3m_pct', 'N/A')}%
- **1 Year:** {evaluation['technical_sentiment']['price_momentum'].get('returns_1y_pct', 'N/A')}%
- **Verdict:** {evaluation['technical_sentiment']['price_momentum'].get('momentum_verdict', 'N/A')}

**💡 Trader's Question:** *Is this stock hot money FOMO, or cold hard value?*  
**Answer:** {evaluation['technical_sentiment']['verdict']}

---

## 5️⃣ SUPPORT & RESISTANCE (The Battlegrounds)

**Current Price:** ${evaluation['support_resistance']['current_price']}  
**Verdict:** {evaluation['support_resistance']['verdict']}

### Key Levels
- **Nearest Support:** ${evaluation['support_resistance']['nearest_support']}
- **Nearest Resistance:** ${evaluation['support_resistance']['nearest_resistance']}
- **Trading Range:** {evaluation['support_resistance'].get('trading_range', {}).get('range_pct', 'N/A')}%

### Support Levels
{chr(10).join([f'- ${level:.2f}' for level in evaluation['support_resistance']['support_levels'][:3]])}

### Resistance Levels
{chr(10).join([f'- ${level:.2f}' for level in evaluation['support_resistance']['resistance_levels'][:3]])}

### Breakout Analysis
- **52-Week High:** ${evaluation['support_resistance']['breakout_analysis'].get('high_52w', 'N/A')}
- **52-Week Low:** ${evaluation['support_resistance']['breakout_analysis'].get('low_52w', 'N/A')}
- **Distance from High:** {evaluation['support_resistance']['breakout_analysis'].get('distance_from_high_pct', 'N/A')}%
- **Distance from Low:** {evaluation['support_resistance']['breakout_analysis'].get('distance_from_low_pct', 'N/A')}%
- **Potential:** {evaluation['support_resistance']['breakout_analysis'].get('potential', 'N/A')}

**💡 Trader's Question:** *Where do I buy the dip, and where do I sell the rip?*  
**Answer:** {evaluation['support_resistance']['verdict']}

---

## 📈 FINAL VERDICT

{evaluation['recommendation']['action']} - {evaluation['recommendation']['reasoning']}

**This framework evaluated {ticker} like a hedge fund pro:**
1. ✅ Checked the numbers (fundamentals)
2. ✅ Read the macro (industry)
3. ✅ Judged the general (leadership)
4. ✅ Took the market's temperature (technicals)
5. ✅ Mapped the battlefield (support/resistance)

**Result:** You now know if {ticker} is worth owning, worth shorting, or worth ignoring — and exactly where to enter and exit.

---

*This analysis is for informational purposes only and should not be considered financial advice.*
"""
    
    return md


if __name__ == '__main__':
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python company_evaluation_api.py <TICKER> [format]")
        print("Example: python company_evaluation_api.py NVDA markdown")
        sys.exit(1)
    
    ticker = sys.argv[1]
    output_format = sys.argv[2] if len(sys.argv) > 2 else 'markdown'
    
    print(f"\n🔍 Evaluating {ticker}...\n")
    result = evaluate_company(ticker, output_format)
    print(result)
    
    # Save to file
    if output_format == 'markdown':
        filename = f"reports/{ticker}_evaluation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md"
        with open(filename, 'w') as f:
            f.write(result)
        print(f"\n📄 Report saved to: {filename}")
    else:
        filename = f"reports/{ticker}_evaluation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(filename, 'w') as f:
            f.write(result)
        print(f"\n📄 Report saved to: {filename}")

