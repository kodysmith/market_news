#!/usr/bin/env python3
"""
Daily Strategy Advisor - Ask QuantEngine for the best strategy or trade for the day

Usage:
    python daily_strategy_advisor.py
    python daily_strategy_advisor.py --asset NVDA
    python daily_strategy_advisor.py --sector technology
    python daily_strategy_advisor.py --risk high
"""

import os
import sys
import logging
import argparse
from pathlib import Path
from datetime import datetime, timedelta
import pandas as pd
import numpy as np

# Set production environment
os.environ['QUANT_ENV'] = 'production'

# Add QuantEngine to path
quant_engine_dir = Path(__file__).parent
sys.path.insert(0, str(quant_engine_dir))

from engine.data_ingestion.data_manager import DataManager
from engine.research.opportunity_scanner import OpportunityScanner
from engine.research.strategy_optimizer import AdaptiveStrategyResearcher
from engine.research.regime_detector import MarketRegimeDetector
from data_broker import QuantBotDataBroker
from llm_integration import OllamaLLM

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

class DailyStrategyAdvisor:
    """Daily strategy and trade recommendation system"""
    
    def __init__(self):
        self.data_manager = DataManager({})
        self.opportunity_scanner = OpportunityScanner({})
        self.strategy_researcher = AdaptiveStrategyResearcher({})
        self.regime_detector = MarketRegimeDetector({})
        self.data_broker = QuantBotDataBroker()
        
        # Initialize LLM
        try:
            self.llm = OllamaLLM(model="qwen2.5:72b")
            logger.info("✅ LLM integration initialized")
        except Exception as e:
            logger.warning(f"⚠️ LLM not available: {e}")
            self.llm = None
        
        logger.info("✅ Daily Strategy Advisor initialized")

    def get_daily_recommendations(self, asset: str = None, sector: str = None, risk_level: str = 'medium'):
        """Get daily strategy and trade recommendations"""
        logger.info("🎯 Generating daily strategy recommendations...")
        
        # Get current market data
        end_date = datetime.now()
        start_date = end_date - timedelta(days=30)  # Last 30 days for analysis
        
        # Determine what to analyze
        if asset:
            tickers = [asset.upper()]
            analysis_type = f"asset {asset}"
        elif sector:
            sector_tickers = self._get_sector_tickers(sector)
            tickers = sector_tickers
            analysis_type = f"{sector} sector"
        else:
            # Analyze major market indices
            tickers = ['SPY', 'QQQ', 'IWM', 'VTI']
            analysis_type = "broad market"
        
        logger.info(f"📊 Analyzing {analysis_type} for daily opportunities")
        
        # Get market data
        try:
            market_data = self.data_manager.get_market_data(tickers, start_date.strftime('%Y-%m-%d'), end_date.strftime('%Y-%m-%d'))
            logger.info(f"✅ Got market data for {len(market_data)} tickers")
        except Exception as e:
            logger.error(f"❌ Failed to get market data: {e}")
            market_data = {}
        
        if not market_data:
            logger.error("❌ No market data available")
            return None
        
        # Detect current market regime
        regime = self._detect_market_regime(market_data)
        logger.info(f"🌍 Current market regime: {regime}")
        
        # Find opportunities
        opportunities = self._find_daily_opportunities(market_data, risk_level)
        
        # Generate strategy recommendations
        strategies = self._generate_strategy_recommendations(market_data, regime, risk_level)
        
        # Add LLM analysis if available
        llm_insights = ""
        if self.llm and market_data:
            try:
                logger.info("🤖 Generating LLM insights...")
                # Analyze the first ticker with LLM
                first_ticker = list(market_data.keys())[0]
                
                # Add market context
                market_context = self._get_market_context(first_ticker, market_data[first_ticker])
                
                llm_analysis = self.llm.analyze_market_data(first_ticker, market_data[first_ticker])
                llm_insights = llm_analysis['llm_analysis']
                
                # Add market context to insights
                if market_context:
                    llm_insights = f"{market_context}\n\n{llm_insights}"
                
                logger.info("✅ LLM analysis completed")
            except Exception as e:
                logger.warning(f"⚠️ LLM analysis failed: {e}")
                llm_insights = ""
        
        # Create daily report
        report = self._create_daily_report(analysis_type, regime, opportunities, strategies, market_data, llm_insights)
        
        return report

    def _get_sector_tickers(self, sector: str) -> list:
        """Get representative tickers for a sector"""
        sector_map = {
            'technology': ['XLK', 'QQQ', 'VGT'],
            'healthcare': ['XLV', 'VHT', 'IBB'],
            'financial': ['XLF', 'VFH', 'KBE'],
            'energy': ['XLE', 'VDE', 'XOP'],
            'consumer': ['XLY', 'VCR', 'XRT'],
            'industrial': ['XLI', 'VIS', 'IYJ'],
            'utilities': ['XLU', 'VPU', 'IDU'],
            'materials': ['XLB', 'VAW', 'IYM'],
            'real_estate': ['VNQ', 'IYR', 'REZ'],
            'communication': ['XLC', 'VOX', 'IYZ']
        }
        return sector_map.get(sector.lower(), ['SPY'])

    def _detect_market_regime(self, market_data: dict) -> str:
        """Detect current market regime"""
        try:
            # Use SPY as market proxy
            if 'SPY' in market_data:
                spy_data = market_data['SPY']
                returns = spy_data['close'].pct_change().dropna()
                
                # Calculate regime indicators
                volatility = returns.rolling(20).std().iloc[-1] * np.sqrt(252)
                trend = (spy_data['close'].iloc[-1] / spy_data['close'].iloc[-20] - 1) * 100
                
                if volatility > 0.25:
                    if trend > 5:
                        return "high_vol_bull"
                    elif trend < -5:
                        return "high_vol_bear"
                    else:
                        return "high_vol_sideways"
                else:
                    if trend > 3:
                        return "low_vol_bull"
                    elif trend < -3:
                        return "low_vol_bear"
                    else:
                        return "low_vol_sideways"
            else:
                return "unknown"
        except Exception as e:
            logger.warning(f"Failed to detect regime: {e}")
            return "unknown"

    def _find_daily_opportunities(self, market_data: dict, risk_level: str) -> list:
        """Find daily trading opportunities"""
        opportunities = []
        
        for ticker, data in market_data.items():
            try:
                # Basic technical analysis
                current_price = float(data['close'].iloc[-1].iloc[0])
                sma_20 = float(data['close'].rolling(20).mean().iloc[-1].iloc[0])
                sma_50 = float(data['close'].rolling(50).mean().iloc[-1].iloc[0])
                volatility = float(data['close'].pct_change().rolling(20).std().iloc[-1].iloc[0]) * np.sqrt(252)
                
                # RSI calculation
                delta = data['close'].diff()
                gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
                loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
                rs = gain / loss
                rsi = 100 - (100 / (1 + rs))
                current_rsi = float(rsi.iloc[-1].iloc[0]) if not pd.isna(rsi.iloc[-1].iloc[0]) else 50
                
                # Volume analysis
                avg_volume = float(data['volume'].rolling(20).mean().iloc[-1].iloc[0])
                current_volume = float(data['volume'].iloc[-1].iloc[0])
                volume_ratio = current_volume / avg_volume if avg_volume > 0 else 1
                
                # Generate opportunity signals
                opportunity = {
                    'ticker': ticker,
                    'current_price': current_price,
                    'signal': 'HOLD',
                    'confidence': 0.5,
                    'reason': 'No clear signal',
                    'risk_level': 'medium',
                    'expected_return': 0.0,
                    'stop_loss': current_price * 0.95,
                    'target_price': current_price * 1.05,
                    'timeframe': '1-3 days'
                }
                
                # Technical signals
                if current_price > sma_20 > sma_50 and current_rsi < 70:
                    opportunity['signal'] = 'BUY'
                    opportunity['confidence'] = 0.7
                    opportunity['reason'] = 'Uptrend with healthy RSI'
                    opportunity['expected_return'] = 0.03
                    opportunity['target_price'] = current_price * 1.05  # 5% above current
                    opportunity['stop_loss'] = current_price * 0.95     # 5% below current
                elif current_price < sma_20 < sma_50 and current_rsi > 30:
                    opportunity['signal'] = 'SELL'
                    opportunity['confidence'] = 0.6
                    opportunity['reason'] = 'Downtrend with oversold RSI'
                    opportunity['expected_return'] = -0.02
                    opportunity['target_price'] = current_price * 0.95  # 5% below current
                    opportunity['stop_loss'] = current_price * 1.05     # 5% above current
                elif current_rsi < 30 and volume_ratio > 1.5:
                    opportunity['signal'] = 'BUY'
                    opportunity['confidence'] = 0.8
                    opportunity['reason'] = 'Oversold with high volume'
                    opportunity['expected_return'] = 0.05
                    opportunity['target_price'] = current_price * 1.08  # 8% above current
                    opportunity['stop_loss'] = current_price * 0.92     # 8% below current
                elif current_rsi > 70 and volume_ratio > 1.5:
                    opportunity['signal'] = 'SELL'
                    opportunity['confidence'] = 0.8
                    opportunity['reason'] = 'Overbought with high volume'
                    opportunity['expected_return'] = -0.03
                    opportunity['target_price'] = current_price * 0.97  # 3% below current
                    opportunity['stop_loss'] = current_price * 1.03     # 3% above current
                
                # Adjust for risk level
                if risk_level == 'low':
                    if opportunity['confidence'] < 0.8:
                        opportunity['signal'] = 'HOLD'
                elif risk_level == 'high':
                    if opportunity['confidence'] > 0.6:
                        opportunity['confidence'] = min(0.9, opportunity['confidence'] + 0.1)
                
                opportunities.append(opportunity)
                
            except Exception as e:
                logger.warning(f"Failed to analyze {ticker}: {e}")
                continue
        
        # Sort by confidence and expected return
        opportunities.sort(key=lambda x: x['confidence'] * abs(x['expected_return']), reverse=True)
        
        return opportunities[:5]  # Top 5 opportunities

    def _generate_strategy_recommendations(self, market_data: dict, regime: str, risk_level: str) -> list:
        """Generate strategy recommendations based on market regime"""
        strategies = []
        
        # Regime-based strategy recommendations
        if 'bull' in regime:
            strategies.extend([
                {
                    'name': 'Momentum Breakout',
                    'description': 'Buy on breakouts above resistance with volume confirmation',
                    'risk_level': 'medium',
                    'expected_return': 0.08,
                    'timeframe': '1-2 weeks',
                    'entry_criteria': 'Price breaks above 20-day high with 2x average volume',
                    'exit_criteria': 'Price falls below 10-day moving average'
                },
                {
                    'name': 'Growth Stock Rotation',
                    'description': 'Focus on high-growth stocks with strong fundamentals',
                    'risk_level': 'high',
                    'expected_return': 0.12,
                    'timeframe': '2-4 weeks',
                    'entry_criteria': 'Strong earnings growth and positive momentum',
                    'exit_criteria': 'Fundamental deterioration or technical breakdown'
                }
            ])
        elif 'bear' in regime:
            strategies.extend([
                {
                    'name': 'Defensive Rotation',
                    'description': 'Rotate into defensive sectors and dividend stocks',
                    'risk_level': 'low',
                    'expected_return': 0.04,
                    'timeframe': '2-6 weeks',
                    'entry_criteria': 'Utilities, consumer staples, healthcare',
                    'exit_criteria': 'Market regime change to bullish'
                },
                {
                    'name': 'Short Volatility',
                    'description': 'Sell volatility spikes in stable markets',
                    'risk_level': 'high',
                    'expected_return': 0.06,
                    'timeframe': '1-3 weeks',
                    'entry_criteria': 'VIX > 30 and market showing signs of stability',
                    'exit_criteria': 'VIX drops below 20 or market breaks down'
                }
            ])
        else:  # sideways/unknown
            strategies.extend([
                {
                    'name': 'Mean Reversion',
                    'description': 'Buy oversold, sell overbought in range-bound markets',
                    'risk_level': 'medium',
                    'expected_return': 0.05,
                    'timeframe': '3-7 days',
                    'entry_criteria': 'RSI < 30 or price touches lower Bollinger Band',
                    'exit_criteria': 'RSI > 70 or price touches upper Bollinger Band'
                },
                {
                    'name': 'Options Income',
                    'description': 'Sell covered calls or cash-secured puts for income',
                    'risk_level': 'medium',
                    'expected_return': 0.03,
                    'timeframe': '1-4 weeks',
                    'entry_criteria': 'Stable stocks with high implied volatility',
                    'exit_criteria': 'Assignment or expiration'
                }
            ])
        
        # Filter by risk level
        if risk_level == 'low':
            strategies = [s for s in strategies if s['risk_level'] == 'low']
        elif risk_level == 'high':
            strategies = [s for s in strategies if s['risk_level'] in ['medium', 'high']]
        
        return strategies[:3]  # Top 3 strategies

    def _get_market_context(self, ticker: str, data: pd.DataFrame) -> str:
        """Get current market context and key levels"""
        try:
            current_price = float(data['close'].iloc[-1])
            sma_20 = float(data['close'].rolling(20).mean().iloc[-1])
            sma_50 = float(data['close'].rolling(50).mean().iloc[-1])
            
            # Calculate recent high/low
            recent_high = float(data['high'].rolling(20).max().iloc[-1])
            recent_low = float(data['low'].rolling(20).min().iloc[-1])
            
            # Determine key levels
            context = f"📊 {ticker} MARKET CONTEXT:\n"
            context += f"- Current Price: ${current_price:.2f}\n"
            context += f"- 20-day SMA: ${sma_20:.2f} ({'ABOVE' if current_price > sma_20 else 'BELOW'})\n"
            context += f"- 50-day SMA: ${sma_50:.2f} ({'ABOVE' if current_price > sma_50 else 'BELOW'})\n"
            context += f"- Recent High: ${recent_high:.2f}\n"
            context += f"- Recent Low: ${recent_low:.2f}\n"
            
            # Key level analysis
            if current_price > sma_20 > sma_50:
                context += f"- BULLISH: {ticker} above both SMAs (uptrend)\n"
            elif current_price < sma_20 < sma_50:
                context += f"- BEARISH: {ticker} below both SMAs (downtrend)\n"
            elif current_price > sma_20 and current_price < sma_50:
                context += f"- MIXED: {ticker} above 20-day but below 50-day SMA\n"
            elif current_price < sma_20 and current_price > sma_50:
                context += f"- MIXED: {ticker} below 20-day but above 50-day SMA\n"
            
            # Support/Resistance levels
            if current_price > recent_high * 0.98:
                context += f"- NEAR RESISTANCE: {ticker} approaching recent high of ${recent_high:.2f}\n"
            elif current_price < recent_low * 1.02:
                context += f"- NEAR SUPPORT: {ticker} approaching recent low of ${recent_low:.2f}\n"
            
            return context
            
        except Exception as e:
            logger.warning(f"Failed to get market context for {ticker}: {e}")
            return ""

    def _create_daily_report(self, analysis_type: str, regime: str, opportunities: list, strategies: list, market_data: dict, llm_insights: str = "") -> str:
        """Create daily strategy report"""
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        
        # Get market summary
        market_summary = ""
        for ticker, data in market_data.items():
            current_price = float(data['close'].iloc[-1].iloc[0])
            daily_change = float(data['close'].pct_change().iloc[-1].iloc[0]) * 100
            market_summary += f"- **{ticker}**: ${current_price:.2f} ({daily_change:+.2f}%)\n"
        
        report = f"""
# Daily Strategy Advisor Report
**Generated:** {timestamp}
**Analysis:** {analysis_type.title()}
**Market Regime:** {regime.replace('_', ' ').title()}

## Market Summary
{market_summary}

## AI Market Analysis - {analysis_type.upper()}
{llm_insights if llm_insights else "*LLM analysis not available*"}

## Top Trading Opportunities
"""
        
        for i, opp in enumerate(opportunities, 1):
            report += f"""
### {i}. {opp['ticker']} - {opp['signal']}
- **Current Price:** ${opp['current_price']:.2f}
- **Signal:** {opp['signal']}
- **Confidence:** {opp['confidence']:.1%}
- **Reason:** {opp['reason']}
- **Expected Return:** {opp['expected_return']:+.1%}
- **Target Price:** ${opp['target_price']:.2f}
- **Stop Loss:** ${opp['stop_loss']:.2f}
- **Timeframe:** {opp['timeframe']}
"""
        
        report += f"""
## Recommended Strategies
"""
        
        for i, strategy in enumerate(strategies, 1):
            report += f"""
### {i}. {strategy['name']}
- **Description:** {strategy['description']}
- **Risk Level:** {strategy['risk_level'].title()}
- **Expected Return:** {strategy['expected_return']:+.1%}
- **Timeframe:** {strategy['timeframe']}
- **Entry Criteria:** {strategy['entry_criteria']}
- **Exit Criteria:** {strategy['exit_criteria']}
"""
        
        report += f"""
## Risk Management
- **Position Sizing:** Limit individual positions to 2-5% of portfolio
- **Stop Losses:** Use the recommended stop loss levels
- **Diversification:** Don't put all capital in one strategy
- **Market Conditions:** Current regime: {regime.replace('_', ' ').title()}

## Disclaimer
This report is for informational purposes only and not financial advice. 
Always do your own research and consider your risk tolerance before trading.

---
*Generated by QuantEngine Daily Strategy Advisor*
"""
        
        return report

    def save_report(self, report: str, filename: str = None):
        """Save report to file"""
        if filename is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f"daily_strategy_report_{timestamp}.md"
        
        reports_dir = Path("reports")
        reports_dir.mkdir(exist_ok=True)
        
        filepath = reports_dir / filename
        with open(filepath, 'w') as f:
            f.write(report)
        
        logger.info(f"✅ Report saved: {filepath}")
        return filepath

def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description='Daily Strategy Advisor')
    parser.add_argument('--asset', help='Specific asset to analyze (e.g., NVDA)')
    parser.add_argument('--sector', help='Sector to analyze (e.g., technology)')
    parser.add_argument('--risk', choices=['low', 'medium', 'high'], default='medium', 
                       help='Risk tolerance level')
    parser.add_argument('--output', help='Output filename for report')
    
    args = parser.parse_args()
    
    advisor = DailyStrategyAdvisor()
    
    try:
        report = advisor.get_daily_recommendations(
            asset=args.asset,
            sector=args.sector,
            risk_level=args.risk
        )
        
        if report:
            print("\n" + "="*80)
            print(report)
            print("="*80)
            
            # Save report
            filepath = advisor.save_report(report, args.output)
            print(f"\n📄 Report saved to: {filepath}")
        else:
            print("❌ Failed to generate recommendations")
            
    except Exception as e:
        logger.error(f"❌ Error generating recommendations: {e}")
        raise

if __name__ == "__main__":
    main()
