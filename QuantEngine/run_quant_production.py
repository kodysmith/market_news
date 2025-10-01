#!/usr/bin/env python3
"""
Production runner for QuantEngine with specific asset/goal targeting

Usage:
    python run_quant_production.py "create a trade strategy report for NVDA over the next two quarters"
    python run_quant_production.py "analyze house prices for the next 6 months"
    python run_quant_production.py "research REITs for Q1 2024"
"""

import os
import sys
import logging
import argparse
from pathlib import Path
from datetime import datetime, timedelta
import re
import pandas as pd
import numpy as np

# Set production environment
os.environ['QUANT_ENV'] = 'production'

# Add QuantEngine to path
quant_engine_dir = Path(__file__).parent
sys.path.insert(0, str(quant_engine_dir))

# Configure logging for production
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('quant_engine_production.log'),
        logging.StreamHandler(sys.stdout)
    ]
)

logger = logging.getLogger(__name__)

class QuantResearchTargeter:
    """Targets QuantEngine research for specific assets and goals"""
    
    def __init__(self):
        self.research_agent = None
        self.data_manager = None
        self.config = None
        
    def parse_research_query(self, query: str) -> dict:
        """Parse natural language query into research parameters"""
        query_lower = query.lower()
        
        # Extract asset/sector
        asset = None
        sector = None
        
        # Common stock tickers
        stock_patterns = {
            'nvda': 'NVDA', 'nvidia': 'NVDA',
            'aapl': 'AAPL', 'apple': 'AAPL',
            'msft': 'MSFT', 'microsoft': 'MSFT',
            'googl': 'GOOGL', 'google': 'GOOGL',
            'amzn': 'AMZN', 'amazon': 'AMZN',
            'tsla': 'TSLA', 'tesla': 'TSLA',
            'meta': 'META', 'facebook': 'META',
            'spy': 'SPY', 'sp500': 'SPY',
            'qqq': 'QQQ', 'nasdaq': 'QQQ',
            'tqqq': 'TQQQ', '3x nasdaq': 'TQQQ'
        }
        
        for pattern, ticker in stock_patterns.items():
            if pattern in query_lower:
                asset = ticker
                break
        
        # Sector analysis
        if 'house' in query_lower or 'housing' in query_lower or 'real estate' in query_lower:
            sector = 'real_estate'
            asset = 'REAL_ESTATE'
        elif 'reit' in query_lower or 'reits' in query_lower:
            sector = 'reits'
            asset = 'REITS'
        elif 'tech' in query_lower or 'technology' in query_lower:
            sector = 'technology'
            asset = 'TECH'
        elif 'energy' in query_lower:
            sector = 'energy'
            asset = 'ENERGY'
        elif 'healthcare' in query_lower or 'health' in query_lower:
            sector = 'healthcare'
            asset = 'HEALTHCARE'
        
        # Extract timeframe
        timeframe = '3 months'  # default
        if 'quarter' in query_lower or 'quarters' in query_lower:
            if 'two' in query_lower or '2' in query_lower:
                timeframe = '6 months'
            elif 'three' in query_lower or '3' in query_lower:
                timeframe = '9 months'
            else:
                timeframe = '3 months'
        elif 'month' in query_lower or 'months' in query_lower:
            if '6' in query_lower or 'six' in query_lower:
                timeframe = '6 months'
            elif '12' in query_lower or 'twelve' in query_lower or 'year' in query_lower:
                timeframe = '12 months'
            else:
                timeframe = '3 months'
        elif 'year' in query_lower or 'years' in query_lower:
            timeframe = '12 months'
        
        # Extract research type
        research_type = 'strategy'
        if 'report' in query_lower or 'analysis' in query_lower:
            research_type = 'analysis'
        elif 'strategy' in query_lower or 'trade' in query_lower:
            research_type = 'strategy'
        elif 'outlook' in query_lower or 'forecast' in query_lower:
            research_type = 'outlook'
        
        return {
            'asset': asset,
            'sector': sector,
            'timeframe': timeframe,
            'research_type': research_type,
            'original_query': query
        }
    
    def initialize_components(self):
        """Initialize QuantEngine components"""
        try:
            from main import load_config, setup_logging
            from research.research_agent import ResearchAgent
            from engine.data_ingestion.data_manager import DataManager
            
            # Load configuration
            self.config = load_config('config/config.yaml')
            setup_logging(self.config)
            
            # Initialize components
            self.data_manager = DataManager(self.config)
            self.research_agent = ResearchAgent(self.config)
            
            logger.info("✅ QuantEngine components initialized")
            
        except Exception as e:
            logger.error(f"Failed to initialize components: {e}")
            raise
    
    def run_targeted_research(self, query: str):
        """Run targeted research for specific asset/goal"""
        logger.info(f"🎯 Running targeted research: {query}")
        
        # Parse the query
        params = self.parse_research_query(query)
        logger.info(f"📊 Research parameters: {params}")
        
        # Initialize components
        self.initialize_components()
        
        # Determine research approach based on asset type
        if params['asset'] and params['asset'] not in ['REAL_ESTATE', 'REITS', 'TECH', 'ENERGY', 'HEALTHCARE']:
            # Individual stock analysis
            return self._run_stock_analysis(params)
        elif params['sector']:
            # Sector analysis
            return self._run_sector_analysis(params)
        else:
            # General market analysis
            return self._run_general_analysis(params)
    
    def _run_stock_analysis(self, params: dict):
        """Run analysis for specific stock"""
        asset = params['asset']
        timeframe = params['timeframe']
        
        logger.info(f"📈 Analyzing {asset} for {timeframe}")
        
        # Get market data
        end_date = datetime.now()
        start_date = end_date - timedelta(days=365)  # 1 year of data
        
        try:
            market_data = self.data_manager.get_market_data(
                [asset], 
                start_date.strftime('%Y-%m-%d'), 
                end_date.strftime('%Y-%m-%d')
            )
            
            if not market_data or asset not in market_data:
                logger.error(f"No data available for {asset}")
                return None
            
            # Generate research report
            report = self._generate_stock_report(asset, market_data[asset], params)
            return report
            
        except Exception as e:
            logger.error(f"Failed to analyze {asset}: {e}")
            return None
    
    def _run_sector_analysis(self, params: dict):
        """Run analysis for sector"""
        sector = params['sector']
        timeframe = params['timeframe']
        
        logger.info(f"🏢 Analyzing {sector} sector for {timeframe}")
        
        # Define sector ETFs
        sector_etfs = {
            'real_estate': ['VNQ', 'IYR'],
            'reits': ['VNQ', 'IYR', 'REZ'],
            'technology': ['XLK', 'QQQ', 'VGT'],
            'energy': ['XLE', 'VDE'],
            'healthcare': ['XLV', 'VHT']
        }
        
        etfs = sector_etfs.get(sector, ['SPY'])
        
        # Get market data
        end_date = datetime.now()
        start_date = end_date - timedelta(days=365)
        
        try:
            market_data = self.data_manager.get_market_data(
                etfs,
                start_date.strftime('%Y-%m-%d'),
                end_date.strftime('%Y-%m-%d')
            )
            
            # Generate sector report
            report = self._generate_sector_report(sector, market_data, params)
            return report
            
        except Exception as e:
            logger.error(f"Failed to analyze {sector} sector: {e}")
            return None
    
    def _run_general_analysis(self, params: dict):
        """Run general market analysis"""
        logger.info("🌍 Running general market analysis")
        
        # Use broad market ETFs
        etfs = ['SPY', 'QQQ', 'IWM', 'VTI']
        
        end_date = datetime.now()
        start_date = end_date - timedelta(days=365)
        
        try:
            market_data = self.data_manager.get_market_data(
                etfs,
                start_date.strftime('%Y-%m-%d'),
                end_date.strftime('%Y-%m-%d')
            )
            
            # Generate general report
            report = self._generate_general_report(market_data, params)
            return report
            
        except Exception as e:
            logger.error(f"Failed to run general analysis: {e}")
            return None
    
    def _generate_stock_report(self, asset: str, data: pd.DataFrame, params: dict):
        """Generate stock-specific research report"""
        logger.info(f"📝 Generating stock report for {asset}")
        
        # Basic technical analysis
        current_price = float(data['close'].iloc[-1].iloc[0])
        price_change = float(data['close'].pct_change().iloc[-1].iloc[0]) * 100
        
        # Simple moving averages
        sma_20 = float(data['close'].rolling(20).mean().iloc[-1].iloc[0])
        sma_50 = float(data['close'].rolling(50).mean().iloc[-1].iloc[0])
        
        # Volatility
        volatility = float(data['close'].pct_change().rolling(20).std().iloc[-1].iloc[0]) * np.sqrt(252) * 100
        
        # Support and Resistance Analysis
        support_resistance = self._analyze_support_resistance(data, params)
        
        # Generate scenarios based on technical analysis
        scenarios = self._generate_scenarios(data, params)
        
        # Create report
        report = f"""
# Research Report: {asset}
**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
**Query:** {params['original_query']}

## Current Market Data
- **Current Price:** ${current_price:.2f}
- **Daily Change:** {price_change:.2f}%
- **20-day SMA:** ${sma_20:.2f}
- **50-day SMA:** ${sma_50:.2f}
- **Volatility:** {volatility:.1f}%

## Support & Resistance Analysis
{support_resistance}

## Price Scenarios for {params['timeframe']}
{scenarios}

## Trading Strategy Recommendations
Based on current technical analysis and market conditions, consider:

1. **Entry Strategy:** Monitor for breakout above resistance levels
2. **Risk Management:** Set stop-loss at key support levels
3. **Position Sizing:** Adjust based on volatility and risk tolerance
4. **Time Horizon:** Align with {params['timeframe']} outlook

## Risk Factors
- Market volatility may impact short-term performance
- Sector-specific risks should be considered
- Economic conditions may affect overall market sentiment

*This report is for informational purposes only and not financial advice.*
"""
        
        # Save report
        report_file = f"reports/{asset.lower()}_research_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md"
        os.makedirs('reports', exist_ok=True)
        
        with open(report_file, 'w') as f:
            f.write(report)
        
        logger.info(f"✅ Report saved: {report_file}")
        
        # Publish to Firebase for the market news app
        try:
            self._publish_to_firebase(report_file, asset, params)
        except Exception as e:
            logger.warning(f"Failed to publish to Firebase: {e}")
        
        return report_file
    
    def _generate_sector_report(self, sector: str, market_data: dict, params: dict):
        """Generate sector-specific research report"""
        logger.info(f"📝 Generating sector report for {sector}")
        
        # Analyze sector ETFs
        sector_analysis = {}
        for etf, data in market_data.items():
            if data is not None and not data.empty:
                current_price = float(data['close'].iloc[-1])
                price_change = float(data['close'].pct_change().iloc[-1]) * 100
                volatility = float(data['close'].pct_change().rolling(20).std().iloc[-1]) * np.sqrt(252) * 100
                
                sector_analysis[etf] = {
                    'price': current_price,
                    'change': price_change,
                    'volatility': volatility
                }
        
        # Create report
        report = f"""
# Sector Research Report: {sector.upper()}
**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
**Query:** {params['original_query']}

## Sector Analysis for {params['timeframe']}
"""
        
        for etf, analysis in sector_analysis.items():
            report += f"""
### {etf}
- **Current Price:** ${analysis['price']:.2f}
- **Daily Change:** {analysis['change']:.2f}%
- **Volatility:** {analysis['volatility']:.1f}%
"""
        
        report += f"""
## Sector Outlook
Based on current market conditions and sector performance:

1. **Trend Analysis:** Monitor sector rotation patterns
2. **Risk Assessment:** Consider sector-specific risks
3. **Opportunity Identification:** Look for relative strength/weakness
4. **Time Horizon:** {params['timeframe']} outlook

## Investment Considerations
- Diversification across sector components
- Risk management through position sizing
- Monitoring of sector rotation trends
- Economic cycle considerations

*This report is for informational purposes only and not financial advice.*
"""
        
        # Save report
        report_file = f"reports/{sector}_sector_research_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md"
        os.makedirs('reports', exist_ok=True)
        
        with open(report_file, 'w') as f:
            f.write(report)
        
        logger.info(f"✅ Report saved: {report_file}")
        return report_file
    
    def _generate_general_report(self, market_data: dict, params: dict):
        """Generate general market research report"""
        logger.info("📝 Generating general market report")
        
        # Analyze broad market ETFs
        market_analysis = {}
        for etf, data in market_data.items():
            if data is not None and not data.empty:
                current_price = float(data['close'].iloc[-1])
                price_change = float(data['close'].pct_change().iloc[-1]) * 100
                volatility = float(data['close'].pct_change().rolling(20).std().iloc[-1]) * np.sqrt(252) * 100
                
                market_analysis[etf] = {
                    'price': current_price,
                    'change': price_change,
                    'volatility': volatility
                }
        
        # Create report
        report = f"""
# General Market Research Report
**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
**Query:** {params['original_query']}

## Market Overview for {params['timeframe']}
"""
        
        for etf, analysis in market_analysis.items():
            report += f"""
### {etf}
- **Current Price:** ${analysis['price']:.2f}
- **Daily Change:** {analysis['change']:.2f}%
- **Volatility:** {analysis['volatility']:.1f}%
"""
        
        report += f"""
## Market Outlook
Based on current market conditions:

1. **Trend Analysis:** Monitor overall market direction
2. **Risk Assessment:** Consider market-wide risks
3. **Opportunity Identification:** Look for relative strength
4. **Time Horizon:** {params['timeframe']} outlook

## Investment Considerations
- Diversification across asset classes
- Risk management through position sizing
- Monitoring of market trends
- Economic cycle considerations

*This report is for informational purposes only and not financial advice.*
"""
        
        # Save report
        report_file = f"reports/general_market_research_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md"
        os.makedirs('reports', exist_ok=True)
        
        with open(report_file, 'w') as f:
            f.write(report)
        
        logger.info(f"✅ Report saved: {report_file}")
        return report_file
    
    def _generate_scenarios(self, data: pd.DataFrame, params: dict):
        """Generate price scenarios based on technical analysis"""
        current_price = float(data['close'].iloc[-1].iloc[0])
        volatility = float(data['close'].pct_change().rolling(20).std().iloc[-1].iloc[0]) * np.sqrt(252)
        
        # Simple scenario generation
        scenarios = f"""
### Bullish Scenario (40% probability)
- **Target Price:** ${current_price * 1.15:.2f} (+15%)
- **Key Levels:** Break above resistance
- **Catalysts:** Positive earnings, sector strength

### Neutral Scenario (40% probability)
- **Target Price:** ${current_price * 1.02:.2f} (+2%)
- **Key Levels:** Range-bound trading
- **Catalysts:** Mixed signals, consolidation

### Bearish Scenario (20% probability)
- **Target Price:** ${current_price * 0.85:.2f} (-15%)
- **Key Levels:** Break below support
- **Catalysts:** Negative news, sector weakness
"""
        
        return scenarios
    
    def _analyze_support_resistance(self, data: pd.DataFrame, params: dict):
        """Analyze support and resistance levels"""
        logger.info("🔍 Analyzing support and resistance levels")
        
        current_price = float(data['close'].iloc[-1].iloc[0])
        
        # Determine lookback period based on timeframe
        timeframe = params.get('timeframe', '3 months')
        if '6 months' in timeframe or 'two quarters' in timeframe:
            lookback = min(120, len(data))  # 6 months of daily data
        elif '3 months' in timeframe or 'quarter' in timeframe:
            lookback = min(60, len(data))   # 3 months of daily data
        elif 'month' in timeframe:
            lookback = min(20, len(data))   # 1 month of daily data
        else:
            lookback = min(40, len(data))    # Default to ~2 months
        
        # Use recent data for analysis
        recent_data = data.tail(lookback)
        
        # Simple support and resistance using moving averages and price levels
        recent_highs = recent_data['high']
        recent_lows = recent_data['low']
        recent_closes = recent_data['close']
        
        # Find key levels using simple statistical methods
        resistance_levels = self._find_key_levels(recent_highs, current_price, 'resistance')
        support_levels = self._find_key_levels(recent_lows, current_price, 'support')
        
        # Format analysis
        analysis = "### Key Resistance Levels\n"
        if resistance_levels:
            for level in resistance_levels[:3]:  # Top 3
                distance = ((level - current_price) / current_price) * 100
                analysis += f"- **${level:.2f}** (+{distance:.1f}% from current)\n"
        else:
            analysis += "- No significant resistance levels identified\n"
        
        analysis += "\n### Key Support Levels\n"
        if support_levels:
            for level in support_levels[:3]:  # Top 3
                distance = ((current_price - level) / current_price) * 100
                analysis += f"- **${level:.2f}** (-{distance:.1f}% from current)\n"
        else:
            analysis += "- No significant support levels identified\n"
        
        # Add current position analysis
        analysis += f"\n### Current Position Analysis\n"
        analysis += f"- **Current Price:** ${current_price:.2f}\n"
        
        if resistance_levels:
            nearest_resistance = min(resistance_levels)
            resistance_distance = ((nearest_resistance - current_price) / current_price) * 100
            analysis += f"- **Nearest Resistance:** ${nearest_resistance:.2f} (+{resistance_distance:.1f}%)\n"
        else:
            analysis += "- **Nearest Resistance:** None identified\n"
        
        if support_levels:
            nearest_support = max(support_levels)
            support_distance = ((current_price - nearest_support) / current_price) * 100
            analysis += f"- **Nearest Support:** ${nearest_support:.2f} (-{support_distance:.1f}%)\n"
        else:
            analysis += "- **Nearest Support:** None identified\n"
        
        return analysis
    
    def _find_key_levels(self, price_data, current_price, level_type):
        """Find key support/resistance levels using statistical methods"""
        levels = []
        
        # Get price statistics
        mean_price = float(price_data.mean().iloc[0])
        std_price = float(price_data.std().iloc[0])
        max_price = float(price_data.max().iloc[0])
        min_price = float(price_data.min().iloc[0])
        
        # Calculate key levels based on statistical bands
        if level_type == 'resistance':
            # Resistance levels above current price
            levels.append(mean_price + std_price)  # 1 std above mean
            levels.append(mean_price + 2 * std_price)  # 2 std above mean
            levels.append(max_price)  # Recent high
            levels.append(current_price * 1.05)  # 5% above current
            levels.append(current_price * 1.10)  # 10% above current
        else:  # support
            # Support levels below current price
            levels.append(mean_price - std_price)  # 1 std below mean
            levels.append(mean_price - 2 * std_price)  # 2 std below mean
            levels.append(min_price)  # Recent low
            levels.append(current_price * 0.95)  # 5% below current
            levels.append(current_price * 0.90)  # 10% below current
        
        # Filter levels based on type
        if level_type == 'resistance':
            levels = [level for level in levels if level > current_price]
        else:  # support
            levels = [level for level in levels if level < current_price]
        
        # Remove duplicates and sort
        levels = sorted(list(set(levels)))
        
        return levels
    
    def _publish_to_firebase(self, report_file: str, asset: str, params: dict):
        """Publish report to Firebase for the market news app"""
        try:
            import requests
            
            # Read the report content
            with open(report_file, 'r') as f:
                content = f.read()
            
            # Parse basic data from the report
            current_price = None
            daily_change = None
            volatility = None
            
            lines = content.split('\n')
            for line in lines:
                if '**Current Price:**' in line:
                    import re
                    price_match = re.search(r'\$([\d.]+)', line)
                    if price_match:
                        current_price = float(price_match.group(1))
                elif '**Daily Change:**' in line:
                    import re
                    change_match = re.search(r'([+-]?[\d.]+)%', line)
                    if change_match:
                        daily_change = float(change_match.group(1))
                elif '**Volatility:**' in line:
                    import re
                    vol_match = re.search(r'([\d.]+)%', line)
                    if vol_match:
                        volatility = float(vol_match.group(1))
            
            # Create payload for Firebase
            payload = {
                "type": "quant_engine_report",
                "timestamp": datetime.now().isoformat(),
                "title": f"Research Report: {asset}",
                "query": params.get('original_query', ''),
                "asset": asset,
                "current_price": current_price,
                "daily_change": daily_change,
                "volatility": volatility,
                "timeframe": params.get('timeframe', '3 months'),
                "content": content,
                "report_path": report_file
            }
            
            # Send to Firebase
            firebase_url = "https://api-hvi4gdtdka-uc.a.run.app"
            response = requests.post(
                f"{firebase_url}/publish-quant-report",
                json=payload,
                headers={'Content-Type': 'application/json'},
                timeout=10
            )
            
            if response.status_code == 200:
                logger.info(f"✅ Report published to Firebase: {asset}")
            else:
                logger.warning(f"⚠️ Failed to publish to Firebase: {response.status_code}")
                
        except Exception as e:
            logger.warning(f"⚠️ Firebase publishing failed: {e}")
    
    def _find_pivot_highs(self, highs, closes, window=5):
        """Find pivot high points"""
        pivot_highs = []
        if len(highs) < window * 2:
            return pivot_highs
            
        for i in range(window, len(highs) - window):
            is_pivot = True
            # Check if current high is higher than surrounding highs
            for j in range(i - window, i + window + 1):
                if j != i and j < len(highs) and highs.iloc[j] >= highs.iloc[i]:
                    is_pivot = False
                    break
            if is_pivot:
                pivot_highs.append(float(highs.iloc[i]))
        return pivot_highs
    
    def _find_pivot_lows(self, lows, closes, window=5):
        """Find pivot low points"""
        pivot_lows = []
        if len(lows) < window * 2:
            return pivot_lows
            
        for i in range(window, len(lows) - window):
            is_pivot = True
            # Check if current low is lower than surrounding lows
            for j in range(i - window, i + window + 1):
                if j != i and j < len(lows) and lows.iloc[j] <= lows.iloc[i]:
                    is_pivot = False
                    break
            if is_pivot:
                pivot_lows.append(float(lows.iloc[i]))
        return pivot_lows
    
    def _calculate_level_strength(self, price_data, levels, tolerance=0.02):
        """Calculate strength of support/resistance levels based on touches"""
        strength = {}
        for level in levels:
            touches = 0
            for price in price_data:
                # Check if price is within tolerance of the level
                if abs(price - level) / level <= tolerance:
                    touches += 1
            strength[level] = touches
        return strength

def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description='QuantEngine Production Research')
    parser.add_argument('query', help='Research query (e.g., "analyze NVDA for next quarter")')
    parser.add_argument('--output', '-o', help='Output file path')
    
    args = parser.parse_args()
    
    logger.info("🚀 Starting QuantEngine Targeted Research")
    logger.info(f"📊 Query: {args.query}")
    
    try:
        # Initialize research targeter
        targeter = QuantResearchTargeter()
        
        # Run targeted research
        result = targeter.run_targeted_research(args.query)
        
        if result:
            logger.info(f"✅ Research completed successfully")
            logger.info(f"📄 Report generated: {result}")
        else:
            logger.error("❌ Research failed")
            sys.exit(1)
            
    except KeyboardInterrupt:
        logger.info("🛑 Research interrupted by user")
    except Exception as e:
        logger.error(f"❌ Research failed: {e}")
        raise
    finally:
        logger.info("🏁 QuantEngine research completed")

if __name__ == "__main__":
    main()
