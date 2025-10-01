#!/usr/bin/env python3
"""
QuantEngine Chat Interface Runner

Interactive command-line interface for asking research questions to the QuantEngine.
Supports both single questions and interactive chat mode.
"""

import asyncio
import argparse
import logging
import sys
from pathlib import Path
from datetime import datetime
from typing import Dict

# Add QuantEngine root to path
quant_engine_root = Path(__file__).parent
if str(quant_engine_root) not in sys.path:
    sys.path.insert(0, str(quant_engine_root))

from chat_interface import QuantResearchChat
from research_agent import AdvancedResearchAgent

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('quant_chat.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger('QuantChat')

class QuantChatRunner:
    """Interactive chat runner for QuantEngine research"""
    
    def __init__(self, use_advanced: bool = False):
        self.use_advanced = use_advanced
        self.chat = None
        self.advanced_agent = None
        
    async def initialize(self):
        """Initialize the chat interface"""
        logger.info("🚀 Initializing QuantEngine Chat Interface")
        
        try:
            if self.use_advanced:
                # Initialize advanced research agent
                config = {
                    "universe": {
                        "equities": ["SPY", "QQQ", "IWM", "VTI", "BND"],
                        "sectors": ["XLE", "XLF", "XLK", "XLV", "XLY", "XLI", "XLC", "XLU", "XLB", "XLRE"],
                        "leveraged": ["TQQQ", "SQQQ", "UVXY", "SVXY"]
                    }
                }
                self.advanced_agent = AdvancedResearchAgent(config)
                logger.info("✅ Advanced Research Agent initialized")
            else:
                # Initialize basic chat interface
                self.chat = QuantResearchChat()
                await self.chat.initialize()
                logger.info("✅ Basic Chat Interface initialized")
                
        except Exception as e:
            logger.error(f"❌ Initialization failed: {e}")
            raise
    
    async def run_single_question(self, question: str):
        """Run a single research question"""
        print(f"\n🔍 Question: {question}")
        print("=" * 80)
        
        try:
            if self.use_advanced and self.advanced_agent:
                # Use advanced agent
                parsed_question = self.chat.parse_question(question) if self.chat else {}
                market_context = await self.chat.get_market_context() if self.chat else {}
                
                report = await self.advanced_agent.generate_comprehensive_report(
                    question, parsed_question, market_context
                )
                
                # Format and display report
                self._display_advanced_report(report)
            else:
                # Use basic chat interface
                response = await self.chat.ask_question(question)
                formatted_response = self.chat.format_response(response)
                print(formatted_response)
                
        except Exception as e:
            logger.error(f"Error processing question: {e}")
            print(f"❌ Error: {e}")
    
    def _display_advanced_report(self, report: Dict):
        """Display advanced research report"""
        
        print(f"# Research Report: {report['question']}")
        print(f"**Generated:** {report['analysis_timestamp']}")
        print(f"**Confidence Score:** {report['confidence_score']:.1%}")
        print("")
        
        # Market Context
        context = report['market_context']
        print("## Market Context")
        print(f"- **Regime:** {context.get('regime', 'Unknown')} (confidence: {context.get('regime_confidence', 0):.1%})")
        print(f"- **Sentiment:** {context.get('market_sentiment', 0):.2f}")
        print(f"- **Volatility:** {context.get('volatility_level', 0):.1f}")
        print("")
        
        # Monte Carlo Analysis
        if report['monte_carlo_analysis']:
            print("## Monte Carlo Analysis")
            for asset, mc_results in report['monte_carlo_analysis'].items():
                print(f"### {asset}")
                stats = mc_results['statistics']
                print(f"- **Expected Return:** {stats['mean_return']:.2%}")
                print(f"- **Volatility:** {stats['std_return']:.2%}")
                print(f"- **Skewness:** {stats['skewness']:.2f}")
                print(f"- **Kurtosis:** {stats['kurtosis']:.2f}")
                
                # Show percentiles
                percentiles = mc_results['percentiles']['returns']
                print(f"- **5th Percentile:** {percentiles[5]:.1%}")
                print(f"- **25th Percentile:** {percentiles[25]:.1%}")
                print(f"- **50th Percentile:** {percentiles[50]:.1%}")
                print(f"- **75th Percentile:** {percentiles[75]:.1%}")
                print(f"- **95th Percentile:** {percentiles[95]:.1%}")
                print("")
        
        # Scenarios
        if report['scenarios']:
            print("## Scenario Analysis")
            for i, scenario in enumerate(report['scenarios'][:10], 1):  # Show top 10
                print(f"### Scenario {i}: {scenario['name']}")
                print(f"**Probability:** {scenario.get('probability', 0):.1%}")
                print(f"**Description:** {scenario.get('description', 'N/A')}")
                
                if 'price_range' in scenario:
                    print(f"**Price Range:** {scenario['price_range']}")
                
                if 'sector_impact' in scenario:
                    print(f"**Sector Impact:** {scenario['sector_impact']}")
                
                if 'key_drivers' in scenario:
                    print("**Key Drivers:**")
                    for driver in scenario['key_drivers']:
                        print(f"- {driver}")
                
                print(f"**Confidence:** {scenario.get('confidence', 0):.1%}")
                print("")
        
        # Recommendations
        if report['recommendations']:
            print("## Recommendations")
            for i, rec in enumerate(report['recommendations'], 1):
                print(f"{i}. {rec}")
            print("")
    
    async def run_interactive_mode(self):
        """Run interactive chat mode"""
        print("\n🤖 QuantEngine Research Chat")
        print("=" * 50)
        print("Ask research questions about markets, stocks, sectors, and economic events.")
        print("Type 'help' for examples, 'quit' to exit.")
        print("")
        
        while True:
            try:
                # Get user input
                question = input("\n💬 Your question: ").strip()
                
                if question.lower() in ['quit', 'exit', 'q']:
                    print("👋 Goodbye!")
                    break
                
                if question.lower() == 'help':
                    self._show_help()
                    continue
                
                if not question:
                    continue
                
                # Process question
                await self.run_single_question(question)
                
            except KeyboardInterrupt:
                print("\n👋 Goodbye!")
                break
            except Exception as e:
                logger.error(f"Error in interactive mode: {e}")
                print(f"❌ Error: {e}")
    
    def _show_help(self):
        """Show help and example questions"""
        print("\n📚 Example Questions:")
        print("")
        print("Market Analysis:")
        print("- How will the Fed interest rate decision impact housing prices in 6 months?")
        print("- What's the outlook for tech stocks given current market conditions?")
        print("- How will inflation affect different sectors?")
        print("")
        print("Stock Research:")
        print("- Research NFLX company for stock price outlook 3 months from now")
        print("- Analyze AAPL stock performance under different scenarios")
        print("- What are the key risks for TSLA stock?")
        print("")
        print("Sector Analysis:")
        print("- How will rising interest rates impact the financial sector?")
        print("- What's the outlook for energy stocks in 2024?")
        print("- How will AI developments affect the technology sector?")
        print("")
        print("Economic Events:")
        print("- How will the next jobs report impact the market?")
        print("- What's the impact of a potential recession on different assets?")
        print("- How will trade tensions affect specific sectors?")
        print("")
        print("Commands:")
        print("- 'help' - Show this help message")
        print("- 'quit' or 'exit' - Exit the chat")
        print("")

def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description='QuantEngine Research Chat Interface')
    parser.add_argument('--question', '-q', type=str, help='Single question to ask')
    parser.add_argument('--advanced', '-a', action='store_true', 
                       help='Use advanced research agent with Monte Carlo analysis')
    parser.add_argument('--interactive', '-i', action='store_true',
                       help='Run in interactive mode')
    
    args = parser.parse_args()
    
    # Determine mode
    if args.question:
        mode = 'single'
    elif args.interactive:
        mode = 'interactive'
    else:
        mode = 'interactive'  # Default to interactive
    
    # Create and run chat
    runner = QuantChatRunner(use_advanced=args.advanced)
    
    async def run():
        await runner.initialize()
        
        if mode == 'single':
            await runner.run_single_question(args.question)
        else:
            await runner.run_interactive_mode()
    
    try:
        asyncio.run(run())
    except KeyboardInterrupt:
        print("\n👋 Goodbye!")
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        print(f"❌ Fatal error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
