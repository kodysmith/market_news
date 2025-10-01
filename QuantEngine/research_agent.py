#!/usr/bin/env python3
"""
Advanced Research Agent for QuantEngine

Extends the basic research capabilities with more sophisticated analysis,
including Monte Carlo simulations, regime-aware modeling, and advanced
scenario generation with probability distributions.
"""

import asyncio
import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta
import json
from pathlib import Path
from scipy import stats
from scipy.optimize import minimize
import warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)

class AdvancedResearchAgent:
    """
    Advanced research agent with sophisticated modeling capabilities
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.monte_carlo_runs = 10000
        self.regime_models = {}
        self.correlation_matrices = {}
        
    async def generate_monte_carlo_scenarios(self, 
                                           asset: str,
                                           time_horizon: str,
                                           market_context: Dict[str, Any],
                                           n_simulations: int = 10000) -> Dict[str, Any]:
        """
        Generate Monte Carlo scenarios for asset price movements
        
        Args:
            asset: Asset symbol (e.g., 'NFLX', 'SPY')
            time_horizon: Time horizon ('3 months', '6 months', '1 year')
            market_context: Current market context
            n_simulations: Number of Monte Carlo simulations
            
        Returns:
            Monte Carlo simulation results with probability distributions
        """
        
        logger.info(f"Running Monte Carlo simulation for {asset} over {time_horizon}")
        
        # Convert time horizon to days
        horizon_days = self._parse_time_horizon(time_horizon)
        
        # Get historical data for the asset
        try:
            # This would integrate with the actual data manager
            historical_data = await self._get_historical_data(asset, days=252)  # 1 year
        except:
            # Fallback to simulated data
            historical_data = self._generate_simulated_data(asset, 252)
        
        # Calculate historical statistics
        returns = historical_data['close'].pct_change().dropna()
        mean_return = returns.mean()
        volatility = returns.std()
        
        # Adjust for current market regime
        regime_adjustment = self._get_regime_adjustment(market_context)
        adjusted_mean = mean_return * regime_adjustment['return_multiplier']
        adjusted_vol = volatility * regime_adjustment['vol_multiplier']
        
        # Generate Monte Carlo paths
        np.random.seed(42)  # For reproducibility
        simulations = []
        
        for _ in range(n_simulations):
            # Generate random returns
            random_returns = np.random.normal(adjusted_mean, adjusted_vol, horizon_days)
            
            # Calculate cumulative price path
            price_path = [100]  # Start at 100
            for ret in random_returns:
                price_path.append(price_path[-1] * (1 + ret))
            
            simulations.append(price_path)
        
        simulations = np.array(simulations)
        
        # Calculate statistics
        final_prices = simulations[:, -1]
        price_changes = (final_prices - 100) / 100
        
        # Calculate percentiles
        percentiles = [5, 10, 25, 50, 75, 90, 95]
        price_percentiles = np.percentile(final_prices, percentiles)
        return_percentiles = np.percentile(price_changes, percentiles)
        
        # Calculate probabilities for different scenarios
        scenarios = self._calculate_scenario_probabilities(price_changes, final_prices)
        
        return {
            'asset': asset,
            'time_horizon': time_horizon,
            'simulations': simulations,
            'final_prices': final_prices,
            'price_changes': price_changes,
            'statistics': {
                'mean_price': np.mean(final_prices),
                'std_price': np.std(final_prices),
                'mean_return': np.mean(price_changes),
                'std_return': np.std(price_changes),
                'skewness': stats.skew(price_changes),
                'kurtosis': stats.kurtosis(price_changes)
            },
            'percentiles': {
                'prices': dict(zip(percentiles, price_percentiles)),
                'returns': dict(zip(percentiles, return_percentiles))
            },
            'scenarios': scenarios,
            'regime_adjustment': regime_adjustment
        }
    
    def _parse_time_horizon(self, time_horizon: str) -> int:
        """Convert time horizon string to days"""
        if '3 months' in time_horizon.lower():
            return 90
        elif '6 months' in time_horizon.lower():
            return 180
        elif '1 year' in time_horizon.lower():
            return 252
        else:
            return 90  # Default to 3 months
    
    async def _get_historical_data(self, asset: str, days: int) -> pd.DataFrame:
        """Get historical data for asset"""
        # This would integrate with the actual data manager
        # For now, return simulated data
        return self._generate_simulated_data(asset, days)
    
    def _generate_simulated_data(self, asset: str, days: int) -> pd.DataFrame:
        """Generate simulated historical data"""
        np.random.seed(hash(asset) % 10000)
        
        # Generate price data
        returns = np.random.normal(0.0005, 0.02, days)  # Daily returns
        prices = 100 * np.exp(np.cumsum(returns))
        
        # Generate volume data
        volumes = np.random.lognormal(15, 0.5, days).astype(int)
        
        dates = pd.date_range(end=datetime.now(), periods=days, freq='D')
        
        return pd.DataFrame({
            'close': prices,
            'volume': volumes,
            'open': prices * (1 + np.random.normal(0, 0.005, days)),
            'high': prices * (1 + np.abs(np.random.normal(0, 0.01, days))),
            'low': prices * (1 - np.abs(np.random.normal(0, 0.01, days)))
        }, index=dates)
    
    def _get_regime_adjustment(self, market_context: Dict[str, Any]) -> Dict[str, float]:
        """Get regime-based adjustments for modeling"""
        regime = market_context.get('regime', 'unknown')
        volatility_level = market_context.get('volatility_level', 20.0)
        
        # Regime-based adjustments
        regime_adjustments = {
            'bull_market': {'return_multiplier': 1.2, 'vol_multiplier': 0.8},
            'bear_market': {'return_multiplier': 0.6, 'vol_multiplier': 1.3},
            'high_volatility': {'return_multiplier': 0.8, 'vol_multiplier': 1.5},
            'low_volatility': {'return_multiplier': 1.1, 'vol_multiplier': 0.7},
            'risk_on': {'return_multiplier': 1.3, 'vol_multiplier': 0.9},
            'risk_off': {'return_multiplier': 0.5, 'vol_multiplier': 1.4}
        }
        
        base_adjustment = regime_adjustments.get(regime, {'return_multiplier': 1.0, 'vol_multiplier': 1.0})
        
        # Adjust for volatility level
        vol_factor = volatility_level / 20.0  # Normalize to VIX 20
        base_adjustment['vol_multiplier'] *= vol_factor
        
        return base_adjustment
    
    def _calculate_scenario_probabilities(self, price_changes: np.ndarray, 
                                        final_prices: np.ndarray) -> List[Dict[str, Any]]:
        """Calculate probabilities for different price scenarios"""
        
        scenarios = []
        
        # Bull scenario (>10% gain)
        bull_prob = np.mean(price_changes > 0.10)
        scenarios.append({
            'name': 'Bull Scenario',
            'description': 'Price increases by more than 10%',
            'probability': bull_prob,
            'price_range': f"${np.percentile(final_prices[price_changes > 0.10], [25, 75])[0]:.0f} - ${np.percentile(final_prices[price_changes > 0.10], [25, 75])[1]:.0f}" if bull_prob > 0 else "N/A",
            'confidence': min(bull_prob * 2, 1.0)
        })
        
        # Moderate gain (0-10%)
        moderate_gain_prob = np.mean((price_changes > 0) & (price_changes <= 0.10))
        scenarios.append({
            'name': 'Moderate Gain',
            'description': 'Price increases by 0-10%',
            'probability': moderate_gain_prob,
            'price_range': f"${np.percentile(final_prices[(price_changes > 0) & (price_changes <= 0.10)], [25, 75])[0]:.0f} - ${np.percentile(final_prices[(price_changes > 0) & (price_changes <= 0.10)], [25, 75])[1]:.0f}" if moderate_gain_prob > 0 else "N/A",
            'confidence': min(moderate_gain_prob * 2, 1.0)
        })
        
        # Flat (-5% to 5%)
        flat_prob = np.mean((price_changes >= -0.05) & (price_changes <= 0.05))
        scenarios.append({
            'name': 'Flat Scenario',
            'description': 'Price changes by -5% to +5%',
            'probability': flat_prob,
            'price_range': f"${np.percentile(final_prices[(price_changes >= -0.05) & (price_changes <= 0.05)], [25, 75])[0]:.0f} - ${np.percentile(final_prices[(price_changes >= -0.05) & (price_changes <= 0.05)], [25, 75])[1]:.0f}" if flat_prob > 0 else "N/A",
            'confidence': min(flat_prob * 2, 1.0)
        })
        
        # Moderate loss (-10% to 0%)
        moderate_loss_prob = np.mean((price_changes >= -0.10) & (price_changes < 0))
        scenarios.append({
            'name': 'Moderate Loss',
            'description': 'Price decreases by 0-10%',
            'probability': moderate_loss_prob,
            'price_range': f"${np.percentile(final_prices[(price_changes >= -0.10) & (price_changes < 0)], [25, 75])[0]:.0f} - ${np.percentile(final_prices[(price_changes >= -0.10) & (price_changes < 0)], [25, 75])[1]:.0f}" if moderate_loss_prob > 0 else "N/A",
            'confidence': min(moderate_loss_prob * 2, 1.0)
        })
        
        # Bear scenario (<-10% loss)
        bear_prob = np.mean(price_changes < -0.10)
        scenarios.append({
            'name': 'Bear Scenario',
            'description': 'Price decreases by more than 10%',
            'probability': bear_prob,
            'price_range': f"${np.percentile(final_prices[price_changes < -0.10], [25, 75])[0]:.0f} - ${np.percentile(final_prices[price_changes < -0.10], [25, 75])[1]:.0f}" if bear_prob > 0 else "N/A",
            'confidence': min(bear_prob * 2, 1.0)
        })
        
        return scenarios
    
    async def analyze_sector_impact(self, 
                                  sector: str,
                                  event: str,
                                  time_horizon: str,
                                  market_context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze impact of specific events on sectors
        
        Args:
            sector: Sector to analyze (e.g., 'tech', 'financial', 'energy')
            event: Event type (e.g., 'fed_decision', 'earnings', 'inflation')
            time_horizon: Time horizon for analysis
            market_context: Current market context
            
        Returns:
            Sector impact analysis with scenarios
        """
        
        logger.info(f"Analyzing {event} impact on {sector} sector")
        
        # Map sectors to ETFs
        sector_etfs = {
            'tech': 'XLK',
            'financial': 'XLF', 
            'energy': 'XLE',
            'healthcare': 'XLV',
            'consumer': 'XLY',
            'industrial': 'XLI',
            'utilities': 'XLU',
            'materials': 'XLB',
            'real_estate': 'XLRE'
        }
        
        etf = sector_etfs.get(sector.lower(), 'SPY')
        
        # Get Monte Carlo analysis for the sector ETF
        mc_results = await self.generate_monte_carlo_scenarios(
            etf, time_horizon, market_context
        )
        
        # Generate event-specific scenarios
        event_scenarios = self._generate_event_scenarios(event, sector, mc_results)
        
        return {
            'sector': sector,
            'etf': etf,
            'event': event,
            'time_horizon': time_horizon,
            'monte_carlo_results': mc_results,
            'event_scenarios': event_scenarios,
            'sector_specific_factors': self._get_sector_factors(sector, event)
        }
    
    def _generate_event_scenarios(self, event: str, sector: str, 
                                mc_results: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate event-specific scenarios"""
        
        scenarios = []
        
        if event == 'fed_decision':
            scenarios = self._generate_fed_scenarios(sector, mc_results)
        elif event == 'earnings':
            scenarios = self._generate_earnings_scenarios(sector, mc_results)
        elif event == 'inflation':
            scenarios = self._generate_inflation_scenarios(sector, mc_results)
        else:
            scenarios = self._generate_generic_scenarios(sector, mc_results)
        
        return scenarios
    
    def _generate_fed_scenarios(self, sector: str, mc_results: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate Fed decision scenarios"""
        
        scenarios = []
        
        # Rate hike scenario
        scenarios.append({
            'name': 'Rate Hike',
            'probability': 0.35,
            'description': 'Fed raises rates by 0.25-0.5%',
            'sector_impact': self._get_sector_impact(sector, 'rate_hike'),
            'price_impact': 'Negative 2-8%',
            'key_drivers': [
                'Higher borrowing costs',
                'Reduced consumer spending',
                'Tighter financial conditions'
            ],
            'confidence': 0.75
        })
        
        # Rate hold scenario
        scenarios.append({
            'name': 'Rate Hold',
            'probability': 0.45,
            'description': 'Fed maintains current rates',
            'sector_impact': self._get_sector_impact(sector, 'rate_hold'),
            'price_impact': 'Neutral to positive 0-3%',
            'key_drivers': [
                'Stable financing environment',
                'Continued economic growth',
                'Supportive monetary policy'
            ],
            'confidence': 0.80
        })
        
        # Rate cut scenario
        scenarios.append({
            'name': 'Rate Cut',
            'probability': 0.20,
            'description': 'Fed cuts rates by 0.25-0.5%',
            'sector_impact': self._get_sector_impact(sector, 'rate_cut'),
            'price_impact': 'Positive 3-10%',
            'key_drivers': [
                'Lower borrowing costs',
                'Increased liquidity',
                'Stimulative monetary policy'
            ],
            'confidence': 0.70
        })
        
        return scenarios
    
    def _get_sector_impact(self, sector: str, scenario: str) -> str:
        """Get sector-specific impact description"""
        
        impacts = {
            'tech': {
                'rate_hike': 'Negative - Higher discount rates hurt growth stocks',
                'rate_hold': 'Neutral to positive - Stable environment supports growth',
                'rate_cut': 'Positive - Lower rates boost growth stock valuations'
            },
            'financial': {
                'rate_hike': 'Positive - Higher rates improve net interest margins',
                'rate_hold': 'Neutral - Stable rate environment',
                'rate_cut': 'Negative - Lower rates compress margins'
            },
            'energy': {
                'rate_hike': 'Mixed - Higher rates vs. economic growth',
                'rate_hold': 'Neutral - Stable economic environment',
                'rate_cut': 'Positive - Stimulative policy supports demand'
            },
            'real_estate': {
                'rate_hike': 'Negative - Higher mortgage rates reduce demand',
                'rate_hold': 'Neutral - Stable financing environment',
                'rate_cut': 'Positive - Lower rates boost property demand'
            }
        }
        
        return impacts.get(sector, {}).get(scenario, 'Moderate impact expected')
    
    def _generate_earnings_scenarios(self, sector: str, mc_results: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate earnings scenarios"""
        
        return [
            {
                'name': 'Earnings Beat',
                'probability': 0.30,
                'description': 'Sector companies beat earnings expectations',
                'sector_impact': 'Positive 5-15%',
                'confidence': 0.70
            },
            {
                'name': 'Earnings Meet',
                'probability': 0.50,
                'description': 'Sector companies meet earnings expectations',
                'sector_impact': 'Neutral 0-5%',
                'confidence': 0.75
            },
            {
                'name': 'Earnings Miss',
                'probability': 0.20,
                'description': 'Sector companies miss earnings expectations',
                'sector_impact': 'Negative -5 to -15%',
                'confidence': 0.70
            }
        ]
    
    def _generate_inflation_scenarios(self, sector: str, mc_results: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate inflation scenarios"""
        
        return [
            {
                'name': 'Inflation Surge',
                'probability': 0.25,
                'description': 'Inflation accelerates above 4%',
                'sector_impact': 'Mixed - depends on pricing power',
                'confidence': 0.65
            },
            {
                'name': 'Inflation Stable',
                'probability': 0.50,
                'description': 'Inflation remains around 2-3%',
                'sector_impact': 'Neutral - stable environment',
                'confidence': 0.80
            },
            {
                'name': 'Inflation Decline',
                'probability': 0.25,
                'description': 'Inflation falls below 2%',
                'sector_impact': 'Positive for growth sectors',
                'confidence': 0.70
            }
        ]
    
    def _generate_generic_scenarios(self, sector: str, mc_results: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate generic scenarios"""
        
        return [
            {
                'name': 'Positive Scenario',
                'probability': 0.40,
                'description': 'Favorable conditions for the sector',
                'sector_impact': 'Positive 3-8%',
                'confidence': 0.70
            },
            {
                'name': 'Neutral Scenario',
                'probability': 0.40,
                'description': 'Mixed conditions for the sector',
                'sector_impact': 'Neutral -2 to +2%',
                'confidence': 0.75
            },
            {
                'name': 'Negative Scenario',
                'probability': 0.20,
                'description': 'Challenging conditions for the sector',
                'sector_impact': 'Negative -5 to -10%',
                'confidence': 0.70
            }
        ]
    
    def _get_sector_factors(self, sector: str, event: str) -> List[str]:
        """Get sector-specific factors to consider"""
        
        factors = {
            'tech': [
                'Innovation cycles and product launches',
                'Regulatory environment and antitrust concerns',
                'Cybersecurity and data privacy issues',
                'International trade tensions',
                'Talent acquisition and retention costs'
            ],
            'financial': [
                'Interest rate environment',
                'Credit quality and loan losses',
                'Regulatory capital requirements',
                'Fintech disruption',
                'Economic cycle positioning'
            ],
            'energy': [
                'Oil and gas price volatility',
                'Renewable energy transition',
                'Environmental regulations',
                'Geopolitical tensions',
                'Infrastructure investments'
            ],
            'real_estate': [
                'Interest rate sensitivity',
                'Demographic trends',
                'Urbanization patterns',
                'Work-from-home impact',
                'Supply and demand dynamics'
            ]
        }
        
        return factors.get(sector, ['General market conditions', 'Economic growth', 'Interest rates'])
    
    async def generate_comprehensive_report(self, 
                                          question: str,
                                          parsed_question: Dict[str, Any],
                                          market_context: Dict[str, Any]) -> Dict[str, Any]:
        """Generate comprehensive research report"""
        
        report = {
            'question': question,
            'analysis_timestamp': datetime.now().isoformat(),
            'market_context': market_context,
            'monte_carlo_analysis': {},
            'sector_analysis': {},
            'scenarios': [],
            'recommendations': [],
            'confidence_score': 0.0
        }
        
        # Run Monte Carlo analysis for each asset
        for asset in parsed_question['assets']:
            mc_results = await self.generate_monte_carlo_scenarios(
                asset, 
                parsed_question['time_horizon'],
                market_context
            )
            report['monte_carlo_analysis'][asset] = mc_results
        
        # Run sector analysis for each sector
        for sector in parsed_question['sectors']:
            for event in parsed_question['events']:
                sector_results = await self.analyze_sector_impact(
                    sector,
                    event,
                    parsed_question['time_horizon'],
                    market_context
                )
                report['sector_analysis'][f"{sector}_{event}"] = sector_results
        
        # Combine all scenarios
        all_scenarios = []
        
        # Add Monte Carlo scenarios
        for asset, mc_results in report['monte_carlo_analysis'].items():
            all_scenarios.extend(mc_results['scenarios'])
        
        # Add sector scenarios
        for key, sector_results in report['sector_analysis'].items():
            all_scenarios.extend(sector_results['event_scenarios'])
        
        report['scenarios'] = all_scenarios
        
        # Calculate overall confidence
        if all_scenarios:
            confidences = [s.get('confidence', 0.5) for s in all_scenarios]
            report['confidence_score'] = np.mean(confidences)
        
        # Generate recommendations
        report['recommendations'] = self._generate_advanced_recommendations(
            all_scenarios, market_context, parsed_question
        )
        
        return report
    
    def _generate_advanced_recommendations(self, 
                                         scenarios: List[Dict[str, Any]],
                                         market_context: Dict[str, Any],
                                         parsed_question: Dict[str, Any]) -> List[str]:
        """Generate advanced recommendations based on analysis"""
        
        recommendations = []
        
        if not scenarios:
            return ["Insufficient data for recommendations"]
        
        # Get highest probability scenarios
        sorted_scenarios = sorted(scenarios, key=lambda x: x.get('probability', 0), reverse=True)
        top_scenarios = sorted_scenarios[:3]
        
        # Generate recommendations based on top scenarios
        for scenario in top_scenarios:
            prob = scenario.get('probability', 0)
            name = scenario.get('name', 'Unknown')
            
            if prob > 0.4:  # High probability scenario
                if 'bull' in name.lower() or 'positive' in name.lower():
                    recommendations.append(f"High probability ({prob:.1%}) of {name} - Consider increasing exposure")
                elif 'bear' in name.lower() or 'negative' in name.lower():
                    recommendations.append(f"High probability ({prob:.1%}) of {name} - Consider defensive positioning")
                else:
                    recommendations.append(f"High probability ({prob:.1%}) of {name} - Monitor closely")
        
        # Add regime-specific recommendations
        regime = market_context.get('regime', 'unknown')
        if regime == 'high_volatility':
            recommendations.append("High volatility environment - Consider volatility strategies and hedging")
        elif regime == 'low_volatility':
            recommendations.append("Low volatility environment - Look for volatility expansion opportunities")
        
        # Add time horizon specific recommendations
        time_horizon = parsed_question.get('time_horizon', '3 months')
        if '6 months' in time_horizon or '1 year' in time_horizon:
            recommendations.append("Longer time horizon - Focus on fundamental analysis and structural trends")
        else:
            recommendations.append("Short time horizon - Focus on technical analysis and momentum")
        
        return recommendations

# Example usage
async def main():
    """Example usage of the AdvancedResearchAgent"""
    
    config = {
        "universe": {
            "equities": ["SPY", "QQQ", "IWM", "VTI"],
            "sectors": ["XLE", "XLF", "XLK", "XLV"]
        }
    }
    
    agent = AdvancedResearchAgent(config)
    
    # Example: Analyze NFLX with Monte Carlo
    market_context = {
        'regime': 'bull_market',
        'regime_confidence': 0.75,
        'volatility_level': 18.0
    }
    
    print("🔬 Running Monte Carlo analysis for NFLX...")
    mc_results = await agent.generate_monte_carlo_scenarios(
        'NFLX', '3 months', market_context
    )
    
    print(f"Monte Carlo Results for NFLX:")
    print(f"Mean return: {mc_results['statistics']['mean_return']:.2%}")
    print(f"Volatility: {mc_results['statistics']['std_return']:.2%}")
    print(f"95% confidence interval: {mc_results['percentiles']['returns'][5]:.1%} to {mc_results['percentiles']['returns'][95]:.1%}")
    
    print("\nScenarios:")
    for scenario in mc_results['scenarios']:
        print(f"- {scenario['name']}: {scenario['probability']:.1%} probability")

if __name__ == "__main__":
    asyncio.run(main())

