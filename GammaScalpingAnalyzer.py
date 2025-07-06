import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json
from scipy import stats

class GammaScalpingAnalyzer:
    def __init__(self):
        self.risk_free_rate = self._get_risk_free_rate()
        self.config = self._load_config()
        
    def _load_config(self):
        """Load configuration from config.json"""
        try:
            with open('config.json', 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            print("Config file not found, using default values")
            return {
                'GAMMA_SCALPING_THRESHOLD': 60.0,
                'PREMIUM_SELLING_THRESHOLD': 30.0,
                'IV_RV_RATIO_LOW': 0.8,
                'IV_RV_RATIO_HIGH': 1.2,
                'VIX_LOW_PERCENTILE': 20.0,
                'VIX_HIGH_PERCENTILE': 80.0,
                'VOLATILITY_ACCELERATION_HIGH': 1.3,
                'VOLATILITY_ACCELERATION_LOW': 0.7
            }
        
    def _get_risk_free_rate(self):
        """Get current risk-free rate from 10-year Treasury"""
        try:
            tnx = yf.Ticker('^TNX').history(period='1d')['Close'].iloc[-1]
            return tnx / 100
        except:
            return 0.02  # Default 2%
    
    def calculate_realized_volatility(self, ticker, days=30):
        """Calculate realized volatility over specified period"""
        try:
            end_date = datetime.now()
            start_date = end_date - timedelta(days=days)
            
            data = yf.download(ticker, start=start_date, end=end_date)
            if data.empty:
                return None
                
            # Calculate daily returns
            returns = np.log(data['Close'] / data['Close'].shift(1)).dropna()
            
            # Annualized realized volatility
            realized_vol = returns.std() * np.sqrt(252)
            return realized_vol
            
        except Exception as e:
            print(f"Error calculating realized volatility for {ticker}: {e}")
            return None
    
    def get_implied_volatility_data(self, ticker):
        """Get implied volatility from options chains"""
        try:
            stock = yf.Ticker(ticker)
            current_price = stock.history(period='1d')['Close'].iloc[-1]
            expirations = stock.options[:2]  # Next 2 expirations
            
            iv_data = []
            
            for expiry in expirations:
                chain = stock.option_chain(expiry)
                
                # Find ATM options
                atm_calls = chain.calls.iloc[(chain.calls['strike'] - current_price).abs().argsort()[:3]]
                atm_puts = chain.puts.iloc[(chain.puts['strike'] - current_price).abs().argsort()[:3]]
                
                # Average IV for ATM options
                call_iv = atm_calls['impliedVolatility'].mean()
                put_iv = atm_puts['impliedVolatility'].mean()
                avg_iv = (call_iv + put_iv) / 2
                
                days_to_expiry = (datetime.strptime(expiry, '%Y-%m-%d').date() - datetime.now().date()).days
                
                iv_data.append({
                    'expiry': expiry,
                    'days_to_expiry': days_to_expiry,
                    'implied_volatility': avg_iv,
                    'call_iv': call_iv,
                    'put_iv': put_iv
                })
            
            return iv_data
            
        except Exception as e:
            print(f"Error getting IV data for {ticker}: {e}")
            return []
    
    def calculate_vix_percentile(self, days=252):
        """Calculate current VIX percentile over specified period"""
        try:
            end_date = datetime.now()
            start_date = end_date - timedelta(days=days)
            
            vix_data = yf.download('^VIX', start=start_date, end=end_date)
            if vix_data.empty:
                return None
                
            current_vix = vix_data['Close'].iloc[-1]
            vix_percentile = stats.percentileofscore(vix_data['Close'], current_vix)
            
            return {
                'current_vix': current_vix,
                'vix_percentile': vix_percentile,
                'vix_mean': vix_data['Close'].mean(),
                'vix_std': vix_data['Close'].std()
            }
            
        except Exception as e:
            print(f"Error calculating VIX percentile: {e}")
            return None
    
    def analyze_volatility_regime(self, ticker):
        """Analyze volatility regime for a specific ticker"""
        try:
            # Get realized volatility (30-day)
            realized_vol_30d = self.calculate_realized_volatility(ticker, 30)
            realized_vol_10d = self.calculate_realized_volatility(ticker, 10)
            
            if realized_vol_30d is None or realized_vol_10d is None:
                return None
            
            # Get implied volatility
            iv_data = self.get_implied_volatility_data(ticker)
            if not iv_data:
                return None
            
            # Use near-term IV (closest expiration)
            near_term_iv = iv_data[0]['implied_volatility']
            
            # Calculate ratios
            iv_rv_ratio = near_term_iv / realized_vol_30d if realized_vol_30d > 0 else 0
            rv_acceleration = realized_vol_10d / realized_vol_30d if realized_vol_30d > 0 else 0
            
            analysis = {
                'ticker': ticker,
                'realized_vol_30d': realized_vol_30d,
                'realized_vol_10d': realized_vol_10d,
                'implied_volatility': near_term_iv,
                'iv_rv_ratio': iv_rv_ratio,
                'rv_acceleration': rv_acceleration,
                'iv_data': iv_data
            }
            
            return analysis
            
        except Exception as e:
            print(f"Error analyzing volatility regime for {ticker}: {e}")
            return None
    
    def determine_gamma_scalping_opportunity(self, ticker):
        """Determine if conditions favor gamma scalping vs premium selling"""
        
        vol_analysis = self.analyze_volatility_regime(ticker)
        if not vol_analysis:
            return None
        
        vix_analysis = self.calculate_vix_percentile()
        if not vix_analysis:
            return None
        
        # Decision criteria using config parameters
        iv_rv_ratio = vol_analysis['iv_rv_ratio']
        vix_percentile = vix_analysis['vix_percentile']
        rv_acceleration = vol_analysis['rv_acceleration']
        current_vix = vix_analysis['current_vix']
        
        # Scoring system (0-100, higher = better for gamma scalping)
        gamma_score = 0
        reasons = []
        
        # Factor 1: IV vs RV (40% weight)
        if iv_rv_ratio < self.config['IV_RV_RATIO_LOW']:  # IV significantly below RV
            gamma_score += 40
            reasons.append(f"IV/RV ratio low ({iv_rv_ratio:.2f}) - options cheap")
        elif iv_rv_ratio > self.config['IV_RV_RATIO_HIGH']:  # IV significantly above RV
            gamma_score -= 20
            reasons.append(f"IV/RV ratio high ({iv_rv_ratio:.2f}) - options expensive")
        else:
            gamma_score += 10
            reasons.append(f"IV/RV ratio neutral ({iv_rv_ratio:.2f})")
        
        # Factor 2: VIX level and percentile (30% weight)
        if vix_percentile < self.config['VIX_LOW_PERCENTILE'] and current_vix < 20:  # Low VIX, potential for vol expansion
            gamma_score += 30
            reasons.append(f"VIX low ({current_vix:.1f}, {vix_percentile:.0f}th percentile) - vol expansion likely")
        elif vix_percentile > self.config['VIX_HIGH_PERCENTILE']:  # High VIX, potential for vol contraction
            gamma_score -= 15
            reasons.append(f"VIX high ({current_vix:.1f}, {vix_percentile:.0f}th percentile) - vol contraction likely")
        else:
            gamma_score += 5
            reasons.append(f"VIX neutral ({current_vix:.1f}, {vix_percentile:.0f}th percentile)")
        
        # Factor 3: Recent volatility acceleration (20% weight)
        if rv_acceleration > self.config['VOLATILITY_ACCELERATION_HIGH']:  # Recent vol pickup
            gamma_score += 20
            reasons.append(f"Recent vol acceleration ({rv_acceleration:.2f}) - momentum")
        elif rv_acceleration < self.config['VOLATILITY_ACCELERATION_LOW']:  # Recent vol decline
            gamma_score -= 10
            reasons.append(f"Recent vol deceleration ({rv_acceleration:.2f}) - declining momentum")
        
        # Factor 4: Term structure (10% weight)
        if len(vol_analysis['iv_data']) > 1:
            front_iv = vol_analysis['iv_data'][0]['implied_volatility']
            back_iv = vol_analysis['iv_data'][1]['implied_volatility']
            term_structure = back_iv / front_iv if front_iv > 0 else 1
            
            if term_structure > 1.1:  # Steep contango
                gamma_score += 10
                reasons.append(f"IV term structure steep ({term_structure:.2f}) - front month cheap")
            elif term_structure < 0.9:  # Backwardation
                gamma_score -= 5
                reasons.append(f"IV term structure inverted ({term_structure:.2f}) - front month expensive")
        
        # Determine recommendation using config thresholds
        if gamma_score >= self.config['GAMMA_SCALPING_THRESHOLD']:
            recommendation = "GAMMA_SCALPING"
            strategy = "Buy straddles/strangles and delta hedge"
        elif gamma_score <= self.config['PREMIUM_SELLING_THRESHOLD']:
            recommendation = "PREMIUM_SELLING"
            strategy = "Sell straddles/strangles to collect premium"
        else:
            recommendation = "NEUTRAL"
            strategy = "Mixed approach or wait for better opportunity"
        
        return {
            'ticker': ticker,
            'recommendation': recommendation,
            'strategy': strategy,
            'gamma_score': gamma_score,
            'reasons': reasons,
            'analysis': {
                'iv_rv_ratio': iv_rv_ratio,
                'vix_percentile': vix_percentile,
                'rv_acceleration': rv_acceleration,
                'current_vix': current_vix,
                'realized_vol_30d': vol_analysis['realized_vol_30d'],
                'implied_volatility': vol_analysis['implied_volatility']
            }
        }
    
    def get_market_gamma_scalping_analysis(self, tickers):
        """Analyze gamma scalping opportunities across multiple tickers"""
        results = []
        
        for ticker in tickers:
            analysis = self.determine_gamma_scalping_opportunity(ticker)
            if analysis:
                results.append(analysis)
        
        # Market-wide summary
        if results:
            gamma_scores = [r['gamma_score'] for r in results]
            avg_gamma_score = np.mean(gamma_scores)
            
            gamma_scalping_count = len([r for r in results if r['recommendation'] == 'GAMMA_SCALPING'])
            premium_selling_count = len([r for r in results if r['recommendation'] == 'PREMIUM_SELLING'])
            
            # Overall market recommendation using config thresholds
            if avg_gamma_score >= self.config['GAMMA_SCALPING_THRESHOLD']:
                market_recommendation = "GAMMA_SCALPING_FAVORED"
            elif avg_gamma_score <= self.config['PREMIUM_SELLING_THRESHOLD']:
                market_recommendation = "PREMIUM_SELLING_FAVORED"
            else:
                market_recommendation = "MIXED_CONDITIONS"
            
            return {
                'market_recommendation': market_recommendation,
                'avg_gamma_score': avg_gamma_score,
                'gamma_scalping_count': gamma_scalping_count,
                'premium_selling_count': premium_selling_count,
                'total_analyzed': len(results),
                'individual_analysis': results
            }
        
        return None

def analyze_gamma_scalping_opportunities():
    """Main function to run gamma scalping analysis"""
    analyzer = GammaScalpingAnalyzer()
    
    # Get tickers from config or use default
    tickers = analyzer.config.get('GAMMA_SCALPING_TICKERS', 
                                 ['SPY', 'QQQ', 'IWM', 'AAPL', 'TSLA', 'NVDA', 'MSFT', 'GOOGL'])
    
    print("Analyzing gamma scalping opportunities...")
    analysis = analyzer.get_market_gamma_scalping_analysis(tickers)
    
    if analysis:
        print(f"\n--- GAMMA SCALPING ANALYSIS ---")
        print(f"Market Recommendation: {analysis['market_recommendation']}")
        print(f"Average Gamma Score: {analysis['avg_gamma_score']:.1f}/100")
        print(f"Gamma Scalping Opportunities: {analysis['gamma_scalping_count']}")
        print(f"Premium Selling Opportunities: {analysis['premium_selling_count']}")
        
        print(f"\n--- INDIVIDUAL TICKER ANALYSIS ---")
        for ticker_analysis in analysis['individual_analysis']:
            print(f"\n{ticker_analysis['ticker']}: {ticker_analysis['recommendation']} (Score: {ticker_analysis['gamma_score']:.0f})")
            print(f"  Strategy: {ticker_analysis['strategy']}")
            for reason in ticker_analysis['reasons'][:2]:  # Show top 2 reasons
                print(f"  • {reason}")
    
    return analysis

if __name__ == "__main__":
    result = analyze_gamma_scalping_opportunities() 