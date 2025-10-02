#!/usr/bin/env python3
"""
Backtesting Framework for Scanner Confidence Calibration
Tests historical performance of RSI-based signals
"""

import yfinance as yf
import pandas as pd
import numpy as np
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Tuple
import talib

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ScannerBacktester:
    def __init__(self):
        self.results = {}
        
    def backtest_rsi_signals(self, ticker: str, start_date: str = "2020-01-01", 
                            end_date: str = None) -> Dict:
        """Backtest RSI signals for a single ticker"""
        try:
            logger.info(f"🔍 Backtesting {ticker} from {start_date}")
            
            # Get historical data
            stock = yf.Ticker(ticker)
            hist = stock.history(start=start_date, end=end_date)
            
            if hist.empty:
                return {'error': 'No historical data available'}
            
            # Calculate RSI
            rsi = talib.RSI(hist['Close'].values, timeperiod=14)
            rsi_series = pd.Series(rsi, index=hist.index)
            
            # Generate signals
            signals = self._generate_signals(rsi_series, hist['Close'])
            
            # Calculate performance metrics
            performance = self._calculate_performance(signals, hist)
            
            # Analyze signal accuracy
            accuracy = self._analyze_signal_accuracy(signals, hist)
            
            return {
                'ticker': ticker,
                'period': f"{start_date} to {end_date or 'present'}",
                'total_days': len(hist),
                'signals_generated': len(signals),
                'performance': performance,
                'accuracy': accuracy,
                'rsi_stats': self._calculate_rsi_stats(rsi_series)
            }
            
        except Exception as e:
            logger.error(f"❌ Backtest failed for {ticker}: {e}")
            return {'error': str(e)}
    
    def _generate_signals(self, rsi: pd.Series, prices: pd.Series) -> List[Dict]:
        """Generate buy/sell signals based on RSI"""
        signals = []
        
        for i in range(1, len(rsi)):
            if pd.isna(rsi.iloc[i]):
                continue
                
            current_rsi = rsi.iloc[i]
            current_price = prices.iloc[i]
            date = rsi.index[i]
            
            signal = None
            signal_type = None
            
            # Buy signals
            if current_rsi < 30 and rsi.iloc[i-1] >= 30:  # RSI crosses below 30
                signal = 'BUY'
                signal_type = 'OVERSOLD_ENTRY'
            elif current_rsi < 20:  # Extreme oversold
                signal = 'BUY'
                signal_type = 'EXTREME_OVERSOLD'
            
            # Sell signals
            elif current_rsi > 70 and rsi.iloc[i-1] <= 70:  # RSI crosses above 70
                signal = 'SELL'
                signal_type = 'OVERBOUGHT_ENTRY'
            elif current_rsi > 80:  # Extreme overbought
                signal = 'SELL'
                signal_type = 'EXTREME_OVERBOUGHT'
            
            if signal:
                signals.append({
                    'date': date,
                    'signal': signal,
                    'signal_type': signal_type,
                    'rsi': current_rsi,
                    'price': current_price,
                    'index': i
                })
        
        return signals
    
    def _calculate_performance(self, signals: List[Dict], hist: pd.DataFrame) -> Dict:
        """Calculate performance metrics for signals"""
        if not signals:
            return {'total_return': 0, 'win_rate': 0, 'avg_return': 0}
        
        returns = []
        wins = 0
        
        for i, signal in enumerate(signals):
            if signal['signal'] == 'BUY':
                # Find next sell signal or end of data
                next_sell = None
                for j in range(i + 1, len(signals)):
                    if signals[j]['signal'] == 'SELL':
                        next_sell = signals[j]
                        break
                
                if next_sell:
                    entry_price = signal['price']
                    exit_price = next_sell['price']
                    return_pct = (exit_price - entry_price) / entry_price
                    returns.append(return_pct)
                    
                    if return_pct > 0:
                        wins += 1
                else:
                    # No exit signal, use last available price
                    entry_price = signal['price']
                    exit_price = hist['Close'].iloc[-1]
                    return_pct = (exit_price - entry_price) / entry_price
                    returns.append(return_pct)
                    
                    if return_pct > 0:
                        wins += 1
            
            elif signal['signal'] == 'SELL':
                # Find next buy signal or end of data
                next_buy = None
                for j in range(i + 1, len(signals)):
                    if signals[j]['signal'] == 'BUY':
                        next_buy = signals[j]
                        break
                
                if next_buy:
                    entry_price = signal['price']
                    exit_price = next_buy['price']
                    return_pct = (entry_price - exit_price) / entry_price  # Short position
                    returns.append(return_pct)
                    
                    if return_pct > 0:
                        wins += 1
                else:
                    # No exit signal, use last available price
                    entry_price = signal['price']
                    exit_price = hist['Close'].iloc[-1]
                    return_pct = (entry_price - exit_price) / entry_price
                    returns.append(return_pct)
                    
                    if return_pct > 0:
                        wins += 1
        
        if not returns:
            return {'total_return': 0, 'win_rate': 0, 'avg_return': 0}
        
        total_return = sum(returns)
        win_rate = wins / len(returns) * 100
        avg_return = np.mean(returns) * 100
        
        return {
            'total_return': round(total_return * 100, 2),
            'win_rate': round(win_rate, 2),
            'avg_return': round(avg_return, 2),
            'total_trades': len(returns),
            'winning_trades': wins,
            'losing_trades': len(returns) - wins
        }
    
    def _analyze_signal_accuracy(self, signals: List[Dict], hist: pd.DataFrame) -> Dict:
        """Analyze accuracy of different signal types"""
        signal_types = {}
        
        for signal in signals:
            signal_type = signal['signal_type']
            if signal_type not in signal_types:
                signal_types[signal_type] = {'total': 0, 'profitable': 0}
            
            signal_types[signal_type]['total'] += 1
            
            # Check if signal was profitable
            entry_price = signal['price']
            entry_date = signal['date']
            
            # Look ahead 5, 10, 20 days to see if price moved in expected direction
            for days_ahead in [5, 10, 20]:
                try:
                    future_date = entry_date + timedelta(days=days_ahead)
                    future_prices = hist[hist.index >= future_date]
                    
                    if not future_prices.empty:
                        future_price = future_prices['Close'].iloc[0]
                        
                        if signal['signal'] == 'BUY':
                            profitable = future_price > entry_price
                        else:  # SELL
                            profitable = future_price < entry_price
                        
                        if profitable:
                            signal_types[signal_type]['profitable'] += 1
                            break  # Count as profitable if any time horizon works
                            
                except Exception:
                    continue
        
        # Calculate accuracy for each signal type
        accuracy = {}
        for signal_type, data in signal_types.items():
            if data['total'] > 0:
                accuracy[signal_type] = {
                    'accuracy': round(data['profitable'] / data['total'] * 100, 2),
                    'total_signals': data['total'],
                    'profitable_signals': data['profitable']
                }
        
        return accuracy
    
    def _calculate_rsi_stats(self, rsi: pd.Series) -> Dict:
        """Calculate RSI statistics"""
        valid_rsi = rsi.dropna()
        
        return {
            'mean': round(valid_rsi.mean(), 2),
            'std': round(valid_rsi.std(), 2),
            'min': round(valid_rsi.min(), 2),
            'max': round(valid_rsi.max(), 2),
            'overbought_days': len(valid_rsi[valid_rsi > 70]),
            'oversold_days': len(valid_rsi[valid_rsi < 30]),
            'extreme_overbought_days': len(valid_rsi[valid_rsi > 80]),
            'extreme_oversold_days': len(valid_rsi[valid_rsi < 20])
        }
    
    def calibrate_confidence_scores(self, tickers: List[str]) -> Dict:
        """Calibrate confidence scores based on historical performance"""
        logger.info(f"🎯 Calibrating confidence scores for {len(tickers)} tickers...")
        
        calibration_results = {}
        
        for ticker in tickers:
            try:
                backtest_result = self.backtest_rsi_signals(ticker)
                
                if 'error' not in backtest_result:
                    calibration_results[ticker] = backtest_result
                    
                    # Calculate confidence calibration
                    win_rate = backtest_result['performance']['win_rate']
                    avg_return = backtest_result['performance']['avg_return']
                    
                    # Base confidence on historical performance
                    base_confidence = min(100, max(0, win_rate + (avg_return * 0.5)))
                    
                    calibration_results[ticker]['calibrated_confidence'] = round(base_confidence, 1)
                    
            except Exception as e:
                logger.error(f"❌ Calibration failed for {ticker}: {e}")
        
        # Calculate overall calibration metrics
        overall_metrics = self._calculate_overall_metrics(calibration_results)
        
        return {
            'calibration_results': calibration_results,
            'overall_metrics': overall_metrics,
            'calibration_timestamp': datetime.now().isoformat()
        }
    
    def _calculate_overall_metrics(self, results: Dict) -> Dict:
        """Calculate overall calibration metrics"""
        if not results:
            return {}
        
        all_win_rates = [r['performance']['win_rate'] for r in results.values()]
        all_avg_returns = [r['performance']['avg_return'] for r in results.values()]
        all_confidences = [r['calibrated_confidence'] for r in results.values()]
        
        return {
            'avg_win_rate': round(np.mean(all_win_rates), 2),
            'avg_return': round(np.mean(all_avg_returns), 2),
            'avg_confidence': round(np.mean(all_confidences), 2),
            'total_tickers': len(results),
            'best_performer': max(results.keys(), key=lambda x: results[x]['performance']['win_rate']),
            'worst_performer': min(results.keys(), key=lambda x: results[x]['performance']['win_rate'])
        }

def main():
    """Test the backtesting framework"""
    backtester = ScannerBacktester()
    
    # Test with a few tickers
    test_tickers = ['AAPL', 'MSFT', 'GOOGL', 'TSLA', 'NVDA']
    
    print("🚀 Running Scanner Backtesting...")
    print("=" * 60)
    
    # Calibrate confidence scores
    calibration = backtester.calibrate_confidence_scores(test_tickers)
    
    print(f"\n📊 CALIBRATION RESULTS")
    print(f"Total Tickers: {calibration['overall_metrics']['total_tickers']}")
    print(f"Average Win Rate: {calibration['overall_metrics']['avg_win_rate']}%")
    print(f"Average Return: {calibration['overall_metrics']['avg_return']}%")
    print(f"Average Confidence: {calibration['overall_metrics']['avg_confidence']}%")
    
    print(f"\n🎯 INDIVIDUAL RESULTS:")
    for ticker, result in calibration['calibration_results'].items():
        perf = result['performance']
        print(f"  {ticker:4} | Win Rate: {perf['win_rate']:5.1f}% | "
              f"Avg Return: {perf['avg_return']:6.2f}% | "
              f"Trades: {perf['total_trades']:3} | "
              f"Confidence: {result['calibrated_confidence']:5.1f}%")
    
    # Save calibration results
    with open('scanner_calibration.json', 'w') as f:
        json.dump(calibration, f, indent=2, default=str)
    
    print(f"\n💾 Calibration results saved to scanner_calibration.json")

if __name__ == "__main__":
    main()
