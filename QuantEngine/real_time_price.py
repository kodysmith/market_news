#!/usr/bin/env python3
"""
Real-time Price Checker

Quick script to get real-time prices and key levels for any stock.
"""

import sys
import yfinance as yf
from datetime import datetime
import pandas as pd
import numpy as np

def get_real_time_analysis(ticker: str):
    """Get real-time analysis for a ticker"""
    try:
        print(f"🔍 Fetching real-time data for {ticker}...")
        
        # Get real-time data
        stock = yf.Ticker(ticker)
        info = stock.info
        
        # Get current price
        current_price = info.get('regularMarketPrice') or info.get('previousClose')
        if not current_price:
            print(f"❌ Could not get current price for {ticker}")
            return
        
        # Get historical data for technical analysis
        hist = stock.history(period="3mo")
        if hist.empty:
            print(f"❌ No historical data for {ticker}")
            return
        
        # Calculate technical indicators
        sma_20 = float(hist['Close'].rolling(20).mean().iloc[-1])
        sma_50 = float(hist['Close'].rolling(50).mean().iloc[-1])
        
        # Calculate RSI
        delta = hist['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        current_rsi = float(rsi.iloc[-1]) if not pd.isna(rsi.iloc[-1]) else 50
        
        # Recent high/low
        recent_high = float(hist['High'].rolling(20).max().iloc[-1])
        recent_low = float(hist['Low'].rolling(20).min().iloc[-1])
        
        # Daily change
        prev_close = float(hist['Close'].iloc[-2])
        daily_change = ((current_price - prev_close) / prev_close) * 100
        
        print(f"\n📊 {ticker} REAL-TIME ANALYSIS")
        print("=" * 50)
        print(f"💰 Current Price: ${current_price:.2f}")
        print(f"📈 Daily Change: {daily_change:+.2f}%")
        print(f"📊 RSI: {current_rsi:.1f}")
        print(f"📈 20-day SMA: ${sma_20:.2f}")
        print(f"📈 50-day SMA: ${sma_50:.2f}")
        print(f"🔝 Recent High: ${recent_high:.2f}")
        print(f"🔻 Recent Low: ${recent_low:.2f}")
        
        # Key level analysis
        print(f"\n🎯 KEY LEVELS:")
        if current_price > sma_20 > sma_50:
            print(f"✅ BULLISH: Above both SMAs (uptrend)")
        elif current_price < sma_20 < sma_50:
            print(f"❌ BEARISH: Below both SMAs (downtrend)")
        elif current_price > sma_20 and current_price < sma_50:
            print(f"⚠️ MIXED: Above 20-day but below 50-day SMA")
        elif current_price < sma_20 and current_price > sma_50:
            print(f"⚠️ MIXED: Below 20-day but above 50-day SMA")
        
        # RSI analysis
        if current_rsi > 80:
            print(f"🔥 EXTREME OVERBOUGHT: RSI {current_rsi:.1f} - Strong SELL signal")
        elif current_rsi > 70:
            print(f"📈 OVERBOUGHT: RSI {current_rsi:.1f} - SELL signal")
        elif current_rsi < 20:
            print(f"❄️ EXTREME OVERSOLD: RSI {current_rsi:.1f} - Strong BUY signal")
        elif current_rsi < 30:
            print(f"📉 OVERSOLD: RSI {current_rsi:.1f} - BUY signal")
        else:
            print(f"⚪ NEUTRAL: RSI {current_rsi:.1f} - No clear signal")
        
        # Support/Resistance
        if current_price > recent_high * 0.98:
            print(f"🚧 NEAR RESISTANCE: Approaching recent high of ${recent_high:.2f}")
        elif current_price < recent_low * 1.02:
            print(f"🛡️ NEAR SUPPORT: Approaching recent low of ${recent_low:.2f}")
        
        # Trading signal
        if current_rsi > 70 and current_price > sma_20:
            signal = "SELL"
            target = current_price * 0.95
            stop = current_price * 1.05
        elif current_rsi < 30 and current_price < sma_20:
            signal = "BUY"
            target = current_price * 1.05
            stop = current_price * 0.95
        else:
            signal = "HOLD"
            target = current_price * 1.02
            stop = current_price * 0.98
        
        print(f"\n🎯 TRADING SIGNAL: {signal}")
        print(f"🎯 Target Price: ${target:.2f}")
        print(f"🛡️ Stop Loss: ${stop:.2f}")
        
        # Market context
        print(f"\n📊 MARKET CONTEXT:")
        print(f"- {ticker} is {'ABOVE' if current_price > sma_20 else 'BELOW'} the 20-day SMA")
        print(f"- {ticker} is {'ABOVE' if current_price > sma_50 else 'BELOW'} the 50-day SMA")
        print(f"- RSI indicates {'overbought' if current_rsi > 70 else 'oversold' if current_rsi < 30 else 'neutral'} conditions")
        
        if current_price > recent_high * 0.98:
            print(f"- ⚠️ WARNING: {ticker} is near resistance at ${recent_high:.2f}")
        elif current_price < recent_low * 1.02:
            print(f"- ⚠️ WARNING: {ticker} is near support at ${recent_low:.2f}")
        
        print(f"\n⏰ Last Updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
    except Exception as e:
        print(f"❌ Error analyzing {ticker}: {e}")

def main():
    """Main entry point"""
    if len(sys.argv) < 2:
        print("Usage: python3 real_time_price.py <TICKER>")
        print("Example: python3 real_time_price.py NVDA")
        return
    
    ticker = sys.argv[1].upper()
    get_real_time_analysis(ticker)

if __name__ == "__main__":
    main()

