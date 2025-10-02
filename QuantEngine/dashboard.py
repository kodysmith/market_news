#!/usr/bin/env python3
"""
Enhanced Market Scanner Web Dashboard
Real-time visualization of market analysis and trading opportunities
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import json
import os
import sys
from datetime import datetime, timedelta
from pathlib import Path
import asyncio
import yfinance as yf

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Configure Streamlit
st.set_page_config(
    page_title="Enhanced Market Scanner",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
    .opportunity-card {
        background-color: #ffffff;
        padding: 1rem;
        border-radius: 0.5rem;
        border: 1px solid #e0e0e0;
        margin-bottom: 1rem;
    }
    .bullish {
        color: #28a745;
        font-weight: bold;
    }
    .bearish {
        color: #dc3545;
        font-weight: bold;
    }
    .neutral {
        color: #6c757d;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

class EnhancedScannerDashboard:
    """Enhanced Market Scanner Dashboard"""
    
    def __init__(self):
        self.data_dir = Path("reports")
        self.data_dir.mkdir(exist_ok=True)
        
    def load_latest_data(self):
        """Load latest scanner data"""
        try:
            # Find latest scan file
            scan_files = list(self.data_dir.glob("enhanced_scan_*.json"))
            if not scan_files:
                return None
            
            latest_file = max(scan_files, key=os.path.getctime)
            
            with open(latest_file, 'r') as f:
                data = json.load(f)
            
            return data
            
        except Exception as e:
            st.error(f"Failed to load data: {e}")
            return None
    
    def load_llm_analysis(self):
        """Load latest LLM analysis"""
        try:
            # Find latest LLM analysis file
            llm_files = list(self.data_dir.glob("llm_analysis_*.json"))
            if not llm_files:
                return None
            
            latest_file = max(llm_files, key=os.path.getctime)
            
            with open(latest_file, 'r') as f:
                data = json.load(f)
            
            return data
            
        except Exception as e:
            st.warning(f"Failed to load LLM analysis: {e}")
            return None
    
    def create_market_overview(self, data):
        """Create market overview section"""
        st.markdown('<div class="main-header">📊 Enhanced Market Scanner Dashboard</div>', unsafe_allow_html=True)
        
        # Key metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(
                "Tickers Scanned",
                data.get('tickers_scanned', 0),
                delta=None
            )
        
        with col2:
            opportunities = data.get('opportunities', [])
            st.metric(
                "Opportunities Found",
                len(opportunities),
                delta=None
            )
        
        with col3:
            gpu_used = data.get('gpu_used', False)
            st.metric(
                "GPU Acceleration",
                "✅ Enabled" if gpu_used else "❌ Disabled",
                delta=None
            )
        
        with col4:
            llm_used = data.get('llm_used', False)
            st.metric(
                "LLM Analysis",
                "✅ Enabled" if llm_used else "❌ Disabled",
                delta=None
            )
        
        # Market sentiment
        if 'market_sentiment' in data:
            sentiment = data['market_sentiment']
            st.markdown("### 📈 Market Sentiment")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                sentiment_class = "bullish" if sentiment['sentiment'] == 'BULLISH' else "bearish" if sentiment['sentiment'] == 'BEARISH' else "neutral"
                st.markdown(f'<div class="metric-card {sentiment_class}">Market Sentiment: {sentiment["sentiment"]}</div>', unsafe_allow_html=True)
            
            with col2:
                st.metric("Confidence", f"{sentiment['confidence']:.1%}")
            
            with col3:
                st.metric("Buy Signals", sentiment.get('buy_signals', 0))
    
    def create_opportunities_table(self, data):
        """Create opportunities table"""
        st.markdown("### 🎯 Trading Opportunities")
        
        opportunities = data.get('opportunities', [])
        if not opportunities:
            st.info("No trading opportunities found")
            return
        
        # Create DataFrame
        df = pd.DataFrame(opportunities)
        
        # Format columns
        df['current_price'] = df['current_price'].apply(lambda x: f"${x:.2f}")
        df['target_price'] = df['target_price'].apply(lambda x: f"${x:.2f}")
        df['stop_loss'] = df['stop_loss'].apply(lambda x: f"${x:.2f}")
        df['confidence'] = df['confidence'].apply(lambda x: f"{x:.1%}")
        df['risk_reward'] = df['risk_reward'].apply(lambda x: f"{x:.2f}")
        df['rsi'] = df['rsi'].apply(lambda x: f"{x:.1f}")
        
        # Display table
        st.dataframe(
            df[['ticker', 'signal', 'confidence', 'current_price', 'target_price', 'stop_loss', 'risk_reward', 'rsi']],
            use_container_width=True
        )
    
    def create_technical_analysis_chart(self, data):
        """Create technical analysis chart"""
        st.markdown("### 📊 Technical Analysis")
        
        opportunities = data.get('opportunities', [])
        if not opportunities:
            st.info("No data available for technical analysis")
            return
        
        # Create scatter plot
        df = pd.DataFrame(opportunities)
        
        fig = go.Figure()
        
        # Add scatter points
        for signal in df['signal'].unique():
            signal_data = df[df['signal'] == signal]
            
            fig.add_trace(go.Scatter(
                x=signal_data['rsi'],
                y=signal_data['confidence'],
                mode='markers',
                name=signal,
                text=signal_data['ticker'],
                hovertemplate='<b>%{text}</b><br>RSI: %{x:.1f}<br>Confidence: %{y:.1%}<extra></extra>',
                marker=dict(size=10)
            ))
        
        fig.update_layout(
            title="RSI vs Confidence by Signal",
            xaxis_title="RSI",
            yaxis_title="Confidence",
            hovermode='closest'
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    def create_llm_analysis_section(self, llm_data):
        """Create LLM analysis section"""
        if not llm_data:
            return
        
        st.markdown("### 🤖 AI Market Analysis")
        
        # Market sentiment
        if 'market_sentiment' in llm_data and 'sentiment' in llm_data['market_sentiment']:
            sentiment = llm_data['market_sentiment']['sentiment']
            st.markdown(f"**Market Sentiment:** {sentiment['sentiment']} ({sentiment['confidence']:.1%})")
        
        # Stock opportunities
        if 'stock_opportunities' in llm_data:
            st.markdown("#### Stock Analysis")
            
            for analysis in llm_data['stock_opportunities']:
                if 'opportunity' in analysis:
                    opp = analysis['opportunity']
                    with st.expander(f"{opp['ticker']} - {opp['signal']} ({opp['confidence']:.1%})"):
                        st.write(analysis['analysis'])
        
        # Sector rotation
        if 'sector_rotation' in llm_data and 'rotation' in llm_data['sector_rotation']:
            rotation = llm_data['sector_rotation']['rotation']
            st.markdown("#### Sector Rotation")
            st.write(f"**Leading Sectors:** {rotation['leading_sectors']}")
            st.write(f"**Lagging Sectors:** {rotation['lagging_sectors']}")
    
    def create_performance_metrics(self, data):
        """Create performance metrics section"""
        st.markdown("### 📈 Performance Metrics")
        
        opportunities = data.get('opportunities', [])
        if not opportunities:
            st.info("No performance data available")
            return
        
        # Calculate metrics
        df = pd.DataFrame(opportunities)
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            avg_confidence = df['confidence'].mean()
            st.metric("Average Confidence", f"{avg_confidence:.1%}")
        
        with col2:
            avg_risk_reward = df['risk_reward'].mean()
            st.metric("Average Risk/Reward", f"{avg_risk_reward:.2f}")
        
        with col3:
            signal_distribution = df['signal'].value_counts()
            most_common_signal = signal_distribution.index[0]
            st.metric("Most Common Signal", most_common_signal)
    
    def create_real_time_data(self):
        """Create real-time data section"""
        st.markdown("### ⏰ Real-Time Market Data")
        
        # Get current market data
        try:
            # Major indices
            indices = ['^GSPC', '^IXIC', '^VIX', '^IWM']
            data = {}
            
            for ticker in indices:
                stock = yf.Ticker(ticker)
                info = stock.info
                data[ticker] = {
                    'price': info.get('regularMarketPrice', 0),
                    'change': info.get('regularMarketChange', 0),
                    'change_pct': info.get('regularMarketChangePercent', 0)
                }
            
            # Display indices
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                sp500 = data['^GSPC']
                st.metric("S&P 500", f"${sp500['price']:.2f}", f"{sp500['change_pct']:+.2f}%")
            
            with col2:
                nasdaq = data['^IXIC']
                st.metric("Nasdaq", f"${nasdaq['price']:.2f}", f"{nasdaq['change_pct']:+.2f}%")
            
            with col3:
                vix = data['^VIX']
                st.metric("VIX", f"{vix['price']:.2f}", f"{vix['change_pct']:+.2f}%")
            
            with col4:
                russell = data['^IWM']
                st.metric("Russell 2000", f"${russell['price']:.2f}", f"{russell['change_pct']:+.2f}%")
                
        except Exception as e:
            st.error(f"Failed to load real-time data: {e}")
    
    def run_dashboard(self):
        """Run the dashboard"""
        # Load data
        data = self.load_latest_data()
        llm_data = self.load_llm_analysis()
        
        if data is None:
            st.error("No scanner data available. Please run the scanner first.")
            return
        
        # Create sections
        self.create_market_overview(data)
        st.markdown("---")
        
        self.create_real_time_data()
        st.markdown("---")
        
        self.create_opportunities_table(data)
        st.markdown("---")
        
        self.create_technical_analysis_chart(data)
        st.markdown("---")
        
        self.create_performance_metrics(data)
        st.markdown("---")
        
        self.create_llm_analysis_section(llm_data)
        
        # Auto-refresh
        if st.button("🔄 Refresh Data"):
            st.rerun()

def main():
    """Main function"""
    dashboard = EnhancedScannerDashboard()
    dashboard.run_dashboard()

if __name__ == "__main__":
    main()
