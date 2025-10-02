#!/usr/bin/env python3
"""
Enhanced Market Scanner Dashboard with AI Chat Integration
Real-time visualization of market analysis and trading opportunities with conversational AI
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
import asyncio
import yfinance as yf
from datetime import datetime, timedelta
from pathlib import Path
import requests
import time

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import chat interfaces
try:
    from sector_research_chat import SectorResearchChat
    from group_analysis_chat import GroupAnalysisChat
    CHAT_AVAILABLE = True
except ImportError as e:
    st.warning(f"Chat interfaces not available: {e}")
    CHAT_AVAILABLE = False

# Configure Streamlit
st.set_page_config(
    page_title="Enhanced Market Scanner with AI Chat",
    page_icon="🤖",
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
    .chat-header {
        font-size: 1.8rem;
        font-weight: bold;
        color: #2e8b57;
        margin-bottom: 1rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
    .chat-message {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
        border-left: 4px solid #2e8b57;
    }
    .user-message {
        background-color: #e3f2fd;
        border-left-color: #2196f3;
    }
    .assistant-message {
        background-color: #f1f8e9;
        border-left-color: #4caf50;
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
    .stTabs [data-baseweb="tab-list"] {
        gap: 2px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        white-space: pre-wrap;
        background-color: #f0f2f6;
        border-radius: 4px 4px 0px 0px;
        gap: 1px;
        padding-left: 20px;
        padding-right: 20px;
    }
    .stTabs [aria-selected="true"] {
        background-color: #1f77b4;
        color: white;
    }
</style>
""", unsafe_allow_html=True)

class EnhancedDashboardWithChat:
    """Enhanced Market Scanner Dashboard with AI Chat Integration"""
    
    def __init__(self):
        self.data_dir = Path("reports")
        self.data_dir.mkdir(exist_ok=True)
        
        # Initialize chat interfaces
        self.sector_chat = None
        self.group_chat = None
        
        if CHAT_AVAILABLE:
            try:
                self.sector_chat = SectorResearchChat()
                self.group_chat = GroupAnalysisChat()
                st.success("✅ AI Chat interfaces initialized")
            except Exception as e:
                st.error(f"❌ Failed to initialize chat interfaces: {e}")
        
        # Expanded symbol list as requested
        self.expanded_symbols = [
            'NVDA', 'NFLX', 'GOOG', 'MSFT', 'FDX', 'JNJ', 'DV', 
            'SPY', 'TQQQ', 'QQQ', 'TSLA', 'HD', 'KO'
        ]
        
        # Initialize session state
        if 'chat_history' not in st.session_state:
            st.session_state.chat_history = []
        if 'selected_chat_type' not in st.session_state:
            st.session_state.selected_chat_type = 'sector'
    
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
    
    def create_market_overview(self, data):
        """Create market overview section"""
        st.markdown('<div class="main-header">🤖 Enhanced Market Scanner with AI Chat</div>', unsafe_allow_html=True)
        
        # Key metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(
                "Tickers Scanned",
                data.get('tickers_scanned', len(self.expanded_symbols)),
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
            st.metric(
                "AI Chat",
                "✅ Available" if CHAT_AVAILABLE else "❌ Unavailable",
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
    
    def create_expanded_symbols_section(self):
        """Create section showing expanded symbol scanning"""
        st.markdown("### 📊 Expanded Symbol Coverage")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("**Technology & Growth**")
            tech_symbols = ['NVDA', 'NFLX', 'GOOG', 'MSFT', 'TSLA']
            for symbol in tech_symbols:
                st.write(f"• {symbol}")
        
        with col2:
            st.markdown("**Market Indices & ETFs**")
            index_symbols = ['SPY', 'TQQQ', 'QQQ']
            for symbol in index_symbols:
                st.write(f"• {symbol}")
        
        with col3:
            st.markdown("**Diverse Sectors**")
            other_symbols = ['FDX', 'JNJ', 'DV', 'HD', 'KO']
            for symbol in other_symbols:
                st.write(f"• {symbol}")
    
    def create_ai_chat_interface(self):
        """Create AI chat interface"""
        st.markdown('<div class="chat-header">🤖 AI Research Assistant</div>', unsafe_allow_html=True)
        
        # Chat type selection
        chat_type = st.radio(
            "Select Analysis Type:",
            ["Sector Research", "Group Analysis"],
            horizontal=True
        )
        
        st.session_state.selected_chat_type = 'sector' if chat_type == "Sector Research" else 'group'
        
        # Chat input
        user_input = st.text_input(
            "Ask me about market analysis, sectors, or stock groups:",
            placeholder="e.g., 'Research the technology sector outlook' or 'Analyze NVDA, MSFT, GOOGL for correlation'",
            key="chat_input"
        )
        
        col1, col2 = st.columns([1, 4])
        with col1:
            send_button = st.button("Send", type="primary")
        
        with col2:
            if st.button("Clear Chat"):
                st.session_state.chat_history = []
                st.rerun()
        
        # Process chat input
        if send_button and user_input:
            self.process_chat_input(user_input, chat_type)
        
        # Display chat history
        self.display_chat_history()
        
        # Example questions
        st.markdown("#### 💡 Example Questions")
        
        if chat_type == "Sector Research":
            examples = [
                "Research the technology sector outlook for the next 3 months",
                "Compare healthcare and financial sectors performance",
                "What's the risk profile of the energy sector?",
                "Analyze semiconductor stocks in the technology sector"
            ]
        else:
            examples = [
                "Analyze NVDA, MSFT, GOOGL for correlation analysis",
                "Compare mega cap tech vs dividend aristocrats",
                "What's the risk profile of the fintech group?",
                "Research TSLA, NVDA, AMD for momentum analysis"
            ]
        
        for example in examples:
            if st.button(f"💬 {example}", key=f"example_{example}"):
                st.session_state.chat_input = example
                st.rerun()
    
    def process_chat_input(self, user_input, chat_type):
        """Process chat input and get AI response"""
        if not CHAT_AVAILABLE:
            st.error("Chat interfaces not available")
            return
        
        # Add user message to history
        st.session_state.chat_history.append({
            'type': 'user',
            'content': user_input,
            'timestamp': datetime.now().strftime("%H:%M:%S")
        })
        
        # Show loading
        with st.spinner("🤖 AI is analyzing your question..."):
            try:
                if chat_type == "Sector Research":
                    response = asyncio.run(self.sector_chat.ask_question(user_input))
                else:
                    response = asyncio.run(self.group_chat.ask_question(user_input))
                
                # Add assistant response to history
                if 'error' in response:
                    ai_response = f"❌ Error: {response['error']}"
                else:
                    ai_response = self.format_ai_response(response)
                
                st.session_state.chat_history.append({
                    'type': 'assistant',
                    'content': ai_response,
                    'timestamp': datetime.now().strftime("%H:%M:%S")
                })
                
            except Exception as e:
                st.error(f"Error processing chat input: {e}")
                st.session_state.chat_history.append({
                    'type': 'assistant',
                    'content': f"❌ Error: {str(e)}",
                    'timestamp': datetime.now().strftime("%H:%M:%S")
                })
        
        st.rerun()
    
    def format_ai_response(self, response):
        """Format AI response for display"""
        if 'llm_analysis' in response:
            llm_analysis = response['llm_analysis']
            
            formatted = "## AI Analysis Results\n\n"
            
            # Display each section
            sections = [
                ('Sector Overview', 'sector_overview'),
                ('Group Overview', 'group_overview'),
                ('Technical Analysis', 'technical_analysis'),
                ('Risk Assessment', 'risk_assessment'),
                ('Trading Recommendations', 'trading_recommendations'),
                ('Portfolio Recommendations', 'portfolio_recommendations')
            ]
            
            for title, key in sections:
                content = llm_analysis.get(key, '')
                if content and content.strip():
                    formatted += f"### {title}\n{content}\n\n"
            
            # If no structured sections, show raw response
            if llm_analysis.get('raw_response'):
                formatted += f"### Full Analysis\n{llm_analysis['raw_response']}\n"
            
            return formatted
        else:
            return "Analysis completed successfully."
    
    def display_chat_history(self):
        """Display chat history"""
        if not st.session_state.chat_history:
            return
        
        st.markdown("#### 💬 Chat History")
        
        for message in st.session_state.chat_history:
            if message['type'] == 'user':
                st.markdown(f"""
                <div class="chat-message user-message">
                    <strong>You ({message['timestamp']}):</strong><br>
                    {message['content']}
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="chat-message assistant-message">
                    <strong>AI Assistant ({message['timestamp']}):</strong><br>
                    {message['content']}
                </div>
                """, unsafe_allow_html=True)
    
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
    
    def create_real_time_data(self):
        """Create real-time data section with expanded symbols"""
        st.markdown("### ⏰ Real-Time Market Data")
        
        # Get current market data for expanded symbols
        try:
            data = {}
            
            # Fetch data for all expanded symbols
            for ticker in self.expanded_symbols:
                try:
                    stock = yf.Ticker(ticker)
                    info = stock.info
                    hist = stock.history(period="1d")
                    
                    if not hist.empty:
                        current_price = hist['Close'].iloc[-1]
                        prev_close = hist['Open'].iloc[-1]
                        change = current_price - prev_close
                        change_pct = (change / prev_close) * 100
                        
                        data[ticker] = {
                            'price': current_price,
                            'change': change,
                            'change_pct': change_pct
                        }
                except Exception as e:
                    st.warning(f"Failed to fetch data for {ticker}: {e}")
            
            # Display in organized sections
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.markdown("**Technology & Growth**")
                tech_symbols = ['NVDA', 'NFLX', 'GOOG', 'MSFT', 'TSLA']
                for symbol in tech_symbols:
                    if symbol in data:
                        info = data[symbol]
                        st.metric(symbol, f"${info['price']:.2f}", f"{info['change_pct']:+.2f}%")
            
            with col2:
                st.markdown("**Indices & ETFs**")
                index_symbols = ['SPY', 'TQQQ', 'QQQ']
                for symbol in index_symbols:
                    if symbol in data:
                        info = data[symbol]
                        st.metric(symbol, f"${info['price']:.2f}", f"{info['change_pct']:+.2f}%")
            
            with col3:
                st.markdown("**Diverse Sectors**")
                other_symbols = ['FDX', 'JNJ', 'DV', 'HD', 'KO']
                for symbol in other_symbols:
                    if symbol in data:
                        info = data[symbol]
                        st.metric(symbol, f"${info['price']:.2f}", f"{info['change_pct']:+.2f}%")
                
        except Exception as e:
            st.error(f"Failed to load real-time data: {e}")
    
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
    
    def run_dashboard(self):
        """Run the enhanced dashboard"""
        # Create tabs
        tab1, tab2, tab3 = st.tabs(["📊 Market Overview", "🤖 AI Chat Assistant", "📈 Real-Time Data"])
        
        with tab1:
            # Load data
            data = self.load_latest_data()
            
            if data is None:
                st.error("No scanner data available. Please run the scanner first.")
                return
            
            # Create sections
            self.create_market_overview(data)
            st.markdown("---")
            
            self.create_expanded_symbols_section()
            st.markdown("---")
            
            self.create_opportunities_table(data)
            st.markdown("---")
            
            self.create_performance_metrics(data)
        
        with tab2:
            self.create_ai_chat_interface()
        
        with tab3:
            self.create_real_time_data()
        
        # Auto-refresh button
        if st.button("🔄 Refresh All Data"):
            st.rerun()

def main():
    """Main function"""
    dashboard = EnhancedDashboardWithChat()
    dashboard.run_dashboard()

if __name__ == "__main__":
    main()
