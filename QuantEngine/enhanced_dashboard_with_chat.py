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
from typing import List, Dict

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

# Import GEX calculator
try:
    from gex_calculator import (
        calculate_gex, 
        get_option_chain_snapshot, 
        get_spot_price,
        calculate_flip_line
    )
    GEX_AVAILABLE = True
except ImportError as e:
    st.warning(f"GEX calculator not available: {e}")
    GEX_AVAILABLE = False

# Import divergence tracker
try:
    from divergence_tracker import DivergenceTracker
    DIVERGENCE_AVAILABLE = True
except ImportError as e:
    st.warning(f"Divergence tracker not available: {e}")
    DIVERGENCE_AVAILABLE = False

# Configure Streamlit
st.set_page_config(
    page_title="Enhanced Market Scanner with AI Chat",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS - Dark theme compatible
st.markdown("""
<style>
    /* Main headers */
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #00d4aa;
        text-align: center;
        margin-bottom: 2rem;
        text-shadow: 0 0 10px rgba(0, 212, 170, 0.3);
    }
    .chat-header {
        font-size: 1.8rem;
        font-weight: bold;
        color: #00d4aa;
        margin-bottom: 1rem;
    }
    
    /* Metric cards - dark theme */
    .metric-card {
        background-color: rgba(255, 255, 255, 0.1);
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #00d4aa;
        color: #ffffff;
        backdrop-filter: blur(10px);
    }
    
    /* Chat messages - dark theme */
    .chat-message {
        background-color: rgba(255, 255, 255, 0.05);
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
        border-left: 4px solid #00d4aa;
        color: #ffffff;
        backdrop-filter: blur(5px);
    }
    .user-message {
        background-color: rgba(0, 212, 170, 0.1);
        border-left-color: #00d4aa;
        color: #ffffff;
    }
    .assistant-message {
        background-color: rgba(0, 150, 136, 0.1);
        border-left-color: #009688;
        color: #ffffff;
    }
    
    /* Opportunity cards - dark theme */
    .opportunity-card {
        background-color: rgba(255, 255, 255, 0.05);
        padding: 1rem;
        border-radius: 0.5rem;
        border: 1px solid rgba(255, 255, 255, 0.1);
        margin-bottom: 1rem;
        color: #ffffff;
        backdrop-filter: blur(5px);
    }
    
    /* Status colors - high contrast */
    .bullish {
        color: #4caf50;
        font-weight: bold;
        text-shadow: 0 0 5px rgba(76, 175, 80, 0.5);
    }
    .bearish {
        color: #f44336;
        font-weight: bold;
        text-shadow: 0 0 5px rgba(244, 67, 54, 0.5);
    }
    .neutral {
        color: #ff9800;
        font-weight: bold;
        text-shadow: 0 0 5px rgba(255, 152, 0, 0.5);
    }
    
    /* Tabs styling - dark theme */
    .stTabs [data-baseweb="tab-list"] {
        gap: 2px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        white-space: pre-wrap;
        background-color: rgba(255, 255, 255, 0.1);
        border-radius: 4px 4px 0px 0px;
        gap: 1px;
        padding-left: 20px;
        padding-right: 20px;
        color: #ffffff;
    }
    .stTabs [aria-selected="true"] {
        background-color: #00d4aa;
        color: #000000;
        font-weight: bold;
    }
    
    /* General text visibility */
    .stMarkdown {
        color: #ffffff;
    }
    
    /* Dataframe styling */
    .dataframe {
        background-color: rgba(255, 255, 255, 0.05);
        color: #ffffff;
    }
    
    /* Input styling */
    .stTextInput > div > div > input {
        background-color: rgba(255, 255, 255, 0.1);
        color: #ffffff;
        border: 1px solid rgba(255, 255, 255, 0.2);
    }
    
    /* Button styling */
    .stButton > button {
        background-color: #00d4aa;
        color: #000000;
        border: none;
        border-radius: 5px;
        font-weight: bold;
    }
    .stButton > button:hover {
        background-color: #00b894;
        color: #000000;
    }
    
    /* Selectbox styling */
    .stSelectbox > div > div {
        background-color: rgba(255, 255, 255, 0.1);
        color: #ffffff;
    }
    
    /* Radio button styling */
    .stRadio > div {
        background-color: rgba(255, 255, 255, 0.05);
        border-radius: 5px;
        padding: 10px;
    }
    
    /* Error messages */
    .stAlert {
        background-color: rgba(244, 67, 54, 0.1);
        border: 1px solid #f44336;
        color: #ffffff;
    }
    
    /* Success messages */
    .stSuccess {
        background-color: rgba(76, 175, 80, 0.1);
        border: 1px solid #4caf50;
        color: #ffffff;
    }
    
    /* Info messages */
    .stInfo {
        background-color: rgba(33, 150, 243, 0.1);
        border: 1px solid #2196f3;
        color: #ffffff;
    }
</style>
""", unsafe_allow_html=True)

class EnhancedDashboardWithChat:
    """Enhanced Market Scanner Dashboard with AI Chat Integration"""
    
    def __init__(self):
        self.data_dir = Path("reports")
        self.data_dir.mkdir(exist_ok=True)
        
        # Load config for API keys
        self.config_path = Path("data/config.json")
        self.config = {}
        if self.config_path.exists():
            with open(self.config_path, 'r') as f:
                self.config = json.load(f)
        
        # Initialize chat interfaces
        self.sector_chat = None
        self.group_chat = None
        
        if CHAT_AVAILABLE:
            try:
                self.sector_chat = SectorResearchChat()
                self.group_chat = GroupAnalysisChat()
            except Exception as e:
                pass  # Silent fail, will show in UI
        
        # Expanded symbol list as requested
        self.expanded_symbols = [
            'NVDA', 'NFLX', 'GOOG', 'MSFT', 'FDX', 'JNJ', 'DV', 
            'SPY', 'TQQQ', 'QQQ', 'TSLA', 'HD', 'KO'
        ]
        
        # GEX tickers from config
        self.gex_tickers = self.config.get("GEX_TICKERS", ["SPY", "SPX", "QQQ"])
        
        # Initialize session state
        if 'chat_history' not in st.session_state:
            st.session_state.chat_history = []
        if 'selected_chat_type' not in st.session_state:
            st.session_state.selected_chat_type = 'sector'
        if 'gex_data' not in st.session_state:
            st.session_state.gex_data = {}
        if 'divergence_tracker' not in st.session_state:
            st.session_state.divergence_tracker = None
        if 'divergence_update_interval' not in st.session_state:
            st.session_state.divergence_update_interval = 5  # minutes
        if 'divergence_auto_refresh' not in st.session_state:
            st.session_state.divergence_auto_refresh = True
        if 'last_divergence_update' not in st.session_state:
            st.session_state.last_divergence_update = None
    
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
    
    def create_divergence_tracker(self):
        """Create SPY/SPX divergence tracker section"""
        st.markdown("---")
        st.markdown("""
        <div style="text-align: center; margin: 2rem 0;">
            <h2 style="color: #00d4aa; font-size: 2rem;">⚡ SPY/SPX Divergence Tracker</h2>
            <p style="color: #888; font-size: 1rem;">
                Track which asset moves first (SPY = retail sentiment, SPX = institutional) 
                and identify arbitrage opportunities
            </p>
        </div>
        """, unsafe_allow_html=True)
        
        if not DIVERGENCE_AVAILABLE:
            st.error("Divergence tracker module not available. Please ensure divergence_tracker.py is in the QuantEngine directory.")
            return
        
        alphavantage_key = self.config.get("ALPHAVANTAGE_API_KEY", "")
        massive_key = self.config.get("MASSIVE_API_KEY", "")  # Optional fallback
        
        if not alphavantage_key:
            st.warning("⚠️ Alpha Vantage API key not configured. Using yfinance as fallback for SPY.")
            alphavantage_key = st.text_input("Enter Alpha Vantage API Key (optional)", type="password", key="div_av_key")
        
        # Note: SPX uses yfinance (Yahoo Finance), Massive API is optional fallback
        if not massive_key:
            st.info("ℹ️ SPX will use Yahoo Finance (yfinance). Massive API key is optional fallback.")
        
        # Get divergence config
        div_config = self.config.get("DIVERGENCE_TRACKER", {})
        
        # Initialize tracker if needed
        if st.session_state.divergence_tracker is None:
            try:
                st.session_state.divergence_tracker = DivergenceTracker(
                    alphavantage_key,
                    massive_api_key=massive_key,
                    config=div_config
                )
            except Exception as e:
                st.error(f"Failed to initialize divergence tracker: {e}")
                return
        
        tracker = st.session_state.divergence_tracker
        
        # Controls
        col1, col2, col3 = st.columns(3)
        
        with col1:
            update_intervals = [1, 5, 15, 30, 60]
            selected_interval = st.selectbox(
                "Update Frequency",
                options=update_intervals,
                index=update_intervals.index(st.session_state.divergence_update_interval) if st.session_state.divergence_update_interval in update_intervals else 1,
                help="How often to fetch new price data"
            )
            st.session_state.divergence_update_interval = selected_interval
        
        with col2:
            auto_refresh = st.checkbox(
                "Auto-refresh",
                value=st.session_state.divergence_auto_refresh,
                help="Automatically update prices at selected interval"
            )
            st.session_state.divergence_auto_refresh = auto_refresh
        
        with col3:
            if st.button("🔄 Refresh Now", use_container_width=True):
                st.session_state.last_divergence_update = None  # Force update
        
        # Check if we need to update
        should_update = False
        now = time.time()
        
        if st.session_state.divergence_auto_refresh:
            if st.session_state.last_divergence_update is None:
                should_update = True
            else:
                interval_seconds = st.session_state.divergence_update_interval * 60
                if (now - st.session_state.last_divergence_update) >= interval_seconds:
                    should_update = True
        else:
            # Manual refresh only
            if st.session_state.last_divergence_update is None:
                should_update = True
        
        # Fetch prices if needed
        if should_update:
            with st.spinner("Fetching SPY and SPX prices..."):
                spy_price, spx_price = tracker.fetch_current_prices()
                st.session_state.last_divergence_update = now
                
                if not spy_price:
                    st.error("❌ Failed to fetch SPY price. Check Alpha Vantage API key (or yfinance fallback) and network connection.")
                    return
                
                if not spx_price:
                    st.error("❌ Failed to fetch SPX price. Using yfinance (Yahoo Finance) for SPX index. Check network connection.")
                    return
        
        # Get current state
        state = tracker.get_current_state()
        
        if not state.get('has_data'):
            st.info("No price data available. Click 'Refresh Now' to fetch initial prices.")
            return
        
        # Display metrics
        st.markdown("### 📊 Current State")
        
        col1, col2, col3, col4, col5 = st.columns(5)
        
        with col1:
            st.metric("SPY Price", f"${state['spy_price']:,.2f}")
        
        with col2:
            st.metric("SPX Price", f"${state['spx_price']:,.2f}")
        
        with col3:
            normalized_spx = state['normalized_spx']
            delta = normalized_spx - state['spy_price']
            delta_pct = state['divergence']['percentage_divergence']
            st.metric(
                "Divergence",
                f"{delta_pct:+.3f}%",
                delta=f"${delta:+.2f}"
            )
        
        with col4:
            first_mover = state.get('first_mover', {})
            if first_mover:
                mover = first_mover.get('asset', 'Unknown')
                mover_label = "🟢 SPY (Retail)" if mover == 'SPY' else "🔵 SPX (Institutional)"
                st.metric("First Mover", mover_label)
            else:
                st.metric("First Mover", "N/A")
        
        with col5:
            duration = state.get('divergence_duration')
            if duration:
                minutes = int(duration // 60)
                seconds = int(duration % 60)
                st.metric("Divergence Duration", f"{minutes}m {seconds}s")
            else:
                st.metric("Divergence Duration", "Converged")
        
        # Direction indicator
        divergence = state['divergence']
        direction = divergence['direction']
        is_diverging = divergence['is_diverging']
        
        if is_diverging:
            if direction == 'spy_high':
                st.warning(f"⚠️ SPY is trading HIGHER than expected (retail bullish). Divergence: {divergence['percentage_divergence']:+.3f}%")
            else:
                st.warning(f"⚠️ SPX is trading HIGHER than expected (institutional bullish). Divergence: {divergence['percentage_divergence']:+.3f}%")
        else:
            st.success(f"✅ SPY and SPX are converged. Difference: {divergence['percentage_divergence']:+.3f}%")
        
        # Get history for charts
        history = tracker.get_divergence_history(limit=500)
        
        # Show current metrics even with just 1 data point
        if len(history) < 2:
            st.info(f"📊 **Collecting data...** ({len(history)} data point{'s' if len(history) != 1 else ''} collected)")
            st.info("💡 **Tip:** Charts will appear after 2+ data points. Current metrics are shown above.")
            
            # Show what we have so far
            if len(history) == 1:
                latest = history[0]
                st.markdown("### 📈 Current Snapshot")
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("SPY Price", f"${latest['spy_price']:,.2f}")
                    st.metric("SPX Price", f"${latest['spx_price']:,.2f}")
                with col2:
                    st.metric("Normalized SPX", f"${latest['normalized_spx']:,.2f}")
                    div_pct = latest['divergence']['percentage_divergence']
                    st.metric("Divergence", f"{div_pct:+.3f}%")
            
            # Show when next update will happen
            if st.session_state.divergence_auto_refresh and st.session_state.last_divergence_update:
                interval_seconds = st.session_state.divergence_update_interval * 60
                time_until_next = interval_seconds - (time.time() - st.session_state.last_divergence_update)
                if time_until_next > 0:
                    minutes = int(time_until_next // 60)
                    seconds = int(time_until_next % 60)
                    st.success(f"⏱️ Next data point in {minutes}m {seconds}s (charts will appear after 2+ points)")
            
            return
        
        # Create visualizations
        st.markdown("---")
        st.markdown("### 📈 Visualizations")
        
        # Tab for different chart views
        chart_tab1, chart_tab2, chart_tab3 = st.tabs([
            "📊 Dual Price Chart",
            "📉 Divergence Chart", 
            "⏱️ Timeline View"
        ])
        
        with chart_tab1:
            self._create_dual_price_chart(history, state)
        
        with chart_tab2:
            self._create_divergence_chart(history, state)
        
        with chart_tab3:
            self._create_timeline_view(history, state)
        
        # Auto-refresh mechanism
        if st.session_state.divergence_auto_refresh and st.session_state.last_divergence_update:
            interval_seconds = st.session_state.divergence_update_interval * 60
            time_until_next = interval_seconds - (now - st.session_state.last_divergence_update)
            if time_until_next > 0:
                st.caption(f"⏱️ Next update in {int(time_until_next // 60)}m {int(time_until_next % 60)}s")
            else:
                # Schedule refresh using placeholder
                placeholder = st.empty()
                with placeholder.container():
                    st.info("🔄 Auto-refreshing...")
                time.sleep(0.1)
                st.rerun()
        elif st.session_state.divergence_auto_refresh:
            st.caption("⏱️ Waiting for first update...")
    
    def _create_dual_price_chart(self, history: List[Dict], state: Dict):
        """Create dual line chart showing SPY and normalized SPX"""
        if len(history) < 2:
            return
        
        # Prepare data
        timestamps = [h['timestamp'] for h in history]
        spy_prices = [h['spy_price'] for h in history]
        normalized_spx = [h['normalized_spx'] for h in history]
        
        # Convert timestamps to datetime
        datetimes = [datetime.fromtimestamp(ts) for ts in timestamps]
        
        fig = go.Figure()
        
        # SPY line
        fig.add_trace(go.Scatter(
            x=datetimes,
            y=spy_prices,
            name='SPY',
            line=dict(color='#00d4aa', width=2),
            mode='lines+markers',
            marker=dict(size=4)
        ))
        
        # Normalized SPX line
        fig.add_trace(go.Scatter(
            x=datetimes,
            y=normalized_spx,
            name='SPX (normalized)',
            line=dict(color='#ff6b6b', width=2),
            mode='lines+markers',
            marker=dict(size=4)
        ))
        
        # Highlight divergence periods
        for i, h in enumerate(history):
            if h['divergence']['is_diverging']:
                if i < len(datetimes) - 1:
                    fig.add_vrect(
                        x0=datetimes[i],
                        x1=datetimes[min(i+1, len(datetimes)-1)],
                        fillcolor="rgba(255, 107, 107, 0.1)",
                        layer="below",
                        line_width=0,
                    )
        
        # Current prices reference lines
        if state.get('spy_price'):
            fig.add_hline(
                y=state['spy_price'],
                line_dash="dot",
                line_color="#00d4aa",
                opacity=0.5,
                annotation_text=f"SPY: ${state['spy_price']:.2f}"
            )
        
        if state.get('normalized_spx'):
            fig.add_hline(
                y=state['normalized_spx'],
                line_dash="dot",
                line_color="#ff6b6b",
                opacity=0.5,
                annotation_text=f"SPX: ${state['normalized_spx']:.2f}"
            )
        
        fig.update_layout(
            title="SPY vs SPX (Normalized) Price Comparison",
            xaxis_title="Time",
            yaxis_title="Price ($)",
            height=500,
            template="plotly_dark",
            hovermode='x unified',
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    def _create_divergence_chart(self, history: List[Dict], state: Dict):
        """Create divergence chart showing delta over time"""
        if len(history) < 2:
            return
        
        timestamps = [h['timestamp'] for h in history]
        divergences = [h['divergence']['percentage_divergence'] for h in history]
        first_movers = [h.get('first_mover', {}).get('asset') if h.get('first_mover') else None for h in history]
        
        datetimes = [datetime.fromtimestamp(ts) for ts in timestamps]
        
        fig = go.Figure()
        
        # Color code by direction and first mover
        colors = []
        for i, h in enumerate(history):
            div = h['divergence']
            mover = first_movers[i]
            
            if div['percentage_divergence'] > 0:
                if mover == 'SPY':
                    colors.append('#00d4aa')  # Green for SPY leading up
                else:
                    colors.append('#4ecdc4')  # Cyan for SPX leading up
            else:
                if mover == 'SPY':
                    colors.append('#ff6b6b')  # Red for SPY leading down
                else:
                    colors.append('#ffa07a')  # Light red for SPX leading down
        
        # Divergence line
        fig.add_trace(go.Scatter(
            x=datetimes,
            y=divergences,
            name='Divergence %',
            line=dict(color='#888', width=2),
            mode='lines+markers',
            marker=dict(size=6, color=colors),
            fill='tozeroy',
            fillcolor='rgba(255, 107, 107, 0.1)'
        ))
        
        # Zero line
        fig.add_hline(y=0, line_dash="dash", line_color="white", line_width=1, annotation_text="Converged")
        
        # Threshold lines
        div_config = self.config.get("DIVERGENCE_TRACKER", {})
        threshold = div_config.get("divergence_threshold_percent", 0.1)
        fig.add_hline(y=threshold, line_dash="dot", line_color="yellow", opacity=0.5, annotation_text=f"Threshold: +{threshold}%")
        fig.add_hline(y=-threshold, line_dash="dot", line_color="yellow", opacity=0.5, annotation_text=f"Threshold: -{threshold}%")
        
        # First mover annotations
        for i, mover in enumerate(first_movers):
            if mover and i % max(1, len(history) // 20) == 0:  # Sample annotations
                fig.add_annotation(
                    x=datetimes[i],
                    y=divergences[i],
                    text="🟢" if mover == 'SPY' else "🔵",
                    showarrow=True,
                    arrowhead=2,
                    arrowsize=1,
                    arrowwidth=2,
                    arrowcolor=colors[i] if i < len(colors) else '#888'
                )
        
        fig.update_layout(
            title="Divergence Over Time (SPY vs SPX)",
            xaxis_title="Time",
            yaxis_title="Divergence (%)",
            height=500,
            template="plotly_dark",
            hovermode='x unified'
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Legend
        st.markdown("""
        <div style="margin-top: 1rem;">
            <strong>Legend:</strong><br>
            🟢 SPY moved first (Retail sentiment) | 🔵 SPX moved first (Institutional sentiment)<br>
            Green markers = Positive divergence | Red markers = Negative divergence
        </div>
        """, unsafe_allow_html=True)
    
    def _create_timeline_view(self, history: List[Dict], state: Dict):
        """Create timeline visualization showing first mover events"""
        if len(history) < 2:
            return
        
        # Extract events
        events = []
        current_divergence_start = None
        
        for i, h in enumerate(history):
            timestamp = h['timestamp']
            dt = datetime.fromtimestamp(timestamp)
            first_mover = h.get('first_mover')
            divergence = h['divergence']
            
            # Divergence start/end events
            if divergence['is_diverging']:
                if current_divergence_start is None:
                    current_divergence_start = timestamp
                    events.append({
                        'time': dt,
                        'type': 'divergence_start',
                        'direction': divergence['direction'],
                        'magnitude': divergence['percentage_divergence']
                    })
            else:
                if current_divergence_start is not None:
                    duration = timestamp - current_divergence_start
                    events.append({
                        'time': dt,
                        'type': 'divergence_end',
                        'duration': duration
                    })
                    current_divergence_start = None
            
            # First mover events
            if first_mover and i > 0:
                prev_mover = history[i-1].get('first_mover', {}).get('asset') if history[i-1].get('first_mover') else None
                if first_mover.get('asset') != prev_mover:
                    events.append({
                        'time': dt,
                        'type': 'first_mover_change',
                        'asset': first_mover.get('asset'),
                        'spy_change_pct': first_mover.get('spy_change_pct', 0),
                        'spx_change_pct': first_mover.get('spx_change_pct', 0)
                    })
        
        if not events:
            st.info("No significant events detected yet. Collecting more data...")
            return
        
        # Display timeline
        st.markdown("### ⏱️ Event Timeline")
        
        for event in events[-20:]:  # Show last 20 events
            event_time = event['time'].strftime("%H:%M:%S")
            
            if event['type'] == 'divergence_start':
                direction_icon = "📈" if event['magnitude'] > 0 else "📉"
                st.markdown(f"""
                <div style="background: rgba(255, 107, 107, 0.1); padding: 0.5rem; border-left: 4px solid #ff6b6b; margin: 0.5rem 0;">
                    <strong>{event_time}</strong> - {direction_icon} Divergence Started<br>
                    Direction: {event['direction']} | Magnitude: {event['magnitude']:+.3f}%
                </div>
                """, unsafe_allow_html=True)
            
            elif event['type'] == 'divergence_end':
                duration_min = int(event['duration'] // 60)
                st.markdown(f"""
                <div style="background: rgba(0, 212, 170, 0.1); padding: 0.5rem; border-left: 4px solid #00d4aa; margin: 0.5rem 0;">
                    <strong>{event_time}</strong> - ✅ Divergence Ended<br>
                    Duration: {duration_min} minutes
                </div>
                """, unsafe_allow_html=True)
            
            elif event['type'] == 'first_mover_change':
                asset = event['asset']
                icon = "🟢" if asset == 'SPY' else "🔵"
                label = "SPY (Retail)" if asset == 'SPY' else "SPX (Institutional)"
                st.markdown(f"""
                <div style="background: rgba(78, 205, 196, 0.1); padding: 0.5rem; border-left: 4px solid #4ecdc4; margin: 0.5rem 0;">
                    <strong>{event_time}</strong> - {icon} First Mover: {label}<br>
                    SPY change: {event['spy_change_pct']:+.2f}% | SPX change: {event['spx_change_pct']:+.2f}%
                </div>
                """, unsafe_allow_html=True)
        
        # Current divergence status
        if state.get('divergence_duration'):
            duration_min = int(state['divergence_duration'] // 60)
            duration_sec = int(state['divergence_duration'] % 60)
            st.markdown(f"""
            <div style="background: rgba(255, 193, 7, 0.2); padding: 1rem; border: 2px solid #ffc107; margin: 1rem 0; border-radius: 0.5rem;">
                <h4>Current Divergence Status</h4>
                <p><strong>Duration:</strong> {duration_min} minutes {duration_sec} seconds</p>
                <p><strong>Magnitude:</strong> {state['divergence']['percentage_divergence']:+.3f}%</p>
                <p><strong>Direction:</strong> {state['divergence']['direction']}</p>
            </div>
            """, unsafe_allow_html=True)
    
    def create_gex_interface(self):
        """Create Gamma Exposure (GEX) calculation interface"""
        st.markdown("""
        <div class="main-header">
            ⚡ Gamma Exposure (GEX) Calculator
        </div>
        <p style="text-align: center; color: #888; margin-bottom: 2rem;">
            Calculate dealer gamma exposure and identify the flip line (zero gamma level) for options chains
        </p>
        """, unsafe_allow_html=True)
        
        if not GEX_AVAILABLE:
            st.error("GEX calculator module not available. Please ensure gex_calculator.py is in the QuantEngine directory.")
            return
        
        # Get API keys
        massive_key = self.config.get("MASSIVE_API_KEY", "")
        alphavantage_key = self.config.get("ALPHAVANTAGE_API_KEY", "")
        
        # Sidebar for API keys and ticker selection
        with st.sidebar:
            st.markdown("### 🔑 API Configuration")
            massive_key = st.text_input(
                "Massive API Key", 
                value=massive_key, 
                type="password",
                help="Required for fetching options chain data"
            )
            alphavantage_key = st.text_input(
                "Alpha Vantage API Key", 
                value=alphavantage_key, 
                type="password",
                help="Required for fetching spot prices"
            )
            
            if st.button("💾 Save API Keys"):
                if self.config_path.exists():
                    self.config["MASSIVE_API_KEY"] = massive_key
                    self.config["ALPHAVANTAGE_API_KEY"] = alphavantage_key
                    with open(self.config_path, 'w') as f:
                        json.dump(self.config, f, indent=4)
                    st.success("API keys saved!")
                else:
                    st.error("Config file not found")
            
            st.markdown("---")
            st.markdown("### 📊 Ticker Selection")
            ticker = st.selectbox(
                "Select Ticker",
                options=self.gex_tickers,
                index=0 if "SPY" in self.gex_tickers else 0
            )
            
            custom_ticker = st.text_input("Or enter custom ticker", value="").upper()
            if custom_ticker:
                ticker = custom_ticker
        
        if not massive_key:
            st.warning("⚠️ Please enter your Massive API key in the sidebar to fetch options data.")
            return
        
        # Fetch and calculate GEX
        with st.spinner(f"Fetching options data for {ticker}..."):
            try:
                # Get options snapshot (API max limit is 250)
                snap = get_option_chain_snapshot(massive_key, ticker, limit=250)
                
                if not snap.get("results"):
                    st.error(f"No options data found for {ticker}")
                    return
                
                # Get spot price
                spot_price = get_spot_price(ticker, massive_key, alphavantage_key)
                
                if not spot_price:
                    st.warning(f"Could not fetch spot price automatically. Please enter manually.")
                    spot_price = st.number_input(
                        f"Enter {ticker} spot price",
                        value=0.0,
                        min_value=0.0,
                        step=0.01,
                        format="%.2f"
                    )
                    if spot_price <= 0:
                        st.error("Spot price is required")
                        return
                else:
                    st.info(f"📊 Current {ticker} spot price: ${spot_price:,.2f}")
                
                # Calculate GEX
                df, agg, metrics, skipped = calculate_gex(
                    snap, 
                    spot_price, 
                    massive_key, 
                    alphavantage_key
                )
                
                if df.empty:
                    st.error("No computable contracts found. Check that contracts have implied volatility data.")
                    return
                
                # Display metrics
                st.markdown("---")
                col1, col2, col3, col4, col5 = st.columns(5)
                
                with col1:
                    st.metric("Spot Price", f"${metrics['spot_price']:,.2f}")
                with col2:
                    st.metric("Total GEX", f"{metrics['total_gex']:,.0f}")
                with col3:
                    st.metric("Put Wall", f"${metrics['put_wall']:,.2f}")
                with col4:
                    st.metric("Call Wall", f"${metrics['call_wall']:,.2f}")
                with col5:
                    if metrics['flip_line']:
                        st.metric("Flip Line", f"${metrics['flip_line']:,.2f}")
                    else:
                        st.metric("Flip Line", "N/A")
                
                # Breakdown
                call_gex = float(df[df["type"] == "call"]["gex"].sum())
                put_gex = float(df[df["type"] == "put"]["gex"].sum())
                
                st.caption(
                    f"📊 Processed: {len(df)} contracts | "
                    f"Skipped: {skipped} | "
                    f"Call GEX: {call_gex:,.0f} | "
                    f"Put GEX: {put_gex:,.0f}"
                )
                
                # Chart
                st.markdown("---")
                st.markdown("### 📈 GEX by Strike")
                
                fig = go.Figure()
                
                # Bar chart
                fig.add_trace(go.Bar(
                    x=agg["strike"],
                    y=agg["gex"],
                    name="Net GEX",
                    marker_color=agg["gex"].apply(lambda x: "#00d4aa" if x > 0 else "#ff4444"),
                    opacity=0.7
                ))
                
                # Zero line
                fig.add_hline(y=0, line_dash="dash", line_color="white", line_width=1)
                
                # Flip line
                if metrics['flip_line']:
                    fig.add_vline(
                        x=metrics['flip_line'],
                        line_dash="dash",
                        line_color="red",
                        line_width=2,
                        annotation_text=f"Flip Line: ${metrics['flip_line']:,.2f}",
                        annotation_position="top"
                    )
                
                # Spot price line
                fig.add_vline(
                    x=spot_price,
                    line_dash="dot",
                    line_color="green",
                    line_width=1.5,
                    opacity=0.7,
                    annotation_text=f"Spot: ${spot_price:,.2f}",
                    annotation_position="bottom"
                )
                
                fig.update_layout(
                    title=f"Gamma Exposure (GEX) for {ticker}",
                    xaxis_title="Strike Price",
                    yaxis_title="GEX",
                    height=500,
                    template="plotly_dark",
                    showlegend=False
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Data table
                st.markdown("### 📋 GEX by Strike (Detailed)")
                st.dataframe(
                    agg.style.format({
                        "strike": "${:,.2f}",
                        "gex": "{:,.0f}"
                    }),
                    use_container_width=True,
                    height=400
                )
                
                # Debug info
                with st.expander("🔍 Debug Information"):
                    st.write(f"**Total contracts processed:** {len(df)}")
                    st.write(f"**Total contracts skipped:** {skipped}")
                    st.write(f"**Spot price used:** ${spot_price:,.2f}")
                    st.write(f"**Call contracts:** {len(df[df['type'] == 'call'])}")
                    st.write(f"**Put contracts:** {len(df[df['type'] == 'put'])}")
                    
                    if st.checkbox("Show sample contracts"):
                        st.dataframe(df.head(20), use_container_width=True)
                    
                    if st.checkbox("Show raw API response"):
                        st.json(snap)
                
            except requests.exceptions.RequestException as e:
                st.error(f"❌ API Error: {e}")
                st.info("Please check your API keys and network connection.")
            except Exception as e:
                st.error(f"❌ Error calculating GEX: {e}")
                import traceback
                with st.expander("Error Details"):
                    st.code(traceback.format_exc())
    
    def run_dashboard(self):
        """Run the enhanced dashboard"""
        # Create tabs - GEX is now the FIRST and most important tab
        tab0, tab1, tab2, tab3 = st.tabs([
            "⚡ GEX Calculator", 
            "📊 Market Overview", 
            "🤖 AI Chat Assistant", 
            "📈 Real-Time Data"
        ])
        
        with tab0:
            self.create_gex_interface()
        
        with tab1:
            # Divergence Tracker (most important feature)
            self.create_divergence_tracker()
            st.markdown("---")
            
            # Load data
            data = self.load_latest_data()
            
            if data is None:
                st.warning("No scanner data available. Divergence tracker is still active above.")
            else:
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
