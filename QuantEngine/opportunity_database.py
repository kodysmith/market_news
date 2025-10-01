#!/usr/bin/env python3
"""
Opportunity Database - Comprehensive Stock Opportunity Scanner

Scans top 10 stocks from each sector to identify trade opportunities.
Stores results in SQLite database for quick querying and training data generation.

Features:
- Sector-based scanning (11 major sectors)
- Top 10 stocks per sector by market cap
- Real-time opportunity detection
- Historical tracking with timestamps
- Training data export capabilities
- Cost-effective (no API limits)
"""

import yfinance as yf
import pandas as pd
import sqlite3
import numpy as np
from datetime import datetime, timedelta
import json
import logging
from typing import Dict, List, Any, Optional
import time
import random

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class OpportunityDatabase:
    """Comprehensive opportunity scanner and database"""
    
    def __init__(self, db_path: str = "opportunity_database.db"):
        self.db_path = db_path
        self.connection = None
        self.sector_stocks = self._get_sector_stocks()
        self._init_database()
    
    def _get_sector_stocks(self) -> Dict[str, List[str]]:
        """
        Define top 10 stocks by market cap for each major sector
        Based on S&P 500 sector classification
        """
        return {
            'Technology': [
                'AAPL', 'MSFT', 'NVDA', 'GOOGL', 'META', 'TSLA', 'AVGO', 'ORCL', 'CRM', 'ADBE'
            ],
            'Healthcare': [
                'JNJ', 'UNH', 'PFE', 'ABBV', 'MRK', 'TMO', 'ABT', 'DHR', 'BMY', 'AMGN'
            ],
            'Financials': [
                'BRK-B', 'JPM', 'BAC', 'WFC', 'GS', 'MS', 'C', 'BLK', 'AXP', 'SPGI'
            ],
            'Consumer Discretionary': [
                'AMZN', 'HD', 'MCD', 'NKE', 'SBUX', 'LOW', 'TJX', 'BKNG', 'CMG', 'ABNB'
            ],
            'Consumer Staples': [
                'PG', 'KO', 'PEP', 'WMT', 'COST', 'CL', 'KMB', 'GIS', 'K', 'HSY'
            ],
            'Industrials': [
                'BA', 'CAT', 'HON', 'UPS', 'RTX', 'LMT', 'GE', 'MMM', 'DE', 'EMR'
            ],
            'Energy': [
                'XOM', 'CVX', 'COP', 'EOG', 'SLB', 'PXD', 'MPC', 'VLO', 'KMI', 'PSX'
            ],
            'Utilities': [
                'NEE', 'DUK', 'SO', 'D', 'AEP', 'EXC', 'SRE', 'XEL', 'PEG', 'WEC'
            ],
            'Real Estate': [
                'AMT', 'PLD', 'CCI', 'EQIX', 'PSA', 'O', 'SPG', 'WELL', 'EXR', 'AVB'
            ],
            'Materials': [
                'LIN', 'APD', 'SHW', 'FCX', 'NEM', 'ECL', 'DOW', 'PPG', 'DD', 'IFF'
            ],
            'Communication Services': [
                'GOOGL', 'META', 'NFLX', 'DIS', 'CMCSA', 'VZ', 'T', 'CHTR', 'TMUS', 'EA'
            ]
        }
    
    def _init_database(self):
        """Initialize SQLite database with opportunity tables"""
        self.connection = sqlite3.connect(self.db_path)
        cursor = self.connection.cursor()
        
        # Create opportunities table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS opportunities (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                ticker TEXT NOT NULL,
                sector TEXT NOT NULL,
                scan_date TIMESTAMP NOT NULL,
                current_price REAL,
                opportunity_type TEXT NOT NULL,
                signal_strength REAL,
                confidence_score REAL,
                entry_price REAL,
                target_price REAL,
                stop_loss REAL,
                risk_reward_ratio REAL,
                technical_score REAL,
                fundamental_score REAL,
                sector_score REAL,
                overall_score REAL,
                rsi REAL,
                macd_signal TEXT,
                trend_direction TEXT,
                volume_signal TEXT,
                support_level REAL,
                resistance_level REAL,
                pe_ratio REAL,
                revenue_growth REAL,
                profit_margin REAL,
                debt_equity REAL,
                market_cap REAL,
                relative_strength REAL,
                analyst_rating TEXT,
                price_target REAL,
                earnings_surprise REAL,
                raw_data TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # Create index for faster queries
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_ticker_date ON opportunities(ticker, scan_date)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_sector_date ON opportunities(sector, scan_date)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_opportunity_type ON opportunities(opportunity_type)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_overall_score ON opportunities(overall_score)')
        
        # Create sector performance table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS sector_performance (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                sector TEXT NOT NULL,
                scan_date TIMESTAMP NOT NULL,
                sector_etf TEXT,
                sector_return_1d REAL,
                sector_return_1w REAL,
                sector_return_1m REAL,
                sector_return_3m REAL,
                sector_return_1y REAL,
                avg_pe_ratio REAL,
                avg_revenue_growth REAL,
                avg_profit_margin REAL,
                total_market_cap REAL,
                opportunity_count INTEGER,
                strong_buy_count INTEGER,
                buy_count INTEGER,
                hold_count INTEGER,
                sell_count INTEGER,
                strong_sell_count INTEGER,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # Create training data table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS training_data (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                ticker TEXT NOT NULL,
                sector TEXT NOT NULL,
                scan_date TIMESTAMP NOT NULL,
                features TEXT NOT NULL,
                target_signal TEXT NOT NULL,
                actual_return_1d REAL,
                actual_return_1w REAL,
                actual_return_1m REAL,
                actual_return_3m REAL,
                prediction_accuracy REAL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        self.connection.commit()
        logger.info("✅ Database initialized successfully")
    
    def scan_all_sectors(self, save_to_db: bool = True) -> Dict[str, Any]:
        """
        Scan all sectors for opportunities
        """
        logger.info("🔍 Starting comprehensive sector scan...")
        
        all_opportunities = []
        sector_performance = {}
        total_opportunities = 0
        
        for sector, stocks in self.sector_stocks.items():
            logger.info(f"📊 Scanning {sector} sector ({len(stocks)} stocks)...")
            
            try:
                sector_opps, sector_perf = self._scan_sector(sector, stocks)
                all_opportunities.extend(sector_opps)
                sector_performance[sector] = sector_perf
                total_opportunities += len(sector_opps)
                
                logger.info(f"✅ {sector}: {len(sector_opps)} opportunities found")
                
                # Rate limiting to avoid API issues
                time.sleep(random.uniform(1, 2))
                
            except Exception as e:
                logger.error(f"❌ Error scanning {sector}: {e}")
                continue
        
        # Save to database
        if save_to_db:
            self._save_opportunities(all_opportunities)
            self._save_sector_performance(sector_performance)
        
        logger.info(f"🎯 Scan complete: {total_opportunities} total opportunities found")
        
        return {
            'total_opportunities': total_opportunities,
            'sector_breakdown': {sector: len(opps) for sector, opps in 
                               [(s, [o for o in all_opportunities if o['sector'] == s]) 
                                for s in self.sector_stocks.keys()]},
            'opportunities': all_opportunities,
            'sector_performance': sector_performance
        }
    
    def _scan_sector(self, sector: str, stocks: List[str]) -> tuple:
        """Scan a single sector for opportunities"""
        opportunities = []
        sector_data = []
        
        for ticker in stocks:
            try:
                # Get stock data
                stock = yf.Ticker(ticker)
                info = stock.info
                history = stock.history(period='1y')
                
                if history.empty:
                    continue
                
                # Analyze opportunity
                opportunity = self._analyze_opportunity(ticker, sector, info, history)
                
                if opportunity:
                    opportunities.append(opportunity)
                    sector_data.append({
                        'ticker': ticker,
                        'market_cap': info.get('marketCap', 0),
                        'pe_ratio': info.get('trailingPE', 0),
                        'revenue_growth': info.get('revenueGrowth', 0),
                        'profit_margin': info.get('profitMargins', 0),
                        'opportunity': opportunity
                    })
                
                # Rate limiting
                time.sleep(random.uniform(0.5, 1.0))
                
            except Exception as e:
                logger.warning(f"⚠️ Error analyzing {ticker}: {e}")
                continue
        
        # Calculate sector performance
        sector_performance = self._calculate_sector_performance(sector, sector_data)
        
        return opportunities, sector_performance
    
    def _analyze_opportunity(self, ticker: str, sector: str, info: Dict, history: pd.DataFrame) -> Optional[Dict[str, Any]]:
        """Analyze individual stock for trading opportunities"""
        try:
            current_price = float(history['Close'].iloc[-1])
            
            # Technical Analysis
            technical_score, technical_signals = self._calculate_technical_score(history)
            
            # Fundamental Analysis
            fundamental_score, fundamental_signals = self._calculate_fundamental_score(info)
            
            # Sector Analysis
            sector_score = self._calculate_sector_score(sector, info)
            
            # Overall Score
            overall_score = (technical_score * 0.4 + fundamental_score * 0.4 + sector_score * 0.2)
            
            # Determine opportunity type
            opportunity_type = self._determine_opportunity_type(
                technical_signals, fundamental_signals, overall_score
            )
            
            # Only record if it's a significant opportunity
            if overall_score < 30 or overall_score > 70:
                return {
                    'ticker': ticker,
                    'sector': sector,
                    'scan_date': datetime.now().isoformat(),
                    'current_price': current_price,
                    'opportunity_type': opportunity_type,
                    'signal_strength': abs(overall_score - 50) / 50,
                    'confidence_score': min(overall_score, 100 - overall_score) / 50,
                    'entry_price': current_price,
                    'target_price': self._calculate_target_price(current_price, opportunity_type, technical_signals),
                    'stop_loss': self._calculate_stop_loss(current_price, opportunity_type),
                    'risk_reward_ratio': self._calculate_risk_reward(current_price, opportunity_type),
                    'technical_score': technical_score,
                    'fundamental_score': fundamental_score,
                    'sector_score': sector_score,
                    'overall_score': overall_score,
                    'rsi': technical_signals.get('rsi', 50),
                    'macd_signal': technical_signals.get('macd_signal', 'neutral'),
                    'trend_direction': technical_signals.get('trend', 'sideways'),
                    'volume_signal': technical_signals.get('volume_signal', 'normal'),
                    'support_level': technical_signals.get('support', current_price * 0.95),
                    'resistance_level': technical_signals.get('resistance', current_price * 1.05),
                    'pe_ratio': info.get('trailingPE', 0),
                    'revenue_growth': info.get('revenueGrowth', 0) * 100 if info.get('revenueGrowth') else 0,
                    'profit_margin': info.get('profitMargins', 0) * 100 if info.get('profitMargins') else 0,
                    'debt_equity': info.get('debtToEquity', 0) / 100 if info.get('debtToEquity') else 0,
                    'market_cap': info.get('marketCap', 0),
                    'relative_strength': self._calculate_relative_strength(ticker, history),
                    'analyst_rating': info.get('recommendationKey', 'hold'),
                    'price_target': info.get('targetMeanPrice', 0),
                    'earnings_surprise': info.get('earningsSurprisePct', 0),
                    'raw_data': json.dumps({
                        'info': {k: v for k, v in info.items() if isinstance(v, (str, int, float, bool))},
                        'last_5_prices': history['Close'].tail().tolist(),
                        'last_5_volumes': history['Volume'].tail().tolist()
                    })
                }
            
            return None
            
        except Exception as e:
            logger.warning(f"⚠️ Error analyzing {ticker}: {e}")
            return None
    
    def _calculate_technical_score(self, history: pd.DataFrame) -> tuple:
        """Calculate technical analysis score and signals"""
        try:
            closes = history['Close']
            volumes = history['Volume']
            
            # RSI
            rsi = self._calculate_rsi(closes)
            
            # MACD
            macd_signal = self._calculate_macd_signal(closes)
            
            # Moving Averages
            sma_20 = closes.rolling(20).mean().iloc[-1]
            sma_50 = closes.rolling(50).mean().iloc[-1]
            current_price = closes.iloc[-1]
            
            # Trend
            if current_price > sma_20 > sma_50:
                trend = 'uptrend'
            elif current_price < sma_20 < sma_50:
                trend = 'downtrend'
            else:
                trend = 'sideways'
            
            # Volume
            avg_volume = volumes.rolling(20).mean().iloc[-1]
            current_volume = volumes.iloc[-1]
            volume_ratio = current_volume / avg_volume if avg_volume > 0 else 1
            
            if volume_ratio > 1.5:
                volume_signal = 'high'
            elif volume_ratio < 0.7:
                volume_signal = 'low'
            else:
                volume_signal = 'normal'
            
            # Support/Resistance
            support = closes.rolling(20).min().iloc[-1]
            resistance = closes.rolling(20).max().iloc[-1]
            
            # Calculate score
            score = 50  # Base score
            
            # RSI contribution
            if 30 <= rsi <= 70:
                score += 10
            elif rsi < 30:
                score += 20  # Oversold
            elif rsi > 70:
                score -= 20  # Overbought
            
            # MACD contribution
            if macd_signal == 'bullish':
                score += 15
            elif macd_signal == 'bearish':
                score -= 15
            
            # Trend contribution
            if trend == 'uptrend':
                score += 20
            elif trend == 'downtrend':
                score -= 20
            
            # Volume contribution
            if volume_signal == 'high' and trend == 'uptrend':
                score += 10
            elif volume_signal == 'high' and trend == 'downtrend':
                score -= 10
            
            # Price position
            price_position = (current_price - support) / (resistance - support) if resistance > support else 0.5
            if price_position < 0.3:
                score += 10  # Near support
            elif price_position > 0.7:
                score -= 10  # Near resistance
            
            return max(0, min(100, score)), {
                'rsi': rsi,
                'macd_signal': macd_signal,
                'trend': trend,
                'volume_signal': volume_signal,
                'support': support,
                'resistance': resistance
            }
            
        except Exception as e:
            logger.warning(f"Technical analysis error: {e}")
            return 50, {}
    
    def _calculate_fundamental_score(self, info: Dict) -> tuple:
        """Calculate fundamental analysis score and signals"""
        try:
            score = 50  # Base score
            
            # P/E Ratio
            pe_ratio = info.get('trailingPE', 0)
            if 10 <= pe_ratio <= 25:
                score += 20
            elif 5 <= pe_ratio < 10:
                score += 15
            elif 25 < pe_ratio <= 40:
                score += 10
            elif pe_ratio > 40:
                score -= 10
            
            # Revenue Growth
            revenue_growth = info.get('revenueGrowth', 0)
            if revenue_growth and revenue_growth > 0.2:
                score += 20
            elif revenue_growth and revenue_growth > 0.1:
                score += 15
            elif revenue_growth and revenue_growth > 0.05:
                score += 10
            elif revenue_growth and revenue_growth < 0:
                score -= 15
            
            # Profit Margin
            profit_margin = info.get('profitMargins', 0)
            if profit_margin and profit_margin > 0.2:
                score += 15
            elif profit_margin and profit_margin > 0.1:
                score += 10
            elif profit_margin and profit_margin > 0.05:
                score += 5
            elif profit_margin and profit_margin < 0:
                score -= 15
            
            # Debt/Equity
            debt_equity = info.get('debtToEquity', 0)
            if debt_equity and debt_equity < 0.5:
                score += 10
            elif debt_equity and debt_equity < 1.0:
                score += 5
            elif debt_equity and debt_equity > 2.0:
                score -= 10
            
            # Free Cash Flow
            fcf = info.get('freeCashflow', 0)
            if fcf and fcf > 0:
                score += 10
            elif fcf and fcf < 0:
                score -= 15
            
            return max(0, min(100, score)), {
                'pe_ratio': pe_ratio,
                'revenue_growth': revenue_growth,
                'profit_margin': profit_margin,
                'debt_equity': debt_equity,
                'free_cash_flow': fcf
            }
            
        except Exception as e:
            logger.warning(f"Fundamental analysis error: {e}")
            return 50, {}
    
    def _calculate_sector_score(self, sector: str, info: Dict) -> float:
        """Calculate sector-relative score"""
        # This would ideally compare against sector averages
        # For now, return a neutral score
        return 50.0
    
    def _calculate_relative_strength(self, ticker: str, history: pd.DataFrame) -> float:
        """Calculate relative strength vs S&P 500"""
        try:
            # Get SPY data
            spy = yf.Ticker('SPY')
            spy_history = spy.history(period='1y')
            
            if spy_history.empty or history.empty:
                return 0.0
            
            # Calculate returns
            stock_return = (history['Close'].iloc[-1] / history['Close'].iloc[0] - 1) * 100
            spy_return = (spy_history['Close'].iloc[-1] / spy_history['Close'].iloc[0] - 1) * 100
            
            return stock_return - spy_return
            
        except Exception as e:
            logger.warning(f"Relative strength calculation error: {e}")
            return 0.0
    
    def _determine_opportunity_type(self, technical_signals: Dict, fundamental_signals: Dict, overall_score: float) -> str:
        """Determine the type of opportunity"""
        if overall_score >= 80:
            return 'STRONG_BUY'
        elif overall_score >= 65:
            return 'BUY'
        elif overall_score >= 35:
            return 'HOLD'
        elif overall_score >= 20:
            return 'SELL'
        else:
            return 'STRONG_SELL'
    
    def _calculate_target_price(self, current_price: float, opportunity_type: str, technical_signals: Dict) -> float:
        """Calculate target price based on opportunity type"""
        resistance = technical_signals.get('resistance', current_price * 1.1)
        support = technical_signals.get('support', current_price * 0.9)
        
        if 'BUY' in opportunity_type:
            return resistance
        elif 'SELL' in opportunity_type:
            return support
        else:
            return current_price
    
    def _calculate_stop_loss(self, current_price: float, opportunity_type: str) -> float:
        """Calculate stop loss based on opportunity type"""
        if 'BUY' in opportunity_type:
            return current_price * 0.95  # 5% stop loss
        elif 'SELL' in opportunity_type:
            return current_price * 1.05  # 5% stop loss for shorts
        else:
            return current_price
    
    def _calculate_risk_reward(self, current_price: float, opportunity_type: str) -> float:
        """Calculate risk/reward ratio"""
        if 'BUY' in opportunity_type:
            target = current_price * 1.1
            stop = current_price * 0.95
            return (target - current_price) / (current_price - stop)
        elif 'SELL' in opportunity_type:
            target = current_price * 0.9
            stop = current_price * 1.05
            return (current_price - target) / (stop - current_price)
        else:
            return 1.0
    
    def _calculate_sector_performance(self, sector: str, sector_data: List[Dict]) -> Dict[str, Any]:
        """Calculate sector performance metrics"""
        if not sector_data:
            return {
                'sector': sector,
                'scan_date': datetime.now().isoformat(),
                'avg_pe_ratio': 0,
                'avg_revenue_growth': 0,
                'avg_profit_margin': 0,
                'total_market_cap': 0,
                'opportunity_count': 0,
                'STRONG_BUY': 0, 'BUY': 0, 'HOLD': 0, 'SELL': 0, 'STRONG_SELL': 0
            }
        
        # Calculate averages
        pe_ratios = [d['pe_ratio'] for d in sector_data if d['pe_ratio'] > 0]
        revenue_growths = [d['revenue_growth'] for d in sector_data if d['revenue_growth'] != 0]
        profit_margins = [d['profit_margin'] for d in sector_data if d['profit_margin'] != 0]
        
        avg_pe = np.mean(pe_ratios) if pe_ratios else 0
        avg_revenue_growth = np.mean(revenue_growths) if revenue_growths else 0
        avg_profit_margin = np.mean(profit_margins) if profit_margins else 0
        total_market_cap = sum([d['market_cap'] for d in sector_data])
        
        # Count opportunities
        opportunity_counts = {'STRONG_BUY': 0, 'BUY': 0, 'HOLD': 0, 'SELL': 0, 'STRONG_SELL': 0}
        for d in sector_data:
            opp_type = d['opportunity']['opportunity_type']
            if opp_type in opportunity_counts:
                opportunity_counts[opp_type] += 1
        
        return {
            'sector': sector,
            'scan_date': datetime.now().isoformat(),
            'avg_pe_ratio': avg_pe,
            'avg_revenue_growth': avg_revenue_growth,
            'avg_profit_margin': avg_profit_margin,
            'total_market_cap': total_market_cap,
            'opportunity_count': len(sector_data),
            **opportunity_counts
        }
    
    def _calculate_rsi(self, closes: pd.Series, period: int = 14) -> float:
        """Calculate RSI"""
        try:
            delta = closes.diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
            rs = gain / loss
            rsi = 100 - (100 / (1 + rs))
            return float(rsi.iloc[-1])
        except:
            return 50.0
    
    def _calculate_macd_signal(self, closes: pd.Series) -> str:
        """Calculate MACD signal"""
        try:
            ema_12 = closes.ewm(span=12).mean()
            ema_26 = closes.ewm(span=26).mean()
            macd = ema_12 - ema_26
            signal = macd.ewm(span=9).mean()
            histogram = macd - signal
            
            if histogram.iloc[-1] > 0 and histogram.iloc[-2] <= 0:
                return 'bullish'
            elif histogram.iloc[-1] < 0 and histogram.iloc[-2] >= 0:
                return 'bearish'
            elif histogram.iloc[-1] > 0:
                return 'positive'
            else:
                return 'negative'
        except:
            return 'neutral'
    
    def _save_opportunities(self, opportunities: List[Dict[str, Any]]):
        """Save opportunities to database"""
        if not opportunities:
            return
        
        cursor = self.connection.cursor()
        
        for opp in opportunities:
            cursor.execute('''
                INSERT INTO opportunities (
                    ticker, sector, scan_date, current_price, opportunity_type,
                    signal_strength, confidence_score, entry_price, target_price,
                    stop_loss, risk_reward_ratio, technical_score, fundamental_score,
                    sector_score, overall_score, rsi, macd_signal, trend_direction,
                    volume_signal, support_level, resistance_level, pe_ratio,
                    revenue_growth, profit_margin, debt_equity, market_cap,
                    relative_strength, analyst_rating, price_target, earnings_surprise,
                    raw_data
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                opp['ticker'], opp['sector'], opp['scan_date'], opp['current_price'],
                opp['opportunity_type'], opp['signal_strength'], opp['confidence_score'],
                opp['entry_price'], opp['target_price'], opp['stop_loss'],
                opp['risk_reward_ratio'], opp['technical_score'], opp['fundamental_score'],
                opp['sector_score'], opp['overall_score'], opp['rsi'], opp['macd_signal'],
                opp['trend_direction'], opp['volume_signal'], opp['support_level'],
                opp['resistance_level'], opp['pe_ratio'], opp['revenue_growth'],
                opp['profit_margin'], opp['debt_equity'], opp['market_cap'],
                opp['relative_strength'], opp['analyst_rating'], opp['price_target'],
                opp['earnings_surprise'], opp['raw_data']
            ))
        
        self.connection.commit()
        logger.info(f"💾 Saved {len(opportunities)} opportunities to database")
    
    def _save_sector_performance(self, sector_performance: Dict[str, Dict[str, Any]]):
        """Save sector performance to database"""
        cursor = self.connection.cursor()
        
        for sector, perf in sector_performance.items():
            cursor.execute('''
                INSERT INTO sector_performance (
                    sector, scan_date, avg_pe_ratio, avg_revenue_growth,
                    avg_profit_margin, total_market_cap, opportunity_count,
                    strong_buy_count, buy_count, hold_count, sell_count, strong_sell_count
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                sector, perf['scan_date'], perf['avg_pe_ratio'],
                perf['avg_revenue_growth'], perf['avg_profit_margin'],
                perf['total_market_cap'], perf['opportunity_count'],
                perf['STRONG_BUY'], perf['BUY'], perf['HOLD'],
                perf['SELL'], perf['STRONG_SELL']
            ))
        
        self.connection.commit()
        logger.info("💾 Saved sector performance to database")
    
    def query_opportunities(self, 
                          ticker: str = None,
                          sector: str = None,
                          opportunity_type: str = None,
                          min_score: float = None,
                          max_score: float = None,
                          days_back: int = 7,
                          limit: int = 100) -> pd.DataFrame:
        """Query opportunities from database"""
        cursor = self.connection.cursor()
        
        query = "SELECT * FROM opportunities WHERE 1=1"
        params = []
        
        if ticker:
            query += " AND ticker = ?"
            params.append(ticker)
        
        if sector:
            query += " AND sector = ?"
            params.append(sector)
        
        if opportunity_type:
            query += " AND opportunity_type = ?"
            params.append(opportunity_type)
        
        if min_score is not None:
            query += " AND overall_score >= ?"
            params.append(min_score)
        
        if max_score is not None:
            query += " AND overall_score <= ?"
            params.append(max_score)
        
        if days_back:
            query += " AND scan_date >= datetime('now', '-{} days')".format(days_back)
        
        query += " ORDER BY scan_date DESC, overall_score DESC LIMIT ?"
        params.append(limit)
        
        df = pd.read_sql_query(query, self.connection, params=params)
        return df
    
    def get_top_opportunities(self, limit: int = 20) -> pd.DataFrame:
        """Get top opportunities by score"""
        return self.query_opportunities(min_score=70, limit=limit)
    
    def get_sector_summary(self, days_back: int = 7) -> pd.DataFrame:
        """Get sector performance summary"""
        query = '''
            SELECT sector, 
                   COUNT(*) as total_opportunities,
                   AVG(overall_score) as avg_score,
                   SUM(CASE WHEN opportunity_type = 'STRONG_BUY' THEN 1 ELSE 0 END) as strong_buy_count,
                   SUM(CASE WHEN opportunity_type = 'BUY' THEN 1 ELSE 0 END) as buy_count,
                   SUM(CASE WHEN opportunity_type = 'HOLD' THEN 1 ELSE 0 END) as hold_count,
                   SUM(CASE WHEN opportunity_type = 'SELL' THEN 1 ELSE 0 END) as sell_count,
                   SUM(CASE WHEN opportunity_type = 'STRONG_SELL' THEN 1 ELSE 0 END) as strong_sell_count
            FROM opportunities 
            WHERE scan_date >= datetime('now', '-{} days')
            GROUP BY sector
            ORDER BY avg_score DESC
        '''.format(days_back)
        
        return pd.read_sql_query(query, self.connection)
    
    def export_training_data(self, days_back: int = 30) -> pd.DataFrame:
        """Export training data for ML models"""
        query = '''
            SELECT ticker, sector, scan_date, overall_score, opportunity_type,
                   technical_score, fundamental_score, sector_score,
                   rsi, pe_ratio, revenue_growth, profit_margin, debt_equity,
                   relative_strength, market_cap
            FROM opportunities 
            WHERE scan_date >= datetime('now', '-{} days')
            ORDER BY scan_date DESC
        '''.format(days_back)
        
        return pd.read_sql_query(query, self.connection)
    
    def close(self):
        """Close database connection"""
        if self.connection:
            self.connection.close()


def main():
    """Main function to run opportunity scanning"""
    db = OpportunityDatabase()
    
    try:
        # Run comprehensive scan
        results = db.scan_all_sectors()
        
        print(f"\n🎯 SCAN RESULTS:")
        print(f"Total Opportunities: {results['total_opportunities']}")
        print(f"\nSector Breakdown:")
        for sector, count in results['sector_breakdown'].items():
            print(f"  {sector}: {count} opportunities")
        
        # Show top opportunities
        print(f"\n🏆 TOP OPPORTUNITIES:")
        top_opps = db.get_top_opportunities(10)
        if not top_opps.empty:
            for _, opp in top_opps.iterrows():
                print(f"  {opp['ticker']} ({opp['sector']}): {opp['opportunity_type']} - Score: {opp['overall_score']:.1f}")
        
        # Show sector summary
        print(f"\n📊 SECTOR SUMMARY:")
        sector_summary = db.get_sector_summary()
        if not sector_summary.empty:
            print(sector_summary.to_string(index=False))
        
    finally:
        db.close()


if __name__ == '__main__':
    main()
