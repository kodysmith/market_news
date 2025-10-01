import { onRequest } from 'firebase-functions/v2/https';
import { setGlobalOptions } from 'firebase-functions/v2';
import * as admin from 'firebase-admin';
import axios from 'axios';

// Set global options for all functions
setGlobalOptions({
  maxInstances: 10,
  memory: '1GiB',
  timeoutSeconds: 60,
});

// Firebase Admin is already initialized in index.ts

interface ChatMessage {
  role: 'user' | 'assistant';
  content: string;
  timestamp: string;
}

interface StockData {
  ticker: string;
  currentPrice: number;
  change: number;
  changePercent: number;
  volume: number;
  marketCap: number;
  peRatio: number;
  high52w: number;
  low52w: number;
}

interface TechnicalAnalysis {
  rsi: number;
  sma20: number;
  sma50: number;
  trend: 'BULLISH' | 'BEARISH' | 'MIXED';
  signal: 'BUY' | 'SELL' | 'HOLD';
  confidence: number;
  support: number;
  resistance: number;
}

/**
 * Main QuantEngine Chat Function
 * Handles conversational AI for trading analysis
 */
export const quantChat = onRequest({
  cors: true,
  memory: '1GiB',
  timeoutSeconds: 60,
}, async (req, res) => {
  try {
    const { message, conversationHistory = [] } = req.body;

    if (!message) {
      res.status(400).json({ error: 'Message is required' });
      return;
    }

    console.log(`🤖 QuantEngine Chat: ${message.substring(0, 50)}...`);

    // Detect if asking about specific stock
    const ticker = detectTicker(message);
    
    let response: string;
    let analysisType = 'general_chat';

    if (ticker) {
      // Get real-time stock analysis
      const stockData = await getStockData(ticker);
      const technicalAnalysis = await getTechnicalAnalysis(ticker);
      const stockAnalysis = {
        ticker,
        stockData,
        technicalAnalysis,
        timestamp: new Date().toISOString()
      };
      response = await generateStockResponse(message, stockAnalysis, conversationHistory);
      analysisType = 'stock_analysis';
    } else if (isTradingSignalRequest(message)) {
      // Generate trading signals
      const signals = await generateTradingSignals();
      response = await generateTradingResponse(message, signals, conversationHistory);
      analysisType = 'trading_signals';
    } else {
      // General chat
      response = await generateGeneralResponse(message, conversationHistory);
    }

    // Store conversation in Firestore
    await storeConversation(message, response, analysisType);

    res.json({
      success: true,
      response,
      analysisType,
      timestamp: new Date().toISOString()
    });

  } catch (error) {
    console.error('QuantEngine Chat Error:', error);
    res.status(500).json({ 
      error: 'Failed to process chat request',
      details: error instanceof Error ? error.message : 'Unknown error'
    });
  }
});

/**
 * Stock Analysis Function
 * Provides detailed technical and fundamental analysis
 */
export const analyzeStock = onRequest({
  cors: true,
  memory: '1GiB',
  timeoutSeconds: 60,
}, async (req, res) => {
  try {
    const { ticker } = req.params;
    
    if (!ticker) {
      res.status(400).json({ error: 'Ticker is required' });
      return;
    }

    console.log(`📊 Analyzing stock: ${ticker}`);

    const stockData = await getStockData(ticker.toUpperCase());
    const technicalAnalysis = await getTechnicalAnalysis(ticker.toUpperCase());
    const fundamentalAnalysis = await getFundamentalAnalysis(ticker.toUpperCase());

    const analysis = {
      ticker: ticker.toUpperCase(),
      stockData,
      technicalAnalysis,
      fundamentalAnalysis,
      timestamp: new Date().toISOString()
    };

    // Store analysis in Firestore
    await admin.firestore().collection('stock_analyses').add(analysis);

    res.json({
      success: true,
      analysis
    });

  } catch (error) {
    console.error('Stock Analysis Error:', error);
    res.status(500).json({ 
      error: 'Failed to analyze stock',
      details: error instanceof Error ? error.message : 'Unknown error'
    });
  }
});

/**
 * Market Scanner Function
 * Scans for overbought/oversold stocks
 */
export const scanMarket = onRequest({
  cors: true,
  memory: '1GiB',
  timeoutSeconds: 60,
}, async (req, res) => {
  try {
    console.log('🔍 Scanning market for opportunities...');

    const popularTickers = ['AAPL', 'MSFT', 'GOOGL', 'NVDA', 'TSLA', 'AMZN', 'META', 'NFLX', 'AMD', 'INTC'];
    const opportunities = [];

    for (const ticker of popularTickers) {
      try {
        const analysis = await getTechnicalAnalysis(ticker);
        
        if (analysis.signal === 'BUY' && analysis.confidence > 70) {
          opportunities.push({
            ticker,
            signal: 'BUY',
            confidence: analysis.confidence,
            reason: 'Strong bullish signals',
            ...analysis
          });
        } else if (analysis.signal === 'SELL' && analysis.confidence > 70) {
          opportunities.push({
            ticker,
            signal: 'SELL',
            confidence: analysis.confidence,
            reason: 'Strong bearish signals',
            ...analysis
          });
        }
      } catch (error) {
        console.warn(`Failed to analyze ${ticker}:`, error);
      }
    }

    // Store scan results
    await admin.firestore().collection('market_scans').add({
      opportunities,
      timestamp: new Date().toISOString(),
      totalScanned: popularTickers.length,
      opportunitiesFound: opportunities.length
    });

    res.json({
      success: true,
      opportunities,
      totalScanned: popularTickers.length,
      timestamp: new Date().toISOString()
    });

  } catch (error) {
    console.error('Market Scan Error:', error);
    res.status(500).json({ 
      error: 'Failed to scan market',
      details: error instanceof Error ? error.message : 'Unknown error'
    });
  }
});

// Helper Functions

async function getStockData(ticker: string): Promise<StockData> {
  try {
    // Use Yahoo Finance API
    const response = await axios.get(`https://query1.finance.yahoo.com/v8/finance/chart/${ticker}?range=1d&interval=1d`);
    const result = response.data.chart.result[0];
    const meta = result.meta;
    const quotes = result.indicators.quote[0];
    
    const currentPrice = meta.regularMarketPrice;
    const prevClose = meta.previousClose;
    const change = currentPrice - prevClose;
    const changePercent = (change / prevClose) * 100;

    return {
      ticker,
      currentPrice,
      change,
      changePercent,
      volume: quotes.volume[quotes.volume.length - 1] || 0,
      marketCap: meta.marketCap || 0,
      peRatio: meta.trailingPE || 0,
      high52w: meta.fiftyTwoWeekHigh || currentPrice,
      low52w: meta.fiftyTwoWeekLow || currentPrice
    };
  } catch (error) {
    throw new Error(`Failed to get stock data for ${ticker}: ${error}`);
  }
}

async function getTechnicalAnalysis(ticker: string): Promise<TechnicalAnalysis> {
  try {
    // Get historical data from Yahoo Finance
    const response = await axios.get(`https://query1.finance.yahoo.com/v8/finance/chart/${ticker}?range=3mo&interval=1d`);
    const result = response.data.chart.result[0];
    const quotes = result.indicators.quote[0];
    
    const prices = quotes.close.filter((price: number) => price !== null);
    const currentPrice = prices[prices.length - 1];
    
    // Calculate RSI
    const rsi = calculateRSI(prices, 14);
    
    // Calculate SMAs
    const sma20 = calculateSMA(prices, 20);
    const sma50 = calculateSMA(prices, 50);
    
    // Determine trend
    let trend: 'BULLISH' | 'BEARISH' | 'MIXED';
    if (currentPrice > sma20 && sma20 > sma50) {
      trend = 'BULLISH';
    } else if (currentPrice < sma20 && sma20 < sma50) {
      trend = 'BEARISH';
    } else {
      trend = 'MIXED';
    }
    
    // Generate signal
    let signal: 'BUY' | 'SELL' | 'HOLD';
    let confidence: number;
    
    if (trend === 'BULLISH' && rsi < 70) {
      signal = 'BUY';
      confidence = 75;
    } else if (trend === 'BEARISH' && rsi > 30) {
      signal = 'SELL';
      confidence = 75;
    } else if (rsi < 30) {
      signal = 'BUY';
      confidence = 60;
    } else if (rsi > 70) {
      signal = 'SELL';
      confidence = 60;
    } else {
      signal = 'HOLD';
      confidence = 50;
    }
    
    // Calculate support and resistance
    const recentHigh = Math.max(...prices.slice(-20));
    const recentLow = Math.min(...prices.slice(-20));
    
    return {
      rsi,
      sma20,
      sma50,
      trend,
      signal,
      confidence,
      support: recentLow,
      resistance: recentHigh
    };
  } catch (error) {
    throw new Error(`Failed to get technical analysis for ${ticker}: ${error}`);
  }
}

async function getFundamentalAnalysis(ticker: string): Promise<any> {
  try {
    // Use Yahoo Finance API for fundamental data
    const response = await axios.get(`https://query1.finance.yahoo.com/v8/finance/chart/${ticker}?range=1d&interval=1d`);
    const result = response.data.chart.result[0];
    const meta = result.meta;
    
    return {
      marketCap: meta.marketCap || 0,
      peRatio: meta.trailingPE || 0,
      pegRatio: meta.pegRatio || 0,
      priceToBook: meta.priceToBook || 0,
      debtToEquity: meta.debtToEquity || 0,
      returnOnEquity: meta.returnOnEquity || 0,
      returnOnAssets: meta.returnOnAssets || 0,
      revenue: meta.totalRevenue || 0,
      profitMargin: meta.profitMargins || 0,
      earningsGrowth: meta.earningsGrowth || 0,
      revenueGrowth: meta.revenueGrowth || 0
    };
  } catch (error) {
    throw new Error(`Failed to get fundamental analysis for ${ticker}: ${error}`);
  }
}

function calculateRSI(prices: number[], period: number = 14): number {
  if (prices.length < period + 1) return 50;
  
  const deltas = [];
  for (let i = 1; i < prices.length; i++) {
    deltas.push(prices[i] - prices[i - 1]);
  }
  
  let gains = 0;
  let losses = 0;
  
  for (let i = deltas.length - period; i < deltas.length; i++) {
    if (deltas[i] > 0) gains += deltas[i];
    else losses -= deltas[i];
  }
  
  const avgGain = gains / period;
  const avgLoss = losses / period;
  
  if (avgLoss === 0) return 100;
  
  const rs = avgGain / avgLoss;
  return 100 - (100 / (1 + rs));
}

function calculateSMA(prices: number[], period: number): number {
  if (prices.length < period) return prices[prices.length - 1];
  
  const sum = prices.slice(-period).reduce((a, b) => a + b, 0);
  return sum / period;
}

function detectTicker(message: string): string | null {
  const tickers = ['AAPL', 'MSFT', 'GOOGL', 'NVDA', 'TSLA', 'AMZN', 'META', 'NFLX', 'AMD', 'INTC'];
  const messageUpper = message.toUpperCase();
  
  for (const ticker of tickers) {
    if (messageUpper.includes(ticker)) {
      return ticker;
    }
  }
  
  return null;
}

function isTradingSignalRequest(message: string): boolean {
  const signalKeywords = [
    'trading signals', 'trading opportunities', 'buy signals', 'sell signals',
    'overbought', 'oversold', 'market scan', 'opportunities'
  ];
  
  const messageLower = message.toLowerCase();
  return signalKeywords.some(keyword => messageLower.includes(keyword));
}

async function generateStockResponse(message: string, analysis: any, conversationHistory: ChatMessage[]): Promise<string> {
  // This would integrate with OpenAI/Anthropic API
  // For now, return a structured response
  return `📊 **${analysis.ticker} Analysis**

💰 **Price Action:**
• Current: $${analysis.stockData.currentPrice.toFixed(2)}
• Change: ${analysis.stockData.changePercent.toFixed(2)}%
• Volume: ${analysis.stockData.volume.toLocaleString()}

📈 **Technical Indicators:**
• RSI: ${analysis.technicalAnalysis.rsi.toFixed(1)} (${analysis.technicalAnalysis.rsi > 70 ? 'Overbought' : analysis.technicalAnalysis.rsi < 30 ? 'Oversold' : 'Neutral'})
• Trend: ${analysis.technicalAnalysis.trend}
• 20-day SMA: $${analysis.technicalAnalysis.sma20.toFixed(2)}
• 50-day SMA: $${analysis.technicalAnalysis.sma50.toFixed(2)}

🎯 **Trading Signal:**
• Recommendation: ${analysis.technicalAnalysis.signal}
• Confidence: ${analysis.technicalAnalysis.confidence}%

💡 **Key Levels:**
• Support: $${analysis.technicalAnalysis.support.toFixed(2)}
• Resistance: $${analysis.technicalAnalysis.resistance.toFixed(2)}

${analysis.technicalAnalysis.signal === 'BUY' ? '🟢 Consider buying on pullbacks to support' : 
  analysis.technicalAnalysis.signal === 'SELL' ? '🔴 Consider selling on rallies to resistance' : 
  '⚪ Wait for clearer signals'}`;
}

async function generateTradingResponse(message: string, signals: any[], conversationHistory: ChatMessage[]): Promise<string> {
  if (signals.length === 0) {
    return "🔍 **Market Scan Results**\n\nNo strong trading opportunities found at the moment. The market appears to be in a consolidation phase. Consider waiting for clearer signals or look for specific stocks you're interested in.";
  }
  
  let response = "🔍 **Market Scan Results**\n\n";
  response += `Found ${signals.length} trading opportunities:\n\n`;
  
  signals.slice(0, 5).forEach((signal, index) => {
    response += `${index + 1}. **${signal.ticker}** - ${signal.signal} (${signal.confidence}%)\n`;
    response += `   • Reason: ${signal.reason}\n`;
    response += `   • RSI: ${signal.rsi.toFixed(1)}\n`;
    response += `   • Trend: ${signal.trend}\n\n`;
  });
  
  return response;
}

async function generateGeneralResponse(message: string, conversationHistory: ChatMessage[]): Promise<string> {
  // This would integrate with OpenAI/Anthropic API
  // For now, return a helpful response
  return `🤖 **AI Trading Assistant**

I'm your AI-powered trading assistant! I can help you with:

📊 **Stock Analysis** - Ask about any stock (e.g., "What is NVDA doing?")
📈 **Trading Signals** - Get buy/sell recommendations
🔍 **Market Scanning** - Find overbought/oversold stocks
📋 **Fundamental Analysis** - Company financials and metrics
💬 **General Questions** - Market trends and strategies

What would you like to know?`;
}

async function storeConversation(userMessage: string, assistantResponse: string, analysisType: string): Promise<void> {
  try {
    await admin.firestore().collection('chat_conversations').add({
      userMessage,
      assistantResponse,
      analysisType,
      timestamp: admin.firestore.FieldValue.serverTimestamp(),
      createdAt: new Date().toISOString()
    });
  } catch (error) {
    console.warn('Failed to store conversation:', error);
  }
}

async function generateTradingSignals(): Promise<any[]> {
  // This would scan multiple stocks and return trading opportunities
  // For now, return empty array
  return [];
}
