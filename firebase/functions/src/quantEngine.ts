import { onRequest } from 'firebase-functions/v2/https';
import { setGlobalOptions } from 'firebase-functions/v2';
import * as admin from 'firebase-admin';
import axios from 'axios';
import { getCache, cachedRequest } from './utils/apiCache';

// Set global options for all functions
setGlobalOptions({
  maxInstances: 10,
  memory: '1GiB',
  timeoutSeconds: 60,
});

// Firebase Admin is already initialized in index.ts

// Initialize API cache with 10-second TTL
const apiCache = getCache(10);

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

interface PriceHistory {
  closes: number[];
  currentPrice: number;
  range: string;
  interval: string;
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
      const priceHistory = await getPriceHistory(ticker);
      const stockAnalysis = {
        ticker,
        stockData,
        priceHistory,
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
    const priceHistory = await getPriceHistory(ticker.toUpperCase());
    const fundamentalAnalysis = await getFundamentalAnalysis(ticker.toUpperCase());

    const analysis = {
      ticker: ticker.toUpperCase(),
      stockData,
      priceHistory,
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
    const opportunities: any[] = [];

    for (const ticker of popularTickers) {
      try {
        // Device-first architecture: return raw history for on-device scoring.
        const stockData = await getStockData(ticker);
        const priceHistory = await getPriceHistory(ticker);
        opportunities.push({ ticker, stockData, priceHistory });
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
    // Use Yahoo Finance API with caching
    const url = `https://query1.finance.yahoo.com/v8/finance/chart/${ticker}?range=1d&interval=1d`;
    
    const response = await cachedRequest(
      () => axios.get(url),
      apiCache,
      url,
      undefined,
      undefined,
      10
    );
    
    const result = response.chart.result[0];
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

async function getPriceHistory(ticker: string, range: string = '3mo', interval: string = '1d'): Promise<PriceHistory> {
  try {
    // Get historical data from Yahoo Finance with caching
    const url = `https://query1.finance.yahoo.com/v8/finance/chart/${ticker}?range=${range}&interval=${interval}`;
    
    const response = await cachedRequest(
      () => axios.get(url),
      apiCache,
      url,
      undefined,
      undefined,
      10
    );
    
    const result = response.chart.result[0];
    const quotes = result.indicators.quote[0];
    
    const closes = (quotes.close as Array<number | null>).filter((price) => price !== null) as number[];
    const currentPrice = closes.length > 0 ? closes[closes.length - 1] : 0;

    return { closes, currentPrice, range, interval };
  } catch (error) {
    throw new Error(`Failed to get price history for ${ticker}: ${error}`);
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

// Technical indicators are intentionally not computed server-side in the
// device-first architecture. Return raw history and compute on-device.

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

📈 **History:**
• Range: ${analysis.priceHistory.range}
• Interval: ${analysis.priceHistory.interval}
• Points: ${analysis.priceHistory.closes.length}

Note: Technical indicators/signals are computed on-device in the current architecture.`;
}

async function generateTradingResponse(message: string, signals: any[], conversationHistory: ChatMessage[]): Promise<string> {
  if (signals.length === 0) {
    return "🔍 **Market Scan Results**\n\nNo strong trading opportunities found at the moment. The market appears to be in a consolidation phase. Consider waiting for clearer signals or look for specific stocks you're interested in.";
  }
  
  let response = "🔍 **Market Scan Results**\n\n";
  response += `Found ${signals.length} trading opportunities:\n\n`;
  
  signals.slice(0, 5).forEach((signal, index) => {
    response += `${index + 1}. **${signal.ticker}**\n`;
    response += `   • Price: $${signal.stockData?.currentPrice?.toFixed?.(2) ?? 'N/A'}\n`;
    response += `   • History points: ${signal.priceHistory?.closes?.length ?? 0}\n\n`;
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
