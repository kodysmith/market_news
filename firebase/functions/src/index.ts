import * as functions from 'firebase-functions';
import * as admin from 'firebase-admin';
import axios from 'axios';
import * as cors from 'cors';
import * as express from 'express';

// Initialize Firebase Admin
admin.initializeApp();

const app = express();
app.use(cors({ origin: true }));
app.options('*', cors({ origin: true }));

// Environment variables - get from process.env for Cloud Functions v2
const FMP_API_KEY = process.env.FMP_API_KEY;

interface VixData {
  date: string;
  close: number;
}

interface MarketSentiment {
  sentiment: string;
  indicators: Array<{
    name: string;
    ticker: string;
    price: string;
    direction: string;
  }>;
}

interface TopStrategy {
  name: string;
  description: string;
  score: number;
}

interface TradeIdea {
  ticker: string;
  strategy: string;
  expiry: string;
  details: string;
  cost: number;
  metric_name: string;
  metric_value: string;
  max_profit?: number;
  max_loss?: number;
  risk_reward_ratio?: number;
  iv_rank?: number;
  current_iv?: number;
}

// Cache for market data (30 minutes)
let cachedReportData: any = null;
let lastCacheTime = 0;
const CACHE_DURATION = 10 * 60 * 1000; // 10 minutes

// Cache for trade recommendations (10 minutes)
const tradeRecCache: { [key: string]: { data: any, timestamp: number } } = {};
const TRADE_REC_CACHE_DURATION = 10 * 60 * 1000; // 10 minutes

// Helper function to calculate historical volatility
async function calculateHistoricalVolatility(symbol: string, days: number = 252): Promise<number | null> {
  try {
    const endDate = new Date();
    const startDate = new Date(endDate.getTime() - (days + 30) * 24 * 60 * 60 * 1000);
    
    const response = await axios.get(`https://financialmodelingprep.com/api/v3/historical-price-full/${symbol}`, {
      params: {
        from: startDate.toISOString().split('T')[0],
        to: endDate.toISOString().split('T')[0],
        apikey: FMP_API_KEY
      }
    });

    if (response.data && response.data.historical && response.data.historical.length > 1) {
      const prices = response.data.historical.reverse().map((day: any) => parseFloat(day.close));
      
      if (prices.length < 20) return null;

      // Calculate daily returns
      const returns: number[] = [];
      for (let i = 1; i < prices.length; i++) {
        const dailyReturn = (prices[i] - prices[i - 1]) / prices[i - 1];
        returns.push(dailyReturn);
      }

      // Calculate annualized volatility
      if (returns.length > 0) {
        const mean = returns.reduce((sum, ret) => sum + ret, 0) / returns.length;
        const variance = returns.reduce((sum, ret) => sum + Math.pow(ret - mean, 2), 0) / returns.length;
        const volatility = Math.sqrt(variance) * Math.sqrt(252); // Annualized
        return volatility * 100; // Convert to percentage
      }
    }
    
    return null;
  } catch (error) {
    console.error(`Error calculating volatility for ${symbol}:`, error);
    return null;
  }
}

// Get VIX data from FMP, fallback to Yahoo Finance
async function getVixData(): Promise<VixData[]> {
  // Try FMP first
  try {
    const fmpResp = await axios.get(
      `https://financialmodelingprep.com/api/v3/historical-price-full/^VIX`,
      { params: { timeseries: 7, apikey: FMP_API_KEY } }
    );
    if (fmpResp.data && fmpResp.data.historical && fmpResp.data.historical.length > 0) {
      return fmpResp.data.historical.map((d: any) => ({
        date: d.date,
        close: d.close,
      }));
    }
    throw new Error('No VIX data from FMP');
  } catch (e) {
    console.error('FMP VIX failed, falling back to Yahoo:', e.message || e);
    // Yahoo fallback
    try {
      const yahooResp = await axios.get(
        `https://query1.finance.yahoo.com/v8/finance/chart/%5EVIX?range=7d&interval=1d`
      );
      const result = yahooResp.data.chart.result[0];
      const timestamps = result.timestamp;
      const closes = result.indicators.quote[0].close;
      return timestamps.map((ts: number, i: number) => ({
        date: new Date(ts * 1000).toISOString().slice(0, 10),
        close: closes[i],
      }));
    } catch (err) {
      console.error('Yahoo VIX fallback failed:', err.message || err);
      return [];
    }
  }
}

// Get market sentiment analysis (FMP primary, Yahoo fallback for indices)
async function getMarketSentiment(): Promise<MarketSentiment> {
  let indices = [];
  // Try FMP first
  try {
    const indicesResponse = await axios.get('https://financialmodelingprep.com/api/v3/quotes/index', {
      params: { apikey: FMP_API_KEY }
    });
    indices = indicesResponse.data || [];
    if (!Array.isArray(indices) || indices.length === 0) throw new Error('No indices from FMP');
  } catch (e) {
    console.error('FMP indices failed, falling back to Yahoo:', e.message || e);
    // Yahoo fallback for S&P 500, Nasdaq, VIX
    try {
      const [spResp, ndqResp, vixResp] = await Promise.all([
        axios.get('https://query1.finance.yahoo.com/v8/finance/chart/%5EGSPC?range=1d&interval=1d'),
        axios.get('https://query1.finance.yahoo.com/v8/finance/chart/%5EIXIC?range=1d&interval=1d'),
        axios.get('https://query1.finance.yahoo.com/v8/finance/chart/%5EVIX?range=1d&interval=1d'),
      ]);
      indices = [
        {
          symbol: '^GSPC',
          price: spResp.data.chart.result[0].meta.regularMarketPrice,
          change: spResp.data.chart.result[0].meta.regularMarketChange
        },
        {
          symbol: '^IXIC',
          price: ndqResp.data.chart.result[0].meta.regularMarketPrice,
          change: ndqResp.data.chart.result[0].meta.regularMarketChange
        },
        {
          symbol: '^VIX',
          price: vixResp.data.chart.result[0].meta.regularMarketPrice,
          change: vixResp.data.chart.result[0].meta.regularMarketChange
        }
      ];
    } catch (err) {
      console.error('Yahoo indices fallback failed:', err.message || err);
      indices = [];
    }
  }

  const spyData = indices.find((idx: any) => idx.symbol === '^GSPC');
  const nasdaqData = indices.find((idx: any) => idx.symbol === '^IXIC');
  const vixData = indices.find((idx: any) => idx.symbol === '^VIX');

  // Simple sentiment calculation based on major indices
  let bullishCount = 0;
  let totalCount = 0;

  const indicators = [
    {
      name: 'S&P 500',
      ticker: '^GSPC',
      price: spyData ? `$${spyData.price.toFixed(2)}` : 'N/A',
      direction: spyData && spyData.change > 0 ? 'UP' : 'DOWN'
    },
    {
      name: 'Nasdaq',
      ticker: '^IXIC',
      price: nasdaqData ? `$${nasdaqData.price.toFixed(2)}` : 'N/A',
      direction: nasdaqData && nasdaqData.change > 0 ? 'UP' : 'DOWN'
    },
    {
      name: 'VIX',
      ticker: '^VIX',
      price: vixData ? `${vixData.price.toFixed(2)}` : 'N/A',
      direction: vixData && vixData.change < 0 ? 'UP' : 'DOWN' // VIX inverse to market
    }
  ];

  indicators.forEach(indicator => {
    if (indicator.direction === 'UP') bullishCount++;
    totalCount++;
  });

  const sentiment = bullishCount >= totalCount / 2 ? 'BULLISH' : 'BEARISH';

  return {
    sentiment,
    indicators
  };
}

// Get top strategy recommendations
async function getTopStrategies(): Promise<TopStrategy[]> {
  const sentiment = (await getMarketSentiment()).sentiment;

  if (sentiment === 'BULLISH') {
    return [
      {
        name: 'Bull Put Spread',
        description: 'Sell out-of-the-money puts to collect premium in bullish markets',
        score: 8.5
      },
      {
        name: 'Covered Call',
        description: 'Generate income on existing stock positions',
        score: 7.8
      },
      {
        name: 'Cash-Secured Put',
        description: 'Sell puts on stocks you want to own at lower prices',
        score: 7.2
      }
    ];
  } else if (sentiment === 'BEARISH') {
    return [
      {
        name: 'Bear Call Spread',
        description: 'Sell out-of-the-money calls to collect premium in bearish markets',
        score: 8.3
      },
      {
        name: 'Long Put',
        description: 'Profit from a decline in stock price with limited risk',
        score: 7.5
      },
      {
        name: 'Protective Put',
        description: 'Hedge long stock positions against downside risk',
        score: 7.0
      }
    ];
  } else { // NEUTRAL or unknown
    return [
      {
        name: 'Iron Condor',
        description: 'Profit from low volatility with limited risk',
        score: 8.0
      },
      {
        name: 'Straddle',
        description: 'Profit from large moves in either direction',
        score: 7.2
      },
      {
        name: 'Calendar Spread',
        description: 'Take advantage of time decay and volatility differences',
        score: 6.8
      }
    ];
  }
}

// Get FMP insights (earnings, gainers, losers, indices, etc.) with fallback for indices
async function getFmpInsights() {
  try {
    const today = new Date().toISOString().split('T')[0];
    const weekLater = new Date(Date.now() + 7 * 24 * 60 * 60 * 1000).toISOString().split('T')[0];

    const [
      economicCalendar,
      earningsCalendar,
      gainers,
      losers,
      indices
    ] = await Promise.all([
      axios.get(`https://financialmodelingprep.com/api/v3/economic_calendar`, {
        params: { from: today, to: weekLater, apikey: FMP_API_KEY }
      }),
      axios.get(`https://financialmodelingprep.com/api/v3/earning_calendar`, {
        params: { from: today, to: weekLater, apikey: FMP_API_KEY }
      }),
      axios.get(`https://financialmodelingprep.com/api/v3/gainers`, {
        params: { apikey: FMP_API_KEY }
      }),
      axios.get(`https://financialmodelingprep.com/api/v3/losers`, {
        params: { apikey: FMP_API_KEY }
      }),
      axios.get(`https://financialmodelingprep.com/api/v3/quotes/index`, {
        params: { apikey: FMP_API_KEY }
      })
    ]);

    // Filter for US-only data
    const usEconomicCalendar = economicCalendar.data.filter((e: any) => 
      e.country === 'US' && ['Medium', 'High'].includes(e.impact)
    );

    const usEarningsCalendar = earningsCalendar.data.filter((e: any) => 
      ['NYSE', 'NASDAQ'].includes(e.exchange)
    );

    const usGainers = gainers.data.filter((g: any) => 
      ['NYSE', 'NASDAQ'].includes(g.exchangeShortName)
    );

    const usLosers = losers.data.filter((l: any) => 
      ['NYSE', 'NASDAQ'].includes(l.exchangeShortName)
    );

    let usIndices = indices.data.filter((i: any) => 
      ['^GSPC', '^DJI', '^IXIC', '^VIX', 'SPY', 'QQQ', 'DIA'].includes(i.symbol)
    );
    // Fallback to Yahoo if indices are empty
    if (!Array.isArray(usIndices) || usIndices.length === 0) {
      try {
        const [spResp, djiResp, ndqResp, vixResp] = await Promise.all([
          axios.get('https://query1.finance.yahoo.com/v8/finance/chart/%5EGSPC?range=1d&interval=1d'),
          axios.get('https://query1.finance.yahoo.com/v8/finance/chart/%5EDJI?range=1d&interval=1d'),
          axios.get('https://query1.finance.yahoo.com/v8/finance/chart/%5EIXIC?range=1d&interval=1d'),
          axios.get('https://query1.finance.yahoo.com/v8/finance/chart/%5EVIX?range=1d&interval=1d'),
        ]);
        usIndices = [
          {
            symbol: '^GSPC',
            price: spResp.data.chart.result[0].meta.regularMarketPrice,
            change: spResp.data.chart.result[0].meta.regularMarketChange
          },
          {
            symbol: '^DJI',
            price: djiResp.data.chart.result[0].meta.regularMarketPrice,
            change: djiResp.data.chart.result[0].meta.regularMarketChange
          },
          {
            symbol: '^IXIC',
            price: ndqResp.data.chart.result[0].meta.regularMarketPrice,
            change: ndqResp.data.chart.result[0].meta.regularMarketChange
          },
          {
            symbol: '^VIX',
            price: vixResp.data.chart.result[0].meta.regularMarketPrice,
            change: vixResp.data.chart.result[0].meta.regularMarketChange
          }
        ];
      } catch (err) {
        console.error('Yahoo indices fallback failed:', err.message || err);
        usIndices = [];
      }
    }

    return {
      economic_calendar: usEconomicCalendar,
      earnings_calendar: usEarningsCalendar,
      top_gainers: usGainers.slice(0, 10),
      top_losers: usLosers.slice(0, 10),
      indices: usIndices
    };
  } catch (error) {
    console.error('Error fetching FMP insights:', error);
    return {
      economic_calendar: [],
      earnings_calendar: [],
      top_gainers: [],
      top_losers: [],
      indices: []
    };
  }
}

// Generate mock trade ideas (simplified version)
async function getTradeIdeas(): Promise<TradeIdea[]> {
  // Mock trade ideas - in production, this would use options chain data
  const mockTrades: TradeIdea[] = [
    {
      ticker: 'AAPL',
      strategy: 'Bull Put Spread',
      expiry: '2025-01-17',
      details: 'Sell 180 Put',
      cost: 0.85,
      metric_name: 'Prob. of Success',
      metric_value: '78.5%',
      max_profit: 0.85,
      max_loss: 4.15,
      risk_reward_ratio: 0.20,
      iv_rank: 45.2,
      current_iv: 28.5
    },
    {
      ticker: 'MSFT',
      strategy: 'Bear Call Spread',
      expiry: '2025-01-17',
      details: 'Sell 420 Call',
      cost: 0.92,
      metric_name: 'Prob. of Success',
      metric_value: '82.1%',
      max_profit: 0.92,
      max_loss: 4.08,
      risk_reward_ratio: 0.23,
      iv_rank: 38.7,
      current_iv: 24.2
    }
  ];

  return mockTrades;
}

// Main report generation function
async function generateReport() {
  try {
    const [
      vixData,
      marketSentiment,
      topStrategies,
      fmpInsights,
      tradeIdeas
    ] = await Promise.all([
      getVixData(),
      getMarketSentiment(),
      getTopStrategies(),
      getFmpInsights(),
      getTradeIdeas()
    ]);

    return {
      generated_at: new Date().toISOString(),
      vix_data: vixData,
      market_sentiment: marketSentiment,
      top_strategies: topStrategies,
      trade_ideas: tradeIdeas,
      economic_calendar: fmpInsights.economic_calendar,
      earnings_calendar: fmpInsights.earnings_calendar,
      top_gainers: fmpInsights.top_gainers,
      top_losers: fmpInsights.top_losers,
      indices: fmpInsights.indices
    };
  } catch (error) {
    console.error('Error generating report:', error);
    throw error;
  }
}

// API key middleware
app.use((req, res, next) => {
  const apiKey = req.get('x-api-key');
  if (process.env.API_SECRET_KEY && apiKey !== process.env.API_SECRET_KEY) {
    // Set CORS headers manually for error responses
    res.set('Access-Control-Allow-Origin', req.get('Origin') || '*');
    res.set('Access-Control-Allow-Headers', 'Origin, X-Requested-With, Content-Type, Accept, x-api-key');
    res.set('Access-Control-Allow-Methods', 'GET, POST, OPTIONS');
    return res.status(403).json({ error: 'Forbidden' });
  }
  next();
});

// API Routes
app.get('/report.json', async (req: express.Request, res: express.Response) => {
  try {
    const now = Date.now();
    
    // Check cache
    if (cachedReportData && (now - lastCacheTime) < CACHE_DURATION) {
      return res.json(cachedReportData);
    }

    // Generate new report
    const reportData = await generateReport();
    
    // Update cache
    cachedReportData = reportData;
    lastCacheTime = now;

    res.json(reportData);
  } catch (error) {
    console.error('Error in /report.json:', error);
    res.status(500).json({ error: 'Failed to generate report' });
  }
});

app.get('/news.json', async (req: express.Request, res: express.Response) => {
  try {
    // Mock news data for now
    const newsData = [
      {
        title: 'Market Update: Indices Show Mixed Performance',
        summary: 'Major indices showed mixed results today with tech stocks leading gains.',
        url: 'https://example.com/news/1',
        published_at: new Date().toISOString(),
        source: 'MarketNews'
      }
    ];

    res.json(newsData);
  } catch (error) {
    console.error('Error in /news.json:', error);
    res.status(500).json({ error: 'Failed to fetch news' });
  }
});

// Health check endpoint
app.get('/health', (req: express.Request, res: express.Response) => {
  res.json({ status: 'healthy', timestamp: new Date().toISOString() });
});

// Utility: Calculate IV Rank (mocked for now)
async function getIvRank(symbol: string): Promise<{iv_rank: number, current_iv: number}> {
  // TODO: Replace with real calculation from options data
  // For now, return random plausible values
  return {
    iv_rank: Math.round(Math.random() * 100),
    current_iv: 15 + Math.random() * 35
  };
}

// Utility: Get recent price trend (mocked for now)
async function getTrend(symbol: string): Promise<string> {
  // TODO: Replace with real trend analysis
  const trends = ['Uptrend', 'Downtrend', 'Sideways'];
  return trends[Math.floor(Math.random() * trends.length)];
}

// Decision logic for trade recommendation
function getTradeRecommendation(iv_rank: number, current_iv: number, trend: string): {recommendation: string, rationale: string, volatility_level: string} {
  let volatility_level = 'Medium';
  if (iv_rank >= 60) volatility_level = 'High';
  else if (iv_rank <= 30) volatility_level = 'Low';

  if (volatility_level === 'High' && trend === 'Sideways') {
    return {
      recommendation: 'Sell Options',
      rationale: 'High IV Rank and stable price: good for premium selling.',
      volatility_level
    };
  } else if (volatility_level === 'High' && trend !== 'Sideways') {
    return {
      recommendation: 'Scalp or Sell Options',
      rationale: 'High volatility and movement: scalp or sell premium.',
      volatility_level
    };
  } else if (volatility_level === 'Low') {
    return {
      recommendation: 'Buy Options',
      rationale: 'Low IV: options are cheap, consider long calls/puts.',
      volatility_level
    };
  } else if (volatility_level === 'Medium' && trend !== 'Sideways') {
    return {
      recommendation: 'Scalp or Buy Options',
      rationale: 'Some movement, not much premium: scalp or buy options.',
      volatility_level
    };
  } else {
    return {
      recommendation: 'Do Nothing',
      rationale: 'No clear edge or illiquid setup.',
      volatility_level
    };
  }
}

// POST /trade_recommendations
app.post('/trade_recommendations', async (req, res) => {
  try {
    const symbols = req.body.symbols;
    if (!Array.isArray(symbols) || symbols.length === 0) {
      return res.status(400).json({ error: 'Missing or invalid symbols list' });
    }
    // Create a cache key based on sorted symbols
    const cacheKey = symbols.slice().sort().join(',');
    const now = Date.now();
    if (
      tradeRecCache[cacheKey] &&
      (now - tradeRecCache[cacheKey].timestamp) < TRADE_REC_CACHE_DURATION
    ) {
      return res.json(tradeRecCache[cacheKey].data);
    }
    const results = await Promise.all(symbols.map(async (symbol: string) => {
      const { iv_rank, current_iv } = await getIvRank(symbol);
      const trend = await getTrend(symbol);
      const { recommendation, rationale, volatility_level } = getTradeRecommendation(iv_rank, current_iv, trend);
      return {
        symbol,
        iv_rank,
        current_iv,
        trend,
        volatility_level,
        recommendation,
        rationale
      };
    }));
    // Cache the result
    tradeRecCache[cacheKey] = { data: results, timestamp: now };
    res.json(results);
  } catch (error) {
    console.error('Error in /trade_recommendations:', error);
    res.status(500).json({ error: 'Failed to generate trade recommendations' });
  }
});

// Export the Express app as a Firebase Cloud Function
export const api = functions.https.onRequest(app); 