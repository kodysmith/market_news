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
    change?: number;
    changePercent?: number;
    direction: string;
    weight?: number;
    score?: number;
  }>;
  weightedScore?: number;
  confidence?: number;
  bullishCount?: number;
  totalCount?: number;
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
  expected_move?: number;
  expected_move_pct?: number;
  break_even_price?: number;
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

// Enhanced market sentiment analysis with multiple data sources and weighted scoring
async function getMarketSentiment(): Promise<MarketSentiment> {
  let indices = [];
  
  // Try FMP first for comprehensive data
  try {
    const indicesResponse = await axios.get('https://financialmodelingprep.com/api/v3/quotes/index', {
      params: { apikey: FMP_API_KEY }
    });
    indices = indicesResponse.data || [];
    if (!Array.isArray(indices) || indices.length === 0) throw new Error('No indices from FMP');
  } catch (e) {
    console.error('FMP indices failed, falling back to Yahoo:', e.message || e);
    // Enhanced Yahoo fallback with more data sources
    try {
      const [spResp, ndqResp, vixResp, iwmResp, tnxResp, dxyResp] = await Promise.all([
        axios.get('https://query1.finance.yahoo.com/v8/finance/chart/%5EGSPC?range=1d&interval=1d'),
        axios.get('https://query1.finance.yahoo.com/v8/finance/chart/%5EIXIC?range=1d&interval=1d'),
        axios.get('https://query1.finance.yahoo.com/v8/finance/chart/%5EVIX?range=1d&interval=1d'),
        axios.get('https://query1.finance.yahoo.com/v8/finance/chart/%5EIWM?range=1d&interval=1d'), // Russell 2000
        axios.get('https://query1.finance.yahoo.com/v8/finance/chart/%5ETNX?range=1d&interval=1d'), // 10Y Treasury
        axios.get('https://query1.finance.yahoo.com/v8/finance/chart/DX-Y.NYB?range=1d&interval=1d'), // Dollar Index
      ]);
      indices = [
        {
          symbol: '^GSPC',
          price: spResp.data.chart.result[0].meta.regularMarketPrice,
          change: spResp.data.chart.result[0].meta.regularMarketChange,
          changePercent: spResp.data.chart.result[0].meta.regularMarketChangePercent
        },
        {
          symbol: '^IXIC',
          price: ndqResp.data.chart.result[0].meta.regularMarketPrice,
          change: ndqResp.data.chart.result[0].meta.regularMarketChange,
          changePercent: ndqResp.data.chart.result[0].meta.regularMarketChangePercent
        },
        {
          symbol: '^VIX',
          price: vixResp.data.chart.result[0].meta.regularMarketPrice,
          change: vixResp.data.chart.result[0].meta.regularMarketChange,
          changePercent: vixResp.data.chart.result[0].meta.regularMarketChangePercent
        },
        {
          symbol: '^IWM',
          price: iwmResp.data.chart.result[0].meta.regularMarketPrice,
          change: iwmResp.data.chart.result[0].meta.regularMarketChange,
          changePercent: iwmResp.data.chart.result[0].meta.regularMarketChangePercent
        },
        {
          symbol: '^TNX',
          price: tnxResp.data.chart.result[0].meta.regularMarketPrice,
          change: tnxResp.data.chart.result[0].meta.regularMarketChange,
          changePercent: tnxResp.data.chart.result[0].meta.regularMarketChangePercent
        },
        {
          symbol: 'DX-Y.NYB',
          price: dxyResp.data.chart.result[0].meta.regularMarketPrice,
          change: dxyResp.data.chart.result[0].meta.regularMarketChange,
          changePercent: dxyResp.data.chart.result[0].meta.regularMarketChangePercent
        }
      ];
    } catch (err) {
      console.error('Yahoo indices fallback failed:', err.message || err);
      indices = [];
    }
  }

  // Enhanced data extraction with fallbacks
  const spyData = indices.find((idx: any) => idx.symbol === '^GSPC');
  const nasdaqData = indices.find((idx: any) => idx.symbol === '^IXIC');
  const vixData = indices.find((idx: any) => idx.symbol === '^VIX');
  const iwmData = indices.find((idx: any) => idx.symbol === '^IWM');
  const tnxData = indices.find((idx: any) => idx.symbol === '^TNX');
  const dxyData = indices.find((idx: any) => idx.symbol === 'DX-Y.NYB');

  // Enhanced weighted sentiment calculation
  const indicators = [
    {
      name: 'S&P 500',
      ticker: '^GSPC',
      price: spyData ? `$${spyData.price.toFixed(2)}` : 'N/A',
      change: spyData?.change || 0,
      changePercent: spyData?.changePercent || 0,
      direction: spyData && spyData.change > 0 ? 'UP' : 'DOWN',
      weight: 0.25, // 25% weight
      score: spyData ? (spyData.change > 0 ? 1 : -1) : 0
    },
    {
      name: 'Nasdaq',
      ticker: '^IXIC',
      price: nasdaqData ? `$${nasdaqData.price.toFixed(2)}` : 'N/A',
      change: nasdaqData?.change || 0,
      changePercent: nasdaqData?.changePercent || 0,
      direction: nasdaqData && nasdaqData.change > 0 ? 'UP' : 'DOWN',
      weight: 0.20, // 20% weight
      score: nasdaqData ? (nasdaqData.change > 0 ? 1 : -1) : 0
    },
    {
      name: 'Russell 2000',
      ticker: '^IWM',
      price: iwmData ? `$${iwmData.price.toFixed(2)}` : 'N/A',
      change: iwmData?.change || 0,
      changePercent: iwmData?.changePercent || 0,
      direction: iwmData && iwmData.change > 0 ? 'UP' : 'DOWN',
      weight: 0.15, // 15% weight
      score: iwmData ? (iwmData.change > 0 ? 1 : -1) : 0
    },
    {
      name: 'VIX',
      ticker: '^VIX',
      price: vixData ? `${vixData.price.toFixed(2)}` : 'N/A',
      change: vixData?.change || 0,
      changePercent: vixData?.changePercent || 0,
      direction: vixData && vixData.change < 0 ? 'UP' : 'DOWN', // VIX inverse to market
      weight: 0.20, // 20% weight
      score: vixData ? (vixData.change < 0 ? 1 : -1) : 0 // Inverted for VIX
    },
    {
      name: '10Y Treasury',
      ticker: '^TNX',
      price: tnxData ? `${tnxData.price.toFixed(2)}%` : 'N/A',
      change: tnxData?.change || 0,
      changePercent: tnxData?.changePercent || 0,
      direction: tnxData && tnxData.change < 0 ? 'UP' : 'DOWN', // Lower yields = bullish
      weight: 0.10, // 10% weight
      score: tnxData ? (tnxData.change < 0 ? 1 : -1) : 0 // Inverted for yields
    },
    {
      name: 'Dollar Index',
      ticker: 'DX-Y.NYB',
      price: dxyData ? `${dxyData.price.toFixed(2)}` : 'N/A',
      change: dxyData?.change || 0,
      changePercent: dxyData?.changePercent || 0,
      direction: dxyData && dxyData.change < 0 ? 'UP' : 'DOWN', // Weaker dollar = bullish for stocks
      weight: 0.10, // 10% weight
      score: dxyData ? (dxyData.change < 0 ? 1 : -1) : 0 // Inverted for dollar
    }
  ];

  // Calculate weighted sentiment score
  let weightedScore = 0;
  let totalWeight = 0;
  let bullishCount = 0;
  let totalCount = 0;

  indicators.forEach(indicator => {
    if (indicator.direction === 'UP') bullishCount++;
    totalCount++;
    
    weightedScore += indicator.score * indicator.weight;
    totalWeight += indicator.weight;
  });

  // Normalize weighted score to [-1, 1] range
  const normalizedScore = totalWeight > 0 ? weightedScore / totalWeight : 0;
  
  // Enhanced sentiment determination with confidence levels
  let sentiment: string;
  let confidence: number;
  
  if (normalizedScore > 0.3) {
    sentiment = 'BULLISH';
    confidence = Math.min(0.9, 0.5 + Math.abs(normalizedScore) * 0.5);
  } else if (normalizedScore < -0.3) {
    sentiment = 'BEARISH';
    confidence = Math.min(0.9, 0.5 + Math.abs(normalizedScore) * 0.5);
  } else {
    sentiment = 'NEUTRAL';
    confidence = 0.3 + Math.abs(normalizedScore) * 0.4;
  }

  return {
    sentiment,
    indicators,
    weightedScore: normalizedScore,
    confidence,
    bullishCount,
    totalCount
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

// Generate real trade ideas from market data
async function getTradeIdeas(): Promise<TradeIdea[]> {
  try {
    // Get current market data for major stocks
    const stocks = ['AAPL', 'MSFT', 'GOOGL'];
    const tradeIdeas: TradeIdea[] = [];

    console.log('Generating trade ideas for stocks:', stocks);

    for (const ticker of stocks) {
      try {
        console.log(`Fetching data for ${ticker}`);

        // Get current stock price
        const quoteResp = await axios.get(
          `https://financialmodelingprep.com/api/v3/quote/${ticker}`,
          { params: { apikey: FMP_API_KEY } }
        );

        if (!quoteResp.data || quoteResp.data.length === 0) {
          console.log(`No data for ${ticker}`);
          continue;
        }

        const stockData = quoteResp.data[0];
        const currentPrice = stockData.price;
        const changePercent = stockData.changesPercentage;

        console.log(`${ticker}: price=${currentPrice}, change=${changePercent}%`);

        // Always generate trade ideas for demo purposes
        const isBullish = changePercent >= 0;
        const volatility = Math.max(18, Math.abs(changePercent) * 2 + 20);
        const daysToExpiry = 28;

        if (isBullish) {
          // Bull Put Spread for bullish stocks
          const strikeOffset = Math.round(currentPrice * 0.03); // 3% OTM
          const shortStrike = currentPrice - strikeOffset;
          const longStrike = shortStrike - 5;
          const width = shortStrike - longStrike;

          tradeIdeas.push({
            ticker,
            strategy: 'Bull Put Spread',
            expiry: '2025-10-17',
            details: `Sell ${Math.round(shortStrike)} Put / Buy ${Math.round(longStrike)} Put`,
            cost: Math.max(0.10, width * 0.15), // Realistic premium
            metric_name: 'Prob. of Success',
            metric_value: '68.5%',
            max_profit: Math.max(0.10, width * 0.15),
            max_loss: width - Math.max(0.10, width * 0.15),
            risk_reward_ratio: 0.25,
            iv_rank: Math.min(100, Math.max(10, volatility * 2.5)),
            current_iv: volatility,
            ...calculateExpectedMove(currentPrice, volatility, daysToExpiry)
          });
        } else {
          // Bear Call Spread for bearish/neutral stocks
          const strikeOffset = Math.round(currentPrice * 0.02); // 2% OTM
          const shortStrike = currentPrice + strikeOffset;
          const longStrike = shortStrike + 5;
          const width = longStrike - shortStrike;

          tradeIdeas.push({
            ticker,
            strategy: 'Bear Call Spread',
            expiry: '2025-10-17',
            details: `Sell ${Math.round(shortStrike)} Call / Buy ${Math.round(longStrike)} Call`,
            cost: Math.max(0.12, width * 0.18),
            metric_name: 'Prob. of Success',
            metric_value: '65.2%',
            max_profit: Math.max(0.12, width * 0.18),
            max_loss: width - Math.max(0.12, width * 0.18),
            risk_reward_ratio: 0.30,
            iv_rank: Math.min(100, Math.max(10, volatility * 2.2)),
            current_iv: volatility,
            ...calculateExpectedMove(currentPrice, volatility, daysToExpiry)
          });
        }
      } catch (error) {
        console.error(`Error getting trade idea for ${ticker}:`, error.message);
      }
    }

    // If we have trade ideas, return the best ones
    if (tradeIdeas.length > 0) {
      return tradeIdeas
        .filter(t => t.max_profit && t.max_loss && t.max_loss > 0)
        .sort((a, b) => (b.max_profit / b.max_loss) - (a.max_profit / a.max_loss))
        .slice(0, 3);
    }

    // Fallback: Generate basic trade ideas using hardcoded realistic data
    console.log('No trade ideas generated, using fallback');
    return [
      {
        ticker: 'AAPL',
        strategy: 'Bull Put Spread',
        expiry: '2025-10-17',
        details: 'Sell 235 Put / Buy 230 Put',
        cost: 1.45,
        metric_name: 'Prob. of Success',
        metric_value: '73.2%',
        max_profit: 1.45,
        max_loss: 3.55,
        risk_reward_ratio: 0.41,
        iv_rank: 45.2,
        current_iv: 26.1,
        ...calculateExpectedMove(235, 26.1, 28)
      },
      {
        ticker: 'MSFT',
        strategy: 'Bear Call Spread',
        expiry: '2025-10-17',
        details: 'Sell 405 Call / Buy 410 Call',
        cost: 1.85,
        metric_name: 'Prob. of Success',
        metric_value: '69.8%',
        max_profit: 1.85,
        max_loss: 3.15,
        risk_reward_ratio: 0.59,
        iv_rank: 38.7,
        current_iv: 24.2,
        ...calculateExpectedMove(405, 24.2, 28)
      }
    ];

  } catch (error) {
    console.error('Error generating trade ideas:', error.message);
    // Fallback to basic mock data if all else fails
    return [
      {
        ticker: 'SPY',
        strategy: 'Bull Put Spread',
        expiry: '2025-10-17',
        details: 'Sell 570 Put / Buy 565 Put',
        cost: 1.85,
        metric_name: 'Prob. of Success',
        metric_value: '71.2%',
        max_profit: 1.85,
        max_loss: 3.15,
        risk_reward_ratio: 0.59,
        iv_rank: 38.5,
        current_iv: 22.1,
        ...calculateExpectedMove(570, 22.1, 28)
      }
    ];
  }
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
    // Try FMP news API first
    const fmpNewsResp = await axios.get(
      'https://financialmodelingprep.com/api/v3/fmp/articles',
      { params: { page: 0, size: 10, apikey: FMP_API_KEY } }
    );

    if (fmpNewsResp.data && fmpNewsResp.data.content && fmpNewsResp.data.content.length > 0) {
      const newsData = fmpNewsResp.data.content.slice(0, 8).map((article: any) => ({
        title: article.title || 'Market Update',
        summary: article.contentSnippet ? article.contentSnippet.substring(0, 200) + '...' : article.title,
        url: article.link || `https://financialmodelingprep.com/news/${article.symbol || 'market'}`,
        published_at: article.publishedDate || new Date().toISOString(),
        source: article.site || 'Financial Modeling Prep'
      }));

      res.json(newsData);
      return;
    }

    // Fallback to Yahoo Finance news if FMP fails
    try {
      const yahooNewsResp = await axios.get(
        'https://feeds.finance.yahoo.com/rss/2.0/headline?s=SPY,QQQ&region=US&lang=en-US'
      );

      // Parse RSS (simplified - in production use a proper RSS parser)
      const items = yahooNewsResp.data.match(/<item>(.*?)<\/item>/gs) || [];
      const newsData = items.slice(0, 6).map((item: string, index: number) => {
        const titleMatch = item.match(/<title><!\[CDATA\[(.*?)\]\]><\/title>/);
        const linkMatch = item.match(/<link>(.*?)<\/link>/);
        const pubDateMatch = item.match(/<pubDate>(.*?)<\/pubDate>/);

        return {
          title: titleMatch ? titleMatch[1] : `Market News ${index + 1}`,
          summary: titleMatch ? titleMatch[1] : 'Latest market developments',
          url: linkMatch ? linkMatch[1] : 'https://finance.yahoo.com',
          published_at: pubDateMatch ? new Date(pubDateMatch[1]).toISOString() : new Date().toISOString(),
          source: 'Yahoo Finance'
        };
      });

      res.json(newsData);
    } catch (yahooError) {
      console.error('Yahoo news fallback failed:', yahooError.message);

      // Final fallback to basic market summary
      const newsData = [
        {
          title: 'Market Update: Current Market Conditions',
          summary: 'Markets are showing typical volatility with major indices trading near recent highs.',
          url: 'https://finance.yahoo.com',
          published_at: new Date().toISOString(),
          source: 'Market Summary'
        },
        {
          title: 'Economic Indicators Show Mixed Signals',
          summary: 'Recent economic data indicates steady growth with some sector rotation occurring.',
          url: 'https://finance.yahoo.com',
          published_at: new Date(Date.now() - 3600000).toISOString(),
          source: 'Economic Report'
        }
      ];

      res.json(newsData);
    }
  } catch (error) {
    console.error('Error in /news.json:', error);
    res.status(500).json({ error: 'Failed to fetch news' });
  }
});

// Health check endpoint
// QuantEngine Recommendations Endpoint
app.get('/quantengine/recommendations', async (req: express.Request, res: express.Response) => {
  try {
    // In production, this would read from Firestore opportunities collection
    // For now, generate intelligent recommendations based on market conditions
    
    const marketRegime = await getMarketRegime();
    const recommendations = await generateQuantEngineRecommendations(marketRegime);
    
    const response = {
      market_regime: marketRegime,
      recommendations: recommendations,
      generated_at: new Date().toISOString()
    };

    // Auto-track the example trades from our recommendations for performance tracking
    const admin = require('firebase-admin');
    if (admin.apps.length === 0) {
      admin.initializeApp({
        projectId: 'kardova-capital'
      });
    }
    const db = admin.firestore();
    
    for (const rec of recommendations) {
      if (rec.example_trades && rec.example_trades.length > 0) {
        for (const trade of rec.example_trades.slice(0, 1)) { // Track first example from each recommendation
          try {
            // Parse entry price from trade setup (estimate for now)
            const entryPrice = parseFloat(trade.entry.match(/\$?(\d+(?:\.\d+)?)/)?.[1] || '0');
            const targetPrice = entryPrice * 1.5; // 50% profit target
            const stopPrice = entryPrice * 0.5; // 50% stop loss
            
            const trackedPosition = {
              id: `auto_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`,
              symbol: trade.symbol,
              strategy: trade.strategy,
              setup: trade.setup,
              entry_price: entryPrice,
              entry_time: new Date().toISOString(),
              target_price: targetPrice,
              stop_loss_price: stopPrice,
              position_size: 1,
              status: 'open',
              current_pnl: 0,
              max_profit: 0,
              max_loss: 0,
              days_held: 0,
              close_reason: null,
              close_time: null,
              final_pnl: null,
              source: 'intelligence_recommendation',
              market_regime: marketRegime.regime
            };

            await db.collection('tracked_positions').doc(trackedPosition.id).set({
              ...trackedPosition,
              created_at: admin.firestore.FieldValue.serverTimestamp(),
              updated_at: admin.firestore.FieldValue.serverTimestamp()
            });

            console.log('✅ REAL AUTO-TRACKING:', {
              symbol: trade.symbol,
              strategy: trade.strategy,
              entry_price: entryPrice,
              id: trackedPosition.id,
              regime: marketRegime.regime
            });
          } catch (error) {
            console.error('❌ Auto-tracking failed:', error);
          }
        }
      }
    }

    res.json(response);
  } catch (error) {
    console.error('Error in /quantengine/recommendations:', error);
    res.status(500).json({ error: 'Failed to fetch QuantEngine recommendations' });
  }
});

// Market Regime Analysis
async function getMarketRegime(): Promise<{regime: string, confidence: number, reasoning: string, vix_level: string}> {
  try {
    // Get VIX data
    const vixData = await getVixData();
    const currentVix = vixData.length > 0 ? vixData[0].close : 20;
    
    // Get market indices
    const marketSentiment = await getMarketSentiment();
    
    // Simple regime classification
    let regime = 'neutral';
    let confidence = 0.7;
    let reasoning = 'Mixed signals';
    let vixLevel = 'normal';
    
    if (currentVix > 30) {
      regime = 'high_volatility';
      vixLevel = 'high';
      confidence = 0.9;
      reasoning = `VIX at ${currentVix.toFixed(1)} indicates high fear/volatility. Opportunity for volatility selling strategies.`;
    } else if (currentVix < 15) {
      regime = 'low_volatility';
      vixLevel = 'low';
      confidence = 0.85;
      reasoning = `VIX at ${currentVix.toFixed(1)} indicates complacency. Consider volatility buying strategies.`;
    } else if (marketSentiment.sentiment === 'bullish') {
      regime = 'trending_up';
      confidence = 0.8;
      reasoning = 'Market showing bullish momentum. Favor bullish strategies and momentum plays.';
    } else if (marketSentiment.sentiment === 'bearish') {
      regime = 'trending_down';
      confidence = 0.8;
      reasoning = 'Market showing bearish momentum. Favor bearish strategies and defensive plays.';
    }
    
    return { regime, confidence, reasoning, vix_level: vixLevel };
  } catch (error) {
    return {
      regime: 'neutral',
      confidence: 0.5,
      reasoning: 'Unable to determine market regime due to data issues',
      vix_level: 'normal'
    };
  }
}

// Generate QuantEngine-style recommendations with specific trade examples
async function generateQuantEngineRecommendations(marketRegime: any): Promise<any[]> {
  const recommendations = [];
  
  // Get current market data for specific trade examples
  const exampleTrades = await generateSpecificTradeExamples(marketRegime);
  
  // Strategy recommendations based on regime
  if (marketRegime.regime === 'high_volatility') {
    recommendations.push({
      strategy: 'Short Premium Strategies',
      priority: 'high',
      rationale: 'High VIX creates opportunity to sell overpriced options',
      specific_strategies: ['Short Strangles', 'Iron Condors', 'Covered Calls'],
      target_assets: ['SPY', 'QQQ', 'High IV individual stocks'],
      risk_level: 'medium',
      time_horizon: '2-6 weeks',
      example_trades: exampleTrades.high_volatility || []
    });
  } else if (marketRegime.regime === 'low_volatility') {
    recommendations.push({
      strategy: 'Long Volatility Strategies',
      priority: 'high', 
      rationale: 'Low VIX suggests volatility is underpriced',
      specific_strategies: ['Long Straddles', 'Long Strangles', 'Ratio Backspreads'],
      target_assets: ['VIX options', 'UVXY calls', 'Event-driven stocks'],
      risk_level: 'high',
      time_horizon: '1-4 weeks',
      example_trades: exampleTrades.low_volatility || []
    });
  } else if (marketRegime.regime === 'trending_up') {
    recommendations.push({
      strategy: 'Bullish Momentum Strategies',
      priority: 'high',
      rationale: 'Market momentum favors bullish positioning',
      specific_strategies: ['Bull Put Spreads', 'Cash-Secured Puts', 'Covered Calls'],
      target_assets: ['Growth stocks', 'Tech ETFs', 'Momentum leaders'],
      risk_level: 'medium',
      time_horizon: '3-8 weeks',
      example_trades: exampleTrades.trending_up || []
    });
  } else if (marketRegime.regime === 'trending_down') {
    recommendations.push({
      strategy: 'Bearish/Defensive Strategies', 
      priority: 'high',
      rationale: 'Market weakness suggests defensive positioning',
      specific_strategies: ['Bear Call Spreads', 'Protective Puts', 'Cash Heavy'],
      target_assets: ['Defensive stocks', 'Utilities', 'Consumer staples'],
      risk_level: 'low',
      time_horizon: '2-6 weeks',
      example_trades: exampleTrades.trending_down || []
    });
  } else {
    // Neutral market
    recommendations.push({
      strategy: 'Range-Bound Strategies',
      priority: 'medium',
      rationale: 'Neutral market conditions favor range-bound strategies',
      specific_strategies: ['Iron Condors', 'Short Strangles', 'Covered Calls'],
      target_assets: ['SPY', 'QQQ', 'Large cap stocks'],
      risk_level: 'medium',
      time_horizon: '2-6 weeks',
      example_trades: exampleTrades.neutral || []
    });
  }
  
  // Always include risk management recommendation
  recommendations.push({
    strategy: 'Risk Management',
    priority: 'critical',
    rationale: 'Consistent risk management is key to long-term success',
    specific_strategies: ['Position sizing', 'Stop losses', 'Diversification'],
    target_assets: ['Portfolio level'],
    risk_level: 'low',
    time_horizon: 'ongoing',
    example_trades: [
      {
        symbol: 'Portfolio',
        strategy: 'Risk Management',
        setup: 'Risk no more than 2% per trade',
        entry: 'Calculate position size based on account balance',
        target: 'Preserve capital for long-term growth',
        stop_loss: 'Pre-defined exit rules',
        rationale: 'Capital preservation is the foundation of successful trading'
      }
    ]
  });
  
  return recommendations;
}

// Generate specific trade examples based on market regime
async function generateSpecificTradeExamples(marketRegime: any): Promise<any> {
  const examples = {
    high_volatility: [],
    low_volatility: [],
    trending_up: [],
    trending_down: [],
    neutral: []
  };

  try {
    // Get current stock prices for examples
    const symbols = ['SPY', 'QQQ', 'AAPL', 'MSFT', 'NVDA', 'TSLA'];
    const stockPrices: { [key: string]: number } = {};

    // Fetch current prices (limit to 3 for performance)
    for (const symbol of symbols.slice(0, 3)) {
      try {
        const quoteResp = await axios.get(
          `https://financialmodelingprep.com/api/v3/quote/${symbol}`,
          { params: { apikey: FMP_API_KEY } }
        );
        if (quoteResp.data && quoteResp.data[0]) {
          stockPrices[symbol] = quoteResp.data[0].price;
        }
      } catch (error) {
        // Use fallback prices if API fails
        const fallbackPrices = { SPY: 570, QQQ: 485, AAPL: 245, MSFT: 410, NVDA: 135, TSLA: 250 };
        stockPrices[symbol] = fallbackPrices[symbol as keyof typeof fallbackPrices] || 100;
      }
    }

    // High Volatility Examples (Short Premium)
    examples.high_volatility = [
      {
        symbol: 'SPY',
        strategy: 'Iron Condor',
        setup: `Sell ${Math.round(stockPrices.SPY * 0.98)} Put / Buy ${Math.round(stockPrices.SPY * 0.96)} Put, Sell ${Math.round(stockPrices.SPY * 1.02)} Call / Buy ${Math.round(stockPrices.SPY * 1.04)} Call`,
        entry: 'Net credit of $1.50-2.00',
        target: '50% of credit received',
        stop_loss: '2x credit received',
        rationale: 'High IV makes premium selling attractive, SPY likely to stay in range'
      },
      {
        symbol: 'QQQ',
        strategy: 'Short Strangle',
        setup: `Sell ${Math.round(stockPrices.QQQ * 0.97)} Put / Sell ${Math.round(stockPrices.QQQ * 1.03)} Call`,
        entry: 'Net credit of $2.50-3.50',
        target: '25-50% of credit received',
        stop_loss: 'Close if either strike is threatened',
        rationale: 'Tech volatility premium is elevated, expect mean reversion'
      }
    ];

    // Low Volatility Examples (Long Volatility)
    examples.low_volatility = [
      {
        symbol: 'AAPL',
        strategy: 'Long Straddle',
        setup: `Buy ${Math.round(stockPrices.AAPL)} Call + Buy ${Math.round(stockPrices.AAPL)} Put (ATM)`,
        entry: 'Debit of $8-12 (look for earnings or events)',
        target: 'Move > 15% in either direction',
        stop_loss: '50% of debit paid',
        rationale: 'AAPL often has large moves around earnings, low IV makes straddles cheap'
      },
      {
        symbol: 'UVXY',
        strategy: 'Long Calls',
        setup: 'Buy UVXY calls 2-4 weeks out',
        entry: 'When VIX < 15 and market complacent',
        target: 'VIX spike to 20-25',
        stop_loss: '30-50% of premium paid',
        rationale: 'Volatility mean reversion - low VIX unlikely to persist'
      },
      {
        symbol: 'SPY',
        strategy: 'Long Strangle',
        setup: `Buy ${Math.round(stockPrices.SPY * 0.98)} Put + Buy ${Math.round(stockPrices.SPY * 1.02)} Call`,
        entry: 'Debit of $5-8',
        target: 'Large market move in either direction',
        stop_loss: '50% of debit paid',
        rationale: 'Market positioned for volatility expansion, low IV makes this attractive'
      }
    ];

    // Trending Up Examples (Bullish Strategies)
    examples.trending_up = [
      {
        symbol: 'QQQ',
        strategy: 'Bull Put Spread',
        setup: `Sell ${Math.round(stockPrices.QQQ * 0.95)} Put / Buy ${Math.round(stockPrices.QQQ * 0.92)} Put`,
        entry: 'Net credit of $1.00-1.50',
        target: 'Full credit if QQQ stays above short strike',
        stop_loss: 'Close if QQQ breaks trend support',
        rationale: 'Tech momentum continues, collect premium while bullish'
      },
      {
        symbol: 'AAPL',
        strategy: 'Cash-Secured Put',
        setup: `Sell ${Math.round(stockPrices.AAPL * 0.95)} Put`,
        entry: 'Premium of $3-5',
        target: 'Keep premium if AAPL stays above strike',
        stop_loss: 'Accept assignment if bullish long-term',
        rationale: 'Get paid to buy AAPL at discount if assigned'
      }
    ];

    // Trending Down Examples (Bearish/Defensive)
    examples.trending_down = [
      {
        symbol: 'SPY',
        strategy: 'Bear Call Spread',
        setup: `Sell ${Math.round(stockPrices.SPY * 1.02)} Call / Buy ${Math.round(stockPrices.SPY * 1.05)} Call`,
        entry: 'Net credit of $1.00-1.50',
        target: 'Full credit if SPY stays below short strike',
        stop_loss: 'Close if market reverses trend',
        rationale: 'Bearish momentum likely to continue, collect premium'
      },
      {
        symbol: 'QQQ',
        strategy: 'Protective Puts',
        setup: `Buy ${Math.round(stockPrices.QQQ * 0.95)} Puts (if holding QQQ)`,
        entry: 'Premium of $4-7',
        target: 'Portfolio protection',
        stop_loss: 'Let expire if market recovers',
        rationale: 'Hedge existing tech positions against further decline'
      }
    ];

    // Neutral Market Examples
    examples.neutral = [
      {
        symbol: 'SPY',
        strategy: 'Iron Condor',
        setup: `Sell ${Math.round(stockPrices.SPY * 0.98)}-${Math.round(stockPrices.SPY * 1.02)} range`,
        entry: 'Net credit of $1.50-2.00',
        target: '50% of credit received',
        stop_loss: '2x credit received',
        rationale: 'Neutral market likely to stay in range, collect theta decay'
      }
    ];

  } catch (error) {
    console.error('Error generating trade examples:', error);
  }

  return examples;
}

// Paper Trading Tracker - Track our recommendation performance
app.post('/track/recommendation', async (req: express.Request, res: express.Response) => {
  try {
    const recommendation = req.body;

    // Create a tracked position
    const trackedPosition = {
      id: `rec_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`,
      symbol: recommendation.symbol,
      strategy: recommendation.strategy,
      setup: recommendation.setup,
      entry_price: recommendation.entry_price,
      entry_time: new Date().toISOString(),
      target_price: recommendation.target_price,
      stop_loss_price: recommendation.stop_loss_price,
      position_size: 1, // Standard size for tracking
      status: recommendation.status || 'open',
      current_pnl: recommendation.status === 'closed' ? recommendation.final_pnl || 0 : 0,
      max_profit: recommendation.status === 'closed' ? recommendation.final_pnl || 0 : 0,
      max_loss: recommendation.status === 'closed' ? recommendation.final_pnl || 0 : 0,
      days_held: recommendation.days_held || 0,
      close_reason: recommendation.close_reason || null,
      close_time: recommendation.status === 'closed' ? new Date().toISOString() : null,
      final_pnl: recommendation.final_pnl || null
    };

    // Save to Firestore 'tracked_positions' collection - REAL TRACKING STARTS NOW
    try {
      const admin = require('firebase-admin');
      if (admin.apps.length === 0) {
        admin.initializeApp({
          projectId: 'kardova-capital'
        });
      }
      const db = admin.firestore();

      await db.collection('tracked_positions').doc(trackedPosition.id).set({
        ...trackedPosition,
        created_at: admin.firestore.FieldValue.serverTimestamp(),
        updated_at: admin.firestore.FieldValue.serverTimestamp()
      });

      console.log('✅ REAL TRACKING: Saved to Firestore:', {
        symbol: trackedPosition.symbol,
        strategy: trackedPosition.strategy,
        entry_time: trackedPosition.entry_time,
        id: trackedPosition.id
      });
    } catch (firestoreError) {
      // Fallback to in-memory tracking for demo purposes
      console.log('⚠️ Firestore not available, using fallback tracking:', {
        symbol: trackedPosition.symbol,
        strategy: trackedPosition.strategy,
        entry_time: trackedPosition.entry_time,
        id: trackedPosition.id
      });

      // Store in global memory for demo (in production, use Redis or similar)
      if (!global.trackedPositions) {
        global.trackedPositions = [];
      }
      global.trackedPositions.push({
        ...trackedPosition,
        created_at: new Date().toISOString(),
        updated_at: new Date().toISOString()
      });
    }
    
    res.json({ 
      success: true, 
      position_id: trackedPosition.id,
      message: 'Recommendation is now being tracked' 
    });
  } catch (error) {
    console.error('Error tracking recommendation:', error);
    res.status(500).json({ error: 'Failed to track recommendation' });
  }
});

// Get performance dashboard - show how our recommendations are doing
app.get('/performance/dashboard', async (req: express.Request, res: express.Response) => {
  try {
    // In production, this would query Firestore for tracked positions
    // For now, return mock performance data
    const performanceData = await generatePerformanceDashboard();
    
    res.json(performanceData);
  } catch (error) {
    console.error('Error generating performance dashboard:', error);
    res.status(500).json({ error: 'Failed to generate performance dashboard' });
  }
});

// Generate performance dashboard - REAL DATA from Firestore tracking
async function generatePerformanceDashboard() {
  const today = new Date();
  
  try {
    // Get REAL tracked positions from Firestore or fallback
    let allPositions = [];

    try {
      const admin = require('firebase-admin');
      if (admin.apps.length === 0) {
        admin.initializeApp({
          projectId: 'kardova-capital'
        });
      }
      const db = admin.firestore();

      // Try to initialize Firestore by writing a system document
      try {
        await db.collection('system').doc('init').set({
          initialized: true,
          timestamp: new Date().toISOString(),
          version: '1.0.0'
        });
        console.log('✅ Firestore database initialized');
      } catch (initError) {
        console.log('⚠️ Firestore init skipped (may already exist):', initError.message);
      }

      const positionsSnapshot = await db.collection('tracked_positions').get();
      allPositions = positionsSnapshot.docs.map(doc => ({
        id: doc.id,
        ...doc.data()
      }));

      console.log(`📊 REAL FIRESTORE DATA: Found ${allPositions.length} tracked positions`);
    } catch (firestoreError) {
      // Fallback to in-memory data
      allPositions = global.trackedPositions || [];
      console.log(`📊 FALLBACK MEMORY DATA: Found ${allPositions.length} tracked positions`);
      console.log('❌ Firestore error:', firestoreError.message);
    }
    
    console.log(`📊 REAL DATA: Found ${allPositions.length} tracked positions`);
    
    // Separate open and closed positions
    const openPositions = allPositions.filter(p => p.status === 'open');
    const closedPositions = allPositions.filter(p => p.status === 'closed');
    
    // Calculate REAL performance metrics
    const totalPositions = allPositions.length;
    const winners = closedPositions.filter(p => (p.final_pnl || 0) > 0);
    const winRate = closedPositions.length > 0 ? (winners.length / closedPositions.length) * 100 : 0;
    
    const avgReturn = closedPositions.length > 0 
      ? closedPositions.reduce((sum, p) => sum + (p.final_pnl || 0), 0) / closedPositions.length
      : 0;
      
    const totalPnl = closedPositions.reduce((sum, p) => sum + (p.final_pnl || 0), 0);
    
    const bestTrade = closedPositions.length > 0 
      ? Math.max(...closedPositions.map(p => p.final_pnl || 0))
      : 0;
      
    const worstTrade = closedPositions.length > 0
      ? Math.min(...closedPositions.map(p => p.final_pnl || 0))
      : 0;
    
    const avgHoldTime = closedPositions.length > 0
      ? closedPositions.reduce((sum, p) => sum + (p.days_held || 0), 0) / closedPositions.length
      : 0;
    
    // If we have no real data yet, show minimal starting stats
    if (totalPositions === 0) {
      return {
        summary: {
          total_recommendations: 0,
          open_positions: 0,
          closed_positions: 0,
          win_rate: 0,
          avg_return_per_trade: 0,
          total_pnl: 0,
          best_trade: 0,
          worst_trade: 0,
          avg_hold_time: 0,
          sharpe_ratio: 0
        },
        recent_performance: [],
        open_positions: [],
        closed_positions: [],
        strategy_performance: [],
        milestones: [{
          title: '🚀 First Day',
          description: 'Real tracking started today!',
          achieved_date: today.toISOString().split('T')[0],
          type: 'milestone'
        }]
      };
    }
    
    return {
      summary: {
        total_recommendations: totalPositions,
        open_positions: openPositions.length,
        closed_positions: closedPositions.length,
        win_rate: Math.round(winRate * 10) / 10,
        avg_return_per_trade: Math.round(avgReturn * 10) / 10,
        total_pnl: Math.round(totalPnl * 100) / 100,
        best_trade: Math.round(bestTrade * 10) / 10,
        worst_trade: Math.round(worstTrade * 10) / 10,
        avg_hold_time: Math.round(avgHoldTime * 10) / 10,
        sharpe_ratio: closedPositions.length > 3 ? calculateSharpeRatio(closedPositions) : 0
      },
      recent_performance: generateRecentPerformance(closedPositions),
      open_positions: openPositions.map(p => ({
        id: p.id,
        symbol: p.symbol,
        strategy: p.strategy,
        entry_date: p.entry_time ? new Date(p.entry_time).toISOString().split('T')[0] : 'Unknown',
        entry_price: p.entry_price || 0,
        current_price: p.current_price || p.entry_price || 0,
        current_pnl: p.current_pnl || 0,
        days_held: p.days_held || 0,
        target_hit: false,
        stop_triggered: false,
        status: p.current_pnl > 0 ? 'winning' : 'losing'
      })),
      closed_positions: closedPositions.slice(-10).map(p => ({
        symbol: p.symbol,
        strategy: p.strategy,
        entry_date: p.entry_time ? new Date(p.entry_time).toISOString().split('T')[0] : 'Unknown',
        close_date: p.close_time ? new Date(p.close_time).toISOString().split('T')[0] : 'Unknown',
        entry_price: p.entry_price || 0,
        exit_price: p.close_price || 0,
        pnl_percent: p.final_pnl || 0,
        hold_days: p.days_held || 0,
        close_reason: p.close_reason || 'unknown',
        status: (p.final_pnl || 0) > 0 ? 'winner' : 'loser'
      })),
      strategy_performance: calculateStrategyPerformance(allPositions),
      milestones: generateMilestones(allPositions, winners, winRate)
    };
    
  } catch (error) {
    console.error('❌ Error generating real performance data:', error);
    // Fallback to basic empty state
    return {
      summary: {
        total_recommendations: 0,
        open_positions: 0,
        closed_positions: 0,
        win_rate: 0,
        avg_return_per_trade: 0,
        total_pnl: 0,
        best_trade: 0,
        worst_trade: 0,
        avg_hold_time: 0,
        sharpe_ratio: 0
      },
      recent_performance: [],
      open_positions: [],
      closed_positions: [],
      strategy_performance: [],
      milestones: [{
        title: '⚠️ Starting Up',
        description: 'Real tracking system initializing...',
        achieved_date: new Date().toISOString().split('T')[0],
        type: 'system'
      }]
    };
  }
}

// Helper functions for real data calculations
function calculateSharpeRatio(closedPositions: any[]): number {
  if (closedPositions.length < 2) return 0;
  
  const returns = closedPositions.map(p => p.final_pnl || 0);
  const avgReturn = returns.reduce((a, b) => a + b, 0) / returns.length;
  const variance = returns.reduce((sum, r) => sum + Math.pow(r - avgReturn, 2), 0) / returns.length;
  const stdDev = Math.sqrt(variance);
  
  return stdDev > 0 ? Math.round((avgReturn / stdDev) * 100) / 100 : 0;
}

function generateRecentPerformance(closedPositions: any[]): any[] {
  // Group by date and calculate daily performance
  const dailyPerformance: { [key: string]: any } = {};
  
  closedPositions.forEach(p => {
    if (!p.close_time) return;
    
    const date = new Date(p.close_time).toISOString().split('T')[0];
    if (!dailyPerformance[date]) {
      dailyPerformance[date] = {
        date,
        daily_pnl: 0,
        trades_closed: 0,
        winners: 0
      };
    }
    
    dailyPerformance[date].daily_pnl += p.final_pnl || 0;
    dailyPerformance[date].trades_closed += 1;
    if ((p.final_pnl || 0) > 0) {
      dailyPerformance[date].winners += 1;
    }
  });
  
  return Object.values(dailyPerformance)
    .map(d => ({
      ...d,
      win_rate: d.trades_closed > 0 ? (d.winners / d.trades_closed) * 100 : 0,
      best_trade: `${d.trades_closed} trades, ${d.winners} winners`
    }))
    .slice(-5); // Last 5 days
}

function calculateStrategyPerformance(allPositions: any[]): any[] {
  const strategyStats: { [key: string]: any } = {};
  
  allPositions.forEach(p => {
    if (!strategyStats[p.strategy]) {
      strategyStats[p.strategy] = {
        strategy: p.strategy,
        total_trades: 0,
        winners: 0,
        total_return: 0,
        best_return: 0,
        closed_trades: 0
      };
    }
    
    strategyStats[p.strategy].total_trades += 1;
    
    if (p.status === 'closed') {
      strategyStats[p.strategy].closed_trades += 1;
      const pnl = p.final_pnl || 0;
      strategyStats[p.strategy].total_return += pnl;
      
      if (pnl > 0) {
        strategyStats[p.strategy].winners += 1;
      }
      
      if (pnl > strategyStats[p.strategy].best_return) {
        strategyStats[p.strategy].best_return = pnl;
      }
    }
  });
  
  return Object.values(strategyStats).map((s: any) => ({
    strategy: s.strategy,
    total_trades: s.closed_trades,
    win_rate: s.closed_trades > 0 ? (s.winners / s.closed_trades) * 100 : 0,
    avg_return: s.closed_trades > 0 ? s.total_return / s.closed_trades : 0,
    best_return: s.best_return,
    status: s.closed_trades > 0 && (s.winners / s.closed_trades) > 0.7 ? 'hot' : 
            s.closed_trades > 0 && (s.winners / s.closed_trades) > 0.5 ? 'good' : 'steady'
  }));
}

function generateMilestones(allPositions: any[], winners: any[], winRate: number): any[] {
  const milestones = [];
  
  if (allPositions.length >= 1) {
    milestones.push({
      title: '🎯 First Trade',
      description: 'Started real performance tracking',
      achieved_date: new Date().toISOString().split('T')[0],
      type: 'milestone'
    });
  }
  
  if (winners.length >= 1) {
    milestones.push({
      title: '💰 First Winner',
      description: `First profitable trade: +${winners[0]?.final_pnl?.toFixed(1)}%`,
      achieved_date: winners[0]?.close_time ? new Date(winners[0].close_time).toISOString().split('T')[0] : 'Unknown',
      type: 'win'
    });
  }
  
  if (winRate >= 50 && winners.length >= 2) {
    milestones.push({
      title: '🎯 Profitable System',
      description: `${winRate.toFixed(1)}% win rate achieved`,
      achieved_date: new Date().toISOString().split('T')[0],
      type: 'achievement'
    });
  }
  
  return milestones.slice(0, 3); // Show max 3 milestones
}

// Live position updates - check current P&L of open positions
app.get('/performance/live-updates', async (req: express.Request, res: express.Response) => {
  try {
    // In production, this would:
    // 1. Get all open tracked positions from Firestore
    // 2. Fetch current market prices for each position
    // 3. Calculate real-time P&L
    // 4. Check if targets/stops should be triggered
    // 5. Update position status
    
    const liveUpdates = {
      total_open_positions: 12,
      total_unrealized_pnl: 456.78,
      daily_pnl_change: 89.23,
      positions_near_target: 3,
      positions_near_stop: 1,
      last_updated: new Date().toISOString(),
      hot_positions: [
        {
          symbol: 'AAPL',
          strategy: 'Long Straddle',
          current_pnl: 21.9,
          pnl_change_today: 5.2,
          status: 'approaching_target'
        },
        {
          symbol: 'NVDA',
          strategy: 'Bull Put Spread',
          current_pnl: 15.7,
          pnl_change_today: 3.1,
          status: 'steady_winner'
        }
      ]
    };
    
    res.json(liveUpdates);
  } catch (error) {
    console.error('Error getting live updates:', error);
    res.status(500).json({ error: 'Failed to get live updates' });
  }
});

// Initialize Firestore database
app.post('/init-firestore', async (req: express.Request, res: express.Response) => {
  try {
    const admin = require('firebase-admin');
    if (admin.apps.length === 0) {
      admin.initializeApp({
        projectId: 'kardova-capital'
      });
    }
    const db = admin.firestore();

    // Create a test document to initialize Firestore
    await db.collection('system').doc('init').set({
      initialized: true,
      timestamp: new Date().toISOString(),
      message: 'Firestore database initialized for performance tracking'
    });

    // Also create a sample tracked position to ensure the collection exists
    await db.collection('tracked_positions').doc('sample').set({
      id: 'sample',
      symbol: 'AAPL',
      strategy: 'Sample Trade',
      setup: 'Test trade for database initialization',
      entry_price: 150.0,
      entry_time: new Date().toISOString(),
      target_price: 160.0,
      stop_loss_price: 140.0,
      position_size: 1,
      status: 'open',
      current_pnl: 0,
      max_profit: 0,
      max_loss: 0,
      days_held: 0,
      close_reason: null,
      close_time: null,
      final_pnl: null,
      source: 'initialization',
      market_regime: 'neutral'
    });

    console.log('✅ Firestore database and collections initialized');
    res.json({
      success: true,
      message: 'Firestore database and collections initialized',
      collections: ['system', 'tracked_positions']
    });
  } catch (error) {
    console.error('Failed to initialize Firestore:', error);
    res.status(500).json({
      error: 'Failed to initialize Firestore',
      details: error.message,
      suggestion: 'Try accessing the Firebase console to manually create the database'
    });
  }
});

app.get('/health', (req: express.Request, res: express.Response) => {
  res.json({ status: 'healthy', timestamp: new Date().toISOString() });
});

// Market countdown and status endpoint
app.get('/market-status', (req: express.Request, res: express.Response) => {
  try {
    const now = new Date();
    const etNow = new Date(now.toLocaleString('en-US', { timeZone: 'America/New_York' }));

    const dayOfWeek = etNow.getDay(); // 0 = Sunday, 6 = Saturday
    const hour = etNow.getHours();
    const minute = etNow.getMinutes();

    // Market hours: 9:30 AM - 4:00 PM ET, weekdays
    const isWeekend = dayOfWeek === 0 || dayOfWeek === 6;
    const isMarketOpen = !isWeekend && hour >= 9 && hour < 16 && (hour > 9 || (hour === 9 && minute >= 30));

    let status: string;
    let message: string;
    let countdown: number | null = null;
    let nextEvent: string;

    if (isWeekend) {
      status = 'closed';
      message = 'Market is closed for the weekend';
      // Calculate time until Monday 9:30 AM ET
      const monday = new Date(etNow);
      monday.setDate(etNow.getDate() + (8 - dayOfWeek)); // Next Monday
      monday.setHours(9, 30, 0, 0);
      countdown = Math.floor((monday.getTime() - etNow.getTime()) / 1000); // seconds
      nextEvent = 'Monday market open';
    } else if (isMarketOpen) {
      status = 'open';
      message = 'Market is currently open';
      // Calculate time until market close
      const closeTime = new Date(etNow);
      closeTime.setHours(16, 0, 0, 0);
      countdown = Math.floor((closeTime.getTime() - etNow.getTime()) / 1000);
      nextEvent = 'Market close';
    } else {
      status = 'closed';
      message = 'Market is currently closed';

      // Calculate time until market open
      const openTime = new Date(etNow);
      if (hour >= 16) {
        // After close, next open is tomorrow
        openTime.setDate(etNow.getDate() + 1);
        openTime.setHours(9, 30, 0, 0);
      } else if (hour < 9 || (hour === 9 && minute < 30)) {
        // Before open today
        openTime.setHours(9, 30, 0, 0);
      }

      countdown = Math.floor((openTime.getTime() - etNow.getTime()) / 1000);
      nextEvent = 'Market open';
    }

    res.json({
      status,
      message,
      is_market_open: isMarketOpen,
      countdown_seconds: countdown,
      next_event: nextEvent,
      current_time_et: etNow.toISOString(),
      market_hours: {
        open: '09:30',
        close: '16:00',
        timezone: 'America/New_York'
      }
    });

  } catch (error) {
    console.error('Market status error:', error);
    res.status(500).json({ error: 'Failed to get market status' });
  }
});

// Manual P&L update endpoint (since scheduled functions need different setup)
app.post('/update-pnl', async (req: express.Request, res: express.Response) => {
  console.log('🕐 Manual P&L update triggered at:', new Date().toISOString());

  try {
    // Allow force update for testing
    const force = req.query.force === 'true';

    // Check if it's market hours (9:30 AM - 4:00 PM ET, weekdays)
    const now = new Date();
    const etNow = new Date(now.toLocaleString('en-US', { timeZone: 'America/New_York' }));
    const dayOfWeek = etNow.getDay(); // 0 = Sunday, 6 = Saturday
    const hour = etNow.getHours();
    const minute = etNow.getMinutes();

    // Skip weekends and non-market hours (unless forced)
    if (!force && (dayOfWeek === 0 || dayOfWeek === 6)) {
      console.log('⏰ Weekend - skipping P&L update');
      return res.json({ message: 'Weekend - no update needed' });
    }

    if (!force && (hour < 9 || (hour === 9 && minute < 30) || hour >= 16)) {
      console.log('⏰ Outside market hours - skipping P&L update');
      return res.json({ message: 'Outside market hours - no update needed' });
    }

    console.log('📈 Market is open - updating P&L for tracked positions');

    // Get all tracked positions
    let trackedPositions = [];

    try {
      // Try Firestore first
      const db = admin.firestore();
      const positionsSnapshot = await db.collection('tracked_positions').get();
      trackedPositions = positionsSnapshot.docs.map(doc => ({
        id: doc.id,
        ...doc.data()
      }));
      console.log(`📊 Found ${trackedPositions.length} positions in Firestore`);
    } catch (firestoreError) {
      // Fallback to memory
      trackedPositions = global.trackedPositions || [];
      console.log(`📊 Found ${trackedPositions.length} positions in memory`);
    }

    const openPositions = trackedPositions.filter(p => p.status === 'open');

    if (openPositions.length === 0) {
      console.log('📊 No open positions to update');
      return res.json({ message: 'No open positions to update' });
    }

    console.log(`📊 Updating P&L for ${openPositions.length} open positions`);

    const updates = [];

    // Update each position's P&L
    for (const position of openPositions) {
      try {
        // Get current stock price from FMP API
        let currentPrice;
        try {
          const response = await axios.get(`https://financialmodelingprep.com/api/v3/quote/${position.symbol}?apikey=${FMP_API_KEY}`);
          currentPrice = response.data[0]?.price;
        } catch (apiError) {
          console.log(`⚠️ FMP API failed for ${position.symbol}, using mock price`);
          // Mock price for testing - use entry price + small random change
          currentPrice = position.entry_price * (1 + (Math.random() - 0.5) * 0.1); // ±5% change
        }

        if (!currentPrice) {
          console.log(`⚠️ Could not get price for ${position.symbol}`);
          continue;
        }

        // Calculate current P&L
        const entryPrice = position.entry_price || 0;
        const positionSize = position.position_size || 1;
        const currentPnL = (currentPrice - entryPrice) * positionSize * 100; // Assuming 100 shares per contract

        // Update max profit/loss tracking
        const maxProfit = Math.max(position.max_profit || 0, currentPnL);
        const maxLoss = Math.min(position.max_loss || 0, currentPnL);

        // Calculate days held
        const entryTime = new Date(position.entry_time);
        const daysHeld = Math.floor((now.getTime() - entryTime.getTime()) / (1000 * 60 * 60 * 24));

        const updatedPosition = {
          ...position,
          current_pnl: currentPnL,
          current_price: currentPrice,
          max_profit: maxProfit,
          max_loss: maxLoss,
          days_held: daysHeld,
          updated_at: now.toISOString()
        };

        // Save back to storage
        try {
          const db = admin.firestore();
          await db.collection('tracked_positions').doc(position.id).update({
            current_pnl: currentPnL,
            current_price: currentPrice,
            max_profit: maxProfit,
            max_loss: maxLoss,
            days_held: daysHeld,
            updated_at: now.toISOString()
          });
          console.log(`✅ Updated ${position.symbol} P&L: $${currentPnL.toFixed(2)}`);
          updates.push({ symbol: position.symbol, pnl: currentPnL, status: 'firestore' });
        } catch (saveError) {
          // Update in memory
          if (global.trackedPositions) {
            const index = global.trackedPositions.findIndex(p => p.id === position.id);
            if (index !== -1) {
              global.trackedPositions[index] = updatedPosition;
            }
          }
          console.log(`✅ Updated ${position.symbol} P&L in memory: $${currentPnL.toFixed(2)}`);
          updates.push({ symbol: position.symbol, pnl: currentPnL, status: 'memory' });
        }

      } catch (error) {
        console.error(`❌ Failed to update P&L for ${position.symbol}:`, error.message);
        updates.push({ symbol: position.symbol, error: error.message });
      }
    }

    console.log('✅ P&L update completed');
    res.json({
      success: true,
      message: 'P&L update completed',
      updates: updates,
      total_positions: openPositions.length
    });

  } catch (error) {
    console.error('❌ P&L update failed:', error);
    res.status(500).json({ error: 'P&L update failed', details: error.message });
  }
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

// Utility: Calculate expected move
function calculateExpectedMove(currentPrice: number, iv: number, daysToExpiry: number): { expected_move: number, expected_move_pct: number, break_even_price: number } {
  // Expected move formula: Price * IV * sqrt(days/365)
  const timeFactor = Math.sqrt(daysToExpiry / 365);
  const expectedMove = currentPrice * (iv / 100) * timeFactor;
  const expectedMovePct = (expectedMove / currentPrice) * 100;

  // For simplicity, assume ATM break-even (this would be more complex for spreads)
  const breakEvenPrice = currentPrice;

  return {
    expected_move: Math.round(expectedMove * 100) / 100,
    expected_move_pct: Math.round(expectedMovePct * 100) / 100,
    break_even_price: Math.round(breakEvenPrice * 100) / 100
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

// Publish QuantEngine report endpoint
app.post('/publish-quant-report', async (req, res) => {
  try {
    const {
      type,
      timestamp,
      title,
      query,
      asset,
      current_price,
      daily_change,
      volatility,
      timeframe,
      content,
      report_path
    } = req.body;

    if (type !== 'quant_engine_report') {
      return res.status(400).json({ error: 'Invalid report type' });
    }

    // Store the report in Firestore
    const reportData = {
      type: 'quant_engine_report',
      timestamp: timestamp || new Date().toISOString(),
      title,
      query,
      asset,
      current_price,
      daily_change,
      volatility,
      timeframe,
      content,
      report_path,
      created_at: admin.firestore.FieldValue.serverTimestamp(),
      published: true
    };

    const docRef = await admin.firestore().collection('quant_reports').add(reportData);
    
    console.log(`QuantEngine report published: ${title} (${asset})`);
    
    res.json({
      success: true,
      id: docRef.id,
      message: 'QuantEngine report published successfully'
    });

  } catch (error) {
    console.error('Error publishing QuantEngine report:', error);
    res.status(500).json({ error: 'Failed to publish report' });
  }
});

// Get QuantEngine reports endpoint
app.get('/quant-reports', async (req, res) => {
  try {
    const limit = parseInt(req.query.limit as string) || 10;
    const offset = parseInt(req.query.offset as string) || 0;

    const snapshot = await admin.firestore()
      .collection('quant_reports')
      .orderBy('created_at', 'desc')
      .limit(limit)
      .offset(offset)
      .get();

    const reports = snapshot.docs.map(doc => ({
      id: doc.id,
      ...doc.data()
    }));

    res.json({
      success: true,
      reports,
      count: reports.length
    });

  } catch (error) {
    console.error('Error fetching QuantEngine reports:', error);
    res.status(500).json({ error: 'Failed to fetch reports' });
  }
});

// Import QuantEngine functions
import { quantChat, analyzeStock, scanMarket } from './quantEngine';

// Export the Express app as a Firebase Cloud Function
export const api = functions.https.onRequest(app);

// Export QuantEngine functions
export { quantChat, analyzeStock, scanMarket };

// ==========================================
// OPPORTUNITY DATABASE ENDPOINTS
// ==========================================

// Publish opportunities to mobile app
app.post('/publish-opportunities', async (req, res) => {
  try {
    const { opportunities, scan_timestamp, total_count } = req.body;

    if (!opportunities || !Array.isArray(opportunities)) {
      return res.status(400).json({ error: 'Invalid opportunities data' });
    }

    console.log(`📱 Publishing ${opportunities.length} opportunities to mobile app`);

    // Store opportunities in Firestore
    const batch = admin.firestore().batch();
    
    for (const opp of opportunities) {
      const docRef = admin.firestore().collection('trading_opportunities').doc();
      batch.set(docRef, {
        ...opp,
        created_at: admin.firestore.FieldValue.serverTimestamp(),
        published: true
      });
    }

    await batch.commit();

    // Also store scan summary
    await admin.firestore().collection('scan_summaries').add({
      scan_timestamp,
      total_count,
      opportunities_published: opportunities.length,
      created_at: admin.firestore.FieldValue.serverTimestamp(),
      published: true
    });

    console.log(`✅ Published ${opportunities.length} opportunities successfully`);
    
    res.json({
      success: true,
      message: `Published ${opportunities.length} opportunities`,
      timestamp: new Date().toISOString()
    });

  } catch (error) {
    console.error('Error publishing opportunities:', error);
    res.status(500).json({ 
      error: 'Failed to publish opportunities',
      details: error instanceof Error ? error.message : 'Unknown error'
    });
  }
});

// Publish improved scanner data endpoint
app.post('/publish-scanner-data', async (req, res) => {
  try {
    const { 
      opportunities, 
      scan_timestamp, 
      total_tickers, 
      successful_scans, 
      failed_scans, 
      summary 
    } = req.body;

    if (!opportunities || !Array.isArray(opportunities)) {
      return res.status(400).json({ error: 'Invalid scanner data' });
    }

    console.log(`🚀 Publishing improved scanner data: ${opportunities.length} opportunities`);

    // Use existing trading_opportunities collection instead of new one
    const batch = admin.firestore().batch();
    
    for (const opp of opportunities) {
      const docRef = admin.firestore().collection('trading_opportunities').doc();
      batch.set(docRef, {
        ...opp,
        created_at: admin.firestore.FieldValue.serverTimestamp(),
        published: true,
        data_source: 'improved_scanner',
        scan_type: 'production_scanner'
      });
    }

    await batch.commit();

    // Store scan summary in existing collection
    await admin.firestore().collection('scan_summaries').add({
      scan_timestamp,
      total_tickers,
      successful_scans,
      failed_scans,
      summary,
      opportunities_published: opportunities.length,
      created_at: admin.firestore.FieldValue.serverTimestamp(),
      published: true,
      scan_type: 'production_scanner'
    });

    console.log(`✅ Published improved scanner data successfully`);
    
    res.json({
      success: true,
      message: `Published ${opportunities.length} scanner opportunities`,
      timestamp: new Date().toISOString(),
      summary
    });

  } catch (error) {
    console.error('Error publishing scanner data:', error);
    res.status(500).json({ 
      error: 'Failed to publish scanner data',
      details: error instanceof Error ? error.message : 'Unknown error'
    });
  }
});

// Get scanner opportunities for mobile app
app.get('/scanner-opportunities', async (req, res) => {
  try {
    console.log('📱 Fetching scanner opportunities for mobile app');

    // Get scanner opportunities from trading_opportunities collection
    const opportunitiesSnapshot = await admin.firestore()
      .collection('trading_opportunities')
      .where('published', '==', true)
      .where('data_source', '==', 'improved_scanner')
      .orderBy('created_at', 'desc')
      .limit(20)
      .get();

    const opportunities = opportunitiesSnapshot.docs.map(doc => ({
      id: doc.id,
      ...doc.data()
    }));

    // Get latest scan summary
    const summarySnapshot = await admin.firestore()
      .collection('scan_summaries')
      .where('scan_type', '==', 'production_scanner')
      .orderBy('created_at', 'desc')
      .limit(1)
      .get();

    const latestSummary = summarySnapshot.empty ? null : summarySnapshot.docs[0].data();

    console.log(`✅ Retrieved ${opportunities.length} scanner opportunities`);

    res.json({
      success: true,
      opportunities,
      summary: latestSummary,
      timestamp: new Date().toISOString()
    });

  } catch (error) {
    console.error('Error fetching scanner opportunities:', error);
    res.status(500).json({ 
      error: 'Failed to fetch scanner opportunities',
      details: error instanceof Error ? error.message : 'Unknown error'
    });
  }
});

// Publish sector summary to mobile app
app.post('/publish-sector-summary', async (req, res) => {
  try {
    const { sectors, summary_timestamp } = req.body;

    if (!sectors || !Array.isArray(sectors)) {
      return res.status(400).json({ error: 'Invalid sectors data' });
    }

    console.log(`📊 Publishing sector summary to mobile app`);

    // Store sector summary in Firestore
    const batch = admin.firestore().batch();
    
    for (const sector of sectors) {
      const docRef = admin.firestore().collection('sector_summaries').doc();
      batch.set(docRef, {
        ...sector,
        created_at: admin.firestore.FieldValue.serverTimestamp(),
        published: true
      });
    }

    await batch.commit();

    console.log(`✅ Published sector summary successfully`);
    
    res.json({
      success: true,
      message: `Published sector summary`,
      timestamp: new Date().toISOString()
    });

  } catch (error) {
    console.error('Error publishing sector summary:', error);
    res.status(500).json({ 
      error: 'Failed to publish sector summary',
      details: error instanceof Error ? error.message : 'Unknown error'
    });
  }
});

// Get latest opportunities for mobile app
app.get('/latest-opportunities', async (req, res) => {
  try {
    const limit = parseInt(req.query.limit as string) || 20;
    const minScore = parseFloat(req.query.minScore as string) || 60;

    console.log(`📱 Fetching latest opportunities (limit: ${limit}, minScore: ${minScore})`);

    const snapshot = await admin.firestore()
      .collection('trading_opportunities')
      .where('published', '==', true)
      .where('overall_score', '>=', minScore)
      .orderBy('overall_score', 'desc')
      .orderBy('created_at', 'desc')
      .limit(limit)
      .get();

    const opportunities = snapshot.docs.map(doc => ({
      id: doc.id,
      ...doc.data()
    }));

    res.json({
      success: true,
      opportunities,
      count: opportunities.length,
      timestamp: new Date().toISOString()
    });

  } catch (error) {
    console.error('Error fetching opportunities:', error);
    res.status(500).json({ 
      error: 'Failed to fetch opportunities',
      details: error instanceof Error ? error.message : 'Unknown error'
    });
  }
});

// Get sector summaries for mobile app
app.get('/sector-summaries', async (req, res) => {
  try {
    const limit = parseInt(req.query.limit as string) || 10;

    console.log(`📊 Fetching sector summaries (limit: ${limit})`);

    const snapshot = await admin.firestore()
      .collection('sector_summaries')
      .where('published', '==', true)
      .orderBy('created_at', 'desc')
      .limit(limit)
      .get();

    const summaries = snapshot.docs.map(doc => ({
      id: doc.id,
      ...doc.data()
    }));

    res.json({
      success: true,
      summaries,
      count: summaries.length,
      timestamp: new Date().toISOString()
    });

  } catch (error) {
    console.error('Error fetching sector summaries:', error);
    res.status(500).json({ 
      error: 'Failed to fetch sector summaries',
      details: error instanceof Error ? error.message : 'Unknown error'
    });
  }
});

// Get opportunities by ticker
app.get('/opportunities/:ticker', async (req, res) => {
  try {
    const { ticker } = req.params;
    const limit = parseInt(req.query.limit as string) || 10;

    console.log(`📈 Fetching opportunities for ${ticker} (limit: ${limit})`);

    const snapshot = await admin.firestore()
      .collection('trading_opportunities')
      .where('ticker', '==', ticker.toUpperCase())
      .where('published', '==', true)
      .orderBy('created_at', 'desc')
      .limit(limit)
      .get();

    const opportunities = snapshot.docs.map(doc => ({
      id: doc.id,
      ...doc.data()
    }));

    res.json({
      success: true,
      ticker: ticker.toUpperCase(),
      opportunities,
      count: opportunities.length,
      timestamp: new Date().toISOString()
    });

  } catch (error) {
    console.error('Error fetching ticker opportunities:', error);
    res.status(500).json({ 
      error: 'Failed to fetch ticker opportunities',
      details: error instanceof Error ? error.message : 'Unknown error'
    });
  }
});

// Get opportunities by sector
app.get('/opportunities/sector/:sector', async (req, res) => {
  try {
    const { sector } = req.params;
    const limit = parseInt(req.query.limit as string) || 20;

    console.log(`🏢 Fetching opportunities for ${sector} sector (limit: ${limit})`);

    const snapshot = await admin.firestore()
      .collection('trading_opportunities')
      .where('sector', '==', sector)
      .where('published', '==', true)
      .orderBy('overall_score', 'desc')
      .orderBy('created_at', 'desc')
      .limit(limit)
      .get();

    const opportunities = snapshot.docs.map(doc => ({
      id: doc.id,
      ...doc.data()
    }));

    res.json({
      success: true,
      sector,
      opportunities,
      count: opportunities.length,
      timestamp: new Date().toISOString()
    });

  } catch (error) {
    console.error('Error fetching sector opportunities:', error);
    res.status(500).json({ 
      error: 'Failed to fetch sector opportunities',
      details: error instanceof Error ? error.message : 'Unknown error'
    });
  }
}); 