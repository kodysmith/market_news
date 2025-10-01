# 🎯 Opportunity Database - Comprehensive Trading Opportunity Scanner

## **📊 Overview**

The Opportunity Database is a comprehensive system that scans **110 top stocks** across **11 major sectors** to identify trading opportunities. It stores all data locally in SQLite for **zero-cost querying** and provides training data for ML model development.

## **🏗️ Architecture**

### **Sector Coverage (11 Sectors)**
- **Technology** (AAPL, MSFT, NVDA, GOOGL, META, TSLA, AVGO, ORCL, CRM, ADBE)
- **Healthcare** (JNJ, UNH, PFE, ABBV, MRK, TMO, ABT, DHR, BMY, AMGN)
- **Financials** (BRK-B, JPM, BAC, WFC, GS, MS, C, BLK, AXP, SPGI)
- **Consumer Discretionary** (AMZN, HD, MCD, NKE, SBUX, LOW, TJX, BKNG, CMG, ABNB)
- **Consumer Staples** (PG, KO, PEP, WMT, COST, CL, KMB, GIS, K, HSY)
- **Industrials** (BA, CAT, HON, UPS, RTX, LMT, GE, MMM, DE, EMR)
- **Energy** (XOM, CVX, COP, EOG, SLB, PXD, MPC, VLO, KMI, PSX)
- **Utilities** (NEE, DUK, SO, D, AEP, EXC, SRE, XEL, PEG, WEC)
- **Real Estate** (AMT, PLD, CCI, EQIX, PSA, O, SPG, WELL, EXR, AVB)
- **Materials** (LIN, APD, SHW, FCX, NEM, ECL, DOW, PPG, DD, IFF)
- **Communication Services** (GOOGL, META, NFLX, DIS, CMCSA, VZ, T, CHTR, TMUS, EA)

### **Data Sources**
- **Yahoo Finance API** - Real-time financial data
- **Real fundamental metrics** - P/E, revenue growth, margins, debt ratios
- **Real technical indicators** - RSI, MACD, moving averages, volume
- **Real market data** - Prices, market cap, analyst ratings

## **🚀 Quick Start**

### **1. Run Full Scan**
```bash
cd /Users/kody/base/MarketNews/QuantEngine
python3 opportunity_database.py
```

### **2. Query Opportunities**
```bash
# Show top opportunities
python3 query_opportunities.py --top

# Show database statistics
python3 query_opportunities.py --stats

# Filter by sector
python3 query_opportunities.py --sector Technology

# Filter by ticker
python3 query_opportunities.py --ticker AAPL

# Export training data
python3 query_opportunities.py --training-data
```

### **3. Schedule Regular Scans**
```bash
# Run scheduler (scans every 4 hours during market hours)
python3 schedule_opportunity_scanner.py
```

## **📈 Opportunity Detection**

### **Scoring System (0-100)**
- **Technical Score (40%)** - RSI, MACD, moving averages, volume, trend
- **Fundamental Score (40%)** - P/E, revenue growth, margins, debt, cash flow
- **Sector Score (20%)** - Relative sector performance

### **Opportunity Types**
- **STRONG_BUY** (Score ≥ 80) - High conviction opportunities
- **BUY** (Score 65-79) - Good opportunities
- **HOLD** (Score 35-64) - Neutral positions
- **SELL** (Score 20-34) - Weak positions
- **STRONG_SELL** (Score < 20) - High conviction shorts

### **Real Example Results**
```
🏆 TOP OPPORTUNITIES (Score >= 70)
Ticker    Sector                Type      Score  Price    Target   R/R
T         Communication        BUY       74.0   $27.52   $29.62   2.0
VZ        Communication        BUY       78.0   $43.83   $44.38   2.0
META      Communication        BUY       76.0   $717.34  $779.72  2.0
GOOGL     Communication        BUY       78.0   $244.90  $254.72  2.0
AVB       Real Estate          BUY       72.0   $191.35  $195.05  2.0
```

## **💾 Database Schema**

### **Opportunities Table**
```sql
CREATE TABLE opportunities (
    id INTEGER PRIMARY KEY,
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
);
```

### **Sector Performance Table**
```sql
CREATE TABLE sector_performance (
    id INTEGER PRIMARY KEY,
    sector TEXT NOT NULL,
    scan_date TIMESTAMP NOT NULL,
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
);
```

### **Training Data Table**
```sql
CREATE TABLE training_data (
    id INTEGER PRIMARY KEY,
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
);
```

## **🔍 Query Examples**

### **Basic Queries**
```bash
# Top 20 opportunities
python3 query_opportunities.py --top --limit 20

# Technology sector only
python3 query_opportunities.py --sector Technology

# High conviction opportunities (score ≥ 80)
python3 query_opportunities.py --min-score 80

# Specific ticker history
python3 query_opportunities.py --ticker AAPL --days 30

# Export to CSV
python3 query_opportunities.py --top --format csv > opportunities.csv

# Export to JSON
python3 query_opportunities.py --sector Healthcare --format json
```

### **Advanced Queries**
```bash
# Strong buy opportunities in last 3 days
python3 query_opportunities.py --type STRONG_BUY --days 3

# Opportunities with score 70-85
python3 query_opportunities.py --min-score 70 --max-score 85

# Export training data for ML
python3 query_opportunities.py --training-data --days 30
```

## **📊 Current Database Stats**

```
📊 DATABASE STATISTICS
==================================================
Total Opportunities: 21
Date Range: 2025-10-01T13:07:12 to 2025-10-01T13:09:42

By Type:
  BUY: 20
  STRONG_BUY: 1

By Sector:
  Communication Services: 4
  Healthcare: 4
  Consumer Discretionary: 3
  Financials: 3
  Technology: 3
  Real Estate: 2
  Consumer Staples: 1
  Energy: 1

Average Scores:
  Overall: 76.38
  Technical: 76.43
  Fundamental: 89.52
  Sector: 50.0
```

## **🤖 Training Data for ML**

### **Export Training Data**
```bash
python3 query_opportunities.py --training-data --days 30
```

### **Features Available**
- **Technical:** RSI, MACD, moving averages, volume, trend
- **Fundamental:** P/E, revenue growth, profit margin, debt/equity
- **Market:** Market cap, relative strength, analyst rating
- **Temporal:** Scan date, time-based features

### **Target Variables**
- **Opportunity Type:** STRONG_BUY, BUY, HOLD, SELL, STRONG_SELL
- **Actual Returns:** 1-day, 1-week, 1-month, 3-month returns
- **Prediction Accuracy:** Model performance metrics

## **⏰ Scheduling**

### **Automatic Scanning**
The scheduler runs scans at:
- **9:00 AM EST** - Morning scan
- **1:00 PM EST** - Midday scan
- **4:00 PM EST** - Evening scan
- **Saturday 10:00 AM EST** - Weekend scan

### **Manual Scheduling**
```bash
# Run scheduler
python3 schedule_opportunity_scanner.py

# Run one-time scan
python3 opportunity_database.py
```

## **💡 Key Benefits**

### **1. Zero-Cost Querying**
- All data stored locally in SQLite
- No API rate limits or costs
- Instant query responses

### **2. Comprehensive Coverage**
- 110 top stocks across 11 sectors
- Real-time fundamental and technical analysis
- Historical tracking with timestamps

### **3. Training Data Generation**
- Export data for ML model training
- Feature engineering capabilities
- Performance tracking over time

### **4. Professional-Grade Analysis**
- Hedge fund-level scoring methodology
- Real financial data from Yahoo Finance
- Risk/reward calculations

### **5. Flexible Querying**
- Filter by ticker, sector, score, date range
- Multiple output formats (table, CSV, JSON)
- Command-line and programmatic access

## **🔧 Technical Details**

### **Rate Limiting**
- 0.5-1.0 second delays between stock requests
- 1-2 second delays between sector scans
- Prevents API rate limiting

### **Error Handling**
- Graceful handling of delisted stocks
- Continues scanning if individual stocks fail
- Comprehensive logging

### **Data Freshness**
- Price data: Real-time (15-20 min delay)
- Financials: Updated quarterly
- Technical indicators: Calculated from real prices

## **📈 Performance Metrics**

### **Scan Performance**
- **Total scan time:** ~5-10 minutes for all 110 stocks
- **Opportunities found:** 15-25 per scan (varies by market conditions)
- **Database size:** ~1-2 MB per 1000 opportunities

### **Query Performance**
- **Simple queries:** < 100ms
- **Complex filters:** < 500ms
- **Full database scan:** < 1 second

## **🚀 Next Steps**

1. **Run regular scans** to build historical data
2. **Export training data** for ML model development
3. **Set up alerts** for high-score opportunities
4. **Integrate with trading systems** for automated execution
5. **Develop ML models** using the training data

---

**This system provides institutional-grade opportunity scanning with zero ongoing costs and unlimited querying capabilities!** 🎯
