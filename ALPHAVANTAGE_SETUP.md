# Alpha Vantage API Integration Setup

## 🎯 **Overview**
Your Bull Put Spread screener now integrates with Alpha Vantage API to fetch real-time options data, current stock prices, and calculate implied volatility metrics.

---

## 🔑 **Getting Your Alpha Vantage API Key**

### **Step 1: Sign Up**
1. Go to [Alpha Vantage](https://www.alphavantage.co/support/#api-key)
2. Click **"Get your free API key today"**
3. Fill out the form with your details
4. **Free tier includes**: 25 requests per day, 5 requests per minute

### **Step 2: Get Premium Access (Recommended)**
For production use, consider upgrading:
- **Basic Plan**: $49.99/month - 1,200 requests/day
- **Premium Plan**: $149.99/month - 5,000 requests/day
- **Enterprise**: Custom pricing for unlimited access

---

## ⚙️ **Configuration**

### **Add Your API Key**
1. Open `/Users/kody/base/MarketNews/market_news_app/.env`
2. Replace the placeholder:
```env
ALPHAVANTAGE_API_KEY=YOUR_ACTUAL_API_KEY_HERE
```

### **Example:**
```env
FMP_API_KEY=your_financial_modeling_prep_key_here
FIREBASE_API_KEY=your_firebase_key_here
ALPHAVANTAGE_API_KEY=ABC123XYZ789YOURKEYHERE
```

---

## 📊 **What the Integration Provides**

### **Real-Time Data:**
- ✅ **Current Stock Prices**: Live pricing for SPY, QQQ, NVDA, etc.
- ✅ **Options Chains**: Complete put/call data with strikes, premiums, Greeks
- ✅ **Implied Volatility**: Real IV calculations for each option
- ✅ **Greeks**: Delta, gamma, theta, vega, rho for risk analysis
- ✅ **Volume & Open Interest**: Liquidity metrics for each strike

### **Calculated Metrics:**
- ✅ **IV Rank**: Historical volatility percentile ranking
- ✅ **Probability of Profit**: Based on delta calculations
- ✅ **Liquidity Scores**: Bid/ask spreads and volume analysis
- ✅ **Technical Scores**: Strike distance from current price
- ✅ **Composite Rankings**: Weighted scoring system

---

## 🔧 **API Endpoints Used**

### **1. Current Stock Price**
```
GET https://www.alphavantage.co/query?function=GLOBAL_QUOTE&symbol=SPY&apikey=YOUR_KEY
```

### **2. Options Chain Data**
```
GET https://www.alphavantage.co/query?function=HISTORICAL_OPTIONS&symbol=SPY&apikey=YOUR_KEY
```

### **3. Intraday Data (for IV calculations)**
```
GET https://www.alphavantage.co/query?function=TIME_SERIES_INTRADAY&symbol=SPY&interval=5min&apikey=YOUR_KEY
```

---

## 🎮 **How It Works**

### **Data Flow:**
1. **User selects symbol** (SPY, QQQ, etc.) in the screener
2. **API fetches**:
   - Current stock price
   - Complete options chain
   - Historical volatility data
3. **Algorithm processes**:
   - Filters options by DTE (10-45 days)
   - Calculates spread combinations
   - Applies quality filters (IV rank, yield, delta)
4. **Results displayed**:
   - Ranked list of best Bull Put Spreads
   - Real-time pricing and Greeks
   - Quality grades and risk levels

### **Filtering Logic:**
```dart
// Example filters applied:
- Days to Expiration: 10-45 days
- IV Rank: ≥ 50% (adjustable)
- Yield: ≥ 33% (adjustable)  
- Delta: 0.15-0.30 (optimal range)
- Bid/Ask Spread: < $0.10 (liquidity)
- Open Interest: > 50 contracts
```

---

## 📈 **Features**

### **Smart Strike Selection:**
- Automatically finds optimal strike combinations
- Ensures proper delta ranges (0.15-0.30)
- Maintains 5-20 point spread widths
- Filters for adequate liquidity

### **Real-Time Calculations:**
- **Credit**: Net premium received (short bid - long ask)
- **Yield**: Credit / spread width
- **Max Profit**: Credit × 100 (per contract)
- **Max Loss**: (Width - Credit) × 100
- **Probability of Profit**: (1 - short put delta) × 100

### **Quality Scoring:**
- **Composite Score**: Weighted algorithm combining:
  - Probability of Profit (40%)
  - Yield (30%)
  - IV Rank (20%)
  - Liquidity (10%)

---

## 🚨 **Rate Limiting**

### **Free Tier Limits:**
- **25 requests/day**
- **5 requests/minute**
- **Suitable for**: Testing and light usage

### **Handling Limits:**
```dart
// The service automatically:
1. Caches responses for 5 minutes
2. Falls back to mock data if rate limited
3. Shows user-friendly error messages
4. Queues requests to respect rate limits
```

---

## 🔄 **Fallback Strategy**

### **If API Fails:**
1. **First**: Try to use cached data (if available)
2. **Second**: Fall back to enhanced mock data with current prices
3. **Third**: Show error message with retry option
4. **Always**: Maintain app functionality

### **Error Messages:**
- `"Using demo data. Add Alpha Vantage API key for live options data."`
- `"Rate limit reached. Please try again in a few minutes."`
- `"API temporarily unavailable. Using cached data."`

---

## 🎯 **Testing the Integration**

### **Step 1: Add API Key**
```bash
# Edit .env file
nano /Users/kody/base/MarketNews/market_news_app/.env

# Add your key:
ALPHAVANTAGE_API_KEY=your_actual_key_here
```

### **Step 2: Restart the App**
```bash
cd /Users/kody/base/MarketNews/market_news_app
flutter run -d chrome
```

### **Step 3: Test the Screener**
1. Navigate to **"Spreads"** tab
2. Select **SPY** from dropdown
3. Adjust filters (DTE, IV Rank, Yield)
4. Look for **real-time data** vs mock data indicators

### **Step 4: Verify Real Data**
- **Real data**: Strike prices near current market levels (~$660 for SPY)
- **Mock data**: Shows "Using demo data" message
- **Live updates**: Prices change when you switch symbols

---

## 🔮 **Future Enhancements**

### **Phase 1: Enhanced Data**
- **Earnings Calendar**: Integrate earnings dates to filter out earnings plays
- **Historical IV**: Store 252-day IV history for accurate IV rank
- **Sector Analysis**: Add sector rotation and relative strength

### **Phase 2: Advanced Features**
- **Backtesting**: Historical performance of generated spreads
- **Paper Trading**: Track hypothetical positions
- **Alerts**: Push notifications for new high-quality setups

### **Phase 3: Portfolio Management**
- **Position Tracking**: Monitor open spreads
- **Greeks Management**: Portfolio-level delta, gamma, theta
- **Risk Management**: Automated stop losses and profit taking

---

## 📊 **Cost Analysis**

### **Free Tier (25 requests/day):**
- **Suitable for**: Personal testing, light usage
- **Limitations**: ~2-3 symbol scans per day
- **Cost**: $0/month

### **Basic Plan ($49.99/month):**
- **Requests**: 1,200/day (50/hour)
- **Suitable for**: Active personal trading
- **Usage**: ~100 symbol scans per day
- **ROI**: Pays for itself with 1-2 successful trades

### **Premium Plan ($149.99/month):**
- **Requests**: 5,000/day (200/hour)  
- **Suitable for**: Professional trading, small fund
- **Usage**: Real-time scanning across all symbols
- **ROI**: Professional-grade options analysis tool

---

## 🎉 **Benefits**

### **Immediate:**
- **Real market data** instead of mock data
- **Live options pricing** and Greeks
- **Current IV ranks** and volatility analysis
- **Professional-grade** screening capabilities

### **Long-term:**
- **Systematic approach** to options trading
- **Consistent profit opportunities** identification
- **Risk management** through proper screening
- **Scalable framework** for portfolio growth

---

Your Bull Put Spread screener is now equipped with institutional-quality options data! 🚀

