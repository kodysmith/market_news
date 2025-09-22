# Bull Put Spread Screener - Mathematical Strategy Framework

## 🎯 **Objective**
Create a systematic, mathematical screener that identifies consistently profitable Bull Put Spread opportunities by scanning option chains across 0-20 DTE (Days to Expiration) with strict quantitative criteria.

---

## 🔑 **Core Criteria for Profitable Bull Put Spreads**

### **1. High Implied Volatility Rank (IVR ≥ 50%)**
- **Logic**: IV must be elevated vs. historical levels to maximize premium collection
- **Calculation**: `IVR = (Current IV - 52w Low IV) / (52w High IV - 52w Low IV) × 100`
- **Threshold**: Only consider spreads where IVR ≥ 50%
- **Rationale**: Selling fear when it's expensive, avoiding low-premium environments

### **2. Short Put Delta: 0.15 – 0.30**
- **Probability Target**: 70–85% chance of expiring worthless
- **Risk Balance**: 
  - Delta > 0.30 = Too directional (essentially long stock position)
  - Delta < 0.15 = Insufficient premium for risk taken
- **Sweet Spot**: 0.20-0.25 delta for optimal risk/reward

### **3. Premium Yield (Credit ÷ Spread Width ≥ 33%)**
- **Formula**: `Yield = Credit Received / (Strike Width) × 100`
- **Minimum**: 33% yield required
- **Example**: $5 wide spread must yield ≥ $1.65 credit
- **Filter**: Reject any spread yielding < 25% of width

### **4. Underlying Strength Requirements**
- **Trend Confirmation**:
  - Price > 50-day SMA > 200-day SMA
  - RSI > 50 (momentum confirmation)
  - Sector relative strength > 0
- **Technical Filters**:
  - No major resistance levels within 5% of current price
  - Volume above 20-day average
  - No bearish divergences on daily chart

### **5. Liquidity Requirements**
- **Bid/Ask Spread**: < $0.05 for each leg
- **Open Interest**: > 100 contracts per strike
- **Volume**: > 50 contracts traded (daily average)
- **Preferred Underlyings**: SPY, QQQ, IWM, NVDA, TSLA, AAPL, MSFT, GOOGL

### **6. Time to Expiration: 30-45 DTE Target**
- **Entry Window**: 35-42 DTE optimal
- **Theta Acceleration**: Maximize time decay in 21-35 DTE range
- **Exit Strategy**: Close at 50-70% max profit or 21 DTE, whichever comes first

### **7. Risk Management Framework**
- **Stop Loss**: 2x initial credit received
- **Example**: $2.00 credit collected → Stop at -$4.00 total loss
- **Profit Target**: 50-70% of maximum profit
- **Time-based Exit**: Close at 21 DTE regardless of P&L

### **8. Event Risk Filters**
- **Earnings Exclusion**: No positions within 7 days of earnings
- **Macro Events**: Avoid FOMC, CPI, NFP, GDP releases
- **Binary Events**: Exclude FDA approvals, court decisions, etc.
- **Ex-dividend**: Account for dividend dates affecting put pricing

---

## 🚀 **Screener Logic Flow**

### **Phase 1: Universe Filtering**
```
1. Scan all optionable stocks with market cap > $10B
2. Filter for average daily volume > 1M shares
3. Remove stocks with earnings in next 21 days
4. Calculate IVR for each underlying
5. Keep only IVR ≥ 50%
```

### **Phase 2: Option Chain Analysis**
```
For each qualified underlying:
1. Pull option chains for 0-20 DTE
2. Calculate delta for all put strikes
3. Identify puts with delta 0.15-0.30
4. Create spread combinations (5, 10, 15, 20 point widths)
5. Calculate theoretical credit for each spread
```

### **Phase 3: Spread Evaluation**
```
For each potential spread:
1. Credit/Width ratio ≥ 0.33
2. Bid/Ask spread < $0.05 per leg
3. Open Interest > 100 per strike
4. Volume > 50 per strike
5. Calculate Probability of Profit (PoP)
```

### **Phase 4: Technical Confirmation**
```
1. Verify underlying trend (SMA alignment)
2. Check RSI > 50
3. Confirm sector relative strength
4. Validate support/resistance levels
5. Check volume patterns
```

### **Phase 5: Ranking & Output**
```
Score = (PoP × 0.4) + (Yield × 0.3) + (IVR × 0.2) + (Liquidity × 0.1)
Sort by composite score (highest first)
Output top 20 opportunities with full metrics
```

---

## 📊 **Screener Output Format**

### **Required Data Points:**
- **Symbol**: Underlying ticker
- **Expiration**: Days to expiration
- **Short Strike**: Put strike being sold
- **Long Strike**: Put strike being bought
- **Credit**: Premium collected
- **Width**: Strike spread width
- **Yield**: Credit/Width percentage
- **Delta**: Short put delta
- **IVR**: Implied volatility rank
- **PoP**: Probability of profit
- **Liquidity Score**: Composite bid/ask + volume score
- **Technical Score**: Trend + momentum score
- **Risk/Reward**: Max profit vs max loss ratio

### **Sample Output:**
```
Rank | Symbol | Exp | Short/Long | Credit | Yield | Delta | IVR | PoP | Score
1    | SPY    | 35d | 420/415    | $1.85  | 37%   | 0.22  | 65% | 78% | 8.7
2    | QQQ    | 28d | 350/345    | $1.65  | 33%   | 0.25  | 58% | 75% | 8.4
3    | NVDA   | 42d | 900/890    | $3.20  | 32%   | 0.20  | 72% | 80% | 8.2
```

---

## 🔧 **Implementation Requirements**

### **Data Sources Needed:**
- **Real-time Options Data**: Strikes, premiums, Greeks, volume, OI
- **Historical Volatility**: 252-day rolling for IVR calculation  
- **Price Data**: OHLCV for technical analysis
- **Earnings Calendar**: To filter out earnings plays
- **Economic Calendar**: FOMC, CPI, NFP dates
- **Sector Performance**: For relative strength analysis

### **Technical Infrastructure:**
- **Options API**: Interactive Brokers, TD Ameritrade, or Polygon.io
- **Calculation Engine**: Python with pandas, numpy, scipy
- **Greeks Calculation**: Black-Scholes model implementation
- **Database**: Store historical IV data for IVR calculations
- **Scheduling**: Daily scans at market close
- **Alerts**: Real-time notifications for new opportunities

### **Performance Metrics to Track:**
- **Win Rate**: Percentage of profitable trades
- **Average Return**: Mean profit per trade
- **Sharpe Ratio**: Risk-adjusted returns
- **Maximum Drawdown**: Largest losing streak
- **Time in Trade**: Average holding period
- **Slippage**: Actual fills vs theoretical prices

---

## 📈 **Advanced Enhancements**

### **Machine Learning Integration:**
- **Feature Engineering**: IV rank, delta, yield, technical indicators
- **Model Training**: Historical performance prediction
- **Backtesting**: 5+ years of historical validation
- **Walk-Forward Analysis**: Out-of-sample testing

### **Portfolio-Level Optimization:**
- **Position Sizing**: Kelly Criterion or fixed fractional
- **Correlation Limits**: Maximum exposure per sector
- **Greeks Management**: Portfolio delta, gamma, theta limits
- **Diversification**: Spread across multiple underlyings

### **Real-Time Monitoring:**
- **P&L Tracking**: Mark-to-market throughout day
- **Greeks Updates**: Delta, gamma, theta changes
- **Early Warning**: Approaching stop losses
- **Market Regime**: Detect volatility regime changes

---

## 🎯 **Success Metrics & KPIs**

### **Target Performance:**
- **Win Rate**: 75-85%
- **Average Return**: 15-25% per trade
- **Sharpe Ratio**: > 2.0
- **Maximum Drawdown**: < 10%
- **Annual Return**: 30-50% (with proper position sizing)

### **Risk Controls:**
- **Single Position Risk**: < 2% of portfolio
- **Sector Concentration**: < 20% in any sector  
- **Total Greeks Exposure**: Delta < ±50, Gamma < 25
- **Volatility Exposure**: Vega < ±100 per $10k portfolio

---

This framework provides a systematic, quantitative approach to identifying high-probability Bull Put Spread opportunities while maintaining strict risk management and avoiding the emotional pitfalls of discretionary trading.

