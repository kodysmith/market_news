# Options Screener Documentation

## Overview

The Options Screener is designed to identify **long-term profitable trades** with **positive expected value (EV)** by analyzing all current options data for SPY and filtering down to only the most promising opportunities.

## Core Philosophy

> **"Only show trades that are mathematically profitable in the long run"**

The screener follows a strict **positive expected value requirement** - if a trade has negative expected value, it will **not appear** in the results, regardless of other attractive metrics.

## Expected Value (EV) Calculation

### Formula
```
EV = (Probability of Profit × Max Profit) - (Probability of Loss × Max Loss)
```

### Example
- **Credit Received**: $0.95
- **Max Loss**: $4.05 (if both puts expire in-the-money)
- **Probability of Profit**: 64%
- **Probability of Loss**: 36%

**EV = (0.64 × $0.95) - (0.36 × $4.05) = $0.61 - $1.46 = -$0.85**

❌ **This trade would be FILTERED OUT** due to negative EV.

### Positive EV Example
- **Credit Received**: $1.50
- **Max Loss**: $3.50
- **Probability of Profit**: 70%
- **Probability of Loss**: 30%

**EV = (0.70 × $1.50) - (0.30 × $3.50) = $1.05 - $1.05 = $0.00**

✅ **This trade would be INCLUDED** (breakeven EV is acceptable).

## Screening Criteria

### 1. **Primary Filter: Positive Expected Value**
- **Requirement**: EV ≥ $0.00
- **Rationale**: Only trades that are profitable in the long run
- **Implementation**: Hard filter - negative EV trades are completely excluded

### 2. **Secondary Filters**

#### **Days to Expiration (DTE)**
- **Range**: 20-45 days
- **Rationale**: Optimal time decay vs. time to manage position
- **Too Short**: <20 days - rapid time decay, hard to manage
- **Too Long**: >45 days - slow time decay, capital tied up

#### **Implied Volatility Rank (IVR)**
- **Minimum**: 50%
- **Rationale**: Need elevated volatility to justify premium collection
- **Calculation**: Current IV vs. 52-week high/low range

#### **Probability of Profit (POP)**
- **Minimum**: 50%
- **Target Range**: 60-80%
- **Rationale**: Balance between safety and premium collection

#### **Premium Yield**
- **Minimum**: 33% of spread width
- **Example**: $5 wide spread needs ≥$1.65 credit
- **Rationale**: Ensure adequate compensation for risk

#### **Liquidity Requirements**
- **Open Interest**: ≥100 contracts per leg
- **Volume**: ≥1 contract traded today
- **Bid-Ask Spread**: ≤$0.10 or ≤2% of underlying price
- **Rationale**: Ensure executable fills at reasonable prices

### 3. **Quality Scoring**

#### **Expected Value Score (40% weight)**
```
EV Score = (EV - Min Possible EV) / (Max Possible EV - Min Possible EV) × 100
```
- **Range**: 0-100
- **Higher is better**

#### **Probability of Profit (30% weight)**
- **Range**: 0-100
- **Higher is better**

#### **Premium Yield (20% weight)**
- **Range**: 0-100
- **Higher is better**

#### **IV Rank (10% weight)**
- **Range**: 0-100
- **Higher is better**

#### **Final Composite Score**
```
Composite Score = (EV Score × 0.4) + (POP × 0.3) + (Yield × 0.2) + (IVR × 0.1)
```

## Data Sources

### **Primary: Alpha Vantage API**
- **Real-time Options Data**: Current bid/ask prices, volume, open interest
- **Greeks**: Delta, gamma, theta, vega, rho
- **Implied Volatility**: Current IV for each strike
- **Rate Limit**: 5 calls/minute (free tier)

### **Fallback: Mock Data**
- **When**: API unavailable or rate limited
- **Quality**: Realistic market conditions
- **Purpose**: Ensure screener always functions

## Screening Process

### **Step 1: Data Collection**
1. Fetch current SPY price
2. Retrieve options chain for all expirations
3. Filter puts by DTE range (20-45 days)
4. Calculate implied volatility rank

### **Step 2: Spread Generation**
1. Identify all valid bull put spread combinations
2. Calculate credit received (short put bid - long put ask)
3. Determine spread width (short strike - long strike)
4. Calculate maximum loss (width - credit)

### **Step 3: Expected Value Analysis**
1. Calculate probability of profit using Black-Scholes
2. Compute expected value for each spread
3. **FILTER OUT** all spreads with EV < $0.00
4. Calculate EV score for remaining spreads

### **Step 4: Quality Assessment**
1. Apply secondary filters (IVR, POP, yield, liquidity)
2. Calculate composite score for each spread
3. Sort by composite score (highest first)
4. Return top 10 opportunities

### **Step 5: Risk Management**
1. Flag spreads near earnings dates
2. Check for major economic events
3. Validate liquidity requirements
4. Ensure executable fills

## Output Format

### **Report Structure**
```json
{
  "asOf": "2025-09-16T18:41:17.702268+00:00",
  "scanner": {
    "universe": ["SPY"],
    "dteWindow": [20, 45],
    "thresholds": {
      "minPOP": 0.5,
      "minEVPer100": 0.0,
      "minExpectedValue": 0.0
    }
  },
  "topIdeas": [
    {
      "ticker": "SPY",
      "strategy": "BULL_PUT",
      "expiry": "2025-10-18",
      "shortK": 510.0,
      "longK": 505.0,
      "width": 5.0,
      "credit": 1.50,
      "maxLoss": 3.50,
      "dte": 33,
      "pop": 0.70,
      "ev": 0.25,
      "ivShort": 0.22,
      "bidAskW": 0.06,
      "oiShort": 1423,
      "oiLong": 1312,
      "volShort": 289,
      "volLong": 214,
      "fillScore": 0.82,
      "id": "spread_hash"
    }
  ],
  "alertsSentThisRun": ["spread_hash"]
}
```

### **Key Metrics Explained**

| Metric | Description | Good Value |
|--------|-------------|------------|
| `ev` | Expected Value | ≥ $0.00 |
| `pop` | Probability of Profit | 60-80% |
| `credit` | Premium Received | High relative to width |
| `maxLoss` | Maximum Loss | Width - Credit |
| `dte` | Days to Expiration | 20-45 |
| `ivShort` | Implied Volatility | 20-40% |
| `fillScore` | Liquidity Score | 0.7+ |

## Market Conditions

### **High Volatility Environment (VIX > 25)**
- **More Opportunities**: Higher premiums = more positive EV trades
- **Better Yields**: 40-60% of spread width common
- **Higher Risk**: More volatile underlying

### **Low Volatility Environment (VIX < 20)**
- **Fewer Opportunities**: Lower premiums = fewer positive EV trades
- **Lower Yields**: 20-30% of spread width typical
- **Lower Risk**: More stable underlying

### **Normal Volatility Environment (VIX 20-25)**
- **Moderate Opportunities**: Balanced risk/reward
- **Typical Yields**: 30-40% of spread width
- **Manageable Risk**: Standard market conditions

## Quality Assurance

### **Automated Checks**
1. **EV Validation**: Every trade must have EV ≥ $0.00
2. **Data Freshness**: Options data < 5 minutes old
3. **Liquidity Verification**: Sufficient volume and open interest
4. **Price Validation**: Bid/ask spreads within acceptable range

### **Manual Review**
1. **Market Context**: Consider overall market conditions
2. **Earnings Calendar**: Avoid trades near earnings
3. **Economic Events**: Check for major announcements
4. **Technical Analysis**: Verify underlying trend

## Performance Metrics

### **Success Criteria**
- **Win Rate**: 60-80% of trades profitable
- **Average EV**: $0.10+ per $100 collateral
- **Max Drawdown**: < 10% of account
- **Sharpe Ratio**: > 1.0

### **Monitoring**
- **Daily**: Review all trades and outcomes
- **Weekly**: Analyze win rate and EV trends
- **Monthly**: Adjust criteria based on performance
- **Quarterly**: Review and update screening parameters

## Troubleshooting

### **No Opportunities Found**
1. **Check Market Conditions**: Low volatility = fewer opportunities
2. **Adjust Criteria**: Lower IVR or yield requirements
3. **Expand Universe**: Consider other tickers (QQQ, IWM)
4. **Wait for Volatility**: Better opportunities during market stress

### **Poor Fill Quality**
1. **Increase Liquidity Requirements**: Higher OI/volume thresholds
2. **Tighten Bid-Ask Limits**: Reduce acceptable spread width
3. **Focus on Major Strikes**: Round numbers (500, 505, 510)
4. **Avoid Earnings**: Skip trades near earnings dates

### **Negative Performance**
1. **Review EV Calculations**: Ensure probability models are accurate
2. **Check Market Regime**: Adapt to changing volatility
3. **Adjust Position Sizing**: Reduce size during difficult periods
4. **Improve Exit Strategy**: Better risk management rules

## Future Enhancements

### **Planned Features**
1. **Earnings Calendar Integration**: Automatic filtering
2. **Macro Event Detection**: Fed meetings, CPI releases
3. **Dynamic Criteria**: Adjust based on market conditions
4. **Machine Learning**: Improve probability calculations
5. **Multi-Strategy Support**: Iron condors, strangles, etc.

### **Data Improvements**
1. **Real-Time Greeks**: More accurate probability calculations
2. **Historical Analysis**: Backtesting capabilities
3. **Sector Analysis**: Relative strength considerations
4. **Correlation Analysis**: Portfolio risk management

## Conclusion

The Options Screener is designed to be **conservative and profitable** by focusing exclusively on trades with positive expected value. This approach ensures that over time, the strategy will be profitable even if individual trades don't always work out.

The key to success is **patience** - waiting for the right opportunities rather than forcing trades in poor market conditions. The screener will tell you when conditions are favorable, and when they're not, it's better to wait.

**Remember**: It's better to miss a trade than to take a bad one. The screener's job is to find the good ones and filter out the rest.

