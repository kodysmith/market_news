# Overbought/Oversold Stock Scanner Guide

## 🎯 **What It Does**

The Overbought/Oversold Scanner continuously monitors stocks for trading opportunities based on RSI (Relative Strength Index) conditions:

- **Overbought (RSI 70-80)**: Stocks that may be due for a pullback
- **Extreme Overbought (RSI > 80)**: Strong sell signals
- **Oversold (RSI 20-30)**: Stocks that may be due for a bounce
- **Extreme Oversold (RSI < 20)**: Strong buy signals

## 🚀 **Quick Start**

### **1. Run a Single Scan**
```bash
# Scan all default stocks
python3 overbought_oversold_scanner.py --scan

# Scan specific stocks
python3 overbought_oversold_scanner.py --scan --stocks AAPL MSFT GOOGL NVDA TSLA
```

### **2. View Current Opportunities**
```bash
# View dashboard
python3 view_opportunities.py

# Filter by category
python3 view_opportunities.py --category overbought
python3 view_opportunities.py --category oversold
```

### **3. Start Continuous Monitoring**
```bash
# Run every 4 hours (default)
python3 schedule_scanner.py

# Run every 2 hours
python3 schedule_scanner.py --interval 2

# Run once and exit
python3 schedule_scanner.py --once
```

## 📊 **Understanding the Output**

### **Dashboard Categories:**
- 🔥 **Extreme Overbought**: RSI > 80 (Strong SELL signals)
- 📈 **Overbought**: RSI 70-80 (SELL signals)
- 📉 **Oversold**: RSI 20-30 (BUY signals)
- ❄️ **Extreme Oversold**: RSI < 20 (Strong BUY signals)

### **Key Metrics:**
- **RSI**: Relative Strength Index (0-100)
- **Confidence**: Algorithm confidence in the signal (0-100%)
- **Target Price**: Expected price target
- **Stop Loss**: Risk management level
- **Volume Ratio**: Current volume vs average (confirms signals)

## 🔄 **Automated Monitoring**

### **Option 1: Continuous Scheduler**
```bash
# Start continuous monitoring (runs every 4 hours)
python3 schedule_scanner.py

# Custom interval (every 2 hours)
python3 schedule_scanner.py --interval 2
```

### **Option 2: Cron Job**
Add to your crontab for automated scanning:
```bash
# Run every 4 hours
0 */4 * * * cd /path/to/QuantEngine && python3 overbought_oversold_scanner.py --scan

# Run every 2 hours during market hours (9 AM - 4 PM EST)
0 9,11,13,15 * * 1-5 cd /path/to/QuantEngine && python3 overbought_oversold_scanner.py --scan
```

## 📈 **Stock Universe**

The scanner monitors 70+ stocks across sectors:

### **Technology**
AAPL, MSFT, GOOGL, AMZN, META, NVDA, TSLA, NFLX, ADBE, CRM

### **Financial**
JPM, BAC, WFC, GS, MS, C, AXP, V, MA, PYPL

### **Healthcare**
JNJ, PFE, UNH, ABBV, MRK, TMO, DHR, BMY, AMGN, GILD

### **Consumer**
KO, PEP, WMT, HD, MCD, NKE, SBUX, TGT, LOW, COST

### **Energy**
XOM, CVX, COP, EOG, SLB, KMI, WMB, PSX, VLO, MPC

### **ETFs**
SPY, QQQ, IWM, VTI, XLK, XLF, XLV, XLE, XLI, XLY

## 🎯 **Trading Strategy**

### **Overbought Stocks (SELL Signals)**
- **Entry**: When RSI crosses above 70
- **Target**: 3-5% below current price
- **Stop Loss**: 3-5% above current price
- **Confirmation**: High volume (>1.5x average)

### **Oversold Stocks (BUY Signals)**
- **Entry**: When RSI crosses below 30
- **Target**: 3-5% above current price
- **Stop Loss**: 3-5% below current price
- **Confirmation**: High volume (>1.5x average)

### **Risk Management**
- **Position Size**: 2-5% of portfolio per trade
- **Stop Losses**: Always use recommended levels
- **Diversification**: Don't put all capital in one signal
- **Confirmation**: Wait for volume confirmation

## 📊 **Data Storage**

- **Data File**: `overbought_oversold_data.json`
- **Reports**: `reports/overbought_oversold_scan_YYYYMMDD_HHMMSS.md`
- **Logs**: `overbought_oversold_scanner.log`

## 🔧 **Customization**

### **Modify Stock Universe**
Edit `overbought_oversold_scanner.py`:
```python
self.stock_universe = [
    'AAPL', 'MSFT', 'GOOGL',  # Add your stocks here
    # ... existing stocks
]
```

### **Adjust RSI Thresholds**
```python
self.overbought_threshold = 70    # Default: 70
self.oversold_threshold = 30      # Default: 30
self.extreme_overbought = 80      # Default: 80
self.extreme_oversold = 20        # Default: 20
```

### **Change Scan Frequency**
```bash
# Every 1 hour
python3 schedule_scanner.py --interval 1

# Every 6 hours
python3 schedule_scanner.py --interval 6
```

## 📱 **Integration with Market News App**

The scanner can be integrated with the Market News App to show real-time opportunities:

1. **Publish to Firebase**: Reports automatically saved
2. **Real-time Updates**: Dashboard updates every scan
3. **Mobile Access**: View opportunities on mobile app

## ⚠️ **Important Notes**

- **Market Hours**: Best results during market hours (9 AM - 4 PM EST)
- **Volume Confirmation**: Always check volume ratios
- **Risk Management**: Use stop losses and position sizing
- **Backtesting**: Test strategies before live trading
- **Disclaimer**: This is for informational purposes only, not financial advice

## 🚀 **Quick Commands Summary**

```bash
# Scan now
python3 overbought_oversold_scanner.py --scan

# View opportunities
python3 view_opportunities.py

# Start continuous monitoring
python3 schedule_scanner.py

# Scan specific stocks
python3 overbought_oversold_scanner.py --scan --stocks AAPL MSFT GOOGL
```

**Happy Trading! 🎯📈**

