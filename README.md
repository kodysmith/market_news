# Market News Application

A comprehensive market analysis and trading opportunity application that combines a Python backend for data analysis with a Flutter frontend for cross-platform mobile access.

## Overview

This application provides real-time market sentiment analysis, options trading strategies, and market insights through two main components:

1. **Python Backend**: Handles data fetching, market analysis, and report generation
2. **Flutter Frontend**: Cross-platform mobile app for displaying market insights

## Architecture

### Backend Components (Python)

#### Core Files

- **`api.py`**: Flask web server that serves market data via REST API
- **`generate_report.py`**: Main report generation engine that coordinates all data sources
- **`MarketDashboard.py`**: Market sentiment analysis using key indicators
- **`Scanner1.py`**: Options trading strategy scanner and analyzer
- **`TickerProvider.py`**: Fetches most active stocks from Yahoo Finance
- **`get_vix_data.py`**: VIX volatility data collection
- **`config.json`**: Configuration settings for trading parameters

#### Key Features

**Market Sentiment Analysis**
- Tracks S&P 500 Futures, Nasdaq 100 Futures, VIX, 10-Year Treasury, USD Index
- Calculates overall market sentiment (Bullish/Bearish/Neutral)
- Provides directional indicators for each market component

**Options Trading Scanner**
- Scans for multiple option strategies:
  - Bull Put Spreads (Bullish)
  - Bear Call Spreads (Bearish)
  - Long Straddles (High Volatility)
  - Long Strangles (High Volatility)
- Uses Black-Scholes model for probability calculations
- Filters trades based on configurable risk parameters

**Data Sources**
- Yahoo Finance API for market data
- Real-time options chain analysis
- VIX historical data for volatility trends
- Treasury yield data for risk-free rates

### Frontend Components (Flutter)

#### Project Structure
```
market_news_app/
├── lib/
│   ├── models/          # Data models
│   │   ├── report_data.dart
│   │   ├── economic_event.dart
│   │   └── vix_data.dart
│   ├── screens/         # UI screens
│   │   └── market_insights_screen.dart
│   ├── services/        # API services
│   │   └── fmp_api_service.dart
│   ├── widgets/         # Custom widgets
│   │   └── daily_strategy_guide.dart
│   └── main.dart        # App entry point
├── assets/
│   └── results.json     # Generated market data
└── pubspec.yaml         # Flutter dependencies
```

#### Key Features

**Market Dashboard**
- Real-time sentiment indicators
- Market direction analysis
- Trade idea recommendations
- VIX volatility tracking

**Market Insights Screen**
- Detailed volatility analysis
- Daily strategy guide
- Economic calendar integration
- Risk assessment tools

## Configuration

### Backend Configuration (`config.json`)
```json
{
    "CREDIT_SPREAD_OTM_PERCENT": 2.0,         # Out-of-money percentage for credit spreads
    "STRANGLE_OTM_PERCENT": 5.0,             # Out-of-money percentage for strangles
    "NUM_MOST_ACTIVE_TICKERS": 10,           # Number of active stocks to scan
    "FIXED_TICKERS": [],                      # Fixed tickers to always include
    "MIN_PROB_SUCCESS": 70.0,                # Minimum probability of success (%)
    "MIN_CREDIT_RECEIVED": 0.10,             # Minimum credit for trades ($)
    "SORT_TRADES_BY": "Prob. of Success",    # Trade sorting criteria
    "GAMMA_SCALPING_TICKERS": [              # Tickers for gamma scalping analysis
        "SPY", "QQQ", "IWM", "AAPL", "TSLA", 
        "NVDA", "MSFT", "GOOGL", "AMZN", "META"
    ],
    "GAMMA_SCALPING_THRESHOLD": 60.0,        # Score threshold for gamma scalping (0-100)
    "PREMIUM_SELLING_THRESHOLD": 30.0,       # Score threshold for premium selling (0-100)
    "IV_RV_RATIO_LOW": 0.8,                  # Low IV/RV ratio threshold
    "IV_RV_RATIO_HIGH": 1.2,                 # High IV/RV ratio threshold
    "VIX_LOW_PERCENTILE": 20.0,              # Low VIX percentile threshold
    "VIX_HIGH_PERCENTILE": 80.0,             # High VIX percentile threshold
    "VOLATILITY_ACCELERATION_HIGH": 1.3,     # High volatility acceleration threshold
    "VOLATILITY_ACCELERATION_LOW": 0.7       # Low volatility acceleration threshold
}
```

### Flutter Configuration
- Environment variables in `.env` file
- Asset files in `assets/` directory
- API keys and endpoints configured via environment

## Setup and Installation

### Prerequisites
- Python 3.8+
- Flutter SDK 3.8+
- Virtual environment (recommended)

### Backend Setup

1. **Create and activate virtual environment:**
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. **Install Python dependencies:**
```bash
pip install -r requirements.txt
```

3. **Run the report generator:**
```bash
python generate_report.py
```

4. **Start the API server:**
```bash
python api.py
```

### Flutter App Setup

1. **Navigate to Flutter app directory:**
```bash
cd market_news_app
```

2. **Install Flutter dependencies:**
```bash
flutter pub get
```

3. **Create environment file:**
```bash
cp .env.example .env
# Edit .env with your API keys
```

4. **Run the Flutter app:**
```bash
flutter run
```

## Usage Workflow

### Daily Market Analysis

1. **Generate Market Report:**
   ```bash
   python generate_report.py
   ```
   - Fetches market sentiment data
   - Scans for trading opportunities
   - Generates `report.json` and `report.html`

2. **Start API Server:**
   ```bash
   python api.py
   ```
   - Serves market data at `http://localhost:5000`
   - Provides `/report.json` endpoint

3. **Launch Flutter App:**
   ```bash
   cd market_news_app && flutter run
   ```
   - Displays market dashboard
   - Shows trade ideas and insights

### Data Flow

1. **Backend Data Collection:**
   - `TickerProvider.py` → Fetches active stocks
   - `MarketDashboard.py` → Analyzes market sentiment
   - `Scanner1.py` → Scans for trading opportunities
   - `get_vix_data.py` → Collects volatility data

2. **Report Generation:**
   - `generate_report.py` → Coordinates all data sources
   - Outputs `report.json` for Flutter app
   - Outputs `report.html` for web viewing

3. **Frontend Display:**
   - Flutter app reads `assets/results.json`
   - Displays market sentiment and trade ideas
   - Provides detailed market insights

## Key Features

### Market Sentiment Analysis
- **Indicators Tracked:**
  - S&P 500 Futures (ES=F)
  - Nasdaq 100 Futures (NQ=F)
  - VIX Fear Index (^VIX)
  - 10-Year Treasury Yield (^TNX)
  - US Dollar Index (DX-Y.NYB)

### Gamma Scalping vs Premium Selling Analysis
- **Volatility Regime Analysis**: Compares implied volatility to realized volatility
- **VIX Percentile Analysis**: Determines current volatility environment
- **Market Timing**: Identifies optimal days for gamma scalping vs premium selling
- **Individual Ticker Analysis**: Provides specific recommendations for each stock
- **Scoring System**: 0-100 scale indicating strategy preference

**Decision Factors:**
- **IV/RV Ratio**: When IV < RV, gamma scalping is favored (options are cheap)
- **VIX Level**: Low VIX suggests potential volatility expansion
- **Volatility Acceleration**: Recent volatility trends influence strategy selection
- **Term Structure**: Relationship between front-month and back-month volatility

### Trading Strategies
- **Bull Put Spreads**: Bullish credit spreads
- **Bear Call Spreads**: Bearish credit spreads
- **Long Straddles**: Non-directional volatility plays
- **Long Strangles**: Wide volatility strategies
- **Gamma Scalping**: Buy options and delta hedge with shares
- **Premium Selling**: Sell options to collect time decay

### Risk Management
- Configurable success probability thresholds
- Minimum credit requirements
- Risk/reward ratio calculations
- Black-Scholes probability modeling
- Volatility-based position sizing

## API Endpoints

### Backend API (`api.py`)
- `GET /`: API status check
- `GET /report.json`: Latest market report data (includes gamma analysis)

### External APIs Used
- Yahoo Finance: Market data and options chains
- Financial Modeling Prep: VIX and economic data

## Development Notes

### Backend Technologies
- **Flask**: Web framework for API server
- **yfinance**: Yahoo Finance data integration
- **pandas/numpy**: Data analysis and manipulation
- **scipy**: Statistical calculations
- **BeautifulSoup**: Web scraping for active stocks
- **GammaScalpingAnalyzer**: Custom volatility regime analysis

### Flutter Technologies
- **Material Design**: UI framework
- **HTTP**: API communication
- **flutter_dotenv**: Environment variable management
- **JSON serialization**: Data modeling
- **Custom Widgets**: Gamma scalping analysis displays

## Usage Examples

### Gamma Scalping Analysis
```bash
# Run standalone gamma analysis
python GammaScalpingAnalyzer.py

# Generate full report with gamma analysis
python generate_report.py
```

**Sample Output:**
```
--- GAMMA SCALPING ANALYSIS ---
Market Recommendation: GAMMA_SCALPING_FAVORED
Average Gamma Score: 67.3/100
Gamma Scalping Opportunities: 6
Premium Selling Opportunities: 2

--- INDIVIDUAL TICKER ANALYSIS ---
SPY: GAMMA_SCALPING (Score: 72)
  Strategy: Buy straddles/strangles and delta hedge
  • IV/RV ratio low (0.73) - options cheap
  • VIX low (16.2, 15th percentile) - vol expansion likely
```

## Troubleshooting

### Common Issues

1. **No market data**: Check internet connection and API limits
2. **Options chain errors**: Verify ticker symbols and market hours
3. **Flutter build issues**: Run `flutter clean && flutter pub get`
4. **API connection errors**: Ensure backend server is running
5. **Gamma analysis errors**: Check options data availability and market hours

### Data Refresh
- Market data updates when `generate_report.py` runs
- Flutter app requires restart to load new data
- API server serves latest available report
- Gamma analysis updates with each report generation

## Future Enhancements

- Real-time data streaming
- Push notifications for trade alerts
- Historical performance tracking
- Advanced charting and visualization
- Paper trading simulation
- Multi-timeframe analysis
- **Gamma scalping backtesting**
- **Real-time delta hedging alerts**
- **Volatility forecasting models**

## Disclaimer

This application is for educational and informational purposes only. It does not constitute financial advice. Options trading and gamma scalping involve significant risk and may not be suitable for all investors. Always conduct your own research and consult with qualified financial professionals before making trading decisions.

## License

This project is for personal use and educational purposes.