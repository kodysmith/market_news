# Options Scanner Backend

A comprehensive options scanner that polls option chains, computes positive-EV spreads, and sends push notifications to your Flutter app.

## Features

- 🔍 **Real-time Options Scanning**: Polls Alpha Vantage for live options data
- 📊 **Positive-EV Analysis**: Calculates probability of profit and expected value
- 🔔 **Smart Alerts**: FCM push notifications with deduplication
- 📈 **Bull Put Spreads**: Focuses on high-probability credit spreads
- 💾 **Data Persistence**: SQLite database for history and deduplication
- 📱 **Flutter Integration**: Generates report.json for your existing app

## Architecture

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Alpha Vantage │───▶│  Options Scanner │───▶│  Flutter App    │
│   (Options API) │    │  (Python Backend)│    │  (FCM + JSON)   │
└─────────────────┘    └──────────────────┘    └─────────────────┘
                              │
                              ▼
                       ┌──────────────────┐
                       │   SQLite DB      │
                       │  (Deduplication) │
                       └──────────────────┘
```

## Quick Start

### 1. Install Dependencies

```bash
cd options_scanner
pip install -r requirements.txt
```

### 2. Configure Environment

Copy `env_example.txt` to `.env` and fill in your credentials:

```bash
cp env_example.txt .env
```

Required variables:
- `ALPHAVANTAGE_API_KEY`: Your Alpha Vantage API key
- `FIREBASE_SERVICE_ACCOUNT_JSON_BASE64`: Base64 encoded Firebase service account JSON

### 3. Run Tests

```bash
python test_scanner.py
```

### 4. Run Scanner

```bash
# Single run
python main_worker.py

# Continuous mode (for testing)
python main_worker.py --continuous
```

## Configuration

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `ALPHAVANTAGE_API_KEY` | Alpha Vantage API key | Required |
| `FIREBASE_SERVICE_ACCOUNT_JSON_BASE64` | Firebase credentials (base64) | Required |
| `PUBLIC_BUCKET_URL` | Report publishing URL | `https://api-hvi4gdtdka-uc.a.run.app` |
| `REPORT_OBJECT_PATH` | Report JSON path | `/report.json` |
| `TICKERS` | Comma-separated tickers | `SPY,QQQ,IWM` |
| `SCAN_INTERVAL_MINUTES` | Scan frequency | `5` |
| `MARKET_TZ` | Market timezone | `America/New_York` |

### Screening Criteria

The scanner uses these filters by default:

- **Strategy**: Bull put credit spreads only
- **DTE**: 20-45 days to expiration
- **Liquidity**: Both legs OI ≥ 100, volume ≥ 1
- **Bid-Ask**: Width ≤ $0.10 or ≤ 2% of underlying
- **EV Threshold**: ≥ $0.10 per $100 collateral
- **POP Threshold**: ≥ 50%

## API Reference

### Main Components

#### `OptionsScanner`
Main orchestrator class that coordinates all components.

```python
scanner = OptionsScanner()
result = scanner.run_once()
```

#### `AlphaVantageClient`
Fetches options data from Alpha Vantage API.

```python
client = AlphaVantageClient(api_key)
spot_price = client.get_spot("SPY")
options = client.get_otm_puts("SPY", spot_price)
```

#### `SpreadAnalyzer`
Analyzes and scores bull put spreads.

```python
analyzer = SpreadAnalyzer()
candidates = analyzer.build_bull_put_candidates(options, spot_price)
filtered = analyzer.apply_filters(candidates)
```

#### `AlertManager`
Manages alert sending and deduplication.

```python
alert_manager = AlertManager(db_manager)
alerts = alert_manager.decide_alerts(spreads)
```

### Data Models

#### `BullPutSpread`
Represents a bull put spread opportunity.

```python
@dataclass
class BullPutSpread:
    ticker: str
    expiry: str
    short_strike: float
    long_strike: float
    width: float
    credit: float
    max_loss: float
    pop: float  # Probability of profit
    ev: float   # Expected value
    dte: int    # Days to expiration
    # ... more fields
```

## Report Format

The scanner generates a `report.json` file with this structure:

```json
{
  "asOf": "2025-09-15T14:35:00Z",
  "scanner": {
    "universe": ["SPY", "QQQ", "IWM"],
    "dteWindow": [20, 45],
    "thresholds": { "minPOP": 0.50, "minEVPer100": 0.10 }
  },
  "topIdeas": [
    {
      "ticker": "SPY",
      "strategy": "BULL_PUT",
      "expiry": "2025-10-18",
      "shortK": 510,
      "longK": 505,
      "width": 5.0,
      "credit": 0.95,
      "maxLoss": 4.05,
      "dte": 33,
      "pop": 0.64,
      "ev": 0.16,
      "ivShort": 0.22,
      "bidAskW": 0.06,
      "oiShort": 1423,
      "oiLong": 1312,
      "volShort": 289,
      "volLong": 214,
      "fillScore": 0.82,
      "id": "hash..."
    }
  ],
  "alertsSentThisRun": ["hash...", "hash..."]
}
```

## FCM Notifications

Push notifications are sent with this payload:

```json
{
  "title": "New positive-EV bull put on SPY",
  "body": "POP 64%, EV $0.16, credit $0.95, 33 DTE (510/505)",
  "data": {
    "type": "opportunity",
    "id": "hash...",
    "ticker": "SPY",
    "strategy": "BULL_PUT",
    "expiry": "2025-10-18",
    "shortK": "510",
    "longK": "505"
  }
}
```

## Deployment

### Cloud Run (Recommended)

1. **Create Dockerfile**:
```dockerfile
FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
CMD ["python", "main_worker.py"]
```

2. **Deploy to Cloud Run**:
```bash
gcloud run deploy options-scanner \
  --source . \
  --platform managed \
  --region us-central1 \
  --set-env-vars ALPHAVANTAGE_API_KEY=your_key
```

3. **Set up Cloud Scheduler**:
```bash
gcloud scheduler jobs create http options-scanner-job \
  --schedule="*/5 9-16 * * 1-5" \
  --uri="https://your-cloud-run-url/run" \
  --http-method=POST
```

### VM with Cron

1. **Install on VM**:
```bash
git clone <your-repo>
cd options_scanner
pip install -r requirements.txt
```

2. **Set up cron job**:
```bash
# Edit crontab
crontab -e

# Add this line (runs every 5 minutes during market hours)
*/5 9-16 * * 1-5 cd /path/to/options_scanner && python main_worker.py
```

## Monitoring

### Logs
The scanner logs all activities with timestamps:
- Market hours detection
- API calls and responses
- Spread analysis results
- Alert sending status
- Report generation

### Database
Check the SQLite database for:
- `pushed_alerts`: Alert deduplication history
- `spread_snapshots`: Historical spread data

### Health Checks
The scanner returns status codes:
- `0`: Success
- `1`: Error (check logs)

## Troubleshooting

### Common Issues

1. **"Market is closed"**: Scanner only runs during market hours (9:30 AM - 4:00 PM ET, weekdays)

2. **"No options data found"**: Check Alpha Vantage API key and rate limits

3. **"Firebase not initialized"**: Verify Firebase service account JSON is base64 encoded correctly

4. **"No spreads meet criteria"**: Adjust thresholds in `apply_filters()` method

### Debug Mode

Run with verbose logging:
```bash
PYTHONPATH=. python -u main_worker.py 2>&1 | tee scanner.log
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Run the test suite: `python test_scanner.py`
5. Submit a pull request

## License

MIT License - see LICENSE file for details.

## Support

For issues and questions:
1. Check the troubleshooting section
2. Review the logs
3. Run the test suite
4. Open an issue on GitHub

