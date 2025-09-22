# Local Testing Guide

This guide shows you how to test the options scanner backend locally.

## Quick Start

### 1. Install Dependencies
```bash
cd options_scanner
pip install -r requirements.txt
```

### 2. Run Tests
```bash
python test_scanner.py
```

### 3. Run Mock Demo
```bash
python test_local.py
```

### 4. Run Interactive Demo
```bash
python demo_real.py
```

## Available Test Scripts

### `test_scanner.py`
Runs the comprehensive test suite to verify all components work correctly.

```bash
python test_scanner.py
```

**Expected Output:**
```
🧪 Running options scanner tests...
✅ All tests passed!
```

### `test_local.py`
Tests the scanner with mock data to verify the complete pipeline works.

```bash
python test_local.py
```

**Expected Output:**
```
🧪 Testing Options Scanner with Mock Data
📊 Created 3 mock spreads
✅ 1 spreads passed filters
📤 1 alerts would be sent
✅ Report saved to report.json
🎉 All tests completed successfully!
```

### `demo_real.py`
Interactive demo that can use real API data or mock data.

```bash
python demo_real.py
```

**Features:**
- Prompts for Alpha Vantage API key
- Falls back to mock data if no key provided
- Shows complete scan results
- Displays generated report.json

### `run_continuous.py`
Runs the scanner continuously for testing (every minute).

```bash
python run_continuous.py
```

**Features:**
- Runs every 60 seconds
- Shows scan results in real-time
- Press Ctrl+C to stop
- Good for testing the complete workflow

## Generated Files

### `report.json`
The scanner generates a report.json file that your Flutter app can consume:

```json
{
  "asOf": "2025-09-16T03:38:26.895625+00:00",
  "scanner": {
    "universe": ["SPY"],
    "dteWindow": [20, 45],
    "thresholds": { "minPOP": 0.5, "minEVPer100": 0.1 }
  },
  "topIdeas": [
    {
      "ticker": "SPY",
      "strategy": "BULL_PUT",
      "expiry": "2025-10-18",
      "shortK": 510.0,
      "longK": 505.0,
      "width": 5.0,
      "credit": 0.95,
      "maxLoss": 4.05,
      "dte": 33,
      "pop": 0.64,
      "ev": 50.0,
      "id": "test_spread_1"
    }
  ],
  "alertsSentThisRun": ["test_spread_1"]
}
```

### `options_scanner.db`
SQLite database file created for deduplication and historical data.

## Testing with Real API

To test with real Alpha Vantage data:

1. **Get API Key**: Sign up at [Alpha Vantage](https://www.alphavantage.co/support/#api-key)

2. **Run Demo**: 
   ```bash
   python demo_real.py
   ```
   Enter your API key when prompted.

3. **Expected Behavior**:
   - Fetches real SPY options data
   - Calculates real spreads and EV
   - May find 0-5 opportunities (depending on market conditions)
   - Generates report.json with real data

## Troubleshooting

### Common Issues

1. **"No spreads meet criteria"**
   - This is normal in low-volatility markets
   - Try adjusting thresholds in `spreads.py`
   - Check if market is open (9:30 AM - 4:00 PM ET)

2. **"API request failed"**
   - Check your Alpha Vantage API key
   - Verify you haven't exceeded rate limits (5 calls/minute for free tier)
   - Check internet connection

3. **"Firebase not initialized"**
   - This is expected with mock Firebase credentials
   - Real Firebase setup requires service account JSON

4. **"Market is closed"**
   - Scanner only runs during market hours
   - Use mock data for testing outside market hours

### Debug Mode

Run with verbose output:
```bash
PYTHONPATH=. python -u test_local.py 2>&1 | tee debug.log
```

## Next Steps

Once local testing is working:

1. **Configure Real API Keys**:
   - Set up Alpha Vantage API key
   - Configure Firebase service account

2. **Deploy to Production**:
   - Use `deploy.sh` script
   - Deploy to Cloud Run or VM
   - Set up cron job for 5-minute intervals

3. **Monitor Performance**:
   - Check logs for errors
   - Monitor report.json updates
   - Verify FCM notifications work

## File Structure

```
options_scanner/
├── main_worker.py          # Main scanner orchestrator
├── test_scanner.py         # Unit tests
├── test_local.py           # Mock data testing
├── demo_real.py            # Interactive demo
├── run_continuous.py       # Continuous testing
├── report.json             # Generated report (after running)
├── options_scanner.db      # SQLite database (after running)
└── LOCAL_TESTING.md        # This guide
```

## Support

If you encounter issues:

1. Check the troubleshooting section above
2. Run `python test_scanner.py` to verify all components work
3. Check the generated logs and error messages
4. Review the README.md for detailed documentation

