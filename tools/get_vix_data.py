import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta

def get_vix_data():
    today = datetime.now()
    # Fetch data for the last 7 days to ensure we get some data
    start_date = today - timedelta(days=7)
    
    try:
        vix_data = yf.download("^VIX", start=start_date, end=today)
        if not vix_data.empty:
            print("VIX Data from yfinance (first 5 rows):")
            print(vix_data.head().to_string())
        else:
            print("No VIX data found for the specified period.")
    except Exception as e:
        print(f"Error fetching VIX data: {e}")

if __name__ == "__main__":
    get_vix_data()
