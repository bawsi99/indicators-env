import sys
import logging
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta

# Try to import patch
try:
    import yf_patch
    yf_patch.patch_yfinance_globally()
except ImportError:
    print("yf_patch not found, using vanilla yfinance")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_fetch():
    symbol = "RELIANCE.NS"
    # Use a fixed date range from last week to avoid weekend issues
    end_dt = datetime(2026, 4, 10) # Friday
    start_dt = end_dt - timedelta(days=365)
    
    ticker = yf.Ticker(symbol)
    print(f"Fetching {symbol} from {start_dt.date()} to {end_dt.date()}...")
    df = ticker.history(start=start_dt.strftime('%Y-%m-%d'), 
                        end=(end_dt + timedelta(days=1)).strftime('%Y-%m-%d'),
                        auto_adjust=True)
    
    if df.empty:
        print("❌ Dataframe is EMPTY")
    else:
        print(f"✅ Success: fetched {len(df)} rows")
        print(df.tail())

if __name__ == "__main__":
    test_fetch()
