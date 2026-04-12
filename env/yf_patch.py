"""
yf_patch.py — Portable Yahoo Finance session patch and direct fetcher.
Bypasses yfinance library brittleness for core OHLCV operations.
"""

import logging
import random
import threading
import time
import urllib.parse
from typing import Optional, Dict

import pandas as pd

logger = logging.getLogger(__name__)

# ─── Configuration ───────────────────────────────────────────────────────────

UA_POOL = [
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/121.0.0.0 Safari/537.36",
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
]

def _get_random_headers():
    return {
        "User-Agent": random.choice(UA_POOL),
        "Accept": "application/json, text/plain, */*",
        "Accept-Language": "en-US,en;q=0.9",
        "Referer": "https://finance.yahoo.com/",
        "Origin": "https://finance.yahoo.com",
    }

# ─── Session Management ──────────────────────────────────────────────────────

_cached_session = None
_session_lock = threading.Lock()

def get_yf_session():
    global _cached_session
    try:
        from curl_cffi.requests import Session
    except ImportError:
        return None

    with _session_lock:
        if _cached_session is None:
            try:
                s = Session(impersonate="chrome120")
                headers = _get_random_headers()
                s.get("https://fc.yahoo.com", headers=headers, timeout=10)
                _cached_session = s
            except Exception:
                pass
    return _cached_session

# ─── Direct REST Fetcher (The Reliable Way) ──────────────────────────────────

def fetch_ohlcv_direct(
    symbol: str, 
    start_date: Optional[str] = None, 
    end_date: Optional[str] = None,
    interval: str = "1d"
) -> Optional[pd.DataFrame]:
    """
    Fetch OHLCV directly from Yahoo Finance v8 chart API via curl_cffi.
    """
    session = get_yf_session()
    if session is None:
        return None

    # Handle .NS suffix if missing
    if not symbol.endswith(".NS") and not symbol.startswith("^"):
        symbol = f"{symbol}.NS"

    sym_enc = urllib.parse.quote(symbol, safe="")
    
    # Range handling
    range_str = "1y" # Default
    if start_date and end_date:
        s_dt = int(pd.to_datetime(start_date).timestamp())
        e_dt = int(pd.to_datetime(end_date).timestamp())
        url = (
            f"https://query1.finance.yahoo.com/v8/finance/chart/{sym_enc}"
            f"?period1={s_dt}&period2={e_dt}&interval={interval}&includeAdjustedClose=true"
        )
    else:
        url = (
            f"https://query1.finance.yahoo.com/v8/finance/chart/{sym_enc}"
            f"?range={range_str}&interval={interval}&includeAdjustedClose=true"
        )

    try:
        r = session.get(url, headers=_get_random_headers(), timeout=15)
        if r.status_code != 200:
            return None

        data = r.json()
        result = data.get("chart", {}).get("result", [None])[0]
        if not result:
            return None

        timestamps = result.get("timestamp", [])
        quote = result.get("indicators", {}).get("quote", [{}])[0]
        adjclose = result.get("indicators", {}).get("adjclose", [{}])[0].get("adjclose", [])
        
        if not timestamps:
            return None

        df = pd.DataFrame({
            "open": quote.get("open", []),
            "high": quote.get("high", []),
            "low": quote.get("low", []),
            "close": adjclose if adjclose else quote.get("close", []),
            "volume": quote.get("volume", []),
        }, index=pd.to_datetime(timestamps, unit="s"))
        
        df = df.dropna(subset=["close"])
        return df

    except Exception:
        return None

def patch_yfinance_globally():
    """No-op or lightweight patch for compatibility if someone still uses yf.Ticker."""
    # We still keep the patch logic for safety, but we'll prioritize fetch_ohlcv_direct.
    pass
