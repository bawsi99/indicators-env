"""
data_loader.py — Fetches OHLCV data via yfinance and computes the full
technical indicator suite used by the StockAnalyzer-Pro indicators agent.
Also provides ground truth labels via 20-day forward return thresholding.

Indicators computed (mirrors calculate_all_indicators_optimized):
  Moving Averages : SMA(20/50/200), EMA(20/50), golden/death cross
  Momentum        : RSI(14), MACD(12/26/9), Stochastic(14/3)
  Volatility      : Bollinger Bands(20,2), ATR(14), volatility regime
  Trend           : ADX(14), +DI/-DI, trend strength
  Volume          : OBV, VWAP, MFI(14), CMF(20), A/D Line, volume ratio
  Levels          : Pivot Points (Standard: R2/R1/P/S1/S2)
  Context         : market regime, [TERM: X] token
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import yfinance as yf

logger = logging.getLogger(__name__)

# ─── Constants ───────────────────────────────────────────────────────────────

TERM_WINDOWS: Dict[str, int] = {
    "intraday": 1,
    "short": 5,
    "medium": 20,
    "long": 60,
}

TERM_THRESHOLDS: Dict[str, float] = {
    "intraday": 0.005,   # ±0.5%
    "short":    0.015,   # ±1.5%
    "medium":   0.025,   # ±2.5%
    "long":     0.050,   # ±5.0%
}

# 100 liquid NSE stocks (diversified across sectors)
NSE_UNIVERSE: List[str] = [
    "RELIANCE", "TCS", "HDFCBANK", "INFY", "ICICIBANK", "HINDUNILVR", "SBIN",
    "BHARTIARTL", "ITC", "KOTAKBANK", "LT", "AXISBANK", "ASIANPAINT", "MARUTI",
    "BAJFINANCE", "TITAN", "SUNPHARMA", "WIPRO", "ULTRACEMCO", "NESTLEIND",
    "POWERGRID", "NTPC", "TECHM", "HCLTECH", "DIVISLAB", "CIPLA", "EICHERMOT",
    "HDFCLIFE", "DRREDDY", "ONGC", "COALINDIA", "TATASTEEL",
    "JSWSTEEL", "ADANIPORTS", "BAJAJ-AUTO", "HEROMOTOCO", "INDUSINDBK",
    "GRASIM", "BRITANNIA", "SBILIFE", "APOLLOHOSP", "TATACONSUM", "PIDILITIND",
    "TORNTPHARM", "HAVELLS", "GODREJCP", "MUTHOOTFIN", "PAGEIND", "COLPAL",
    "BERGEPAINT", "DABUR", "MARICO", "EMAMILTD", "BALKRISIND", "CUMMINSIND",
    "VOLTAS", "WHIRLPOOL", "TVSMOTOR", "BOSCHLTD", "SCHAEFFLER", "ASTRAL",
    "POLYCAB", "KANSAINER", "AARTIIND", "DEEPAKNTR", "PIIND", "LALPATHLAB",
    "METROPOLIS", "AUROPHARMA", "BIOCON", "GLENMARK", "LUPIN", "ALKEM",
    "IPCALAB", "LAURUSLABS", "GRANULES", "ABBOTINDIA", "PFIZER", "SANOFI",
    "KAJARIACER", "CENTURYTEX", "RAMCOCEM", "JKCEMENT", "SHREECEM", "AMBUJACEMENT",
    "INDIGO", "SPICEJET", "IRCTC", "CONCOR", "GMRINFRA", "HUDCO",
    "BANDHANBNK", "IDFCFIRSTB", "FEDERALBNK", "RBLBANK", "CANBK", "PNB",
    "BANKBARODA", "UNIONBANK",
]


# ─── Core fetch + indicator computation ──────────────────────────────────────

def fetch_ohlcv(symbol: str, end_date: str, lookback_days: int = 300) -> Optional[pd.DataFrame]:
    """
    Fetch OHLCV data for `symbol.NS` up to `end_date` via yfinance.
    Returns a clean DataFrame with columns: open, high, low, close, volume.
    """
    try:
        end_dt = pd.to_datetime(end_date) + timedelta(days=1)
        start_dt = end_dt - timedelta(days=lookback_days)
        ticker = yf.Ticker(f"{symbol}.NS")
        df = ticker.history(
            start=start_dt.strftime("%Y-%m-%d"),
            end=end_dt.strftime("%Y-%m-%d"),
            auto_adjust=True,
        )
        if df.empty or len(df) < 30:
            return None
        df.columns = df.columns.str.lower()
        df = df[["open", "high", "low", "close", "volume"]].dropna()
        return df
    except Exception as e:
        logger.warning(f"[DataLoader] fetch_ohlcv failed for {symbol} on {end_date}: {e}")
        return None


def compute_indicators(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Compute the full indicator suite mirroring calculate_all_indicators_optimized.
    All values are current (scalar), no historical arrays returned.
    """
    ind: Dict[str, Any] = {}
    close = df["close"]
    high  = df["high"]
    low   = df["low"]
    vol   = df["volume"]
    cp    = float(close.iloc[-1])

    # ── Moving Averages ──────────────────────────────────────────────────────
    sma20  = close.rolling(20).mean()
    sma50  = close.rolling(50).mean()
    sma200 = close.rolling(200).mean() if len(df) >= 200 else sma50
    ema20  = close.ewm(span=20, adjust=False).mean()
    ema50  = close.ewm(span=50, adjust=False).mean()

    golden_cross = bool(sma20.iloc[-1] > sma50.iloc[-1] and sma20.iloc[-2] <= sma50.iloc[-2])
    death_cross  = bool(sma20.iloc[-1] < sma50.iloc[-1] and sma20.iloc[-2] >= sma50.iloc[-2])

    ind["moving_averages"] = {
        "sma_20": _safe(sma20.iloc[-1], cp),
        "sma_50": _safe(sma50.iloc[-1], cp),
        "sma_200": _safe(sma200.iloc[-1], cp),
        "ema_20": _safe(ema20.iloc[-1], cp),
        "ema_50": _safe(ema50.iloc[-1], cp),
        "price_to_sma200_pct": round((cp / _safe(sma200.iloc[-1], cp) - 1) * 100, 2),
        "sma20_to_sma50_pct":  round((sma20.iloc[-1] / _safe(sma50.iloc[-1], cp) - 1) * 100, 2),
        "golden_cross": golden_cross,
        "death_cross": death_cross,
        "signal": "bullish" if sma20.iloc[-1] > sma50.iloc[-1] else "bearish",
    }

    # ── RSI(14) ──────────────────────────────────────────────────────────────
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = (-delta).clip(lower=0)
    avg_gain = gain.ewm(alpha=1/14, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/14, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    rsi_val = _safe(rsi.iloc[-1], 50.0)

    ind["rsi"] = {
        "rsi_14": rsi_val,
        "trend": "up" if rsi.iloc[-1] > rsi.iloc[-2] else "down",
        "status": (
            "overbought" if rsi_val > 70 else
            "near_overbought" if rsi_val > 60 else
            "near_oversold" if rsi_val < 40 else
            "oversold" if rsi_val < 30 else "neutral"
        ),
        "signal": "oversold" if rsi_val < 30 else "overbought" if rsi_val > 70 else "neutral",
    }

    # ── MACD(12/26/9) ────────────────────────────────────────────────────────
    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    macd_line = ema12 - ema26
    signal_line = macd_line.ewm(span=9, adjust=False).mean()
    histogram = macd_line - signal_line

    ind["macd"] = {
        "macd_line": round(float(macd_line.iloc[-1]), 4),
        "signal_line": round(float(signal_line.iloc[-1]), 4),
        "histogram": round(float(histogram.iloc[-1]), 4),
        "signal": "bullish" if macd_line.iloc[-1] > signal_line.iloc[-1] else "bearish",
        "crossover": (
            "bullish_cross" if macd_line.iloc[-1] > signal_line.iloc[-1] and macd_line.iloc[-2] <= signal_line.iloc[-2]
            else "bearish_cross" if macd_line.iloc[-1] < signal_line.iloc[-1] and macd_line.iloc[-2] >= signal_line.iloc[-2]
            else "none"
        ),
    }

    # ── Bollinger Bands(20, 2) ───────────────────────────────────────────────
    mb = sma20
    std = close.rolling(20).std()
    ub = mb + 2 * std
    lb = mb - 2 * std
    bw = (ub.iloc[-1] - lb.iloc[-1]) / _safe(mb.iloc[-1], cp)
    pct_b = (cp - lb.iloc[-1]) / (ub.iloc[-1] - lb.iloc[-1]) if (ub.iloc[-1] - lb.iloc[-1]) > 0 else 0.5

    ind["bollinger_bands"] = {
        "upper": _safe(ub.iloc[-1], cp),
        "middle": _safe(mb.iloc[-1], cp),
        "lower": _safe(lb.iloc[-1], cp),
        "percent_b": round(pct_b, 3),
        "bandwidth": round(bw, 4),
        "squeeze": bool(bw < 0.1),
        "signal": "squeeze" if bw < 0.1 else "expansion",
    }

    # ── ATR(14) + Volatility ─────────────────────────────────────────────────
    tr1 = high - low
    tr2 = (high - close.shift()).abs()
    tr3 = (low - close.shift()).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(14).mean()
    atr_20avg = atr.rolling(20).mean()
    vol_ratio = atr.iloc[-1] / atr_20avg.iloc[-1] if _safe(atr_20avg.iloc[-1], 0) > 0 else 1.0

    ind["volatility"] = {
        "atr_14": _safe(atr.iloc[-1], 0.0),
        "atr_20_avg": _safe(atr_20avg.iloc[-1], 0.0),
        "volatility_ratio": round(vol_ratio, 2),
        "bb_squeeze": bool(bw < 0.1),
        "regime": "high" if vol_ratio > 1.5 else "low" if vol_ratio < 0.7 else "normal",
    }

    # ── ADX(14) ──────────────────────────────────────────────────────────────
    up_move   = high.diff()
    down_move = low.shift() - low
    plus_dm   = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm  = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    plus_dm_s  = pd.Series(plus_dm,  index=df.index).rolling(14).mean()
    minus_dm_s = pd.Series(minus_dm, index=df.index).rolling(14).mean()
    atr14     = atr
    plus_di   = 100 * plus_dm_s / atr14.replace(0, np.nan)
    minus_di  = 100 * minus_dm_s / atr14.replace(0, np.nan)
    dx        = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)
    adx       = dx.rolling(14).mean()

    adx_val     = _safe(adx.iloc[-1], 20.0)
    plus_di_val = _safe(plus_di.iloc[-1], 25.0)
    minus_di_val= _safe(minus_di.iloc[-1], 25.0)

    ind["adx"] = {
        "adx": adx_val,
        "plus_di": plus_di_val,
        "minus_di": minus_di_val,
        "trend_direction": "bullish" if plus_di_val > minus_di_val else "bearish",
        "trend_strength": "strong" if adx_val > 25 else "weak",
    }

    # ── Stochastic(14, 3) ────────────────────────────────────────────────────
    lowest_low    = low.rolling(14).min()
    highest_high  = high.rolling(14).max()
    stoch_k = 100 * (close - lowest_low) / (highest_high - lowest_low).replace(0, np.nan)
    stoch_d = stoch_k.rolling(3).mean()

    ind["stochastic"] = {
        "k": _safe(stoch_k.iloc[-1], 50.0),
        "d": _safe(stoch_d.iloc[-1], 50.0),
        "signal": (
            "oversold" if _safe(stoch_k.iloc[-1], 50.0) < 20 else
            "overbought" if _safe(stoch_k.iloc[-1], 50.0) > 80 else "neutral"
        ),
    }

    # ── OBV ──────────────────────────────────────────────────────────────────
    obv  = (np.sign(close.diff()) * vol).fillna(0).cumsum()
    ind["volume"] = {
        "obv": round(float(obv.iloc[-1]), 0),
        "obv_trend": "up" if obv.iloc[-1] > obv.iloc[-5] else "down",
        "volume_ratio": round(float(vol.iloc[-1] / vol.rolling(20).mean().iloc[-1]), 2) if vol.rolling(20).mean().iloc[-1] > 0 else 1.0,
        "signal": "high_volume" if vol.iloc[-1] > 1.5 * vol.rolling(20).mean().iloc[-1] else "normal",
    }

    # ── VWAP ─────────────────────────────────────────────────────────────────
    tp   = (high + low + close) / 3
    vwap = (tp * vol).cumsum() / vol.cumsum().replace(0, np.nan)
    vwap_val = _safe(vwap.iloc[-1], cp)

    # ── MFI(14) ──────────────────────────────────────────────────────────────
    mf_raw = tp * vol
    pos_mf = mf_raw.where(tp > tp.shift(), 0.0)
    neg_mf = mf_raw.where(tp < tp.shift(), 0.0)
    mfr    = pos_mf.rolling(14).sum() / neg_mf.rolling(14).sum().replace(0, np.nan)
    mfi    = 100 - (100 / (1 + mfr))
    mfi_val= _safe(mfi.iloc[-1], 50.0)

    # ── CMF(20) ──────────────────────────────────────────────────────────────
    clv = ((close - low) - (high - close)) / (high - low).replace(0, np.nan)
    cmf = (clv * vol).rolling(20).sum() / vol.rolling(20).sum().replace(0, np.nan)
    cmf_val = _safe(cmf.iloc[-1], 0.0)

    # ── A/D Line ─────────────────────────────────────────────────────────────
    ad_line = (clv * vol).fillna(0).cumsum()
    ad_trend = "up" if ad_line.iloc[-1] > ad_line.iloc[-20] else "down"

    ind["enhanced_volume"] = {
        "vwap": round(vwap_val, 2),
        "price_vs_vwap_pct": round((cp / vwap_val - 1) * 100, 2) if vwap_val > 0 else 0.0,
        "mfi": round(mfi_val, 2),
        "mfi_status": "overbought" if mfi_val > 80 else "oversold" if mfi_val < 20 else "neutral",
        "cmf": round(cmf_val, 4),
        "cmf_signal": "bullish" if cmf_val > 0 else "bearish",
        "ad_line_trend": ad_trend,
    }

    # ── Pivot Points (Standard, based on previous day) ────────────────────────
    H = float(high.iloc[-2])
    L = float(low.iloc[-2])
    C = float(close.iloc[-2])
    P = (H + L + C) / 3
    ind["pivot_points"] = {
        "pivot": round(P, 2),
        "r1": round(2 * P - L, 2),
        "r2": round(P + (H - L), 2),
        "s1": round(2 * P - H, 2),
        "s2": round(P - (H - L), 2),
    }

    return ind


def compute_ground_truth(symbol: str, end_date: str, term: str = "medium") -> Optional[str]:
    """
    Compute the forward-return ground truth label for a (symbol, date, term) triplet.
    Returns "Bullish", "Bearish", or "Neutral", or None if data unavailable.
    """
    window    = TERM_WINDOWS.get(term, 20)
    threshold = TERM_THRESHOLDS.get(term, 0.025)
    try:
        end_dt   = pd.to_datetime(end_date)
        fetch_end = end_dt + timedelta(days=window + 15)  # extra buffer for weekends/holidays
        ticker   = yf.Ticker(f"{symbol}.NS")
        df       = ticker.history(
            start=end_date,
            end=fetch_end.strftime("%Y-%m-%d"),
            auto_adjust=True,
        )
        if df.empty or len(df) < window:
            return None
        df.columns = df.columns.str.lower()
        entry_price  = float(df["close"].iloc[0])
        exit_price   = float(df["close"].iloc[min(window, len(df) - 1)])
        forward_ret  = (exit_price - entry_price) / entry_price
        if forward_ret > threshold:
            return "Bullish"
        elif forward_ret < -threshold:
            return "Bearish"
        else:
            return "Neutral"
    except Exception as e:
        logger.warning(f"[DataLoader] ground_truth failed for {symbol} on {end_date}: {e}")
        return None


def build_observation(symbol: str, date: str, term: str = "medium") -> Optional[Dict[str, Any]]:
    """
    Full pipeline: fetch OHLCV → compute indicators → package as observation dict.
    Returns None if data is insufficient.
    """
    df = fetch_ohlcv(symbol, date)
    if df is None:
        return None
    indicators = compute_indicators(df)
    cp = float(df["close"].iloc[-1])
    return {
        "symbol": symbol,
        "date": date,
        "term": term.upper(),
        "current_price": round(cp, 2),
        "indicators": indicators,
    }


def fetch_macro_context(date: str) -> Dict[str, Any]:
    """
    Fetch macro context from NIFTY50 index for a given date.
    Used for Task 3 (long-term) observations to give the agent market-wide context.
    Returns a dict with nifty_trend, nifty_return_20d, and market_regime.
    Falls back gracefully if NIFTY50 data is unavailable.
    """
    try:
        end_dt  = pd.to_datetime(date) + timedelta(days=1)
        start_dt = end_dt - timedelta(days=60)
        ticker = yf.Ticker("^NSEI")
        df = ticker.history(
            start=start_dt.strftime("%Y-%m-%d"),
            end=end_dt.strftime("%Y-%m-%d"),
            auto_adjust=True,
        )
        if df.empty or len(df) < 20:
            return {"nifty_trend": "Unknown", "nifty_return_20d": 0.0, "market_regime": "Unknown"}
        df.columns = df.columns.str.lower()
        close = df["close"]
        lookback = min(21, len(close))
        ret_20d = float((close.iloc[-1] - close.iloc[-lookback]) / close.iloc[-lookback])
        trend   = "Bullish" if ret_20d > 0.02 else "Bearish" if ret_20d < -0.02 else "Neutral"
        regime  = "trending" if abs(ret_20d) > 0.03 else "ranging"
        return {
            "nifty_trend":       trend,
            "nifty_return_20d":  round(ret_20d * 100, 2),
            "market_regime":     regime,
        }
    except Exception as e:
        logger.warning(f"[DataLoader] fetch_macro_context failed for {date}: {e}")
        return {"nifty_trend": "Unknown", "nifty_return_20d": 0.0, "market_regime": "Unknown"}


def build_multi_step_episode(
    symbol: str,
    start_date: str,
    n_steps: int = 5,
    term: str = "medium",
    lookback_days: int = 300,
    include_macro: bool = False,
) -> Optional[List[Tuple[Dict[str, Any], str, float]]]:
    """
    Build n_steps consecutive (observation_dict, ground_truth, actual_1day_return) tuples.
    Single OHLCV fetch per call — no per-step API calls.

    Returns list of n_steps tuples, or None if data is insufficient.
      observation_dict     : full indicator snapshot for that trading day
      ground_truth         : N-day forward return label (Bullish/Bearish/Neutral)
      actual_1day_return   : next-day return fraction (used for portfolio reward)

    Args:
        include_macro: if True, fetches NIFTY50 macro context once and embeds in each obs_dict.
                       Used for Task 3 (long-term) to give the agent market-wide awareness.
    """
    window    = TERM_WINDOWS.get(term, 20)
    threshold = TERM_THRESHOLDS.get(term, 0.025)
    try:
        start_dt    = pd.to_datetime(start_date)
        fetch_start = (start_dt - timedelta(days=lookback_days)).strftime("%Y-%m-%d")
        fetch_end   = (start_dt + timedelta(days=n_steps * 3 + window + 20)).strftime("%Y-%m-%d")

        ticker  = yf.Ticker(f"{symbol}.NS")
        full_df = ticker.history(start=fetch_start, end=fetch_end, auto_adjust=True)
        if full_df.empty or len(full_df) < lookback_days // 2:
            return None
        full_df.columns = full_df.columns.str.lower()
        full_df = full_df[["open", "high", "low", "close", "volume"]].dropna()
        full_df.index = pd.to_datetime(full_df.index).tz_localize(None)

        # n_steps consecutive trading days starting at or after start_date
        available_dates = full_df[full_df.index >= start_dt].index[:n_steps]
        if len(available_dates) < n_steps:
            return None

        # Fetch macro context once for the whole episode (Task 3 only)
        macro_ctx = fetch_macro_context(start_date) if include_macro else None

        steps = []
        for step_dt in available_dates:
            hist = full_df[full_df.index <= step_dt].tail(lookback_days)
            if len(hist) < 60:
                return None
            indicators = compute_indicators(hist)
            cp = float(hist["close"].iloc[-1])
            obs_dict: Dict[str, Any] = {
                "symbol":        symbol,
                "date":          step_dt.strftime("%Y-%m-%d"),
                "term":          term.upper(),
                "current_price": round(cp, 2),
                "indicators":    indicators,
            }
            if macro_ctx is not None:
                obs_dict["macro"] = macro_ctx

            # GT: N-day forward return label
            future = full_df[full_df.index > step_dt].head(window + 5)
            if len(future) < window:
                return None
            exit_ = float(future["close"].iloc[min(window, len(future)) - 1])
            fwd_ret = (exit_ - cp) / cp
            gt = "Bullish" if fwd_ret > threshold else "Bearish" if fwd_ret < -threshold else "Neutral"

            # Actual 1-day return (next trading day's close vs today's close)
            # Drives the portfolio reward in indicators_env.py
            next_day = full_df[full_df.index > step_dt].head(1)
            if len(next_day) >= 1:
                actual_1day_return = round((float(next_day["close"].iloc[0]) - cp) / cp, 6)
            else:
                actual_1day_return = 0.0

            steps.append((obs_dict, gt, actual_1day_return))

        return steps if len(steps) == n_steps else None

    except Exception as e:
        logger.warning(f"[DataLoader] build_multi_step_episode failed for {symbol}/{start_date}: {e}")
        return None


def generate_scenario_pool(
    symbols: Optional[List[str]] = None,
    start_date: str = "2019-01-01",
    end_date: str = "2024-12-31",
    term: str = "medium",
    max_scenarios: int = 50_000,
) -> List[Dict[str, str]]:
    """
    Pre-generate a pool of (symbol, date) pairs that have valid ground truth labels.
    Used to populate the environment's scenario queue.
    """
    if symbols is None:
        symbols = NSE_UNIVERSE
    window = TERM_WINDOWS.get(term, 20)
    # Generate monthly sample dates (avoids look-ahead: stops window days before end_date)
    cutoff = (pd.to_datetime(end_date) - timedelta(days=window + 5)).strftime("%Y-%m-%d")
    dates = pd.bdate_range(start=start_date, end=cutoff, freq="10B").strftime("%Y-%m-%d").tolist()

    pool = []
    for sym in symbols:
        for dt in dates:
            pool.append({"symbol": sym, "date": dt, "term": term})
            if len(pool) >= max_scenarios:
                break
        if len(pool) >= max_scenarios:
            break
    return pool


# ─── Helpers ─────────────────────────────────────────────────────────────────

def _safe(val: Any, default: float) -> float:
    """Return float val, defaulting if NaN/None."""
    try:
        v = float(val)
        return default if np.isnan(v) else round(v, 4)
    except Exception:
        return default


# ─── Batch offline dataset generator  (no per-episode API calls) ─────────────

def generate_dataset_offline(
    symbols: Optional[List[str]] = None,
    start_date: str = "2020-01-01",
    end_date: str = "2024-06-30",
    term: str = "medium",
    dates_per_stock: int = 15,
    max_total: int = 5000,
    save_path: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """
    Batch dataset builder: 1 yfinance API call per stock → many training examples.

    For each stock we download full history ONCE, then slice out `dates_per_stock`
    evenly-spaced windows. Each window gives indicators + forward-return GT.
    This avoids per-episode yfinance calls and prevents Colab rate-limiting.

    Args:
        symbols     : list of NSE symbols (default: first 30 of NSE_UNIVERSE)
        start_date  : earliest date to sample (needs lookback buffer)
        end_date    : latest date to sample (will stop `window` days before this)
        term        : prediction term (intraday/short/medium/long)
        dates_per_stock : candidate dates to sample per stock
        max_total   : cap on total dataset size
        save_path   : if given, saves as JSON for later reload

    Returns:
        List of dicts: {symbol, date, term, current_price, indicators, ground_truth, prompt}
    """
    import json, time

    if symbols is None:
        symbols = NSE_UNIVERSE[:30]

    window    = TERM_WINDOWS.get(term, 20)
    threshold = TERM_THRESHOLDS.get(term, 0.025)
    lookback  = 300  # days of history needed for indicators

    # Generate candidate dates (evenly spread, no weekends)
    cutoff = (pd.to_datetime(end_date) - timedelta(days=window + 5)).strftime("%Y-%m-%d")
    all_dates = pd.bdate_range(
        start=(pd.to_datetime(start_date) + timedelta(days=lookback)).strftime("%Y-%m-%d"),
        end=cutoff,
    )
    step = max(1, len(all_dates) // dates_per_stock)
    sample_dates = [d.strftime("%Y-%m-%d") for d in all_dates[::step]][:dates_per_stock]

    dataset: List[Dict[str, Any]] = []

    for sym_idx, symbol in enumerate(symbols):
        if len(dataset) >= max_total:
            break
        try:
            # ── Single API call: fetch full history for this stock ──────────
            fetch_start = (pd.to_datetime(start_date) - timedelta(days=5)).strftime("%Y-%m-%d")
            fetch_end   = (pd.to_datetime(end_date) + timedelta(days=window + 20)).strftime("%Y-%m-%d")
            ticker = yf.Ticker(f"{symbol}.NS")
            full_df = ticker.history(
                start=fetch_start, end=fetch_end, auto_adjust=True
            )
            if full_df.empty or len(full_df) < lookback:
                logger.warning(f"[Offline] {symbol}: insufficient data ({len(full_df)} rows). Skipping.")
                continue
            full_df.columns = full_df.columns.str.lower()
            full_df = full_df[["open", "high", "low", "close", "volume"]].dropna()
            full_df.index = pd.to_datetime(full_df.index).tz_localize(None)

            logger.info(f"[Offline] {symbol} ({sym_idx+1}/{len(symbols)}): {len(full_df)} rows fetched → slicing {len(sample_dates)} dates")

            for date_str in sample_dates:
                if len(dataset) >= max_total:
                    break
                try:
                    target_dt = pd.to_datetime(date_str)

                    # Slice history up to this date (lookback window for indicators)
                    hist = full_df[full_df.index <= target_dt].tail(lookback)
                    if len(hist) < 60:
                        continue

                    # Ground truth: forward return from this date
                    future = full_df[full_df.index > target_dt].head(window + 5)
                    if len(future) < window:
                        continue
                    entry = float(hist["close"].iloc[-1])
                    exit_ = float(future["close"].iloc[min(window, len(future)) - 1])
                    fwd_ret = (exit_ - entry) / entry
                    if fwd_ret > threshold:
                        gt = "Bullish"
                    elif fwd_ret < -threshold:
                        gt = "Bearish"
                    else:
                        gt = "Neutral"

                    # Compute indicators from historical slice
                    indicators = compute_indicators(hist)

                    dataset.append({
                        "symbol": symbol,
                        "date": date_str,
                        "term": term.upper(),
                        "current_price": round(entry, 2),
                        "indicators": indicators,
                        "ground_truth": gt,
                    })
                except Exception as e:
                    logger.debug(f"[Offline] {symbol}/{date_str} skipped: {e}")
                    continue

            # Small pause between stocks to be polite to yfinance
            time.sleep(0.3)

        except Exception as e:
            logger.warning(f"[Offline] {symbol}: fetch failed: {e}")
            continue

    logger.info(f"[Offline] Dataset complete: {len(dataset)} episodes from {len(symbols)} stocks")

    if save_path:
        import json as _json
        with open(save_path, "w") as f:
            _json.dump(dataset, f)
        logger.info(f"[Offline] Saved to {save_path}")

    return dataset

