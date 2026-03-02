"""
Dual-source data layer:
  SmartAPI  -> live OHLCV candles (true real-time)
  yfinance  -> sentiment, fundamentals, 52w data
"""
from __future__ import annotations
import os, time
from datetime import datetime, timedelta
from typing import Optional
import pandas as pd
import yfinance as yf
from dotenv import load_dotenv

load_dotenv()

POSITIVE_KW = ["surge","rally","profit","growth","buy","upgrade","record",
               "strong","beat","outperform","dividend","win","gain","rise",
               "bullish","breakout","target","upside"]
NEGATIVE_KW = ["fall","drop","loss","sell","downgrade","weak","miss",
               "underperform","crash","debt","fraud","cut","bearish",
               "risk","concern","pressure","below"]

def to_yf_ticker(nse_symbol: str) -> str:
    clean = nse_symbol.replace("-EQ","").replace("-BE","").strip()
    return f"{clean}.NS"

def fetch_yf_sentiment(nse_symbol: str) -> dict:
    ticker = to_yf_ticker(nse_symbol)
    score = 50
    pos = neg = headlines_count = 0
    try:
        news = yf.Ticker(ticker).news or []
        headlines = [
            (n.get("content",{}).get("title") or n.get("title","")).lower()
            for n in news[:20]
        ]
        headlines = [h for h in headlines if h]
        headlines_count = len(headlines)
        pos = sum(1 for h in headlines for w in POSITIVE_KW if w in h)
        neg = sum(1 for h in headlines for w in NEGATIVE_KW if w in h)
        total = pos + neg
        if total > 0:
            score = round(pos / total * 100, 2)
        time.sleep(0.3)
    except Exception as e:
        print(f"  Sentiment error ({nse_symbol}): {e}")
    return {
        "sentiment_score": score,
        "sentiment_positive": pos,
        "sentiment_negative": neg,
        "sentiment_headlines": headlines_count,
    }

def fetch_yf_fundamentals(nse_symbol: str) -> dict:
    ticker = to_yf_ticker(nse_symbol)
    result = {"pe_ratio": None, "market_cap": None, "eps": None,
              "52w_high": None, "52w_low": None,
              "avg_volume_10d": None, "analyst_target": None, "beta": None}
    try:
        info = yf.Ticker(ticker).info
        result.update({
            "pe_ratio":       info.get("trailingPE"),
            "market_cap":     info.get("marketCap"),
            "eps":            info.get("trailingEps"),
            "52w_high":       info.get("fiftyTwoWeekHigh"),
            "52w_low":        info.get("fiftyTwoWeekLow"),
            "avg_volume_10d": info.get("averageVolume10days"),
            "analyst_target": info.get("targetMeanPrice"),
            "beta":           info.get("beta"),
        })
        time.sleep(0.3)
    except Exception as e:
        print(f"  Fundamentals error ({nse_symbol}): {e}")
    return result

def fetch_yf_candles(nse_symbol: str, period: str = "6mo",
                     interval: str = "15m") -> Optional[pd.DataFrame]:
    try:
        ticker = to_yf_ticker(nse_symbol)
        df = yf.download(ticker, period=period, interval=interval,
                         progress=False, auto_adjust=True)
        if df.empty:
            return None
        df = df.reset_index()
        df.columns = [c.lower() if isinstance(c,str) else c[0].lower() for c in df.columns]
        df = df.rename(columns={"date":"datetime","index":"datetime"})
        return df[["datetime","open","high","low","close","volume"]]
    except Exception as e:
        print(f"  yfinance candles error ({nse_symbol}): {e}")
    return None

def fetch_smartapi_candles(symbol_token: str, interval: str = "FIFTEEN_MINUTE", days: int = 30):
    """Delegates to smartapi_client — single session, chunked, retry-safe."""
    from src.trading_ensemble.data.smartapi_client import login_from_env, fetch_candles_chunked
    from datetime import datetime, timedelta
    session  = login_from_env()
    end_dt   = datetime.now()
    start_dt = end_dt - timedelta(days=days)
    return fetch_candles_chunked(
        session.smart, exchange="NSE", symbol_token=str(symbol_token),
        interval=interval,
        start=start_dt.strftime("%Y-%m-%d 0915"),
        end=end_dt.strftime("%Y-%m-%d 1530"),
        chunk_days=60,
    )

def fetch_candles(nse_symbol: str, symbol_token: str = None,
                  interval: str = "FIFTEEN_MINUTE") -> Optional[pd.DataFrame]:
    if symbol_token:
        df = fetch_smartapi_candles(symbol_token, interval)
        if df is not None and not df.empty:
            print(f"  [{nse_symbol}] SmartAPI live ({len(df)} candles)")
            return df
    df = fetch_yf_candles(nse_symbol)
    if df is not None and not df.empty:
        print(f"  [{nse_symbol}] yfinance fallback ({len(df)} candles, 15min delay)")
        return df
    print(f"  [{nse_symbol}] No data source available")
    return None
