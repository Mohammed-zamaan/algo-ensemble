# market_regime.py
# HASH-009: Market Regime Detector
# Uses NIFTY50 + BANKNIFTY trend + India VIX volatility proxy.
# Output: BULL | SIDEWAYS | BEAR | CRISIS

import os
import pandas as pd
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()

USE_YFINANCE = os.getenv("REGIME_USE_YFINANCE", "true").lower() == "true"

TICKERS = {"NIFTY": "^NSEI", "BANKNIFTY": "^NSEBANK", "INDIAVIX": "^INDIAVIX"}

VIX_CAUTION = float(os.getenv("VIX_CAUTION", 20))
VIX_CRISIS  = float(os.getenv("VIX_CRISIS", 30))
SLOPE_DAYS  = int(os.getenv("REGIME_SLOPE_DAYS", 5))

def _ma(series: pd.Series, n: int) -> pd.Series:
    return series.rolling(n).mean()

def _slope(series: pd.Series, n: int) -> float:
    s = series.dropna().tail(n)
    if len(s) < n:
        return 0.0
    return float((s.iloc[-1] - s.iloc[0]) / s.iloc[0] * 100)

def fetch_index_history(ticker: str, period: str = "1y") -> pd.DataFrame:
    if not USE_YFINANCE:
        raise RuntimeError("REGIME_USE_YFINANCE=false but no alternate source configured")
    import yfinance as yf
    df = yf.Ticker(ticker).history(period=period, interval="1d")
    if df.empty:
        raise ValueError(f"No data for {ticker}")
    return df

def classify_trend(close: pd.Series) -> dict:
    ma50  = _ma(close, 50)
    ma200 = _ma(close, 200)
    last = close.iloc[-1]
    last50 = ma50.iloc[-1]
    last200 = ma200.iloc[-1]
    slope = _slope(ma50, SLOPE_DAYS)

    if pd.isna(last50) or pd.isna(last200):
        return {"trend": "UNKNOWN", "slope": slope}

    if last > last50 and last50 > last200 and slope > 0:
        return {"trend": "BULL", "slope": slope}
    if last < last50 and last50 < last200 and slope < 0:
        return {"trend": "BEAR", "slope": slope}
    return {"trend": "SIDEWAYS", "slope": slope}

def detect_regime() -> dict:
    nifty = fetch_index_history(TICKERS["NIFTY"])
    bank  = fetch_index_history(TICKERS["BANKNIFTY"])
    vix   = fetch_index_history(TICKERS["INDIAVIX"], period="6mo")

    nifty_trend = classify_trend(nifty["Close"])
    bank_trend  = classify_trend(bank["Close"])
    vix_last = float(vix["Close"].iloc[-1])

    if vix_last >= VIX_CRISIS:
        regime = "CRISIS"
    else:
        trends = [nifty_trend["trend"], bank_trend["trend"]]
        if trends.count("BEAR") >= 1 and vix_last >= VIX_CAUTION:
            regime = "BEAR"
        elif trends.count("BULL") == 2 and vix_last < VIX_CAUTION:
            regime = "BULL"
        elif trends.count("BEAR") == 2:
            regime = "BEAR"
        else:
            regime = "SIDEWAYS"

    return {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "regime": regime,
        "nifty_trend": nifty_trend["trend"],
        "banknifty_trend": bank_trend["trend"],
        "nifty_ma50_slope_pct": round(nifty_trend["slope"], 4),
        "banknifty_ma50_slope_pct": round(bank_trend["slope"], 4),
        "india_vix": round(vix_last, 2),
        "vix_caution": VIX_CAUTION,
        "vix_crisis": VIX_CRISIS,
    }

def regime_multiplier(regime: str) -> float:
    regime = regime.upper()
    if regime == "BULL":
        return 1.0
    if regime == "SIDEWAYS":
        return 0.5
    if regime == "BEAR":
        return 0.25
    if regime == "CRISIS":
        return 0.0
    return 0.5

if __name__ == "__main__":
    info = detect_regime()
    mult = regime_multiplier(info["regime"])
    print("\n[REGIME] Market Regime Detector")
    for k, v in info.items():
        print(f"  {k:>20}: {v}")
    print(f"  {'risk_multiplier':>20}: {mult}")
