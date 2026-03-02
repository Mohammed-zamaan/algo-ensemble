"""
Stock Screener - Multi-metric scoring system
Metrics: Technical + Sentiment + Volatility + Volume
"""
from __future__ import annotations
import os, re, time
from pathlib import Path
from dotenv import load_dotenv
import pandas as pd
import numpy as np
import requests
from bs4 import BeautifulSoup
from comet_ml import Experiment

load_dotenv()

from src.trading_ensemble.core.engine import (
    validate_ohlcv, compute_indicators,
    compute_filters, compute_dynamic_donchian, compute_signals,
)
from src.trading_ensemble.core.params import StrategyParams

# ─── WEIGHTS (must sum to 100) ───────────────────────────────────────────────
WEIGHTS = {
    "technical":  40,   # Donchian signals, ADX, ATR
    "sentiment":  25,   # News + NSE announcements
    "volatility": 20,   # ATR%, volume spike
    "momentum":   15,   # Signal rate, trend consistency
}

# ─── 1. TECHNICAL SCORE ──────────────────────────────────────────────────────
def technical_score(df_raw: pd.DataFrame) -> dict:
    p = StrategyParams()
    df = validate_ohlcv(df_raw.copy())
    df = compute_indicators(df, p)
    df = compute_filters(df, p)
    df = compute_dynamic_donchian(df, p)
    all_rows = compute_signals(df)

    total       = len(all_rows)
    long_entries= int(all_rows["long_signal"].sum())
    exits       = int(all_rows["don_exit_signal"].sum())
    signal_rate = round(long_entries / total * 100, 2) if total > 0 else 0

    # ADX strength (last 20 candles avg)
    adx_mean = float(all_rows["adx"].tail(20).mean()) if "adx" in all_rows.columns else 0
    adx_score = min(adx_mean / 40 * 100, 100)  # 40 ADX = max score

    # ATR% (normalized volatility)
    atrp_mean = float(all_rows["atrp"].tail(20).mean()) if "atrp" in all_rows.columns else 0

    # Signal consistency (entries vs exits balance)
    balance = min(long_entries, exits) / max(long_entries, exits) * 100 if max(long_entries, exits) > 0 else 0

    score = (
        min(signal_rate * 5, 40) +   # Signal rate contributes up to 40pts
        adx_score * 0.4 +             # ADX contributes up to 40pts
        balance * 0.2                 # Balance contributes up to 20pts
    )

    return {
        "long_entries": long_entries,
        "exits": exits,
        "signal_rate_pct": signal_rate,
        "adx_mean": round(adx_mean, 2),
        "atrp_mean": round(atrp_mean, 2),
        "technical_score": round(min(score, 100), 2),
    }

# ─── 2. SENTIMENT SCORE ──────────────────────────────────────────────────────
def sentiment_score(symbol: str) -> dict:
    """
    Scrapes Moneycontrol news headlines for the symbol.
    Scores based on positive/negative keyword presence.
    """
    POSITIVE = ["surge", "rally", "profit", "growth", "buy", "upgrade",
                "record", "strong", "beat", "outperform", "dividend", "win"]
    NEGATIVE = ["fall", "drop", "loss", "sell", "downgrade", "weak",
                "miss", "underperform", "crash", "debt", "fraud", "cut"]

    clean_sym = symbol.replace("-EQ", "").replace("-", "")
    score = 50  # neutral default
    headlines_found = 0

    try:
        url = f"https://www.moneycontrol.com/stocks/coms/news.php?sc_id={clean_sym}"
        headers = {"User-Agent": "Mozilla/5.0"}
        resp = requests.get(url, headers=headers, timeout=5)
        soup = BeautifulSoup(resp.text, "html.parser")
        headlines = [h.get_text(strip=True).lower() for h in soup.find_all(["h2","h3","li"], limit=20)]
        headlines_found = len(headlines)

        pos = sum(1 for h in headlines for w in POSITIVE if w in h)
        neg = sum(1 for h in headlines for w in NEGATIVE if w in h)
        total_kw = pos + neg

        if total_kw > 0:
            score = round((pos / total_kw) * 100, 2)
        time.sleep(0.5)  # polite scraping
    except Exception:
        pass  # fallback to neutral 50

    return {
        "sentiment_score": score,
        "sentiment_headlines": headlines_found,
    }

# ─── 3. VOLATILITY SCORE ─────────────────────────────────────────────────────
def volatility_score(df_raw: pd.DataFrame) -> dict:
    df = df_raw.copy()
    df.columns = [c.strip().lower() for c in df.columns]

    # Volume spike: last 5 candles vs 20-candle average
    vol_ma   = df["volume"].rolling(20).mean()
    vol_spike= float((df["volume"].tail(5).mean() / vol_ma.tail(5).mean())) if vol_ma.tail(5).mean() > 0 else 1
    vol_score= min(vol_spike * 50, 100)

    # Price range % (high-low / close)
    df["range_pct"] = (df["high"] - df["low"]) / df["close"] * 100
    range_mean = float(df["range_pct"].tail(20).mean())
    range_score= min(range_mean * 10, 100)

    # 52-week high proximity
    high_52w  = float(df["high"].max())
    last_close= float(df["close"].iloc[-1])
    proximity = (last_close / high_52w) * 100

    return {
        "vol_spike":       round(vol_spike, 2),
        "range_pct_mean":  round(range_mean, 2),
        "high_52w":        round(high_52w, 2),
        "last_close":      round(last_close, 2),
        "proximity_52w":   round(proximity, 2),
        "volatility_score":round((vol_score + range_score + proximity) / 3, 2),
    }

# ─── 4. MOMENTUM SCORE ───────────────────────────────────────────────────────
def momentum_score(df_raw: pd.DataFrame) -> dict:
    df = df_raw.copy()
    df.columns = [c.strip().lower() for c in df.columns]

    # Price momentum: last close vs 20-candle ago
    if len(df) >= 20:
        ret_20 = (df["close"].iloc[-1] - df["close"].iloc[-20]) / df["close"].iloc[-20] * 100
    else:
        ret_20 = 0

    # Consecutive up candles (last 5)
    df["up"] = df["close"] > df["open"]
    consec_up = int(df["up"].tail(5).sum())

    score = min(max(ret_20 * 5 + 50, 0), 100) * 0.6 + consec_up / 5 * 100 * 0.4

    return {
        "return_20c_pct":  round(ret_20, 2),
        "consec_up_5c":    consec_up,
        "momentum_score":  round(score, 2),
    }

# ─── 5. MASTER SCORER ────────────────────────────────────────────────────────
def composite_score(tech: dict, sent: dict, vol: dict, mom: dict) -> float:
    return round(
        tech["technical_score"]  * WEIGHTS["technical"]  / 100 +
        sent["sentiment_score"]  * WEIGHTS["sentiment"]  / 100 +
        vol["volatility_score"]  * WEIGHTS["volatility"] / 100 +
        mom["momentum_score"]    * WEIGHTS["momentum"]   / 100,
        2
    )

# ─── 6. MAIN ─────────────────────────────────────────────────────────────────
def main():
    source = Path(os.getenv("CSV_SOURCE", "trading_candles"))
    files  = sorted(source.glob("*.csv"))
    if not files:
        raise SystemExit(f"No CSVs in {source}/")

    print(f"Screening {len(files)} stocks...\n")
    results = []

    for f in files:
        symbol = f.name.split("_")[0]
        print(f"  Analyzing {symbol}...", end=" ")

        try:
            df_raw = pd.read_csv(f)

            tech = technical_score(df_raw)
            sent = sentiment_score(symbol)
            vol  = volatility_score(df_raw)
            mom  = momentum_score(df_raw)
            comp = composite_score(tech, sent, vol, mom)

            row = {"symbol": symbol, "composite_score": comp,
                   **tech, **sent, **vol, **mom}
            results.append(row)

            # Log to Comet
            exp = Experiment(
                api_key=os.getenv("COMET_API_KEY"),
                project_name="stock-screener",
                workspace=os.getenv("COMET_WORKSPACE", "zamaan"),
            )
            exp.set_name(symbol)
            exp.log_metrics({k: v for k, v in row.items() if isinstance(v, (int, float))})
            exp.log_parameter("symbol", symbol)
            exp.end()

            print(f"Score={comp} ✅")

        except Exception as e:
            print(f"❌ {e}")

    # Save ranked output
    out = pd.DataFrame(results).sort_values("composite_score", ascending=False)
    out_path = Path("results/screener_ranked.csv")
    out_path.parent.mkdir(exist_ok=True)
    out.to_csv(out_path, index=False)

    print(f"\n{'='*50}")
    print("TOP 10 STOCKS BY COMPOSITE SCORE:")
    print(out[["symbol","composite_score","technical_score",
               "sentiment_score","volatility_score","momentum_score"]].head(10).to_string(index=False))
    print(f"\n✅ Full results → {out_path}")
    print("📊 Comet → https://www.comet.com/zamaan/stock-screener")

if __name__ == "__main__":
    main()
