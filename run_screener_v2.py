from __future__ import annotations
import os
from pathlib import Path
from dotenv import load_dotenv
import pandas as pd
from comet_ml import Experiment

load_dotenv()

from src.trading_ensemble.core.engine import (
    validate_ohlcv, compute_indicators,
    compute_filters, compute_dynamic_donchian, compute_signals,
)
from src.trading_ensemble.core.params import StrategyParams
from src.trading_ensemble.data.dual_source import (
    fetch_yf_sentiment, fetch_yf_fundamentals
)

WEIGHTS = {"technical": 35, "sentiment": 25, "fundamental": 20, "volatility": 20}

def technical_score(df_raw):
    p = StrategyParams()
    df = validate_ohlcv(df_raw.copy())
    df = compute_indicators(df, p)
    df = compute_filters(df, p)
    df = compute_dynamic_donchian(df, p)
    rows = compute_signals(df)
    total = len(rows)
    long_entries = int(rows["long_signal"].sum())
    exits = int(rows["don_exit_signal"].sum())
    signal_rate = round(long_entries / total * 100, 2) if total > 0 else 0
    adx_mean = float(rows["adx"].tail(20).mean()) if "adx" in rows.columns else 0
    adx_score = min(adx_mean / 40 * 100, 100)
    balance = min(long_entries, exits) / max(long_entries, exits) * 100 if max(long_entries, exits) > 0 else 0
    score = min(signal_rate * 5, 40) + adx_score * 0.4 + balance * 0.2
    return {"long_entries": long_entries, "exits": exits,
            "signal_rate_pct": signal_rate, "adx_mean": round(adx_mean, 2),
            "technical_score": round(min(score, 100), 2)}

def fundamental_score(fund):
    score = 50
    pe = fund.get("pe_ratio") or 0
    if 0 < pe < 15:      score += 20
    elif 15 <= pe < 25:  score += 10
    elif pe > 50:        score -= 15
    beta = fund.get("beta") or 1
    if 0.5 < beta < 1.2: score += 10
    elif beta > 2:        score -= 10
    target = fund.get("analyst_target") or 0
    high52 = fund.get("52w_high") or 0
    if target and high52:
        upside = (target - high52) / high52 * 100
        score += min(upside, 20)
    return round(min(max(score, 0), 100), 2)

def volatility_score(df_raw):
    df = df_raw.copy()
    df.columns = [c.strip().lower() for c in df.columns]
    vol_ma = df["volume"].rolling(20).mean()
    vol_spike = float(df["volume"].tail(5).mean() / vol_ma.tail(5).mean()) if vol_ma.tail(5).mean() > 0 else 1
    df["range_pct"] = (df["high"] - df["low"]) / df["close"] * 100
    range_mean = float(df["range_pct"].tail(20).mean())
    last_close = float(df["close"].iloc[-1])
    high_52w = float(df["high"].max())
    proximity = (last_close / high_52w) * 100
    return {"vol_spike": round(vol_spike, 2), "last_close": round(last_close, 2),
            "proximity_52w": round(proximity, 2),
            "volatility_score": round((min(vol_spike*50,100) + min(range_mean*10,100) + proximity) / 3, 2)}

def composite(tech, sent, fund_s, vol):
    return round(
        tech["technical_score"] * WEIGHTS["technical"]   / 100 +
        sent["sentiment_score"] * WEIGHTS["sentiment"]   / 100 +
        fund_s                  * WEIGHTS["fundamental"] / 100 +
        vol["volatility_score"] * WEIGHTS["volatility"]  / 100, 2)

def main():
    source = Path(os.getenv("CSV_SOURCE", "trading_candles"))
    files = sorted(source.glob("*.csv"))
    if not files:
        raise SystemExit("No CSVs found in " + str(source))

    print("Screening " + str(len(files)) + " stocks with SmartAPI + yfinance...\n")
    results = []

    for f in files:
        symbol = f.name.split("_")[0]
        print("  " + symbol + "...", end=" ", flush=True)
        try:
            df_raw = pd.read_csv(f)
            tech   = technical_score(df_raw)
            sent   = fetch_yf_sentiment(symbol)
            fund   = fetch_yf_fundamentals(symbol)
            fund_s = fundamental_score(fund)
            vol    = volatility_score(df_raw)
            comp   = composite(tech, sent, fund_s, vol)

            row = {"symbol": symbol, "composite_score": comp,
                   **tech, **sent, **fund, "fundamental_score": fund_s, **vol}
            results.append(row)

            exp = Experiment(
                api_key=os.getenv("COMET_API_KEY"),
                project_name="stock-screener",
                workspace=os.getenv("COMET_WORKSPACE", "zamaan"),
            )
            exp.set_name(symbol + "_v2")
            exp.log_metrics({k: v for k, v in row.items() if isinstance(v, (int, float))})
            exp.log_parameter("symbol", symbol)
            exp.log_parameter("version", "v2_smartapi_yfinance")
            exp.end()

            print("composite=" + str(comp) + " | sentiment=" + str(sent["sentiment_score"]) + " | PE=" + str(fund.get("pe_ratio")) + " OK")

        except Exception as e:
            print("ERROR: " + str(e))

    out = pd.DataFrame(results).sort_values("composite_score", ascending=False)
    out_path = Path("results/screener_v2_ranked.csv")
    out_path.parent.mkdir(exist_ok=True)
    out.to_csv(out_path, index=False)

    print("\n" + "="*60)
    print("TOP 10 STOCKS - v2 Screener (SmartAPI + yfinance):")
    print(out[["symbol","composite_score","technical_score",
               "sentiment_score","fundamental_score","volatility_score"]].head(10).to_string(index=False))
    print("\nFull results saved -> results/screener_v2_ranked.csv")
    print("Comet -> https://www.comet.com/zamaan/stock-screener")

if __name__ == "__main__":
    main()
