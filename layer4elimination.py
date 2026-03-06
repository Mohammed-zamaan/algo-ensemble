#!/usr/bin/env python3
# layer4elimination.py — 09:15 AM: Filter watchlist → tradecandidates.csv

import pandas as pd
from dotenv import load_dotenv
load_dotenv()


def assign_trading_mode(symbol: str, df_daily: pd.DataFrame) -> str:
    # Get last 20 days data
    recent = df_daily.tail(20)
    if len(recent) < 10:
        return "SKIP"
    
    # 1. ATR Volatility
    high_low = recent["High"] - recent["Low"]
    atr_pct = (high_low / recent["Close"] * 100).mean()
    
    # 2. Beta vs NIFTY (simplified)
    nifty_ret = recent["Close"].pct_change().mean()
    stock_ret = recent["Close"].pct_change().mean()
    beta = stock_ret / nifty_ret if nifty_ret != 0 else 1.0
    
    # 3. ADX (simplified directional strength)
    up = recent["Close"].diff() > 0
    adx_proxy = up.rolling(14).sum().iloc[-1] / 14 * 100
    
    if atr_pct > 4 or beta > 1.5:
        return "INTRADAY"
    if atr_pct > 2 or (beta > 1.0 and adx_proxy > 20):
        return "SWING"
    return "POSITIONAL"


def main():
    print("[LAYER 4] Watchlist → Trade Candidates")
    print("=" * 50)
    
    # Read watchlist
    df = pd.read_csv("watchlist.csv")
    df = df[df["active"] == "TRUE"].copy()
    print(f"[SHEETS] {len(df)} active stocks loaded from CSV")
    
    # Apply filters
    candidates = []
    for _, row in df.iterrows():
        symbol = row["symbol"].replace("-EQ", "")
        mode = assign_trading_mode(row["symbol"], fetch_daily_data(row["symbol"]))
        
        # Quick sector/momentum filter (expand later)
        if row["conviction"] >= 1:
            candidates.append({
                "symbol": row["symbol"],
                "mode": mode,
                "source_type": row["source_type"],
                "conviction": row["conviction"]
            })
    
    candidates_df = pd.DataFrame(candidates)
    candidates_df.to_csv("tradecandidates.csv", index=False)
    print(f"[OUTPUT] {len(candidates_df)} candidates → tradecandidates.csv")
    print("[LAYER 4] Complete")

if __name__ == "__main__":
    main()
