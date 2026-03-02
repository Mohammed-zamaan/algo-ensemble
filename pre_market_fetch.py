"""
pre_market_fetch.py — Run at 08:45 AM before market open.
Fetches sentiment + fundamentals for all Comet stocks and caches to JSON.
Live pipeline reads from cache instead of calling yfinance at runtime.
"""
import json
from src.trading_ensemble.data.comet_screener import get_top_stocks
from src.trading_ensemble.data.dual_source import fetch_yf_sentiment, fetch_yf_fundamentals

CACHE_FILE = "data/pre_market_cache.json"

def main():
    stocks = get_top_stocks()
    cache  = {}
    total  = len(stocks)
    for i, (_, row) in enumerate(stocks.iterrows(), 1):
        symbol = row["symbol"].replace(".NS", "")
        print(f"  [{i}/{total}] Fetching {symbol}...")
        cache[symbol] = {
            **fetch_yf_sentiment(symbol),
            **fetch_yf_fundamentals(symbol),
        }
    import os; os.makedirs("data", exist_ok=True)
    with open(CACHE_FILE, "w") as f:
        json.dump(cache, f, indent=2)
    print(f"Cache saved -> {CACHE_FILE}")

if __name__ == "__main__":
    main()
