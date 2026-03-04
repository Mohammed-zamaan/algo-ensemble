"""
Complete ensemble pipeline with conviction-based sizing.
Reads watchlist from Google Sheets → Fetches data → Runs strategy.
"""
import os
from pathlib import Path
import pandas as pd
from dotenv import load_dotenv

from src.trading_ensemble.data.sheets_client_oauth import read_master_universe
from src.trading_ensemble.data.dual_source import fetch_candles, fetch_yf_sentiment
from src.trading_ensemble.core.engine import build_feature_pipeline, backtest_long_only
from src.trading_ensemble.core.params import StrategyParams

load_dotenv()

SPREADSHEET_ID = "1mH4ZzZegQy4_RV0p0JFfOpOq0sD8g5tlquvzi0l4718"  # Your sheet ID

def main():
    print("=== Algo Ensemble with Conviction System ===\n")
    
    # 1. Read watchlist from Google Sheets
    print("📊 Reading watchlist from Google Sheets...")
    universe = read_master_universe(SPREADSHEET_ID)
    print(f"✅ Loaded {len(universe)} active stocks\n")
    
    # 2. Process each stock
    results = []
    for idx, row in universe.iterrows():
        symbol = row['symbol']
        conviction = int(row['conviction'])
        
        print(f"[{idx+1}/{len(universe)}] {symbol} (conviction={conviction})...")
        
        # Fetch OHLCV data
        df = fetch_candles(symbol, symbol_token=None)
        if df is None or df.empty:
            print(f"  ❌ No data\n")
            continue
        
        # Run strategy with conviction
        p = StrategyParams()
        bars = build_feature_pipeline(df, p)
        
        # Custom backtest with conviction (modify backtest_long_only to accept conviction)
        # For now, using base backtest
        result = backtest_long_only(bars, p)
        result['symbol'] = symbol
        result['conviction'] = conviction
        results.append(result)
        
        print(f"  ✅ Equity: ₹{result['final_equity']:.2f} | Trades: {len(result['trades'])}\n")
    
    # 3. Aggregate results
    summary = pd.DataFrame([{
        'symbol': r['symbol'],
        'conviction': r['conviction'],
        'final_equity': r['final_equity'],
        'net_profit': r['net_profit'],
        'num_trades': len(r['trades'])
    } for r in results])
    
    # Save results
    out_path = Path("results/conviction_ensemble.csv")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(out_path, index=False)
    
    print(f"\n🎯 Results saved to {out_path}")
    print(f"Total P&L: ₹{summary['net_profit'].sum():.2f}")

if __name__ == "__main__":
    main()
