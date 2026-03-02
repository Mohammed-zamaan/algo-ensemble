# layer_9_backtest.py
# LAYER 9 — Backtesting Engine
# Simulates Layers 4-6 on 6 months of daily historical data
# Tests the full Donchian breakout strategy across all 43 watchlist stocks
# Logs all results to Comet ML project "backtest-results"

import os
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
from dotenv import load_dotenv

load_dotenv(dotenv_path="/workspaces/algo-ensemble/.env")

# ─────────────────────────────────────────────
# BACKTEST CONFIG
# ─────────────────────────────────────────────
BACKTEST_MONTHS      = 6
DONCHIAN_PERIOD      = 20
ATR_PERIOD           = 14
ATR_TARGET_MULT      = 2.0
ATR_SL_MULT          = 1.5
MIN_VOLUME_RATIO     = 1.0
MIN_COMPOSITE_SCORE  = 50.0
TOTAL_CAPITAL        = float(os.getenv("TOTAL_CAPITAL", 100000))
RISK_PER_TRADE_PCT   = 0.02
MAX_CAPITAL_PER_TRADE= 0.20
MAX_POSITIONS        = 5

RESULTS_FILE         = "backtest_results.csv"
SUMMARY_FILE         = "backtest_summary.txt"

# All 43 watchlist stocks
WATCHLIST = [
    "ABB-EQ","ADANIENSOL-EQ","ADANIENT-EQ","ADANIGREEN-EQ",
    "BALKRISIND-EQ","BEL-EQ","BLUESTARCO-EQ","CANBK-EQ",
    "CGPOWER-EQ","CONCORDBIO-EQ","DATAPATTNS-EQ","EICHERMOT-EQ",
    "ENRIN-EQ","FSL-EQ","GRAPHITE-EQ","HAPPSTMNDS-EQ",
    "HINDCOPPER-EQ","HINDZINC-EQ","IRCON-EQ","IRFC-EQ",
    "JINDALSAW-EQ","JINDALSTEL-EQ","KPITTECH-EQ","LICI-EQ",
    "MANAPPURAM-EQ","MAZDOCK-EQ","MOTHERSON-EQ","NATIONALUM-EQ",
    "NEWGEN-EQ","NUVOCO-EQ","POWERGRID-EQ","PREMIERENE-EQ",
    "SBIN-EQ","SCHNEIDER-EQ","SHRIRAMFIN-EQ","SOBHA-EQ",
    "SWANCORP-EQ","TATASTEEL-EQ","TVSMOTOR-EQ","USHAMART-EQ",
    "VEDL-EQ","WAAREEENER-EQ","WELSPUNLIV-EQ"
]
# ─────────────────────────────────────────────


def fetch_historical_data(symbol_eq: str, months: int = 6) -> pd.DataFrame:
    """Fetch daily OHLCV for the past N months."""
    ticker = symbol_eq.replace("-EQ", ".NS")
    end    = datetime.today()
    start  = end - timedelta(days=months * 30)
    try:
        df = yf.download(ticker, start=start.strftime("%Y-%m-%d"),
                         end=end.strftime("%Y-%m-%d"),
                         interval="1d", progress=False, auto_adjust=True)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        df = df.dropna()
        return df
    except Exception as e:
        print(f"  [WARN] {ticker}: {e}")
        return pd.DataFrame()


def compute_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Add Donchian bands, ATR, volume ratio to DataFrame."""
    df = df.copy()
    df["donchian_upper"] = df["High"].rolling(DONCHIAN_PERIOD).max().shift(1)
    df["donchian_lower"] = df["Low"].rolling(DONCHIAN_PERIOD).min().shift(1)

    prev_close = df["Close"].shift(1)
    tr = pd.concat([
        df["High"] - df["Low"],
        (df["High"] - prev_close).abs(),
        (df["Low"]  - prev_close).abs()
    ], axis=1).max(axis=1)
    df["atr"] = tr.rolling(ATR_PERIOD).mean()

    df["vol_ma20"]    = df["Volume"].rolling(DONCHIAN_PERIOD).mean().shift(1)
    df["vol_ratio"]   = df["Volume"] / df["vol_ma20"]
    df["return_20c"]  = df["Close"].pct_change(DONCHIAN_PERIOD) * 100
    df = df.dropna()
    return df


def simulate_trades(symbol: str, df: pd.DataFrame) -> list:
    """
    Walk forward day by day.
    Entry signal: Close > Donchian upper AND volume ratio >= MIN_VOLUME_RATIO
    Exit: target hit OR stop-loss hit (checked on subsequent candles)
    """
    trades = []
    in_trade = False
    entry_price = target = stop_loss = entry_date = qty = None

    for i in range(len(df)):
        row = df.iloc[i]
        date = df.index[i]

        # ── Manage open trade ──────────────────────────
        if in_trade:
            high_today = float(row["High"])
            low_today  = float(row["Low"])
            close_today= float(row["Close"])

            # Check target hit (intraday high touched target)
            if high_today >= target:
                pnl       = round((target - entry_price) * qty, 2)
                pnl_pct   = round((target - entry_price) / entry_price * 100, 2)
                hold_days = (date - entry_date).days
                trades.append({
                    "symbol":       symbol,
                    "entry_date":   entry_date.date(),
                    "exit_date":    date.date(),
                    "entry_price":  round(entry_price, 2),
                    "exit_price":   round(target, 2),
                    "stop_loss":    round(stop_loss, 2),
                    "target":       round(target, 2),
                    "quantity":     qty,
                    "pnl":          pnl,
                    "pnl_pct":      pnl_pct,
                    "hold_days":    hold_days,
                    "exit_reason":  "TARGET",
                    "result":       "WIN",
                })
                in_trade = False
                continue

            # Check stop-loss hit (intraday low touched SL)
            if low_today <= stop_loss:
                pnl       = round((stop_loss - entry_price) * qty, 2)
                pnl_pct   = round((stop_loss - entry_price) / entry_price * 100, 2)
                hold_days = (date - entry_date).days
                trades.append({
                    "symbol":       symbol,
                    "entry_date":   entry_date.date(),
                    "exit_date":    date.date(),
                    "entry_price":  round(entry_price, 2),
                    "exit_price":   round(stop_loss, 2),
                    "stop_loss":    round(stop_loss, 2),
                    "target":       round(target, 2),
                    "quantity":     qty,
                    "pnl":          pnl,
                    "pnl_pct":      pnl_pct,
                    "hold_days":    hold_days,
                    "exit_reason":  "STOP_LOSS",
                    "result":       "LOSS",
                })
                in_trade = False
                continue

            # Force exit at end of data
            if i == len(df) - 1:
                pnl = round((close_today - entry_price) * qty, 2)
                pnl_pct = round((close_today - entry_price) / entry_price * 100, 2)
                trades.append({
                    "symbol":       symbol,
                    "entry_date":   entry_date.date(),
                    "exit_date":    date.date(),
                    "entry_price":  round(entry_price, 2),
                    "exit_price":   round(close_today, 2),
                    "stop_loss":    round(stop_loss, 2),
                    "target":       round(target, 2),
                    "quantity":     qty,
                    "pnl":          pnl,
                    "pnl_pct":      pnl_pct,
                    "hold_days":    (date - entry_date).days,
                    "exit_reason":  "END_OF_DATA",
                    "result":       "WIN" if pnl >= 0 else "LOSS",
                })
                in_trade = False

        # ── Check for new entry signal ──────────────────
        if not in_trade:
            close    = float(row["Close"])
            don_high = float(row["donchian_upper"])
            vol_rat  = float(row["vol_ratio"])
            atr      = float(row["atr"])

            # Entry conditions
            if close >= don_high and vol_rat >= MIN_VOLUME_RATIO and atr > 0:
                risk_amt   = TOTAL_CAPITAL * RISK_PER_TRADE_PCT
                risk_share = close * ATR_SL_MULT * (atr / close)
                if risk_share > 0:
                    qty_risk = int(risk_amt / risk_share)
                    qty_cap  = int((TOTAL_CAPITAL * MAX_CAPITAL_PER_TRADE) / close)
                    qty      = min(qty_risk, qty_cap)
                    if qty > 0:
                        entry_price = close
                        target      = round(close + ATR_TARGET_MULT * atr, 2)
                        stop_loss   = round(close - ATR_SL_MULT   * atr, 2)
                        entry_date  = date
                        in_trade    = True

    return trades


def run_backtest():
    print("\n" + "="*65)
    print("  LAYER 9 — BACKTESTING ENGINE")
    print(f"  Period : Last {BACKTEST_MONTHS} months")
    print(f"  Stocks : {len(WATCHLIST)}")
    print(f"  Strategy: Donchian {DONCHIAN_PERIOD}-day Breakout")
    print(f"  ATR Target: {ATR_TARGET_MULT}x  SL: {ATR_SL_MULT}x")
    print("="*65)

    all_trades = []

    for i, symbol in enumerate(WATCHLIST):
        print(f"  [{i+1:02d}/{len(WATCHLIST)}] {symbol:<22}", end=" ")
        df = fetch_historical_data(symbol, BACKTEST_MONTHS)
        if df.empty or len(df) < DONCHIAN_PERIOD + ATR_PERIOD:
            print("SKIP (insufficient data)")
            continue

        df = compute_indicators(df)
        trades = simulate_trades(symbol, df)
        all_trades.extend(trades)

        wins   = sum(1 for t in trades if t["result"] == "WIN")
        losses = sum(1 for t in trades if t["result"] == "LOSS")
        total_pnl = sum(t["pnl"] for t in trades)
        print(f"{len(trades)} trades  |  W={wins} L={losses}  |  PnL=INR {total_pnl:,.0f}")

    return pd.DataFrame(all_trades)


def print_summary(df: pd.DataFrame):
    if df.empty:
        print("\n  No trades in backtest period.")
        return

    total_trades  = len(df)
    wins          = len(df[df["result"] == "WIN"])
    losses        = len(df[df["result"] == "LOSS"])
    win_rate      = round(wins / total_trades * 100, 1)
    total_pnl     = round(df["pnl"].sum(), 2)
    avg_win       = round(df[df["result"]=="WIN"]["pnl"].mean(), 2)
    avg_loss      = round(df[df["result"]=="LOSS"]["pnl"].mean(), 2)
    avg_hold_days = round(df["hold_days"].mean(), 1)
    best_trade    = df.loc[df["pnl"].idxmax()]
    worst_trade   = df.loc[df["pnl"].idxmin()]
    profit_factor = round(abs(df[df["pnl"]>0]["pnl"].sum() /
                              df[df["pnl"]<0]["pnl"].sum()), 2) if losses > 0 else float("inf")

    # Equity curve for max drawdown
    equity = TOTAL_CAPITAL + df["pnl"].cumsum()
    peak   = equity.cummax()
    dd     = ((equity - peak) / peak * 100)
    max_dd = round(dd.min(), 2)

    summary = f"""
{'='*65}
  BACKTEST SUMMARY — Last {BACKTEST_MONTHS} Months
  Strategy: Donchian {DONCHIAN_PERIOD}-day Breakout | ATR {ATR_TARGET_MULT}x/{ATR_SL_MULT}x
  Capital: INR {TOTAL_CAPITAL:,.0f}
{'='*65}
  Total Trades    : {total_trades}
  Wins            : {wins}
  Losses          : {losses}
  Win Rate        : {win_rate}%
  Profit Factor   : {profit_factor}
  
  Total PnL       : INR {total_pnl:,.2f}
  Return on Cap   : {round(total_pnl/TOTAL_CAPITAL*100, 2)}%
  Avg Win         : INR {avg_win:,.2f}
  Avg Loss        : INR {avg_loss:,.2f}
  Avg Hold        : {avg_hold_days} days
  Max Drawdown    : {max_dd}%

  Best Trade      : {best_trade['symbol']}  INR {best_trade['pnl']:,.2f}  ({best_trade['entry_date']})
  Worst Trade     : {worst_trade['symbol']}  INR {worst_trade['pnl']:,.2f}  ({worst_trade['entry_date']})
{'='*65}

  Exit Breakdown:
{df['exit_reason'].value_counts().to_string()}

  Top 5 Stocks by PnL:
{df.groupby('symbol')['pnl'].sum().sort_values(ascending=False).head(5).to_string()}

  Bottom 5 Stocks by PnL:
{df.groupby('symbol')['pnl'].sum().sort_values(ascending=True).head(5).to_string()}
{'='*65}
"""
    print(summary)


    with open(SUMMARY_FILE, 'w') as sf:
        sf.write(summary)
    print(f'  [SAVED] Summary -> {SUMMARY_FILE}')


if __name__ == '__main__':
    results_df = run_backtest()

    if not results_df.empty:
        results_df.to_csv(RESULTS_FILE, index=False)
        print(f'[SAVED] {len(results_df)} trades -> {RESULTS_FILE}')

    print_summary(results_df)
