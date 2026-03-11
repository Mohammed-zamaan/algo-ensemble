# positional_layer_9_backtest.py
# Backtests the positional system over the past 12 months on DAILY bars

import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
from src.trading_ensemble.data.comet_screener import get_top_stocks

# Strategy params (must match positional layers)
DONCHIAN_PERIOD = 55
VOLUME_MULTIPLIER = 1.3
ATR_PERIOD = 21
ATR_SL_MULT = 3.0
ATR_TARGET_MULT = 8.0
MIN_RR = 2.5
RISK_PER_TRADE = 0.03
CAPITAL = 500000
LOOKBACK_MONTHS = 12


def compute_atr(df, period=21):
    high, low, close = df["High"], df["Low"], df["Close"]
    tr = pd.concat([
        high - low,
        (high - close.shift(1)).abs(),
        (low - close.shift(1)).abs(),
    ], axis=1).max(axis=1)
    return tr.rolling(period).mean()


def backtest_symbol(symbol, df):
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    if len(df) < DONCHIAN_PERIOD + ATR_PERIOD + 5:
        return []

    df = df.copy()
    df["atr"] = compute_atr(df, ATR_PERIOD)
    df["don_high"] = df["High"].shift(1).rolling(DONCHIAN_PERIOD).max()
    df["avg_volume"] = df["Volume"].shift(1).rolling(50).mean()
    df["week52_high"] = df["High"].rolling(252, min_periods=50).max()
    df.dropna(inplace=True)

    trades = []
    active_trade = None

    for i in range(len(df)):
        row = df.iloc[i]

        if active_trade:
            if row["Low"] <= active_trade["stop_loss"]:
                pnl = (active_trade["stop_loss"] - active_trade["entry"]) * active_trade["qty"]
                trades.append({**active_trade, "exit": active_trade["stop_loss"], "result": "SL", "pnl": round(pnl, 2), "exit_date": df.index[i]})
                active_trade = None
            elif row["High"] >= active_trade["target"]:
                pnl = (active_trade["target"] - active_trade["entry"]) * active_trade["qty"]
                trades.append({**active_trade, "exit": active_trade["target"], "result": "TARGET", "pnl": round(pnl, 2), "exit_date": df.index[i]})
                active_trade = None

        if active_trade is None:
            breakout = row["Close"] > row["don_high"]
            volume_ok = row["Volume"] > row["avg_volume"] * VOLUME_MULTIPLIER
            near_52w = row["Close"] >= row["week52_high"] * 0.95
            if breakout and volume_ok and near_52w:
                entry = row["Close"]
                sl = entry - ATR_SL_MULT * row["atr"]
                target = entry + ATR_TARGET_MULT * row["atr"]
                rr = (target - entry) / (entry - sl) if (entry - sl) > 0 else 0
                if rr >= MIN_RR:
                    risk_amt = CAPITAL * RISK_PER_TRADE
                    qty = max(1, int(risk_amt / (entry - sl)))
                    active_trade = {
                        "symbol": symbol,
                        "entry": entry,
                        "stop_loss": sl,
                        "target": target,
                        "qty": qty,
                        "entry_date": df.index[i],
                        "rr": round(rr, 2),
                    }

    return trades


def run_backtest():
    end = datetime.today()
    start = end - timedelta(days=LOOKBACK_MONTHS * 30)

    stocks = get_top_stocks()
    print(f"Backtesting positional system on {len(stocks)} stocks | {start.date()} to {end.date()}")

    all_trades = []
    for _, row in stocks.iterrows():
        symbol = row["symbol"]
        try:
            df = yf.download(symbol, start=start, end=end, interval="1d", progress=False)
            if df.empty:
                continue
            all_trades.extend(backtest_symbol(symbol, df))
        except Exception as exc:
            print(f"Error on {symbol}: {exc}")

    return pd.DataFrame(all_trades)


if __name__ == "__main__":
    results = run_backtest()
    print(results.tail(20))
