# backtest.py — HASH-013: 3-Year Walk-Forward Backtest Engine
# Tests algo-ensemble strategy on NSE historical data 2023-01-01 to 2026-03-06
# Simulates: Donchian breakout + ATR sizing + Regime filter + DD protection

import os, sys, json, warnings
import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
from dotenv import load_dotenv

warnings.filterwarnings("ignore")
load_dotenv()

# ── CONFIG ────────────────────────────────────────────────────────────────────
START_DATE    = "2023-01-01"
END_DATE      = "2026-03-06"
CAPITAL       = float(os.getenv("TOTAL_CAPITAL", 500000))
RISK_PCT      = float(os.getenv("RISK_PCT", 2.0))
MAX_POSITIONS = int(os.getenv("MAX_POSITIONS", 5))
MIN_RR        = 1.5
SL_MULT       = 2.0
TGT_MULT      = 4.0
DONCHIAN_N    = 20
ATR_N         = 14
VIX_CAUTION   = 20.0
VIX_CRISIS    = 30.0

UNIVERSE = [
    "SBIN.NS","HDFCBANK.NS","ICICIBANK.NS","KOTAKBANK.NS","AXISBANK.NS",
    "RELIANCE.NS","TCS.NS","INFY.NS","WIPRO.NS","HCLTECH.NS",
    "TATASTEEL.NS","JSWSTEEL.NS","HINDALCO.NS","COALINDIA.NS",
    "MARUTI.NS","TATAMOTORS.NS","BAJAJ-AUTO.NS","HEROMOTOCO.NS",
    "SUNPHARMA.NS","DRREDDY.NS","CIPLA.NS","DIVISLAB.NS",
    "LT.NS","NTPC.NS","POWERGRID.NS",
    "ASIANPAINT.NS","TITAN.NS","NESTLEIND.NS","HINDUNILVR.NS",
]

# ── HELPERS ───────────────────────────────────────────────────────────────────
def fetch_data(ticker, start, end):
    try:
        df = yf.download(ticker, start=start, end=end, interval="1d",
                         auto_adjust=True, progress=False, show_errors=False)
        if len(df) < 60:
            return None
        df.columns = [c[0] if isinstance(c, tuple) else c for c in df.columns]
        return df
    except Exception:
        return None

def fetch_index(ticker, start, end):
    try:
        df = yf.download(ticker, start=start, end=end, interval="1d",
                         auto_adjust=True, progress=False, show_errors=False)
        if df.empty:
            return pd.Series(dtype=float), pd.DataFrame()
        df.columns = [c[0] if isinstance(c, tuple) else c for c in df.columns]
        return df["Close"], df
    except Exception:
        return pd.Series(dtype=float), pd.DataFrame()

def compute_atr(df, n=14):
    hl = df["High"] - df["Low"]
    hc = (df["High"] - df["Close"].shift()).abs()
    lc = (df["Low"]  - df["Close"].shift()).abs()
    tr = pd.concat([hl, hc, lc], axis=1).max(axis=1)
    return tr.rolling(n).mean()

def compute_donchian(df, n=20):
    return df["High"].shift(1).rolling(n).max(), df["Low"].shift(1).rolling(n).min()

def compute_adx(df, n=14):
    atr     = compute_atr(df, n)
    pdm     = df["High"].diff().clip(lower=0)
    mdm     = (-df["Low"].diff()).clip(lower=0)
    pdi     = 100 * pdm.rolling(n).mean() / (atr + 1e-9)
    mdi     = 100 * mdm.rolling(n).mean() / (atr + 1e-9)
    dx      = (100 * (pdi - mdi).abs() / (pdi + mdi + 1e-9))
    return dx.rolling(n).mean()

def get_regime(nifty_close, vix_series, idx):
    vix = float(vix_series.reindex([idx], method="nearest").iloc[0]) if len(vix_series) else 20.0
    if vix >= VIX_CRISIS:
        return "CRISIS", vix
    ni = nifty_close.loc[:idx]
    if len(ni) < 50:
        return "SIDEWAYS", vix
    ma50 = float(ni.tail(50).mean())
    ma200= float(ni.tail(200).mean()) if len(ni) >= 200 else ma50
    last = float(ni.iloc[-1])
    slp  = (float(ni.tail(5).iloc[-1]) - float(ni.tail(5).iloc[0])) / float(ni.tail(5).iloc[0]) * 100 if len(ni) >= 5 else 0
    if last > ma50 > ma200 and slp > 0:
        return "BULL", vix
    if last < ma50 < ma200 and slp < 0:
        return "BEAR", vix
    return "SIDEWAYS", vix

def regime_mult(r):
    return {"BULL":1.0,"SIDEWAYS":0.5,"BEAR":0.25,"CRISIS":0.0}.get(r, 0.5)

def dd_mult(peak, cur):
    dd = (peak - cur) / peak * 100
    if dd < 5:  return 1.0
    if dd < 10: return 0.5
    if dd < 15: return 0.25
    return 0.0

# ── MAIN BACKTEST ─────────────────────────────────────────────────────────────
def run_backtest():
    print(f"\n{'='*65}")
    print(f"  algo-ensemble Backtest  |  {START_DATE} to {END_DATE}")
    print(f"  Capital: INR {CAPITAL:,.0f}  |  Universe: {len(UNIVERSE)} stocks")
    print(f"{'='*65}\n")

    vix_close,  _        = fetch_index("^INDIAVIX", START_DATE, END_DATE)
    nifty_close, _       = fetch_index("^NSEI",     START_DATE, END_DATE)
    bank_close, _        = fetch_index("^NSEBANK",  START_DATE, END_DATE)
    print(f"  [OK] NIFTY={len(nifty_close)} | BANKNIFTY={len(bank_close)} | VIX={len(vix_close)} days")

    all_data = {}
    for t in UNIVERSE:
        df = fetch_data(t, START_DATE, END_DATE)
        if df is not None:
            all_data[t] = df
    print(f"  [OK] {len(all_data)}/{len(UNIVERSE)} stocks loaded\n")

    dates = sorted(set().union(*[set(df.index) for df in all_data.values()]))
    dates = [d for d in dates if pd.Timestamp(START_DATE) <= d <= pd.Timestamp(END_DATE)]

    equity      = CAPITAL
    peak_eq     = CAPITAL
    positions   = {}
    trades      = []
    equity_curve= []

    for today in dates:
        # ── close positions ──
        to_close = []
        for ticker, pos in positions.items():
            df = all_data.get(ticker)
            if df is None or today not in df.index:
                continue
            row   = df.loc[today]
            lo, hi = float(row["Low"]), float(row["High"])
            if lo <= pos["sl"]:
                pnl = (pos["sl"] - pos["entry"]) * pos["qty"]
                equity += pnl; peak_eq = max(peak_eq, equity)
                trades.append({**pos, "exit":pos["sl"], "pnl":round(pnl,2),
                    "reason":"SL", "date":str(today.date()),
                    "hold_days":(today - pos["entry_date"]).days})
                to_close.append(ticker)
            elif hi >= pos["target"]:
                pnl = (pos["target"] - pos["entry"]) * pos["qty"]
                equity += pnl; peak_eq = max(peak_eq, equity)
                trades.append({**pos, "exit":pos["target"], "pnl":round(pnl,2),
                    "reason":"TARGET", "date":str(today.date()),
                    "hold_days":(today - pos["entry_date"]).days})
                to_close.append(ticker)
        for t in to_close:
            del positions[t]

        equity_curve.append({"date":str(today.date()), "equity": float(round(equity,2))})

        if len(positions) >= MAX_POSITIONS:
            continue

        # ── regime ──
        regime, vix = get_regime(nifty_close, vix_close, today)
        combined = regime_mult(regime) * dd_mult(peak_eq, equity)
        if combined == 0:
            continue

        # ── scan for signals ──
        for ticker, df in all_data.items():
            if ticker in positions or len(positions) >= MAX_POSITIONS:
                continue
            if today not in df.index:
                continue
            idx_pos = df.index.get_loc(today)
            if idx_pos < DONCHIAN_N + ATR_N + 5:
                continue
            slc   = df.iloc[:idx_pos+1]
            upper, _ = compute_donchian(slc, DONCHIAN_N)
            atr   = compute_atr(slc, ATR_N)
            adx   = compute_adx(slc, ATR_N)
            close = float(df.loc[today, "Close"])
            prev_upper = float(upper.iloc[-2]) if len(upper) >= 2 else np.nan
            curr_atr   = float(atr.iloc[-1])
            curr_adx   = float(adx.iloc[-1])
            ma50       = float(slc["Close"].tail(50).mean())
            vol_avg    = float(slc["Volume"].tail(20).mean())
            vol_today  = float(df.loc[today, "Volume"])

            if any(pd.isna(v) for v in [prev_upper, curr_atr, curr_adx]):
                continue
            if not (close > prev_upper and close > ma50 and
                    curr_adx > 20 and vol_today >= 1.5 * vol_avg):
                continue

            sl     = close - SL_MULT * curr_atr
            target = close + TGT_MULT * curr_atr
            rr     = (target - close) / (close - sl) if (close - sl) > 0 else 0
            if rr < MIN_RR:
                continue

            risk_amt = equity * (RISK_PCT / 100) * combined
            qty      = int(risk_amt / (close - sl)) if (close - sl) > 0 else 0
            if qty <= 0:
                continue

            positions[ticker] = {
                "ticker": ticker, "entry": close, "sl": sl, "target": target,
                "qty": qty, "entry_date": today, "regime": regime,
                "vix": round(vix, 2), "rr": round(rr, 2)
            }

    # force-close remaining
    for ticker, pos in positions.items():
        df = all_data.get(ticker)
        if df is None:
            continue
        ep  = float(df["Close"].iloc[-1])
        pnl = (ep - pos["entry"]) * pos["qty"]
        equity += pnl
        trades.append({**pos, "exit": ep, "pnl": round(pnl,2),
            "reason":"EOD", "date": str(df.index[-1].date()),
            "hold_days": (df.index[-1] - pos["entry_date"]).days})

    return trades, equity_curve, equity

# ── METRICS ───────────────────────────────────────────────────────────────────
def compute_metrics(trades, equity_curve, final_equity):
    df = pd.DataFrame(trades)
    eq = pd.DataFrame(equity_curve)
    eq["equity"] = pd.to_numeric(eq["equity"])
    if len(df) == 0:
        return {}

    wins   = df[df["pnl"] > 0]
    losses = df[df["pnl"] <= 0]
    pf     = wins["pnl"].sum() / abs(losses["pnl"].sum()) if losses["pnl"].sum() != 0 else 999
    eq_vals= eq["equity"].values
    peak   = np.maximum.accumulate(eq_vals)
    dd     = (peak - eq_vals) / peak * 100
    daily_r= eq["equity"].pct_change().dropna()
    sharpe = float(daily_r.mean() / daily_r.std() * np.sqrt(252)) if daily_r.std() > 0 else 0
    df["month"] = pd.to_datetime(df["date"]).dt.to_period("M")
    monthly = df.groupby("month")["pnl"].sum()

    return {
        "Total Trades":    len(df),
        "Win Rate %":      round(len(wins)/len(df)*100, 1),
        "Avg Win INR":     round(float(wins["pnl"].mean()), 0) if len(wins) else 0,
        "Avg Loss INR":    round(float(losses["pnl"].mean()), 0) if len(losses) else 0,
        "Profit Factor":   round(pf, 2),
        "Total PnL INR":   round(float(df["pnl"].sum()), 0),
        "Total Return %":  round((final_equity - CAPITAL) / CAPITAL * 100, 2),
        "Max Drawdown %":  round(float(dd.max()), 2),
        "Sharpe Ratio":    round(sharpe, 2),
        "Avg Hold Days":   round(float(df["hold_days"].mean()), 1),
        "Positive Months": f"{(monthly > 0).sum()}/{len(monthly)}",
        "Final Capital INR": round(final_equity, 0),
    }

if __name__ == "__main__":
    trades, equity_curve, final_equity = run_backtest()
    metrics = compute_metrics(trades, equity_curve, final_equity)

    print(f"\n{'='*55}")
    print("  BACKTEST RESULTS")
    print(f"{'='*55}")
    for k, v in metrics.items():
        print(f"  {k:<24}: {v}")
    print(f"{'='*55}\n")

    os.makedirs("state", exist_ok=True)
    pd.DataFrame(trades).to_csv("state/backtest_trades.csv", index=False)
    pd.DataFrame(equity_curve).to_csv("state/backtest_equity.csv", index=False)
    with open("state/backtest_metrics.json","w") as f:
        json.dump(metrics, f, indent=2)
    print("  Saved: state/backtest_trades.csv")
    print("  Saved: state/backtest_equity.csv")
    print("  Saved: state/backtest_metrics.json")
    print("  Run  : python backtest_chart.py  to generate equity curve PNG")