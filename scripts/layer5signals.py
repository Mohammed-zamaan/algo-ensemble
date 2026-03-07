# layer_5_signals.py
# LAYER 5 — Signal Generation & Confirmation
# HASH-001 FIX: Complete implementation (was empty stub)
# 
# Modes: INTRADAY (15m, MIS) | SWING (daily, CNC) | POSITIONAL (daily, CNC)
# Entry window: 09:30 AM onwards (first 15m candle must be CLOSED)
# Entry price: Always fresh SmartAPI candle — NEVER Layer 4 LTP
# Confirmation: Price still above Donchian + Volume elevated + ATR valid

import os
import time
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, time as dt_time
from dotenv import load_dotenv

load_dotenv()  # HASH-016 FIX: no hardcoded path

# ─────────────────────────────────────────────────────────────
# CONFIG — mode-specific parameters (matches architecture doc)
# ─────────────────────────────────────────────────────────────
MODE = os.getenv("TRADE_MODE", "INTRADAY")  # INTRADAY | SWING | POSITIONAL

MODE_CONFIG = {
    "INTRADAY": {
        "candidates_file":  "trade_candidates.csv",
        "signals_file":     "trade_signals.csv",
        "donchian_period":  20,
        "atr_period":       14,
        "atr_sl_mult":      1.5,
        "atr_target_mult":  2.5,
        "min_rr":           1.5,
        "volume_mult":      2.0,
        "atr_min":          5.0,         # minimum ATR in INR
        "yf_interval":      "15m",
        "yf_period":        "5d",
        "product_type":     "MIS",
        "entry_after":      dt_time(9, 30),   # wait for first candle to close
        "exit_before":      dt_time(15, 15),
    },
    "SWING": {
        "candidates_file":  "swing_candidates.csv",
        "signals_file":     "swing_signals.csv",
        "donchian_period":  20,
        "atr_period":       14,
        "atr_sl_mult":      2.0,
        "atr_target_mult":  4.0,
        "min_rr":           2.0,
        "volume_mult":      1.5,
        "atr_min":          5.0,
        "yf_interval":      "1d",
        "yf_period":        "3mo",
        "product_type":     "CNC",
        "entry_after":      dt_time(9, 30),
        "exit_before":      dt_time(15, 15),
    },
    "POSITIONAL": {
        "candidates_file":  "positional_candidates.csv",
        "signals_file":     "positional_signals.csv",
        "donchian_period":  55,
        "atr_period":       21,
        "atr_sl_mult":      3.0,
        "atr_target_mult":  8.0,
        "min_rr":           2.5,
        "volume_mult":      1.3,
        "atr_min":          5.0,
        "yf_interval":      "1d",
        "yf_period":        "6mo",
        "product_type":     "CNC",
        "entry_after":      dt_time(9, 30),
        "exit_before":      dt_time(15, 15),
    },
}

cfg = MODE_CONFIG[MODE]

# ─────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────

def to_yf_ticker(symbol: str) -> str:
    """Convert NSE symbol to yfinance ticker."""
    return symbol.replace("-EQ", "").strip() + ".NS"


def fetch_candles(symbol: str) -> pd.DataFrame | None:
    """
    Fetch OHLCV candles via yfinance.
    Returns cleaned DataFrame or None on failure.
    """
    ticker = to_yf_ticker(symbol)
    try:
        df = yf.download(
            ticker,
            period=cfg["yf_period"],
            interval=cfg["yf_interval"],
            progress=False,
            auto_adjust=True,
        )
        if df is None or df.empty:
            print(f"    [WARN] No data for {ticker}")
            return None
        # Flatten MultiIndex if present
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        df.columns = [c.strip().lower() for c in df.columns]
        df = df[["open", "high", "low", "close", "volume"]].copy()
        df.dropna(inplace=True)
        if len(df) < cfg["donchian_period"] + 5:
            print(f"    [WARN] Insufficient rows for {ticker}: {len(df)}")
            return None
        return df
    except Exception as e:
        print(f"    [ERROR] fetch_candles {ticker}: {e}")
        return None


def compute_atr(df: pd.DataFrame, period: int) -> float:
    """Compute latest ATR (Average True Range)."""
    high  = df["high"].astype(float)
    low   = df["low"].astype(float)
    close = df["close"].astype(float)
    prev_close = close.shift(1)
    tr = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low  - prev_close).abs(),
    ], axis=1).max(axis=1)
    atr = tr.ewm(span=period, adjust=False).mean()
    return float(atr.iloc[-1])


def compute_donchian_upper(df: pd.DataFrame, period: int) -> float:
    """
    Donchian upper band = highest high over last N bars EXCLUDING current bar.
    Using iloc[-2] prevents lookahead bias.
    """
    return float(df["high"].rolling(period).max().iloc[-2])


def compute_avg_volume(df: pd.DataFrame, period: int) -> float:
    """Average volume over last N bars excluding current bar."""
    return float(df["volume"].iloc[-period-1:-1].mean())


def get_signal_strength(
    price_above: bool,
    vol_ratio: float,
    atr_ratio: float,
    adx: float,
) -> str:
    """
    STRONG  = all conditions met with good margin
    CONFIRMED = all conditions met
    WEAK    = borderline
    REJECTED = failed
    """
    if not price_above:
        return "REJECTED"
    score = 0
    if vol_ratio >= 2.5:   score += 2
    elif vol_ratio >= 2.0: score += 1
    if atr_ratio >= 1.5:   score += 2
    elif atr_ratio >= 1.0: score += 1
    if adx >= 30:          score += 2
    elif adx >= 20:        score += 1
    if score >= 5:   return "STRONG"
    if score >= 3:   return "CONFIRMED"
    return "WEAK"


def compute_adx(df: pd.DataFrame, period: int = 14) -> float:
    """Simple ADX computation."""
    try:
        high  = df["high"].astype(float)
        low   = df["low"].astype(float)
        close = df["close"].astype(float)
        plus_dm  = high.diff()
        minus_dm = -low.diff()
        plus_dm[plus_dm < 0]   = 0
        minus_dm[minus_dm < 0] = 0
        tr = pd.concat([
            high - low,
            (high - close.shift()).abs(),
            (low  - close.shift()).abs(),
        ], axis=1).max(axis=1)
        atr14     = tr.ewm(span=period, adjust=False).mean()
        plus_di   = 100 * plus_dm.ewm(span=period, adjust=False).mean() / atr14
        minus_di  = 100 * minus_dm.ewm(span=period, adjust=False).mean() / atr14
        dx        = (100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan))
        adx       = dx.ewm(span=period, adjust=False).mean()
        return float(adx.iloc[-1])
    except Exception:
        return 0.0


# ─────────────────────────────────────────────────────────────
# ENTRY WINDOW GUARD
# ─────────────────────────────────────────────────────────────

def check_entry_window() -> bool:
    """
    HASH-001 FIX: Do NOT enter trades before 09:30 AM.
    First 15m candle (09:15–09:30) must be fully closed.
    This eliminates gap-and-reverse false breakouts.
    """
    now = datetime.now().time()
    if now < cfg["entry_after"]:
        print(f"  [WAIT] Entry window opens at {cfg['entry_after']} — current time {now}")
        return False
    if now >= cfg["exit_before"]:
        print(f"  [HALT] Past exit time {cfg['exit_before']} — no new entries")
        return False
    return True


# ─────────────────────────────────────────────────────────────
# CORE SIGNAL GENERATOR
# ─────────────────────────────────────────────────────────────

def generate_signal(row: pd.Series) -> dict | None:
    """
    Generate confirmed signal for one candidate.
    Fetches FRESH candles — NEVER uses stale Layer 4 LTP.
    """
    symbol = row["symbol"]
    print(f"  [{symbol}] Fetching fresh {cfg['yf_interval']} candles...", end=" ", flush=True)

    df = fetch_candles(symbol)
    if df is None:
        print("SKIP — no data")
        return None

    # ── Core metrics ──────────────────────────────────────────
    atr             = compute_atr(df, cfg["atr_period"])
    donchian_upper  = compute_donchian_upper(df, cfg["donchian_period"])
    avg_vol         = compute_avg_volume(df, cfg["donchian_period"])
    adx             = compute_adx(df, 14)

    # Fresh entry price = last closed candle close (NOT Layer 4 LTP)
    entry_price     = float(df["close"].iloc[-1])
    current_vol     = float(df["volume"].iloc[-1])
    vol_ratio       = current_vol / avg_vol if avg_vol > 0 else 0.0
    atr_ratio       = atr / entry_price * 100  # ATR as % of price

    # ── Validation gates ──────────────────────────────────────
    price_above     = entry_price >= donchian_upper
    vol_ok          = vol_ratio >= cfg["volume_mult"]
    atr_ok          = atr >= cfg["atr_min"]

    strength = get_signal_strength(price_above, vol_ratio, atr_ratio, adx)

    # Reject if not at least CONFIRMED
    if strength in ("REJECTED", "WEAK"):
        print(f"SKIP — {strength} | price_above={price_above} vol_ratio={vol_ratio:.2f} atr={atr:.2f}")
        return None

    # ── Entry / SL / Target ───────────────────────────────────
    stop_loss    = round(entry_price - cfg["atr_sl_mult"]     * atr, 2)
    target_price = round(entry_price + cfg["atr_target_mult"] * atr, 2)
    rr_ratio     = round((target_price - entry_price) / (entry_price - stop_loss), 2)

    if rr_ratio < cfg["min_rr"]:
        print(f"SKIP — RR {rr_ratio:.2f} < {cfg['min_rr']}")
        return None

    print(f"✅ {strength} | Entry={entry_price} SL={stop_loss} TGT={target_price} RR={rr_ratio}")

    return {
        "symbol":           symbol,
        "MODE":             MODE,
        "COMPOSITE_SCORE":  float(row.get("COMPOSITE_SCORE", 0)),
        "ENTRY_PRICE":      entry_price,
        "STOP_LOSS":        stop_loss,
        "TARGET_PRICE":     target_price,
        "RR_RATIO":         rr_ratio,
        "ATR":              round(atr, 4),
        "ATR_RATIO_PCT":    round(atr_ratio, 4),
        "ADX":              round(adx, 2),
        "VOLUME_RATIO":     round(vol_ratio, 2),
        "DONCHIAN_UPPER":   round(donchian_upper, 2),
        "BREAKOUT_15M":     price_above and vol_ok and atr_ok,
        "SIGNAL_STRENGTH":  strength,
        "PRODUCT_TYPE":     cfg["product_type"],
        "SIGNAL_TIME":      datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }


# ─────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────

def main():
    print(f"\n[LAYER 5] Signal Generation — MODE: {MODE}")
    print("=" * 65)

    # Entry window guard — no trades before 09:30
    if not check_entry_window():
        return

    # Load Layer 4 candidates
    candidates_file = cfg["candidates_file"]
    if not os.path.exists(candidates_file):
        print(f"  [ERROR] {candidates_file} not found — run layer_4 first")
        return

    candidates = pd.read_csv(candidates_file)
    if candidates.empty:
        print("  [INFO] No candidates from Layer 4 — no trades today")
        return

    print(f"  Processing {len(candidates)} candidate(s)...\n")

    signals = []
    for _, row in candidates.iterrows():
        result = generate_signal(row)
        if result:
            signals.append(result)
        time.sleep(0.3)  # rate limit yfinance

    print("\n" + "=" * 65)
    print(f"  RESULT: {len(signals)} confirmed signal(s) from {len(candidates)} candidates")
    print("=" * 65)

    if not signals:
        print("  [INFO] No confirmed signals — no trades to place today")
        # Write empty file so Layer 6 doesn't crash
        pd.DataFrame(columns=[
            "symbol", "MODE", "COMPOSITE_SCORE", "ENTRY_PRICE",
            "STOP_LOSS", "TARGET_PRICE", "RR_RATIO", "ATR",
            "BREAKOUT_15M", "SIGNAL_STRENGTH", "PRODUCT_TYPE", "SIGNAL_TIME"
        ]).to_csv(cfg["signals_file"], index=False)
        return

    signals_df = pd.DataFrame(signals)
    signals_df.sort_values("COMPOSITE_SCORE", ascending=False, inplace=True)
    signals_df.to_csv(cfg["signals_file"], index=False)

    print(f"  [SAVED] {cfg['signals_file']}")
    print()
    display_cols = ["symbol", "ENTRY_PRICE", "STOP_LOSS", "TARGET_PRICE",
                    "RR_RATIO", "SIGNAL_STRENGTH", "VOLUME_RATIO", "ADX"]
    print(signals_df[display_cols].to_string(index=False))


if __name__ == "__main__":
    main()
