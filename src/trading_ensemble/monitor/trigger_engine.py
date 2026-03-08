from __future__ import annotations

from datetime import datetime
from typing import Any

import pandas as pd
import yfinance as yf


def to_yf_ticker(symbol: str) -> str:
    return symbol.replace("-EQ", "").strip() + ".NS"


def mode_to_market_data(mode: str) -> dict[str, str]:
    mode = str(mode).upper()
    if mode == "INTRADAY":
        return {"period": "2d", "interval": "15m"}
    if mode == "SWING":
        return {"period": "3mo", "interval": "1d"}
    if mode == "POSITIONAL":
        return {"period": "6mo", "interval": "1d"}
    return {"period": "2d", "interval": "15m"}


def load_setup_signals(store) -> pd.DataFrame:
    query = """
        SELECT
            s.signal_id,
            s.symbol,
            s.strategy_mode AS MODE,
            s.entry_price AS ENTRY_PRICE,
            s.stop_loss AS STOP_LOSS,
            s.target_price AS TARGET_PRICE,
            s.rr_ratio AS RR_RATIO,
            s.atr AS ATR,
            s.adx AS ADX,
            s.volume_ratio AS VOLUME_RATIO,
            s.signal_strength AS SIGNAL_STRENGTH,
            s.product_type AS PRODUCT_TYPE,
            s.signal_time AS SIGNAL_TIME,
            r.run_id,
            r.started_at
        FROM signals s
        LEFT JOIN runs r
            ON s.run_id = r.run_id
        WHERE s.signal_strength = 'SETUP'
        ORDER BY s.signal_id DESC
    """
    with store.connect() as conn:
        df = pd.read_sql_query(query, conn)

    if df.empty:
        return df

    # backfill from CSV-style fields if they exist in future migrations
    for col in [
        "DONCHIAN_UPPER",
        "BREAKOUT_READY",
        "VOLUME_READY",
        "ATR_READY",
        "SETUP_REASON",
        "WATCHLIST_CONVICTION",
        "SECTOR",
        "PRIORITY",
    ]:
        if col not in df.columns:
            df[col] = None

    df["MODE"] = df["MODE"].astype(str).str.upper()
    return df


def fetch_latest_market_snapshot(symbol: str, mode: str) -> dict[str, Any] | None:
    cfg = mode_to_market_data(mode)
    ticker = to_yf_ticker(symbol)

    try:
        df = yf.download(
            ticker,
            period=cfg["period"],
            interval=cfg["interval"],
            progress=False,
            auto_adjust=True,
        )
        if df is None or df.empty:
            return None

        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)

        df.columns = [str(c).strip().lower() for c in df.columns]
        df = df[["open", "high", "low", "close", "volume"]].copy().dropna()
        if df.empty:
            return None

        latest = df.iloc[-1]

        # rolling levels from live snapshot series
        donchian_period = 20 if str(mode).upper() in {"INTRADAY", "SWING"} else 55
        if len(df) >= donchian_period:
            live_donchian_upper = float(df["high"].rolling(donchian_period).max().iloc[-2])
            avg_volume = float(df["volume"].iloc[-donchian_period - 1 : -1].mean()) if len(df) >= donchian_period + 1 else float(df["volume"].mean())
        else:
            live_donchian_upper = float(df["high"].max())
            avg_volume = float(df["volume"].mean())

        return {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "close": float(latest["close"]),
            "high": float(latest["high"]),
            "low": float(latest["low"]),
            "volume": float(latest["volume"]),
            "avg_volume": avg_volume,
            "live_donchian_upper": live_donchian_upper,
            "rows": len(df),
        }
    except Exception:
        return None


def evaluate_setup_for_promotion(row: pd.Series, snapshot: dict[str, Any] | None) -> dict[str, Any]:
    symbol = str(row["symbol"])
    mode = str(row["MODE"]).upper()

    if snapshot is None:
        return {
            "symbol": symbol,
            "MODE": mode,
            "promotion_candidate": False,
            "promotion_reason": "NO_MARKET_DATA",
        }

    latest_close = float(snapshot["close"])
    latest_volume = float(snapshot["volume"])
    avg_volume = float(snapshot["avg_volume"]) if snapshot["avg_volume"] > 0 else 0.0
    live_volume_ratio = latest_volume / avg_volume if avg_volume > 0 else 0.0

    # use live-computed Donchian
    donchian_upper = float(snapshot["live_donchian_upper"])
    breakout_ready = latest_close >= donchian_upper

    volume_threshold = 2.0 if mode == "INTRADAY" else 1.5 if mode == "SWING" else 1.3
    volume_ready = live_volume_ratio >= volume_threshold

    atr_ready = True  # Phase 20B keeps ATR permissive; tighten in 20C if needed

    promotion_candidate = breakout_ready and volume_ready and atr_ready

    if not breakout_ready:
        reason = "WAITING_BREAKOUT"
    elif not volume_ready:
        reason = "WAITING_VOLUME"
    elif not atr_ready:
        reason = "WAITING_ATR"
    else:
        reason = "PROMOTED"

    return {
        "symbol": symbol,
        "MODE": mode,
        "promotion_candidate": bool(promotion_candidate),
        "promotion_reason": reason,
        "latest_close": round(latest_close, 4),
        "latest_volume": round(latest_volume, 2),
        "live_volume_ratio": round(live_volume_ratio, 4),
        "live_donchian_upper": round(donchian_upper, 4),
        "snapshot_time": snapshot["timestamp"],
        "signal_id": row.get("signal_id"),
        "ENTRY_PRICE": row.get("ENTRY_PRICE"),
        "STOP_LOSS": row.get("STOP_LOSS"),
        "TARGET_PRICE": row.get("TARGET_PRICE"),
        "RR_RATIO": row.get("RR_RATIO"),
        "PRODUCT_TYPE": row.get("PRODUCT_TYPE"),
        "SIGNAL_TIME": row.get("SIGNAL_TIME"),
    }


def dedupe_setup_rows(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df

    work = df.copy()
    sort_cols = [col for col in ["signal_id", "SIGNAL_TIME", "started_at"] if col in work.columns]
    if sort_cols:
        work = work.sort_values(sort_cols, ascending=False)

    work = work.drop_duplicates(subset=["symbol", "MODE"], keep="first")
    return work.reset_index(drop=True)