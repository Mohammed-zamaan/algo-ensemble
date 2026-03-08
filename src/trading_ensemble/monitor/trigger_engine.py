from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any

import pandas as pd
import yfinance as yf


@dataclass(frozen=True)
class SetupSignal:
    symbol: str
    mode: str
    signal_time: str
    entry_price: float
    stop_loss: float
    target_price: float
    rr_ratio: float
    donchian_upper: float | None
    volume_ratio: float | None
    breakout_ready: bool
    volume_ready: bool
    atr_ready: bool
    setup_reason: str
    watchlist_conviction: float
    sector: str
    priority: int


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
    """
    Read current SETUP signals from SQLite.
    """
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

    df["MODE"] = df["MODE"].astype(str).str.upper()
    return df


def load_setup_signals_from_csv(path: str) -> pd.DataFrame:
    """
    Fallback helper if you want to inspect the signals CSV instead of SQLite.
    """
    df = pd.read_csv(path)
    if df.empty:
        return df
    if "SIGNAL_STATUS" not in df.columns:
        return pd.DataFrame()
    df = df[df["SIGNAL_STATUS"] == "SETUP"].copy()
    if not df.empty and "MODE" in df.columns:
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
        return {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "close": float(latest["close"]),
            "high": float(latest["high"]),
            "low": float(latest["low"]),
            "volume": float(latest["volume"]),
            "rows": len(df),
        }
    except Exception:
        return None


def evaluate_setup_for_promotion(row: pd.Series, snapshot: dict[str, Any] | None) -> dict[str, Any]:
    """
    Phase 20A skeleton:
    - load setup
    - fetch latest snapshot
    - decide if it's eligible for future promotion logic

    This does NOT place orders yet.
    """
    if snapshot is None:
        return {
            "symbol": str(row["symbol"]),
            "mode": str(row["MODE"]),
            "promotion_candidate": False,
            "reason": "NO_MARKET_DATA",
        }

    entry_price = float(snapshot["close"])
    donchian_upper = row.get("DONCHIAN_UPPER")
    breakout_ready = False

    if pd.notna(donchian_upper):
        breakout_ready = entry_price >= float(donchian_upper)

    return {
        "symbol": str(row["symbol"]),
        "mode": str(row["MODE"]),
        "promotion_candidate": bool(breakout_ready),
        "reason": "BREAKOUT_READY" if breakout_ready else "WAITING_BREAKOUT",
        "latest_close": entry_price,
        "snapshot_time": snapshot["timestamp"],
    }


def dedupe_setup_rows(df: pd.DataFrame) -> pd.DataFrame:
    """
    Keep latest setup per symbol+mode.
    """
    if df.empty:
        return df

    work = df.copy()
    sort_cols = [col for col in ["signal_id", "SIGNAL_TIME", "started_at"] if col in work.columns]
    if sort_cols:
        work = work.sort_values(sort_cols, ascending=False)

    subset = [col for col in ["symbol", "MODE"] if col in work.columns]
    if subset:
        work = work.drop_duplicates(subset=subset, keep="first")

    return work.reset_index(drop=True)