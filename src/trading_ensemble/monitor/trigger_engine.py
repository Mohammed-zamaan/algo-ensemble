from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any

import pandas as pd

from trading_ensemble.core.timeutils import fmt_ist, now_ist
from trading_ensemble.core.timestamp_normalizer import normalize_external_timestamp
from trading_ensemble.data.smartapi_client import (
    load_scrip_master,
    login_from_env,
    resolve_symbol_to_token_offline,
)

logger = logging.getLogger(__name__)


def mode_to_market_data(mode: str) -> dict[str, Any]:
    mode = str(mode).upper()
    if mode == "INTRADAY":
        return {
            "interval": "FIFTEEN_MINUTE",
            "lookback_days": 3,
            "donchian_period": 20,
            "max_age_minutes": 20,
        }
    if mode == "SWING":
        return {
            "interval": "ONE_DAY",
            "lookback_days": 120,
            "donchian_period": 20,
            "max_age_minutes": 60 * 36,
        }
    if mode == "POSITIONAL":
        return {
            "interval": "ONE_DAY",
            "lookback_days": 240,
            "donchian_period": 55,
            "max_age_minutes": 60 * 36,
        }
    return {
        "interval": "FIFTEEN_MINUTE",
        "lookback_days": 3,
        "donchian_period": 20,
        "max_age_minutes": 20,
    }


@dataclass
class ExecutionDataProvider:
    smart: Any
    scrip_df: pd.DataFrame

    @classmethod
    def from_env(cls) -> "ExecutionDataProvider":
        session = login_from_env()
        return cls(smart=session.smart, scrip_df=load_scrip_master())

    def fetch_candles(self, symbol: str, mode: str) -> pd.DataFrame:
        cfg = mode_to_market_data(mode)
        _, token = resolve_symbol_to_token_offline(symbol, exchange="NSE", scrip_df=self.scrip_df)
        now = now_ist().replace(tzinfo=None)
        start = (now - timedelta(days=int(cfg["lookback_days"]))).strftime("%Y-%m-%d %H%M")
        end = now.strftime("%Y-%m-%d %H%M")

        params = {
            "exchange": "NSE",
            "symboltoken": str(token),
            "interval": str(cfg["interval"]),
            "fromdate": start,
            "todate": end,
        }
        raw = self.smart.getCandleData(params)
        return _candles_to_frame(raw)


@dataclass
class ReferenceDataProvider:
    def fetch_last_price(self, symbol: str) -> float | None:
        try:
            from jugaad_trader.nse import NSELive

            quote = NSELive().stock_quote(symbol.replace("-EQ", "").strip().upper())
            info = quote.get("priceInfo", {})
            last_price = info.get("lastPrice")

            raw_ts = (
                quote.get("metadata", {}).get("lastUpdateTime")
                or quote.get("metadata", {}).get("lastUpdateDate")
                or info.get("lastUpdateTime")
            )
            if raw_ts is not None:
                normalize_external_timestamp(
                    source="NSE_REFERENCE_QUOTE",
                    raw_value=raw_ts,
                    assume_exchange_local_ist=True,
                    max_future_seconds=120,
                    max_age_minutes=30,
                )

            return float(last_price) if last_price is not None else None
        except Exception:
            return None


def _candles_to_frame(raw: Any) -> pd.DataFrame:
    cols = ["datetime", "open", "high", "low", "close", "volume"]
    data = raw.get("data") if isinstance(raw, dict) else None
    if not data:
        return pd.DataFrame(columns=cols)

    df = pd.DataFrame(data, columns=cols)
    df["datetime"] = df["datetime"].apply(
        lambda x: normalize_external_timestamp(
            source="ANGEL_SMARTAPI_TRIGGER",
            raw_value=x,
            assume_exchange_local_ist=True,
            max_future_seconds=120,
        )
    )
    df["datetime"] = df["datetime"].apply(lambda x: x.value_ist if x is not None else None)

    for col in ["open", "high", "low", "close", "volume"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(subset=["datetime", "open", "high", "low", "close", "volume"])
    return df.sort_values("datetime").drop_duplicates(subset=["datetime"]).reset_index(drop=True)


def _age_minutes(latest_candle_ts: datetime, now: datetime) -> float:
    return max(0.0, (now - latest_candle_ts).total_seconds() / 60.0)


def _safe_gap_pct(current: float, required: float) -> float:
    if required <= 0:
        return 0.0
    return round(max(0.0, ((required - current) / required) * 100), 4)


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

    df["MODE"] = df["MODE"].astype(str).str.upper()
    return df


def fetch_latest_market_snapshot(symbol: str, mode: str) -> dict[str, Any] | None:
    cfg = mode_to_market_data(mode)
    reference_invoked = False
    mismatch_outcome = "NOT_INVOKED"

    try:
        execution_provider = ExecutionDataProvider.from_env()
        angel_df = execution_provider.fetch_candles(symbol, mode)
    except Exception as exc:
        logger.error(json.dumps({
            "event": "trigger_data_error",
            "symbol": symbol,
            "mode": str(mode).upper(),
            "source_used": "ANGEL_SMARTAPI",
            "error": str(exc),
            "nse_verification_invoked": False,
        }))
        return None

    if angel_df.empty:
        logger.warning(json.dumps({
            "event": "trigger_data_invalid",
            "symbol": symbol,
            "mode": str(mode).upper(),
            "source_used": "ANGEL_SMARTAPI",
            "reason": "EMPTY_CANDLES",
            "nse_verification_invoked": False,
        }))
        return None

    latest = angel_df.iloc[-1]
    latest_ts = latest["datetime"]
    now = now_ist()
    freshness_minutes = _age_minutes(latest_ts, now)
    min_rows = int(cfg["donchian_period"]) + 2
    complete_enough = len(angel_df) >= min_rows
    fresh_enough = freshness_minutes <= float(cfg["max_age_minutes"])
    suspicious_or_stale = (not complete_enough) or (not fresh_enough)

    if suspicious_or_stale:
        reference_invoked = True
        ref_provider = ReferenceDataProvider()
        ref_last_price = ref_provider.fetch_last_price(symbol)
        if ref_last_price is None:
            mismatch_outcome = "NSE_UNAVAILABLE"
        else:
            angel_close = float(latest["close"])
            diff_pct = abs(angel_close - ref_last_price) / max(angel_close, 1e-9) * 100
            mismatch_outcome = "MISMATCH" if diff_pct > 1.5 else "MATCH"

    if (not complete_enough) or (not fresh_enough) or (mismatch_outcome == "MISMATCH"):
        logger.warning(json.dumps({
            "event": "trigger_data_blocked",
            "symbol": symbol,
            "mode": str(mode).upper(),
            "source_used": "ANGEL_SMARTAPI",
            "candle_timestamp": latest_ts.strftime("%Y-%m-%d %H:%M:%S"),
            "freshness_age_minutes": round(freshness_minutes, 2),
            "nse_verification_invoked": reference_invoked,
            "mismatch_outcome": mismatch_outcome,
            "complete_enough": complete_enough,
            "fresh_enough": fresh_enough,
        }))
        return None

    donchian_period = int(cfg["donchian_period"])
    if len(angel_df) >= donchian_period + 1:
        live_donchian_upper = float(angel_df["high"].rolling(donchian_period).max().iloc[-2])
        avg_volume = float(angel_df["volume"].iloc[-donchian_period - 1 : -1].mean())
    else:
        live_donchian_upper = float(angel_df["high"].max())
        avg_volume = float(angel_df["volume"].mean())

    logger.info(json.dumps({
        "event": "trigger_data_ready",
        "symbol": symbol,
        "mode": str(mode).upper(),
        "source_used": "ANGEL_SMARTAPI",
        "candle_timestamp": latest_ts.strftime("%Y-%m-%d %H:%M:%S"),
        "freshness_age_minutes": round(freshness_minutes, 2),
        "nse_verification_invoked": reference_invoked,
        "mismatch_outcome": mismatch_outcome,
    }))

    return {
        "timestamp": fmt_ist(),
        "close": float(latest["close"]),
        "high": float(latest["high"]),
        "low": float(latest["low"]),
        "volume": float(latest["volume"]),
        "avg_volume": avg_volume,
        "live_donchian_upper": live_donchian_upper,
        "rows": len(angel_df),
    }


def evaluate_setup_for_promotion(row: pd.Series, snapshot: dict[str, Any] | None) -> dict[str, Any]:
    symbol = str(row["symbol"])
    mode = str(row["MODE"]).upper()

    if snapshot is None:
        return {
            "symbol": symbol,
            "MODE": mode,
            "promotion_candidate": False,
            "promotion_reason": "NO_MARKET_DATA",
            "latest_close": None,
            "live_donchian_upper": None,
            "breakout_gap_pct": None,
            "latest_volume": None,
            "live_volume_ratio": None,
            "volume_gap_pct": None,
            "snapshot_time": None,
        }

    latest_close = float(snapshot["close"])
    latest_volume = float(snapshot["volume"])
    avg_volume = float(snapshot["avg_volume"]) if snapshot["avg_volume"] > 0 else 0.0
    live_volume_ratio = latest_volume / avg_volume if avg_volume > 0 else 0.0

    donchian_upper = float(snapshot["live_donchian_upper"])
    breakout_ready = latest_close >= donchian_upper

    volume_threshold = 2.0 if mode == "INTRADAY" else 1.5 if mode == "SWING" else 1.3
    volume_ready = live_volume_ratio >= volume_threshold
    atr_ready = True

    breakout_gap_pct = _safe_gap_pct(latest_close, donchian_upper)
    volume_gap_pct = _safe_gap_pct(live_volume_ratio, volume_threshold)

    promotion_candidate = breakout_ready and volume_ready and atr_ready

    if not breakout_ready:
        reason = "WAITING_BREAKOUT"
    elif not volume_ready:
        reason = "WAITING_VOLUME"
    elif not atr_ready:
        reason = "WAITING_ATR"
    else:
        reason = "PROMOTED"

    readiness_score = round(
        100.0 - breakout_gap_pct - volume_gap_pct,
        4,
    )

    return {
        "symbol": symbol,
        "MODE": mode,
        "promotion_candidate": bool(promotion_candidate),
        "promotion_reason": reason,
        "latest_close": round(latest_close, 4),
        "live_donchian_upper": round(donchian_upper, 4),
        "breakout_gap_pct": breakout_gap_pct,
        "latest_volume": round(latest_volume, 2),
        "live_volume_ratio": round(live_volume_ratio, 4),
        "volume_gap_pct": volume_gap_pct,
        "readiness_score": readiness_score,
        "snapshot_time": snapshot["timestamp"],
        "signal_id": row.get("signal_id"),
        "ENTRY_PRICE": row.get("ENTRY_PRICE"),
        "STOP_LOSS": row.get("STOP_LOSS"),
        "TARGET_PRICE": row.get("TARGET_PRICE"),
        "RR_RATIO": row.get("RR_RATIO"),
        "PRODUCT_TYPE": row.get("PRODUCT_TYPE"),
        "SIGNAL_TIME": row.get("SIGNAL_TIME"),
        "WATCHLIST_CONVICTION": row.get("WATCHLIST_CONVICTION", 2),
        "SECTOR": row.get("SECTOR", "UNKNOWN"),
        "PRIORITY": row.get("PRIORITY", 0),
        "COMPOSITE_SCORE": row.get("COMPOSITE_SCORE", 0),
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


def has_open_position(store, symbol: str) -> bool:
    with store.connect() as conn:
        row = conn.execute(
            """
            SELECT 1
            FROM positions
            WHERE symbol = ? AND status = 'OPEN'
            LIMIT 1
            """,
            (symbol,),
        ).fetchone()
    return row is not None


def has_existing_order_today(store, symbol: str) -> bool:
    now = now_ist()
    day_start_ist = now.replace(hour=0, minute=0, second=0, microsecond=0).isoformat(timespec="seconds")
    next_day_start_ist = (now.replace(hour=0, minute=0, second=0, microsecond=0) + pd.Timedelta(days=1)).isoformat(timespec="seconds")
    with store.connect() as conn:
        row = conn.execute(
            """
            SELECT 1
            FROM orders
            WHERE symbol = ?
              AND created_at >= ?
              AND created_at < ?
            LIMIT 1
            """,
            (symbol, day_start_ist, next_day_start_ist),
        ).fetchone()
    return row is not None


def filter_promotion_candidates(store, promoted_df: pd.DataFrame) -> pd.DataFrame:
    if promoted_df.empty:
        return promoted_df

    rows = []
    for _, row in promoted_df.iterrows():
        symbol = str(row["symbol"])

        if has_open_position(store, symbol):
            continue
        if has_existing_order_today(store, symbol):
            continue

        rows.append(row)

    if not rows:
        return pd.DataFrame(columns=promoted_df.columns)

    return pd.DataFrame(rows).reset_index(drop=True)


def convert_promotions_to_signals(promoted_df: pd.DataFrame) -> pd.DataFrame:
    if promoted_df.empty:
        return pd.DataFrame()

    rows = []
    promoted_at = fmt_ist()

    for _, row in promoted_df.iterrows():
        rows.append(
            {
                "symbol": row["symbol"],
                "MODE": row["MODE"],
                "COMPOSITE_SCORE": float(row.get("COMPOSITE_SCORE", 0) or 0),
                "ENTRY_PRICE": float(row["latest_close"]),
                "STOP_LOSS": float(row["STOP_LOSS"]),
                "TARGET_PRICE": float(row["TARGET_PRICE"]),
                "RR_RATIO": float(row["RR_RATIO"]),
                "ATR": None,
                "ATR_RATIO_PCT": None,
                "ADX": None,
                "VOLUME_RATIO": float(row.get("live_volume_ratio", 0) or 0),
                "DONCHIAN_UPPER": float(row.get("live_donchian_upper", 0) or 0),
                "BREAKOUT_15M": True,
                "BREAKOUT_READY": True,
                "VOLUME_READY": True,
                "ATR_READY": True,
                "PRODUCT_TYPE": row.get("PRODUCT_TYPE", "MIS"),
                "SIGNAL_TIME": promoted_at,
                "SIGNAL_STRENGTH": "PROMOTED",
                "SIGNAL_STATUS": "CONFIRMED",
                "SETUP_REASON": "PROMOTED_BY_MONITOR",
                "WATCHLIST_CONVICTION": float(row.get("WATCHLIST_CONVICTION", 2) or 2),
                "SECTOR": row.get("SECTOR", "UNKNOWN"),
                "PRIORITY": int(row.get("PRIORITY", 0) or 0),
                "PROMOTED_AT": promoted_at,
                "PROMOTION_REASON": row.get("promotion_reason", "PROMOTED"),
            }
        )

    return pd.DataFrame(rows)
