from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from datetime import timedelta

import pandas as pd

from trading_ensemble.core.timeutils import now_ist
from trading_ensemble.core.timestamp_normalizer import normalize_external_timestamp
from trading_ensemble.data.smartapi_client import (
    fetch_candles_live,
    load_scrip_master,
    login_from_env,
)

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class RegimeSnapshot:
    regime: str
    adx_like: float
    atr_pct: float
    index_price: float
    index_sma20: float
    trend_gap_pct: float


def _unknown_snapshot() -> RegimeSnapshot:
    return RegimeSnapshot("UNKNOWN", 0.0, 0.0, 0.0, 0.0, 0.0)


def _resolve_index_token(scrip_df: pd.DataFrame, symbol: str) -> str:
    work = scrip_df.copy()
    work.columns = [str(c).strip().lower() for c in work.columns]
    if "token" not in work.columns:
        raise RuntimeError("Scrip master missing token column")

    sym = str(symbol).strip().upper()
    exch_col = work["exch_seg"].astype(str).str.upper() if "exch_seg" in work.columns else pd.Series([""] * len(work))
    sym_col = work["symbol"].astype(str).str.upper() if "symbol" in work.columns else pd.Series([""] * len(work))
    name_col = work["name"].astype(str).str.upper() if "name" in work.columns else pd.Series([""] * len(work))

    search_terms = [sym, "NIFTY", "NIFTY 50", "NIFTY50", "NIFTY-I"]

    for term in search_terms:
        exact = work[(exch_col == "NSE") & (sym_col == term)]
        if not exact.empty:
            return str(exact.iloc[0]["token"])

    for term in search_terms:
        by_name = work[(exch_col == "NSE") & (name_col.str.contains(term, na=False))]
        if not by_name.empty:
            return str(by_name.iloc[0]["token"])

    raise RuntimeError(f"Unable to resolve SmartAPI token for index symbol={symbol}")


def _fetch_index_candles(symbol: str) -> tuple[pd.DataFrame | None, dict]:
    now = now_ist()
    start = (now - timedelta(days=120)).strftime("%Y-%m-%d %H:%M")
    end = now.strftime("%Y-%m-%d %H:%M")

    source = "ANGEL_SMARTAPI"
    try:
        session = login_from_env()
        scrip_df = load_scrip_master()
        token = _resolve_index_token(scrip_df, symbol)
        candles = fetch_candles_live(
            session.smart,
            exchange="NSE",
            symbol_token=token,
            interval="ONE_DAY",
            start=start,
            end=end,
        )
    except Exception as exc:
        logger.error(json.dumps({
            "event": "regime_input_fetch_failed",
            "source": source,
            "symbol": symbol,
            "reason": str(exc),
        }))
        return None, {"source": source, "reason": str(exc)}

    if candles is None or candles.empty:
        reason = "EMPTY_DATA"
        logger.warning(json.dumps({
            "event": "regime_input_fetch_failed",
            "source": source,
            "symbol": symbol,
            "reason": reason,
        }))
        return None, {"source": source, "reason": reason}

    work = candles.copy()
    work.columns = [str(c).strip().lower() for c in work.columns]
    required = ["datetime", "high", "low", "close"]
    if any(c not in work.columns for c in required):
        reason = "INCOMPLETE_COLUMNS"
        logger.warning(json.dumps({
            "event": "regime_input_fetch_failed",
            "source": source,
            "symbol": symbol,
            "reason": reason,
            "columns": list(work.columns),
        }))
        return None, {"source": source, "reason": reason}

    work = work[required].dropna().reset_index(drop=True)
    if work.empty:
        return None, {"source": source, "reason": "EMPTY_AFTER_CLEAN"}

    parsed = work["datetime"].apply(
        lambda raw: normalize_external_timestamp(
            source="REGIME_INPUT_CANDLE",
            raw_value=raw,
            assume_exchange_local_ist=True,
            max_future_seconds=180,
        )
    )
    work["datetime"] = parsed.apply(lambda x: x.value_ist if x else None)
    work = work.dropna(subset=["datetime"]).reset_index(drop=True)
    if work.empty:
        return None, {"source": source, "reason": "MALFORMED_TIMESTAMPS"}

    latest_ts = work["datetime"].iloc[-1]
    freshness_hours = max(0.0, (now - latest_ts).total_seconds() / 3600.0)
    if freshness_hours > 60.0:
        reason = "STALE_INPUT"
        logger.warning(json.dumps({
            "event": "regime_input_stale",
            "source": source,
            "symbol": symbol,
            "timestamp": latest_ts.isoformat(),
            "freshness_hours": round(freshness_hours, 3),
            "reason": reason,
        }))
        return None, {"source": source, "reason": reason, "freshness_hours": freshness_hours}

    logger.info(json.dumps({
        "event": "regime_input_ready",
        "source": source,
        "symbol": symbol,
        "timestamp": latest_ts.isoformat(),
        "freshness_hours": round(freshness_hours, 3),
        "rows": len(work),
    }))

    return work, {
        "source": source,
        "freshness_hours": freshness_hours,
        "latest_timestamp": latest_ts.isoformat(),
    }


def fetch_index_regime(symbol: str = "NIFTY") -> RegimeSnapshot:
    try:
        df, meta = _fetch_index_candles(symbol)
        if df is None or df.empty:
            logger.warning(json.dumps({
                "event": "regime_evaluation_blocked",
                "symbol": symbol,
                "reason": (meta or {}).get("reason", "UNKNOWN"),
                "source": (meta or {}).get("source", "UNKNOWN"),
            }))
            return _unknown_snapshot()

        if len(df) < 25:
            logger.warning(json.dumps({
                "event": "regime_evaluation_blocked",
                "symbol": symbol,
                "source": (meta or {}).get("source", "UNKNOWN"),
                "reason": "INSUFFICIENT_ROWS",
                "rows": len(df),
            }))
            return _unknown_snapshot()

        high = df["high"].astype(float)
        low = df["low"].astype(float)
        close = df["close"].astype(float)

        prev_close = close.shift(1)
        tr = pd.concat(
            [
                high - low,
                (high - prev_close).abs(),
                (low - prev_close).abs(),
            ],
            axis=1,
        ).max(axis=1)

        atr = tr.ewm(span=14, adjust=False).mean()
        atr_pct = float((atr.iloc[-1] / close.iloc[-1]) * 100.0)

        sma20 = close.rolling(20).mean()
        index_price = float(close.iloc[-1])
        index_sma20 = float(sma20.iloc[-1])
        trend_gap_pct = ((index_price - index_sma20) / index_sma20) * 100.0 if index_sma20 else 0.0

        adx_like = min(abs(trend_gap_pct) * 8.0, 100.0)

        daily_return_pct = ((close.iloc[-1] - close.iloc[-2]) / close.iloc[-2]) * 100.0 if len(close) >= 2 else 0.0
        high_low_pct = ((high.iloc[-1] - low.iloc[-1]) / close.iloc[-1]) * 100.0 if close.iloc[-1] else 0.0

        if atr_pct >= 2.8 or abs(daily_return_pct) >= 2.5 or high_low_pct >= 3.5:
            regime = "CRISIS"
        elif atr_pct >= 2.2:
            regime = "HIGH_VOL"
        elif abs(trend_gap_pct) >= 1.0:
            regime = "TREND"
        else:
            regime = "CHOP"

        return RegimeSnapshot(
            regime=regime,
            adx_like=round(adx_like, 2),
            atr_pct=round(atr_pct, 4),
            index_price=round(index_price, 2),
            index_sma20=round(index_sma20, 2),
            trend_gap_pct=round(trend_gap_pct, 4),
        )
    except Exception as exc:
        logger.error(json.dumps({
            "event": "regime_evaluation_failed",
            "symbol": symbol,
            "reason": str(exc),
        }))
        return _unknown_snapshot()


def apply_regime_overrides(control_panel, regime_snapshot: RegimeSnapshot) -> dict:
    base_risk = float(getattr(control_panel, "risk_multiplier", 1.0)) if control_panel else 1.0
    base_orders = int(getattr(control_panel, "max_new_orders_per_run", 4)) if control_panel else 4

    trend_risk = float(getattr(control_panel, "trend_risk_multiplier", 1.15)) if control_panel else 1.15
    chop_risk = float(getattr(control_panel, "chop_risk_multiplier", 0.6)) if control_panel else 0.6
    high_vol_risk = float(getattr(control_panel, "high_vol_risk_multiplier", 0.5)) if control_panel else 0.5
    crisis_risk = float(getattr(control_panel, "crisis_risk_multiplier", 0.25)) if control_panel else 0.25

    trend_orders = int(getattr(control_panel, "trend_max_new_orders", base_orders)) if control_panel else base_orders
    chop_orders = int(getattr(control_panel, "chop_max_new_orders", max(1, min(2, base_orders)))) if control_panel else max(1, min(2, base_orders))
    high_vol_orders = int(getattr(control_panel, "high_vol_max_new_orders", 1)) if control_panel else 1
    crisis_orders = int(getattr(control_panel, "crisis_max_new_orders", 1)) if control_panel else 1

    regime = regime_snapshot.regime

    if regime == "TREND":
        return {
            "market_regime": regime,
            "effective_risk_multiplier": round(base_risk * trend_risk, 4),
            "effective_max_new_orders_per_run": trend_orders,
        }
    if regime == "CHOP":
        return {
            "market_regime": regime,
            "effective_risk_multiplier": round(base_risk * chop_risk, 4),
            "effective_max_new_orders_per_run": chop_orders,
        }
    if regime == "HIGH_VOL":
        return {
            "market_regime": regime,
            "effective_risk_multiplier": round(base_risk * high_vol_risk, 4),
            "effective_max_new_orders_per_run": high_vol_orders,
        }
    if regime == "CRISIS":
        return {
            "market_regime": regime,
            "effective_risk_multiplier": round(base_risk * crisis_risk, 4),
            "effective_max_new_orders_per_run": crisis_orders,
        }

    return {
        "market_regime": regime,
        "effective_risk_multiplier": round(base_risk, 4),
        "effective_max_new_orders_per_run": base_orders,
    }
