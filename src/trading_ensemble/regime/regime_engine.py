from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
import yfinance as yf


@dataclass(frozen=True)
class RegimeSnapshot:
    regime: str
    adx_like: float
    atr_pct: float
    index_price: float
    index_sma20: float
    trend_gap_pct: float


def fetch_index_regime(symbol: str = "^NSEI") -> RegimeSnapshot:
    try:
        df = yf.download(
            symbol,
            period="3mo",
            interval="1d",
            progress=False,
            auto_adjust=True,
        )
        if df is None or df.empty:
            return RegimeSnapshot("UNKNOWN", 0.0, 0.0, 0.0, 0.0, 0.0)

        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)

        df.columns = [str(c).strip().lower() for c in df.columns]
        df = df[["high", "low", "close"]].copy().dropna()

        if len(df) < 25:
            return RegimeSnapshot("UNKNOWN", 0.0, 0.0, 0.0, 0.0, 0.0)

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

        # lightweight ADX-like proxy
        adx_like = min(abs(trend_gap_pct) * 8.0, 100.0)

        if atr_pct >= 2.2:
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
    except Exception:
        return RegimeSnapshot("UNKNOWN", 0.0, 0.0, 0.0, 0.0, 0.0)


def apply_regime_overrides(control_panel, regime_snapshot: RegimeSnapshot) -> dict:
    base_risk = float(getattr(control_panel, "risk_multiplier", 1.0)) if control_panel else 1.0
    base_orders = int(getattr(control_panel, "max_new_orders_per_run", 4)) if control_panel else 4

    trend_risk = float(getattr(control_panel, "trend_risk_multiplier", 1.15)) if control_panel else 1.15
    chop_risk = float(getattr(control_panel, "chop_risk_multiplier", 0.6)) if control_panel else 0.6
    high_vol_risk = float(getattr(control_panel, "high_vol_risk_multiplier", 0.5)) if control_panel else 0.5

    trend_orders = int(getattr(control_panel, "trend_max_new_orders", base_orders)) if control_panel else base_orders
    chop_orders = int(getattr(control_panel, "chop_max_new_orders", max(1, min(2, base_orders)))) if control_panel else max(1, min(2, base_orders))
    high_vol_orders = int(getattr(control_panel, "high_vol_max_new_orders", 1)) if control_panel else 1

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

    return {
        "market_regime": regime,
        "effective_risk_multiplier": round(base_risk, 4),
        "effective_max_new_orders_per_run": base_orders,
    }
