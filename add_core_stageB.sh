#!/usr/bin/env bash
set -euo pipefail

cd /workspaces/algo-ensemble

mkdir -p src/trading_ensemble/core

# indicators.py (RMA, ATR, ADX) extracted from StageB [file:35]
cat > src/trading_ensemble/core/indicators.py << 'EOF'
"""Indicators (Wilder RMA, ATR, ADX) from StageB notebook [file:35]."""
from __future__ import annotations

import numpy as np
import pandas as pd


def rma(series: pd.Series, length: int) -> pd.Series:
    """Wilder RMA via EMA(alpha=1/length)."""
    return series.ewm(alpha=1.0 / float(length), adjust=False).mean()


def true_range(high: pd.Series, low: pd.Series, close: pd.Series) -> pd.Series:
    prev_close = close.shift(1)
    tr = pd.concat(
        [
            (high - low).abs(),
            (high - prev_close).abs(),
            (low - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    return tr


def atr_wilder(high: pd.Series, low: pd.Series, close: pd.Series, length: int) -> pd.Series:
    return rma(true_range(high, low, close), length)


def adx_wilder(high: pd.Series, low: pd.Series, close: pd.Series, di_len: int, adx_smooth: int) -> pd.Series:
    up = high.diff()
    dn = -low.diff()

    plus_dm = np.where((up > dn) & (up > 0), up, 0.0)
    minus_dm = np.where((dn > up) & (dn > 0), dn, 0.0)

    tr = true_range(high, low, close)
    tr_rma = rma(tr, di_len)

    plus_rma = rma(pd.Series(plus_dm, index=high.index), di_len)
    minus_rma = rma(pd.Series(minus_dm, index=high.index), di_len)

    plus_di = 100.0 * (plus_rma / tr_rma)
    minus_di = 100.0 * (minus_rma / tr_rma)

    dx = 100.0 * (plus_di.sub(minus_di).abs() / plus_di.add(minus_di))
    adx = rma(dx, adx_smooth)
    return adx
EOF

# engine.py (validation, features, donchian regime, signals, backtest) [file:35]
cat > src/trading_ensemble/core/engine.py << 'EOF'
"""Feature pipeline + long-only backtest extracted from StageB [file:35]."""
from __future__ import annotations

from dataclasses import asdict
from typing import Any, Dict, Tuple, List, Optional

import numpy as np
import pandas as pd

from .params import StrategyParams
from .indicators import atr_wilder, adx_wilder


def validate_ohlcv(df: pd.DataFrame) -> pd.DataFrame:
    required = ["datetime", "open", "high", "low", "close", "volume"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"OHLCV missing columns: {missing}")

    out = df.copy()
    out["datetime"] = pd.to_datetime(out["datetime"], errors="coerce")
    out = out.sort_values("datetime").reset_index(drop=True)

    for c in ["open", "high", "low", "close", "volume"]:
        out[c] = pd.to_numeric(out[c], errors="coerce")

    out = out.dropna(subset=["datetime", "open", "high", "low", "close"]).reset_index(drop=True)

    bad = (
        (out["high"] < out["low"])
        | (out["high"] < out["open"])
        | (out["high"] < out["close"])
        | (out["low"] > out["open"])
        | (out["low"] > out["close"])
    )
    out = out.loc[~bad].reset_index(drop=True)
    return out


def _session_mask(df: pd.DataFrame, start_hhmm: str, end_hhmm: str) -> pd.Series:
    t = df["datetime"].dt.strftime("%H%M")
    return (t >= start_hhmm) & (t <= end_hhmm)


def compute_indicators(df: pd.DataFrame, p: StrategyParams) -> pd.DataFrame:
    out = df.copy()

    out["sma"] = out["close"].rolling(p.sma_len, min_periods=p.sma_len).mean()
    out["atr"] = atr_wilder(out["high"], out["low"], out["close"], p.atr_len)
    out["adx"] = adx_wilder(out["high"], out["low"], out["close"], p.di_len, p.adx_smooth)

    out["atrp"] = 100.0 * (out["atr"] / out["close"])
    out["atrp_sm"] = out["atrp"].rolling(p.atrp_len, min_periods=p.atrp_len).mean()
    return out


def compute_filters(df: pd.DataFrame, p: StrategyParams) -> pd.DataFrame:
    out = df.copy()

    out["in_session"] = True
    if p.use_session:
        out["in_session"] = _session_mask(out, p.session_start, p.session_end)

    if p.use_sma:
        trend_long = out["close"] > out["sma"]
    else:
        trend_long = pd.Series(True, index=out.index)

    if p.use_adx:
        adx_ok = out["adx"] >= p.adx_min
        if p.adx_rising:
            adx_ok = adx_ok & (out["adx"] > out["adx"].shift(1))
    else:
        adx_ok = pd.Series(True, index=out.index)

    out["allow_long"] = out["in_session"] & trend_long & adx_ok
    return out


def choose_regime_params(atrp_sm: float, p: StrategyParams) -> Tuple[int, int, float]:
    if np.isnan(atrp_sm):
        return p.don_entry_base, p.don_exit_base, p.trail_mult_lo

    is_high = atrp_sm >= p.atrp_hi
    is_low = atrp_sm <= p.atrp_lo

    don_entry = p.don_entry_hi if is_high else (p.don_entry_lo if is_low else p.don_entry_base)
    don_exit = p.don_exit_hi if is_high else (p.don_exit_lo if is_low else p.don_exit_base)
    trail_mult = p.trail_mult_hi if is_high else p.trail_mult_lo
    return don_entry, don_exit, trail_mult


def compute_dynamic_donchian(df: pd.DataFrame, p: StrategyParams) -> pd.DataFrame:
    out = df.copy()
    n = len(out)

    up_entry = np.full(n, np.nan)
    dn_exit = np.full(n, np.nan)
    trail_mult_eff = np.full(n, np.nan)

    highs = out["high"].to_numpy()
    lows = out["low"].to_numpy()

    for i in range(n):
        le, lx, tm = choose_regime_params(float(out["atrp_sm"].iloc[i]), p)
        trail_mult_eff[i] = tm

        if i - le >= 0:
            up_entry[i] = np.max(highs[i - le : i])
        if i - lx >= 0:
            dn_exit[i] = np.min(lows[i - lx : i])

    out["don_up_entry"] = up_entry
    out["don_dn_exit"] = dn_exit
    out["trail_mult_eff"] = trail_mult_eff
    return out


def compute_signals(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["long_signal"] = out["allow_long"] & (out["close"] > out["don_up_entry"])
    out["don_exit_signal"] = out["close"] < out["don_dn_exit"]
    return out


def _commission(value: float, commission_pct: float) -> float:
    return float(value) * float(commission_pct) / 100.0


def size_qty_no_leverage(equity: float, price: float, atr: float, p: StrategyParams) -> int:
    if price <= 0 or atr <= 0 or np.isnan(price) or np.isnan(atr):
        return 0

    qty_cash_cap = int(np.floor(equity / price))
    risk_cash = equity * (p.risk_pct / 100.0)
    stop_dist = max(p.stop_mult * atr, 1e-9)
    qty_risk = int(np.floor(risk_cash / stop_dist))

    qty = max(0, min(qty_cash_cap, qty_risk))
    return qty


def backtest_long_only(df: pd.DataFrame, p: StrategyParams) -> Dict[str, Any]:
    equity = float(p.initial_capital)

    in_pos = False
    qty = 0
    entry_price = np.nan
    entry_i: Optional[int] = None

    hi_since = np.nan
    bars_in_pos = 0

    trades: List[Dict[str, Any]] = []

    for i in range(1, len(df)):
        row = df.iloc[i]
        h = float(row["high"])
        l = float(row["low"])
        c = float(row["close"])
        atr = float(row["atr"]) if not np.isnan(row["atr"]) else np.nan

        if np.isnan(atr) or atr <= 0:
            continue

        if in_pos:
            bars_in_pos += 1
            hi_since = max(hi_since, h) if not np.isnan(hi_since) else h

            if bool(row["don_exit_signal"]):
                exit_price = c
                gross = qty * (exit_price - entry_price)
                cost = _commission(qty * entry_price, p.commission_pct) + _commission(qty * exit_price, p.commission_pct)
                net = gross - cost
                equity += net
                trades.append({"entryi": entry_i, "exiti": i, "entry": entry_price, "exit": exit_price, "qty": qty, "pl": net, "reason": "DonchianExit"})
                in_pos = False
                qty = 0
                entry_price = np.nan
                entry_i = None
                hi_since = np.nan
                bars_in_pos = 0
                continue

            long_stop_init = entry_price - (p.stop_mult * atr)

            tm = float(row["trail_mult_eff"]) if not np.isnan(row["trail_mult_eff"]) else p.trail_mult_lo
            progress_ok = (hi_since - entry_price) >= (p.trail_start_atr * atr)
            trail_allowed = (bars_in_pos >= p.min_hold_bars) and progress_ok
            trail_stop = hi_since - (tm * atr)

            stop_price = max(long_stop_init, trail_stop if trail_allowed else long_stop_init)

            if l <= stop_price:
                exit_price = stop_price
                gross = qty * (exit_price - entry_price)
                cost = _commission(qty * entry_price, p.commission_pct) + _commission(qty * exit_price, p.commission_pct)
                net = gross - cost
                equity += net
                trades.append({"entryi": entry_i, "exiti": i, "entry": entry_price, "exit": exit_price, "qty": qty, "pl": net, "reason": "Stop"})
                in_pos = False
                qty = 0
                entry_price = np.nan
                entry_i = None
                hi_since = np.nan
                bars_in_pos = 0
                continue

        if (not in_pos) and bool(row["long_signal"]):
            qty_new = size_qty_no_leverage(equity=equity, price=c, atr=atr, p=p)
            if qty_new >= p.min_qty:
                in_pos = True
                qty = qty_new
                entry_price = c
                entry_i = i
                hi_since = h
                bars_in_pos = 0

    if in_pos:
        exit_price = float(df.iloc[-1]["close"])
        gross = qty * (exit_price - entry_price)
        cost = _commission(qty * entry_price, p.commission_pct) + _commission(qty * exit_price, p.commission_pct)
        net = gross - cost
        equity += net
        trades.append({"entryi": entry_i, "exiti": len(df) - 1, "entry": entry_price, "exit": exit_price, "qty": qty, "pl": net, "reason": "LastBar"})

    trades_df = pd.DataFrame(trades)
    return {
        "final_equity": equity,
        "net_profit": equity - float(p.initial_capital),
        "trades": trades_df,
        "params": asdict(p),
    }


def build_feature_pipeline(df: pd.DataFrame, p: StrategyParams) -> pd.DataFrame:
    df0 = validate_ohlcv(df)
    df1 = compute_indicators(df0, p)
    df2 = compute_filters(df1, p)
    df3 = compute_dynamic_donchian(df2, p)
    df4 = compute_signals(df3)
    return df4


def run_strategy_workflow(df: pd.DataFrame, p: StrategyParams) -> Dict[str, Any]:
    bars = build_feature_pipeline(df, p)
    if not p.long_only:
        raise NotImplementedError("Only long_only=True supported in this engine.")
    res = backtest_long_only(bars, p)
    res["bars"] = bars
    return res
EOF

echo "Created indicators.py and engine.py"
