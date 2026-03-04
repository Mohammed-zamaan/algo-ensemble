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
    """Vectorised Donchian with regime-adaptive windows. Replaces Python for-loop."""
    out = df.copy()
    out["_up_lo"]   = out["high"].shift(1).rolling(p.don_entry_lo,   min_periods=p.don_entry_lo).max()
    out["_up_base"] = out["high"].shift(1).rolling(p.don_entry_base, min_periods=p.don_entry_base).max()
    out["_up_hi"]   = out["high"].shift(1).rolling(p.don_entry_hi,   min_periods=p.don_entry_hi).max()
    out["_dn_lo"]   = out["low"].shift(1).rolling(p.don_exit_lo,     min_periods=p.don_exit_lo).min()
    out["_dn_base"] = out["low"].shift(1).rolling(p.don_exit_base,   min_periods=p.don_exit_base).min()
    out["_dn_hi"]   = out["low"].shift(1).rolling(p.don_exit_hi,     min_periods=p.don_exit_hi).min()
    is_hi = out["atrp_sm"] >= p.atrp_hi
    is_lo = out["atrp_sm"] <= p.atrp_lo
    out["don_up_entry"]   = np.where(is_hi, out["_up_hi"], np.where(is_lo, out["_up_lo"], out["_up_base"]))
    out["don_dn_exit"]    = np.where(is_hi, out["_dn_hi"], np.where(is_lo, out["_dn_lo"], out["_dn_base"]))
    out["trail_mult_eff"] = np.where(is_hi, p.trail_mult_hi, p.trail_mult_lo)
    out.drop(columns=[c for c in out.columns if c.startswith("_up_") or c.startswith("_dn_")], inplace=True)
    return out


def compute_signals(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["long_signal"] = out["allow_long"] & (out["close"] > out["don_up_entry"])
    out["don_exit_signal"] = out["close"] < out["don_dn_exit"]
    return out


def _commission(value: float, commission_pct: float) -> float:
    return float(value) * float(commission_pct) / 100.0


def compute_conviction_multiplier(conviction: int, is_crisis: bool = False) -> float:
    """
    Compute position size multiplier based on conviction level.
    
    Conviction levels:
    - 1 (MANUAL): 0.9x
    - 2 (ANGEL_ONE): 1.0x  
    - 3 (BOTH): 1.1x
    
    Crisis regime: conviction 3 → 1.0x (suppress boost)
    """
    if is_crisis and conviction == 3:
        return 1.0
    
    multipliers = {1: 0.9, 2: 1.0, 3: 1.1}
    return multipliers.get(conviction, 1.0)

def size_qty_with_conviction(equity: float, price: float, atr: float, 
                             conviction: int, p: StrategyParams, 
                             is_crisis: bool = False) -> int:
    """Size position with conviction multiplier applied."""
    if price <= 0 or atr <= 0 or np.isnan(price) or np.isnan(atr):
        return 0
    
    # Base position size (same as before)
    qty_cash_cap = int(np.floor(equity / price))
    risk_cash = equity * (p.risk_pct / 100.0)
    stop_dist = max(p.stop_mult * atr, 1e-9)
    qty_risk = int(np.floor(risk_cash / stop_dist))
    qty_base = max(0, min(qty_cash_cap, qty_risk))
    
    # Apply conviction multiplier
    conviction_mult = compute_conviction_multiplier(conviction, is_crisis)
    qty_final = int(np.floor(qty_base * conviction_mult))
    
    # Cap at 1.1x of base (max boost)
    qty_max = int(np.floor(qty_base * 1.1))
    return max(0, min(qty_final, qty_max))



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
