from __future__ import annotations

from datetime import datetime, time as dt_time

import numpy as np
import pandas as pd
import yfinance as yf

from trading_ensemble.data.sheets_output import maybe_write_output
from trading_ensemble.strategy.modes import resolve_symbol_modes
from ..engine import PipelineStage


MODE_CONFIG = {
    "INTRADAY": {
        "donchian_period": 20,
        "atr_period": 14,
        "atr_sl_mult": 1.5,
        "atr_target_mult": 2.5,
        "min_rr": 1.5,
        "volume_mult": 2.0,
        "atr_min": 5.0,
        "yf_interval": "15m",
        "yf_period": "5d",
        "product_type": "MIS",
        "entry_after": dt_time(9, 30),
        "exit_before": dt_time(15, 15),
    },
    "SWING": {
        "donchian_period": 20,
        "atr_period": 14,
        "atr_sl_mult": 2.0,
        "atr_target_mult": 4.0,
        "min_rr": 2.0,
        "volume_mult": 1.5,
        "atr_min": 5.0,
        "yf_interval": "1d",
        "yf_period": "3mo",
        "product_type": "CNC",
        "entry_after": dt_time(9, 30),
        "exit_before": dt_time(15, 15),
    },
    "POSITIONAL": {
        "donchian_period": 55,
        "atr_period": 21,
        "atr_sl_mult": 3.0,
        "atr_target_mult": 8.0,
        "min_rr": 2.5,
        "volume_mult": 1.3,
        "atr_min": 5.0,
        "yf_interval": "1d",
        "yf_period": "6mo",
        "product_type": "CNC",
        "entry_after": dt_time(9, 30),
        "exit_before": dt_time(15, 15),
    },
}


def to_yf_ticker(symbol: str) -> str:
    return symbol.replace("-EQ", "").strip() + ".NS"


def fetch_candles(symbol: str, cfg: dict) -> pd.DataFrame | None:
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
            return None

        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)

        df.columns = [c.strip().lower() for c in df.columns]
        df = df[["open", "high", "low", "close", "volume"]].copy()
        df.dropna(inplace=True)

        if len(df) < cfg["donchian_period"] + 5:
            return None

        return df
    except Exception:
        return None


def compute_atr(df: pd.DataFrame, period: int) -> float:
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

    atr = tr.ewm(span=period, adjust=False).mean()
    return float(atr.iloc[-1])


def compute_donchian_upper(df: pd.DataFrame, period: int) -> float:
    return float(df["high"].rolling(period).max().iloc[-2])


def compute_avg_volume(df: pd.DataFrame, period: int) -> float:
    return float(df["volume"].iloc[-period - 1 : -1].mean())


def compute_adx(df: pd.DataFrame, period: int = 14) -> float:
    try:
        high = df["high"].astype(float)
        low = df["low"].astype(float)
        close = df["close"].astype(float)

        plus_dm = high.diff()
        minus_dm = -low.diff()
        plus_dm[plus_dm < 0] = 0
        minus_dm[minus_dm < 0] = 0

        tr = pd.concat(
            [
                high - low,
                (high - close.shift()).abs(),
                (low - close.shift()).abs(),
            ],
            axis=1,
        ).max(axis=1)

        atr14 = tr.ewm(span=period, adjust=False).mean()
        plus_di = 100 * plus_dm.ewm(span=period, adjust=False).mean() / atr14
        minus_di = 100 * minus_dm.ewm(span=period, adjust=False).mean() / atr14
        dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)
        adx = dx.ewm(span=period, adjust=False).mean()
        return float(adx.iloc[-1])
    except Exception:
        return 0.0


def get_signal_strength(price_above: bool, vol_ratio: float, atr_ratio: float, adx: float) -> str:
    if not price_above:
        return "REJECTED"

    score = 0

    if vol_ratio >= 2.5:
        score += 2
    elif vol_ratio >= 2.0:
        score += 1

    if atr_ratio >= 1.5:
        score += 2
    elif atr_ratio >= 1.0:
        score += 1

    if adx >= 30:
        score += 2
    elif adx >= 20:
        score += 1

    if score >= 5:
        return "STRONG"
    if score >= 3:
        return "CONFIRMED"
    return "WEAK"


def generate_signal(row: pd.Series, mode: str) -> tuple[dict | None, str]:
    cfg = MODE_CONFIG[mode]
    symbol = row["symbol"]

    df = fetch_candles(symbol, cfg)
    if df is None:
        return None, "no_data"

    atr = compute_atr(df, cfg["atr_period"])
    donchian_upper = compute_donchian_upper(df, cfg["donchian_period"])
    avg_vol = compute_avg_volume(df, cfg["donchian_period"])
    adx = compute_adx(df, 14)

    entry_price = float(df["close"].iloc[-1])
    current_vol = float(df["volume"].iloc[-1])
    vol_ratio = current_vol / avg_vol if avg_vol > 0 else 0.0
    atr_ratio = atr / entry_price * 100

    price_above = entry_price >= donchian_upper
    vol_ok = vol_ratio >= cfg["volume_mult"]
    atr_ok = atr >= cfg["atr_min"]

    stop_loss = round(entry_price - cfg["atr_sl_mult"] * atr, 2)
    target_price = round(entry_price + cfg["atr_target_mult"] * atr, 2)
    rr_ratio = round((target_price - entry_price) / (entry_price - stop_loss), 2) if (entry_price - stop_loss) > 0 else 0.0

    base = {
        "symbol": symbol,
        "MODE": mode,
        "COMPOSITE_SCORE": float(row.get("COMPOSITE_SCORE", 0)),
        "ENTRY_PRICE": entry_price,
        "STOP_LOSS": stop_loss,
        "TARGET_PRICE": target_price,
        "RR_RATIO": rr_ratio,
        "ATR": round(atr, 4),
        "ATR_RATIO_PCT": round(atr_ratio, 4),
        "ADX": round(adx, 2),
        "VOLUME_RATIO": round(vol_ratio, 2),
        "DONCHIAN_UPPER": round(donchian_upper, 2),
        "BREAKOUT_READY": bool(price_above),
        "VOLUME_READY": bool(vol_ok),
        "ATR_READY": bool(atr_ok),
        "PRODUCT_TYPE": cfg["product_type"],
        "SIGNAL_TIME": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }

    # Setup but not triggered yet
    if not price_above:
        return {
            **base,
            "BREAKOUT_15M": False,
            "SIGNAL_STRENGTH": "SETUP",
            "SIGNAL_STATUS": "SETUP",
            "SETUP_REASON": "WAITING_BREAKOUT",
        }, "setup"

    strength = get_signal_strength(price_above, vol_ratio, atr_ratio, adx)

    if strength == "WEAK":
        return {
            **base,
            "BREAKOUT_15M": False,
            "SIGNAL_STRENGTH": "SETUP",
            "SIGNAL_STATUS": "SETUP",
            "SETUP_REASON": "WEAK_CONFIRMATION",
        }, "setup"

    if not vol_ok:
        return {
            **base,
            "BREAKOUT_15M": False,
            "SIGNAL_STRENGTH": "SETUP",
            "SIGNAL_STATUS": "SETUP",
            "SETUP_REASON": "LOW_VOLUME_CONFIRMATION",
        }, "setup"

    if not atr_ok:
        return {
            **base,
            "BREAKOUT_15M": False,
            "SIGNAL_STRENGTH": "SETUP",
            "SIGNAL_STATUS": "SETUP",
            "SETUP_REASON": "LOW_ATR",
        }, "setup"

    if rr_ratio < cfg["min_rr"]:
        return {
            **base,
            "BREAKOUT_15M": False,
            "SIGNAL_STRENGTH": "SETUP",
            "SIGNAL_STATUS": "SETUP",
            "SETUP_REASON": "LOW_RR",
        }, "setup"

    return {
        **base,
        "BREAKOUT_15M": True,
        "SIGNAL_STRENGTH": strength,
        "SIGNAL_STATUS": "CONFIRMED",
        "SETUP_REASON": "",
    }, "confirmed"


class SignalsStage(PipelineStage):
    name = "signals"

    def run(self, context):
        settings = context["settings"]
        store = context["store"]
        run_id = context["run_id"]
        control_panel = context.get("control_panel")

        candidates_df = context.get("candidates_df", pd.DataFrame())
        watchlist = context.get("watchlist", [])

        if candidates_df.empty:
            print("No candidates available for signal generation")
            context["signals_df"] = pd.DataFrame()
            empty_df = pd.DataFrame()
            empty_df.to_csv(settings.trade_signals_path, index=False)
            maybe_write_output(settings, control_panel, "ConfirmedSignals", empty_df)
            return

        watchlist_by_symbol = {w.symbol.upper(): w for w in watchlist}

        signals = []
        diag = {
            "no_data": 0,
            "setup": 0,
            "confirmed": 0,
        }

        for _, row in candidates_df.iterrows():
            symbol = str(row["symbol"]).strip().upper()
            watchlist_obj = watchlist_by_symbol.get(symbol)

            if watchlist_obj is None:
                eligible_modes = [settings.default_mode.upper()]
            else:
                eligible_modes = resolve_symbol_modes(watchlist_obj, settings)

            for mode in eligible_modes:
                result, reason = generate_signal(row, mode)
                diag[reason] = diag.get(reason, 0) + 1

                if result:
                    result["WATCHLIST_CONVICTION"] = getattr(watchlist_obj, "conviction", 2) if watchlist_obj else 2
                    result["SECTOR"] = getattr(watchlist_obj, "sector", "UNKNOWN") if watchlist_obj else "UNKNOWN"
                    result["PRIORITY"] = getattr(watchlist_obj, "priority", 0) if watchlist_obj else 0
                    signals.append(result)

        signals_df = pd.DataFrame(signals)
        context["signals_df"] = signals_df

        print("Signals diagnostics:")
        print(f"  rejected_no_data   = {diag['no_data']}")
        print(f"  setup_signals      = {diag['setup']}")
        print(f"  confirmed_signals  = {diag['confirmed']}")

        for _, row in signals_df.iterrows():
            store.insert_signal(
                run_id=run_id,
                symbol=str(row["symbol"]),
                strategy_mode=str(row.get("MODE", settings.default_mode.upper())),
                entry_price=float(row["ENTRY_PRICE"]),
                stop_loss=float(row["STOP_LOSS"]),
                target_price=float(row["TARGET_PRICE"]),
                rr_ratio=float(row["RR_RATIO"]),
                atr=float(row.get("ATR", 0.0)) if pd.notna(row.get("ATR")) else None,
                adx=float(row.get("ADX", 0.0)) if pd.notna(row.get("ADX")) else None,
                volume_ratio=float(row.get("VOLUME_RATIO", 0.0)) if pd.notna(row.get("VOLUME_RATIO")) else None,
                signal_strength=str(row.get("SIGNAL_STRENGTH", "")),
                product_type=str(row.get("PRODUCT_TYPE", "")),
                signal_time=str(row.get("SIGNAL_TIME", "")),
            )

        signals_df.to_csv(settings.trade_signals_path, index=False)
        maybe_write_output(settings, control_panel, "ConfirmedSignals", signals_df)
        print(f"Saved {len(signals_df)} signals (setup + confirmed) -> {settings.trade_signals_path}")