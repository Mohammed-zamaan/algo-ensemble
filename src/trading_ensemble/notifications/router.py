from __future__ import annotations

from pathlib import Path

import pandas as pd

from trading_ensemble.core.timeutils import fmt_ist, now_ist
from trading_ensemble.notifications.telegram import send_telegram_message


CACHE_DIR = Path("state")
CACHE_DIR.mkdir(parents=True, exist_ok=True)
ALERT_CACHE_FILE = CACHE_DIR / "telegram_alert_cache.csv"


def _load_cache() -> pd.DataFrame:
    if ALERT_CACHE_FILE.exists():
        try:
            return pd.read_csv(ALERT_CACHE_FILE)
        except Exception:
            pass
    return pd.DataFrame(columns=["alert_key", "sent_at"])


def _save_cache(df: pd.DataFrame) -> None:
    df.to_csv(ALERT_CACHE_FILE, index=False)


def _already_sent(alert_key: str) -> bool:
    cache = _load_cache()
    if cache.empty:
        return False
    return alert_key in set(cache["alert_key"].astype(str))


def _mark_sent(alert_key: str) -> None:
    cache = _load_cache()
    new_row = pd.DataFrame(
        [{"alert_key": alert_key, "sent_at": fmt_ist()}]
    )
    cache = pd.concat([cache, new_row], ignore_index=True)
    _save_cache(cache)


def _day_key() -> str:
    return now_ist().date().isoformat()


def send_near_trigger_alerts(alerts_df: pd.DataFrame) -> None:
    if alerts_df is None or alerts_df.empty:
        return

    for _, row in alerts_df.iterrows():
        symbol = str(row.get("symbol", "UNKNOWN"))
        mode = str(row.get("MODE", "UNKNOWN"))
        alert_level = str(row.get("ALERT_LEVEL", "WARM"))
        score = row.get("readiness_score", "")
        breakout_gap = row.get("breakout_gap_pct", "")
        volume_gap = row.get("volume_gap_pct", "")

        alert_key = f"{_day_key()}|near_trigger|{symbol}|{mode}"
        if _already_sent(alert_key):
            continue

        text = (
            f"NEAR TRIGGER [{alert_level}]\n"
            f"{symbol} [{mode}]\n"
            f"Readiness: {score}\n"
            f"Breakout gap: {breakout_gap}%\n"
            f"Volume gap: {volume_gap}%"
        )

        if send_telegram_message(text):
            _mark_sent(alert_key)


def send_promotion_alerts(promoted_df: pd.DataFrame) -> None:
    if promoted_df is None or promoted_df.empty:
        return

    for _, row in promoted_df.iterrows():
        symbol = str(row.get("symbol", "UNKNOWN"))
        mode = str(row.get("MODE", "UNKNOWN"))
        latest_close = row.get("latest_close", "")
        donchian = row.get("live_donchian_upper", "")
        vol_ratio = row.get("live_volume_ratio", "")

        alert_key = f"{_day_key()}|promoted|{symbol}|{mode}"
        if _already_sent(alert_key):
            continue

        text = (
            f"PROMOTED\n"
            f"{symbol} [{mode}]\n"
            f"Live close: {latest_close}\n"
            f"Donchian: {donchian}\n"
            f"Volume ratio: {vol_ratio}"
        )

        if send_telegram_message(text):
            _mark_sent(alert_key)


def send_order_alerts(orders_df: pd.DataFrame) -> None:
    if orders_df is None or orders_df.empty:
        return

    for _, row in orders_df.iterrows():
        symbol = str(row.get("symbol", "UNKNOWN"))
        mode = str(row.get("MODE", "UNKNOWN"))
        qty = row.get("QUANTITY", "")
        entry = row.get("ENTRY_PRICE", "")
        sl = row.get("STOP_LOSS", "")
        tgt = row.get("TARGET_PRICE", "")
        status = row.get("STATUS", "")

        alert_key = f"{_day_key()}|order|{symbol}|{mode}"
        if _already_sent(alert_key):
            continue

        text = (
            f"ORDER APPROVED\n"
            f"{symbol} [{mode}]\n"
            f"Qty: {qty}\n"
            f"Entry: {entry}\n"
            f"SL: {sl}\n"
            f"Target: {tgt}\n"
            f"Status: {status}"
        )

        if send_telegram_message(text):
            _mark_sent(alert_key)


def send_execution_alerts(execution_df: pd.DataFrame) -> None:
    if execution_df is None or execution_df.empty:
        return

    for _, row in execution_df.iterrows():
        symbol = str(row.get("symbol", "UNKNOWN"))
        mode = str(row.get("MODE", "UNKNOWN"))
        qty = row.get("QUANTITY", "")
        fill = row.get("FILL_PRICE", "")
        status = row.get("STATUS", "")

        alert_key = f"{_day_key()}|execution|{symbol}|{mode}"
        if _already_sent(alert_key):
            continue

        text = (
            f"EXECUTED\n"
            f"{symbol} [{mode}]\n"
            f"Status: {status}\n"
            f"Qty: {qty}\n"
            f"Fill: {fill}"
        )

        if send_telegram_message(text):
            _mark_sent(alert_key)
