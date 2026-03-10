from __future__ import annotations

import os
import requests
from dotenv import load_dotenv

load_dotenv()


def telegram_enabled() -> bool:
    value = os.getenv("TELEGRAM_ALERTS_ENABLED", "false").strip().lower()
    return value in {"1", "true", "yes", "y", "on"}


def get_telegram_config() -> tuple[str, str] | tuple[None, None]:
    bot_token = os.getenv("TELEGRAM_BOT_TOKEN", "").strip()
    chat_id = os.getenv("TELEGRAM_CHAT_ID", "").strip()
    if not bot_token or not chat_id:
        return None, None
    return bot_token, chat_id


def send_telegram_message(text: str) -> bool:
    if not telegram_enabled():
        return False

    bot_token, chat_id = get_telegram_config()
    if not bot_token or not chat_id:
        return False

    url = f"https://api.telegram.org/bot{bot_token}/sendMessage"
    payload = {
        "chat_id": chat_id,
        "text": text,
    }

    try:
        response = requests.post(url, data=payload, timeout=10)
        return response.ok
    except Exception:
        return False