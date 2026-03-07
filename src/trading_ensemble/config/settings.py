from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv


load_dotenv()


def _get_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "y", "on"}


def _get_float(name: str, default: float) -> float:
    value = os.getenv(name)
    if value is None or value.strip() == "":
        return default
    return float(value)


def _get_str(name: str, default: str) -> str:
    value = os.getenv(name)
    if value is None or value.strip() == "":
        return default
    return value.strip()


@dataclass(frozen=True)
class Settings:
    trade_mode: str
    paper_trade: bool
    system_trading_enabled: bool
    total_account_capital: float
    state_db_path: Path

    angel_api_key: str
    angel_client_code: str
    angel_pin: str
    angel_totp_secret: str

    comet_api_key: str
    comet_workspace: str
    comet_project_name: str

    watchlist_gsheet_id: str
    watchlist_gsheet_gid: str

    regime_override: str

    @classmethod
    def from_env(cls) -> "Settings":
        return cls(
            trade_mode=_get_str("TRADE_MODE", "INTRADAY"),
            paper_trade=_get_bool("PAPER_TRADE", True),
            system_trading_enabled=_get_bool("SYSTEM_TRADING_ENABLED", False),
            total_account_capital=_get_float("TOTAL_ACCOUNT_CAPITAL", 100000.0),
            state_db_path=Path(_get_str("STATE_DB_PATH", "state/trading.db")),

            angel_api_key=_get_str("ANGEL_API_KEY", ""),
            angel_client_code=_get_str("ANGEL_CLIENT_CODE", ""),
            angel_pin=_get_str("ANGEL_PIN", ""),
            angel_totp_secret=_get_str("ANGEL_TOTP_SECRET", ""),

            comet_api_key=_get_str("COMET_API_KEY", ""),
            comet_workspace=_get_str("COMET_WORKSPACE", ""),
            comet_project_name=_get_str("COMET_PROJECT_NAME", "stock-screener"),

            watchlist_gsheet_id=_get_str("WATCHLIST_GSHEET_ID", ""),
            watchlist_gsheet_gid=_get_str("WATCHLIST_GSHEET_GID", "0"),

            regime_override=_get_str("REGIME_OVERRIDE", ""),
        )

    def validate_for_runtime(self) -> None:
        if self.total_account_capital <= 0:
            raise ValueError("TOTAL_ACCOUNT_CAPITAL must be > 0")

        if not self.paper_trade:
            required = {
                "ANGEL_API_KEY": self.angel_api_key,
                "ANGEL_CLIENT_CODE": self.angel_client_code,
                "ANGEL_PIN": self.angel_pin,
                "ANGEL_TOTP_SECRET": self.angel_totp_secret,
            }
            missing = [key for key, value in required.items() if not value]
            if missing:
                raise ValueError(
                    f"Missing required live trading environment variables: {', '.join(missing)}"
                )