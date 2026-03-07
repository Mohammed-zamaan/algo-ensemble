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
    enabled_modes: str
    default_mode: str

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

    watchlist_source: str
    watchlist_gsheet_id: str
    watchlist_gsheet_gid: str
    watchlist_gsheet_tab: str
    google_service_account_json: str

    watchlist_csv_path: str
    trade_candidates_path: str
    trade_signals_path: str
    trade_orders_path: str
    execution_log_path: str
    paper_trades_path: str

    regime_override: str

    @classmethod
    def from_env(cls) -> "Settings":
        return cls(
            trade_mode=_get_str("TRADE_MODE", "INTRADAY"),
            enabled_modes=_get_str("ENABLED_MODES", "INTRADAY,SWING,POSITIONAL"),
            default_mode=_get_str("DEFAULT_MODE", "INTRADAY"),

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

            watchlist_source=_get_str("WATCHLIST_SOURCE", "csv"),
            watchlist_gsheet_id=_get_str("WATCHLIST_GSHEET_ID", ""),
            watchlist_gsheet_gid=_get_str("WATCHLIST_GSHEET_GID", "0"),
            watchlist_gsheet_tab=_get_str("WATCHLIST_GSHEET_TAB", "MasterWatchlist"),
            google_service_account_json=_get_str(
                "GOOGLE_SERVICE_ACCOUNT_JSON",
                "credentials/service-account.json",
            ),

            watchlist_csv_path=_get_str("WATCHLIST_CSV_PATH", "watchlist.csv"),
            trade_candidates_path=_get_str("TRADE_CANDIDATES_PATH", "trade_candidates.csv"),
            trade_signals_path=_get_str("TRADE_SIGNALS_PATH", "trade_signals.csv"),
            trade_orders_path=_get_str("TRADE_ORDERS_PATH", "trade_orders.csv"),
            execution_log_path=_get_str("EXECUTION_LOG_PATH", "state/execution_log.csv"),
            paper_trades_path=_get_str("PAPER_TRADES_PATH", "state/paper_trades.csv"),

            regime_override=_get_str("REGIME_OVERRIDE", ""),
        )

    def validate_for_runtime(self) -> None:
        if self.total_account_capital <= 0:
            raise ValueError("TOTAL_ACCOUNT_CAPITAL must be > 0")

        if self.watchlist_source.lower() == "google_sheets":
            if not self.watchlist_gsheet_id:
                raise ValueError("WATCHLIST_GSHEET_ID is required when WATCHLIST_SOURCE=google_sheets")
            if not self.watchlist_gsheet_tab:
                raise ValueError("WATCHLIST_GSHEET_TAB is required when WATCHLIST_SOURCE=google_sheets")
            if not self.google_service_account_json:
                raise ValueError("GOOGLE_SERVICE_ACCOUNT_JSON is required when WATCHLIST_SOURCE=google_sheets")

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