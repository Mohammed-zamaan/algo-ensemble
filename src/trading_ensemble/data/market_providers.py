from __future__ import annotations

from dataclasses import dataclass
from datetime import timedelta
from typing import Any

import pandas as pd

from trading_ensemble.core.timeutils import now_ist
from trading_ensemble.data.smartapi_client import (
    fetch_candles_live,
    load_scrip_master,
    login_from_env,
    resolve_symbol_to_token_offline,
)


MODE_FEED_CONFIG: dict[str, dict[str, Any]] = {
    "INTRADAY": {
        "interval": "FIFTEEN_MINUTE",
        "lookback_days": 5,
        "min_rows": 40,
        "max_age_minutes": 20,
    },
    "SWING": {
        "interval": "ONE_DAY",
        "lookback_days": 120,
        "min_rows": 30,
        "max_age_minutes": 60 * 36,
    },
    "POSITIONAL": {
        "interval": "ONE_DAY",
        "lookback_days": 240,
        "min_rows": 70,
        "max_age_minutes": 60 * 36,
    },
}


@dataclass
class ExecutionMarketDataProvider:
    smart: Any
    scrip_df: pd.DataFrame

    @classmethod
    def from_env(cls) -> "ExecutionMarketDataProvider":
        session = login_from_env()
        return cls(smart=session.smart, scrip_df=load_scrip_master())

    def fetch_mode_candles(self, symbol: str, mode: str) -> pd.DataFrame | None:
        cfg = MODE_FEED_CONFIG.get(str(mode).upper(), MODE_FEED_CONFIG["INTRADAY"])
        _, token = resolve_symbol_to_token_offline(symbol, exchange="NSE", scrip_df=self.scrip_df)

        now = now_ist()
        start = (now - timedelta(days=int(cfg["lookback_days"]))).strftime("%Y-%m-%d %H%M")
        end = now.strftime("%Y-%m-%d %H%M")

        df = fetch_candles_live(
            self.smart,
            exchange="NSE",
            symbol_token=str(token),
            interval=str(cfg["interval"]),
            start=start,
            end=end,
        )
        if df is None or df.empty:
            return None

        work = df.copy()
        work.columns = [str(c).strip().lower() for c in work.columns]
        required = ["datetime", "open", "high", "low", "close", "volume"]
        for col in required:
            if col not in work.columns:
                return None

        work = work[required].dropna().reset_index(drop=True)
        if len(work) < int(cfg["min_rows"]):
            return None

        latest_ts = work["datetime"].iloc[-1]
        freshness_mins = max(0.0, (now - latest_ts).total_seconds() / 60.0)
        if freshness_mins > float(cfg["max_age_minutes"]):
            return None

        return work
