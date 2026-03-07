from __future__ import annotations

from datetime import datetime

from trading_ensemble.config.settings import Settings
from trading_ensemble.state.store import StateStore


def run_pipeline() -> None:
    settings = Settings.from_env()
    settings.validate_for_runtime()

    store = StateStore(settings.state_db_path)
    store.initialize()

    print("=" * 60)
    print("TRADING ENSEMBLE PIPELINE")
    print("=" * 60)
    print(f"Started at: {datetime.utcnow().isoformat()}Z")
    print(f"Mode: {settings.trade_mode}")
    print(f"Paper trade: {settings.paper_trade}")
    print(f"Capital base: {settings.total_account_capital}")
    print(f"State DB: {settings.state_db_path}")
    print("Pipeline foundation initialized successfully.")
    print("=" * 60)


if __name__ == "__main__":
    run_pipeline()