from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass
class BacktestConfig:
    start_date: str
    end_date: str
    bars_path: Path | None = None

    trade_mode: str = "INTRADAY"
    backtest_mode: str = "daily"
    intraday_interval: str = "15m"

    initial_capital: float = 1_000_000.0
    benchmark_symbol: str = "NIFTY50"
    universe_name: str = "default"
    symbols: tuple[str, ...] = ()
    watchlist_file: Path | None = None
    output_dir: Path = Path("artifacts/backtest")

    max_positions_per_day: int = 5
    max_concurrent_positions: int = 5
    slippage_bps: float = 5.0
    stop_loss_pct: float = 0.01
    target_pct: float = 0.02
    position_notional_frac: float = 0.10

    breakout_lookback: int = 20
    avg_volume_lookback: int = 20
    volume_multiplier: float = 1.20
    max_hold_days: int = 5

    data_source: str = "file"
    cache_dir: Path = Path("artifacts/backtest_cache")
    smartapi_exchange: str = "NSE"
    smartapi_interval: str = "ONE_DAY"
