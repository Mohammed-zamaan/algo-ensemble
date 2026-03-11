from dataclasses import dataclass
from pathlib import Path
from typing import Tuple


@dataclass
class BacktestConfig:
    start_date: str
    end_date: str
    bars_path: Path
    trade_mode: str = "INTRADAY"
    backtest_mode: str = "daily"
    intraday_interval: str = "15m"
    initial_capital: float = 1_000_000.0
    universe_name: str = "default"
    symbols: Tuple[str, ...] = ()
    watchlist_file: Path | None = None
    output_dir: Path | None = None
    max_positions_per_day: int = 5
    max_concurrent_positions: int = 5
    slippage_bps: float = 5.0
    stop_loss_pct: float = 0.01
    target_pct: float = 0.02
    position_notional_frac: float = 0.10