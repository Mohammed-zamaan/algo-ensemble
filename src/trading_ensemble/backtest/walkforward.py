from __future__ import annotations

from dataclasses import dataclass

from trading_ensemble.backtest.config import BacktestConfig
from trading_ensemble.backtest.engine import BacktestEngine


@dataclass(frozen=True)
class WalkForwardWindow:
    train_start: str
    train_end: str
    test_start: str
    test_end: str


def run_walkforward(config: BacktestConfig, windows: list[WalkForwardWindow]) -> list[dict]:
    """Walk-forward skeleton.

    TODO: separate train/test optimization and out-of-sample evaluation.
    """

    outputs: list[dict] = []
    for window in windows:
        window_cfg = BacktestConfig(
            start_date=window.test_start,
            end_date=window.test_end,
            bars_path=config.bars_path,
            trade_mode=config.trade_mode,
            backtest_mode=config.backtest_mode,
            intraday_interval=config.intraday_interval,
            initial_capital=config.initial_capital,
            benchmark_symbol=config.benchmark_symbol,
            universe_name=config.universe_name,
            output_dir=config.output_dir,
            symbols=config.symbols,
            watchlist_file=config.watchlist_file,
            max_positions_per_day=config.max_positions_per_day,
            max_concurrent_positions=config.max_concurrent_positions,
            slippage_bps=config.slippage_bps,
            stop_loss_pct=config.stop_loss_pct,
            target_pct=config.target_pct,
            position_notional_frac=config.position_notional_frac,
            breakout_lookback=config.breakout_lookback,
            avg_volume_lookback=config.avg_volume_lookback,
            volume_multiplier=config.volume_multiplier,
            max_hold_days=config.max_hold_days,
            data_source=config.data_source,
            cache_dir=config.cache_dir,
            smartapi_exchange=config.smartapi_exchange,
            smartapi_interval=config.smartapi_interval,
        )
        outputs.append(BacktestEngine.from_config(window_cfg).run())
    return outputs
