from __future__ import annotations

import argparse
import json
from pathlib import Path

from trading_ensemble.backtest.config import BacktestConfig
from trading_ensemble.backtest.walkforward import WalkForwardWindow, run_walkforward


def _parse_windows(values: list[str]) -> list[WalkForwardWindow]:
    windows: list[WalkForwardWindow] = []
    for raw in values:
        parts = raw.split(":")
        if len(parts) != 4:
            raise ValueError(f"Invalid window format: {raw}")
        windows.append(
            WalkForwardWindow(
                train_start=parts[0],
                train_end=parts[1],
                test_start=parts[2],
                test_end=parts[3],
            )
        )
    return windows


def main() -> None:
    parser = argparse.ArgumentParser(description="Phase 27 walk-forward backtest CLI")
    parser.add_argument("--bars-path", required=True)
    parser.add_argument("--trade-mode", default="INTRADAY")
    parser.add_argument("--backtest-mode", default="daily", choices=["daily", "intraday_replay"])
    parser.add_argument("--intraday-interval", default="15m")
    parser.add_argument("--initial-capital", type=float, default=1_000_000.0)
    parser.add_argument("--window", action="append", required=True, help="train_start:train_end:test_start:test_end")
    parser.add_argument("--slippage-bps", type=float, default=5.0)
    parser.add_argument("--stop-loss-pct", type=float, default=0.01)
    parser.add_argument("--target-pct", type=float, default=0.02)
    args = parser.parse_args()

    windows = _parse_windows(args.window)
    base_cfg = BacktestConfig(
        start_date=windows[0].test_start,
        end_date=windows[-1].test_end,
        bars_path=Path(args.bars_path),
        trade_mode=args.trade_mode,
        backtest_mode=args.backtest_mode,
        intraday_interval=args.intraday_interval,
        initial_capital=args.initial_capital,
        slippage_bps=args.slippage_bps,
        stop_loss_pct=args.stop_loss_pct,
        target_pct=args.target_pct,
    )
    result = run_walkforward(base_cfg, windows)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
