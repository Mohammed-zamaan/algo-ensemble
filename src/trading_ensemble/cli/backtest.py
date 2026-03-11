from __future__ import annotations

import argparse
import json
from pathlib import Path

from trading_ensemble.backtest.config import BacktestConfig
from trading_ensemble.backtest.engine import BacktestEngine


def _parse_symbols(raw: str) -> tuple[str, ...]:
    if not raw.strip():
        return ()
    return tuple(s.strip().upper() for s in raw.split(",") if s.strip())


def main() -> None:
    parser = argparse.ArgumentParser(description="Phase 27 backtest CLI (daily baseline + intraday replay)")
    parser.add_argument("--start-date", required=True)
    parser.add_argument("--end-date", required=True)
    parser.add_argument("--bars-path", required=True, help="CSV/parquet with symbol,date,open,high,low,close,volume")
    parser.add_argument("--trade-mode", default="INTRADAY")
    parser.add_argument("--backtest-mode", default="daily", choices=["daily", "intraday_replay"])
    parser.add_argument("--intraday-interval", default="15m")
    parser.add_argument("--initial-capital", type=float, default=1_000_000.0)
    parser.add_argument("--universe", default="default")
    parser.add_argument("--symbols", default="", help="Comma-separated symbols")
    parser.add_argument("--watchlist-file", default="", help="Optional exported watchlist CSV/parquet")
    parser.add_argument("--output-dir", default="artifacts/backtest")
    parser.add_argument("--max-positions-per-day", type=int, default=5)
    parser.add_argument("--max-concurrent-positions", type=int, default=5)
    parser.add_argument("--slippage-bps", type=float, default=5.0)
    parser.add_argument("--stop-loss-pct", type=float, default=0.01)
    parser.add_argument("--target-pct", type=float, default=0.02)
    parser.add_argument("--position-notional-frac", type=float, default=0.10)
    args = parser.parse_args()

    watchlist_file = Path(args.watchlist_file) if args.watchlist_file else None
    config = BacktestConfig(
        start_date=args.start_date,
        end_date=args.end_date,
        bars_path=Path(args.bars_path),
        trade_mode=args.trade_mode,
        backtest_mode=args.backtest_mode,
        intraday_interval=args.intraday_interval,
        initial_capital=args.initial_capital,
        universe_name=args.universe,
        symbols=_parse_symbols(args.symbols),
        watchlist_file=watchlist_file,
        output_dir=Path(args.output_dir),
        max_positions_per_day=args.max_positions_per_day,
        max_concurrent_positions=args.max_concurrent_positions,
        slippage_bps=args.slippage_bps,
        stop_loss_pct=args.stop_loss_pct,
        target_pct=args.target_pct,
        position_notional_frac=args.position_notional_frac,
    )
    result = BacktestEngine.from_config(config).run()
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
