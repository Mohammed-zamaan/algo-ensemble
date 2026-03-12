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
    parser.add_argument("--bars-path", default="", help="CSV/parquet with symbol,date,open,high,low,close,volume")
    parser.add_argument("--data-source", default="file", choices=["file", "smartapi_cache"])
    parser.add_argument("--cache-dir", default="artifacts/backtest_cache")
    parser.add_argument("--smartapi-exchange", default="NSE")
    parser.add_argument("--smartapi-interval", default="ONE_DAY")
    parser.add_argument("--trade-mode", default="INTRADAY")
    parser.add_argument("--backtest-mode", default="daily", choices=["daily", "intraday_replay"])
    parser.add_argument("--intraday-interval", default="15m")
    parser.add_argument("--initial-capital", type=float, default=1_000_000.0)
    parser.add_argument("--benchmark-symbol", default="NIFTY50")
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
    parser.add_argument("--breakout-lookback", type=int, default=20)
    parser.add_argument("--avg-volume-lookback", type=int, default=20)
    parser.add_argument("--volume-multiplier", type=float, default=1.20)
    parser.add_argument("--max-hold-days", type=int, default=5)
    args = parser.parse_args()

    if args.data_source == "file" and not args.bars_path:
        raise SystemExit("--bars-path is required when --data-source=file")
    if args.data_source == "smartapi_cache" and not args.symbols and not args.watchlist_file:
        raise SystemExit("Provide --symbols or --watchlist-file when --data-source=smartapi_cache")

    watchlist_file = Path(args.watchlist_file) if args.watchlist_file else None
    bars_path = Path(args.bars_path) if args.bars_path else None
    config = BacktestConfig(
        start_date=args.start_date,
        end_date=args.end_date,
        bars_path=bars_path,
        trade_mode=args.trade_mode,
        backtest_mode=args.backtest_mode,
        intraday_interval=args.intraday_interval,
        initial_capital=args.initial_capital,
        benchmark_symbol=args.benchmark_symbol,
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
        breakout_lookback=args.breakout_lookback,
        avg_volume_lookback=args.avg_volume_lookback,
        volume_multiplier=args.volume_multiplier,
        max_hold_days=args.max_hold_days,
        data_source=args.data_source,
        cache_dir=Path(args.cache_dir),
        smartapi_exchange=args.smartapi_exchange,
        smartapi_interval=args.smartapi_interval,
    )
    result = BacktestEngine.from_config(config).run()
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
