from __future__ import annotations

from pathlib import Path
import math

import pandas as pd

from trading_ensemble.backtest.config import BacktestConfig
from trading_ensemble.backtest.data_provider import (
    FileHistoricalDataProvider,
    HistoricalDataRequest,
    fetch_and_cache_smartapi_daily_bars,
)
from trading_ensemble.backtest.metrics import compute_backtest_metrics
from trading_ensemble.backtest.portfolio import PortfolioEngine
from trading_ensemble.backtest.recorder import BacktestRecorder
from trading_ensemble.backtest.report import write_backtest_outputs
from trading_ensemble.backtest.universe import load_universe


class BacktestEngine:
    def __init__(self, config: BacktestConfig):
        self.config = config
        self.provider = self._build_provider(config)
        self.recorder = BacktestRecorder()
        self.portfolio = PortfolioEngine(
            cash=float(config.initial_capital),
            max_concurrent_positions=int(config.max_concurrent_positions),
            max_positions_per_day=int(config.max_positions_per_day),
        )

    @classmethod
    def from_config(cls, config: BacktestConfig):
        return cls(config)

    def _build_provider(self, config: BacktestConfig) -> FileHistoricalDataProvider:
        if config.data_source == "smartapi_cache":
            symbols = list(config.symbols)
            if not symbols and config.watchlist_file is not None:
                symbols = load_universe(config.universe_name, [], config.watchlist_file)
            combined_path = fetch_and_cache_smartapi_daily_bars(
                symbols=symbols,
                start_date=config.start_date,
                end_date=config.end_date,
                cache_dir=config.cache_dir,
                exchange=config.smartapi_exchange,
                interval=config.smartapi_interval,
            )
            return FileHistoricalDataProvider(combined_path)

        if config.bars_path is None:
            raise ValueError("bars_path is required when data_source='file'")
        return FileHistoricalDataProvider(config.bars_path)

    def run(self):
        if self.config.backtest_mode != "daily":
            raise NotImplementedError("intraday_replay is not implemented yet; daily baseline is the active mode")
        return self._run_daily_baseline()

    def _resolve_universe(self) -> list[str]:
        universe = load_universe(
            self.config.universe_name,
            list(self.config.symbols),
            self.config.watchlist_file,
        )
        if not universe:
            universe = self.provider.list_symbols()
        return sorted(set(universe))

    def _prepare_symbol_frames(self, symbols: list[str]) -> dict[str, pd.DataFrame]:
        frames: dict[str, pd.DataFrame] = {}
        for symbol in symbols:
            df = self.provider.get_bars(HistoricalDataRequest(symbol=symbol, end_date=self.config.end_date))
            if df.empty:
                continue
            df = df.sort_values("date").reset_index(drop=True).copy()
            df["prev_breakout_high"] = (
                df["high"].rolling(window=int(self.config.breakout_lookback), min_periods=int(self.config.breakout_lookback)).max().shift(1)
            )
            df["prev_avg_volume"] = (
                df["volume"].rolling(window=int(self.config.avg_volume_lookback), min_periods=int(self.config.avg_volume_lookback)).mean().shift(1)
            )
            frames[symbol] = df
        return frames

    @staticmethod
    def _ts(value: pd.Timestamp, suffix: str) -> str:
        return f"{value.strftime('%Y-%m-%d')}T{suffix}+05:30"

    def _entry_price(self, open_price: float) -> float:
        return float(open_price) * (1.0 + float(self.config.slippage_bps) / 10000.0)

    def _exit_price(self, reference_price: float) -> float:
        return float(reference_price) * (1.0 - float(self.config.slippage_bps) / 10000.0)

    def _run_daily_baseline(self) -> dict:
        start_ts = pd.Timestamp(self.config.start_date).normalize()
        end_ts = pd.Timestamp(self.config.end_date).normalize()
        symbols = self._resolve_universe()
        frames = self._prepare_symbol_frames(symbols)
        dates = self.provider.trading_dates(self.config.start_date, self.config.end_date)
        if not dates:
            raise ValueError("No trading dates available in the requested range")

        pending_entries: list[dict] = []
        blocked_reasons: dict[str, int] = {}
        session_rows = 0

        for current_date in dates:
            current_date = pd.Timestamp(current_date).normalize()
            if current_date < start_ts or current_date > end_ts:
                continue
            session_date = current_date.strftime("%Y-%m-%d")
            self.portfolio.reset_day(session_date)
            session_rows += 1

            todays_pending = [p for p in pending_entries if p["entry_date"] == current_date]
            pending_entries = [p for p in pending_entries if p["entry_date"] != current_date]
            for pending in todays_pending:
                symbol = pending["symbol"]
                frame = frames.get(symbol)
                if frame is None:
                    blocked_reasons["missing_symbol_frame"] = blocked_reasons.get("missing_symbol_frame", 0) + 1
                    continue
                row_match = frame[frame["date"] == current_date]
                if row_match.empty:
                    blocked_reasons["missing_entry_bar"] = blocked_reasons.get("missing_entry_bar", 0) + 1
                    continue
                row = row_match.iloc[0]
                if symbol in self.portfolio.positions:
                    blocked_reasons["already_in_position"] = blocked_reasons.get("already_in_position", 0) + 1
                    continue
                if not self.portfolio.can_open(session_date):
                    blocked_reasons["position_limit"] = blocked_reasons.get("position_limit", 0) + 1
                    continue
                entry_price = self._entry_price(float(row["open"]))
                budget = float(self.portfolio.cash) * float(self.config.position_notional_frac)
                qty = math.floor(budget / entry_price) if entry_price > 0 else 0
                if qty <= 0:
                    blocked_reasons["insufficient_cash"] = blocked_reasons.get("insufficient_cash", 0) + 1
                    continue
                opened = self.portfolio.open_position(
                    symbol=symbol,
                    session_date=session_date,
                    qty=qty,
                    entry_price=entry_price,
                    trigger_ts=pending["trigger_ts"],
                    promoted_ts=self._ts(current_date, "09:15:00"),
                )
                if opened is None:
                    blocked_reasons["open_rejected"] = blocked_reasons.get("open_rejected", 0) + 1
                    continue
                self.recorder.record_order(
                    {
                        "session_date": session_date,
                        "symbol": symbol,
                        "side": "BUY",
                        "qty": qty,
                        "order_price": round(entry_price, 6),
                        "trigger_ts": pending["trigger_ts"],
                        "submitted_ts": self._ts(current_date, "09:15:00"),
                        "status": "FILLED",
                    }
                )

            price_map: dict[str, float] = {}
            for symbol in list(self.portfolio.positions.keys()):
                frame = frames.get(symbol)
                if frame is None:
                    continue
                row_match = frame[frame["date"] == current_date]
                if row_match.empty:
                    continue
                row = row_match.iloc[0]
                price_map[symbol] = float(row["close"])
                position = self.portfolio.positions[symbol]
                entry_price = float(position["entry_price"])
                stop_price = entry_price * (1.0 - float(self.config.stop_loss_pct))
                target_price = entry_price * (1.0 + float(self.config.target_pct))
                self.portfolio.increment_holding_period(symbol)
                bars_held = int(self.portfolio.positions[symbol].get("bars_held", 0))

                exit_reason = None
                exit_price = None
                if float(row["low"]) <= stop_price:
                    exit_reason = "stop_loss"
                    exit_price = self._exit_price(stop_price)
                elif float(row["high"]) >= target_price:
                    exit_reason = "target"
                    exit_price = self._exit_price(target_price)
                elif bars_held >= int(self.config.max_hold_days):
                    exit_reason = "max_hold"
                    exit_price = self._exit_price(float(row["close"]))

                if exit_reason is not None and exit_price is not None:
                    trade = self.portfolio.close_position(
                        symbol=symbol,
                        exit_price=exit_price,
                        exit_reason=exit_reason,
                        exit_ts=self._ts(current_date, "15:25:00"),
                    )
                    if trade is not None:
                        self.recorder.record_trade(trade)
                        self.recorder.record_order(
                            {
                                "session_date": session_date,
                                "symbol": symbol,
                                "side": "SELL",
                                "qty": trade["qty"],
                                "order_price": round(exit_price, 6),
                                "submitted_ts": self._ts(current_date, "15:25:00"),
                                "status": "FILLED",
                                "exit_reason": exit_reason,
                            }
                        )
                        price_map.pop(symbol, None)

            next_dates = [d for d in dates if d > current_date]
            next_date = pd.Timestamp(next_dates[0]).normalize() if next_dates else None
            for symbol, frame in frames.items():
                row_match = frame[frame["date"] == current_date]
                if row_match.empty:
                    continue
                row = row_match.iloc[0]
                price_map.setdefault(symbol, float(row["close"]))

                breakout_high = row.get("prev_breakout_high")
                avg_volume = row.get("prev_avg_volume")
                if pd.isna(breakout_high) or pd.isna(avg_volume):
                    continue
                signal_ok = bool(
                    float(row["close"]) > float(breakout_high)
                    and float(row["volume"]) >= float(avg_volume) * float(self.config.volume_multiplier)
                )
                if not signal_ok or next_date is None:
                    continue
                if symbol in self.portfolio.positions:
                    blocked_reasons["signal_while_open"] = blocked_reasons.get("signal_while_open", 0) + 1
                    continue
                if any(p["symbol"] == symbol for p in pending_entries):
                    blocked_reasons["duplicate_pending_signal"] = blocked_reasons.get("duplicate_pending_signal", 0) + 1
                    continue
                trigger_ts = self._ts(current_date, "15:20:00")
                promotion_row = {
                    "session_date": session_date,
                    "symbol": symbol,
                    "signal_date": session_date,
                    "entry_date": next_date.strftime("%Y-%m-%d"),
                    "close": round(float(row["close"]), 6),
                    "prev_breakout_high": round(float(breakout_high), 6),
                    "volume": float(row["volume"]),
                    "prev_avg_volume": round(float(avg_volume), 6),
                    "trigger_ts": trigger_ts,
                }
                pending_entries.append(
                    {
                        "symbol": symbol,
                        "entry_date": next_date,
                        "trigger_ts": trigger_ts,
                    }
                )
                self.recorder.record_promotion(promotion_row)

            snapshot = self.portfolio.snapshot(price_map)
            self.recorder.record_equity(
                {
                    "date": session_date,
                    "cash": round(snapshot["cash"], 6),
                    "market_value": round(snapshot["market_value"], 6),
                    "equity": round(snapshot["equity"], 6),
                    "open_positions": len(self.portfolio.positions),
                    "pending_entries": len(pending_entries),
                }
            )

        last_date = pd.Timestamp(dates[-1]).normalize()
        last_session_date = last_date.strftime("%Y-%m-%d")
        for symbol in list(self.portfolio.positions.keys()):
            frame = frames.get(symbol)
            if frame is None:
                continue
            row_match = frame[frame["date"] <= end_ts]
            if row_match.empty:
                continue
            row = row_match.iloc[-1]
            trade = self.portfolio.close_position(
                symbol=symbol,
                exit_price=self._exit_price(float(row["close"])),
                exit_reason="forced_eod",
                exit_ts=self._ts(pd.Timestamp(row["date"]).normalize(), "15:29:00"),
            )
            if trade is not None:
                self.recorder.record_trade(trade)
                self.recorder.record_order(
                    {
                        "session_date": last_session_date,
                        "symbol": symbol,
                        "side": "SELL",
                        "qty": trade["qty"],
                        "order_price": trade["exit_price"],
                        "submitted_ts": trade["exit_ts"],
                        "status": "FILLED",
                        "exit_reason": "forced_eod",
                    }
                )

        trades_df = self.recorder.trades_df()
        equity_df = self.recorder.equity_df()
        promoted_df = self.recorder.promotions_df()
        orders_df = self.recorder.orders_df()
        metrics = compute_backtest_metrics(
            trades_df=trades_df,
            equity_df=equity_df,
            initial_capital=float(self.config.initial_capital),
        )
        summary = {
            "status": "ok",
            "backtest_mode": self.config.backtest_mode,
            "trade_mode": self.config.trade_mode,
            "data_source": self.config.data_source,
            "start_date": self.config.start_date,
            "end_date": self.config.end_date,
            "bars_path": str(self.config.bars_path) if self.config.bars_path is not None else None,
            "cache_dir": str(self.config.cache_dir),
            "output_dir": str(self.config.output_dir),
            "symbols_requested": len(self.config.symbols),
            "symbols_tested": len(frames),
            "sessions": session_rows,
            "pending_entries_left": len(pending_entries),
            "blocked_reasons": blocked_reasons,
        }
        outputs = write_backtest_outputs(
            output_dir=Path(self.config.output_dir),
            trades_df=trades_df,
            equity_df=equity_df,
            promoted_df=promoted_df,
            orders_df=orders_df,
            summary=summary,
            metrics=metrics,
        )
        return {**summary, **metrics, **outputs}
