from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd


@dataclass
class BacktestRecorder:
    """Captures backtest events and tabular outputs."""

    events: list[dict] = field(default_factory=list)
    trades: list[dict] = field(default_factory=list)
    equity_curve: list[dict] = field(default_factory=list)
    promotions: list[dict] = field(default_factory=list)
    orders: list[dict] = field(default_factory=list)

    def record(self, event: dict) -> None:
        self.events.append(event)

    def record_trade(self, trade: dict) -> None:
        self.trades.append(trade)

    def record_equity(self, row: dict) -> None:
        self.equity_curve.append(row)

    def record_promotion(self, row: dict) -> None:
        self.promotions.append(row)

    def record_order(self, row: dict) -> None:
        self.orders.append(row)

    def trades_df(self) -> pd.DataFrame:
        return pd.DataFrame(self.trades)

    def equity_df(self) -> pd.DataFrame:
        return pd.DataFrame(self.equity_curve)

    def promotions_df(self) -> pd.DataFrame:
        return pd.DataFrame(self.promotions)

    def orders_df(self) -> pd.DataFrame:
        return pd.DataFrame(self.orders)
