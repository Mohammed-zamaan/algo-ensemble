from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class PortfolioEngine:
    """Portfolio accounting for daily and intraday replay modes."""

    cash: float
    max_concurrent_positions: int = 5
    max_positions_per_day: int = 5
    positions: dict[str, dict] = field(default_factory=dict)
    opened_today: int = 0
    current_session_date: str | None = None

    def reset_day(self, session_date: str) -> None:
        if self.current_session_date != session_date:
            self.opened_today = 0
            self.current_session_date = session_date

    def can_open(self, session_date: str) -> bool:
        self.reset_day(session_date)
        if len(self.positions) >= int(self.max_concurrent_positions):
            return False
        return self.opened_today < int(self.max_positions_per_day)

    def open_position(
        self,
        *,
        symbol: str,
        session_date: str,
        qty: float,
        entry_price: float,
        trigger_ts: str,
        promoted_ts: str,
    ) -> dict | None:
        if symbol in self.positions:
            return None
        if not self.can_open(session_date):
            return None
        qty = float(qty)
        entry_price = float(entry_price)
        notional = qty * entry_price
        if qty <= 0 or entry_price <= 0 or notional > float(self.cash):
            return None
        position = {
            "symbol": symbol,
            "qty": qty,
            "entry_price": entry_price,
            "entry_notional": notional,
            "entry_ts": str(promoted_ts),
            "trigger_ts": str(trigger_ts),
            "promoted_ts": str(promoted_ts),
            "session_date": session_date,
            "bars_held": 0,
        }
        self.cash -= notional
        self.positions[symbol] = position
        self.opened_today += 1
        return position

    def increment_holding_period(self, symbol: str) -> None:
        position = self.positions.get(symbol)
        if position is not None:
            position["bars_held"] = int(position.get("bars_held", 0)) + 1

    def close_position(self, *, symbol: str, exit_price: float, exit_reason: str, exit_ts: str) -> dict | None:
        position = self.positions.pop(symbol, None)
        if not position:
            return None
        qty = float(position["qty"])
        entry_price = float(position["entry_price"])
        exit_price = float(exit_price)
        exit_notional = exit_price * qty
        self.cash += exit_notional
        pnl = (exit_price - entry_price) * qty
        pnl_pct = ((exit_price / entry_price) - 1.0) if entry_price else 0.0
        return {
            "session_date": position["session_date"],
            "symbol": symbol,
            "qty": qty,
            "entry_price": round(entry_price, 6),
            "exit_price": round(exit_price, 6),
            "entry_ts": position["entry_ts"],
            "trigger_ts": position["trigger_ts"],
            "promotion_ts": position["promoted_ts"],
            "exit_ts": str(exit_ts),
            "bars_held": int(position.get("bars_held", 0)),
            "entry_notional": round(float(position.get("entry_notional", 0.0)), 6),
            "exit_notional": round(exit_notional, 6),
            "exit_reason": exit_reason,
            "pnl": round(pnl, 6),
            "pnl_pct": round(pnl_pct, 6),
        }

    def market_value(self, price_by_symbol: dict[str, float] | None = None) -> float:
        if not price_by_symbol:
            return 0.0
        total = 0.0
        for symbol, position in self.positions.items():
            price = float(price_by_symbol.get(symbol, position["entry_price"]))
            total += price * float(position["qty"])
        return total

    def snapshot(self, price_by_symbol: dict[str, float] | None = None) -> dict:
        market_value = self.market_value(price_by_symbol)
        return {
            "cash": float(self.cash),
            "positions": dict(self.positions),
            "market_value": float(market_value),
            "equity": float(self.cash + market_value),
        }
