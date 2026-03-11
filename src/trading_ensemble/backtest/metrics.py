from __future__ import annotations

import pandas as pd


def compute_backtest_metrics(*, trades_df: pd.DataFrame, equity_df: pd.DataFrame, initial_capital: float) -> dict:
    """Compute baseline summary metrics for daily backtest scaffold."""

    total_trades = int(len(trades_df))
    total_pnl = float(trades_df["pnl"].sum()) if (not trades_df.empty and "pnl" in trades_df.columns) else 0.0
    wins = int((trades_df["pnl"] > 0).sum()) if (not trades_df.empty and "pnl" in trades_df.columns) else 0
    win_rate = (wins / total_trades) if total_trades else 0.0

    final_equity = float(equity_df["equity"].iloc[-1]) if (not equity_df.empty and "equity" in equity_df.columns) else float(initial_capital)
    return_pct = ((final_equity / float(initial_capital)) - 1.0) * 100.0 if initial_capital else 0.0

    return {
        "total_trades": total_trades,
        "winning_trades": wins,
        "win_rate": round(win_rate, 4),
        "total_pnl": round(total_pnl, 2),
        "final_equity": round(final_equity, 2),
        "return_pct": round(return_pct, 4),
    }
