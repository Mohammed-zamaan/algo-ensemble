from __future__ import annotations

from pathlib import Path

import pandas as pd


def build_backtest_report(summary: dict, metrics: dict) -> dict:
    return {
        "summary": summary,
        "metrics": metrics,
        "status": "OK",
    }


def write_backtest_outputs(
    *,
    output_dir: Path,
    trades_df: pd.DataFrame,
    equity_df: pd.DataFrame,
    promoted_df: pd.DataFrame,
    orders_df: pd.DataFrame,
    summary: dict,
    metrics: dict,
) -> dict:
    """Write baseline backtest outputs as CSV files."""

    output_dir.mkdir(parents=True, exist_ok=True)
    trades_path = output_dir / "trades.csv"
    equity_path = output_dir / "equity_curve.csv"
    summary_path = output_dir / "summary.csv"
    promoted_path = output_dir / "promoted_signals.csv"
    orders_path = output_dir / "orders.csv"

    trades_df.to_csv(trades_path, index=False)
    equity_df.to_csv(equity_path, index=False)
    promoted_df.to_csv(promoted_path, index=False)
    orders_df.to_csv(orders_path, index=False)
    pd.DataFrame([{**summary, **metrics}]).to_csv(summary_path, index=False)

    return {
        "trades_csv": str(trades_path),
        "equity_csv": str(equity_path),
        "summary_csv": str(summary_path),
        "promoted_signals_csv": str(promoted_path),
        "orders_csv": str(orders_path),
    }
