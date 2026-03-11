from __future__ import annotations

import pandas as pd

from trading_ensemble.core.timeutils import fmt_ist


def _section(title: str) -> list[dict]:
    return [
        {"section": title, "field": "", "value": ""},
    ]


def _kv_rows(items: list[tuple[str, object]]) -> list[dict]:
    return [{"section": "", "field": k, "value": v} for k, v in items]


def _table_rows(title: str, df: pd.DataFrame, columns: list[str], limit: int = 5) -> list[dict]:
    rows: list[dict] = []
    rows.extend(_section(title))

    if df is None or df.empty:
        rows.append({"section": "", "field": "status", "value": "EMPTY"})
        return rows

    use_cols = [c for c in columns if c in df.columns]
    if not use_cols:
        rows.append({"section": "", "field": "status", "value": "NO_COLUMNS"})
        return rows

    work = df[use_cols].head(limit).copy()

    # header row
    rows.append({
        "section": "",
        "field": " | ".join(use_cols),
        "value": "",
    })

    for _, row in work.iterrows():
        values = [str(row.get(col, "")) for col in use_cols]
        rows.append({
            "section": "",
            "field": " | ".join(values),
            "value": "",
        })

    return rows


def build_command_center(context: dict) -> pd.DataFrame:
    control_panel = context.get("control_panel")
    candidates_df = context.get("candidates_df", pd.DataFrame())
    signals_df = context.get("signals_df", pd.DataFrame())
    orders_df = context.get("orders_df", pd.DataFrame())
    execution_df = context.get("execution_results_df", pd.DataFrame())
    trigger_eval_df = context.get("trigger_eval_df", pd.DataFrame())
    promoted_df = context.get("promoted_df", pd.DataFrame())
    alerts_df = context.get("near_trigger_alerts_df", pd.DataFrame())
    watchlist = context.get("watchlist", [])

    confirmed_count = 0
    setup_count = 0
    if signals_df is not None and not signals_df.empty and "SIGNAL_STATUS" in signals_df.columns:
        confirmed_count = int((signals_df["SIGNAL_STATUS"] == "CONFIRMED").sum())
        setup_count = int((signals_df["SIGNAL_STATUS"] == "SETUP").sum())

    rows: list[dict] = []

    rows.extend(_section("RUN STATUS"))
    rows.extend(_kv_rows([
        ("generated_at", fmt_ist()),
        ("watchlist_symbols", len(watchlist)),
        ("shortlisted_candidates", len(candidates_df)),
        ("setup_signals", setup_count),
        ("confirmed_signals", confirmed_count),
        ("approved_orders", len(orders_df)),
        ("executed_orders", len(execution_df)),
        ("system_trading_enabled", getattr(control_panel, "system_trading_enabled", True) if control_panel else True),
        ("pause_new_entries", getattr(control_panel, "pause_new_entries", False) if control_panel else False),
        ("force_exit_all", getattr(control_panel, "force_exit_all", False) if control_panel else False),
        ("market_regime", context.get("market_regime", "UNKNOWN")),
        ("effective_risk_multiplier", context.get("effective_risk_multiplier", "")),
        ("effective_max_new_orders", context.get("effective_max_new_orders_per_run", "")),
        ("crisis_pause_new_entries", getattr(control_panel, "crisis_pause_new_entries", False) if control_panel else False),
    ]))

    rows.extend(_table_rows(
        "NEAR TRIGGER ALERTS",
        alerts_df,
        ["symbol", "MODE", "ALERT_LEVEL", "readiness_score", "breakout_gap_pct", "volume_gap_pct"],
        limit=5,
    ))

    rows.extend(_table_rows(
        "TOP TRIGGER EVALUATIONS",
        trigger_eval_df,
        ["symbol", "MODE", "promotion_reason", "readiness_score", "breakout_gap_pct", "volume_gap_pct"],
        limit=5,
    ))

    rows.extend(_table_rows(
        "PROMOTED SIGNALS",
        promoted_df,
        ["symbol", "MODE", "promotion_reason", "latest_close", "live_volume_ratio"],
        limit=5,
    ))

    rows.extend(_table_rows(
        "APPROVED ORDERS",
        orders_df,
        ["symbol", "MODE", "QUANTITY", "ENTRY_PRICE", "STOP_LOSS", "TARGET_PRICE", "STATUS"],
        limit=5,
    ))

    rows.extend(_table_rows(
        "RECENT EXECUTIONS",
        execution_df,
        ["symbol", "MODE", "STATUS", "QUANTITY", "FILL_PRICE", "FILL_TIME"],
        limit=5,
    ))

    return pd.DataFrame(rows)
