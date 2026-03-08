from __future__ import annotations

import time
from datetime import datetime, time as dt_time

import pandas as pd

from trading_ensemble.data.sheets_output import maybe_write_output
from trading_ensemble.monitor.trigger_engine import (
    convert_promotions_to_signals,
    dedupe_setup_rows,
    evaluate_setup_for_promotion,
    fetch_latest_market_snapshot,
    filter_promotion_candidates,
    load_setup_signals,
)
from trading_ensemble.pipeline.stages.execution import ExecutionStage
from trading_ensemble.pipeline.stages.risk import RiskStage


def market_is_open(now: datetime) -> bool:
    current = now.time()
    return dt_time(9, 15) <= current <= dt_time(15, 30)


def entry_window_open(now: datetime) -> bool:
    current = now.time()
    return dt_time(9, 30) <= current <= dt_time(15, 0)


def build_monitor_status_df(
    cycle_time: str,
    setups_seen: int,
    promotion_candidates: int,
    promoted_confirmed: int,
    approved_orders: int,
    executed_orders: int,
    market_open: bool,
    entry_open: bool,
    poll_seconds: int,
) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"metric": "cycle_time", "value": cycle_time},
            {"metric": "market_open", "value": market_open},
            {"metric": "entry_window_open", "value": entry_open},
            {"metric": "setups_seen", "value": setups_seen},
            {"metric": "promotion_candidates", "value": promotion_candidates},
            {"metric": "promoted_confirmed", "value": promoted_confirmed},
            {"metric": "approved_orders", "value": approved_orders},
            {"metric": "executed_orders", "value": executed_orders},
            {"metric": "poll_seconds", "value": poll_seconds},
        ]
    )


def run_trigger_cycle(context: dict) -> pd.DataFrame:
    settings = context["settings"]
    control_panel = context.get("control_panel")
    store = context["store"]

    now = datetime.now()
    cycle_time = now.strftime("%Y-%m-%d %H:%M:%S")
    is_market_open = market_is_open(now)
    is_entry_open = entry_window_open(now)

    print("Trigger monitor cycle:")
    print(f"  cycle_time           = {cycle_time}")
    print(f"  market_open          = {is_market_open}")
    print(f"  entry_window_open    = {is_entry_open}")

    if control_panel is not None and not getattr(control_panel, "trigger_monitor_enabled", True):
        print("  monitor disabled by ControlPanel")
        return pd.DataFrame()

    if not is_market_open or not is_entry_open:
        print("  monitor idle: outside entry window")
        return pd.DataFrame()

    setups_df = load_setup_signals(store)
    setups_df = dedupe_setup_rows(setups_df)

    if setups_df.empty:
        print("  no setup signals to monitor")
        maybe_write_output(settings, control_panel, "PromotedSignals", pd.DataFrame())
        status_df = build_monitor_status_df(
            cycle_time=cycle_time,
            setups_seen=0,
            promotion_candidates=0,
            promoted_confirmed=0,
            approved_orders=0,
            executed_orders=0,
            market_open=is_market_open,
            entry_open=is_entry_open,
            poll_seconds=int(getattr(control_panel, "trigger_poll_seconds", 60)) if control_panel else 60,
        )
        maybe_write_output(settings, control_panel, "TriggerMonitorStatus", status_df)
        return pd.DataFrame()

    evaluations = []
    for _, row in setups_df.iterrows():
        snapshot = fetch_latest_market_snapshot(str(row["symbol"]), str(row["MODE"]))
        evaluations.append(evaluate_setup_for_promotion(row, snapshot))

    eval_df = pd.DataFrame(evaluations)
    promoted_df = eval_df[eval_df["promotion_candidate"] == True].copy() if not eval_df.empty else pd.DataFrame()

    max_promotions = int(getattr(control_panel, "max_promotions_per_cycle", 1)) if control_panel else 1
    promoted_df = filter_promotion_candidates(store, promoted_df)
    if not promoted_df.empty:
        promoted_df = promoted_df.head(max_promotions).reset_index(drop=True)

    maybe_write_output(settings, control_panel, "PromotedSignals", promoted_df)

    promoted_signals_df = convert_promotions_to_signals(promoted_df)
    approved_orders_count = 0
    executed_orders_count = 0

    if not promoted_signals_df.empty:
        print(f"  promoted_confirmed   = {len(promoted_signals_df)}")

        promotion_context = {
            "settings": settings,
            "store": store,
            "run_id": context.get("run_id"),
            "control_panel": control_panel,
            "signals_df": promoted_signals_df,
        }

        RiskStage().run(promotion_context)
        orders_df = promotion_context.get("orders_df", pd.DataFrame())
        approved_orders_count = len(orders_df)

        ExecutionStage().run(promotion_context)
        execution_df = promotion_context.get("execution_results_df", pd.DataFrame())
        executed_orders_count = len(execution_df)

    candidates = len(promoted_df)

    status_df = build_monitor_status_df(
        cycle_time=cycle_time,
        setups_seen=len(setups_df),
        promotion_candidates=candidates,
        promoted_confirmed=len(promoted_signals_df),
        approved_orders=approved_orders_count,
        executed_orders=executed_orders_count,
        market_open=is_market_open,
        entry_open=is_entry_open,
        poll_seconds=int(getattr(control_panel, "trigger_poll_seconds", 60)) if control_panel else 60,
    )
    maybe_write_output(settings, control_panel, "TriggerMonitorStatus", status_df)

    print(f"  setups_seen          = {len(setups_df)}")
    print(f"  promotion_candidates = {candidates}")
    print(f"  approved_orders      = {approved_orders_count}")
    print(f"  executed_orders      = {executed_orders_count}")

    return eval_df


def run_trigger_loop(context_factory, max_cycles: int | None = None) -> None:
    cycles = 0

    while True:
        context = context_factory()
        control_panel = context.get("control_panel")
        poll_seconds = int(getattr(control_panel, "trigger_poll_seconds", 60)) if control_panel else 60

        run_trigger_cycle(context)

        cycles += 1
        if max_cycles is not None and cycles >= max_cycles:
            print(f"Trigger monitor exiting after {cycles} cycle(s)")
            return

        time.sleep(poll_seconds)