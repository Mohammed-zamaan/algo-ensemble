from __future__ import annotations

import time
from datetime import datetime, time as dt_time

import pandas as pd

from trading_ensemble.data.sheets_output import maybe_write_output
from trading_ensemble.monitor.trigger_engine import (
    dedupe_setup_rows,
    evaluate_setup_for_promotion,
    fetch_latest_market_snapshot,
    load_setup_signals,
)


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
        status_df = build_monitor_status_df(
            cycle_time=cycle_time,
            setups_seen=0,
            promotion_candidates=0,
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
    candidates = int(eval_df["promotion_candidate"].sum()) if not eval_df.empty and "promotion_candidate" in eval_df.columns else 0

    status_df = build_monitor_status_df(
        cycle_time=cycle_time,
        setups_seen=len(setups_df),
        promotion_candidates=candidates,
        market_open=is_market_open,
        entry_open=is_entry_open,
        poll_seconds=int(getattr(control_panel, "trigger_poll_seconds", 60)) if control_panel else 60,
    )
    maybe_write_output(settings, control_panel, "TriggerMonitorStatus", status_df)

    print(f"  setups_seen          = {len(setups_df)}")
    print(f"  promotion_candidates = {candidates}")

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
