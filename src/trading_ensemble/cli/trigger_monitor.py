from __future__ import annotations

import argparse

from trading_ensemble.config.control_panel import load_control_panel
from trading_ensemble.config.settings import Settings
from trading_ensemble.monitor.scheduler import run_trigger_cycle, run_trigger_loop
from trading_ensemble.state.store import StateStore


def build_context() -> dict:
    settings = Settings.from_env()
    store = StateStore(settings.state_db_path)
    store.initialize()

    try:
        control_panel = load_control_panel(settings)
    except Exception as exc:
        print(f"Failed to load ControlPanel for trigger monitor: {exc}")
        control_panel = None

    run_id = store.create_run(
        mode="TRIGGER_MONITOR",
        paper_trade=settings.paper_trade,
        status="STARTED",
    )

    return {
        "settings": settings,
        "store": store,
        "control_panel": control_panel,
        "run_id": run_id,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Phase 20A trigger monitor skeleton")
    parser.add_argument("--once", action="store_true", help="Run exactly one monitor cycle and exit.")
    parser.add_argument("--cycles", type=int, default=None, help="Run N cycles then exit.")
    args = parser.parse_args()

    if args.once:
        context = build_context()
        try:
            run_trigger_cycle(context)
            context["store"].finish_run(context["run_id"], status="COMPLETED", notes="Trigger monitor single cycle")
        except Exception as exc:
            context["store"].finish_run(context["run_id"], status="FAILED", notes=str(exc))
            raise
        return

    run_trigger_loop(context_factory=build_context, max_cycles=args.cycles)


if __name__ == "__main__":
    main()
