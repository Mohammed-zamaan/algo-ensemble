from __future__ import annotations

import argparse

from trading_ensemble.config.control_panel import load_control_panel
from trading_ensemble.config.settings import Settings
from trading_ensemble.state.store import StateStore
from trading_ensemble.monitor.scheduler import run_trigger_cycle, run_trigger_loop


def build_context() -> dict:
    settings = Settings.from_env()
    store = StateStore(settings.state_db_path)
    store.initialize()

    try:
        control_panel = load_control_panel(settings)
    except Exception as exc:
        print(f"Failed to load ControlPanel for trigger monitor: {exc}")
        control_panel = None

    return {
        "settings": settings,
        "store": store,
        "control_panel": control_panel,
    }


def main():
    parser = argparse.ArgumentParser(description="Phase 20A trigger monitor skeleton")
    parser.add_argument(
        "--once",
        action="store_true",
        help="Run exactly one monitor cycle and exit.",
    )
    parser.add_argument(
        "--cycles",
        type=int,
        default=None,
        help="Run N cycles then exit.",
    )
    args = parser.parse_args()

    if args.once:
        run_trigger_cycle(build_context())
        return

    run_trigger_loop(
        context_factory=build_context,
        max_cycles=args.cycles,
    )


if __name__ == "__main__":
    main()