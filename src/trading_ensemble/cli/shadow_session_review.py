from __future__ import annotations

import argparse
from typing import Any

from trading_ensemble.config.settings import Settings
from trading_ensemble.state.store import StateStore


def _format_dict_block(title: str, values: dict[str, Any]) -> list[str]:
    lines = [f"{title}:"]
    if not values:
        lines.append("  - none")
        return lines
    for key in sorted(values):
        lines.append(f"  - {key}: {values[key]}")
    return lines


def _format_active_breakers(active_breakers: list[dict[str, Any]]) -> list[str]:
    lines = ["active_breakers_at_session_end:"]
    if not active_breakers:
        lines.append("  - none")
        return lines

    for idx, breaker in enumerate(active_breakers, start=1):
        lines.append(
            "  - "
            f"#{idx} name={breaker.get('breaker_name', 'UNKNOWN')} "
            f"reason={breaker.get('last_reason', 'UNKNOWN')} "
            f"policy={breaker.get('reset_policy', 'UNKNOWN')} "
            f"trigger_count={breaker.get('trigger_count', 0)} "
            f"tripped_at={breaker.get('tripped_at', '')}"
        )
    return lines


def format_shadow_session_summary(summary: dict[str, Any]) -> str:
    lines = [
        "=== Shadow Session Summary ===",
        f"run_id: {summary.get('run_id', 'UNKNOWN')}",
        f"unknown_or_ambiguous_intents: {summary.get('unknown_or_ambiguous_intents', 0)}",
        f"duplicate_prevented_count: {summary.get('duplicate_prevented_count', 0)}",
        f"manual_resolutions_performed: {summary.get('manual_resolutions_performed', 0)}",
        "",
    ]

    lines.extend(_format_dict_block("breaker_trips_by_type", summary.get("breaker_trips_by_type", {})))
    lines.extend(_format_active_breakers(summary.get("active_breakers_at_session_end", [])))
    lines.extend(_format_dict_block("blocked_entry_reasons", summary.get("blocked_entry_reasons", {})))
    lines.extend(_format_dict_block("reconciliation_outcome_counts", summary.get("reconciliation_outcome_counts", {})))

    return "\n".join(lines)


def print_shadow_session_summary(store: StateStore, run_id: int) -> str:
    summary = store.get_shadow_session_summary(run_id=run_id)
    rendered = format_shadow_session_summary(summary)
    print(rendered)
    return rendered


def main() -> None:
    parser = argparse.ArgumentParser(description="Print operator-facing shadow-session summary for a run_id")
    parser.add_argument("--run-id", type=int, required=True, help="Run ID to summarize")
    args = parser.parse_args()

    settings = Settings.from_env()
    store = StateStore(settings.state_db_path)
    store.initialize()
    print_shadow_session_summary(store, run_id=args.run_id)


if __name__ == "__main__":
    main()
