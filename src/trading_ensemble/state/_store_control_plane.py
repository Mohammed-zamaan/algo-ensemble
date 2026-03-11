from __future__ import annotations

import json
from datetime import timedelta
from typing import TYPE_CHECKING

from trading_ensemble.core.timeutils import fmt_ist, now_ist

if TYPE_CHECKING:
    from trading_ensemble.state.store import StateStore


def record_circuit_breaker_event(store: "StateStore", *, breaker_name: str, reason: str, severity: str, metadata_json: str = "") -> None:
    created_at = fmt_ist()
    with store.connect() as conn:
        conn.execute(
            """
            INSERT INTO circuit_breaker_events (breaker_name, reason, severity, metadata_json, created_at)
            VALUES (?, ?, ?, ?, ?)
            """,
            (breaker_name, reason, severity, metadata_json, created_at),
        )


def upsert_circuit_breaker_state(
    store: "StateStore",
    *,
    breaker_name: str,
    reason: str,
    reset_policy: str = "MANUAL",
    cooldown_seconds: int = 0,
) -> None:
    now = now_ist()
    updated_at = now.isoformat()
    cooldown_until = None
    if reset_policy == "COOLDOWN" and cooldown_seconds > 0:
        cooldown_until = (now + timedelta(seconds=cooldown_seconds)).isoformat()

    with store.connect() as conn:
        conn.execute(
            """
            INSERT INTO circuit_breaker_state (
                breaker_name, active, tripped_at, last_reason, trigger_count,
                reset_policy, cooldown_seconds, cooldown_until, last_reset_at, updated_at
            ) VALUES (?, 1, ?, ?, 1, ?, ?, ?, NULL, ?)
            ON CONFLICT(breaker_name) DO UPDATE SET
                active = 1,
                tripped_at = COALESCE(circuit_breaker_state.tripped_at, excluded.tripped_at),
                last_reason = excluded.last_reason,
                trigger_count = circuit_breaker_state.trigger_count + 1,
                reset_policy = excluded.reset_policy,
                cooldown_seconds = excluded.cooldown_seconds,
                cooldown_until = excluded.cooldown_until,
                updated_at = excluded.updated_at
            """,
            (breaker_name, updated_at, reason, reset_policy, cooldown_seconds, cooldown_until, updated_at),
        )


def get_circuit_breaker_state(store: "StateStore", breaker_name: str) -> dict | None:
    with store.connect() as conn:
        row = conn.execute(
            """
            SELECT breaker_name, active, tripped_at, last_reason, trigger_count,
                   reset_policy, cooldown_seconds, cooldown_until, last_reset_at, updated_at
            FROM circuit_breaker_state
            WHERE breaker_name = ?
            LIMIT 1
            """,
            (breaker_name,),
        ).fetchone()
    if row is None:
        return None
    return {
        "breaker_name": row[0],
        "active": bool(row[1]),
        "tripped_at": row[2],
        "last_reason": row[3],
        "trigger_count": int(row[4] or 0),
        "reset_policy": row[5],
        "cooldown_seconds": int(row[6] or 0),
        "cooldown_until": row[7],
        "last_reset_at": row[8],
        "updated_at": row[9],
    }


def list_active_circuit_breakers(store: "StateStore") -> list[dict]:
    now_iso = now_ist().isoformat()
    with store.connect() as conn:
        conn.execute(
            """
            UPDATE circuit_breaker_state
            SET active = 0,
                last_reset_at = ?,
                updated_at = ?
            WHERE active = 1
              AND reset_policy = 'COOLDOWN'
              AND cooldown_until IS NOT NULL
              AND cooldown_until <= ?
            """,
            (now_iso, now_iso, now_iso),
        )
        rows = conn.execute(
            """
            SELECT breaker_name, active, tripped_at, last_reason, trigger_count,
                   reset_policy, cooldown_seconds, cooldown_until, last_reset_at, updated_at
            FROM circuit_breaker_state
            WHERE active = 1
            ORDER BY updated_at DESC
            """
        ).fetchall()
    return [
        {
            "breaker_name": r[0],
            "active": bool(r[1]),
            "tripped_at": r[2],
            "last_reason": r[3],
            "trigger_count": int(r[4] or 0),
            "reset_policy": r[5],
            "cooldown_seconds": int(r[6] or 0),
            "cooldown_until": r[7],
            "last_reset_at": r[8],
            "updated_at": r[9],
        }
        for r in rows
    ]


def reset_circuit_breaker(store: "StateStore", *, breaker_name: str, reset_by: str, notes: str = "") -> None:
    now_iso = now_ist().isoformat()
    with store.connect() as conn:
        conn.execute(
            """
            UPDATE circuit_breaker_state
            SET active = 0,
                last_reset_at = ?,
                updated_at = ?,
                last_reason = CASE
                    WHEN ? = '' THEN last_reason
                    ELSE ?
                END
            WHERE breaker_name = ?
            """,
            (now_iso, now_iso, notes, notes, breaker_name),
        )
    store.record_shadow_session_event(
        event_type="circuit_breaker_reset",
        payload_json=f'{{"breaker_name":"{breaker_name}","reset_by":"{reset_by}"}}',
    )


def resolve_execution_intent(
    store: "StateStore",
    *,
    idempotency_key: str,
    resolved_state: str,
    resolution_type: str,
    resolved_by: str,
    notes: str = "",
) -> None:
    timestamp = fmt_ist()
    with store.connect() as conn:
        existing = conn.execute(
            "SELECT execution_state FROM execution_intents WHERE idempotency_key = ? LIMIT 1",
            (idempotency_key,),
        ).fetchone()
        if existing is None:
            raise ValueError(f"Execution intent not found: {idempotency_key}")
        previous_state = str(existing[0])
        conn.execute(
            """
            UPDATE execution_intents
            SET execution_state = ?, notes = ?, updated_at = ?
            WHERE idempotency_key = ?
            """,
            (resolved_state, notes or f"Resolved by {resolved_by}", timestamp, idempotency_key),
        )
        conn.execute(
            """
            INSERT INTO execution_intent_resolutions (
                idempotency_key, previous_state, resolved_state,
                resolution_type, resolved_by, notes, resolved_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (idempotency_key, previous_state, resolved_state, resolution_type, resolved_by, notes, timestamp),
        )


def list_execution_intent_resolutions(store: "StateStore", idempotency_key: str) -> list[dict]:
    with store.connect() as conn:
        rows = conn.execute(
            """
            SELECT idempotency_key, previous_state, resolved_state, resolution_type, resolved_by, notes, resolved_at
            FROM execution_intent_resolutions
            WHERE idempotency_key = ?
            ORDER BY resolution_id ASC
            """,
            (idempotency_key,),
        ).fetchall()
    return [
        {
            "idempotency_key": r[0],
            "previous_state": r[1],
            "resolved_state": r[2],
            "resolution_type": r[3],
            "resolved_by": r[4],
            "notes": r[5],
            "resolved_at": r[6],
        }
        for r in rows
    ]


def record_shadow_session_event(store: "StateStore", *, event_type: str, idempotency_key: str = "", payload_json: str = "") -> None:
    created_at = fmt_ist()
    with store.connect() as conn:
        conn.execute(
            """
            INSERT INTO shadow_session_events (event_type, idempotency_key, payload_json, created_at)
            VALUES (?, ?, ?, ?)
            """,
            (event_type, idempotency_key or None, payload_json, created_at),
        )


def list_shadow_session_events(store: "StateStore") -> list[dict]:
    with store.connect() as conn:
        rows = conn.execute(
            """
            SELECT event_type, idempotency_key, payload_json, created_at
            FROM shadow_session_events
            ORDER BY event_id ASC
            """
        ).fetchall()
    return [
        {
            "event_type": r[0],
            "idempotency_key": r[1],
            "payload_json": r[2],
            "created_at": r[3],
        }
        for r in rows
    ]


def list_circuit_breaker_events(store: "StateStore") -> list[dict]:
    with store.connect() as conn:
        rows = conn.execute(
            """
            SELECT breaker_name, reason, severity, metadata_json, created_at
            FROM circuit_breaker_events
            ORDER BY event_id ASC
            """
        ).fetchall()
    return [
        {
            "breaker_name": r[0],
            "reason": r[1],
            "severity": r[2],
            "metadata_json": r[3],
            "created_at": r[4],
        }
        for r in rows
    ]


def get_shadow_session_summary(store: "StateStore", run_id: int) -> dict:
    unknown_or_ambiguous = 0
    manual_resolutions = 0
    with store.connect() as conn:
        row = conn.execute(
            """
            SELECT COUNT(1)
            FROM execution_intents
            WHERE run_id = ? AND execution_state IN ('UNKNOWN', 'AMBIGUOUS')
            """,
            (run_id,),
        ).fetchone()
        unknown_or_ambiguous = int(row[0] or 0)

        row = conn.execute(
            """
            SELECT COUNT(1)
            FROM execution_intent_resolutions r
            JOIN execution_intents i ON i.idempotency_key = r.idempotency_key
            WHERE i.run_id = ? AND UPPER(r.resolution_type) = 'MANUAL'
            """,
            (run_id,),
        ).fetchone()
        manual_resolutions = int(row[0] or 0)

    breaker_trips_by_type: dict[str, int] = {}
    blocked_entry_reasons: dict[str, int] = {}
    reconciliation_outcomes: dict[str, int] = {}
    duplicate_prevented_count = 0

    for event in store.list_shadow_session_events():
        payload = {}
        try:
            payload = json.loads(event.get("payload_json") or "{}")
        except Exception:
            payload = {}
        if int(payload.get("run_id", -1)) != int(run_id):
            continue

        event_type = str(event.get("event_type") or "")
        if event_type == "breaker_trip":
            reasons = str(payload.get("reason") or "").split(",")
            for reason in [r.strip() for r in reasons if r.strip()]:
                breaker_trips_by_type[reason] = breaker_trips_by_type.get(reason, 0) + 1
        elif event_type == "blocked_entry":
            reason = str(payload.get("reason") or "UNKNOWN")
            blocked_entry_reasons[reason] = blocked_entry_reasons.get(reason, 0) + 1
        elif event_type == "reconciliation_outcome":
            outcome = str(payload.get("outcome") or "UNKNOWN")
            reconciliation_outcomes[outcome] = reconciliation_outcomes.get(outcome, 0) + 1
        elif event_type == "duplicate_prevented":
            duplicate_prevented_count += 1

    return {
        "run_id": run_id,
        "breaker_trips_by_type": breaker_trips_by_type,
        "active_breakers_at_session_end": store.list_active_circuit_breakers(),
        "unknown_or_ambiguous_intents": unknown_or_ambiguous,
        "duplicate_prevented_count": duplicate_prevented_count,
        "blocked_entry_reasons": blocked_entry_reasons,
        "reconciliation_outcome_counts": reconciliation_outcomes,
        "manual_resolutions_performed": manual_resolutions,
    }
