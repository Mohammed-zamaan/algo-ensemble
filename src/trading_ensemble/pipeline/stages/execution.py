from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Any

import pandas as pd

from trading_ensemble.core.timeutils import fmt_ist, now_ist
from trading_ensemble.data.sheets_output import maybe_write_output
from trading_ensemble.notifications.router import send_execution_alerts
from ..engine import PipelineStage

logger = logging.getLogger(__name__)

RECON_FILLED = "FILLED"
RECON_REJECTED = "REJECTED"
RECON_CANCELLED = "CANCELLED"
RECON_OPEN_WORKING = "OPEN/WORKING"
RECON_NOT_FOUND = "NOT_FOUND"
RECON_BROKER_UNAVAILABLE = "BROKER_UNAVAILABLE"
RECON_AMBIGUOUS = "AMBIGUOUS"


@dataclass
class ExecutionCircuitBreaker:
    tripped: bool = False
    reasons: list[str] = field(default_factory=list)
    auth_failures: int = 0
    broker_failures: int = 0
    stale_market_data_hits: int = 0
    unknown_intents: int = 0

    auth_threshold: int = 1
    broker_failure_threshold: int = 3
    stale_market_threshold: int = 3
    unknown_intents_threshold: int = 3

    def trip(self, reason: str) -> None:
        self.tripped = True
        self.reasons.append(reason)


def build_idempotency_key(order: pd.Series) -> str:
    trade_date_ist = now_ist().date().isoformat()
    symbol = str(order.get("symbol", "")).strip().upper()
    mode = str(order.get("MODE", "")).strip().upper()
    side = str(order.get("TRANSACTION", "BUY")).strip().upper()
    signal_time = str(order.get("SIGNAL_TIME", "")).strip()

    if not signal_time:
        signal_time = f"{order.get('ENTRY_PRICE', '')}|{order.get('STOP_LOSS', '')}|{order.get('TARGET_PRICE', '')}"

    return f"{trade_date_ist}|{symbol}|{mode}|{side}|{signal_time}"


def _get_broker_client(live_enabled: bool) -> Any | None:
    if not live_enabled:
        return None
    try:
        from trading_ensemble.data.smartapi_client import login_from_env

        session = login_from_env()
        return session.smart
    except Exception:
        return None


def _fetch_broker_orderbook(smart_client: Any) -> list[dict] | None:
    if smart_client is None:
        return None
    try:
        if hasattr(smart_client, "orderBook"):
            raw = smart_client.orderBook()
        elif hasattr(smart_client, "getOrderBook"):
            raw = smart_client.getOrderBook()
        else:
            return None
        data = raw.get("data") if isinstance(raw, dict) else None
        if isinstance(data, list):
            return data
        return []
    except Exception:
        return None


def _map_broker_status(raw_status: str) -> str:
    status = str(raw_status or "").strip().upper()
    if status in {"COMPLETE", "FILLED", "EXECUTED", "TRADED"}:
        return RECON_FILLED
    if status in {"REJECTED"}:
        return RECON_REJECTED
    if status in {"CANCELLED", "CANCELED"}:
        return RECON_CANCELLED
    if status in {"OPEN", "TRIGGER PENDING", "PENDING", "MODIFY PENDING", "AMO REQ RECEIVED"}:
        return RECON_OPEN_WORKING
    return RECON_AMBIGUOUS


def _detect_stale_market_data_hits(context: dict) -> int:
    eval_df = context.get("trigger_eval_df", pd.DataFrame())
    if eval_df is None or eval_df.empty or "snapshot_time" not in eval_df.columns:
        return 0
    return int(eval_df["snapshot_time"].isna().sum())


def _breaker_policy_for_reason(reason: str) -> tuple[str, int]:
    if reason == "REPEATED_STALE_MARKET_DATA":
        return ("COOLDOWN", 900)
    return ("MANUAL", 0)


def _shadow_event(store: Any, *, event_type: str, idempotency_key: str = "", payload: dict | None = None) -> None:
    try:
        store.record_shadow_session_event(
            event_type=event_type,
            idempotency_key=idempotency_key,
            payload_json=json.dumps(payload or {}),
        )
    except Exception:
        pass


class ExecutionStage(PipelineStage):
    name = "execution"

    def run(self, context):
        settings = context["settings"]
        store = context["store"]
        run_id = context["run_id"]
        control_panel = context.get("control_panel")
        orders_df = context.get("orders_df", pd.DataFrame())

        def emit_shadow_session_summary() -> None:
            try:
                summary = store.get_shadow_session_summary(run_id=run_id)
                logger.info(json.dumps({
                    "event": "shadow_session_summary",
                    "run_id": run_id,
                    **summary,
                }))
                store.record_shadow_session_event(
                    event_type="session_summary",
                    payload_json=json.dumps({"run_id": run_id, "summary": summary}),
                )
            except Exception:
                pass

        def empty_execution(message: str):
            print(message)
            context["execution_results_df"] = pd.DataFrame()
            maybe_write_output(settings, control_panel, "ExecutionLog", pd.DataFrame())
            emit_shadow_session_summary()
            return

        def safe_store_write(func, *, breaker: ExecutionCircuitBreaker, breaker_name: str, payload: dict | None = None):
            try:
                return func()
            except Exception as exc:
                breaker.trip(f"DB_WRITE_FAILURE:{breaker_name}")
                logger.error(json.dumps({
                    "event": "execution_circuit_breaker",
                    "breaker_name": breaker_name,
                    "reason": "DB_WRITE_FAILURE",
                    "error": str(exc),
                    "metadata": payload or {},
                }))
                try:
                    store.record_circuit_breaker_event(
                        breaker_name=breaker_name,
                        reason="DB_WRITE_FAILURE",
                        severity="CRITICAL",
                        metadata_json=json.dumps(payload or {}),
                    )
                except Exception:
                    pass
                return None

        system_enabled = True if control_panel is None else bool(getattr(control_panel, "system_trading_enabled", True))
        pause_new_entries = False if control_panel is None else bool(getattr(control_panel, "pause_new_entries", False))
        force_exit_all = False if control_panel is None else bool(getattr(control_panel, "force_exit_all", False))

        print("Execution diagnostics:")
        print(f"  mode                 = {'PAPER' if settings.paper_trade else 'LIVE'}")
        print(f"  system_enabled       = {system_enabled}")
        print(f"  pause_new_entries    = {pause_new_entries}")
        print(f"  force_exit_all       = {force_exit_all}")
        print(f"  orders_in            = {len(orders_df)}")

        if not system_enabled:
            return empty_execution("Execution blocked: system_trading_enabled is FALSE")
        if force_exit_all:
            return empty_execution("Execution blocked: force_exit_all requested, flatten logic not implemented yet")
        if pause_new_entries:
            return empty_execution("Execution blocked: pause_new_entries is TRUE")

        Path("state").mkdir(parents=True, exist_ok=True)

        breaker = ExecutionCircuitBreaker()
        active_breakers = store.list_active_circuit_breakers()
        if active_breakers:
            reason = ",".join(sorted({str(b.get("last_reason") or b.get("breaker_name") or "UNKNOWN") for b in active_breakers}))
            logger.error(json.dumps({
                "event": "execution_blocked_by_active_breaker_state",
                "reason": reason,
                "active_breakers": active_breakers,
                "orders_blocked": len(orders_df),
            }))
            _shadow_event(store, event_type="blocked_entry", payload={"run_id": run_id, "reason": reason, "active_breakers": active_breakers})
            return empty_execution(f"Execution blocked by active breaker state: {reason}")

        breaker.stale_market_data_hits = _detect_stale_market_data_hits(context)
        if breaker.stale_market_data_hits >= breaker.stale_market_threshold:
            breaker.trip("REPEATED_STALE_MARKET_DATA")

        live_mode = not settings.paper_trade
        broker_client = _get_broker_client(live_mode)
        if live_mode and broker_client is None:
            breaker.auth_failures += 1
            if breaker.auth_failures >= breaker.auth_threshold:
                breaker.trip("AUTH_OR_SESSION_FAILURE")

        orderbook = _fetch_broker_orderbook(broker_client) if broker_client is not None else None
        if live_mode and broker_client is not None and orderbook is None:
            breaker.broker_failures += 1
            if breaker.broker_failures >= breaker.broker_failure_threshold:
                breaker.trip("BROKER_ORDERBOOK_FAILURE")

        inflight = store.list_inflight_execution_intents()
        for intent in inflight:
            source = "LOCAL_ONLY"
            broker_lookup_attempted = False
            outcome = RECON_AMBIGUOUS
            mapped_state = "UNKNOWN"
            broker_order_id = str(intent.get("broker_order_id") or "")

            if orderbook is not None and broker_order_id:
                source = "ANGEL_ORDERBOOK"
                broker_lookup_attempted = True
                match = next((o for o in orderbook if str(o.get("orderid", "")) == broker_order_id), None)
                if match is None:
                    by_broker = store.get_execution_intent_by_broker_order_id(broker_order_id)
                    if by_broker is not None:
                        outcome = store.reconcile_execution_intent(by_broker["idempotency_key"])
                    else:
                        outcome = RECON_NOT_FOUND
                else:
                    outcome = _map_broker_status(match.get("status") or match.get("orderstatus"))
            elif live_mode and broker_client is None:
                source = "BROKER_UNAVAILABLE"
                outcome = RECON_BROKER_UNAVAILABLE
            else:
                try:
                    local = store.reconcile_execution_intent(intent["idempotency_key"])
                except Exception as exc:
                    breaker.trip("DB_WRITE_FAILURE:RECONCILE_LOCAL")
                    logger.error(json.dumps({
                        "event": "execution_circuit_breaker",
                        "breaker_name": "RECONCILE_LOCAL",
                        "reason": "DB_WRITE_FAILURE",
                        "error": str(exc),
                        "metadata": {"idempotency_key": intent["idempotency_key"]},
                    }))
                    outcome = RECON_AMBIGUOUS
                else:
                    if local in {"FILLED", "REJECTED", "CANCELLED"}:
                        outcome = local
                    else:
                        outcome = RECON_AMBIGUOUS

            if outcome == RECON_FILLED:
                mapped_state = "FILLED"
            elif outcome == RECON_REJECTED:
                mapped_state = "REJECTED"
            elif outcome == RECON_CANCELLED:
                mapped_state = "CANCELLED"
            elif outcome == RECON_OPEN_WORKING:
                mapped_state = "SUBMITTED"
            elif outcome == RECON_NOT_FOUND:
                mapped_state = "UNKNOWN"
            elif outcome == RECON_BROKER_UNAVAILABLE:
                mapped_state = "UNKNOWN"
            else:
                mapped_state = "UNKNOWN"

            safe_store_write(
                lambda: store.update_execution_intent_state(
                    idempotency_key=intent["idempotency_key"],
                    state=mapped_state,
                    broker_order_id=broker_order_id,
                    notes=f"Reconcile source={source} outcome={outcome}",
                ),
                breaker=breaker,
                breaker_name="RECONCILIATION",
                payload={"idempotency_key": intent["idempotency_key"], "outcome": outcome},
            )

            safe_store_write(
                lambda: store.update_order_status_by_idempotency(
                    idempotency_key=intent["idempotency_key"],
                    status=("PENDING" if mapped_state == "SUBMITTED" else mapped_state),
                    broker_order_id=broker_order_id,
                ),
                breaker=breaker,
                breaker_name="ORDER_STATUS_UPDATE",
                payload={"idempotency_key": intent["idempotency_key"], "mapped_state": mapped_state},
            )

            logger.warning(json.dumps({
                "event": "execution_reconciliation",
                "idempotency_key": intent["idempotency_key"],
                "reconciliation_source": source,
                "broker_lookup_attempted": broker_lookup_attempted,
                "mapping_result": mapped_state,
                "outcome": outcome,
            }))
            _shadow_event(
                store,
                event_type="reconciliation_outcome",
                idempotency_key=intent["idempotency_key"],
                payload={
                    "run_id": run_id,
                    "source": source,
                    "broker_lookup_attempted": broker_lookup_attempted,
                    "mapping_result": mapped_state,
                    "outcome": outcome,
                },
            )

            if outcome in {RECON_BROKER_UNAVAILABLE, RECON_AMBIGUOUS}:
                breaker.broker_failures += 1

        breaker.unknown_intents = store.count_execution_intents_by_states(["UNKNOWN"])
        if breaker.unknown_intents >= breaker.unknown_intents_threshold:
            breaker.trip("REPEATED_UNKNOWN_EXECUTION_INTENTS")

        if breaker.broker_failures >= breaker.broker_failure_threshold:
            breaker.trip("REPEATED_BROKER_FAILURES")

        if breaker.tripped:
            reason = ",".join(breaker.reasons)
            try:
                store.record_circuit_breaker_event(
                    breaker_name="NEW_ENTRY_EXECUTION",
                    reason=reason,
                    severity="CRITICAL",
                    metadata_json=json.dumps({
                        "unknown_intents": breaker.unknown_intents,
                        "broker_failures": breaker.broker_failures,
                        "stale_market_data_hits": breaker.stale_market_data_hits,
                        "auth_failures": breaker.auth_failures,
                    }),
                )
                for one_reason in breaker.reasons:
                    reset_policy, cooldown_seconds = _breaker_policy_for_reason(one_reason)
                    store.upsert_circuit_breaker_state(
                        breaker_name="NEW_ENTRY_EXECUTION",
                        reason=one_reason,
                        reset_policy=reset_policy,
                        cooldown_seconds=cooldown_seconds,
                    )
            except Exception:
                pass

            logger.error(json.dumps({
                "event": "execution_blocked_by_circuit_breaker",
                "reason": reason,
                "orders_blocked": len(orders_df),
            }))
            _shadow_event(store, event_type="breaker_trip", payload={"run_id": run_id, "reason": reason, "orders_blocked": len(orders_df)})
            _shadow_event(store, event_type="blocked_entry", payload={"run_id": run_id, "reason": reason, "orders_blocked": len(orders_df)})
            return empty_execution(f"Execution blocked by circuit breaker: {reason}")

        if orders_df.empty:
            return empty_execution("No orders to execute")

        if live_mode:
            return empty_execution("Live execution boundary not integrated yet - refusing to place live orders")

        now = fmt_ist()
        records = []
        executed = 0
        skipped = 0

        for _, order in orders_df.iterrows():
            idempotency_key = build_idempotency_key(order)
            claim_ok, existing_state = store.claim_execution_intent(
                idempotency_key=idempotency_key,
                run_id=run_id,
                symbol=str(order["symbol"]),
                strategy_mode=str(order.get("MODE", settings.trade_mode)),
                side=str(order.get("TRANSACTION", "BUY")),
                signal_time=str(order.get("SIGNAL_TIME", "")),
            )

            logger.info(json.dumps({
                "event": "execution_claim",
                "idempotency_key": idempotency_key,
                "claim_result": "CLAIMED" if claim_ok else "EXISTS",
                "existing_state": existing_state,
            }))

            if not claim_ok:
                skipped += 1
                requires_resolution = existing_state in {"UNKNOWN", "AMBIGUOUS", "SUBMITTING", "SUBMITTED", "ACKED"}
                logger.warning(json.dumps({
                    "event": "execution_duplicate_prevented",
                    "idempotency_key": idempotency_key,
                    "existing_state": existing_state,
                    "requires_manual_resolution": requires_resolution,
                    "broker_submission_attempted": False,
                }))
                _shadow_event(
                    store,
                    event_type="duplicate_prevented",
                    idempotency_key=idempotency_key,
                    payload={"run_id": run_id, "existing_state": existing_state, "requires_manual_resolution": requires_resolution},
                )
                continue

            try:
                store.update_execution_intent_state(idempotency_key=idempotency_key, state="SUBMITTING")
                fill_price = float(order["LIMIT_PRICE"])
                broker_order_id = f"PAPER-{date.today()}-{order['symbol']}"

                record = {
                    **order.to_dict(),
                    "IDEMPOTENCY_KEY": idempotency_key,
                    "STATUS": "PAPER_FILLED",
                    "FILL_PRICE": fill_price,
                    "FILL_TIME": now,
                    "order_id": broker_order_id,
                }
                records.append(record)

                store.record_order_submission(
                    run_id=run_id,
                    symbol=str(order["symbol"]),
                    side=str(order.get("TRANSACTION", "BUY")),
                    quantity=int(order["QUANTITY"]),
                    order_type=str(order.get("ORDER_TYPE", "LIMIT")),
                    product_type=str(order.get("PRODUCT_TYPE", "MIS")),
                    order_status="PAPER_FILLED",
                    broker_order_id=broker_order_id,
                    idempotency_key=idempotency_key,
                    intent_state="SUBMITTED",
                    notes="Paper order submission recorded",
                )
                store.update_execution_intent_state(idempotency_key=idempotency_key, state="ACKED", broker_order_id=broker_order_id, notes="Paper order acknowledged")

                store.insert_fill(
                    run_id=run_id,
                    symbol=str(order["symbol"]),
                    quantity=int(order["QUANTITY"]),
                    fill_price=fill_price,
                    fill_status="PAPER_FILLED",
                    broker_order_id=broker_order_id,
                )
                store.upsert_position(
                    symbol=str(order["symbol"]),
                    quantity=int(order["QUANTITY"]),
                    average_price=fill_price,
                    product_type=str(order.get("PRODUCT_TYPE", "MIS")),
                    strategy_mode=str(order.get("MODE", settings.trade_mode)),
                    status="OPEN",
                )
                store.update_execution_intent_state(idempotency_key=idempotency_key, state="FILLED", broker_order_id=broker_order_id, notes="Paper fill persisted")

                executed += 1
                time.sleep(0.05)
            except Exception as exc:
                skipped += 1
                store.update_execution_intent_state(idempotency_key=idempotency_key, state="UNKNOWN", notes=f"Submission ambiguous: {exc}")
                logger.error(json.dumps({
                    "event": "execution_submit_error",
                    "idempotency_key": idempotency_key,
                    "broker_submission_attempted": True,
                    "reconciliation_outcome": "AMBIGUOUS",
                    "error": str(exc),
                }))
                _shadow_event(
                    store,
                    event_type="unknown_intent_created",
                    idempotency_key=idempotency_key,
                    payload={"run_id": run_id, "error": str(exc)},
                )

        execution_results_df = pd.DataFrame(records)
        context["execution_results_df"] = execution_results_df

        paper_path = Path(settings.paper_trades_path)
        if paper_path.exists() and not execution_results_df.empty:
            prior = pd.read_csv(paper_path)
            execution_results_df = pd.concat([prior, execution_results_df], ignore_index=True)

        execution_results_df.to_csv(paper_path, index=False)
        maybe_write_output(settings, control_panel, "ExecutionLog", execution_results_df)
        send_execution_alerts(execution_results_df)
        emit_shadow_session_summary()

        print(f"  executed             = {executed}")
        print(f"  skipped              = {skipped}")
        print(f"Saved paper executions -> {paper_path}")
