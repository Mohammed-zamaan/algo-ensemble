import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import pandas as pd

from trading_ensemble.cli.shadow_session_review import format_shadow_session_summary, print_shadow_session_summary
from trading_ensemble.pipeline.stages.execution import ExecutionStage, _map_broker_status
from trading_ensemble.state.store import StateStore


class ExecutionReliabilityTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.db_path = Path(self.tmp.name) / "trading.db"
        self.paper_path = Path(self.tmp.name) / "paper.csv"
        self.store = StateStore(self.db_path)
        self.store.initialize()

    def tearDown(self):
        self.tmp.cleanup()

    def _settings(self, paper_trade=True):
        return SimpleNamespace(
            paper_trade=paper_trade,
            trade_mode="INTRADAY",
            paper_trades_path=str(self.paper_path),
        )

    def _order_df(self):
        return pd.DataFrame([
            {
                "symbol": "ABC-EQ",
                "MODE": "INTRADAY",
                "TRANSACTION": "BUY",
                "SIGNAL_TIME": "2026-03-11T09:45:00+05:30",
                "LIMIT_PRICE": 100.0,
                "ENTRY_PRICE": 100.0,
                "STOP_LOSS": 95.0,
                "TARGET_PRICE": 110.0,
                "QUANTITY": 1,
                "ORDER_TYPE": "LIMIT",
                "PRODUCT_TYPE": "MIS",
            }
        ])

    def _run_stage(self, context, patches=None):
        patches = patches or []
        with mock.patch("trading_ensemble.pipeline.stages.execution.maybe_write_output", lambda *a, **k: None), \
             mock.patch("trading_ensemble.pipeline.stages.execution.send_execution_alerts", lambda *a, **k: None):
            if not patches:
                ExecutionStage().run(context)
                return
            ctx = []
            for p in patches:
                ctx.append(p)
            for c in ctx:
                c.__enter__()
            try:
                ExecutionStage().run(context)
            finally:
                for c in reversed(ctx):
                    c.__exit__(None, None, None)

    def test_duplicate_evaluation_same_trade_blocked(self):
        context = {"settings": self._settings(True), "store": self.store, "run_id": 1, "orders_df": self._order_df()}
        self._run_stage(context)
        self._run_stage(context)

        with self.store.connect() as conn:
            order_count = conn.execute("SELECT COUNT(1) FROM orders").fetchone()[0]
        self.assertEqual(order_count, 1)

    def test_restart_after_submitting_reconciles_to_unknown(self):
        self.store.claim_execution_intent(
            idempotency_key="k-submit",
            run_id=1,
            symbol="ABC-EQ",
            strategy_mode="INTRADAY",
            side="BUY",
            signal_time="t",
        )
        self.store.update_execution_intent_state(idempotency_key="k-submit", state="SUBMITTING")

        context = {"settings": self._settings(True), "store": self.store, "run_id": 1, "orders_df": pd.DataFrame()}
        self._run_stage(context)

        intent = self.store.get_execution_intent("k-submit")
        self.assertEqual(intent["execution_state"], "UNKNOWN")

    def test_broker_unavailable_reconciliation_trips_breaker(self):
        context = {"settings": self._settings(False), "store": self.store, "run_id": 1, "orders_df": self._order_df()}
        patches = [
            mock.patch("trading_ensemble.pipeline.stages.execution._get_broker_client", return_value=None),
        ]
        self._run_stage(context, patches=patches)

        events = self.store.list_circuit_breaker_events()
        self.assertTrue(any(e["breaker_name"] == "NEW_ENTRY_EXECUTION" for e in events))

    def test_db_write_failure_during_reconciliation_trips_breaker(self):
        self.store.claim_execution_intent(
            idempotency_key="k-db",
            run_id=1,
            symbol="ABC-EQ",
            strategy_mode="INTRADAY",
            side="BUY",
            signal_time="t",
        )

        context = {"settings": self._settings(True), "store": self.store, "run_id": 1, "orders_df": self._order_df()}
        patches = [
            mock.patch.object(self.store, "update_execution_intent_state", side_effect=RuntimeError("db fail")),
        ]
        self._run_stage(context, patches=patches)

        events = self.store.list_circuit_breaker_events()
        self.assertTrue(any("DB_WRITE_FAILURE" in e["reason"] for e in events))

    def test_repeated_unknown_intents_trip_breaker(self):
        for i in range(3):
            key = f"k-unk-{i}"
            self.store.claim_execution_intent(
                idempotency_key=key,
                run_id=1,
                symbol="ABC-EQ",
                strategy_mode="INTRADAY",
                side="BUY",
                signal_time="t",
            )
            self.store.update_execution_intent_state(idempotency_key=key, state="UNKNOWN")

        context = {"settings": self._settings(True), "store": self.store, "run_id": 1, "orders_df": self._order_df()}
        self._run_stage(context)

        events = self.store.list_circuit_breaker_events()
        self.assertTrue(any("REPEATED_UNKNOWN_EXECUTION_INTENTS" in e["reason"] for e in events))

    def test_stale_market_data_breaker_blocks_execution(self):
        eval_df = pd.DataFrame({"snapshot_time": [None, None, None]})
        context = {
            "settings": self._settings(True),
            "store": self.store,
            "run_id": 1,
            "orders_df": self._order_df(),
            "trigger_eval_df": eval_df,
        }
        self._run_stage(context)

        with self.store.connect() as conn:
            order_count = conn.execute("SELECT COUNT(1) FROM orders").fetchone()[0]
        self.assertEqual(order_count, 0)

    def test_conflicting_local_vs_broker_state_prefers_broker_mapping(self):
        self.store.claim_execution_intent(
            idempotency_key="k-conflict",
            run_id=1,
            symbol="ABC-EQ",
            strategy_mode="INTRADAY",
            side="BUY",
            signal_time="t",
        )
        self.store.record_order_submission(
            run_id=1,
            symbol="ABC-EQ",
            side="BUY",
            quantity=1,
            order_type="LIMIT",
            product_type="MIS",
            order_status="PAPER_FILLED",
            broker_order_id="B1",
            idempotency_key="k-conflict",
            intent_state="SUBMITTED",
        )

        context = {"settings": self._settings(False), "store": self.store, "run_id": 1, "orders_df": pd.DataFrame()}
        patches = [
            mock.patch("trading_ensemble.pipeline.stages.execution._get_broker_client", return_value=object()),
            mock.patch("trading_ensemble.pipeline.stages.execution._fetch_broker_orderbook", return_value=[{"orderid": "B1", "status": "REJECTED"}]),
        ]
        self._run_stage(context, patches=patches)

        intent = self.store.get_execution_intent("k-conflict")
        self.assertEqual(intent["execution_state"], "REJECTED")

    def test_unexpected_broker_status_maps_to_ambiguous(self):
        self.assertEqual(_map_broker_status("SOMETHING_NEW"), "AMBIGUOUS")


    def test_breaker_manual_reset_behavior(self):
        self.store.upsert_circuit_breaker_state(
            breaker_name="NEW_ENTRY_EXECUTION",
            reason="AUTH_OR_SESSION_FAILURE",
            reset_policy="MANUAL",
            cooldown_seconds=0,
        )
        active_before = self.store.list_active_circuit_breakers()
        self.assertTrue(any(b["breaker_name"] == "NEW_ENTRY_EXECUTION" for b in active_before))

        self.store.reset_circuit_breaker(breaker_name="NEW_ENTRY_EXECUTION", reset_by="ops")
        active_after = self.store.list_active_circuit_breakers()
        self.assertFalse(any(b["breaker_name"] == "NEW_ENTRY_EXECUTION" for b in active_after))

    def test_breaker_cooldown_reset_distinction(self):
        self.store.upsert_circuit_breaker_state(
            breaker_name="NEW_ENTRY_EXECUTION",
            reason="REPEATED_STALE_MARKET_DATA",
            reset_policy="COOLDOWN",
            cooldown_seconds=60,
        )
        with self.store.connect() as conn:
            conn.execute("UPDATE circuit_breaker_state SET cooldown_until = ? WHERE breaker_name = ?", ("2000-01-01T00:00:00+05:30", "NEW_ENTRY_EXECUTION"))
        active = self.store.list_active_circuit_breakers()
        self.assertFalse(any(b["breaker_name"] == "NEW_ENTRY_EXECUTION" for b in active))

    def test_manual_resolution_of_unknown_intent_is_auditable(self):
        self.store.claim_execution_intent(
            idempotency_key="k-resolve",
            run_id=1,
            symbol="ABC-EQ",
            strategy_mode="INTRADAY",
            side="BUY",
            signal_time="t",
        )
        self.store.update_execution_intent_state(idempotency_key="k-resolve", state="UNKNOWN", notes="needs manual review")

        self.store.resolve_execution_intent(
            idempotency_key="k-resolve",
            resolved_state="REJECTED",
            resolution_type="MANUAL",
            resolved_by="operator",
            notes="broker confirms reject",
        )

        intent = self.store.get_execution_intent("k-resolve")
        self.assertEqual(intent["execution_state"], "REJECTED")
        resolutions = self.store.list_execution_intent_resolutions("k-resolve")
        self.assertEqual(len(resolutions), 1)
        self.assertEqual(resolutions[0]["previous_state"], "UNKNOWN")

    def test_prevent_resubmission_before_unknown_resolution(self):
        context = {"settings": self._settings(True), "store": self.store, "run_id": 1, "orders_df": self._order_df()}
        self._run_stage(context)

        order = self._order_df().iloc[0]
        key = f"{order['SIGNAL_TIME'][0:10]}|{order['symbol']}|{order['MODE']}|{order['TRANSACTION']}|{order['SIGNAL_TIME']}"
        # build via production helper to avoid drift
        from trading_ensemble.pipeline.stages.execution import build_idempotency_key
        key = build_idempotency_key(order)

        self.store.update_execution_intent_state(idempotency_key=key, state="UNKNOWN", notes="ambiguous ack")
        self._run_stage(context)

        events = self.store.list_shadow_session_events()
        duplicate_events = [e for e in events if e["event_type"] == "duplicate_prevented" and e["idempotency_key"] == key]
        self.assertTrue(duplicate_events)



    def test_shadow_session_summary_fields_from_persisted_state(self):
        run_id = 77
        self.store.claim_execution_intent(
            idempotency_key="k-sum-1",
            run_id=run_id,
            symbol="ABC-EQ",
            strategy_mode="INTRADAY",
            side="BUY",
            signal_time="t",
        )
        self.store.update_execution_intent_state(idempotency_key="k-sum-1", state="UNKNOWN")
        self.store.resolve_execution_intent(
            idempotency_key="k-sum-1",
            resolved_state="REJECTED",
            resolution_type="MANUAL",
            resolved_by="ops",
            notes="manual close",
        )

        self.store.upsert_circuit_breaker_state(
            breaker_name="NEW_ENTRY_EXECUTION",
            reason="AUTH_OR_SESSION_FAILURE",
            reset_policy="MANUAL",
            cooldown_seconds=0,
        )

        self.store.record_shadow_session_event(
            event_type="breaker_trip",
            payload_json='{"run_id":77,"reason":"AUTH_OR_SESSION_FAILURE,REPEATED_BROKER_FAILURES"}',
        )
        self.store.record_shadow_session_event(
            event_type="blocked_entry",
            payload_json='{"run_id":77,"reason":"AUTH_OR_SESSION_FAILURE"}',
        )
        self.store.record_shadow_session_event(
            event_type="duplicate_prevented",
            idempotency_key="k-sum-1",
            payload_json='{"run_id":77,"existing_state":"UNKNOWN","requires_manual_resolution":true}',
        )
        self.store.record_shadow_session_event(
            event_type="reconciliation_outcome",
            idempotency_key="k-sum-1",
            payload_json='{"run_id":77,"outcome":"AMBIGUOUS"}',
        )

        summary = self.store.get_shadow_session_summary(run_id=run_id)
        self.assertEqual(summary["run_id"], run_id)
        self.assertEqual(summary["duplicate_prevented_count"], 1)
        self.assertEqual(summary["breaker_trips_by_type"]["AUTH_OR_SESSION_FAILURE"], 1)
        self.assertEqual(summary["breaker_trips_by_type"]["REPEATED_BROKER_FAILURES"], 1)
        self.assertEqual(summary["blocked_entry_reasons"]["AUTH_OR_SESSION_FAILURE"], 1)
        self.assertEqual(summary["reconciliation_outcome_counts"]["AMBIGUOUS"], 1)
        self.assertEqual(summary["manual_resolutions_performed"], 1)
        self.assertTrue(summary["active_breakers_at_session_end"])



    def test_shadow_session_review_formatter_output_shape(self):
        summary = {
            "run_id": 11,
            "breaker_trips_by_type": {"AUTH_OR_SESSION_FAILURE": 1},
            "active_breakers_at_session_end": [
                {
                    "breaker_name": "NEW_ENTRY_EXECUTION",
                    "last_reason": "AUTH_OR_SESSION_FAILURE",
                    "reset_policy": "MANUAL",
                    "trigger_count": 2,
                    "tripped_at": "2026-01-01T09:15:00+05:30",
                }
            ],
            "unknown_or_ambiguous_intents": 3,
            "duplicate_prevented_count": 4,
            "blocked_entry_reasons": {"AUTH_OR_SESSION_FAILURE": 1},
            "reconciliation_outcome_counts": {"AMBIGUOUS": 2},
            "manual_resolutions_performed": 1,
        }
        rendered = format_shadow_session_summary(summary)
        self.assertIn("run_id: 11", rendered)
        self.assertIn("unknown_or_ambiguous_intents: 3", rendered)
        self.assertIn("duplicate_prevented_count: 4", rendered)
        self.assertIn("manual_resolutions_performed: 1", rendered)
        self.assertIn("breaker_trips_by_type:", rendered)
        self.assertIn("active_breakers_at_session_end:", rendered)

    def test_shadow_session_review_uses_store_summary(self):
        run_id = 101
        self.store.record_shadow_session_event(
            event_type="duplicate_prevented",
            payload_json='{"run_id":101,"existing_state":"UNKNOWN","requires_manual_resolution":true}',
        )
        rendered = print_shadow_session_summary(self.store, run_id=run_id)
        self.assertIn("run_id: 101", rendered)
        self.assertIn("duplicate_prevented_count: 1", rendered)



if __name__ == "__main__":
    unittest.main()
