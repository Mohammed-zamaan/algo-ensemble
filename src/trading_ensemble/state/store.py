from __future__ import annotations

import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

from trading_ensemble.core.timeutils import fmt_ist
from trading_ensemble.state import _store_control_plane as _cp


SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS runs (
    run_id INTEGER PRIMARY KEY AUTOINCREMENT,
    started_at TEXT NOT NULL,
    finished_at TEXT,
    mode TEXT NOT NULL,
    paper_trade INTEGER NOT NULL,
    status TEXT NOT NULL,
    notes TEXT
);

CREATE TABLE IF NOT EXISTS orders (
    order_id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id INTEGER,
    symbol TEXT NOT NULL,
    side TEXT NOT NULL,
    quantity INTEGER NOT NULL,
    order_type TEXT NOT NULL,
    product_type TEXT NOT NULL,
    status TEXT NOT NULL,
    broker_order_id TEXT,
    idempotency_key TEXT,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    FOREIGN KEY(run_id) REFERENCES runs(run_id)
);

CREATE TABLE IF NOT EXISTS fills (
    fill_id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id INTEGER,
    symbol TEXT NOT NULL,
    quantity INTEGER NOT NULL,
    fill_price REAL NOT NULL,
    fill_time TEXT NOT NULL,
    fill_status TEXT NOT NULL,
    broker_order_id TEXT,
    FOREIGN KEY(run_id) REFERENCES runs(run_id)
);

CREATE TABLE IF NOT EXISTS positions (
    position_id INTEGER PRIMARY KEY AUTOINCREMENT,
    symbol TEXT NOT NULL,
    quantity INTEGER NOT NULL,
    average_price REAL NOT NULL,
    product_type TEXT NOT NULL,
    strategy_mode TEXT NOT NULL,
    status TEXT NOT NULL,
    opened_at TEXT NOT NULL,
    closed_at TEXT
);

CREATE TABLE IF NOT EXISTS candidates (
    candidate_id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id INTEGER NOT NULL,
    symbol TEXT NOT NULL,
    composite_score REAL,
    volatility_score REAL,
    ltp REAL,
    donchian_upper REAL,
    volume_ratio REAL,
    breakout INTEGER,
    created_at TEXT NOT NULL,
    FOREIGN KEY(run_id) REFERENCES runs(run_id)
);

CREATE TABLE IF NOT EXISTS signals (
    signal_id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id INTEGER NOT NULL,
    symbol TEXT NOT NULL,
    strategy_mode TEXT NOT NULL,
    entry_price REAL NOT NULL,
    stop_loss REAL NOT NULL,
    target_price REAL NOT NULL,
    rr_ratio REAL NOT NULL,
    atr REAL,
    adx REAL,
    volume_ratio REAL,
    signal_strength TEXT,
    product_type TEXT,
    signal_time TEXT,
    created_at TEXT NOT NULL,
    FOREIGN KEY(run_id) REFERENCES runs(run_id)
);

CREATE TABLE IF NOT EXISTS equity_snapshots (
    snapshot_id INTEGER PRIMARY KEY AUTOINCREMENT,
    captured_at TEXT NOT NULL,
    cash REAL NOT NULL,
    positions_market_value REAL NOT NULL,
    total_equity REAL NOT NULL
);

CREATE TABLE IF NOT EXISTS execution_intents (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    idempotency_key TEXT NOT NULL UNIQUE,
    run_id INTEGER,
    symbol TEXT NOT NULL,
    strategy_mode TEXT,
    side TEXT,
    signal_time TEXT,
    execution_state TEXT NOT NULL,
    broker_order_id TEXT,
    notes TEXT,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    FOREIGN KEY(run_id) REFERENCES runs(run_id)
);


CREATE TABLE IF NOT EXISTS circuit_breaker_events (
    event_id INTEGER PRIMARY KEY AUTOINCREMENT,
    breaker_name TEXT NOT NULL,
    reason TEXT NOT NULL,
    severity TEXT NOT NULL,
    metadata_json TEXT,
    created_at TEXT NOT NULL
);


CREATE TABLE IF NOT EXISTS circuit_breaker_state (
    breaker_name TEXT PRIMARY KEY,
    active INTEGER NOT NULL,
    tripped_at TEXT,
    last_reason TEXT,
    trigger_count INTEGER NOT NULL DEFAULT 0,
    reset_policy TEXT NOT NULL DEFAULT 'MANUAL',
    cooldown_seconds INTEGER NOT NULL DEFAULT 0,
    cooldown_until TEXT,
    last_reset_at TEXT,
    updated_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS execution_intent_resolutions (
    resolution_id INTEGER PRIMARY KEY AUTOINCREMENT,
    idempotency_key TEXT NOT NULL,
    previous_state TEXT NOT NULL,
    resolved_state TEXT NOT NULL,
    resolution_type TEXT NOT NULL,
    resolved_by TEXT NOT NULL,
    notes TEXT,
    resolved_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS shadow_session_events (
    event_id INTEGER PRIMARY KEY AUTOINCREMENT,
    event_type TEXT NOT NULL,
    idempotency_key TEXT,
    payload_json TEXT,
    created_at TEXT NOT NULL
);

CREATE UNIQUE INDEX IF NOT EXISTS idx_orders_idempotency_key ON orders(idempotency_key);
"""


class StateStore:
    def __init__(self, db_path: Path) -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)

    @contextmanager
    def connect(self) -> Iterator[sqlite3.Connection]:
        conn = sqlite3.connect(self.db_path)
        try:
            yield conn
            conn.commit()
        finally:
            conn.close()

    def initialize(self) -> None:
        with self.connect() as conn:
            conn.executescript(SCHEMA_SQL)
            cols = [r[1] for r in conn.execute("PRAGMA table_info(orders)").fetchall()]
            if "idempotency_key" not in cols:
                conn.execute("ALTER TABLE orders ADD COLUMN idempotency_key TEXT")
            conn.execute("CREATE UNIQUE INDEX IF NOT EXISTS idx_orders_idempotency_key ON orders(idempotency_key)")

    def create_run(self, mode: str, paper_trade: bool, status: str = "STARTED") -> int:
        started_at = fmt_ist()
        with self.connect() as conn:
            cur = conn.execute(
                """
                INSERT INTO runs (started_at, mode, paper_trade, status, notes)
                VALUES (?, ?, ?, ?, ?)
                """,
                (started_at, mode, int(paper_trade), status, ""),
            )
            return int(cur.lastrowid)

    def finish_run(self, run_id: int, status: str = "COMPLETED", notes: str = "") -> None:
        finished_at = fmt_ist()
        with self.connect() as conn:
            conn.execute(
                """
                UPDATE runs
                SET finished_at = ?,
                    status = ?,
                    notes = ?
                WHERE run_id = ?
                """,
                (finished_at, status, notes, run_id),
            )

    def claim_execution_intent(
        self,
        *,
        idempotency_key: str,
        run_id: int,
        symbol: str,
        strategy_mode: str,
        side: str,
        signal_time: str,
    ) -> tuple[bool, str]:
        timestamp = fmt_ist()
        with self.connect() as conn:
            cur = conn.execute(
                """
                INSERT OR IGNORE INTO execution_intents (
                    idempotency_key, run_id, symbol, strategy_mode, side,
                    signal_time, execution_state, broker_order_id, notes, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, 'PENDING_SUBMIT', '', '', ?, ?)
                """,
                (idempotency_key, run_id, symbol, strategy_mode, side, signal_time, timestamp, timestamp),
            )
            inserted = cur.rowcount == 1
            row = conn.execute(
                "SELECT execution_state FROM execution_intents WHERE idempotency_key = ?",
                (idempotency_key,),
            ).fetchone()

        state = str(row[0]) if row else "UNKNOWN"
        claimed = inserted
        return claimed, state

    def update_execution_intent_state(
        self,
        *,
        idempotency_key: str,
        state: str,
        broker_order_id: str = "",
        notes: str = "",
    ) -> None:
        timestamp = fmt_ist()
        with self.connect() as conn:
            conn.execute(
                """
                UPDATE execution_intents
                SET execution_state = ?,
                    broker_order_id = CASE WHEN ? = '' THEN broker_order_id ELSE ? END,
                    notes = CASE WHEN ? = '' THEN notes ELSE ? END,
                    updated_at = ?
                WHERE idempotency_key = ?
                """,
                (state, broker_order_id, broker_order_id, notes, notes, timestamp, idempotency_key),
            )

    def list_inflight_execution_intents(self) -> list[dict]:
        with self.connect() as conn:
            rows = conn.execute(
                """
                SELECT idempotency_key, execution_state, broker_order_id, symbol, strategy_mode
                FROM execution_intents
                WHERE execution_state IN ('PENDING_SUBMIT', 'SUBMITTING', 'SUBMITTED', 'ACKED', 'UNKNOWN')
                """
            ).fetchall()
        return [
            {
                "idempotency_key": r[0],
                "execution_state": r[1],
                "broker_order_id": r[2],
                "symbol": r[3],
                "strategy_mode": r[4],
            }
            for r in rows
        ]

    def reconcile_execution_intent(self, idempotency_key: str) -> str:
        with self.connect() as conn:
            row = conn.execute(
                """
                SELECT status, broker_order_id
                FROM orders
                WHERE idempotency_key = ?
                ORDER BY order_id DESC
                LIMIT 1
                """,
                (idempotency_key,),
            ).fetchone()

        if row is None:
            self.update_execution_intent_state(idempotency_key=idempotency_key, state="UNKNOWN", notes="No order row found during restart reconciliation")
            return "UNKNOWN"

        status = str(row[0]).upper()
        broker_order_id = str(row[1] or "")
        if status in {"PAPER_FILLED", "FILLED"}:
            self.update_execution_intent_state(idempotency_key=idempotency_key, state="FILLED", broker_order_id=broker_order_id, notes="Reconciled from persisted orders")
            return "FILLED"
        if status in {"REJECTED", "CANCELLED"}:
            self.update_execution_intent_state(idempotency_key=idempotency_key, state=status, broker_order_id=broker_order_id, notes="Reconciled from persisted orders")
            return status

        self.update_execution_intent_state(idempotency_key=idempotency_key, state="UNKNOWN", broker_order_id=broker_order_id, notes="Ambiguous persisted order status")
        return "UNKNOWN"


    def get_execution_intent(self, idempotency_key: str) -> dict | None:
        with self.connect() as conn:
            row = conn.execute(
                """
                SELECT idempotency_key, execution_state, broker_order_id, symbol, strategy_mode, side, signal_time, notes
                FROM execution_intents
                WHERE idempotency_key = ?
                LIMIT 1
                """,
                (idempotency_key,),
            ).fetchone()
        if row is None:
            return None
        return {
            "idempotency_key": row[0],
            "execution_state": row[1],
            "broker_order_id": row[2],
            "symbol": row[3],
            "strategy_mode": row[4],
            "side": row[5],
            "signal_time": row[6],
            "notes": row[7],
        }

    def update_order_status_by_idempotency(self, *, idempotency_key: str, status: str, broker_order_id: str = "") -> None:
        timestamp = fmt_ist()
        with self.connect() as conn:
            conn.execute(
                """
                UPDATE orders
                SET status = ?,
                    broker_order_id = CASE WHEN ? = '' THEN broker_order_id ELSE ? END,
                    updated_at = ?
                WHERE idempotency_key = ?
                """,
                (status, broker_order_id, broker_order_id, timestamp, idempotency_key),
            )

    def count_execution_intents_by_states(self, states: list[str]) -> int:
        if not states:
            return 0
        placeholders = ",".join(["?" for _ in states])
        with self.connect() as conn:
            row = conn.execute(
                f"SELECT COUNT(1) FROM execution_intents WHERE execution_state IN ({placeholders})",
                tuple(states),
            ).fetchone()
        return int(row[0] or 0)

    def record_circuit_breaker_event(self, *, breaker_name: str, reason: str, severity: str, metadata_json: str = "") -> None:
        _cp.record_circuit_breaker_event(self, breaker_name=breaker_name, reason=reason, severity=severity, metadata_json=metadata_json)

    def upsert_circuit_breaker_state(
        self,
        *,
        breaker_name: str,
        reason: str,
        reset_policy: str = "MANUAL",
        cooldown_seconds: int = 0,
    ) -> None:
        _cp.upsert_circuit_breaker_state(
            self,
            breaker_name=breaker_name,
            reason=reason,
            reset_policy=reset_policy,
            cooldown_seconds=cooldown_seconds,
        )

    def get_circuit_breaker_state(self, breaker_name: str) -> dict | None:
        return _cp.get_circuit_breaker_state(self, breaker_name)

    def list_active_circuit_breakers(self) -> list[dict]:
        return _cp.list_active_circuit_breakers(self)

    def reset_circuit_breaker(self, *, breaker_name: str, reset_by: str, notes: str = "") -> None:
        _cp.reset_circuit_breaker(self, breaker_name=breaker_name, reset_by=reset_by, notes=notes)

    def resolve_execution_intent(
        self,
        *,
        idempotency_key: str,
        resolved_state: str,
        resolution_type: str,
        resolved_by: str,
        notes: str = "",
    ) -> None:
        _cp.resolve_execution_intent(
            self,
            idempotency_key=idempotency_key,
            resolved_state=resolved_state,
            resolution_type=resolution_type,
            resolved_by=resolved_by,
            notes=notes,
        )

    def list_execution_intent_resolutions(self, idempotency_key: str) -> list[dict]:
        return _cp.list_execution_intent_resolutions(self, idempotency_key)

    def record_shadow_session_event(self, *, event_type: str, idempotency_key: str = "", payload_json: str = "") -> None:
        _cp.record_shadow_session_event(self, event_type=event_type, idempotency_key=idempotency_key, payload_json=payload_json)

    def list_shadow_session_events(self) -> list[dict]:
        return _cp.list_shadow_session_events(self)

    def get_execution_intent_by_broker_order_id(self, broker_order_id: str) -> dict | None:
        with self.connect() as conn:
            row = conn.execute(
                """
                SELECT idempotency_key, execution_state, broker_order_id, symbol, strategy_mode, side, signal_time, notes
                FROM execution_intents
                WHERE broker_order_id = ?
                ORDER BY id DESC
                LIMIT 1
                """,
                (broker_order_id,),
            ).fetchone()
        if row is None:
            return None
        return {
            "idempotency_key": row[0],
            "execution_state": row[1],
            "broker_order_id": row[2],
            "symbol": row[3],
            "strategy_mode": row[4],
            "side": row[5],
            "signal_time": row[6],
            "notes": row[7],
        }

    def reconcile_execution_intent_by_broker_order_id(self, broker_order_id: str) -> str:
        intent = self.get_execution_intent_by_broker_order_id(broker_order_id)
        if intent is None:
            return "NOT_FOUND"
        return self.reconcile_execution_intent(intent["idempotency_key"])

    def record_order_submission(
        self,
        *,
        run_id: int,
        symbol: str,
        side: str,
        quantity: int,
        order_type: str,
        product_type: str,
        order_status: str,
        broker_order_id: str,
        idempotency_key: str,
        intent_state: str = "SUBMITTED",
        notes: str = "",
    ) -> None:
        timestamp = fmt_ist()
        with self.connect() as conn:
            conn.execute(
                """
                INSERT INTO orders (
                    run_id, symbol, side, quantity, order_type, product_type,
                    status, broker_order_id, idempotency_key, created_at, updated_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (run_id, symbol, side, quantity, order_type, product_type, order_status, broker_order_id, idempotency_key, timestamp, timestamp),
            )
            conn.execute(
                """
                UPDATE execution_intents
                SET execution_state = ?,
                    broker_order_id = ?,
                    notes = CASE WHEN ? = '' THEN notes ELSE ? END,
                    updated_at = ?
                WHERE idempotency_key = ?
                """,
                (intent_state, broker_order_id, notes, notes, timestamp, idempotency_key),
            )

    def list_circuit_breaker_events(self) -> list[dict]:
        return _cp.list_circuit_breaker_events(self)

    def get_shadow_session_summary(self, run_id: int) -> dict:
        return _cp.get_shadow_session_summary(self, run_id)

    def insert_order(
        self,
        run_id: int,
        symbol: str,
        side: str,
        quantity: int,
        order_type: str,
        product_type: str,
        status: str,
        broker_order_id: str = "",
        idempotency_key: str = "",
    ) -> None:
        timestamp = fmt_ist()
        with self.connect() as conn:
            conn.execute(
                """
                INSERT INTO orders (
                    run_id, symbol, side, quantity, order_type, product_type,
                    status, broker_order_id, idempotency_key, created_at, updated_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (run_id, symbol, side, quantity, order_type, product_type, status, broker_order_id, idempotency_key or None, timestamp, timestamp),
            )

    def insert_fill(
        self,
        run_id: int,
        symbol: str,
        quantity: int,
        fill_price: float,
        fill_status: str,
        broker_order_id: str = "",
    ) -> None:
        fill_time = fmt_ist()
        with self.connect() as conn:
            conn.execute(
                """
                INSERT INTO fills (
                    run_id, symbol, quantity, fill_price, fill_time,
                    fill_status, broker_order_id
                )
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (run_id, symbol, quantity, fill_price, fill_time, fill_status, broker_order_id),
            )

    def upsert_position(
        self,
        symbol: str,
        quantity: int,
        average_price: float,
        product_type: str,
        strategy_mode: str,
        status: str = "OPEN",
    ) -> None:
        opened_at = fmt_ist()
        closed_at = fmt_ist() if status == "CLOSED" else None
        with self.connect() as conn:
            existing = conn.execute(
                """
                SELECT position_id FROM positions
                WHERE symbol = ? AND status = 'OPEN'
                ORDER BY position_id DESC
                LIMIT 1
                """,
                (symbol,),
            ).fetchone()

            if existing:
                conn.execute(
                    """
                    UPDATE positions
                    SET quantity = ?, average_price = ?, product_type = ?, strategy_mode = ?, status = ?,
                        closed_at = CASE
                            WHEN ? IS NULL THEN closed_at
                            ELSE ?
                        END
                    WHERE position_id = ?
                    """,
                    (quantity, average_price, product_type, strategy_mode, status, closed_at, closed_at, existing[0]),
                )
            else:
                conn.execute(
                    """
                    INSERT INTO positions (
                        symbol, quantity, average_price, product_type,
                        strategy_mode, status, opened_at
                    )
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                    """,
                    (symbol, quantity, average_price, product_type, strategy_mode, status, opened_at),
                )

    def insert_candidate(
        self,
        run_id: int,
        symbol: str,
        composite_score: float | None,
        volatility_score: float | None,
        ltp: float | None,
        donchian_upper: float | None,
        volume_ratio: float | None,
        breakout: bool | None,
    ) -> None:
        created_at = fmt_ist()
        with self.connect() as conn:
            conn.execute(
                """
                INSERT INTO candidates (
                    run_id, symbol, composite_score, volatility_score, ltp,
                    donchian_upper, volume_ratio, breakout, created_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    run_id,
                    symbol,
                    composite_score,
                    volatility_score,
                    ltp,
                    donchian_upper,
                    volume_ratio,
                    int(bool(breakout)) if breakout is not None else None,
                    created_at,
                ),
            )

    def insert_signal(
        self,
        run_id: int,
        symbol: str,
        strategy_mode: str,
        entry_price: float,
        stop_loss: float,
        target_price: float,
        rr_ratio: float,
        atr: float | None,
        adx: float | None,
        volume_ratio: float | None,
        signal_strength: str | None,
        product_type: str | None,
        signal_time: str | None,
    ) -> None:
        created_at = fmt_ist()
        with self.connect() as conn:
            conn.execute(
                """
                INSERT INTO signals (
                    run_id, symbol, strategy_mode, entry_price, stop_loss,
                    target_price, rr_ratio, atr, adx, volume_ratio,
                    signal_strength, product_type, signal_time, created_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    run_id,
                    symbol,
                    strategy_mode,
                    entry_price,
                    stop_loss,
                    target_price,
                    rr_ratio,
                    atr,
                    adx,
                    volume_ratio,
                    signal_strength,
                    product_type,
                    signal_time,
                    created_at,
                ),
            )
