from __future__ import annotations

import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator


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

CREATE TABLE IF NOT EXISTS equity_snapshots (
    snapshot_id INTEGER PRIMARY KEY AUTOINCREMENT,
    captured_at TEXT NOT NULL,
    cash REAL NOT NULL,
    positions_market_value REAL NOT NULL,
    total_equity REAL NOT NULL
);
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

    def create_run(self, mode: str, paper_trade: bool, status: str = "STARTED") -> int:
        with self.connect() as conn:
            cur = conn.execute(
                """
                INSERT INTO runs (started_at, mode, paper_trade, status, notes)
                VALUES (datetime('now'), ?, ?, ?, ?)
                """,
                (mode, int(paper_trade), status, ""),
            )
            return int(cur.lastrowid)

    def finish_run(self, run_id: int, status: str = "COMPLETED", notes: str = "") -> None:
        with self.connect() as conn:
            conn.execute(
                """
                UPDATE runs
                SET finished_at = datetime('now'),
                    status = ?,
                    notes = ?
                WHERE run_id = ?
                """,
                (status, notes, run_id),
            )

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
    ) -> None:
        with self.connect() as conn:
            conn.execute(
                """
                INSERT INTO orders (
                    run_id, symbol, side, quantity, order_type, product_type,
                    status, broker_order_id, created_at, updated_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, datetime('now'), datetime('now'))
                """,
                (run_id, symbol, side, quantity, order_type, product_type, status, broker_order_id),
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
        with self.connect() as conn:
            conn.execute(
                """
                INSERT INTO fills (
                    run_id, symbol, quantity, fill_price, fill_time,
                    fill_status, broker_order_id
                )
                VALUES (?, ?, ?, ?, datetime('now'), ?, ?)
                """,
                (run_id, symbol, quantity, fill_price, fill_status, broker_order_id),
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
                    SET quantity = ?, average_price = ?, product_type = ?, strategy_mode = ?, status = ?
                    WHERE position_id = ?
                    """,
                    (quantity, average_price, product_type, strategy_mode, status, existing[0]),
                )
            else:
                conn.execute(
                    """
                    INSERT INTO positions (
                        symbol, quantity, average_price, product_type,
                        strategy_mode, status, opened_at
                    )
                    VALUES (?, ?, ?, ?, ?, ?, datetime('now'))
                    """,
                    (symbol, quantity, average_price, product_type, strategy_mode, status),
                )