from __future__ import annotations

import time
from datetime import date, datetime
from pathlib import Path

import pandas as pd

from trading_ensemble.data.sheets_output import maybe_write_output
from trading_ensemble.notifications.router import send_execution_alerts
from ..engine import PipelineStage


class ExecutionStage(PipelineStage):
    name = "execution"

    def run(self, context):
        settings = context["settings"]
        store = context["store"]
        run_id = context["run_id"]
        control_panel = context.get("control_panel")
        orders_df = context.get("orders_df", pd.DataFrame())

        def empty_execution(message: str):
            print(message)
            context["execution_results_df"] = pd.DataFrame()
            maybe_write_output(settings, control_panel, "ExecutionLog", pd.DataFrame())
            return

        system_enabled = True if control_panel is None else bool(
            getattr(control_panel, "system_trading_enabled", True)
        )
        pause_new_entries = False if control_panel is None else bool(
            getattr(control_panel, "pause_new_entries", False)
        )
        force_exit_all = False if control_panel is None else bool(
            getattr(control_panel, "force_exit_all", False)
        )

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

        if orders_df.empty:
            return empty_execution("No orders to execute")

        if not settings.paper_trade:
            return empty_execution("Live execution boundary not integrated yet - refusing to place live orders")

        Path("state").mkdir(parents=True, exist_ok=True)

        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        records = []
        executed = 0
        skipped = 0

        for _, order in orders_df.iterrows():
            try:
                fill_price = float(order["LIMIT_PRICE"])
                broker_order_id = f"PAPER-{date.today()}-{order['symbol']}"

                record = {
                    **order.to_dict(),
                    "STATUS": "PAPER_FILLED",
                    "FILL_PRICE": fill_price,
                    "FILL_TIME": now,
                    "order_id": broker_order_id,
                }
                records.append(record)

                store.insert_order(
                    run_id=run_id,
                    symbol=str(order["symbol"]),
                    side=str(order.get("TRANSACTION", "BUY")),
                    quantity=int(order["QUANTITY"]),
                    order_type=str(order.get("ORDER_TYPE", "LIMIT")),
                    product_type=str(order.get("PRODUCT_TYPE", "MIS")),
                    status="PAPER_FILLED",
                    broker_order_id=broker_order_id,
                )

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

                print(
                    f"[PAPER] {order['symbol']:<20} "
                    f"Entry={fill_price:<8.2f} "
                    f"SL={float(order['STOP_LOSS']):<8.2f} "
                    f"TGT={float(order['TARGET_PRICE']):<8.2f} "
                    f"Qty={int(order['QUANTITY']):<5}"
                )
                executed += 1
                time.sleep(0.15)
            except Exception as exc:
                skipped += 1
                print(f"[SKIP] {order.get('symbol', 'UNKNOWN')}: {exc}")

        execution_results_df = pd.DataFrame(records)
        context["execution_results_df"] = execution_results_df

        paper_path = Path(settings.paper_trades_path)
        if paper_path.exists() and not execution_results_df.empty:
            prior = pd.read_csv(paper_path)
            execution_results_df = pd.concat([prior, execution_results_df], ignore_index=True)

        execution_results_df.to_csv(paper_path, index=False)
        maybe_write_output(settings, control_panel, "ExecutionLog", execution_results_df)
        send_execution_alerts(execution_results_df)

        print(f"  executed             = {executed}")
        print(f"  skipped              = {skipped}")
        print(f"Saved paper executions -> {paper_path}")