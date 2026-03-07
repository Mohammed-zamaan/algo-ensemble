from __future__ import annotations

from datetime import datetime, date
from pathlib import Path

import pandas as pd

from ..engine import PipelineStage


class ExecutionStage(PipelineStage):
    name = "execution"

    def run(self, context):
        settings = context["settings"]
        store = context["store"]
        run_id = context["run_id"]
        orders_df = context.get("orders_df", pd.DataFrame())

        if orders_df.empty:
            print("No orders to execute")
            context["execution_results_df"] = pd.DataFrame()
            return

        if not settings.paper_trade:
            print("Live execution boundary not integrated yet - refusing to place live orders")
            context["execution_results_df"] = pd.DataFrame()
            return

        Path("state").mkdir(parents=True, exist_ok=True)

        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        records = []

        for _, order in orders_df.iterrows():
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

        execution_results_df = pd.DataFrame(records)
        context["execution_results_df"] = execution_results_df

        paper_path = Path(settings.paper_trades_path)
        if paper_path.exists():
            prior = pd.read_csv(paper_path)
            execution_results_df = pd.concat([prior, execution_results_df], ignore_index=True)

        execution_results_df.to_csv(paper_path, index=False)
        print(f"Saved paper executions -> {paper_path}")