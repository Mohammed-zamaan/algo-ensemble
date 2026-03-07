from __future__ import annotations

from datetime import datetime, date
from pathlib import Path

import pandas as pd

from ..engine import PipelineStage


class ExecutionStage(PipelineStage):
    name = "execution"

    def run(self, context):
        settings = context["settings"]
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
            record = {
                **order.to_dict(),
                "STATUS": "PAPER_FILLED",
                "FILL_PRICE": fill_price,
                "FILL_TIME": now,
                "order_id": f"PAPER-{date.today()}-{order['symbol']}",
            }
            records.append(record)

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