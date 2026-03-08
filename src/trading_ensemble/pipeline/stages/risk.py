from __future__ import annotations

import pandas as pd

from trading_ensemble.data.sheets_output import maybe_write_output
from trading_ensemble.risk.allocator import AccountAllocator, CandidateSignal

from ..engine import PipelineStage


MIN_RR_RATIO = 1.5
MAX_POSITIONS = 5


class RiskStage(PipelineStage):
    name = "risk"

    def run(self, context):
        settings = context["settings"]
        store = context["store"]
        run_id = context["run_id"]
        control_panel = context.get("control_panel")
        signals_df = context.get("signals_df", pd.DataFrame())

        if signals_df.empty:
            print("No signals available for risk stage")
            context["orders_df"] = pd.DataFrame()
            empty_df = pd.DataFrame()
            empty_df.to_csv(settings.trade_orders_path, index=False)
            maybe_write_output(settings, control_panel, "ApprovedOrders", empty_df)
            return

        confirmed = signals_df[signals_df["SIGNAL_STATUS"] == "CONFIRMED"].copy()

        if confirmed.empty:
            print("No confirmed signals available for risk stage")
            context["orders_df"] = pd.DataFrame()
            empty_df = pd.DataFrame()
            empty_df.to_csv(settings.trade_orders_path, index=False)
            maybe_write_output(settings, control_panel, "ApprovedOrders", empty_df)
            return

        filtered = confirmed[confirmed["RR_RATIO"] >= MIN_RR_RATIO].copy()
        if filtered.empty:
            print("No confirmed signals passed minimum RR filter")
            context["orders_df"] = pd.DataFrame()
            empty_df = pd.DataFrame()
            empty_df.to_csv(settings.trade_orders_path, index=False)
            maybe_write_output(settings, control_panel, "ApprovedOrders", empty_df)
            return

        filtered = filtered.nlargest(MAX_POSITIONS, "COMPOSITE_SCORE")

        candidates = [
            CandidateSignal(
                symbol=row["symbol"],
                entry_price=float(row["ENTRY_PRICE"]),
                stop_loss=float(row["STOP_LOSS"]),
                conviction=float(row.get("WATCHLIST_CONVICTION", 2)),
                sector=str(row.get("SECTOR", "UNKNOWN")),
            )
            for _, row in filtered.iterrows()
        ]

        allocator = AccountAllocator(settings.total_account_capital)
        decisions = allocator.allocate(candidates)

        orders = []
        for decision in decisions:
            signal_row = filtered[filtered["symbol"] == decision.symbol].iloc[0]

            order = {
                "symbol": decision.symbol,
                "MODE": signal_row.get("MODE", settings.trade_mode),
                "ENTRY_PRICE": float(signal_row["ENTRY_PRICE"]),
                "LIMIT_PRICE": float(signal_row["ENTRY_PRICE"]),
                "TARGET_PRICE": float(signal_row["TARGET_PRICE"]),
                "STOP_LOSS": float(signal_row["STOP_LOSS"]),
                "QUANTITY": int(decision.quantity),
                "CAPITAL_USED": float(decision.allocation_amount),
                "RR_RATIO": float(signal_row["RR_RATIO"]),
                "SIGNAL_STRENGTH": signal_row.get("SIGNAL_STRENGTH", "CONFIRMED"),
                "SIGNAL_STATUS": signal_row.get("SIGNAL_STATUS", "CONFIRMED"),
                "ORDER_TYPE": "LIMIT",
                "PRODUCT_TYPE": signal_row.get("PRODUCT_TYPE", "MIS"),
                "EXCHANGE": "NSE",
                "TRANSACTION": "BUY",
                "STATUS": "PAPER" if settings.paper_trade else "PENDING",
                "conviction": float(decision.conviction),
                "sector": decision.sector,
            }
            orders.append(order)

            store.insert_order(
                run_id=run_id,
                symbol=str(order["symbol"]),
                side=str(order.get("TRANSACTION", "BUY")),
                quantity=int(order["QUANTITY"]),
                order_type=str(order.get("ORDER_TYPE", "LIMIT")),
                product_type=str(order.get("PRODUCT_TYPE", "MIS")),
                status=str(order.get("STATUS", "PENDING")),
                broker_order_id="",
            )

        orders_df = pd.DataFrame(orders)
        context["orders_df"] = orders_df

        orders_df.to_csv(settings.trade_orders_path, index=False)
        maybe_write_output(settings, control_panel, "ApprovedOrders", orders_df)
        print(f"Saved {len(orders_df)} approved orders -> {settings.trade_orders_path}")