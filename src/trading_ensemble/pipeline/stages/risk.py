from __future__ import annotations

import pandas as pd

from trading_ensemble.data.sheets_output import maybe_write_output
from trading_ensemble.risk.allocator import AccountAllocator, CandidateSignal
from trading_ensemble.notifications.router import send_order_alerts

from ..engine import PipelineStage


MIN_RR_RATIO = 1.5
DEFAULT_MAX_POSITIONS = 5


class RiskStage(PipelineStage):
    name = "risk"

    def run(self, context):
        settings = context["settings"]
        store = context["store"]
        run_id = context["run_id"]
        control_panel = context.get("control_panel")
        signals_df = context.get("signals_df", pd.DataFrame())

        def empty_orders(message: str):
            print(message)
            context["orders_df"] = pd.DataFrame()
            empty_df = pd.DataFrame()
            empty_df.to_csv(settings.trade_orders_path, index=False)
            maybe_write_output(settings, control_panel, "ApprovedOrders", empty_df)
            return

        if control_panel is not None:
            if not getattr(control_panel, "system_trading_enabled", True):
                return empty_orders("System trading disabled by ControlPanel")

            if getattr(control_panel, "pause_new_entries", False):
                return empty_orders("New entries paused by ControlPanel")

        if signals_df.empty:
            return empty_orders("No signals available for risk stage")

        confirmed = signals_df[signals_df["SIGNAL_STATUS"] == "CONFIRMED"].copy()

        if confirmed.empty:
            return empty_orders("No confirmed signals available for risk stage")

        filtered = confirmed[confirmed["RR_RATIO"] >= MIN_RR_RATIO].copy()
        if filtered.empty:
            return empty_orders("No confirmed signals passed minimum RR filter")

        filtered["MODE"] = filtered["MODE"].astype(str).str.upper()

        max_new_orders_per_run = (
            getattr(control_panel, "max_new_orders_per_run", DEFAULT_MAX_POSITIONS)
            if control_panel is not None
            else DEFAULT_MAX_POSITIONS
        )
        max_intraday_orders = (
            getattr(control_panel, "max_intraday_orders", max_new_orders_per_run)
            if control_panel is not None
            else max_new_orders_per_run
        )
        max_swing_orders = (
            getattr(control_panel, "max_swing_orders", max_new_orders_per_run)
            if control_panel is not None
            else max_new_orders_per_run
        )
        max_positional_orders = (
            getattr(control_panel, "max_positional_orders", max_new_orders_per_run)
            if control_panel is not None
            else max_new_orders_per_run
        )
        risk_multiplier = (
            float(getattr(control_panel, "risk_multiplier", 1.0))
            if control_panel is not None
            else 1.0
        )

        mode_caps = {
            "INTRADAY": max_intraday_orders,
            "SWING": max_swing_orders,
            "POSITIONAL": max_positional_orders,
        }
        mode_selected = {
            "INTRADAY": 0,
            "SWING": 0,
            "POSITIONAL": 0,
        }

        # rank higher-conviction, higher-score signals first
        sort_cols = []
        ascending = []
        if "PRIORITY" in filtered.columns:
            sort_cols.append("PRIORITY")
            ascending.append(True)
        if "WATCHLIST_CONVICTION" in filtered.columns:
            sort_cols.append("WATCHLIST_CONVICTION")
            ascending.append(False)
        sort_cols.extend(["COMPOSITE_SCORE", "RR_RATIO"])
        ascending.extend([False, False])

        filtered = filtered.sort_values(sort_cols, ascending=ascending).reset_index(drop=True)

        selected_rows = []
        for _, row in filtered.iterrows():
            mode = str(row.get("MODE", settings.trade_mode)).upper()
            if mode not in mode_caps:
                continue
            if mode_selected[mode] >= mode_caps[mode]:
                continue
            if len(selected_rows) >= max_new_orders_per_run:
                break

            selected_rows.append(row)
            mode_selected[mode] += 1

        if not selected_rows:
            return empty_orders("No confirmed signals remained after per-mode caps")

        selected_df = pd.DataFrame(selected_rows).reset_index(drop=True)

        effective_capital = settings.total_account_capital * max(risk_multiplier, 0.0)

        candidates = [
            CandidateSignal(
                symbol=row["symbol"],
                entry_price=float(row["ENTRY_PRICE"]),
                stop_loss=float(row["STOP_LOSS"]),
                conviction=float(row.get("WATCHLIST_CONVICTION", 2)),
                sector=str(row.get("SECTOR", "UNKNOWN")),
            )
            for _, row in selected_df.iterrows()
        ]

        allocator = AccountAllocator(effective_capital)
        decisions = allocator.allocate(candidates)

        print("Risk diagnostics:")
        print(f"  confirmed_signals_in   = {len(confirmed)}")
        print(f"  rr_passed_signals      = {len(filtered)}")
        print(f"  selected_intraday      = {mode_selected['INTRADAY']}")
        print(f"  selected_swing         = {mode_selected['SWING']}")
        print(f"  selected_positional    = {mode_selected['POSITIONAL']}")
        print(f"  total_orders_selected  = {len(decisions)}")
        print(f"  risk_multiplier        = {risk_multiplier:.2f}")
        print(f"  effective_capital      = {effective_capital:.2f}")

        orders = []
        for decision in decisions:
            signal_row = selected_df[selected_df["symbol"] == decision.symbol].iloc[0]

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
        send_order_alerts(orders_df)
        print(f"Saved {len(orders_df)} approved orders -> {settings.trade_orders_path}")