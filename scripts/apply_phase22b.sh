#!/usr/bin/env bash
set -e

echo "Applying Phase 22B: Telegram alerts for approved orders and executions"

ROOT="src/trading_ensemble"

########################################
# Patch notifications/router.py
########################################
ROUTER="$ROOT/notifications/router.py"

if ! grep -q "send_order_alerts" "$ROUTER"; then
cat >> "$ROUTER" <<'PY'


def send_order_alerts(orders_df: pd.DataFrame) -> None:
    if orders_df is None or orders_df.empty:
        return

    for _, row in orders_df.iterrows():
        symbol = str(row.get("symbol", "UNKNOWN"))
        mode = str(row.get("MODE", "UNKNOWN"))
        qty = row.get("QUANTITY", "")
        entry = row.get("ENTRY_PRICE", "")
        sl = row.get("STOP_LOSS", "")
        tgt = row.get("TARGET_PRICE", "")
        status = row.get("STATUS", "")

        alert_key = f"{_day_key()}|order|{symbol}|{mode}"
        if _already_sent(alert_key):
            continue

        text = (
            f"ORDER APPROVED\n"
            f"{symbol} [{mode}]\n"
            f"Qty: {qty}\n"
            f"Entry: {entry}\n"
            f"SL: {sl}\n"
            f"Target: {tgt}\n"
            f"Status: {status}"
        )

        if send_telegram_message(text):
            _mark_sent(alert_key)


def send_execution_alerts(execution_df: pd.DataFrame) -> None:
    if execution_df is None or execution_df.empty:
        return

    for _, row in execution_df.iterrows():
        symbol = str(row.get("symbol", "UNKNOWN"))
        mode = str(row.get("MODE", "UNKNOWN"))
        qty = row.get("QUANTITY", "")
        fill = row.get("FILL_PRICE", "")
        status = row.get("STATUS", "")

        alert_key = f"{_day_key()}|execution|{symbol}|{mode}"
        if _already_sent(alert_key):
            continue

        text = (
            f"EXECUTED\n"
            f"{symbol} [{mode}]\n"
            f"Status: {status}\n"
            f"Qty: {qty}\n"
            f"Fill: {fill}"
        )

        if send_telegram_message(text):
            _mark_sent(alert_key)
PY
fi

########################################
# Patch risk.py imports and call
########################################
RISK="$ROOT/pipeline/stages/risk.py"

python - <<'PY'
from pathlib import Path

path = Path("src/trading_ensemble/pipeline/stages/risk.py")
text = path.read_text()

import_line = "from trading_ensemble.notifications.router import send_order_alerts\n"
anchor = "from trading_ensemble.risk.allocator import AccountAllocator, CandidateSignal\n"
if import_line not in text:
    text = text.replace(anchor, anchor + import_line)

needle = '        maybe_write_output(settings, control_panel, "ApprovedOrders", orders_df)\n'
insert = needle + '        send_order_alerts(orders_df)\n'
if 'send_order_alerts(orders_df)' not in text:
    text = text.replace(needle, insert)

path.write_text(text)
PY

########################################
# Patch execution.py imports and call
########################################
EXEC="$ROOT/pipeline/stages/execution.py"

python - <<'PY'
from pathlib import Path

path = Path("src/trading_ensemble/pipeline/stages/execution.py")
text = path.read_text()

import_line = "from trading_ensemble.notifications.router import send_execution_alerts\n"
anchor = "from trading_ensemble.data.sheets_output import maybe_write_output\n"
if import_line not in text:
    text = text.replace(anchor, anchor + import_line)

needle = '        maybe_write_output(settings, control_panel, "ExecutionLog", execution_results_df)\n'
insert = needle + '        send_execution_alerts(execution_results_df)\n'
if 'send_execution_alerts(execution_results_df)' not in text:
    text = text.replace(needle, insert)

path.write_text(text)
PY

echo "Phase 22B patch applied successfully."