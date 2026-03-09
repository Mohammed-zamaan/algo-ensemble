#!/usr/bin/env bash
set -e

echo "Applying Phase 21: Near Trigger Alerts"

ROOT="src/trading_ensemble"

#########################################
# Patch 1 — ControlPanel fields
#########################################

CONTROL_PANEL="$ROOT/config/control_panel.py"

if ! grep -q "near_trigger_alerts_enabled" "$CONTROL_PANEL"; then
echo "Patching ControlPanel..."

sed -i '/write_outputs_to_sheets/a\
    near_trigger_alerts_enabled: bool = True\
    near_trigger_min_score: float = 95.0\
    near_trigger_max_breakout_gap_pct: float = 1.0\
    max_near_trigger_alerts: int = 10
' "$CONTROL_PANEL"

sed -i '/write_outputs_to_sheets=_to_bool/a\
        near_trigger_alerts_enabled=_to_bool(raw.get("near_trigger_alerts_enabled"), True),\
        near_trigger_min_score=_to_float(raw.get("near_trigger_min_score"), 95.0),\
        near_trigger_max_breakout_gap_pct=_to_float(raw.get("near_trigger_max_breakout_gap_pct"), 1.0),\
        max_near_trigger_alerts=_to_int(raw.get("max_near_trigger_alerts"), 10),
' "$CONTROL_PANEL"

fi

#########################################
# Patch 2 — scheduler alerts builder
#########################################

SCHEDULER="$ROOT/monitor/scheduler.py"

if ! grep -q "build_near_trigger_alerts" "$SCHEDULER"; then
echo "Patching scheduler near-trigger alerts..."

cat >> "$SCHEDULER" << 'EOF'


def build_near_trigger_alerts(eval_df, control_panel):
    import pandas as pd

    if eval_df is None or eval_df.empty:
        return pd.DataFrame()

    if control_panel is None:
        return pd.DataFrame()

    if not getattr(control_panel, "near_trigger_alerts_enabled", True):
        return pd.DataFrame()

    min_score = float(getattr(control_panel, "near_trigger_min_score", 95.0))
    max_breakout_gap = float(getattr(control_panel, "near_trigger_max_breakout_gap_pct", 1.0))
    max_alerts = int(getattr(control_panel, "max_near_trigger_alerts", 10))

    alerts = eval_df.copy()

    if "promotion_candidate" in alerts.columns:
        alerts = alerts[alerts["promotion_candidate"] == False]

    if "readiness_score" in alerts.columns:
        alerts = alerts[alerts["readiness_score"] >= min_score]

    if "breakout_gap_pct" in alerts.columns:
        alerts = alerts[alerts["breakout_gap_pct"] <= max_breakout_gap]

    if alerts.empty:
        return alerts

    alerts = alerts.sort_values(
        ["readiness_score", "breakout_gap_pct", "volume_gap_pct"],
        ascending=[False, True, True],
    ).head(max_alerts).reset_index(drop=True)

    alerts["ALERT_LEVEL"] = alerts["readiness_score"].apply(
        lambda x: "HOT" if float(x) >= 99 else "WARM"
    )

    return alerts

EOF

fi

#########################################
# Patch 3 — Sheets output schema
#########################################

SHEETS="$ROOT/data/sheets_output.py"

if ! grep -q "NearTriggerAlerts" "$SHEETS"; then
echo "Patching sheets output schema..."

sed -i '/TriggerMonitorStatus/a\
    "NearTriggerAlerts": [\
        "symbol",\
        "MODE",\
        "ALERT_LEVEL",\
        "promotion_reason",\
        "readiness_score",\
        "breakout_gap_pct",\
        "volume_gap_pct",\
        "latest_close",\
        "live_donchian_upper",\
        "live_volume_ratio",\
        "snapshot_time",\
        "ENTRY_PRICE",\
        "STOP_LOSS",\
        "TARGET_PRICE",\
        "RR_RATIO",\
        "PRODUCT_TYPE",\
    ],
' "$SHEETS"

fi

echo "Phase 21 patch applied successfully."