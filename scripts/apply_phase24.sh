#!/usr/bin/env bash
set -e

echo "Applying Phase 24: crisis regime detection and capital preservation mode"

ROOT="src/trading_ensemble"

########################################
# Patch regime/regime_engine.py
########################################
python - <<'PY'
from pathlib import Path
path = Path("src/trading_ensemble/regime/regime_engine.py")
text = path.read_text()

text = text.replace(
    '        if atr_pct >= 2.2:\n            regime = "HIGH_VOL"\n        elif abs(trend_gap_pct) >= 1.0:\n            regime = "TREND"\n        else:\n            regime = "CHOP"\n',
    '        daily_return_pct = ((close.iloc[-1] - close.iloc[-2]) / close.iloc[-2]) * 100.0 if len(close) >= 2 else 0.0\n'
    '        high_low_pct = ((high.iloc[-1] - low.iloc[-1]) / close.iloc[-1]) * 100.0 if close.iloc[-1] else 0.0\n\n'
    '        if atr_pct >= 2.8 or abs(daily_return_pct) >= 2.5 or high_low_pct >= 3.5:\n'
    '            regime = "CRISIS"\n'
    '        elif atr_pct >= 2.2:\n'
    '            regime = "HIGH_VOL"\n'
    '        elif abs(trend_gap_pct) >= 1.0:\n'
    '            regime = "TREND"\n'
    '        else:\n'
    '            regime = "CHOP"\n'
)

text = text.replace(
    '    high_vol_risk = float(getattr(control_panel, "high_vol_risk_multiplier", 0.5)) if control_panel else 0.5\n\n'
    '    trend_orders = int(getattr(control_panel, "trend_max_new_orders", base_orders)) if control_panel else base_orders\n'
    '    chop_orders = int(getattr(control_panel, "chop_max_new_orders", max(1, min(2, base_orders)))) if control_panel else max(1, min(2, base_orders))\n'
    '    high_vol_orders = int(getattr(control_panel, "high_vol_max_new_orders", 1)) if control_panel else 1\n',
    '    high_vol_risk = float(getattr(control_panel, "high_vol_risk_multiplier", 0.5)) if control_panel else 0.5\n'
    '    crisis_risk = float(getattr(control_panel, "crisis_risk_multiplier", 0.25)) if control_panel else 0.25\n\n'
    '    trend_orders = int(getattr(control_panel, "trend_max_new_orders", base_orders)) if control_panel else base_orders\n'
    '    chop_orders = int(getattr(control_panel, "chop_max_new_orders", max(1, min(2, base_orders)))) if control_panel else max(1, min(2, base_orders))\n'
    '    high_vol_orders = int(getattr(control_panel, "high_vol_max_new_orders", 1)) if control_panel else 1\n'
    '    crisis_orders = int(getattr(control_panel, "crisis_max_new_orders", 1)) if control_panel else 1\n'
)

text = text.replace(
    '    if regime == "HIGH_VOL":\n'
    '        return {\n'
    '            "market_regime": regime,\n'
    '            "effective_risk_multiplier": round(base_risk * high_vol_risk, 4),\n'
    '            "effective_max_new_orders_per_run": high_vol_orders,\n'
    '        }\n\n'
    '    return {\n',
    '    if regime == "HIGH_VOL":\n'
    '        return {\n'
    '            "market_regime": regime,\n'
    '            "effective_risk_multiplier": round(base_risk * high_vol_risk, 4),\n'
    '            "effective_max_new_orders_per_run": high_vol_orders,\n'
    '        }\n'
    '    if regime == "CRISIS":\n'
    '        return {\n'
    '            "market_regime": regime,\n'
    '            "effective_risk_multiplier": round(base_risk * crisis_risk, 4),\n'
    '            "effective_max_new_orders_per_run": crisis_orders,\n'
    '        }\n\n'
    '    return {\n'
)

path.write_text(text)
PY

########################################
# Patch config/control_panel.py
########################################
python - <<'PY'
from pathlib import Path
path = Path("src/trading_ensemble/config/control_panel.py")
text = path.read_text()

if "crisis_risk_multiplier" not in text:
    text = text.replace(
        '    high_vol_max_new_orders: int = 1\n',
        '    high_vol_max_new_orders: int = 1\n'
        '    crisis_risk_multiplier: float = 0.25\n'
        '    crisis_max_new_orders: int = 1\n'
        '    crisis_pause_new_entries: bool = False\n'
    )

if 'crisis_risk_multiplier=_to_float(raw.get("crisis_risk_multiplier"), 0.25),' not in text:
    text = text.replace(
        '        high_vol_max_new_orders=_to_int(raw.get("high_vol_max_new_orders"), 1),\n',
        '        high_vol_max_new_orders=_to_int(raw.get("high_vol_max_new_orders"), 1),\n'
        '        crisis_risk_multiplier=_to_float(raw.get("crisis_risk_multiplier"), 0.25),\n'
        '        crisis_max_new_orders=_to_int(raw.get("crisis_max_new_orders"), 1),\n'
        '        crisis_pause_new_entries=_to_bool(raw.get("crisis_pause_new_entries"), False),\n'
    )

path.write_text(text)
PY

########################################
# Patch scheduler.py to expose crisis pause
########################################
python - <<'PY'
from pathlib import Path
path = Path("src/trading_ensemble/monitor/scheduler.py")
text = path.read_text()

if 'if context["market_regime"] == "CRISIS" and getattr(control_panel, "crisis_pause_new_entries", False):' not in text:
    needle = '    print(f"  effective_max_orders = {context[\'effective_max_new_orders_per_run\']}")\n\n'
    insert = needle + '    if context["market_regime"] == "CRISIS" and getattr(control_panel, "crisis_pause_new_entries", False):\n' \
                      '        print("  crisis mode active: new entries paused")\n' \
                      '        return pd.DataFrame()\n\n'
    text = text.replace(needle, insert)

path.write_text(text)
PY

########################################
# Patch dashboard/command_center.py
########################################
python - <<'PY'
from pathlib import Path
path = Path("src/trading_ensemble/dashboard/command_center.py")
text = path.read_text()

if '("crisis_pause_new_entries", getattr(control_panel, "crisis_pause_new_entries", False) if control_panel else False),' not in text:
    text = text.replace(
        '        ("effective_max_new_orders", context.get("effective_max_new_orders_per_run", "")),\n',
        '        ("effective_max_new_orders", context.get("effective_max_new_orders_per_run", "")),\n'
        '        ("crisis_pause_new_entries", getattr(control_panel, "crisis_pause_new_entries", False) if control_panel else False),\n'
    )

path.write_text(text)
PY

echo "Phase 24 patch applied successfully."