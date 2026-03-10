#!/usr/bin/env bash
set -e

echo "Applying Phase 23: market regime detection and risk throttling"

ROOT="src/trading_ensemble"

mkdir -p "$ROOT/regime"

########################################
# regime/regime_engine.py
########################################
cat > "$ROOT/regime/regime_engine.py" <<'PY'
from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
import yfinance as yf


@dataclass(frozen=True)
class RegimeSnapshot:
    regime: str
    adx_like: float
    atr_pct: float
    index_price: float
    index_sma20: float
    trend_gap_pct: float


def fetch_index_regime(symbol: str = "^NSEI") -> RegimeSnapshot:
    try:
        df = yf.download(
            symbol,
            period="3mo",
            interval="1d",
            progress=False,
            auto_adjust=True,
        )
        if df is None or df.empty:
            return RegimeSnapshot("UNKNOWN", 0.0, 0.0, 0.0, 0.0, 0.0)

        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)

        df.columns = [str(c).strip().lower() for c in df.columns]
        df = df[["high", "low", "close"]].copy().dropna()

        if len(df) < 25:
            return RegimeSnapshot("UNKNOWN", 0.0, 0.0, 0.0, 0.0, 0.0)

        high = df["high"].astype(float)
        low = df["low"].astype(float)
        close = df["close"].astype(float)

        prev_close = close.shift(1)
        tr = pd.concat(
            [
                high - low,
                (high - prev_close).abs(),
                (low - prev_close).abs(),
            ],
            axis=1,
        ).max(axis=1)

        atr = tr.ewm(span=14, adjust=False).mean()
        atr_pct = float((atr.iloc[-1] / close.iloc[-1]) * 100.0)

        sma20 = close.rolling(20).mean()
        index_price = float(close.iloc[-1])
        index_sma20 = float(sma20.iloc[-1])
        trend_gap_pct = ((index_price - index_sma20) / index_sma20) * 100.0 if index_sma20 else 0.0

        # lightweight ADX-like proxy
        adx_like = min(abs(trend_gap_pct) * 8.0, 100.0)

        if atr_pct >= 2.2:
            regime = "HIGH_VOL"
        elif abs(trend_gap_pct) >= 1.0:
            regime = "TREND"
        else:
            regime = "CHOP"

        return RegimeSnapshot(
            regime=regime,
            adx_like=round(adx_like, 2),
            atr_pct=round(atr_pct, 4),
            index_price=round(index_price, 2),
            index_sma20=round(index_sma20, 2),
            trend_gap_pct=round(trend_gap_pct, 4),
        )
    except Exception:
        return RegimeSnapshot("UNKNOWN", 0.0, 0.0, 0.0, 0.0, 0.0)


def apply_regime_overrides(control_panel, regime_snapshot: RegimeSnapshot) -> dict:
    base_risk = float(getattr(control_panel, "risk_multiplier", 1.0)) if control_panel else 1.0
    base_orders = int(getattr(control_panel, "max_new_orders_per_run", 4)) if control_panel else 4

    trend_risk = float(getattr(control_panel, "trend_risk_multiplier", 1.15)) if control_panel else 1.15
    chop_risk = float(getattr(control_panel, "chop_risk_multiplier", 0.6)) if control_panel else 0.6
    high_vol_risk = float(getattr(control_panel, "high_vol_risk_multiplier", 0.5)) if control_panel else 0.5

    trend_orders = int(getattr(control_panel, "trend_max_new_orders", base_orders)) if control_panel else base_orders
    chop_orders = int(getattr(control_panel, "chop_max_new_orders", max(1, min(2, base_orders)))) if control_panel else max(1, min(2, base_orders))
    high_vol_orders = int(getattr(control_panel, "high_vol_max_new_orders", 1)) if control_panel else 1

    regime = regime_snapshot.regime

    if regime == "TREND":
        return {
            "market_regime": regime,
            "effective_risk_multiplier": round(base_risk * trend_risk, 4),
            "effective_max_new_orders_per_run": trend_orders,
        }
    if regime == "CHOP":
        return {
            "market_regime": regime,
            "effective_risk_multiplier": round(base_risk * chop_risk, 4),
            "effective_max_new_orders_per_run": chop_orders,
        }
    if regime == "HIGH_VOL":
        return {
            "market_regime": regime,
            "effective_risk_multiplier": round(base_risk * high_vol_risk, 4),
            "effective_max_new_orders_per_run": high_vol_orders,
        }

    return {
        "market_regime": regime,
        "effective_risk_multiplier": round(base_risk, 4),
        "effective_max_new_orders_per_run": base_orders,
    }
PY

########################################
# Patch control_panel.py
########################################
python - <<'PY'
from pathlib import Path
path = Path("src/trading_ensemble/config/control_panel.py")
text = path.read_text()

if "trend_risk_multiplier" not in text:
    text = text.replace(
        '    max_near_trigger_alerts: int = 10\n',
        '    max_near_trigger_alerts: int = 10\n'
        '    trend_risk_multiplier: float = 1.15\n'
        '    chop_risk_multiplier: float = 0.60\n'
        '    high_vol_risk_multiplier: float = 0.50\n'
        '    trend_max_new_orders: int = 4\n'
        '    chop_max_new_orders: int = 2\n'
        '    high_vol_max_new_orders: int = 1\n'
    )

if 'trend_risk_multiplier=_to_float(raw.get("trend_risk_multiplier"), 1.15),' not in text:
    text = text.replace(
        '        max_near_trigger_alerts=_to_int(raw.get("max_near_trigger_alerts"), 10),\n',
        '        max_near_trigger_alerts=_to_int(raw.get("max_near_trigger_alerts"), 10),\n'
        '        trend_risk_multiplier=_to_float(raw.get("trend_risk_multiplier"), 1.15),\n'
        '        chop_risk_multiplier=_to_float(raw.get("chop_risk_multiplier"), 0.60),\n'
        '        high_vol_risk_multiplier=_to_float(raw.get("high_vol_risk_multiplier"), 0.50),\n'
        '        trend_max_new_orders=_to_int(raw.get("trend_max_new_orders"), 4),\n'
        '        chop_max_new_orders=_to_int(raw.get("chop_max_new_orders"), 2),\n'
        '        high_vol_max_new_orders=_to_int(raw.get("high_vol_max_new_orders"), 1),\n'
    )

path.write_text(text)
PY

########################################
# Patch risk.py to use effective overrides
########################################
python - <<'PY'
from pathlib import Path
path = Path("src/trading_ensemble/pipeline/stages/risk.py")
text = path.read_text()

old1 = '''        max_new_orders_per_run = (
            getattr(control_panel, "max_new_orders_per_run", DEFAULT_MAX_POSITIONS)
            if control_panel is not None
            else DEFAULT_MAX_POSITIONS
        )
'''
new1 = '''        max_new_orders_per_run = int(context.get(
            "effective_max_new_orders_per_run",
            getattr(control_panel, "max_new_orders_per_run", DEFAULT_MAX_POSITIONS)
            if control_panel is not None
            else DEFAULT_MAX_POSITIONS
        ))
'''
text = text.replace(old1, new1)

old2 = '''        risk_multiplier = (
            float(getattr(control_panel, "risk_multiplier", 1.0))
            if control_panel is not None
            else 1.0
        )
'''
new2 = '''        risk_multiplier = float(
            context.get(
                "effective_risk_multiplier",
                float(getattr(control_panel, "risk_multiplier", 1.0))
                if control_panel is not None
                else 1.0
            )
        )
'''
text = text.replace(old2, new2)

if 'print(f"  market_regime           = {context.get(\'market_regime\', \'UNKNOWN\')}")' not in text:
    text = text.replace(
        '        print(f"  risk_multiplier        = {risk_multiplier:.2f}")\n',
        '        print(f"  market_regime           = {context.get(\'market_regime\', \'UNKNOWN\')}")\n'
        '        print(f"  risk_multiplier        = {risk_multiplier:.2f}")\n'
    )

path.write_text(text)
PY

########################################
# Patch dashboard/command_center.py
########################################
python - <<'PY'
from pathlib import Path
path = Path("src/trading_ensemble/dashboard/command_center.py")
text = path.read_text()

if '("market_regime", context.get("market_regime", "UNKNOWN")),' not in text:
    text = text.replace(
        '        ("force_exit_all", getattr(control_panel, "force_exit_all", False) if control_panel else False),\n',
        '        ("force_exit_all", getattr(control_panel, "force_exit_all", False) if control_panel else False),\n'
        '        ("market_regime", context.get("market_regime", "UNKNOWN")),\n'
        '        ("effective_risk_multiplier", context.get("effective_risk_multiplier", "")),\n'
        '        ("effective_max_new_orders", context.get("effective_max_new_orders_per_run", "")),\n'
    )

path.write_text(text)
PY

########################################
# Patch scheduler.py to calculate regime
########################################
python - <<'PY'
from pathlib import Path
path = Path("src/trading_ensemble/monitor/scheduler.py")
text = path.read_text()

if 'from trading_ensemble.regime.regime_engine import (' not in text:
    text = text.replace(
        'from trading_ensemble.notifications.router import (\n    send_near_trigger_alerts,\n    send_promotion_alerts,\n)\n',
        'from trading_ensemble.notifications.router import (\n    send_near_trigger_alerts,\n    send_promotion_alerts,\n)\n'
        'from trading_ensemble.regime.regime_engine import (\n    apply_regime_overrides,\n    fetch_index_regime,\n)\n'
    )

needle = '    setups_df = load_setup_signals(store)\n'
insert = '''    regime_snapshot = fetch_index_regime()
    regime_state = apply_regime_overrides(control_panel, regime_snapshot)
    context["market_regime"] = regime_state["market_regime"]
    context["effective_risk_multiplier"] = regime_state["effective_risk_multiplier"]
    context["effective_max_new_orders_per_run"] = regime_state["effective_max_new_orders_per_run"]

    print(f"  market_regime        = {context['market_regime']}")
    print(f"  effective_risk_mult  = {context['effective_risk_multiplier']}")
    print(f"  effective_max_orders = {context['effective_max_new_orders_per_run']}")

'''
if 'regime_snapshot = fetch_index_regime()' not in text:
    text = text.replace(needle, insert + needle)

path.write_text(text)
PY

echo "Phase 23 patch applied successfully."
