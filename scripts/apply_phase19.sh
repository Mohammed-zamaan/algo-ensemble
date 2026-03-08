#!/usr/bin/env bash
set -e

echo "Applying Phase 19 patches..."

PROJECT_ROOT=$(pwd)

########################################
# 1. Create pipeline summary module
########################################

mkdir -p src/trading_ensemble/pipeline

cat << 'EOF' > src/trading_ensemble/pipeline/summary.py
from __future__ import annotations

from datetime import datetime
import pandas as pd


def build_run_summary(context) -> pd.DataFrame:
    control_panel = context.get("control_panel")
    candidates_df = context.get("candidates_df", pd.DataFrame())
    signals_df = context.get("signals_df", pd.DataFrame())
    orders_df = context.get("orders_df", pd.DataFrame())
    execution_df = context.get("execution_results_df", pd.DataFrame())
    watchlist = context.get("watchlist", [])
    comet_ranked_count = context.get("comet_ranked_count", 0)

    confirmed_count = 0
    setup_count = 0

    if not signals_df.empty and "SIGNAL_STATUS" in signals_df.columns:
        confirmed_count = int((signals_df["SIGNAL_STATUS"] == "CONFIRMED").sum())
        setup_count = int((signals_df["SIGNAL_STATUS"] == "SETUP").sum())

    rows = [
        {"metric": "run_time", "value": datetime.now().strftime("%Y-%m-%d %H:%M:%S")},
        {"metric": "watchlist_symbols", "value": len(watchlist)},
        {"metric": "comet_ranked_symbols", "value": comet_ranked_count},
        {"metric": "shortlisted_candidates", "value": len(candidates_df)},
        {"metric": "setup_signals", "value": setup_count},
        {"metric": "confirmed_signals", "value": confirmed_count},
        {"metric": "approved_orders", "value": len(orders_df)},
        {"metric": "executed_orders", "value": len(execution_df)},
        {"metric": "system_trading_enabled", "value": getattr(control_panel, "system_trading_enabled", True) if control_panel else True},
        {"metric": "pause_new_entries", "value": getattr(control_panel, "pause_new_entries", False) if control_panel else False},
        {"metric": "force_exit_all", "value": getattr(control_panel, "force_exit_all", False) if control_panel else False},
        {"metric": "paper_trade", "value": getattr(control_panel, "paper_trade", True) if control_panel else True},
    ]

    return pd.DataFrame(rows)
EOF


########################################
# 2. Patch engine to return context
########################################

sed -i 's/print("Pipeline finished")/print("Pipeline finished")\n        return context/' \
src/trading_ensemble/pipeline/engine.py || true


########################################
# 3. Patch elimination stage
########################################

sed -i '/Comet ranked symbols fetched/a\
        context["comet_ranked_count"] = len(top_df)
' src/trading_ensemble/pipeline/stages/elimination.py || true


########################################
# 4. Patch signals stage sorting
########################################

sed -i '/signals_df = pd.DataFrame(signals)/a\
        if not signals_df.empty:\n\
            signals_df["STATUS_SORT"] = signals_df["SIGNAL_STATUS"].map({"CONFIRMED": 0, "SETUP": 1}).fillna(9)\n\
            signals_df = signals_df.sort_values(\n\
                ["STATUS_SORT", "TRIGGER_READINESS_SCORE", "SETUP_QUALITY_SCORE", "COMPOSITE_SCORE"],\n\
                ascending=[True, False, False, False],\n\
            ).drop(columns=["STATUS_SORT"]).reset_index(drop=True)\n' \
src/trading_ensemble/pipeline/stages/signals.py || true


########################################
# 5. Patch CLI pipeline to write summary
########################################

sed -i '/PipelineEngine/a\
from trading_ensemble.pipeline.summary import build_run_summary\n\
from trading_ensemble.data.sheets_output import maybe_write_output\n' \
src/trading_ensemble/cli/pipeline.py || true

sed -i '/engine.run()/a\
    summary_df = build_run_summary(context)\n\
    maybe_write_output(context["settings"], context.get("control_panel"), "RunSummary", summary_df)\n' \
src/trading_ensemble/cli/pipeline.py || true


echo "Phase 19 patches applied successfully."