from __future__ import annotations
import re
import pandas as pd
from comet_ml.api import API

COMET_WORKSPACE = "zamaan"
COMET_PROJECT   = "stock-screener"

def get_top_stocks(min_score: float = 60) -> pd.DataFrame:
    api  = API()
    exps = api.get_experiments(COMET_WORKSPACE, COMET_PROJECT)
    records = []
    for exp in exps:
        name = exp.name or ""
        m = re.match(r"^([A-Z&]+)-EQ", name)
        if not m:
            continue
        symbol  = m.group(1)
        metrics = {mt["name"]: mt["valueCurrent"]
                   for mt in exp.get_metrics_summary() or []}
        score   = metrics.get("composite_score", 0)
        if score >= min_score:
            records.append({"symbol": symbol + ".NS", "composite_score": score})
    return pd.DataFrame(records).sort_values("composite_score", ascending=False)
