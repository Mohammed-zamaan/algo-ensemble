from __future__ import annotations

import pandas as pd

from trading_ensemble.config.control_panel import load_control_panel
from trading_ensemble.config.watchlist import load_master_watchlist
from ..engine import PipelineStage


class PremarketStage(PipelineStage):
    name = "premarket"

    def run(self, context):
        settings = context["settings"]

        print("Running premarket data preparation")

        try:
            control_panel = load_control_panel(settings)
            context["control_panel"] = control_panel
            print("Loaded ControlPanel")
        except Exception as e:
            print(f"Failed to load ControlPanel: {e}")
            context["control_panel"] = None

        try:
            watchlist = load_master_watchlist(settings)

            if not watchlist:
                print("Watchlist loaded but empty")

            symbols = [s.symbol for s in watchlist]
            watchlist_df = pd.DataFrame([w.__dict__ for w in watchlist])

            context["watchlist"] = watchlist
            context["watchlist_symbols"] = symbols
            context["watchlist_df"] = watchlist_df

            print(f"Loaded {len(symbols)} watchlist symbols")
        except Exception as e:
            print(f"Failed to load watchlist: {e}")
            context["watchlist"] = []
            context["watchlist_symbols"] = []
            context["watchlist_df"] = pd.DataFrame()

        context["premarket_ready"] = True