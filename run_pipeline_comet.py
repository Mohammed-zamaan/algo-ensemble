from __future__ import annotations
import os
from pathlib import Path
from dotenv import load_dotenv
import pandas as pd
from comet_ml import Experiment

load_dotenv()

from src.trading_ensemble.core.engine import (
    validate_ohlcv, compute_indicators,
    compute_filters, compute_dynamic_donchian, compute_signals,
)
from src.trading_ensemble.core.params import StrategyParams


def run_experiment(csv_path: Path, p: StrategyParams) -> None:
    experiment = Experiment(
        api_key=os.getenv("COMET_API_KEY"),
        project_name=os.getenv("COMET_PROJECT_NAME", "algo-ensemble"),
        workspace=os.getenv("COMET_WORKSPACE", "zamaan"),
    )
    experiment.set_name(csv_path.stem)

    experiment.log_parameters({
        "csv_file": csv_path.name,
        "symbol": csv_path.name.split("_")[0],
        "timeframe": "15m",
    })

    try:
        df = pd.read_csv(csv_path)
        p = StrategyParams()

        df = validate_ohlcv(df)
        df = compute_indicators(df, p)
        df = compute_filters(df, p)
        df = compute_dynamic_donchian(df, p)
        all_rows = compute_signals(df)

        # ✅ Filter only actual signal rows
        long_entries = all_rows[all_rows["long_signal"] == True]
        exits        = all_rows[all_rows["don_exit_signal"] == True]
        all_signals  = all_rows[all_rows["long_signal"] | all_rows["don_exit_signal"]]

        experiment.log_metrics({
            "total_candles":   len(all_rows),
            "long_entries":    len(long_entries),
            "exits":           len(exits),
            "total_signals":   len(all_signals),
            "signal_rate_pct": round(len(all_signals) / len(all_rows) * 100, 2),
        })

        # Save filtered signals
        out_dir = Path("results/stageB")
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"{csv_path.stem}__signals.csv"
        all_signals.to_csv(out_path, index=False)
        experiment.log_asset(str(out_path), file_name=out_path.name)

        print(f"✅ {csv_path.name}: {len(all_rows)} candles → {len(long_entries)} entries / {len(exits)} exits")

    except Exception as e:
        experiment.log_other("error", str(e))
        print(f"❌ {csv_path.name}: {e!r}")
    finally:
        experiment.end()


def main():
    source = Path(os.getenv("CSV_SOURCE", "trading_candles"))
    files = sorted(source.glob("*.csv"))

    if not files:
        raise SystemExit(f"No CSVs found in {source}/")

    print(f"Found {len(files)} CSV files → running {len(files)} Comet experiments\n")
    p = StrategyParams()

    for f in files:
        run_experiment(f, p)

    print("\n🚀 All done → https://www.comet.com/zamaan/algo-ensemble")


if __name__ == "__main__":
    main()
