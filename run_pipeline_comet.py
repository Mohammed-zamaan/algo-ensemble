# run_pipeline_comet.py
from __future__ import annotations
import os
from pathlib import Path
from dotenv import load_dotenv
import pandas as pd
import comet_ml
from comet_ml import Experiment

load_dotenv()

from src.trading_ensemble.core.engine import (
    validate_ohlcv, compute_indicators,
    compute_filters, compute_dynamic_donchian, compute_signals,
)
from src.trading_ensemble.core.params import StrategyParams


def run_experiment(csv_path: Path, p: StrategyParams) -> None:
    # Start Comet experiment - auto-logs system metrics, code, git hash
    experiment = Experiment(
        api_key=os.getenv("COMET_API_KEY"),
        project_name=os.getenv("COMET_PROJECT_NAME", "algo-ensemble"),
    )
    experiment.set_name(csv_path.stem)  # Name = CSV filename

    # Log strategy parameters
    experiment.log_parameters({
        "csv_file": csv_path.name,
        "donchian_period": getattr(p, "donchian_period", "dynamic"),
        "atr_period": getattr(p, "atr_period", None),
        "session_start": getattr(p, "session_start_hhmm", None),
        "session_end": getattr(p, "session_end_hhmm", None),
    })

    try:
        # Run full StageB pipeline
        df = pd.read_csv(csv_path)
        df.columns = [c.strip().lower() for c in df.columns]
        df = validate_ohlcv(df)
        df = compute_indicators(df, p)
        df = compute_filters(df, p)
        df = compute_dynamic_donchian(df, p)
        signals = compute_signals(df)

        # Log metrics to Comet dashboard
        experiment.log_metrics({
            "total_candles": len(df),
            "total_signals": len(signals),
            "signal_rate_pct": round(len(signals) / len(df) * 100, 2) if len(df) > 0 else 0,
            "buy_signals": int((signals.get("signal", signals.get("side", pd.Series())) == 1).sum()) if "signal" in signals.columns or "side" in signals.columns else 0,
        })

        # Save + log signals CSV as artifact
        out_dir = Path("results/stageB")
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"{csv_path.stem}__signals.csv"
        signals.to_csv(out_path, index=False)
        experiment.log_asset(str(out_path), file_name=out_path.name)

        print(f"✅ {csv_path.name}: {len(df)} candles → {len(signals)} signals")

    except Exception as e:
        experiment.log_other("error", str(e))
        print(f"❌ {csv_path.name}: {e!r}")

    finally:
        experiment.end()


def main():
    source = Path(os.getenv("CSV_SOURCE", "data"))
    files = sorted(source.glob("*.csv"))

    if not files:
        raise SystemExit(f"No CSVs found in {source}/")

    print(f"Found {len(files)} CSV files → running {len(files)} Comet experiments")
    p = StrategyParams()

    for f in files:
        run_experiment(f, p)

    print("\n🚀 All experiments logged → https://www.comet.com")


if __name__ == "__main__":
    main()
