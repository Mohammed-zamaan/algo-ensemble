from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd

from src.trading_ensemble.core.engine import (
    validate_ohlcv,
    compute_indicators,
    compute_filters,
    compute_dynamic_donchian,
    compute_signals,
)
from src.trading_ensemble.core.params import StrategyParams


def find_csv_files(pattern_or_dir: str) -> List[Path]:
    p = Path(pattern_or_dir)
    if p.exists() and p.is_dir():
        files = sorted(p.glob("*.csv"))
        return files
    # treat as glob pattern (e.g., "data/test/*.csv" or "data/**/*.csv")
    files = sorted(Path(".").glob(pattern_or_dir))
    return [f for f in files if f.is_file() and f.suffix.lower() == ".csv"]


def normalize_ohlcv_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [c.strip().lower() for c in df.columns]

    # common variants -> canonical names expected by your engine
    mapping = {
        "o": "open",
        "h": "high",
        "l": "low",
        "c": "close",
        "vol": "volume",
        "qty": "volume",
        "tottrdqnty": "volume",
        "timestamp": "datetime",
        "date": "datetime",
        "time": "datetime",
    }
    for src, dst in mapping.items():
        if src in df.columns and dst not in df.columns:
            df = df.rename(columns={src: dst})

    # If both "datetime" and something like "date_time" exists, keep as-is; engine doesn't require datetime.
    return df


def analyze_one_csv(csv_path: Path, p: StrategyParams) -> Tuple[pd.DataFrame, Dict]:
    df = pd.read_csv(csv_path)
    df = normalize_ohlcv_columns(df)

    # Run full StageB pipeline
    df = validate_ohlcv(df)
    df = compute_indicators(df, p)
    df = compute_filters(df, p)
    df = compute_dynamic_donchian(df, p)
    signals = compute_signals(df)

    meta = {
        "file": str(csv_path),
        "rows_in": int(len(df)),
        "rows_signals": int(len(signals)),
        "cols_in": ",".join(list(df.columns)),
        "cols_signals": ",".join(list(signals.columns)),
    }
    return signals, meta


def safe_stem(path: Path) -> str:
    s = path.stem
    s = re.sub(r"[^A-Za-z0-9._-]+", "_", s)
    return s


def main():
    # 1) Set where your CSVs live:
    #    - Directory: "data/test_csvs"
    #    - Or pattern: "data/**/*.csv"
    source = os.getenv("CSV_SOURCE", "data")  # change default if you want
    files = find_csv_files(source)

    if not files:
        raise SystemExit(
            f"No CSV files found for CSV_SOURCE={source!r}. "
            "Set CSV_SOURCE to a folder (e.g. data/test) or a glob (e.g. 'data/**/*.csv')."
        )

    out_dir = Path("results") / "stageB"
    out_dir.mkdir(parents=True, exist_ok=True)

    p = StrategyParams()

    summary_rows = []
    for f in files:
        try:
            signals, meta = analyze_one_csv(f, p)

            out_file = out_dir / f"{safe_stem(f)}__signals.csv"
            signals.to_csv(out_file, index=False)

            meta["out_file"] = str(out_file)
            summary_rows.append(meta)

            print(f"[OK] {f} -> signals={len(signals)} -> {out_file}")
        except Exception as e:
            summary_rows.append({"file": str(f), "error": repr(e)})
            print(f"[ERR] {f} -> {e!r}")

    summary = pd.DataFrame(summary_rows)
    summary_path = out_dir / "_summary.csv"
    summary.to_csv(summary_path, index=False)
    print(f"Wrote summary: {summary_path}")

    # Optional: combine all signal files into one (handy for quick inspection)
    signal_files = sorted(out_dir.glob("*__signals.csv"))
    if signal_files:
        combined = pd.concat((pd.read_csv(x) for x in signal_files), ignore_index=True)
        combined_path = out_dir / "_all_signals_combined.csv"
        combined.to_csv(combined_path, index=False)
        print(f"Wrote combined signals: {combined_path}")


if __name__ == "__main__":
    main()
