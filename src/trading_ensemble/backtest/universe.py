from __future__ import annotations

from pathlib import Path

import pandas as pd


def load_universe(name: str, symbols: list[str] | None = None, watchlist_file: Path | None = None) -> list[str]:
    """Load universe from explicit symbols or exported MasterWatchlist file.

    Precedence:
    1) explicit `symbols`
    2) `watchlist_file` with symbol-like column (`symbol`, `nse_code`, or `code`)
    3) empty list for default scaffold behavior
    """

    if symbols:
        return sorted({str(s).strip().upper() for s in symbols if str(s).strip()})

    if watchlist_file is not None and watchlist_file.exists():
        if watchlist_file.suffix.lower() == ".parquet":
            df = pd.read_parquet(watchlist_file)
        else:
            df = pd.read_csv(watchlist_file)
        cols = {str(c).strip().lower(): c for c in df.columns}
        col = cols.get("symbol") or cols.get("nse_code") or cols.get("code")
        if col is None:
            raise ValueError("watchlist file missing symbol/nse_code/code column")
        return sorted({str(v).strip().upper() for v in df[col].dropna().tolist() if str(v).strip()})

    _ = name
    return []
