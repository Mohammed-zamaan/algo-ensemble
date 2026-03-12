from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd


REQUIRED_COLS = {"symbol", "date", "open", "high", "low", "close", "volume"}


@dataclass(frozen=True)
class HistoricalDataRequest:
    symbol: str
    start_date: str | None = None
    end_date: str | None = None


class FileHistoricalDataProvider:
    def __init__(self, bars_path: Path):
        self.bars_path = Path(bars_path)
        self._df = self._load()

    def _load(self) -> pd.DataFrame:
        if not self.bars_path.exists():
            raise FileNotFoundError(f"Historical bars file not found: {self.bars_path}")

        suffix = self.bars_path.suffix.lower()
        if suffix == ".csv":
            df = pd.read_csv(self.bars_path)
        elif suffix in {".parquet", ".pq"}:
            df = pd.read_parquet(self.bars_path)
        else:
            raise ValueError(f"Unsupported bars file format: {self.bars_path}")

        df = self._normalize_columns(df)
        missing = REQUIRED_COLS - set(df.columns)
        if missing:
            raise ValueError(f"Historical bars missing required columns: {sorted(missing)}")

        df["symbol"] = df["symbol"].astype(str).str.strip().str.upper()
        df["date"] = pd.to_datetime(df["date"]).dt.normalize()

        numeric_cols = ["open", "high", "low", "close", "volume"]
        for col in numeric_cols:
            df[col] = pd.to_numeric(df[col], errors="coerce")

        df = df.dropna(subset=["symbol", "date", "open", "high", "low", "close", "volume"]).copy()
        df = df.sort_values(["symbol", "date"]).reset_index(drop=True)
        return df

    @staticmethod
    def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
        rename_map = {}
        for col in df.columns:
            c = str(col).strip().lower()
            if c in {"datetime", "timestamp", "time"}:
                rename_map[col] = "date"
            elif c in {"ltp", "adj close", "adj_close"}:
                rename_map[col] = "close"
            elif c in {"ticker", "code", "nse_code"}:
                rename_map[col] = "symbol"
            else:
                rename_map[col] = c
        return df.rename(columns=rename_map)

    def list_symbols(self) -> list[str]:
        return sorted(self._df["symbol"].unique().tolist())

    def trading_dates(self, start_date: str, end_date: str) -> list[pd.Timestamp]:
        start = pd.Timestamp(start_date).normalize()
        end = pd.Timestamp(end_date).normalize()
        dts = self._df.loc[(self._df["date"] >= start) & (self._df["date"] <= end), "date"].drop_duplicates()
        return sorted(dts.tolist())

    def get_bars(self, req: HistoricalDataRequest) -> pd.DataFrame:
        df = self._df[self._df["symbol"] == req.symbol].copy()
        if req.start_date:
            df = df[df["date"] >= pd.Timestamp(req.start_date).normalize()]
        if req.end_date:
            df = df[df["date"] <= pd.Timestamp(req.end_date).normalize()]
        return df.reset_index(drop=True)