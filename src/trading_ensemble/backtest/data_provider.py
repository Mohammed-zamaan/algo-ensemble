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

        df = normalize_bar_frame(df)
        missing = REQUIRED_COLS - set(df.columns)
        if missing:
            raise ValueError(f"Historical bars missing required columns: {sorted(missing)}")

        return df

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


def normalize_bar_frame(df: pd.DataFrame) -> pd.DataFrame:
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
    df = df.rename(columns=rename_map)

    df["symbol"] = df["symbol"].astype(str).str.strip().str.upper()
    df["date"] = pd.to_datetime(df["date"]).dt.tz_localize(None).dt.normalize()

    numeric_cols = ["open", "high", "low", "close", "volume"]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(subset=["symbol", "date", "open", "high", "low", "close", "volume"]).copy()
    df = df.sort_values(["symbol", "date"]).drop_duplicates(subset=["symbol", "date"], keep="last").reset_index(drop=True)
    return df


def fetch_and_cache_smartapi_daily_bars(
    *,
    symbols: list[str],
    start_date: str,
    end_date: str,
    cache_dir: Path,
    exchange: str = "NSE",
    interval: str = "ONE_DAY",
) -> Path:
    if not symbols:
        raise ValueError("symbols are required when data_source='smartapi_cache'")

    from trading_ensemble.data.smartapi_client import (
        fetch_candles_chunked,
        load_scrip_master,
        login_from_env,
        resolve_symbol_to_token_offline,
    )

    cache_dir = Path(cache_dir)
    raw_dir = cache_dir / "smartapi_daily"
    raw_dir.mkdir(parents=True, exist_ok=True)

    session = login_from_env()
    scrip_df = load_scrip_master()
    frames: list[pd.DataFrame] = []

    start_ts = pd.Timestamp(start_date).normalize()
    end_ts = pd.Timestamp(end_date).normalize()
    from_str = start_ts.strftime("%Y-%m-%d 0915")
    to_str = end_ts.strftime("%Y-%m-%d 1530")

    for symbol in sorted({str(s).strip().upper() for s in symbols if str(s).strip()}):
        resolved_symbol, token = resolve_symbol_to_token_offline(symbol, exchange=exchange, scrip_df=scrip_df)
        cache_file = raw_dir / f"{resolved_symbol}.parquet"

        existing = pd.DataFrame(columns=list(REQUIRED_COLS))
        if cache_file.exists():
            existing = normalize_bar_frame(pd.read_parquet(cache_file))

        fetched = fetch_candles_chunked(
            session.smart,
            exchange=exchange,
            symbol_token=str(token),
            interval=interval,
            start=from_str,
            end=to_str,
            chunk_days=60,
            sleep_seconds=1.0,
            max_retries=8,
        )
        if fetched.empty and existing.empty:
            continue

        if not fetched.empty:
            fetched = fetched.rename(columns={"datetime": "date"}).copy()
            fetched["symbol"] = resolved_symbol
            fetched = fetched[["symbol", "date", "open", "high", "low", "close", "volume"]]

        merged = pd.concat([existing, fetched], ignore_index=True) if not fetched.empty else existing
        merged = normalize_bar_frame(merged)
        merged = merged[(merged["date"] >= start_ts) & (merged["date"] <= end_ts)].copy()
        merged.to_parquet(cache_file, index=False)
        frames.append(merged)

    if not frames:
        raise ValueError("No SmartAPI historical bars were fetched for the requested symbols/date range")

    combined = normalize_bar_frame(pd.concat(frames, ignore_index=True))
    combined_path = cache_dir / "combined_daily_bars.parquet"
    combined.to_parquet(combined_path, index=False)
    return combined_path
