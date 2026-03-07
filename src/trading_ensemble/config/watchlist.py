from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List

import pandas as pd


@dataclass(frozen=True)
class WatchlistSymbol:
    symbol: str
    active: bool = True
    sector: str = "UNKNOWN"
    priority: int = 0
    allowed_modes: str = ""
    force_override: str = ""
    conviction: int = 2
    notes: str = ""


REQUIRED_COLUMNS = {"symbol"}


def _normalize_df(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [str(c).strip().lower() for c in df.columns]

    missing = REQUIRED_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(f"Watchlist missing required columns: {sorted(missing)}")

    defaults = {
        "active": True,
        "sector": "UNKNOWN",
        "priority": 0,
        "allowed_modes": "",
        "force_override": "",
        "conviction": 2,
        "notes": "",
    }

    for col, default in defaults.items():
        if col not in df.columns:
            df[col] = default

    return df


def _to_symbols(df: pd.DataFrame) -> List[WatchlistSymbol]:
    symbols: List[WatchlistSymbol] = []

    for _, row in df.iterrows():
        symbol = str(row["symbol"]).strip().upper()
        if not symbol:
            continue

        active_raw = row["active"]
        if isinstance(active_raw, bool):
            active = active_raw
        else:
            active = str(active_raw).strip().lower() in {"1", "true", "yes", "y"}

        allowed_modes = str(row["allowed_modes"]).strip().upper()
        force_override = str(row["force_override"]).strip().upper()

        symbols.append(
            WatchlistSymbol(
                symbol=symbol,
                active=active,
                sector=str(row["sector"]).strip().upper() or "UNKNOWN",
                priority=int(row["priority"]) if str(row["priority"]).strip() else 0,
                allowed_modes=allowed_modes,
                force_override=force_override,
                conviction=int(row["conviction"]) if str(row["conviction"]).strip() else 2,
                notes=str(row["notes"]).strip(),
            )
        )

    return [s for s in symbols if s.active]


def load_watchlist_csv(path: str | Path) -> List[WatchlistSymbol]:
    csv_path = Path(path)
    if not csv_path.exists():
        raise FileNotFoundError(f"Watchlist file not found: {csv_path}")

    df = pd.read_csv(csv_path)
    df = _normalize_df(df)
    return _to_symbols(df)


def load_watchlist_google_service_account(
    spreadsheet_id: str,
    worksheet_name: str,
    service_account_file: str,
) -> List[WatchlistSymbol]:
    import gspread

    gc = gspread.service_account(filename=service_account_file)
    sheet = gc.open_by_key(spreadsheet_id)
    worksheet = sheet.worksheet(worksheet_name)

    records = worksheet.get_all_records()
    df = pd.DataFrame(records)
    df = _normalize_df(df)
    return _to_symbols(df)


def load_master_watchlist(settings) -> List[WatchlistSymbol]:
    source = settings.watchlist_source.strip().lower()

    if source == "google_sheets":
        return load_watchlist_google_service_account(
            spreadsheet_id=settings.watchlist_gsheet_id,
            worksheet_name=settings.watchlist_gsheet_tab,
            service_account_file=settings.google_service_account_json,
        )

    return load_watchlist_csv(settings.watchlist_csv_path)