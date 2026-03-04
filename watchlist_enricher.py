# watchlist_enricher.py
# ─────────────────────────────────────────────────────────────────
# Run once after adding new symbols to column A of the sheet.
# Auto-fills: sector, industry, company name, market cap, ISIN, series.
# Preserves: active flag, notes, manually entered sectors.
# Zero hardcoded symbols. Zero hardcoded sectors.
#
# Usage:
#   python watchlist_enricher.py              # enrich only new/blank rows
#   python watchlist_enricher.py --force      # re-enrich everything
# ─────────────────────────────────────────────────────────────────

from __future__ import annotations
import argparse, io, os, time
from datetime import datetime
from pathlib import Path

import pandas as pd
import requests
import yfinance as yf
from dotenv import load_dotenv

load_dotenv(dotenv_path="/workspaces/algo-ensemble/.env")

GSHEET_ID  = os.getenv("WATCHLIST_GSHEET_ID", "")
GSHEET_GID = os.getenv("WATCHLIST_GSHEET_GID", "0")
LOCAL_CSV  = Path("data/watchlist_master.csv")

SHEET_COLUMNS = [
    "symbol", "nse_code", "active", "company_name",
    "sector", "industry", "market_cap_cr",
    "isin", "series", "last_updated", "notes",
]


def to_nse_code(symbol: str) -> str:
    return symbol.replace("-EQ", "").replace("-BE", "").strip().upper()


def fetch_jugaad(code: str) -> dict:
    try:
        from jugaad_trader.nse import NSELive
        q = NSELive().stock_quote(code)
        time.sleep(0.4)
        return {
            "isin":         q.get("info", {}).get("isin", ""),
            "series":       q.get("metadata", {}).get("series", "EQ"),
            "sector":       q.get("industryInfo", {}).get("macro", ""),
            "industry":     q.get("industryInfo", {}).get("sector", ""),
            "company_name": q.get("info", {}).get("companyName", ""),
        }
    except Exception as e:
        print(f"      jugaad failed ({code}): {e}")
        return {}


def fetch_yfinance(code: str) -> dict:
    try:
        info = yf.Ticker(f"{code}.NS").info
        mc = info.get("marketCap")
        time.sleep(0.3)
        return {
            "sector":        info.get("sector", ""),
            "industry":      info.get("industry", ""),
            "market_cap_cr": round(mc / 1e7, 1) if mc else "",
            "company_name":  info.get("longName", ""),
        }
    except Exception as e:
        print(f"      yfinance failed ({code}): {e}")
        return {}


def pick(*values) -> str:
    for v in values:
        if v and str(v).strip() and str(v).strip().upper() not in ("NAN", "NONE", ""):
            return str(v).strip()
    return ""


def enrich_one(symbol: str, existing: dict | None = None) -> dict:
    code = to_nse_code(symbol)
    jug  = fetch_jugaad(code)
    yfd  = fetch_yfinance(code)

    # Sector priority: jugaad → yfinance → existing manual → blank
    sector   = pick(jug.get("sector"),   yfd.get("sector"))
    industry = pick(jug.get("industry"), yfd.get("industry"))
    company  = pick(jug.get("company_name"), yfd.get("company_name"))
    isin     = pick(jug.get("isin"))
    series   = pick(jug.get("series"), "EQ")
    mcap     = pick(str(yfd.get("market_cap_cr", "")))

    # Preserve manually set sector (don't overwrite if user filled it)
    if existing:
        ex_sector = str(existing.get("sector", "")).strip().upper()
        if ex_sector and ex_sector not in ("", "UNKNOWN", "NAN"):
            sector = ex_sector  # keep manual entry

    return {
        "symbol":        symbol.upper(),
        "nse_code":      code,
        "active":        existing.get("active", "TRUE") if existing else "TRUE",
        "company_name":  company,
        "sector":        sector.upper() if sector else "",
        "industry":      industry,
        "market_cap_cr": mcap,
        "isin":          isin,
        "series":        series,
        "last_updated":  datetime.now().strftime("%Y-%m-%d %H:%M"),
        "notes":         existing.get("notes", "") if existing else "",
    }


def read_sheet() -> tuple[list[str], dict]:
    df = None
    if GSHEET_ID:
        try:
            url = (f"https://docs.google.com/spreadsheets/d/{GSHEET_ID}"
                   f"/export?format=csv&gid={GSHEET_GID}")
            r = requests.get(url, timeout=10)
            r.raise_for_status()
            df = pd.read_csv(io.StringIO(r.text))
            df.columns = [c.strip().lower() for c in df.columns]
        except Exception as e:
            print(f"  GSheets unavailable: {e}")

    if df is None and LOCAL_CSV.exists():
        df = pd.read_csv(LOCAL_CSV)
        df.columns = [c.strip().lower() for c in df.columns]

    if df is None or df.empty:
        raise RuntimeError(
            "No sheet found.\n"
            "Create data/watchlist_master.csv with a 'symbol' column,\n"
            "add your NSE symbols (e.g. ABB-EQ), then run this script."
        )

    if "symbol" not in df.columns:
        raise ValueError("Sheet must have a 'symbol' column in row 1.")

    df = df[df["symbol"].notna() & (df["symbol"].astype(str).str.strip() != "")]
    df["symbol"] = df["symbol"].str.strip().str.upper()

    symbols  = df["symbol"].tolist()
    existing = {row["symbol"]: row.to_dict() for _, row in df.iterrows()}
    return symbols, existing


def write_back(enriched_df: pd.DataFrame):
    for col in SHEET_COLUMNS:
        if col not in enriched_df.columns:
            enriched_df[col] = ""
    enriched_df = enriched_df[SHEET_COLUMNS]

    # Always save local CSV
    os.makedirs(LOCAL_CSV.parent, exist_ok=True)
    enriched_df.to_csv(LOCAL_CSV, index=False)
    print(f"\n  [Saved] Local CSV → {LOCAL_CSV}")

    # Write back to Google Sheet if service account available
    creds_path = os.getenv("GOOGLE_SERVICE_ACCOUNT_JSON", "")
    if not creds_path or not Path(creds_path).exists():
        print("  [GSheets] Write-back skipped — no service account JSON.")
        print("  Set GOOGLE_SERVICE_ACCOUNT_JSON in .env to enable auto write-back.")
        return

    try:
        import gspread
        from google.oauth2.service_account import Credentials
        creds = Credentials.from_service_account_file(
            creds_path,
            scopes=["https://www.googleapis.com/auth/spreadsheets"],
        )
        gc = gspread.authorize(creds)
        ws = gc.open_by_key(GSHEET_ID).get_worksheet(int(GSHEET_GID))
        data = [enriched_df.columns.tolist()] + enriched_df.fillna("").values.tolist()
        ws.clear()
        ws.update("A1", data)
        print(f"  [GSheets] {len(enriched_df)} rows written back ✓")
    except Exception as e:
        print(f"  [GSheets Write] Failed: {e}")


def run_enricher(force_all: bool = False):
    print(f"\n{'='*60}")
    print(f"  WATCHLIST ENRICHER  ({datetime.now().strftime('%
