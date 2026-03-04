import os
import pandas as pd
from dotenv import load_dotenv
load_dotenv()

USE_SHEETS     = os.getenv("USE_SHEETS", "false").lower() == "true"
SPREADSHEET_ID = os.getenv("SPREADSHEET_ID", "")
CREDS_PATH     = os.getenv("GOOGLE_CREDS_PATH", "credentials.json")
WATCHLIST_CSV  = os.getenv("WATCHLIST_CSV", "watchlist.csv")
SCOPES = ["https://spreadsheets.google.com/feeds","https://www.googleapis.com/auth/drive"]
CONVICTION_MULT = {1: 0.9, 2: 1.0, 3: 1.1}

def get_sheets_client():
    import gspread
    from google.oauth2.service_account import Credentials
    creds = Credentials.from_service_account_file(CREDS_PATH, scopes=SCOPES)
    return gspread.authorize(creds)

def read_master_universe():
    if USE_SHEETS:
        return _read_from_sheets()
    return _read_from_csv()

def _read_from_sheets():
    print("  [SHEETS] Reading MasterUniverse from Google Sheets...")
    client = get_sheets_client()
    sh = client.open_by_key(SPREADSHEET_ID)
    ws = sh.worksheet("MasterUniverse")
    df = pd.DataFrame(ws.get_all_records())
    df = _clean_universe(df)
    print(f"  [SHEETS] {len(df)} active stocks loaded")
    return df

def _read_from_csv():
    if not os.path.exists(WATCHLIST_CSV):
        _create_watchlist_template()
    print(f"  [SHEETS] Reading watchlist from {WATCHLIST_CSV}...")
    df = pd.read_csv(WATCHLIST_CSV)
    df = _clean_universe(df)
    print(f"  [SHEETS] {len(df)} active stocks loaded from CSV")
    return df

def _clean_universe(df):
    required = ["symbol","source_type","conviction","active"]
    for col in required:
        if col not in df.columns:
            raise ValueError(f"[SHEETS] Missing column: {col}")
    df["symbol"]      = df["symbol"].str.strip().str.upper()
    df["source_type"] = df["source_type"].str.strip().str.upper()
    df["conviction"]  = pd.to_numeric(df["conviction"], errors="coerce").fillna(1).astype(int)
    df["active"]      = df["active"].astype(str).str.upper().isin(["TRUE","1","YES"])
    df = df[df["active"] == True].copy()
    df["symbol"]      = df["symbol"].apply(lambda s: s if s.endswith("-EQ") else s + "-EQ")
    df["conviction_mult"] = df["conviction"].map(CONVICTION_MULT).fillna(1.0)
    df.reset_index(drop=True, inplace=True)
    return df

def _create_watchlist_template():
    pd.DataFrame([
        {"symbol":"TATASTEEL-EQ","source_type":"MANUAL","conviction":2,"active":True,"mode":"INTRADAY","notes":"Example"},
        {"symbol":"SBIN-EQ","source_type":"ANGEL_ONE","conviction":1,"active":True,"mode":"INTRADAY","notes":"Example"},
        {"symbol":"HDFCBANK-EQ","source_type":"BOTH","conviction":3,"active":True,"mode":"INTRADAY","notes":"Example"},
    ]).to_csv(WATCHLIST_CSV, index=False)
    print(f"  [SHEETS] Template created: {WATCHLIST_CSV}")

def write_layer_output(sheet_name, df):
    if not USE_SHEETS or df.empty:
        return
    try:
        client = get_sheets_client()
        sh = client.open_by_key(SPREADSHEET_ID)
        ws = sh.worksheet(sheet_name)
        ws.clear()
        ws.update([df.columns.tolist()] + df.values.tolist())
        print(f"  [SHEETS] Written {len(df)} rows to {sheet_name}")
    except Exception as e:
        print(f"  [SHEETS] Write failed for {sheet_name}: {e}")

def get_conviction_for_symbol(symbol, universe_df):
    row = universe_df[universe_df["symbol"] == symbol]
    if row.empty:
        return "MANUAL", 2, 1.0
    return str(row["source_type"].iloc[0]), int(row["conviction"].iloc[0]), float(row["conviction_mult"].iloc[0])
