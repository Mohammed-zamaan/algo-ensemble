"""Google Sheets client for reading watchlist and conviction data."""
import os
from typing import List, Dict
import pandas as pd
from google.oauth2.service_account import Credentials
import gspread

SCOPES = ['https://www.googleapis.com/auth/spreadsheets']

def get_sheets_client():
    """Initialize Google Sheets client with service account."""
    creds_path = os.getenv('GOOGLE_SHEETS_CREDS_JSON', 'credentials.json')
    creds = Credentials.from_service_account_file(creds_path, scopes=SCOPES)
    return gspread.authorize(creds)

def read_master_universe(spreadsheet_id: str) -> pd.DataFrame:
    """Read MasterUniverse sheet with conviction data."""
    client = get_sheets_client()
    sheet = client.open_by_key(spreadsheet_id).worksheet('MasterUniverse')
    
    data = sheet.get_all_records()
    df = pd.DataFrame(data)
    
    # Filter only active stocks
    df = df[df['active'] == True].copy()
    
    # Map source_type to conviction (as per your spec)
    conviction_map = {'MANUAL': 1, 'ANGEL_ONE': 2, 'BOTH': 3}
    df['conviction'] = df['source_type'].map(conviction_map).fillna(1)
    
    return df[['symbol', 'nse_code', 'source_type', 'conviction', 'conviction_notes']]

def write_news_to_sheet(spreadsheet_id: str, news_data: pd.DataFrame):
    """Write yfinance news to NewsUniverse sheet."""
    client = get_sheets_client()
    sheet = client.open_by_key(spreadsheet_id).worksheet('NewsUniverse')
    sheet.clear()
    sheet.update([news_data.columns.values.tolist()] + news_data.values.tolist())
