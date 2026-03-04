"""Google Sheets client using OAuth2 user authentication (no service account)."""
import os
import pickle
from pathlib import Path
from typing import List, Dict
import pandas as pd
from google.auth.transport.requests import Request
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build

SCOPES = ['https://www.googleapis.com/auth/spreadsheets']

def get_sheets_service():
    """Authenticate using OAuth2 and return Sheets API service."""
    creds = None
    token_path = 'token.pickle'
    
    # Token stores user's access/refresh tokens
    if os.path.exists(token_path):
        with open(token_path, 'rb') as token:
            creds = pickle.load(token)
    
    # If no valid credentials, let user log in
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            flow = InstalledAppFlow.from_client_secrets_file(
                'credentials_oauth.json', SCOPES)
            creds = flow.run_local_server(port=0)
        
        # Save credentials for next run
        with open(token_path, 'wb') as token:
            pickle.dump(creds, token)
    
    service = build('sheets', 'v4', credentials=creds)
    return service

def read_master_universe(spreadsheet_id: str) -> pd.DataFrame:
    """Read MasterUniverse sheet with conviction data."""
    service = get_sheets_service()
    
    # Read data from MasterUniverse sheet
    result = service.spreadsheets().values().get(
        spreadsheetId=spreadsheet_id,
        range='MasterUniverse!A:M'
    ).execute()
    
    values = result.get('values', [])
    if not values:
        return pd.DataFrame()
    
    # Convert to DataFrame
    df = pd.DataFrame(values[1:], columns=values[0])
    
    # Filter only active stocks
    df['active'] = df['active'].map({'TRUE': True, 'FALSE': False, True: True, False: False})
    df = df[df['active'] == True].copy()
    
    # Map source_type to conviction
    conviction_map = {'MANUAL': 1, 'ANGEL_ONE': 2, 'BOTH': 3}
    df['conviction'] = df['source_type'].map(conviction_map).fillna(1).astype(int)
    
    return df[['symbol', 'nse_code', 'source_type', 'conviction', 'conviction_notes']]

def write_news_to_sheet(spreadsheet_id: str, news_data: pd.DataFrame):
    """Write yfinance news to NewsUniverse sheet."""
    service = get_sheets_service()
    
    # Clear existing data
    service.spreadsheets().values().clear(
        spreadsheetId=spreadsheet_id,
        range='NewsUniverse!A:Z'
    ).execute()
    
    # Prepare data
    values = [news_data.columns.tolist()] + news_data.values.tolist()
    
    # Write new data
    service.spreadsheets().values().update(
        spreadsheetId=spreadsheet_id,
        range='NewsUniverse!A1',
        valueInputOption='RAW',
        body={'values': values}
    ).execute()
