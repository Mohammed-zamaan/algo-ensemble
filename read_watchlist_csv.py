import pandas as pd

def read_master_universe_csv(csv_path='watchlist.csv'):
    """Read watchlist from CSV file exported from Google Sheets."""
    df = pd.read_csv(csv_path)
    
    # Filter only active stocks
    df = df[df['active'] == True].copy()
    
    # Map source_type to conviction
    conviction_map = {'MANUAL': 1, 'ANGEL_ONE': 2, 'BOTH': 3}
    df['conviction'] = df['source_type'].map(conviction_map).fillna(1)
    
    return df[['symbol', 'nse_code', 'source_type', 'conviction', 'conviction_notes']]

if __name__ == '__main__':
    df = read_master_universe_csv()
    print(df)
