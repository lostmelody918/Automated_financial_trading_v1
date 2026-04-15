import os
from FinMind.data import DataLoader
from database import DatabaseManager
from datetime import datetime
import pandas as pd

class DataFetcher:
    """Fetches financial data from FinMind and saves it to the local database."""
    
    def __init__(self, api_token: str = None):
        """Initializes the DataLoader, optionally with an API token for higher rate limits."""
        self.dl = DataLoader()
        if api_token:
            self.dl.login_by_token(api_token)
        self.db = DatabaseManager()

    def fetch_stock_daily(self, stock_id: str, start_date: str, end_date: str = None):
        """Fetches daily stock prices for a given stock ID and date range."""
        if not end_date:
            end_date = datetime.today().strftime('%Y-%m-%d')
            
        print(f"Fetching data for {stock_id} from {start_date} to {end_date}...")
        try:
            # FinMind DataLoader
            df = self.dl.taiwan_stock_daily(
                stock_id=stock_id,
                start_date=start_date,
                end_date=end_date
            )
            
            if not df.empty:
                # Rename columns for standardization
                df.rename(columns={
                    'Trading_Volume': 'Volume',
                    'Trading_money': 'Turnover',
                    'open': 'Open',
                    'max': 'High',
                    'min': 'Low',
                    'close': 'Close',
                    'spread': 'Change',
                    'Trading_turnover': 'Transactions'
                }, inplace=True, errors='ignore')
                
                # Enforce correct types
                for col in ['Open', 'High', 'Low', 'Close', 'Volume', 'Turnover', 'Change']:
                    if col in df.columns:
                        df[col] = pd.to_numeric(df[col], errors='coerce')
                
                table_name = f"stock_{stock_id}_daily"
                self.db.save_dataframe(df, table_name, if_exists='replace')
                return df
            else:
                print(f"No data returned for {stock_id}.")
                return pd.DataFrame()
                
        except Exception as e:
            print(f"Error fetching data for {stock_id}: {e}")
            return pd.DataFrame()

if __name__ == "__main__":
    fetcher = DataFetcher()
    # Let's fetch TSMC (2330) data from 2020-01-01 to today as a test
    print("Testing data fetcher...")
    tsmc_df = fetcher.fetch_stock_daily('2330', '2023-01-01')
    if not tsmc_df.empty:
        print(f"Head of TSMC data:\n{tsmc_df.head()}")