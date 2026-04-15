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
        """Fetches daily stock prices and institutional data, then merges them."""
        if not end_date:
            end_date = datetime.today().strftime('%Y-%m-%d')
            
        print(f"Fetching price data for {stock_id} from {start_date} to {end_date}...")
        try:
            # 1. Fetch Daily Prices
            df_price = self.dl.taiwan_stock_daily(
                stock_id=stock_id,
                start_date=start_date,
                end_date=end_date
            )
            
            if df_price.empty:
                print(f"No price data returned for {stock_id}.")
                return pd.DataFrame()

            df_price.rename(columns={
                'Trading_Volume': 'Volume',
                'Trading_money': 'Turnover',
                'open': 'Open',
                'max': 'High',
                'min': 'Low',
                'close': 'Close',
                'spread': 'Change',
                'Trading_turnover': 'Transactions'
            }, inplace=True, errors='ignore')
            
            for col in ['Open', 'High', 'Low', 'Close', 'Volume', 'Turnover', 'Change']:
                if col in df_price.columns:
                    df_price[col] = pd.to_numeric(df_price[col], errors='coerce')

            # 2. Fetch Institutional Investors Data
            print(f"Fetching institutional investors data for {stock_id}...")
            df_inst = self.dl.taiwan_stock_institutional_investors(
                stock_id=stock_id,
                start_date=start_date,
                end_date=end_date
            )

            if not df_inst.empty:
                # Calculate Net Buy
                df_inst['net_buy'] = df_inst['buy'] - df_inst['sell']
                
                # Simplify names for pivoting
                name_map = {
                    'Foreign_Investor': 'Foreign',
                    'Investment_Trust': 'Trust',
                    'Dealer_self': 'Dealer',
                    'Dealer_Hedging': 'Dealer',
                    'Foreign_Dealer_Self': 'Foreign'
                }
                df_inst['name'] = df_inst['name'].map(name_map).fillna(df_inst['name'])
                
                # Aggregate by date and name
                df_inst_agg = df_inst.groupby(['date', 'name'])['net_buy'].sum().reset_index()
                
                # Pivot to wide format: columns = [date, Foreign, Trust, Dealer]
                df_inst_pivot = df_inst_agg.pivot(index='date', columns='name', values='net_buy').reset_index()
                df_inst_pivot.fillna(0, inplace=True)
                
                # Merge with price data
                df = pd.merge(df_price, df_inst_pivot, on='date', how='left')
                df.fillna(0, inplace=True) # Fill days without institutional trades with 0
            else:
                print("No institutional data found. Proceeding with price data only.")
                df = df_price
                df['Foreign'] = 0
                df['Trust'] = 0
                df['Dealer'] = 0

            # Save to Database
            table_name = f"stock_{stock_id}_daily"
            self.db.save_dataframe(df, table_name, if_exists='replace')
            return df
                
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