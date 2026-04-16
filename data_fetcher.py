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

    def fetch_stock_info(self):
        """Fetches the list of all Taiwan stocks including their industry categories."""
        print("Fetching Taiwan stock information and industry categories...")
        try:
            df = self.dl.taiwan_stock_info()
            if not df.empty:
                self.db.save_dataframe(df, "taiwan_stock_info", if_exists='replace')
                return df
            return pd.DataFrame()
        except Exception as e:
            print(f"Error fetching stock info: {e}")
            return pd.DataFrame()

    def fetch_realtime_tick(self, stock_id: str):
        """Fetches the latest tick data for a specific stock."""
        # Simulated real-time fetch using FinMind's daily data for the current date
        # In a real scenario, this would call a real-time API or WebSocket
        try:
            df = self.dl.taiwan_stock_daily(
                stock_id=stock_id,
                start_date=datetime.today().strftime('%Y-%m-%d')
            )
            return df.tail(1) if not df.empty else pd.DataFrame()
        except Exception as e:
            print(f"Error fetching realtime tick for {stock_id}: {e}")
            return pd.DataFrame()

    def fetch_stock_financials(self, stock_id: str, start_date: str):
        """抓取損益表與股利資料，並確保獲得最新一季的完整資訊。"""
        print(f"Fetching latest financial data for {stock_id}...")
        results = {"financials": pd.DataFrame(), "dividends": pd.DataFrame(), "error": None}

        if stock_id == "TAIEX":
            results["error"] = "大盤指數無基本面資料。"
            return results

        try:
            # 1. Financial Statement - 強制抓取最近 5 年的資料以確保包含最新季度
            df_fin = self.dl.taiwan_stock_financial_statement(
                stock_id=stock_id,
                start_date="2019-01-01" # 拉長範圍確保資料完整
            )
            if df_fin is not None and not df_fin.empty:
                df_fin = df_fin.sort_values(['date', 'type'], ascending=[False, True])
                self.db.save_dataframe(df_fin, f"stock_{stock_id}_financials", if_exists='replace')
                results["financials"] = df_fin

            # 2. Dividend Data
            df_div = self.dl.taiwan_stock_dividend(stock_id=stock_id, start_date=start_date)
            if df_div is not None and not df_div.empty:
                results["dividends"] = df_div

            return results
        except Exception as e:
            results["error"] = f"API 抓取異常: {str(e)}"
            return results


    def fetch_stock_chips(self, stock_id: str, start_date: str, end_date: str = None):
        """抓取籌碼面資訊：資券變化、大戶持股比例"""
        if not end_date:
            end_date = datetime.today().strftime('%Y-%m-%d')
        print(f"Fetching chip data for {stock_id}...")
        
        try:
            # 1. 融資融券
            df_margin = self.dl.taiwan_stock_margin_purchase_short_sale(
                stock_id=stock_id, start_date=start_date, end_date=end_date
            )
            # 2. 大戶持股比例 (每週)
            df_holding = self.dl.taiwan_stock_holding_shares_per(
                stock_id=stock_id, start_date=start_date, end_date=end_date
            )
            
            if not df_margin.empty:
                self.db.save_dataframe(df_margin, f"stock_{stock_id}_margin", if_exists='replace')
            if not df_holding.empty:
                self.db.save_dataframe(df_holding, f"stock_{stock_id}_holdings", if_exists='replace')
                
            return {"margin": df_margin, "holdings": df_holding}
        except Exception as e:
            print(f"Error fetching chips: {e}")
            return {"margin": pd.DataFrame(), "holdings": pd.DataFrame()}

    def fetch_industry_prices(self, stock_ids: list, start_date: str, end_date: str):
        """Fetches daily prices for multiple stocks to calculate correlation."""
        combined_data = {}
        for sid in stock_ids:
            try:
                df = self.dl.taiwan_stock_daily(stock_id=sid, start_date=start_date, end_date=end_date)
                if not df.empty:
                    combined_data[sid] = df.set_index('date')['close']
            except:
                continue
        return pd.DataFrame(combined_data)

if __name__ == "__main__":
    fetcher = DataFetcher()
    # Let's fetch TSMC (2330) data from 2020-01-01 to today as a test
    print("Testing data fetcher...")
    tsmc_df = fetcher.fetch_stock_daily('2330', '2023-01-01')
    if not tsmc_df.empty:
        print(f"Head of TSMC data:\n{tsmc_df.head()}")