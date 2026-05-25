import os
import pandas as pd
import numpy as np
import shioaji as sj
from datetime import datetime, timedelta

class DayTradingDataEngine:
    def __init__(self, symbol="2330"):
        self.symbol = symbol
        self.api = sj.Shioaji()
        api_key = os.environ.get('SHIOAJI_API_KEY', '')
        secret_key = os.environ.get('SHIOAJI_SECRET_KEY', '')
        if api_key and secret_key:
            self.api.login(api_key, secret_key, contracts_timeout=10000)
        else:
            print("Warning: Shioaji API keys missing.")

    def fetch_intraday_data(self, days=60):
        """獲取近 60 天 K 線，使用 Shioaji 進行抓取"""
        print(f"📥 下載 {self.symbol} 近 {days} 天 K 線數據...")
        
        try:
            contract = self.api.Contracts.Stocks[self.symbol]
        except KeyError:
            print(f"Contract {self.symbol} not found.")
            return pd.DataFrame()
            
        start_date = (datetime.now() - timedelta(days=days)).strftime("%Y-%m-%d")
        end_date = datetime.now().strftime("%Y-%m-%d")
        
        kbars = self.api.kbars(contract, start=start_date, end=end_date)
        df = pd.DataFrame({**kbars})
        
        if df.empty:
            return df
            
        df['ts'] = pd.to_datetime(df['ts'])
        df.rename(columns={'ts': 'date'}, inplace=True)
        
        df['time'] = df['date'].dt.time
        df['date_only'] = df['date'].dt.date
        df['day_of_week'] = df['date'].dt.dayofweek
        df['day'] = df['date'].dt.day
        
        # 由於 Shioaji 不易抓取外期指數，這裡暫時將其他特徵填 0 (與舊版相容結構)
        df['Dividends'] = 0.0
        df['Stock Splits'] = 0.0
        df['tsm_ret_1d'] = 0.0
        df['vix_ret_1d'] = 0.0
        df['ixic_ret_1d'] = 0.0
    def fetch_active_option_intraday_data(self, days=1):
        """獲取目前市場上最活躍的選擇權合約的 K 線資料"""
        print(f"📥 搜尋市場上最活躍的選擇權合約...")
        
        all_contracts = []
        for cat in ['TXO', 'TX1', 'TX2', 'TX4', 'TX5']:
            if hasattr(self.api.Contracts.Options, cat):
                all_contracts.extend([c for c in getattr(self.api.Contracts.Options, cat)])
                
        # 由於合約太多，我們只抓最近兩個月份/週別的來比較
        if not all_contracts:
            print("找不到任何選擇權合約")
            return pd.DataFrame(), None
            
        delivery_months = sorted(list(set(c.delivery_month for c in all_contracts)))
        if len(delivery_months) > 2:
            near_months = delivery_months[:2]
            all_contracts = [c for c in all_contracts if c.delivery_month in near_months]

        # 批次取得 snapshots
        try:
            snapshots = self.api.snapshots(all_contracts)
        except Exception as e:
            print(f"取得選擇權 Snapshots 失敗: {e}")
            return pd.DataFrame(), None
            
        volumes = [(all_contracts[i], s.total_volume) for i, s in enumerate(snapshots)]
        volumes.sort(key=lambda x: x[1], reverse=True)
        
        best_contract = volumes[0][0]
        print(f"⭐ 找到最活躍合約: {best_contract.symbol} (成交量: {volumes[0][1]})")
        
        start_date = (datetime.now() - timedelta(days=days)).strftime("%Y-%m-%d")
        end_date = datetime.now().strftime("%Y-%m-%d")
        
        kbars = self.api.kbars(best_contract, start=start_date, end=end_date)
        df = pd.DataFrame({**kbars})
        
        if df.empty:
            print(f"⚠️ {best_contract.symbol} K 線資料為空")
            return df, best_contract
            
        df['ts'] = pd.to_datetime(df['ts'])
        df.rename(columns={'ts': 'date'}, inplace=True)
        
        df['time'] = df['date'].dt.time
        df['date_only'] = df['date'].dt.date
        df['day_of_week'] = df['date'].dt.dayofweek
        df['day'] = df['date'].dt.day
        
        # 由於 Shioaji 不易抓取外期指數，這裡暫時將其他特徵填 0 (與舊版相容結構)
        df['Dividends'] = 0.0
        df['Stock Splits'] = 0.0
        df['tsm_ret_1d'] = 0.0
        df['vix_ret_1d'] = 0.0
        df['ixic_ret_1d'] = 0.0
        df['n225_ret'] = 0.0
        df['n225_slope_5'] = 0.0
        
        # 1. 加入交割日特徵
        df['is_wednesday'] = (df['day_of_week'] == 2).astype(int)
        df['is_monthly_settlement'] = ((df['is_wednesday'] == 1) & (df['day'] >= 15) & (df['day'] <= 21)).astype(int)
        df['is_weekly_settlement'] = ((df['is_wednesday'] == 1) & (df['is_monthly_settlement'] == 0)).astype(int)
        
        # 2. 連假特徵
        unique_dates = pd.to_datetime(pd.Series(df['date_only'].unique()))
        date_diff = (unique_dates.shift(-1) - unique_dates).dt.days
        pre_holiday_dates = unique_dates[date_diff > 3].dt.date.values
        post_holiday_dates = unique_dates[date_diff.shift(1) > 3].dt.date.values
        df['is_pre_holiday'] = df['date_only'].isin(pre_holiday_dates).astype(int)
        df['is_post_holiday'] = df['date_only'].isin(post_holiday_dates).astype(int)

        # 基礎特徵
        df['ret'] = df['Close'].pct_change()
        df['slope_5'] = (df['Close'] - df['Close'].shift(5)) / 5.0
        df['slope_10'] = (df['Close'] - df['Close'].shift(10)) / 10.0
        
        df['typical_price'] = (df['High'] + df['Low'] + df['Close']) / 3
        df['mock_volume'] = df['Volume'].replace(0, 1)
        df['vol_price'] = df['typical_price'] * df['mock_volume']
        df['cum_vol_price'] = df.groupby('date_only')['vol_price'].cumsum()
        df['cum_vol'] = df.groupby('date_only')['mock_volume'].cumsum()
        df['vwap'] = df['cum_vol_price'] / (df['cum_vol'] + 1e-9)
        df['vwap_bias'] = (df['Close'] - df['vwap']) / (df['vwap'] + 1e-9)
        
        # MACD
        exp1 = df['Close'].ewm(span=12, adjust=False).mean()
        exp2 = df['Close'].ewm(span=26, adjust=False).mean()
        df['macd'] = exp1 - exp2
        df['signal'] = df['macd'].ewm(span=9, adjust=False).mean()
        df['macd_hist'] = df['macd'] - df['signal']
        
        # ATR (波動率)
        df['h_l'] = df['High'] - df['Low']
        df['h_pc'] = abs(df['High'] - df['Close'].shift(1))
        df['l_pc'] = abs(df['Low'] - df['Close'].shift(1))
        df['tr'] = df[['h_l', 'h_pc', 'l_pc']].max(axis=1)
        df['atr'] = df['tr'].rolling(14).mean()

        # RSI
        delta = df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / (loss + 1e-9)
        df['rsi'] = 100 - (100 / (1 + rs))

        # Bollinger Bands & Squeeze (擠壓)
        df['sma20'] = df['Close'].rolling(window=20).mean()
        df['std20'] = df['Close'].rolling(window=20).std()
        df['bb_upper'] = df['sma20'] + (df['std20'] * 2)
        df['bb_lower'] = df['sma20'] - (df['std20'] * 2)
        df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / df['sma20']
        
        df['bb_width_ma20'] = df['bb_width'].rolling(20).mean()
        df['is_squeeze'] = (df['bb_width'] < df['bb_width_ma20']).astype(int)

        # 成交量變化
        df['v_ma5'] = df['Volume'].rolling(5).mean()
        df['v_rel'] = df['Volume'] / (df['v_ma5'] + 1e-9)

        # 清理
        df.drop(columns=['day', 'is_wednesday', 'typical_price', 'vol_price', 'cum_vol_price', 'cum_vol', 'h_l', 'h_pc', 'l_pc', 'tr', 'sma20', 'std20', 'v_ma5', 'bb_width_ma20', 'Amount'], inplace=True, errors='ignore')

        return df.dropna(), best_contract
