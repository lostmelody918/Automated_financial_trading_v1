import yfinance as yf
import pandas as pd
import numpy as np

class DayTradingDataEngine:
    def __init__(self, symbol="^TWII"):
        self.symbol = symbol

    def fetch_intraday_data(self, days=60):
        """獲取近 60 天 5 分鐘 K 線，加入交割日與連假特徵 (Calendar Anomalies)"""
        print(f"📥 下載 {self.symbol} 近 {days} 天 5 分鐘 K 線數據...")
        ticker = yf.Ticker(self.symbol)
        df = ticker.history(period=f"{days}d", interval="5m")
        if df.empty:
            return df
        
        df.reset_index(inplace=True)
        if df['Datetime'].dt.tz is not None:
            df['Datetime'] = df['Datetime'].dt.tz_convert('Asia/Taipei')
        
        df.rename(columns={'Datetime': 'date'}, inplace=True)
        df['time'] = df['date'].dt.time
        df['date_only'] = df['date'].dt.date
        df['day_of_week'] = df['date'].dt.dayofweek # 0=Mon, 2=Wed, 4=Fri
        
        # 1. 加入交割日特徵 (每週三結算日，波動與 Gamma 放大)
        df['is_settlement_day'] = (df['day_of_week'] == 2).astype(int)
        
        # 2. 加入連假特徵 (如果下一個交易日與今日相差超過 3 天，視為連假前夕)
        unique_dates = pd.to_datetime(pd.Series(df['date_only'].unique()))
        date_diff = (unique_dates.shift(-1) - unique_dates).dt.days
        pre_holiday_dates = unique_dates[date_diff > 3].dt.date.values
        
        post_holiday_dates = unique_dates[date_diff.shift(1) > 3].dt.date.values
        
        df['is_pre_holiday'] = df['date_only'].isin(pre_holiday_dates).astype(int)
        df['is_post_holiday'] = df['date_only'].isin(post_holiday_dates).astype(int)

        # 基礎特徵
        df['ret'] = df['Close'].pct_change()
        
        # VWAP
        df['typical_price'] = (df['High'] + df['Low'] + df['Close']) / 3
        df['vol_price'] = df['typical_price'] * df['Volume']
        df['cum_vol_price'] = df.groupby('date_only')['vol_price'].cumsum()
        df['cum_vol'] = df.groupby('date_only')['Volume'].cumsum()
        df['vwap'] = df['cum_vol_price'] / (df['cum_vol'] + 1e-9)
        
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
        
        return df.dropna()
