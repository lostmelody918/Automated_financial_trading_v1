import yfinance as yf
import pandas as pd
import numpy as np

class DayTradingDataEngine:
    def __init__(self, symbol="^TWII"):
        self.symbol = symbol

    def fetch_intraday_data(self, days=60):
        """獲取近 60 天 5 分鐘 K 線，加入台股與美指期(NQ=F)特徵"""
        print(f"📥 下載 {self.symbol} 與 NQ=F (那斯達克期貨) 近 {days} 天 5 分鐘 K 線數據...")
        
        # 抓取台股大盤
        ticker = yf.Ticker(self.symbol)
        df = ticker.history(period=f"{days}d", interval="5m")
        if df.empty:
            return df
            
        # 抓取日經225 (亞洲盤高度連動)
        n225_ticker = yf.Ticker("^N225")
        df_n225 = n225_ticker.history(period=f"{days}d", interval="5m")

        # 抓取台積電 ADR, VIX, 和 那斯達克綜合指數(^IXIC) (日線資料，用於每日開盤情緒)
        try:
            tsm_df = yf.Ticker("TSM").history(period=f"{days+10}d", interval="1d")
            vix_df = yf.Ticker("^VIX").history(period=f"{days+10}d", interval="1d")
            ixic_df = yf.Ticker("^IXIC").history(period=f"{days+10}d", interval="1d")
            
            tsm_df['tsm_ret_1d'] = tsm_df['Close'].pct_change()
            vix_df['vix_ret_1d'] = vix_df['Close'].pct_change()
            ixic_df['ixic_ret_1d'] = ixic_df['Close'].pct_change()
            
            # 將日線資料的 index 轉為與台灣時間對齊的日期 (美股收盤通常是台灣時間凌晨，所以我們把它對應到當天的台股開盤)
            tsm_df.index = tsm_df.index.tz_convert('Asia/Taipei').date
            vix_df.index = vix_df.index.tz_convert('Asia/Taipei').date
            ixic_df.index = ixic_df.index.tz_convert('Asia/Taipei').date
        except Exception as e:
            print(f"⚠️ 無法取得 TSM, VIX, 或 IXIC 數據: {e}")
            tsm_df = pd.DataFrame()
            vix_df = pd.DataFrame()
            ixic_df = pd.DataFrame()
        
        df.reset_index(inplace=True)
        if df['Datetime'].dt.tz is not None:
            df['Datetime'] = df['Datetime'].dt.tz_convert('Asia/Taipei')
            
        # 合併 TSM, VIX, IXIC 特徵
        df['date_only'] = df['Datetime'].dt.date
        df['tsm_ret_1d'] = 0.0
        df['vix_ret_1d'] = 0.0
        df['ixic_ret_1d'] = 0.0
        
        if not tsm_df.empty:
            # 取得前一個交易日的美股收盤表現 (shift 1 確保沒有未來數據)
            tsm_dict = tsm_df['tsm_ret_1d'].shift(1).to_dict()
            df['tsm_ret_1d'] = df['date_only'].map(tsm_dict).fillna(0)
            
        if not vix_df.empty:
            vix_dict = vix_df['vix_ret_1d'].shift(1).to_dict()
            df['vix_ret_1d'] = df['date_only'].map(vix_dict).fillna(0)
            
        if not ixic_df.empty:
            ixic_dict = ixic_df['ixic_ret_1d'].shift(1).to_dict()
            df['ixic_ret_1d'] = df['date_only'].map(ixic_dict).fillna(0)
            
        if not df_n225.empty:
            df_n225.reset_index(inplace=True)
            if df_n225['Datetime'].dt.tz is not None:
                df_n225['Datetime'] = df_n225['Datetime'].dt.tz_convert('Asia/Taipei')
            
            # 計算日經特徵
            df_n225['n225_ret'] = df_n225['Close'].pct_change()
            df_n225['n225_slope_5'] = (df_n225['Close'] - df_n225['Close'].shift(5)) / 5.0
            
            # 只保留需要的欄位並與台股對齊
            df_n225_subset = df_n225[['Datetime', 'n225_ret', 'n225_slope_5']]
            df = pd.merge(df, df_n225_subset, on='Datetime', how='left')
            
            # 填補日經沒有交易時段的空值 (向前填補)
            df['n225_ret'] = df['n225_ret'].fillna(0)
            df['n225_slope_5'] = df['n225_slope_5'].ffill().fillna(0)
        else:
            print("⚠️ 無法取得 ^N225 數據，將使用 0 填補")
            df['n225_ret'] = 0
            df['n225_slope_5'] = 0
        
        df.rename(columns={'Datetime': 'date'}, inplace=True)
        df['time'] = df['date'].dt.time
        df['date_only'] = df['date'].dt.date
        df['day_of_week'] = df['date'].dt.dayofweek
        df['day'] = df['date'].dt.day
        
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
        
        # VWAP (如果 yfinance 回傳的指數成交量為 0，則使用 1 來計算時間加權平均價 TWAP)
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
        
        df['bb_width_ma100'] = df['bb_width'].rolling(100).mean()
        df['is_squeeze'] = (df['bb_width'] < df['bb_width_ma100']).astype(int)

        # 成交量變化
        df['v_ma5'] = df['Volume'].rolling(5).mean()
        df['v_rel'] = df['Volume'] / (df['v_ma5'] + 1e-9)

        # 清理
        df.drop(columns=['day', 'is_wednesday', 'typical_price', 'vol_price', 'cum_vol_price', 'cum_vol', 'h_l', 'h_pc', 'l_pc', 'tr', 'sma20', 'std20', 'v_ma5', 'bb_width_ma100'], inplace=True, errors='ignore')

        return df.dropna()
