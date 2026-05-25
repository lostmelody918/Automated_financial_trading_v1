import os
import pandas as pd
import numpy as np
import shioaji as sj
from datetime import datetime, timedelta

def fetch_hft_data(symbol="2330", period="2y", interval="1h"):
    """
    Fetches historical data for symbol using Shioaji.
    """
    print(f"Fetching data for {symbol} using Shioaji...")
    
    api = sj.Shioaji()
    api_key = os.environ.get('SHIOAJI_API_KEY', '')
    secret_key = os.environ.get('SHIOAJI_SECRET_KEY', '')
    
    if api_key and secret_key:
        api.login(api_key, secret_key, contracts_timeout=10000)
    else:
        print("Warning: Shioaji API keys missing for HFT data.")
        return pd.DataFrame()
        
    try:
        contract = api.Contracts.Stocks[symbol]
    except KeyError:
        print(f"Contract {symbol} not found.")
        return pd.DataFrame()
        
    start_date = (datetime.now() - timedelta(days=730)).strftime("%Y-%m-%d")
    end_date = datetime.now().strftime("%Y-%m-%d")
    
    kbars = api.kbars(contract, start=start_date, end=end_date)
    df = pd.DataFrame({**kbars})
    
    if df.empty: return df
    
    df['ts'] = pd.to_datetime(df['ts'])
    df.rename(columns={'ts': 'date'}, inplace=True)
    df['date_only'] = df['date'].dt.date
    
    # Technical Indicators (Vectorized)
    df['ema8'] = df['Close'].ewm(span=8, adjust=False).mean()
    df['ema21'] = df['Close'].ewm(span=21, adjust=False).mean()
    
    df['sma20'] = df['Close'].rolling(20).mean()
    df['std20'] = df['Close'].rolling(20).std()
    df['bb_up'] = df['sma20'] + (df['std20'] * 2.2)
    df['bb_dn'] = df['sma20'] - (df['std20'] * 2.2)
    df['bbw'] = (df['bb_up'] - df['bb_dn']) / (df['sma20'] + 1e-9)
    df['bbw_mean'] = df['bbw'].rolling(100).mean()
    df['bbw_std'] = df['bbw'].rolling(100).std()
    df['bbw_zscore'] = (df['bbw'] - df['bbw_mean']) / (df['bbw_std'] + 1e-9)
    
    df['ema_htf'] = df['Close'].ewm(span=144, adjust=False).mean()
    
    # Hull Moving Average
    def hma(series, n):
        wma1 = series.rolling(n//2).apply(lambda x: np.sum(x * np.arange(1, n//2 + 1)) / np.sum(np.arange(1, n//2 + 1)), raw=True)
        wma2 = series.rolling(n).apply(lambda x: np.sum(x * np.arange(1, n + 1)) / np.sum(np.arange(1, n + 1)), raw=True)
        diff = 2 * wma1 - wma2
        return diff.rolling(int(np.sqrt(n))).apply(lambda x: np.sum(x * np.arange(1, int(np.sqrt(n)) + 1)) / np.sum(np.arange(1, int(np.sqrt(n)) + 1)), raw=True)
    
    df['hma9'] = hma(df['Close'], 9)
    df['hma21'] = hma(df['Close'], 21)
    df['hma9_slope'] = (df['hma9'].diff(3) / df['hma9'].shift(3)) * 100
    df['hma21_slope'] = (df['hma21'].diff(3) / df['hma21'].shift(3)) * 100
    
    # Trend Strength
    df['up_move'] = df['High'].diff()
    df['dn_move'] = df['Low'].diff().abs()
    df['plus_dm'] = np.where((df['up_move'] > df['dn_move']) & (df['up_move'] > 0), df['up_move'], 0)
    df['minus_dm'] = np.where((df['dn_move'] > df['up_move']) & (df['dn_move'] > 0), df['dn_move'], 0)
    df['tr'] = np.maximum(df['High'] - df['Low'], np.maximum((df['High'] - df['Close'].shift()).abs(), (df['Low'] - df['Close'].shift()).abs()))
    df['plus_di'] = 100 * (df['plus_dm'].rolling(14).mean() / (df['tr'].rolling(14).mean() + 1e-9))
    df['minus_di'] = 100 * (df['minus_dm'].rolling(14).mean() / (df['tr'].rolling(14).mean() + 1e-9))
    df['adx'] = 100 * (df['plus_di'] - df['minus_di']).abs() / (df['plus_di'] + df['minus_di'] + 1e-9)
    
    df['atr'] = (df['High'] - df['Low']).rolling(14).mean()
    # ATR Change Rate (Simulates Vega expansion/contraction)
    df['atr_roc'] = df['atr'].pct_change(3) # 3-period change rate
    
    # VWAP (Weekly anchor)
    df['typical_price'] = (df['High'] + df['Low'] + df['Close']) / 3
    df['mock_volume'] = df['Volume'].replace(0, 1)
    df['vol_price'] = df['typical_price'] * df['mock_volume']
    df['year_week'] = df['date'].dt.isocalendar().year.astype(str) + '-' + df['date'].dt.isocalendar().week.astype(str)
    df['vwap'] = df.groupby('year_week')['vol_price'].transform(lambda x: x.cumsum() / df.loc[x.index, 'mock_volume'].cumsum())
    df['vwap'] = df['vwap'].ewm(span=5, adjust=False).mean() # Smoothing to avoid exact match on Monday morning
    
    return df.dropna().reset_index(drop=True)
