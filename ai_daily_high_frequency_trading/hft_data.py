import yfinance as yf
import pandas as pd
import numpy as np

def fetch_hft_data(symbol="^TWII", period="2y", interval="1h"):
    """
    Fetches historical data for ^TWII. 
    Note: yfinance limits for intraday:
    - 1h: 730 days (~2 years)
    """
    print(f"Fetching {period} of {interval} data for {symbol}...")
    ticker = yf.Ticker(symbol)
    
    df = ticker.history(period=period, interval=interval)
    if df.empty: return df
    
    df.reset_index(inplace=True)
    if 'Datetime' in df.columns:
        if df['Datetime'].dt.tz is not None:
            df['Datetime'] = df['Datetime'].dt.tz_convert('Asia/Taipei')
        df.rename(columns={'Datetime': 'date'}, inplace=True)
    elif 'Date' in df.columns:
        df['date'] = pd.to_datetime(df['Date']).dt.tz_localize('Asia/Taipei')
    
    df['date_only'] = df['date'].dt.date
    
    # Technical Indicators (Vectorized)
    df['ema8'] = df['Close'].ewm(span=8, adjust=False).mean()
    df['ema21'] = df['Close'].ewm(span=21, adjust=False).mean()
    
    df['sma20'] = df['Close'].rolling(20).mean()
    df['std20'] = df['Close'].rolling(20).std()
    df['bb_up'] = df['sma20'] + (df['std20'] * 2.2)
    df['bb_dn'] = df['sma20'] - (df['std20'] * 2.2)
    df['bbw'] = (df['bb_up'] - df['bb_dn']) / (df['sma20'] + 1e-9)
    
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
    
    # VWAP (Intraday anchor)
    df['typical_price'] = (df['High'] + df['Low'] + df['Close']) / 3
    df['mock_volume'] = df['Volume'].replace(0, 1)
    df['vol_price'] = df['typical_price'] * df['mock_volume']
    df['vwap'] = df.groupby('date_only')['vol_price'].transform(lambda x: x.cumsum() / df.loc[x.index, 'mock_volume'].cumsum())
    
    return df.dropna().reset_index(drop=True)
