import pandas as pd
import numpy as np

class HFTFeatureEngine:
    """高頻特徵工程引擎：處理分鐘級與 Tick 級數據"""
    
    @staticmethod
    def calculate_lob_imbalance(bid_vol, ask_vol):
        """計算掛單失衡度 (Limit Order Book Imbalance)"""
        return (bid_vol - ask_vol) / (bid_vol + ask_vol + 1e-9)

    @staticmethod
    def calculate_ofi(df):
        """計算訂單流失衡 (Order Flow Imbalance)
        df 需要包含: bid_p, bid_v, ask_p, ask_v
        """
        # 簡化版 OFI 計算
        df['delta_bid_v'] = np.where(df['bid_p'] >= df['bid_p'].shift(1), df['bid_v'], 0) - \
                            np.where(df['bid_p'] <= df['bid_p'].shift(1), df['bid_v'].shift(1), 0)
        df['delta_ask_v'] = np.where(df['ask_p'] <= df['ask_p'].shift(1), df['ask_v'], 0) - \
                            np.where(df['ask_p'] >= df['ask_p'].shift(1), df['ask_v'].shift(1), 0)
        return df['delta_bid_v'] - df['delta_ask_v']

    @staticmethod
    def add_intraday_momentum(df, window=20):
        """加入盤中動量特徵"""
        df['vwap'] = (df['Close'] * df['Volume']).rolling(window=window).sum() / df['Volume'].rolling(window=window).sum()
        df['vwap_dist'] = (df['Close'] - df['vwap']) / df['vwap']
        df['log_ret'] = np.log(df['Close'] / df['Close'].shift(1))
        df['volatility'] = df['log_ret'].rolling(window=window).std()
        return df
