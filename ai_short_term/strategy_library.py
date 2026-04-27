import numpy as np
import pandas as pd

class StrategyLibrary:
    """策略庫：包含多種短線交易邏輯"""

    @staticmethod
    def mean_reversion_signal(price_series, vwap, z_score_threshold=2.0):
        """策略 1: 均值回歸 (基於 VWAP)"""
        if not isinstance(price_series, pd.Series):
            return 0
            
        # 計算 Z-Score 序列
        z_scores = (price_series - vwap) / (price_series.rolling(20).std() + 1e-9)
        last_z = z_scores.iloc[-1]
        
        if last_z > z_score_threshold:
            return -1 # 做空
        elif last_z < -z_score_threshold:
            return 1 # 做多
        return 0

    @staticmethod
    def momentum_breakout_signal(price_input, volume, volume_ma):
        """
        策略 2: 動量突破 (基於價量爆發)
        price_input: 可以是回報率 (float) 或 價格序列 (Series)
        """
        if isinstance(price_input, pd.Series):
            price_ret = price_input.pct_change().iloc[-1]
        else:
            price_ret = price_input
            
        # 對於大盤指數，成交量增加 10% 即視為放量 (個股才需要 1.5 倍)
        vol_surge = volume > (volume_ma * 1.1)
        
        # 大盤突破門檻通常較低，設為 0.15% (0.0015)
        if price_ret > 0.0015 and vol_surge:
            return 1 # 強力買入
        elif price_ret < -0.0015 and vol_surge:
            return -1 # 強力賣出
        return 0

    @staticmethod
    def stat_arb_signal(stock_a_price, stock_b_price, hedge_ratio=1.0):
        """策略 3: 統計套利 (Pairs Trading)"""
        spread = stock_a_price - hedge_ratio * stock_b_price
        z_score = (spread - spread.mean()) / spread.std()
        if z_score > 2.0: return -1 
        if z_score < -2.0: return 1  
        return 0
