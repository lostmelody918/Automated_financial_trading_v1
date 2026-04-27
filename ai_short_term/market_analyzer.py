import pandas as pd
import numpy as np

class MarketRegimeAnalyzer:
    """市場環境分析器：識別當前市場狀態 (Regime Detection)"""
    
    REGIMES = {0: "Low Volatility / Ranging", 1: "High Volatility / Trending", 2: "Crisis / Panic"}

    def __init__(self):
        self.correlation_window = 20

    def get_market_state(self, tw_returns, us_returns):
        """
        計算台美連動性與波動率，返回當前市場 regime
        tw_returns: 台股近況回報率序列
        us_returns: 美股 (SOX/Nasdaq) 近況回報率序列
        """
        # 1. 計算相關性 (Coupling Factor)
        correlation = tw_returns.corr(us_returns)
        
        # 2. 計算波動率 (Volatility)
        volatility = tw_returns.std() * np.sqrt(252)
        
        # 3. 邏輯判斷
        if volatility > 0.3:
            return 2 # Crisis/High Vol
        elif abs(correlation) > 0.7:
            return 1 # Trending / Highly Coupled
        else:
            return 0 # Ranging / Decoupled

    def get_cross_market_features(self, df_tw, df_us):
        """獲取跨市場特徵"""
        # 假設 df_us 是昨晚美股數據
        features = {
            'us_last_ret': df_us['Close'].pct_change().iloc[-1],
            'adr_premium': (df_us['Close'].iloc[-1] / df_tw['Close'].iloc[-1]) - 1 # 簡化溢價
        }
        return features
