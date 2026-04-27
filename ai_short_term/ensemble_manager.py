import torch
import numpy as np
from ai_short_term.strategy_library import StrategyLibrary
from ai_short_term.market_analyzer import MarketRegimeAnalyzer

class StrategyEnsembleManager:
    """策略組合管理器 (大腦)：根據市場環境動態調整策略權重"""
    
    def __init__(self):
        self.analyzer = MarketRegimeAnalyzer()
        self.strategies = StrategyLibrary()
        
    def get_final_signal(self, market_data, ai_pred_prob):
        """
        整合 AI 預測與多重傳統策略
        market_data: 包含即時價格、VWAP、美股資訊的字典
        ai_pred_prob: CNN-LSTM 模型輸出的漲跌機率
        """
        # 1. 識別環境 (Regime)
        # 這裡簡化為從 market_data 提取
        regime = market_data.get('regime', 0)
        
        # 2. 獲取各策略原始信號
        s1 = self.strategies.mean_reversion_signal(market_data['price_series'], market_data['vwap'])
        s2 = self.strategies.momentum_breakout_signal(market_data['price'], market_data['volume'], market_data['vol_ma'])
        
        # 3. 權重分配邏輯 (Regime-based Weighting)
        if regime == 1: # Trending 市場：偏好動量
            weights = {'ai': 0.5, 'momentum': 0.4, 'reversion': 0.1}
        elif regime == 0: # Ranging 市場：偏好回歸
            weights = {'ai': 0.3, 'momentum': 0.1, 'reversion': 0.6}
        else: # Crisis：偏好避險/保守
            weights = {'ai': 0.2, 'momentum': 0.2, 'reversion': 0.6}
            
        # 4. 融合信號 (假設 ai_pred_prob 已轉換為 -1, 0, 1)
        final_signal = (weights['ai'] * ai_pred_prob + 
                        weights['momentum'] * s2 + 
                        weights['reversion'] * s1)
        
        return final_signal

if __name__ == "__main__":
    # 範例成本設定 (考慮手續費與交易稅)
    # 台股短線：手續費 0.1425% (打折前), 交易稅 0.15% (當沖減半)
    # 預設大約成本: 0.25% - 0.3%
    APPROX_COST = 0.0025
    print(f"✅ 策略組合管理器初始化完成。當前預估當沖成本: {APPROX_COST*100:.2f}%")
