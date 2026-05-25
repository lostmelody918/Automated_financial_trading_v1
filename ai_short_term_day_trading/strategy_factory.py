import numpy as np

class StrategyFactory:
    """模組化策略工廠：可快速新增、抽換金融策略"""
    
    @staticmethod
    def get_strategy(strategy_name="composite"):
        return CompositeOptionsStrategy()

class CompositeOptionsStrategy:
    """
    華爾街量化波段當沖：AI-Driven Options Asymmetry
    在訓練出高準確度的 CNN-Transformer 複合模型後，直接由 AI 接管交易方向，釋放最大獲利潛能。
    """
    def __init__(self):
        self.name = "AI-Driven Asymmetry"

    def generate_signal(self, df_slice, ai_score=None, last_win=False):
        if ai_score is None: return 0

        prob_down = ai_score[0]
        prob_neutral = ai_score[1]
        prob_up = ai_score[2]

        last_row = df_slice.iloc[-1]
        atr = last_row.get('atr', 0)
        macd_hist = last_row.get('macd_hist', 0)
        vwap_bias = last_row.get('vwap_bias', 0)
        vix_ret = last_row.get('vix_ret_1d', 0)
        rsi = last_row.get('rsi', 50)

        # Volatility-Adjusted Momentum & Risk Management (Ref: Deep Momentum Networks)
        # 1. AI Confidence Threshold
        threshold = 0.52
        if last_win:
            threshold = 0.45

        # 2. VIX Shock Protection (Avoid trading in chaotic volatility spikes > 5%)
        if abs(vix_ret) > 0.05:
            return 0

        # 3. ATR Volatility Filter (Avoid entering when volatility is extremely high)
        # 放寬 ATR 的極端值過濾，適應高價位的相對特徵
        if atr > 500:
            return 0

        # 黃金規則 2：順勢動能過濾 (恢復高勝率的特徵區間)
        if prob_up > threshold:
            # 做多：MACD 為正向動能，VWAP_bias 相對不偏離太遠 (改用相對的正負號)
            if macd_hist > 0 and vwap_bias < 0.05 and rsi > 45:
                return 1

        elif prob_down > threshold + 0.10: # 嚴格限制做空 (要求更高的 AI 信心)
            # 做空：MACD 為負向動能
            if macd_hist < 0 and vwap_bias > -0.05 and rsi < 55:
                return -1

        return 0
