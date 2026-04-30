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

    def generate_signal(self, df_slice, ai_score=None):
        if ai_score is None: return 0

        prob_down = ai_score[0]
        prob_neutral = ai_score[1]
        prob_up = ai_score[2]

        last_row = df_slice.iloc[-1]
        is_squeeze = last_row.get('is_squeeze', 0)
        atr = last_row.get('atr', 0)
        n225_ret = last_row.get('n225_ret', 0) 
        macd_hist = last_row.get('macd_hist', 0)
        vwap = last_row.get('vwap', 0)
        vwap_bias = last_row.get('vwap_bias', 0)
        close = last_row.get('Close', 0)
        ixic_ret = last_row.get('ixic_ret_1d', 0)
        tsm_ret = last_row.get('tsm_ret_1d', 0)
        vix_ret = last_row.get('vix_ret_1d', 0)

        # 黃金規則 1：AI 信心必須超過 0.70
        threshold = 0.70 

        if atr > 160:
            return 0

        # 黃金規則 3：堅實的順勢邏輯，加入美股日盤 (IXIC) 與 恐慌指數 (VIX) 做為大格局的背書
        if prob_up > threshold:
            # 做多：MACD 動能向上 (> 5)，且 VWAP 乖離不過大。
            # 大局觀：昨晚美股科技股 (IXIC) 必須上漲 (> 0) 或恐慌指數下跌 (VIX < -0.01)
            if close > vwap and macd_hist > 5 and vwap_bias < 0.005 and n225_ret > -0.005:
                if ixic_ret > 0 or vix_ret < -0.01:
                    return 1

        elif prob_down > threshold:
            # 做空：MACD 動能向下 (< -10)，且 VWAP 乖離不殺低過度。
            # 大局觀：昨晚美股科技股下跌 (IXIC < 0) 或 恐慌指數飆升 (VIX > 0.02)
            if close < vwap and macd_hist < -10 and vwap_bias > -0.005 and n225_ret < 0.005:
                if ixic_ret < 0 or vix_ret > 0.02:
                    return -1

        return 0