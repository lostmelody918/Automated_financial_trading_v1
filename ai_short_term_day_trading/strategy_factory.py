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

    def generate_signal(self, df_slice, ai_score=0.0):
        # 完全信任 AI 模型輸出的機率 (>0.5 強烈看多，<-0.5 強烈看空)
        # 此處模擬 AI 模型已在 offline 使用 4 年歷史數據完成高頻訓練，並接管即時盤中決策
        
        if ai_score > 0.5:
            return 1
        elif ai_score < -0.5:
            return -1
            
        return 0