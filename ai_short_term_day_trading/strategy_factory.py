import numpy as np

class StrategyFactory:
    """模組化策略工廠：直接使用 AI 決策"""
    @staticmethod
    def get_strategy(strategy_name="composite"):
        return CompositeOptionsStrategy()

class CompositeOptionsStrategy:
    def __init__(self):
        self.name = "AI Direct Output"

    def generate_signal(self, df_slice, ai_score=None, last_win=False):
        if ai_score is None or len(ai_score) != 7: return 0
        
        # ai_score indices:
        # 0: Strong Down (-3), 1: Med Down (-2), 2: Weak Down (-1)
        # 3: Hold (0)
        # 4: Weak Up (1), 5: Med Up (2), 6: Strong Up (3)
        
        pred_class = np.argmax(ai_score)
        
        # Map class to signal strength
        mapping = {
            0: -3,
            1: -2,
            2: -1,
            3: 0,
            4: 1,
            5: 2,
            6: 3
        }
        
        return mapping.get(pred_class, 0)