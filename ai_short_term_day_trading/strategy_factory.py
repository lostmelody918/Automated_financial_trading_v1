import numpy as np

class StrategyFactory:
    """模組化策略工廠：雙軌制動能共振系統"""
    @staticmethod
    def get_strategy(strategy_name="composite"):
        return CompositeOptionsStrategy()

class CompositeOptionsStrategy:
    def __init__(self):
        self.name = "Dual-Track Momentum"

    def generate_signal(self, df_slice, ai_score=None, last_win=False):
        if ai_score is None: return 0

        prob_down, prob_neutral, prob_up = ai_score[0], ai_score[1], ai_score[2]
        last_row = df_slice.iloc[-1]

        macd_hist = last_row.get('macd_hist', 0)
        slope_vwap = last_row.get('slope_vwap', 0)
        atr = last_row.get('atr', 0)
        vol_surge = last_row.get('vol_surge_ratio', 1.0)
        is_squeeze = last_row.get('is_squeeze', 0)
        foreign_oi_mom = last_row.get('foreign_net_oi_momentum', 0)
        pc_ratio = last_row.get('pc_ratio', 1.0)

        # 基礎 AI 門檻
        base_threshold = 0.50 if last_win else 0.52
        if is_squeeze == 1: base_threshold -= 0.05

        if atr > 100: return 0 # 防禦快市

        # ==========================================
        # 🚀 軌道一：強勢共振 (大波段，訊號 2 / -2)
        # 嚴格條件：AI高自信 + 有斜率 + 爆量/擠壓 + 籌碼支持
        # ==========================================
        # 放寬 AI 門檻從 0.58 -> 0.55，斜率從 0.3 -> 0.2
        strong_up = (prob_up > 0.55 and slope_vwap > 0.2 and macd_hist > 0 and
                     (vol_surge > 1.5 or is_squeeze == 1) and pc_ratio < 1.15)

        strong_down = (prob_down > 0.55 and slope_vwap < -0.2 and macd_hist < 0 and
                       (vol_surge > 1.5 or is_squeeze == 1) and foreign_oi_mom < 0)

        if strong_up: return 3
        if strong_down: return -3

        # ==========================================
        # 📈 Level 2：標準波段 (趨勢成型)
        # ==========================================
        # 放寬 AI 門檻從 0.51 -> 0.48，斜率從 0.125 -> 0.08
        l2_up = (prob_up > 0.48 and slope_vwap > 0.08 and macd_hist > 0 and pc_ratio < 1.4)
        l2_down = (prob_down > 0.48 and slope_vwap < -0.08 and macd_hist < 0)

        if l2_up: return 2
        if l2_down: return -2

        # ==========================================
        # ⚡ Level 1：短線打帶跑 (動能初現或布林擠壓)
        # ==========================================
        # 極度放寬 AI 門檻從 0.38 -> 0.15 (應使用者要求強制提高交易頻率)
        base_threshold = 0.12 if last_win else 0.15
        if is_squeeze == 1: base_threshold -= 0.05

        # 只要有一邊勝率大於另一邊，且突破極低門檻
        l1_up = (prob_up > base_threshold) and (prob_up > prob_down + 0.02)
        l1_down = (prob_down > base_threshold) and (prob_down > prob_up + 0.02)

        if l1_up: return 1
        if l1_down: return -1

        return 0