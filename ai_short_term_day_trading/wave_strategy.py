import numpy as np

class WaveDayStrategy:
    """華爾街量化波段當沖：Institutional Gap & Go + VWAP Golden Cross"""
    
    def __init__(self):
        pass

    def generate_signal(self, df_slice, ai_score=0.0):
        # 取得今天開盤價與昨天收盤價
        today_date = df_slice['date_only'].iloc[-1]
        today_data = df_slice[df_slice['date_only'] == today_date]
        
        if len(today_data) < 2:
            return 0
            
        today_open = today_data['Open'].iloc[0]
        last_close = df_slice['Close'].iloc[-1]
        last_vwap = df_slice['vwap'].iloc[-1]
        
        yesterday_data = df_slice[df_slice['date_only'] != today_date]
        if yesterday_data.empty: return 0
        yest_close = yesterday_data['Close'].iloc[-1]
        
        gap_pct = (today_open - yest_close) / yest_close
        
        macd_hist = df_slice['macd_hist'].iloc[-1]
        prev_macd_hist = df_slice['macd_hist'].iloc[-2]
        
        signal = 0
        
        # 1. Institutional Gap & Go (外資跳空追擊)
        # 如果大盤跳空上漲超過 0.2%，且開盤後前幾根 K 線沒有跌破開盤價，直接做多
        if gap_pct > 0.002 and last_close >= today_open * 0.9995:
            signal = 1
            
        # 2. VWAP Golden Cross (盤中轉強點)
        # 價格由下往上穿越 VWAP，且 MACD 為正向擴張
        elif last_close > last_vwap and df_slice['Close'].iloc[-2] <= df_slice['vwap'].iloc[-2]:
            if macd_hist > 0 and macd_hist > prev_macd_hist:
                signal = 1
                
        # 3. 恐慌崩盤做空 (破底翻空)
        # 如果跳空大跌超過 0.3%，且價格在 VWAP 之下，追空
        elif gap_pct < -0.003 and last_close < last_vwap:
            signal = -1
            
        return signal