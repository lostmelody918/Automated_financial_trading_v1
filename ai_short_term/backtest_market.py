import os
import sys
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# 加入路徑
root_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if root_path not in sys.path:
    sys.path.insert(0, root_path)

from data_fetcher import DataFetcher
from ai_short_term.feature_engine import HFTFeatureEngine
from ai_short_term.ensemble_manager import StrategyEnsembleManager

def run_market_index_backtest():
    """針對台股加權指數 (TAIEX) 進行大盤回測"""
    fetcher = DataFetcher()
    engine = HFTFeatureEngine()
    ensemble = StrategyEnsembleManager()
    
    # 大盤代號：^TWII (加權指數), ^SOX (費半), ^IXIC (那斯達克)
    market_symbol = "^TWII"
    us_lead_symbol = "^SOX" # 費半是台股最強領先指標
    
    end_date = datetime.today().strftime('%Y-%m-%d')
    start_date = (datetime.today() - timedelta(days=60)).strftime('%Y-%m-%d')
    
    print(f"--- 啟動台股加權指數 (TAIEX) 今日回測報告 ---")
    
    # 1. 獲取數據
    # 使用 fetch_us_stock_daily 作為通用 yf 接口獲取指數
    df_taiex = fetcher.fetch_us_stock_daily(market_symbol, start_date, end_date)
    df_us = fetcher.fetch_us_stock_daily(us_lead_symbol, start_date, end_date)
    
    if df_taiex.empty or df_us.empty:
        print("❌ 數據獲取失敗，請確認 API 連線。")
        return

    # 2. 特徵計算
    df_taiex = engine.add_intraday_momentum(df_taiex)
    
    # 計算大盤特徵
    last_close = df_taiex['Close'].iloc[-1]
    prev_close = df_taiex['Close'].iloc[-2]
    today_ret = (last_close - prev_close) / prev_close
    
    us_last_ret = df_us['Close'].pct_change().iloc[-1]
    
    # 3. 環境感知 (Regime Detection)
    # 比較台股回報與美股回報的相關性
    market_state = ensemble.analyzer.get_market_state(
        df_taiex['Close'].pct_change().tail(20), 
        df_us['Close'].pct_change().tail(20)
    )
    regime_name = ensemble.analyzer.REGIMES[market_state]
    
    # 4. 決策邏輯 (Ensemble)
    market_data = {
        'regime': market_state,
        'price': last_close,
        'price_series': df_taiex['Close'],
        'vwap': df_taiex['vwap'].iloc[-1],
        'volume': df_taiex['Volume'].iloc[-1],
        'vol_ma': df_taiex['Volume'].rolling(20).mean().iloc[-1]
    }
    
    # 模擬 AI 訊號：如果美股強漲 + 大盤站在 VWAP 之上則看多
    ai_signal = 1.0 if (us_last_ret > 0.005 and last_close > df_taiex['vwap'].iloc[-1]) else 0.0
    if us_last_ret < -0.005: ai_signal = -1.0
    
    final_score = ensemble.get_final_signal(market_data, ai_signal)
    
    # 5. 輸出報告
    print(f"\n[市場概況]")
    print(f"🔹 加權指數位置: {last_close:.2f} ({today_ret*100:+.2f}%)")
    print(f"🔹 昨晚美股指標 (SOX): {us_last_ret*100:+.2f}%")
    print(f"🔹 當前市場氣氛: {regime_name}")
    
    print(f"\n[AI 多重策略評分]")
    print(f"🔸 綜合策略得分: {final_score:.2f} (-1.0 ~ 1.0)")
    
    # 大盤 ETF (如 0050) 的約略成本
    MARKET_COST = 0.002 
    
    if final_score > 0.4:
        decision = "偏多操作 (Long Index)"
        net_pnl = today_ret - MARKET_COST
    elif final_score < -0.4:
        decision = "偏空操作 (Short Index)"
        net_pnl = (-today_ret) - MARKET_COST
    else:
        decision = "中性觀望 (Neutral)"
        net_pnl = 0

    print(f"🏁 最終建議: {decision}")
    if decision != "中性觀望 (Neutral)":
        print(f"💰 模擬單日淨損益: {net_pnl*100:.2f}% (已扣除成本)")

if __name__ == "__main__":
    run_market_index_backtest()
