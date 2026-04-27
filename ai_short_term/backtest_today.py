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

def run_today_backtest(stock_id="2330"):
    fetcher = DataFetcher()
    engine = HFTFeatureEngine()
    ensemble = StrategyEnsembleManager()
    
    # 1. 獲取數據 (台股 + 美股連動)
    end_date = datetime.today().strftime('%Y-%m-%d')
    start_date = (datetime.today() - timedelta(days=30)).strftime('%Y-%m-%d')
    
    print(f"--- 啟動今日回測報告 ({end_date}) ---")
    
    # 獲取台股數據
    df_tw = fetcher.fetch_stock_daily(stock_id, start_date, end_date)
    # 獲取美股連動 (費半指數)
    df_sox = fetcher.fetch_us_stock_daily("^SOX", start_date, end_date)
    
    if df_tw.empty or df_sox.empty:
        print("❌ 數據獲取失敗，請檢查 API 或網路。")
        return

    # 2. 特徵工程
    df = engine.add_intraday_momentum(df_tw)
    
    # 計算美股昨晚表現作為今日開盤指引
    us_last_ret = df_sox['Close'].pct_change().iloc[-1]
    print(f"📊 昨晚美股 (SOX) 表現: {us_last_ret*100:.2f}%")

    # 3. 模擬當前市場環境
    # 計算近 20 日相關性與波動率
    market_state = ensemble.analyzer.get_market_state(
        df['Close'].pct_change().tail(20), 
        df_sox['Close'].pct_change().tail(20)
    )
    regime_name = ensemble.analyzer.REGIMES[market_state]
    print(f"及時市場環境 (Regime): {regime_name}")

    # 4. 產生今日策略信號
    # 這裡我們先假設 AI 預測值為 0 (因為尚未訓練完成)，主要看傳統策略與環境加權
    current_price = df['Close'].iloc[-1]
    current_vwap = df['vwap'].iloc[-1]
    
    # 構造 market_data 傳入 ensemble
    market_data = {
        'regime': market_state,
        'price': current_price,
        'price_series': df['Close'],
        'vwap': current_vwap,
        'volume': df['Volume'].iloc[-1],
        'vol_ma': df['Volume'].rolling(20).mean().iloc[-1]
    }
    
    # 假設 AI 訊號暫時用 0 代替，或用簡單的趨勢判斷模擬
    ai_simulated_signal = 1.0 if us_last_ret > 0.01 else 0.0
    
    final_score = ensemble.get_final_signal(market_data, ai_simulated_signal)
    
    # 5. 決策輸出
    print(f"\n💡 策略得分: {final_score:.2f}")
    
    COST = 0.0025 # 0.25% 交易成本
    
    if final_score > 0.5:
        action = "Strong Buy / 多單進場"
        potential_pnl = (df['Close'].pct_change().iloc[-1]) - COST
    elif final_score < -0.5:
        action = "Strong Sell / 空單進場"
        potential_pnl = (-df['Close'].pct_change().iloc[-1]) - COST
    else:
        action = "Wait / 觀望"
        potential_pnl = 0

    print(f"🏁 建議行動: {action}")
    if action != "Wait / 觀望":
        print(f"📉 預估扣除成本後淨損益: {potential_pnl*100:.2f}%")

if __name__ == "__main__":
    # 以台積電 (2330) 作為大盤領先指標進行回測
    run_today_backtest("2330")
