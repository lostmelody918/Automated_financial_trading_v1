import os
import sys
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

root_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if root_path not in sys.path:
    sys.path.insert(0, root_path)

from data_fetcher import DataFetcher
from ai_short_term.feature_engine import HFTFeatureEngine
from ai_short_term.ensemble_manager import StrategyEnsembleManager

def run_historical_backtest(start_date="2024-01-01", end_date=None, threshold=0.3):
    if end_date is None:
        end_date = datetime.today().strftime('%Y-%m-%d')
        
    fetcher = DataFetcher()
    engine = HFTFeatureEngine()
    ensemble = StrategyEnsembleManager()
    
    print(f"📥 正在獲取歷史數據 ({start_date} ~ {end_date})...")
    df_taiex = fetcher.fetch_us_stock_daily("^TWII", start_date, end_date)
    df_us = fetcher.fetch_us_stock_daily("^SOX", start_date, end_date)
    
    if df_taiex.empty or df_us.empty:
        print("❌ 數據獲取失敗。")
        return
        
    # 對齊日期
    df_taiex['date'] = pd.to_datetime(df_taiex['date'])
    df_us['date'] = pd.to_datetime(df_us['date'])
    df_merged = pd.merge(df_taiex, df_us, on='date', suffixes=('_tw', '_us'), how='inner')
    
    df_tw_only = df_merged[['date', 'Open_tw', 'High_tw', 'Low_tw', 'Close_tw', 'Volume_tw']].rename(
        columns={'Open_tw': 'Open', 'High_tw': 'High', 'Low_tw': 'Low', 'Close_tw': 'Close', 'Volume_tw': 'Volume'})
        
    df_tw_only = engine.add_intraday_momentum(df_tw_only)
    
    # 回測參數
    COST = 0.002 # 0.2% 雙邊總和成本預估
    window = 20
    
    results = []
    
    print("⏳ 執行歷史回測中...")
    
    for i in range(window, len(df_tw_only)-1):
        # 取得直到今日的數據片段
        curr_tw = df_tw_only.iloc[:i+1].copy()
        curr_us = df_merged.iloc[:i+1].copy()
        
        tw_ret_series = curr_tw['Close'].pct_change().dropna()
        us_ret_series = curr_us['Close_us'].pct_change().dropna()
        
        if len(tw_ret_series) < 20: continue
            
        market_state = ensemble.analyzer.get_market_state(
            tw_ret_series.tail(20), 
            us_ret_series.tail(20)
        )
        
        last_close = curr_tw['Close'].iloc[-1]
        vwap = curr_tw['vwap'].iloc[-1]
        volume = curr_tw['Volume'].iloc[-1]
        vol_ma = curr_tw['Volume'].rolling(20).mean().iloc[-1]
        us_last_ret = us_ret_series.iloc[-1]
        
        market_data = {
            'regime': market_state,
            'price': last_close,
            'price_series': curr_tw['Close'],
            'vwap': vwap,
            'volume': volume,
            'vol_ma': vol_ma
        }
        
        # 優化 AI 模擬訊號：更敏銳地捕捉美股跳空與趨勢
        ai_signal = 0.0
        if us_last_ret > 0.01:
            ai_signal = 1.0 # 美股大漲，做多意願強
        elif us_last_ret > 0.002 and last_close > vwap:
            ai_signal = 0.5 # 美股小漲且站上均價
        elif us_last_ret < -0.01:
            ai_signal = -1.0 # 美股大跌，做空意願強
        elif us_last_ret < -0.002 and last_close < vwap:
            ai_signal = -0.5 # 美股小跌且跌破均價
        
        score = ensemble.get_final_signal(market_data, ai_signal)
        
        # 隔日實際報酬率
        next_ret = (df_tw_only['Close'].iloc[i+1] - df_tw_only['Close'].iloc[i]) / df_tw_only['Close'].iloc[i]
        
        trade_ret = 0
        action = 0
        if score >= threshold:
            trade_ret = next_ret - COST
            action = 1
        elif score <= -threshold:
            trade_ret = -next_ret - COST
            action = -1
            
        results.append({
            'date': curr_tw['date'].iloc[-1],
            'score': score,
            'action': action,
            'market_ret': next_ret,
            'strategy_ret': trade_ret,
            'regime': market_state
        })
        
    df_res = pd.DataFrame(results)
    
    # 績效指標計算
    df_res['cum_market'] = (1 + df_res['market_ret']).cumprod()
    df_res['cum_strategy'] = (1 + df_res['strategy_ret']).cumprod()
    
    total_trades = (df_res['action'] != 0).sum()
    win_trades = (df_res['strategy_ret'] > 0).sum()
    win_rate = win_trades / total_trades if total_trades > 0 else 0
    
    max_dd = 0
    peak = df_res['cum_strategy'].iloc[0] if not df_res.empty else 1
    for val in df_res['cum_strategy']:
        if val > peak: peak = val
        dd = (peak - val) / peak
        if dd > max_dd: max_dd = dd
        
    strategy_total_return = df_res['cum_strategy'].iloc[-1] - 1 if not df_res.empty else 0
    market_total_return = df_res['cum_market'].iloc[-1] - 1 if not df_res.empty else 0
    
    print("\n" + "="*40)
    print("📈 --- 回測績效報告 (大盤短線當沖 v2 優化版) ---")
    print(f"測試期間: {df_res['date'].iloc[0].strftime('%Y-%m-%d')} ~ {df_res['date'].iloc[-1].strftime('%Y-%m-%d')}")
    print(f"觸發閾值: {threshold}")
    print(f"總交易天數: {len(df_res)} 天")
    print(f"策略進場次數: {total_trades} 次")
    print(f"勝率 (Win Rate): {win_rate*100:.2f}%")
    print(f"大盤買進持有報酬: {market_total_return*100:.2f}%")
    print(f"策略累積淨報酬: {strategy_total_return*100:.2f}%")
    print(f"最大回撤 (MDD): {max_dd*100:.2f}%")
    print("="*40)
    
    return df_res, strategy_total_return, win_rate

if __name__ == "__main__":
    run_historical_backtest("2024-01-01", threshold=0.3)
