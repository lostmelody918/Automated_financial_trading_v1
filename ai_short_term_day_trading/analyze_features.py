import pandas as pd
import numpy as np

def analyze_trades():
    import os
    file_path = os.path.join(os.path.dirname(__file__), 'data_learn', 'trade_features_log.csv')
    df = pd.read_csv(file_path)
    if df.empty:
        print("No trades to analyze.")
        return

    df['is_win'] = df['ret'] > 0
    
    print("--- 交易特徵分析 (Trade Feature Analysis) ---")
    print(f"Total Trades: {len(df)}")
    print(f"Win Rate: {df['is_win'].mean()*100:.2f}%")
    
    # 根據勝負來分群分析平均特徵
    features_to_analyze = ['prob_up', 'prob_down', 'atr', 'macd_hist', 'rsi', 'vwap_bias', 'is_squeeze', 'n225_ret', 'n225_slope_5', 'tsm_ret_1d', 'vix_ret_1d', 'ixic_ret_1d']
    
    # 分為做多(1)與做空(-1)
    df_long = df[df['signal'] == 1]
    df_short = df[df['signal'] == -1]
    
    print(f"\n[做多 Long Trades] Count: {len(df_long)}, Win Rate: {df_long['is_win'].mean()*100:.2f}%")
    if not df_long.empty:
        print(df_long.groupby('is_win')[features_to_analyze].mean().T)
        
    print(f"\n[做空 Short Trades] Count: {len(df_short)}, Win Rate: {df_short['is_win'].mean()*100:.2f}%")
    if not df_short.empty:
        print(df_short.groupby('is_win')[features_to_analyze].mean().T)

    # 尋找更好的過濾條件
    print("\n--- 優化建議 (Optimization Ideas) ---")
    
    # 例如：如果只在 prob_up > 0.8 做多？
    if not df_long.empty:
        high_prob_long = df_long[df_long['prob_up'] > 0.8]
        print(f"Long if prob_up > 0.8 -> Count: {len(high_prob_long)}, Win Rate: {high_prob_long['is_win'].mean()*100:.2f}%")
        
    if not df_short.empty:
        high_prob_short = df_short[df_short['prob_down'] > 0.8]
        print(f"Short if prob_down > 0.8 -> Count: {len(high_prob_short)}, Win Rate: {high_prob_short['is_win'].mean()*100:.2f}%")

    # 嘗試結合 VWAP 乖離率
    if not df_long.empty:
        vwap_filtered = df_long[df_long['vwap_bias'] < 0.005] # 不要追高
        print(f"Long if vwap_bias < 0.005 -> Count: {len(vwap_filtered)}, Win Rate: {vwap_filtered['is_win'].mean()*100:.2f}%")
        
    if not df_short.empty:
        vwap_filtered = df_short[df_short['vwap_bias'] > -0.005] # 不要殺低
        print(f"Short if vwap_bias > -0.005 -> Count: {len(vwap_filtered)}, Win Rate: {vwap_filtered['is_win'].mean()*100:.2f}%")
        
if __name__ == "__main__":
    analyze_trades()
