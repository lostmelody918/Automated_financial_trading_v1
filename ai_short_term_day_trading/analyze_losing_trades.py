import pandas as pd
import os

log_path = os.path.join(os.path.dirname(__file__), "data_learn", "trade_features_log.csv")
if not os.path.exists(log_path):
    print("No log found.")
    exit()

df = pd.read_csv(log_path)
df['entry_date'] = pd.to_datetime(df['entry_date'])
df['hour'] = df['entry_date'].dt.hour
df['minute'] = df['entry_date'].dt.minute
df['is_win'] = df['ret'] > 0

loss_df = df[~df['is_win']]
win_df = df[df['is_win']]

with open(os.path.join(os.path.dirname(__file__), "data_learn", "loss_analysis_report.txt"), "w", encoding="utf-8") as f:
    f.write("=== 虧損交易分析報告 (Loss Analysis Report) ===\n")
    f.write(f"總交易次數: {len(df)}\n")
    f.write(f"獲利次數: {len(win_df)}, 虧損次數: {len(loss_df)}\n")
    f.write(f"勝率: {len(win_df)/len(df)*100:.2f}%\n\n")

    f.write("--- 各時段虧損分佈 (Losses by Hour) ---\n")
    hourly_loss = loss_df['hour'].value_spacing = loss_df.groupby('hour').size()
    hourly_win = win_df.groupby('hour').size()
    for hour in range(8, 15):
        l_cnt = hourly_loss.get(hour, 0)
        w_cnt = hourly_win.get(hour, 0)
        f.write(f"Hour {hour}: Losses = {l_cnt}, Wins = {w_cnt}\n")
    
    f.write("\n--- 特徵平均值比較 (Feature Averages: Win vs Loss) ---\n")
    features_to_check = ['atr', 'macd_hist', 'rsi', 'vwap_bias', 'vix_ret_1d', 'n225_ret', 'prob_up', 'prob_down']
    for feat in features_to_check:
        if feat in df.columns:
            mean_loss = loss_df[feat].mean()
            mean_win = win_df[feat].mean()
            f.write(f"{feat:12s} -> Win: {mean_win:8.4f} | Loss: {mean_loss:8.4f}\n")

    f.write("\n--- 虧損交易的極端值 (Extreme Values in Losses) ---\n")
    f.write("1. 高波動 (High ATR) 導致虧損比例:\n")
    high_atr_loss = len(loss_df[loss_df['atr'] > loss_df['atr'].median()])
    f.write(f"   大於中位數 ATR 的虧損筆數: {high_atr_loss} / {len(loss_df)}\n")
    
    f.write("2. VIX 暴增 (VIX Spike) 導致虧損:\n")
    vix_loss = len(loss_df[loss_df['vix_ret_1d'] > 0.05])
    f.write(f"   VIX 單日暴增 > 5% 的虧損筆數: {vix_loss} / {len(loss_df)}\n")

    f.write("3. 逆勢交易 (Counter-Trend MACD) 導致虧損:\n")
    counter_trend_loss = len(loss_df[((loss_df['signal']==1) & (loss_df['macd_hist']<0)) | ((loss_df['signal']==-1) & (loss_df['macd_hist']>0))])
    f.write(f"   逆勢 (多單但MACD<0 或 空單但MACD>0): {counter_trend_loss} / {len(loss_df)}\n")

print("Analysis complete. Report saved to data_learn/loss_analysis_report.txt")