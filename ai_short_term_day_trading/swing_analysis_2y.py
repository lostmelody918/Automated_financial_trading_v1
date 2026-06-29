import pandas as pd
import numpy as np
import os
import sys
sys.path.append(os.path.dirname(__file__))
from data_engine import DayTradingDataEngine

def analyze_2y_swings():
    print("啟動 2 年 (730天) 巨量波段特徵萃取引擎...")
    engine = DayTradingDataEngine()
    
    # 強制獲取 730 天
    df = engine.fetch_intraday_data(days=730)
    if df.empty:
        print("無法取得歷史數據。")
        return
        
    print(f"成功載入 {len(df)} 根 K 線數據。開始計算前瞻 2 小時 (120 分鐘) 波段特徵...")
    
    # 建立波段標籤
    # 假設 df 每一列是 1 分鐘 K 線
    lookforward = 120
    
    # 快速計算前瞻最大最小值 (使用 rolling 但反向)
    # df[::-1].rolling().max()[::-1]
    df_rev = df[::-1].copy()
    
    # 因為是 1 分鐘 K 線，120 根就是 120 分鐘
    future_max = df_rev['High'].rolling(window=lookforward, min_periods=1).max()[::-1]
    future_min = df_rev['Low'].rolling(window=lookforward, min_periods=1).min()[::-1]
    
    df['future_max'] = future_max
    df['future_min'] = future_min
    
    # 定義黃金做多波段 (2小時內噴超過 80 點，且回撤不超過 25 點)
    df['is_golden_long'] = ((df['future_max'] - df['Close']) >= 80) & ((df['Close'] - df['future_min']) <= 25)
    
    # 定義黃金做空波段 (2小時內殺超過 80 點，且反彈不超過 25 點)
    df['is_golden_short'] = ((df['Close'] - df['future_min']) >= 80) & ((df['future_max'] - df['Close']) <= 25)
    
    # 定義無價值死水 (2小時內最高最低振幅不超過 40 點)
    df['is_chop'] = (df['future_max'] - df['future_min']) <= 40
    
    print(f"標籤計算完成！")
    print(f"黃金做多波段樣本數: {df['is_golden_long'].sum()}")
    print(f"黃金做空波段樣本數: {df['is_golden_short'].sum()}")
    print(f"無價值死水樣本數: {df['is_chop'].sum()}")
    
    features_to_analyze = [
        'macd_hist', 'rsi_fast', 'dist_from_ma20', 'pullback_from_high', 'bounce_from_low',
        'spot_futures_proxy', 'bb_width', 'is_squeeze', 'slope_vwap', 'slope_ma20',
        'vol_surge_ratio', 'pv_divergence', 'obv_bias', 'obv_slope', 'orderbook_imbalance',
        'volume_delta', 'cvd_bias', 'close_frac_diff', 'trend_wavelet', 'noise_wavelet',
        'atr', 'lower_shadow', 'upper_shadow', 'momentum_explosion', 'gap_amplitude',
        'us_tw_gap_divergence', 'vol_range_ratio'
    ]
    
    # 過濾出存在這些特徵的欄位
    available_features = [f for f in features_to_analyze if f in df.columns]
    
    print("\n==================================================")
    print("📈 --- 黃金波段特徵突破口深度分析 ---")
    print("==================================================")
    
    analysis_results = []
    
    for f in available_features:
        mean_long = df.loc[df['is_golden_long'], f].mean()
        mean_short = df.loc[df['is_golden_short'], f].mean()
        mean_chop = df.loc[df['is_chop'], f].mean()
        mean_all = df[f].mean()
        
        analysis_results.append({
            'Feature': f,
            'Golden_Long': mean_long,
            'Golden_Short': mean_short,
            'Chop_Deadwater': mean_chop,
            'Global_Mean': mean_all
        })
        
        print(f"➤ 特徵 [{f}]")
        print(f"   - 黃金做多波段均值: {mean_long:8.4f}")
        print(f"   - 黃金做空波段均值: {mean_short:8.4f}")
        print(f"   - 死水盤均值      : {mean_chop:8.4f}")
        print(f"   - 全域平均值      : {mean_all:8.4f}")
        print("-" * 50)
        
    df_results = pd.DataFrame(analysis_results)
    os.makedirs("data_learn", exist_ok=True)
    out_path = os.path.join("data_learn", "golden_swing_analysis_2y.csv")
    df_results.to_csv(out_path, index=False, encoding='utf-8-sig')
    print(f"\n✅ 深度分析結果已匯出至 {out_path}")

if __name__ == '__main__':
    analyze_2y_swings()
