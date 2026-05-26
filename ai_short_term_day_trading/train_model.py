import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
import os
import json
from data_engine import DayTradingDataEngine
from composite_ai import CompositeDayTradingAI
from model_manager import TradingModelManager

def train_trading_model(df_daily_chips_input=None):
    """
    完整訓練流程：包含三大法人籌碼變數與高基期選擇權利金目標優化
    """
    engine = DayTradingDataEngine()
    # 擴大回測/訓練天數，確保能涵蓋長週期籌碼與均線運算
    df_raw, _ = engine.fetch_active_option_intraday_data(days=60)

    if df_raw.empty:
        print("無法獲取即時/歷史 K 線數據")
        return

    # 1. 融入三大法人籌碼數據
    if df_daily_chips_input is not None:
        df = engine.integrate_institutional_chips(df_raw, df_daily_chips_input)
    else:
        # 若無外部輸入，模擬生成相容結構欄位以維持 pipeline 暢通
        print("⚠️ 未偵測到外部籌碼日誌，啟用基本結構適配中...")
        df = df_raw.copy()
        df['foreign_net_oi'] = 0.0
        df['dealer_net_oi'] = 0.0
        df['foreign_oi_zscore'] = 0.0
        df['dealer_oi_zscore'] = 0.0
        df['foreign_oi_momentum'] = 0.0
        df['dealer_oi_momentum'] = 0.0
        df['pc_ratio'] = 1.0
        df['pc_ratio_momentum'] = 0.0

    # 2. 自動過濾特徵欄位 (排除非結構化時間欄位)
    exclude_cols = ['date', 'time', 'date_only', 'day_of_week', 'label', 'future_max', 'future_min', 'max_up_ret', 'max_down_ret']
    feature_cols = [c for c in df.columns if c not in exclude_cols]

    # 進行特徵空間標準化
    df_feat = df[feature_cols].copy()
    mean = df_feat.mean()
    std = df_feat.std() + 1e-9
    df_normalized = (df_feat - mean) / std

    # 3. 建立 42,442 點高基期專屬標籤：未來 5 根 K 線內漲跌幅必須大於 8%
    future_window = 5
    df['future_max'] = df['Close'].shift(-future_window).rolling(window=future_window).max()
    df['future_min'] = df['Close'].shift(-future_window).rolling(window=future_window).min()
    df['max_up_ret'] = (df['future_max'] - df['Close']) / df['Close']
    df['max_down_ret'] = (df['Close'] - df['future_min']) / df['Close']

    df.dropna(subset=['future_max', 'future_min'], inplace=True)
    df_normalized = df_normalized.iloc[:len(df)]

    THRESHOLD = 0.08 # 權利金變動幅度門檻
    def classify_trend(row):
        if row['max_up_ret'] > THRESHOLD and row['max_up_ret'] > row['max_down_ret']: return 2 # 波段噴發多單
        if row['max_down_ret'] > THRESHOLD and row['max_down_ret'] > row['max_up_ret']: return 0 # 波段崩跌空單
        return 1 # 雜訊盤整

    df['label'] = df.apply(classify_trend, axis=1)

    # 4. 時序滑動視窗構建 (Windowing)
    X, y = [], []
    window_size = 25
    data_values = df_normalized.values
    label_values = df['label'].values

    for i in range(window_size, len(data_values)):
        X.append(data_values[i-window_size:i])
        y.append(label_values[i])

    X = torch.tensor(np.array(X), dtype=torch.float32)
    y = torch.tensor(np.array(y), dtype=torch.long)

    # 5. 模型初始化 (網路結構維度隨籌碼特徵自動擴展)
    input_dim = len(feature_cols)
    model = CompositeDayTradingAI(input_dim=input_dim, d_model=128, nhead=8, num_layers=3)
    optimizer = optim.Adam(model.parameters(), lr=0.0002, weight_decay=1e-4)

    # 反比類別權重計算，修正盤整樣本過多的不平衡問題
    counts = df['label'].value_counts().sort_index().values
    weights = 1.0 / (counts + 1e-9)
    weights = weights / weights.sum() * 3.0
    class_weights = torch.tensor(weights, dtype=torch.float32)

    criterion = nn.CrossEntropyLoss(weight=class_weights)

    print(f"🚀 開始特徵融合分類訓練... 輸入特徵維度: {input_dim}, 總樣本數: {len(X)}")
    model.train()
    epochs = 100
    batch_size = 64

    for epoch in range(epochs):
        epoch_loss = 0
        for i in range(0, len(X), batch_size):
            batch_X = X[i:i+batch_size]
            batch_y = y[i:i+batch_size]

            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        if (epoch+1) % 10 == 0:
            print(f"Epoch [{epoch+1}/{epochs}], Loss: {epoch_loss/(len(X)/batch_size):.4f}")

    # 6. 版本化儲存
    manager = TradingModelManager(model_dir=os.path.join(os.path.dirname(__file__), "saved_models"))
    manager.save_model(model, optimizer, {"loss": epoch_loss/len(X)}, {"window_size": window_size, "epochs": epochs})

    norm_params = {"mean": mean.to_dict(), "std": std.to_dict(), "feature_cols": feature_cols}
    with open(os.path.join(os.path.dirname(__file__), "saved_models", "norm_params.json"), "w", encoding='utf-8') as f: json.dump(norm_params, f)
    print("✅ 籌碼特徵融合模型訓練完畢。")

if __name__ == "__main__":
    # 可在此讀入由交易所或外部資料庫下載的三大法人歷史 CSV 進行真實訓練
    # ex: df_chips_history = pd.read_csv("institutional_daily.csv")
    train_trading_model(df_daily_chips_input=None)