import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
import os
from data_engine import DayTradingDataEngine
from composite_ai import CompositeDayTradingAI
from model_manager import TradingModelManager

def train_trading_model():
    # 1. 獲取數據
    engine = DayTradingDataEngine()
    df, _ = engine.fetch_active_option_intraday_data(days=15) # 使用選項權資料訓練，解決 Domain Mismatch
    
    if df.empty:
        print("無法獲取數據")
        return

    # 2. 準備特徵
    feature_cols = [c for c in df.columns if c not in ['date', 'time', 'date_only', 'day_of_week']]
    
    # 標準化
    df_feat = df[feature_cols].copy()
    mean = df_feat.mean()
    std = df_feat.std() + 1e-9
    df_normalized = (df_feat - mean) / std
    
    # 3. 建立標籤 (分類: 0=大跌波段, 1=盤整, 2=大漲波段)
    # 預測未來 5 根 K 線內的最大變動率 (捕捉大斜率波段)
    future_window = 5
    df['future_max'] = df['Close'].shift(-future_window).rolling(window=future_window).max()
    df['future_min'] = df['Close'].shift(-future_window).rolling(window=future_window).min()

    df['max_up_ret'] = (df['future_max'] - df['Close']) / df['Close']
    df['max_down_ret'] = (df['Close'] - df['future_min']) / df['Close']

    df.dropna(inplace=True)
    df_normalized = df_normalized.iloc[:len(df)]

    # 標籤定義：波段漲跌幅大於 8% 視為有價差的波段，確保扣除成本後有利潤
    THRESHOLD = 0.08
    def classify_trend(row):
        if row['max_up_ret'] > THRESHOLD and row['max_up_ret'] > row['max_down_ret']: return 2 # 大漲波段
        if row['max_down_ret'] > THRESHOLD and row['max_down_ret'] > row['max_up_ret']: return 0 # 大跌波段
        return 1 # 盤整/無足夠價差

    df['label'] = df.apply(classify_trend, axis=1)

    X = []
    y = []
    window_size = 40
 # 增加窗口大小
    
    data_values = df_normalized.values
    label_values = df['label'].values
    
    for i in range(window_size, len(data_values)):
        X.append(data_values[i-window_size:i])
        y.append(label_values[i])
        
    X = torch.tensor(np.array(X), dtype=torch.float32)
    y = torch.tensor(np.array(y), dtype=torch.long)
    
    # 4. 初始化模型
    input_dim = len(feature_cols)
    model = CompositeDayTradingAI(input_dim=input_dim, d_model=128, nhead=8, num_layers=3) # 增加複雜度
    optimizer = optim.Adam(model.parameters(), lr=0.0002, weight_decay=1e-4) # 降低學習率
    
    # 計算類別權重
    counts = df['label'].value_counts().sort_index().values
    weights = 1.0 / (counts + 1e-9)
    weights = weights / weights.sum() * 3.0
    class_weights = torch.tensor(weights, dtype=torch.float32)
    print(f"⚖️ 類別權重: {class_weights}")
    
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    
    # 5. 訓練循環
    print(f"🚀 開始終極分類訓練... 樣本數: {len(X)}")
    model.train()
    epochs = 100 # 增加到 100 輪
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
        
        if (epoch+1) % 5 == 0:
            print(f"Epoch [{epoch+1}/{epochs}], Loss: {epoch_loss/(len(X)/batch_size):.4f}")
            
    # 6. 儲存模型
    manager = TradingModelManager(model_dir=os.path.join(os.path.dirname(__file__), "saved_models"))
    metrics = {"loss": epoch_loss/len(X)}
    hyperparams = {"window_size": window_size, "input_dim": input_dim, "epochs": epochs}
    manager.save_model(model, optimizer, metrics, hyperparams)
    
    # 儲存 Normalization 參數以便推理使用
    norm_params = {
        "mean": mean.to_dict(),
        "std": std.to_dict(),
        "feature_cols": feature_cols
    }
    import json
    with open(os.path.join(os.path.dirname(__file__), "saved_models", "norm_params.json"), "w") as f:
        json.dump(norm_params, f)
    print("✅ 訓練完成並儲存模型與正規化參數。")

if __name__ == "__main__":
    train_trading_model()
