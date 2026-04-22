import os
import sys
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import pandas_ta as ta
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset

# 1. 設置專案路徑
root_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if root_path not in sys.path:
    sys.path.insert(0, root_path)

from database import DatabaseManager

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---
# 2. 精簡版 LSTM 模型 (提升泛化能力)
# ---
class MultiFactorLSTM(nn.Module):
    def __init__(self, input_size, hidden_size=128, num_layers=2, output_size=1):
        super(MultiFactorLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # 減少層數至 2 層，增加 Dropout 比例防止過擬合
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=0.4)
        self.bn = nn.BatchNorm1d(hidden_size)
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, 64),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(64, output_size)
        )

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)
        out, _ = self.lstm(x, (h0, c0))
        out = self.bn(out[:, -1, :])
        return self.fc(out)

# ---
# 3. 收益率資料處理器
# ---
class StockDataProcessor:
    def __init__(self, window_size=60):
        self.window_size = window_size
        self.feature_scaler = StandardScaler()
        self.target_scaler = StandardScaler()
        # 加入 'Log_Ret' 作為核心特徵
        self.feature_cols = ['Log_Ret', 'Volume', 'RSI', 'MACD', 'SMA20', 'Foreign']

    def add_indicators(self, df):
        df = df.copy()
        for col in ['Close', 'Open', 'High', 'Low', 'Volume', 'Foreign']:
            df[col] = pd.to_numeric(df[col], errors='coerce')

        # 計算對數收益率 (Stationary Target)
        df['Log_Ret'] = np.log(df['Close'] / df['Close'].shift(1))

        # 其他指標
        df['RSI'] = ta.rsi(df['Close'], length=14)
        macd = ta.macd(df['Close'])
        df['MACD'] = macd['MACD_12_26_9']
        df['SMA20'] = ta.sma(df['Close'], length=20)

        return df.dropna()

    def process_split_data(self, df_train, df_test):
        df_train = self.add_indicators(df_train)
        df_test = self.add_indicators(df_test)

        train_f = df_train[self.feature_cols].values
        train_t = df_train[['Log_Ret']].values # 預測目標改為收益率
        test_f = df_test[self.feature_cols].values
        test_t = df_test[['Log_Ret']].values

        scaled_train_f = self.feature_scaler.fit_transform(train_f)
        scaled_train_t = self.target_scaler.fit_transform(train_t)

        scaled_test_f = self.feature_scaler.transform(test_f)
        scaled_test_t = self.target_scaler.transform(test_t)

        X_train, y_train = self._create_windows(scaled_train_f, scaled_train_t)
        X_test, y_test = self._create_windows(scaled_test_f, scaled_test_t)

        return (torch.FloatTensor(X_train), torch.FloatTensor(y_train),
                torch.FloatTensor(X_test), torch.FloatTensor(y_test), df_test)

    def _create_windows(self, features, targets):
        X, y = [], []
        for i in range(len(features) - self.window_size):
            X.append(features[i:i+self.window_size])
            y.append(targets[i+self.window_size])
        return np.array(X), np.array(y)

# ---
# 4. 訓練與動態價格還原
# ---
def train_and_predict(stock_id="2330"):
    plt.rcParams['font.sans-serif'] = ['Microsoft JhengHei']
    plt.rcParams['axes.unicode_minus'] = False

    db = DatabaseManager()
    df = db.load_dataframe(f"stock_{stock_id}_daily").sort_values('date').reset_index(drop=True)

    split_idx = int(len(df) * 0.8)
    df_train, df_test = df.iloc[:split_idx], df.iloc[split_idx:]

    processor = StockDataProcessor(window_size=60)
    X_train, y_train, X_test, y_test, processed_test_df = processor.process_split_data(df_train, df_test)

    train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=64, shuffle=True)
    model = MultiFactorLSTM(input_size=X_train.shape[2]).to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0005) # 降低 Learning Rate 追求穩定

    print(f"🚀 收益率預測模型啟動 (個股: {stock_id})")

    epochs = 300
    model.train()
    for epoch in range(epochs):
        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            optimizer.zero_grad()
            loss = criterion(model(batch_x), batch_y)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        if (epoch+1) % 20 == 0:
            print(f'Epoch [{epoch+1}/{epochs}], Avg Loss: {epoch_loss/len(train_loader):.6f}')

    # 預測與還原
    model.eval()
    with torch.no_grad():
        preds_ret_scaled = model(X_test.to(device)).cpu().numpy()
        # 反向縮放得到預測的「對數收益率」
        preds_ret = processor.target_scaler.inverse_transform(preds_ret_scaled)

        # 價格還原邏輯：P_t = P_{t-1} * exp(r_t)
        # 注意：我們需要對應回 df_test 裡的 Close 價格
        actual_close = processed_test_df['Close'].values[processor.window_size:]
        prev_close = processed_test_df['Close'].values[processor.window_size-1:-1]

        # 預測價格 = 前一天實際價格 * exp(預測收益率)
        predicted_prices = prev_close * np.exp(preds_ret.flatten())

    # 繪圖
    plt.figure(figsize=(15, 7))
    plt.plot(actual_close, label='實際收盤價', color='#1a73e8', alpha=0.8)
    plt.plot(predicted_prices, label='收益率還原預測值', color='#d93025', linestyle='--')

    plt.title(f'{stock_id} 收益率模型預測圖 (LSTM)', fontsize=16)
    plt.legend()
    plt.grid(True, alpha=0.3)

    save_path = f"ai_research/{stock_id}_returns_model.png"
    plt.savefig(save_path)
    plt.show()

if __name__ == "__main__":
    train_and_predict("2330")