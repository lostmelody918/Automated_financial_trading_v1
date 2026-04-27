import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
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

try:
    from database import DatabaseManager
    from analysis import KLineAnalyzer
    from data_fetcher import DataFetcher
except ImportError:
    print("❌ 警告：找不到必要的模組，請確認路徑。")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---
# 2. Attention 模組
# ---
class ContextualAttention(nn.Module):
    def __init__(self, hidden_size):
        super(ContextualAttention, self).__init__()
        self.query_proj = nn.Linear(hidden_size, hidden_size)
        self.key_proj = nn.Linear(hidden_size, hidden_size)

    def forward(self, lstm_output, last_hidden):
        query = self.query_proj(last_hidden).unsqueeze(1)
        keys = self.key_proj(lstm_output)
        scores = torch.bmm(query, keys.transpose(1, 2))
        weights = F.softmax(scores, dim=2)
        context = torch.bmm(weights, lstm_output).squeeze(1)
        return context

# ---
# 3. 分類型 Attention-LSTM 模型架構
# ---
class ClassificationAttentionLSTM(nn.Module):
    # ⚠️ 關鍵：output_size 變為 3 (對應 做空0, 盤整1, 做多2)
    def __init__(self, input_size, hidden_size=64, num_layers=2, output_size=3):
        super(ClassificationAttentionLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.embedding = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.LeakyReLU(0.1)
        )

        self.lstm = nn.LSTM(hidden_size, hidden_size, num_layers, batch_first=True, dropout=0.2)
        self.attention = ContextualAttention(hidden_size)
        self.ln = nn.LayerNorm(hidden_size * 2)

        self.fc = nn.Sequential(
            nn.Linear(hidden_size * 2, 32),
            nn.LeakyReLU(0.1),
            nn.Dropout(0.2),
            nn.Linear(32, output_size) # 輸出 3 個類別的 Logits
        )

    def forward(self, x):
        emb_x = self.embedding(x)
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)
        lstm_out, _ = self.lstm(emb_x, (h0, c0))

        last_hidden = lstm_out[:, -1, :]
        context = self.attention(lstm_out, last_hidden)
        combined = torch.cat([context, last_hidden], dim=1)

        out = self.ln(combined)
        return self.fc(out)

# ---
# 4. 數據處理器 (標籤工程重構)
# ---
class StockDataProcessor:
    def __init__(self, window_size=15):
        self.window_size = window_size
        self.feature_scaler = StandardScaler()
        self.feature_cols = [
            'Log_Ret', 'Volume', 'RSI', 'MACD', 'MA20', 'Foreign',
            'Momentum_1', 'Momentum_3', 'Momentum_5', 'Momentum_15', 'Momentum_30',
            'Sentiment_Score', 'Nasdaq_Ret', 'SOX_Ret', 'VIX_Ret', 'Volatility', 'ADR_Premium'
        ]

    def add_indicators(self, df, stock_id, is_us=False):
        # ... (此段與原本的特徵提取完全相同，保留你的強大特徵) ...
        df = df.copy()
        for col in ['Close', 'Open', 'High', 'Low', 'Volume']:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        df['date'] = pd.to_datetime(df['date']).dt.strftime('%Y-%m-%d')
        df['Foreign'] = pd.to_numeric(df.get('Foreign', 0), errors='coerce').fillna(0)
        df = KLineAnalyzer.add_indicators(df)
        df['Log_Ret'] = np.log(df['Close'] / df['Close'].shift(1))

        if 'MACD' not in df.columns:
            macd = ta.macd(df['Close'])
            df['MACD'] = macd['MACD_12_26_9'] if macd is not None else 0

        # 情緒與美股特徵 (簡化為直接填0，請自行替換回原本的 get_new 抓取邏輯)
        # 為保持代碼精簡突出重點，此處略過 API 抓取細節，實務上請保留你的 DataFetcher
        for col in ['Sentiment_Score', 'Nasdaq_Ret', 'SOX_Ret', 'VIX_Ret', 'ADR_Premium', 'Volatility']:
            if col not in df.columns: df[col] = 0.0

        return df.dropna()

    def process_split_data(self, df_train, df_test, stock_id, is_us=False):
        df_train = self.add_indicators(df_train, stock_id, is_us)
        df_test = self.add_indicators(df_test, stock_id, is_us)

        train_f = df_train[self.feature_cols].values
        test_f = df_test[self.feature_cols].values

        # ⚠️ 關鍵：Target 不再做 StandardScaler，保留原始收益率供後續分類
        train_t = df_train[['Log_Ret']].values
        test_t = df_test[['Log_Ret']].values

        scaled_train_f = self.feature_scaler.fit_transform(train_f)
        scaled_test_f = self.feature_scaler.transform(test_f)

        X_train, y_train, _ = self._create_windows(scaled_train_f, train_t)
        X_test, y_test, raw_ret_test = self._create_windows(scaled_test_f, test_t)

        return torch.FloatTensor(X_train), torch.LongTensor(y_train), \
               torch.FloatTensor(X_test), torch.LongTensor(y_test), raw_ret_test

    def _create_windows(self, features, targets):
        X, y, raw_returns = [], [], []
        # 🚀 分類閾值設定：大於 0.3% 視為有效上漲
        threshold = 0.003

        for i in range(len(features) - self.window_size):
            X.append(features[i:i+self.window_size])
            ret = targets[i+self.window_size][0]
            raw_returns.append(ret)

            # 將連續數值轉換為 3 個決策類別
            if ret > threshold:
                y.append(2) # 類別 2: 看多
            elif ret < -threshold:
                y.append(0) # 類別 0: 看空
            else:
                y.append(1) # 類別 1: 雜訊/盤整

        return np.array(X), np.array(y), np.array(raw_returns)

# ---
# 5. 主訓練流程 (重構為分類任務)
# ---
def train_and_predict(stock_id="2330", market="TW"):
    plt.rcParams['font.sans-serif'] = ['Microsoft JhengHei']
    plt.rcParams['axes.unicode_minus'] = False
    is_us = (market == "US")

    db = DatabaseManager()
    table_name = f"us_stock_{stock_id}_daily" if is_us else f"stock_{stock_id}_daily"
    df = db.load_dataframe(table_name).sort_values('date').reset_index(drop=True)

    if df.empty or len(df) < 100:
        print(f"❌ 錯誤: 資料量不足。")
        return

    split_idx = int(len(df) * 0.8)
    df_train, df_test = df.iloc[:split_idx], df.iloc[split_idx:]

    processor = StockDataProcessor(window_size=15)
    X_train, y_train, X_test, y_test, raw_ret_test = processor.process_split_data(df_train, df_test, stock_id, is_us)

    train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=32, shuffle=True)

    model = ClassificationAttentionLSTM(input_size=X_train.shape[2], output_size=3).to(device)

    # 🚀 動態計算類別權重 (解決盤整天數過多導致的樣本不平衡)
    class_counts = np.bincount(y_train.numpy(), minlength=3)
    weights = len(y_train) / (3.0 * (class_counts + 1e-5)) # 加上微小值防止除以零
    class_weights = torch.FloatTensor(weights).to(device)

    # 改用 CrossEntropyLoss 訓練分類器
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=50)

    print(f"🚀 【決策分類型】 Advanced-LSTM 啟動 (標的: {stock_id})")
    print(f"🔹 訓練集標籤分佈 -> 空: {class_counts[0]}, 盤整: {class_counts[1]}, 多: {class_counts[2]}")

    epochs = 250
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0
        correct = 0
        total = 0

        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

            # 計算訓練準確率
            _, predicted = torch.max(outputs.data, 1)
            total += batch_y.size(0)
            correct += (predicted == batch_y).sum().item()

        scheduler.step()
        if (epoch+1) % 25 == 0:
            acc = 100 * correct / total
            print(f'Epoch [{epoch+1}/{epochs}], Loss: {epoch_loss/len(train_loader):.4f}, Train Acc: {acc:.2f}%')

    # -----------------------------------------
    # 讓模型自己產出策略：直接讀取類別預測
    # -----------------------------------------
    model.eval()
    with torch.no_grad():
        outputs = model(X_test.to(device))

        # 模型輸出的是機率分佈，取最大機率的類別為決策
        _, predicted_classes = torch.max(outputs.data, 1)
        predicted_classes = predicted_classes.cpu().numpy()
        actual_classes = y_test.numpy()

        # 🎯 將模型的類別轉換回交易訊號
        signals = np.zeros(len(predicted_classes))
        signals[predicted_classes == 2] = 1.0  # 模型預測做多
        signals[predicted_classes == 0] = -1.0 # 模型預測做空
        signals[predicted_classes == 1] = 0.0  # 模型預測盤整

        # 計算真實報酬
        strategy_returns = signals * raw_ret_test
        cumulative_market = np.exp(np.cumsum(raw_ret_test)) * 100
        cumulative_strategy = np.exp(np.cumsum(strategy_returns)) * 100

        # 統計純模型的決策勝率
        active_days = (signals != 0)
        if np.sum(active_days) > 0:
            # 模型看多且實際報酬大於0，或模型看空且實際報酬小於0
            correct_trades = ((signals[active_days] > 0) & (raw_ret_test[active_days] > 0)) | \
                             ((signals[active_days] < 0) & (raw_ret_test[active_days] < 0))
            strategy_win_rate = (correct_trades.sum() / np.sum(active_days)) * 100
        else:
            strategy_win_rate = 0.0

        print(f"\n📈 --- AI 純粹學習決策報告 ---")
        print(f"總交易天數: {len(raw_ret_test)} 天")
        print(f"AI 決定做多: {np.sum(signals == 1)} 天 | 做空: {np.sum(signals == -1)} 天 | 空手: {np.sum(signals == 0)} 天")
        print(f"純模型決策勝率 (無外加策略): {strategy_win_rate:.2f}%")
        print(f"大盤買進持有最終報酬: {cumulative_market[-1]:.2f}%")
        print(f"AI 裸機決策最終報酬: {cumulative_strategy[-1]:.2f}%")

    # -----------------------------------------
    # 繪製純 AI 決策資金曲線
    # -----------------------------------------
    plt.figure(figsize=(14, 8))
    plt.plot(cumulative_market, label=f'{stock_id} 買進持有', color='#bdc3c7', lw=2, alpha=0.8)
    plt.plot(cumulative_strategy, label='AI 分類器裸機報酬', color='#d93025', lw=2.5)

    buy_idx = np.where(signals == 1.0)[0]
    short_idx = np.where(signals == -1.0)[0]

    plt.scatter(buy_idx, cumulative_strategy[buy_idx], marker='^', color='#27ae60', s=50, label='AI 看多', zorder=5)
    plt.scatter(short_idx, cumulative_strategy[short_idx], marker='v', color='#8e44ad', s=50, label='AI 看空', zorder=5)

    plt.title(f'{stock_id} 機器學習原生策略：三分類決策模型', fontsize=16, fontweight='bold')
    plt.xlabel('測試期天數')
    plt.ylabel('累積資金成長率 (%)')
    plt.axhline(100, color='black', linestyle='--', alpha=0.3)
    plt.legend(loc='upper left')
    plt.grid(True, linestyle=':', alpha=0.6)

    save_path = f"ai_research/{stock_id}_classification_equity.png"
    plt.savefig(save_path, dpi=300)
    plt.show()

if __name__ == "__main__":
    train_and_predict("2330", market="TW")