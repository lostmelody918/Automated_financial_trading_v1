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
# 2. Attention 模組群
# ---
class FeatureAttention(nn.Module):
    def __init__(self, input_size):
        super(FeatureAttention, self).__init__()
        self.feature_weights = nn.Sequential(
            nn.Linear(input_size, input_size),
            nn.Sigmoid()
        )

    def forward(self, x):
        weights = self.feature_weights(x)
        return x * weights, weights

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
    def __init__(self, input_size, hidden_size=64, num_layers=2, output_size=3):
        super(ClassificationAttentionLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.feature_attention = FeatureAttention(input_size)
        self.embedding = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.LeakyReLU(0.1)
        )
        self.lstm = nn.LSTM(hidden_size, hidden_size, num_layers, batch_first=True, dropout=0.2)
        self.time_attention = ContextualAttention(hidden_size)
        self.ln = nn.LayerNorm(hidden_size * 2)
        self.fc = nn.Sequential(
            nn.Linear(hidden_size * 2, 32),
            nn.LeakyReLU(0.1),
            nn.Dropout(0.2),
            nn.Linear(32, output_size)
        )

    def forward(self, x):
        weighted_x, _ = self.feature_attention(x)
        emb_x = self.embedding(weighted_x)

        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)
        lstm_out, _ = self.lstm(emb_x, (h0, c0))

        last_hidden = lstm_out[:, -1, :]
        context = self.time_attention(lstm_out, last_hidden)

        combined = torch.cat([context, last_hidden], dim=1)
        out = self.ln(combined)
        return self.fc(out)

# ---
# 4. 數據處理器 (加入波動收斂與加速度特徵)
# ---
class StockDataProcessor:
    def __init__(self, window_size=15):
        self.window_size = window_size
        self.feature_scaler = StandardScaler()
        # 🚀 新增 BBW (布林寬度) 與 MACD_Hist (動能加速度)
        self.feature_cols = [
            'Log_Ret', 'Volume_Ratio', 'RSI', 'MACD', 'MACD_Hist', 'Bias20', 'BBW',
            'Slope_5D', 'Foreign_Trend', 'Momentum_3', 'Momentum_5', 'Breakout_20D',
            'Trend_Regime', 'Nasdaq_Ret', 'SOX_Ret', 'VIX_Ret', 'ADR_Premium'
        ]

    def add_indicators(self, df, stock_id, is_us=False):
        df = df.copy()
        for col in ['Close', 'Open', 'High', 'Low', 'Volume']:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        df['date'] = pd.to_datetime(df['date']).dt.strftime('%Y-%m-%d')
        df['Foreign'] = pd.to_numeric(df.get('Foreign', 0), errors='coerce').fillna(0)

        df = KLineAnalyzer.add_indicators(df)

        df['Log_Ret'] = np.log(df['Close'] / df['Close'].shift(1))
        df['Bias20'] = (df['Close'] - df['MA20']) / df['MA20']
        df['Volume_Ratio'] = df['Volume'] / df['Volume'].rolling(20, min_periods=1).mean()

        # 🌟 新增 1：MACD 柱狀圖 (捕捉加速度)
        macd = ta.macd(df['Close'])
        if macd is not None:
            df['MACD'] = macd['MACD_12_26_9']
            df['MACD_Hist'] = macd['MACDh_12_26_9'] # 柱狀圖
        else:
            df['MACD'], df['MACD_Hist'] = 0, 0

        # 🌟 新增 2：布林通道寬度 BBW (捕捉波動收斂/盤整)
        std20 = df['Close'].rolling(20, min_periods=1).std()
        df['BBW'] = (4 * std20) / df['MA20']

        # 趨勢與斜率
        df['Breakout_20D'] = df['Close'] / df['High'].rolling(20, min_periods=1).max()
        df['MA60'] = df['Close'].rolling(60, min_periods=1).mean()
        df['Trend_Regime'] = df['MA20'] / df['MA60']
        df['Foreign_Trend'] = df['Foreign'].rolling(10, min_periods=1).mean()

        df['MA5'] = df['Close'].rolling(5, min_periods=1).mean()
        df['Slope_5D'] = df['MA5'] / df['MA5'].shift(3) - 1

        # --- 🚀 終極寬容標籤：容忍洗盤，捕捉二波 ---
        future_highs = [df['High'].shift(-i) for i in range(1, 6)]
        future_lows = [df['Low'].shift(-i) for i in range(1, 6)]

        df['Future_Max_5D'] = np.maximum.reduce(future_highs) / df['Close'] - 1
        df['Future_Min_5D'] = np.minimum.reduce(future_lows) / df['Close'] - 1

        df['Next_Ret'] = np.log(df['Close'].shift(-1) / df['Close'])

        # --- 外部特徵 ---
        for col in ['Sentiment_Score', 'ADR_Premium', 'VIX_Ret', 'Nasdaq_Ret', 'SOX_Ret']:
            if col not in df.columns: df[col] = 0.0

        if not is_us:
            try:
                fetcher = DataFetcher()
                nasdaq = fetcher.fetch_us_stock_daily("^IXIC", df['date'].min(), df['date'].max())
                sox = fetcher.fetch_us_stock_daily("^SOX", df['date'].min(), df['date'].max())
                vix = fetcher.fetch_us_stock_daily("^VIX", df['date'].min(), df['date'].max())

                for label, index_df in [('Nasdaq', nasdaq), ('SOX', sox), ('VIX', vix)]:
                    if not index_df.empty:
                        index_df['date'] = pd.to_datetime(index_df['date']).dt.strftime('%Y-%m-%d')
                        index_df[f'{label}_Ret'] = np.log(index_df['Close'] / index_df['Close'].shift(1))
                        index_df[f'{label}_Ret'] = index_df[f'{label}_Ret'].shift(1)
                        df = pd.merge(df, index_df[['date', f'{label}_Ret']], on='date', how='left')

                adr_mapping = {"2330": {"symbol": "TSM", "ratio": 5}}
                if stock_id in adr_mapping:
                    adr_info = adr_mapping[stock_id]
                    adr_df = fetcher.fetch_us_stock_daily(adr_info['symbol'], df['date'].min(), df['date'].max())
                    usd_twd = fetcher.fetch_us_stock_daily("TWD=X", df['date'].min(), df['date'].max())

                    if not adr_df.empty and not usd_twd.empty:
                        adr_df['date'] = pd.to_datetime(adr_df['date']).dt.strftime('%Y-%m-%d')
                        usd_twd['date'] = pd.to_datetime(usd_twd['date']).dt.strftime('%Y-%m-%d')
                        adr_df['ADR_Close_prev'] = adr_df['Close'].shift(1)
                        usd_twd['USDTWD_prev'] = usd_twd['Close'].shift(1)
                        merged_adr = pd.merge(df[['date', 'Close']], adr_df[['date', 'ADR_Close_prev']], on='date', how='left')
                        merged_adr = pd.merge(merged_adr, usd_twd[['date', 'USDTWD_prev']], on='date', how='left')
                        df['ADR_Premium'] = ((merged_adr['ADR_Close_prev'] * merged_adr['USDTWD_prev']) / adr_info['ratio'] / merged_adr['Close'] - 1) * 100
            except Exception:
                pass

        for col in self.feature_cols:
            if col not in df.columns: df[col] = 0.0
            else: df[col] = df[col].ffill().fillna(0)

        return df.dropna()

    def process_split_data(self, df_train, df_test, stock_id, is_us=False):
        df_train = self.add_indicators(df_train, stock_id, is_us)
        df_test = self.add_indicators(df_test, stock_id, is_us)

        train_f = df_train[self.feature_cols].values
        test_f = df_test[self.feature_cols].values

        train_max = df_train[['Future_Max_5D']].values
        train_min = df_train[['Future_Min_5D']].values
        test_max = df_test[['Future_Max_5D']].values
        test_min = df_test[['Future_Min_5D']].values

        raw_ret_test = df_test[['Next_Ret']].values

        scaled_train_f = self.feature_scaler.fit_transform(train_f)
        scaled_test_f = self.feature_scaler.transform(test_f)

        X_train, y_train, _ = self._create_windows(scaled_train_f, train_max, train_min)
        X_test, y_test, ret_test_aligned = self._create_windows(scaled_test_f, test_max, test_min, raw_ret_test)

        return torch.FloatTensor(X_train), torch.LongTensor(y_train), \
               torch.FloatTensor(X_test), torch.LongTensor(y_test), ret_test_aligned

    def _create_windows(self, features, targets_max, targets_min, raw_ret_test=None):
        X, y, raw_returns = [], [], []

        for i in range(len(features) - self.window_size):
            X.append(features[i:i+self.window_size])
            f_max = targets_max[i+self.window_size][0]
            f_min = targets_min[i+self.window_size][0]

            if raw_ret_test is not None:
                raw_returns.append(raw_ret_test[i+self.window_size][0])

            # 🚀 屏障加寬：要求漲幅達 3%，且容忍 2.5% 的深蹲洗盤！
            if f_max > 0.030 and f_min > -0.025:
                y.append(2) # 續漲/牛旗波段
            elif f_min < -0.030 and f_max < 0.025:
                y.append(0) # 續跌波段
            else:
                y.append(1) # 無方向震盪

        return np.array(X), np.array(y), np.array(raw_returns)

# ---
# 5. 主訓練流程
# ---
def train_and_predict(stock_id="2330", market="TW"):
    plt.rcParams['font.sans-serif'] = ['Microsoft JhengHei']
    plt.rcParams['axes.unicode_minus'] = False
    is_us = (market == "US")

    db = DatabaseManager()
    table_name = f"us_stock_{stock_id}_daily" if is_us else f"stock_{stock_id}_daily"
    df = db.load_dataframe(table_name).sort_values('date').reset_index(drop=True)

    if df.empty or len(df) < 100: return

    split_idx = int(len(df) * 0.8)
    df_train, df_test = df.iloc[:split_idx], df.iloc[split_idx:]

    processor = StockDataProcessor(window_size=15)
    X_train, y_train, X_test, y_test, raw_ret_test = processor.process_split_data(df_train, df_test, stock_id, is_us)

    train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=32, shuffle=True)
    model = ClassificationAttentionLSTM(input_size=X_train.shape[2], output_size=3).to(device)

    class_counts = np.bincount(y_train.numpy(), minlength=3)
    weights = len(y_train) / (3.0 * (class_counts + 1e-5))
    class_weights = torch.FloatTensor(weights).to(device)

    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=50)

    print(f"🚀 【牛旗中繼突破型】 啟動 (標的: {stock_id})")
    print(f"🔹 標籤分佈 -> 續跌: {class_counts[0]}, 震盪: {class_counts[1]}, 續漲: {class_counts[2]}")

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

            _, predicted = torch.max(outputs.data, 1)
            total += batch_y.size(0)
            correct += (predicted == batch_y).sum().item()

        scheduler.step()
        if (epoch+1) % 25 == 0:
            print(f'Epoch [{epoch+1}/{epochs}], Loss: {epoch_loss/len(train_loader):.4f}, Train Acc: {100 * correct / total:.2f}%')

    # -----------------------------------------
    # 產出策略訊號
    # -----------------------------------------
    model.eval()
    with torch.no_grad():
        outputs = model(X_test.to(device))
        probs = F.softmax(outputs, dim=1).cpu().numpy()
        signals = np.zeros(len(probs))

        # 信心門檻 45%，要求更高的確信度才出手
        conf_threshold = 0.45

        for i in range(len(probs)):
            if probs[i, 2] > conf_threshold: signals[i] = 1.0
            elif probs[i, 0] > conf_threshold: signals[i] = -1.0
            else: signals[i] = 0.0

        strategy_returns = signals * raw_ret_test.flatten()
        cumulative_market = np.exp(np.cumsum(raw_ret_test.flatten())) * 100
        cumulative_strategy = np.exp(np.cumsum(strategy_returns)) * 100

        active_days = (signals != 0)
        if np.sum(active_days) > 0:
            correct_trades = ((signals[active_days] > 0) & (raw_ret_test.flatten()[active_days] > 0)) | \
                             ((signals[active_days] < 0) & (raw_ret_test.flatten()[active_days] < 0))
            strategy_win_rate = (correct_trades.sum() / np.sum(active_days)) * 100
        else:
            strategy_win_rate = 0.0

        print(f"\n📈 --- AI 牛旗突破報告 ---")
        print(f"總交易天數: {len(raw_ret_test)} 天")
        print(f"強勢抱緊: {np.sum(signals == 1)} 天 | 趨勢放空: {np.sum(signals == -1)} 天 | 收斂觀望: {np.sum(signals == 0)} 天")
        print(f"AI 波段勝率: {strategy_win_rate:.2f}%")
        print(f"大盤買進持有報酬: {cumulative_market[-1]:.2f}%")
        print(f"AI 決策最終報酬: {cumulative_strategy[-1]:.2f}%")

    # -----------------------------------------
    # 繪製資金曲線
    # -----------------------------------------
    plt.figure(figsize=(14, 8))
    plt.plot(cumulative_market, label=f'{stock_id} 買進持有', color='#bdc3c7', lw=2, alpha=0.8)
    plt.plot(cumulative_strategy, label='AI 寬容屏障報酬', color='#d93025', lw=2.5)

    buy_idx = np.where(signals == 1.0)[0]
    short_idx = np.where(signals == -1.0)[0]

    plt.scatter(buy_idx, cumulative_strategy[buy_idx], marker='^', color='#27ae60', s=70, label='強勢抱緊', zorder=5)
    plt.scatter(short_idx, cumulative_strategy[short_idx], marker='v', color='#8e44ad', s=70, label='趨勢破線放空', zorder=5)

    plt.title(f'{stock_id} 機器學習：牛旗突破與波動收斂捕捉', fontsize=16, fontweight='bold')
    plt.xlabel('測試期天數')
    plt.ylabel('累積資金成長率 (%)')
    plt.axhline(100, color='black', linestyle='--', alpha=0.3)
    plt.legend(loc='upper left')
    plt.grid(True, linestyle=':', alpha=0.6)

    save_path = f"ai_research/{stock_id}_classification_equity.png"
    plt.savefig(save_path, dpi=300)
    plt.show()

    # ==========================================
    # 🔍 附錄：XAI 特徵重要性分析
    # ==========================================
    print("\n🕵️‍♂️ 啟動 XAI 特徵重要性分析...")

    model.eval()
    with torch.no_grad():
        base_outputs = model(X_test.to(device))
        _, base_preds = torch.max(base_outputs, 1)
        base_acc = (base_preds == y_test.to(device)).float().mean().item()

    importances = {}
    for i, feature_name in enumerate(processor.feature_cols):
        X_shuffled = X_test.clone()
        indices = torch.randperm(X_shuffled.size(0))
        X_shuffled[:, :, i] = X_shuffled[indices, :, i]

        with torch.no_grad():
            shuffled_outputs = model(X_shuffled.to(device))
            _, shuffled_preds = torch.max(shuffled_outputs, 1)
            shuffled_acc = (shuffled_preds == y_test.to(device)).float().mean().item()

        importances[feature_name] = base_acc - shuffled_acc

    df_imp = pd.DataFrame.from_dict(importances, orient='index', columns=['Importance'])
    df_imp = df_imp.sort_values(by='Importance', ascending=True)

    plt.figure(figsize=(10, 8))
    colors = ['#27ae60' if val > 0 else '#e74c3c' for val in df_imp['Importance']]
    df_imp['Importance'].plot(kind='barh', color=colors)
    plt.title(f'{stock_id} 波段特徵重要性 (包含 BBW 波動收斂)', fontsize=14, fontweight='bold')
    plt.xlabel('準確率下降幅度 (影響力)', fontsize=12)
    plt.ylabel('特徵名稱', fontsize=12)
    plt.grid(True, axis='x', linestyle='--', alpha=0.6)

    plt.tight_layout()
    save_path_imp = f"ai_research/{stock_id}_feature_importance.png"
    plt.savefig(save_path_imp, dpi=300)
    plt.show()

if __name__ == "__main__":
    train_and_predict("2330", market="TW")