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

# 🌟 模組 A：特徵注意力機制 (自動靜音干擾特徵)
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

# 🌟 模組 B：時間步注意力機制 (歷史情境回溯)
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
    # 輸出層變為 3 (0:看空, 1:盤整/空手, 2:看多)
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
# 4. 數據處理器 (波段標籤工程)
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
        df = df.copy()
        for col in ['Close', 'Open', 'High', 'Low', 'Volume']:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        df['date'] = pd.to_datetime(df['date']).dt.strftime('%Y-%m-%d')

        if 'Foreign' in df.columns:
            df['Foreign'] = pd.to_numeric(df['Foreign'], errors='coerce')
        else:
            df['Foreign'] = 0

        df = KLineAnalyzer.add_indicators(df)

        # 基礎當日特徵
        df['Log_Ret'] = np.log(df['Close'] / df['Close'].shift(1))

        # 🚀 關鍵 1：建立未來 3 天波段報酬 (Target)，教導 AI 放眼未來
        df['Target_Ret'] = np.log(df['Close'].shift(-3) / df['Close'])

        # 🚀 計算真實隔日報酬 (Next_Ret)，用於後續客觀計算資金曲線
        df['Next_Ret'] = np.log(df['Close'].shift(-1) / df['Close'])

        if 'MACD' not in df.columns:
            macd = ta.macd(df['Close'])
            df['MACD'] = macd['MACD_12_26_9'] if macd is not None else 0

        # --- 外部特徵 (若無法抓取則補 0) ---
        df['Sentiment_Score'] = 0.0
        df['ADR_Premium'] = 0.0
        df['VIX_Ret'] = 0.0

        try:
            if not is_us:
                from get_new import AdvancedSentimentAnalyzer
                analyzer = AdvancedSentimentAnalyzer()
                news_df = analyzer.fetch_and_analyze(stock_id)
                if not news_df.empty:
                    news_df['date_str'] = news_df['date_obj'].dt.strftime('%Y-%m-%d')
                    daily_sentiment = news_df.groupby('date_str')['加權分數'].mean().to_dict()
                    df['Sentiment_Score'] = df['date'].map(daily_sentiment).fillna(0)
            else:
                fetcher = DataFetcher()
                news_df = fetcher.fetch_stock_news(stock_id, start_date=df['date'].min())
                if not news_df.empty:
                    from analysis import SentimentAnalyzer
                    _, results = SentimentAnalyzer.analyze_sentiment(news_df.to_dict('records'), is_us=True)
                    news_sentiments = {r['date'][:10]: r['score'] for r in results}
                    df['Sentiment_Score'] = df['date'].map(news_sentiments).fillna(0)
        except Exception as e:
            pass # 略過錯誤印出保持乾淨

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

                adr_mapping = {"2330": {"symbol": "TSM", "ratio": 5}, "2303": {"symbol": "UMC", "ratio": 5}, "3711": {"symbol": "ASX", "ratio": 2}}
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
                        converted_price = (merged_adr['ADR_Close_prev'] * merged_adr['USDTWD_prev']) / adr_info['ratio']
                        df['ADR_Premium'] = (converted_price / merged_adr['Close'] - 1) * 100
            except Exception as e:
                pass

        for col in ['Nasdaq_Ret', 'SOX_Ret', 'VIX_Ret', 'ADR_Premium', 'Volatility']:
            if col not in df.columns:
                df[col] = 0.0
            else:
                df[col] = df[col].ffill().fillna(0)

        # 刪除 NaN (包含因為 Target_Ret 而產生的最後 3 筆缺失值)
        return df.dropna()

    def process_split_data(self, df_train, df_test, stock_id, is_us=False):
        df_train = self.add_indicators(df_train, stock_id, is_us)
        df_test = self.add_indicators(df_test, stock_id, is_us)

        for col in self.feature_cols:
            if col not in df_train.columns: df_train[col] = 0
            if col not in df_test.columns: df_test[col] = 0

        train_f = df_train[self.feature_cols].values
        test_f = df_test[self.feature_cols].values

        # 訓練與測試的目標皆使用 Target_Ret (未來3天)
        train_t = df_train[['Target_Ret']].values
        test_t = df_test[['Target_Ret']].values

        # 保留真實的次日報酬，供測試集計算每日資金曲線
        raw_ret_test = df_test[['Next_Ret']].values

        scaled_train_f = self.feature_scaler.fit_transform(train_f)
        scaled_test_f = self.feature_scaler.transform(test_f)

        X_train, y_train, _ = self._create_windows(scaled_train_f, train_t)
        X_test, y_test, ret_test_aligned = self._create_windows(scaled_test_f, test_t, raw_ret_test)

        return torch.FloatTensor(X_train), torch.LongTensor(y_train), \
               torch.FloatTensor(X_test), torch.LongTensor(y_test), ret_test_aligned

    def _create_windows(self, features, targets, raw_ret_test=None):
        X, y, raw_returns = [], [], []
        # 🚀 提高波段閾值：3 天內預期漲幅達 1.5% 才會觸發買點
        threshold = 0.015

        for i in range(len(features) - self.window_size):
            X.append(features[i:i+self.window_size])
            ret = targets[i+self.window_size][0]

            if raw_ret_test is not None:
                raw_returns.append(raw_ret_test[i+self.window_size][0])

            if ret > threshold:
                y.append(2) # 看多
            elif ret < -threshold:
                y.append(0) # 看空
            else:
                y.append(1) # 盤整

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

    if df.empty or len(df) < 100:
        print(f"❌ 錯誤: 資料量不足。")
        return

    split_idx = int(len(df) * 0.8)
    df_train, df_test = df.iloc[:split_idx], df.iloc[split_idx:]

    processor = StockDataProcessor(window_size=15)
    X_train, y_train, X_test, y_test, raw_ret_test = processor.process_split_data(df_train, df_test, stock_id, is_us)

    train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=32, shuffle=True)
    model = ClassificationAttentionLSTM(input_size=X_train.shape[2], output_size=3).to(device)

    # 處理類別不平衡
    class_counts = np.bincount(y_train.numpy(), minlength=3)
    weights = len(y_train) / (3.0 * (class_counts + 1e-5))
    class_weights = torch.FloatTensor(weights).to(device)

    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=50)

    print(f"🚀 【波段信心型】 Attention-LSTM 啟動 (標的: {stock_id})")
    print(f"🔹 標籤分佈 -> 空: {class_counts[0]}, 盤整: {class_counts[1]}, 多: {class_counts[2]}")

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
            acc = 100 * correct / total
            print(f'Epoch [{epoch+1}/{epochs}], Loss: {epoch_loss/len(train_loader):.4f}, Train Acc: {acc:.2f}%')

    # -----------------------------------------
    # 產出策略訊號 (導入 Softmax 信心濾網)
    # -----------------------------------------
    model.eval()
    with torch.no_grad():
        outputs = model(X_test.to(device))

        # 將 Logits 轉換為機率 (0% ~ 100%)
        probs = F.softmax(outputs, dim=1).cpu().numpy()
        signals = np.zeros(len(probs))

        # 🚀 關鍵：45% 信心濾網，沒有把握就強制空手
        conf_threshold = 0.45

        for i in range(len(probs)):
            if probs[i, 2] > conf_threshold:
                signals[i] = 1.0  # 高度確信波段做多
            elif probs[i, 0] > conf_threshold:
                signals[i] = -1.0 # 高度確信波段做空
            else:
                signals[i] = 0.0  # 盤整或信心不足

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

        print(f"\n📈 --- AI 信心波段決策報告 ---")
        print(f"總交易天數: {len(raw_ret_test)} 天")
        print(f"高確信做多: {np.sum(signals == 1)} 天 | 高確信做空: {np.sum(signals == -1)} 天 | 觀望空手: {np.sum(signals == 0)} 天")
        print(f"AI 高確信交易勝率: {strategy_win_rate:.2f}%")
        print(f"大盤買進持有最終報酬: {cumulative_market[-1]:.2f}%")
        print(f"AI 波段決策最終報酬: {cumulative_strategy[-1]:.2f}%")

    # -----------------------------------------
    # 繪製純 AI 決策資金曲線
    # -----------------------------------------
    plt.figure(figsize=(14, 8))
    plt.plot(cumulative_market, label=f'{stock_id} 買進持有', color='#bdc3c7', lw=2, alpha=0.8)
    plt.plot(cumulative_strategy, label='AI 波段信心濾網報酬', color='#d93025', lw=2.5)

    buy_idx = np.where(signals == 1.0)[0]
    short_idx = np.where(signals == -1.0)[0]

    plt.scatter(buy_idx, cumulative_strategy[buy_idx], marker='^', color='#27ae60', s=70, label='強烈看多', zorder=5)
    plt.scatter(short_idx, cumulative_strategy[short_idx], marker='v', color='#8e44ad', s=70, label='強烈看空', zorder=5)

    plt.title(f'{stock_id} 機器學習原生策略：AI 信心波段過濾', fontsize=16, fontweight='bold')
    plt.xlabel('測試期天數')
    plt.ylabel('累積資金成長率 (%)')
    plt.axhline(100, color='black', linestyle='--', alpha=0.3)
    plt.legend(loc='upper left')
    plt.grid(True, linestyle=':', alpha=0.6)

    save_path = f"ai_research/{stock_id}_classification_equity.png"
    plt.savefig(save_path, dpi=300)
    plt.show()

    # ==========================================
    # 🔍 附錄：XAI 特徵重要性分析 (Permutation Importance)
    # ==========================================
    print("\n🕵️‍♂️ 啟動 XAI 特徵重要性分析 (觀察大波段驅動因子)...")

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
    plt.title(f'{stock_id} 波段預測特徵重要性 (T+3 Target)', fontsize=14, fontweight='bold')
    plt.xlabel('準確率下降幅度 (影響力)', fontsize=12)
    plt.ylabel('特徵名稱', fontsize=12)
    plt.grid(True, axis='x', linestyle='--', alpha=0.6)

    plt.tight_layout()
    save_path_imp = f"ai_research/{stock_id}_feature_importance.png"
    plt.savefig(save_path_imp, dpi=300)
    plt.show()

    print("\n📊 影響力最高的前 5 大特徵 (決定波段的關鍵)：")
    print(df_imp.tail(5).sort_values(by='Importance', ascending=False))

if __name__ == "__main__":
    train_and_predict("2330", market="TW")