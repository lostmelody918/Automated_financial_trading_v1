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

# 1. 設置專案路徑 (解決從不同資料夾執行的路徑問題)
root_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if root_path not in sys.path:
    sys.path.insert(0, root_path)

try:
    from database import DatabaseManager
    from analysis import KLineAnalyzer
    from data_fetcher import DataFetcher
except ImportError:
    print("❌ 警告：找不到必要的模組，請確認路徑。")

# 設定運算裝置
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---
# 2. Attention 模組群
# ---

# 🌟 模組 A：特徵注意力機制 (Feature Attention) -> 解決不同股票特徵權重不同的問題
class FeatureAttention(nn.Module):
    def __init__(self, input_size):
        super(FeatureAttention, self).__init__()
        # 學習每個特徵的權重 (0 ~ 1 之間)，用來放大重要特徵、靜音干擾特徵
        self.feature_weights = nn.Sequential(
            nn.Linear(input_size, input_size),
            nn.Sigmoid()
        )

    def forward(self, x):
        # x shape: (batch, seq_len, input_features)
        weights = self.feature_weights(x)
        return x * weights, weights

# 🌟 模組 B：時間步注意力機制 (Time-Step Contextual Attention)
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
# 3. 分類型 Attention-LSTM 模型架構 (雙重 Attention 決策版)
# ---
class ClassificationAttentionLSTM(nn.Module):
    # ⚠️ 輸出層變為 3 (類別 0:做空, 類別 1:空手, 類別 2:做多)
    def __init__(self, input_size, hidden_size=64, num_layers=2, output_size=3):
        super(ClassificationAttentionLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # 裝載 Feature Attention 於最前端
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
        # 1. 特徵動態靜音/放大
        weighted_x, _ = self.feature_attention(x)
        # 2. 特徵映射
        emb_x = self.embedding(weighted_x)

        # 3. 時序記憶處理
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)
        lstm_out, _ = self.lstm(emb_x, (h0, c0))

        # 4. 歷史情境回溯 (Time Attention)
        last_hidden = lstm_out[:, -1, :]
        context = self.time_attention(lstm_out, last_hidden)

        # 殘差結合並輸出分類 Logits
        combined = torch.cat([context, last_hidden], dim=1)
        out = self.ln(combined)
        return self.fc(out)

# ---
# 4. 數據處理器 (標籤工程重構為多分類任務)
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
        df['Log_Ret'] = np.log(df['Close'] / df['Close'].shift(1))

        if 'MACD' not in df.columns:
            macd = ta.macd(df['Close'])
            df['MACD'] = macd['MACD_12_26_9'] if macd is not None else 0

        # 情緒特徵
        df['Sentiment_Score'] = 0.0
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
            print(f"Sentiment update failed: {e}")

        # 外部大盤與籌碼特徵
        df['ADR_Premium'] = 0.0
        df['VIX_Ret'] = 0.0

        if not is_us:
            try:
                print(f"📥 正在提取美股大盤連動特徵與 ADR 溢價率...")
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
                print(f"⚠️ 美股特徵提取失敗: {e}")
        else:
            try:
                fetcher = DataFetcher()
                vix = fetcher.fetch_us_stock_daily("^VIX", df['date'].min(), df['date'].max())
                if not vix.empty:
                    vix['date'] = pd.to_datetime(vix['date']).dt.strftime('%Y-%m-%d')
                    vix['VIX_Ret'] = np.log(vix['Close'] / vix['Close'].shift(1))
                    df = pd.merge(df, vix[['date', 'VIX_Ret']], on='date', how='left')
            except Exception as e:
                print(f"⚠️ VIX 提取失敗: {e}")

        for col in ['Nasdaq_Ret', 'SOX_Ret', 'VIX_Ret', 'ADR_Premium', 'Volatility']:
            if col not in df.columns:
                df[col] = 0.0
            else:
                df[col] = df[col].ffill().fillna(0)

        return df.dropna()

    def process_split_data(self, df_train, df_test, stock_id, is_us=False):
        df_train = self.add_indicators(df_train, stock_id, is_us)
        df_test = self.add_indicators(df_test, stock_id, is_us)

        for col in self.feature_cols:
            if col not in df_train.columns: df_train[col] = 0
            if col not in df_test.columns: df_test[col] = 0

        train_f = df_train[self.feature_cols].values
        test_f = df_test[self.feature_cols].values

        # Target 保留原始收益率供分類轉換
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
        # 🚀 分類閾值：判斷足以覆蓋手續費並有賺頭的波動
        threshold = 0.003

        for i in range(len(features) - self.window_size):
            X.append(features[i:i+self.window_size])
            ret = targets[i+self.window_size][0]
            raw_returns.append(ret)

            # 將數值轉換為 3 個決策類別
            if ret > threshold:
                y.append(2) # 類別 2: 看多
            elif ret < -threshold:
                y.append(0) # 類別 0: 看空
            else:
                y.append(1) # 類別 1: 盤整避險

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

    # 實例化雙重 Attention 模型
    model = ClassificationAttentionLSTM(input_size=X_train.shape[2], output_size=3).to(device)

    # 🚀 計算類別權重 (解決股市大部分時間都在盤整的樣本不均問題)
    class_counts = np.bincount(y_train.numpy(), minlength=3)
    weights = len(y_train) / (3.0 * (class_counts + 1e-5))
    class_weights = torch.FloatTensor(weights).to(device)

    # 使用 CrossEntropyLoss 取代數值迴歸的 Loss
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=50)

    print(f"🚀 【決策分類型】 雙重 Attention-LSTM 啟動 (標的: {stock_id})")
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

            _, predicted = torch.max(outputs.data, 1)
            total += batch_y.size(0)
            correct += (predicted == batch_y).sum().item()

        scheduler.step()
        if (epoch+1) % 25 == 0:
            acc = 100 * correct / total
            print(f'Epoch [{epoch+1}/{epochs}], Loss: {epoch_loss/len(train_loader):.4f}, Train Acc: {acc:.2f}%')

    # -----------------------------------------
    # 讓模型產出策略訊號
    # -----------------------------------------
    model.eval()
    with torch.no_grad():
        outputs = model(X_test.to(device))
        _, predicted_classes = torch.max(outputs.data, 1)
        predicted_classes = predicted_classes.cpu().numpy()

        signals = np.zeros(len(predicted_classes))
        signals[predicted_classes == 2] = 1.0  # 看多
        signals[predicted_classes == 0] = -1.0 # 看空
        signals[predicted_classes == 1] = 0.0  # 盤整

        strategy_returns = signals * raw_ret_test
        cumulative_market = np.exp(np.cumsum(raw_ret_test)) * 100
        cumulative_strategy = np.exp(np.cumsum(strategy_returns)) * 100

        active_days = (signals != 0)
        if np.sum(active_days) > 0:
            correct_trades = ((signals[active_days] > 0) & (raw_ret_test[active_days] > 0)) | \
                             ((signals[active_days] < 0) & (raw_ret_test[active_days] < 0))
            strategy_win_rate = (correct_trades.sum() / np.sum(active_days)) * 100
        else:
            strategy_win_rate = 0.0

        print(f"\n📈 --- AI 純粹學習決策報告 ---")
        print(f"總交易天數: {len(raw_ret_test)} 天")
        print(f"做多: {np.sum(signals == 1)} 天 | 做空: {np.sum(signals == -1)} 天 | 空手: {np.sum(signals == 0)} 天")
        print(f"純模型決策勝率: {strategy_win_rate:.2f}%")
        print(f"大盤買進持有最終報酬: {cumulative_market[-1]:.2f}%")
        print(f"AI 裸機決策最終報酬: {cumulative_strategy[-1]:.2f}%")

    # -----------------------------------------
    # 繪製資金曲線
    # -----------------------------------------
    plt.figure(figsize=(14, 8))
    plt.plot(cumulative_market, label=f'{stock_id} 買進持有', color='#bdc3c7', lw=2, alpha=0.8)
    plt.plot(cumulative_strategy, label='AI 決策分類器報酬', color='#d93025', lw=2.5)

    buy_idx = np.where(signals == 1.0)[0]
    short_idx = np.where(signals == -1.0)[0]

    plt.scatter(buy_idx, cumulative_strategy[buy_idx], marker='^', color='#27ae60', s=50, label='AI 買進', zorder=5)
    plt.scatter(short_idx, cumulative_strategy[short_idx], marker='v', color='#8e44ad', s=50, label='AI 放空', zorder=5)

    plt.title(f'{stock_id} 機器學習原生策略：雙重 Attention 決策模型', fontsize=16, fontweight='bold')
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
    print("\n🕵️‍♂️ 啟動 XAI 特徵重要性分析 (這可能需要幾秒鐘)...")

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

        drop_in_acc = base_acc - shuffled_acc
        importances[feature_name] = drop_in_acc

    df_imp = pd.DataFrame.from_dict(importances, orient='index', columns=['Importance'])
    df_imp = df_imp.sort_values(by='Importance', ascending=True)

    plt.figure(figsize=(10, 8))
    colors = ['#27ae60' if val > 0 else '#e74c3c' for val in df_imp['Importance']]
    df_imp['Importance'].plot(kind='barh', color=colors)
    plt.title(f'{stock_id} LSTM 模型特徵重要性 (Permutation Importance)', fontsize=14, fontweight='bold')
    plt.xlabel('準確率下降幅度 (Drop in Accuracy)', fontsize=12)
    plt.ylabel('特徵名稱', fontsize=12)
    plt.grid(True, axis='x', linestyle='--', alpha=0.6)

    plt.tight_layout()
    save_path_imp = f"ai_research/{stock_id}_feature_importance.png"
    plt.savefig(save_path_imp, dpi=300)
    plt.show()

    print("\n📊 影響力最高的前 5 大特徵：")
    print(df_imp.tail(5).sort_values(by='Importance', ascending=False))

if __name__ == "__main__":
    train_and_predict("2330", market="TW")