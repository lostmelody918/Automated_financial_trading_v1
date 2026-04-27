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
    from analysis import KLineAnalyzer, MomentumAnalyzer, SentimentAnalyzer
    from data_fetcher import DataFetcher
except ImportError:
    print("❌ 警告：找不到必要的模組，請確認路徑。")

# 設定運算裝置
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---
# 2. 終極版損失函數：Directional L1 Loss (打破均值回歸與滯後)
# ---
class DirectionalL1Loss(nn.Module):
    """
    使用 SmoothL1Loss 解決純 L1 在接近 0 時的梯度震盪問題。
    同時保留方向懲罰，確保抓取轉折。
    """
    def __init__(self, alpha=0.5):
        super(DirectionalL1Loss, self).__init__()
        self.alpha = alpha
        self.smooth_l1 = nn.SmoothL1Loss() # 改為SmoothL1Loss

    def forward(self, pred, target):
        loss = self.smooth_l1(pred, target)
        penalty = torch.mean(F.relu(-pred * target))
        return loss + self.alpha * penalty

# ---
# 3. Attention 模組 (Contextual Query-Key Attention)
# ---
class ContextualAttention(nn.Module):
    def __init__(self, hidden_size):
        super(ContextualAttention, self).__init__()
        # 用於轉換 Query 和 Key 的權重矩陣
        self.query_proj = nn.Linear(hidden_size, hidden_size)
        self.key_proj = nn.Linear(hidden_size, hidden_size)

    def forward(self, lstm_output, last_hidden):
        # lstm_output shape: (batch, seq_len, hidden_size)
        # last_hidden shape: (batch, hidden_size)

        # 1. 將最後一天的狀態作為 Query (擴展維度以進行矩陣乘法)
        query = self.query_proj(last_hidden).unsqueeze(1) # (batch, 1, hidden_size)

        # 2. 將所有歷史狀態作為 Keys
        keys = self.key_proj(lstm_output) # (batch, seq_len, hidden_size)

        # 3. 計算注意力分數 (Dot-Product)
        # bmm 是 batch matrix multiplication
        scores = torch.bmm(query, keys.transpose(1, 2)) # (batch, 1, seq_len)

        # 4. 歸一化權重並套用至 Values (即原本的 lstm_output)
        weights = F.softmax(scores, dim=2)
        context = torch.bmm(weights, lstm_output).squeeze(1) # (batch, hidden_size)

        return context

# ---
# 4. Attention-LSTM 模型架構 (特徵嵌入 + 殘差連接版)
# ---
class AdvancedAttentionLSTM(nn.Module):
    def __init__(self, input_size, hidden_size=64, num_layers=2, output_size=1):
        super(AdvancedAttentionLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # 🚀 升級 1：特徵嵌入層 (Feature Embedding)
        # 先讓 17 個特徵互相作用，提取更高階的特徵表達
        self.embedding = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.LeakyReLU(0.1)
        )

        # LSTM 接收的是經過嵌入的高階特徵
        self.lstm = nn.LSTM(hidden_size, hidden_size, num_layers, batch_first=True, dropout=0.2)

        # 導入 Contextual Attention
        self.attention = ContextualAttention(hidden_size)

        # 擴大 LayerNorm 維度，因為我們將使用殘差連接(Concat)
        self.ln = nn.LayerNorm(hidden_size * 2)

        self.fc = nn.Sequential(
            nn.Linear(hidden_size * 2, 32),
            nn.LeakyReLU(0.1),
            nn.Dropout(0.2),
            nn.Linear(32, output_size)
        )

    def forward(self, x):
        # x shape: (batch, seq_len, input_size)

        # 1. 特徵映射
        emb_x = self.embedding(x) # (batch, seq_len, hidden_size)

        # 2. 時序處理
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)
        lstm_out, _ = self.lstm(emb_x, (h0, c0))

        # 取出序列的最後一步狀態作為 Attention 的 Query
        last_hidden = lstm_out[:, -1, :]

        # 3. 獲取注意力上下文
        context = self.attention(lstm_out, last_hidden)

        # 殘差連接 (Residual/Concat Connection)
        # 將「過去的重點(context)」與「今天當下的狀態(last_hidden)」結合
        # 這樣模型絕對不會忘記昨天剛發生的最新資訊
        combined = torch.cat([context, last_hidden], dim=1)

        out = self.ln(combined)
        return self.fc(out)

# ---
# 5. 數據處理器
# ---
class StockDataProcessor:
    def __init__(self, window_size=15):
        self.window_size = window_size
        self.feature_scaler = StandardScaler()
        self.target_scaler = StandardScaler()
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

        # 新聞情緒
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

        # 加入美股聯動特徵、VIX 與 ADR 溢價率
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
                        # 美股昨晚影響台股今日
                        index_df[f'{label}_Ret'] = index_df[f'{label}_Ret'].shift(1)
                        df = pd.merge(df, index_df[['date', f'{label}_Ret']], on='date', how='left')

                # 計算 ADR 溢價率
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

        # 關鍵修正：確保假日缺漏值使用 ffill() 向前填充，然後再 fillna(0)
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

        train_f, train_t = df_train[self.feature_cols].values, df_train[['Log_Ret']].values
        test_f, test_t = df_test[self.feature_cols].values, df_test[['Log_Ret']].values

        scaled_train_f = self.feature_scaler.fit_transform(train_f)
        scaled_train_t = self.target_scaler.fit_transform(train_t)
        scaled_test_f = self.feature_scaler.transform(test_f)
        scaled_test_t = self.target_scaler.transform(test_t)

        X_train, y_train = self._create_windows(scaled_train_f, scaled_train_t)
        X_test, y_test = self._create_windows(scaled_test_f, scaled_test_t)

        return torch.FloatTensor(X_train), torch.FloatTensor(y_train), \
               torch.FloatTensor(X_test), torch.FloatTensor(y_test), df_test

    def _create_windows(self, features, targets):
        X, y = [], []
        for i in range(len(features) - self.window_size):
            X.append(features[i:i+self.window_size])
            y.append(targets[i+self.window_size])
        return np.array(X), np.array(y)

# ---
# 6. 主訓練流程
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
    X_train, y_train, X_test, y_test, processed_test_df = processor.process_split_data(df_train, df_test, stock_id, is_us)

    train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=32, shuffle=True)

    # 模型瘦身，防止過度擬合
    model = AdvancedAttentionLSTM(input_size=X_train.shape[2]).to(device)

    # 啟用全新 Directional L1 Loss (強烈方向懲罰)
    criterion = DirectionalL1Loss(alpha=0.5)
    # 讓 Embedding 層與 Contextual Attention 能夠自由放大特徵權重
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=50)

    print(f"🚀 防滯後多因子 Attention-LSTM 啟動 (標的: {stock_id}, 市場: {market})")
    print(f"🔹 輸入維度: {X_train.shape[2]}, 特徵: {processor.feature_cols}")

    # 訓練次數降為 300，避免死背歷史數據
    epochs = 300
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0
        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            optimizer.zero_grad()
            output = model(batch_x)
            loss = criterion(output, batch_y)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        scheduler.step()
        if (epoch+1) % 25 == 0:
            print(f'Epoch [{epoch+1}/{epochs}], Loss: {epoch_loss/len(train_loader):.6f}')

    # 預測與效能評估
    model.eval()
    with torch.no_grad():
        preds_scaled = model(X_test.to(device)).cpu().numpy()
        preds_ret = processor.target_scaler.inverse_transform(preds_scaled)
        actual_ret = processor.target_scaler.inverse_transform(y_test.numpy())

        # 🎯 計算方向預測準確率 (抓轉折點的成功率)
        correct_dir = ((preds_ret > 0) == (actual_ret > 0)).sum()
        dir_accuracy = (correct_dir / len(actual_ret)) * 100
        print(f"\n🎯 測試集方向預測準確率: {dir_accuracy:.2f}%")

        # 價格還原
        actual_close = processed_test_df['Close'].values[processor.window_size:]
        prev_close = processed_test_df['Close'].values[processor.window_size-1:-1]
        predicted_prices = prev_close * np.exp(preds_ret.flatten())

    # 繪圖
    timeframes = [15, 50, 100]
    fig, axes = plt.subplots(len(timeframes), 1, figsize=(15, 18))

    for i, days in enumerate(timeframes):
        d = min(days, len(actual_close))
        axes[i].plot(actual_close[-d:], label='實際價格', color='#1a73e8', lw=2)
        axes[i].plot(predicted_prices[-d:], label='防滯後預測 (L1 + LayerNorm)', color='#d93025', ls='--', lw=2)
        axes[i].set_title(f'最近 {d} 天預測對比 (L1 Loss + Alpha 15)', fontsize=14)
        axes[i].legend()
        axes[i].grid(True, alpha=0.3)

    plt.tight_layout()
    save_path = f"ai_research/{stock_id}_anti_lag_full_results.png"
    plt.savefig(save_path)
    print(f"✅ 訓練完成！包含轉折點評分的圖表已儲存至: {save_path}")

if __name__ == "__main__":
    # 範例：訓練 2330 台股
    train_and_predict("2330", market="TW")