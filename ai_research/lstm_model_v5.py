import os
from platform import processor
from pyexpat import model
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
# 2. 損失函數：Directional L1 Loss
# ---
class DirectionalL1Loss(nn.Module):
    """
    使用 L1 Loss 取代 Huber/MSE，打破模型「預測平均值(0)」的保守傾向。
    並加入適度的方向懲罰，引導模型捕捉轉折。
    """
    def __init__(self, alpha=0.5): # 降低 alpha，減輕模型的「恐懼感」
        super(DirectionalL1Loss, self).__init__()
        self.alpha = alpha
        self.l1 = nn.L1Loss()

    def forward(self, pred, target):
        loss_l1 = self.l1(pred, target)
        # 方向懲罰：猜錯方向時 (一正一負)，給予懲罰
        penalty = torch.mean(F.relu(-pred * target))
        return loss_l1 + self.alpha * penalty

# ---
# 3. Attention 模組
# ---
class Attention(nn.Module):
    def __init__(self, hidden_size):
        super(Attention, self).__init__()
        self.attn = nn.Linear(hidden_size, 1)

    def forward(self, lstm_output):
        weights = F.softmax(self.attn(lstm_output), dim=1)
        context = torch.sum(weights * lstm_output, dim=1)
        return context

# ---
# 4. Attention-LSTM 模型架構 (急救版：LeakyReLU + 降維)
# ---
class AttentionLSTM(nn.Module):
    # 降低隱藏層維度至 64，減少模型負擔與過擬合風險
    def __init__(self, input_size, hidden_size=64, num_layers=2, output_size=1):
        super(AttentionLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=0.2)
        self.attention = Attention(hidden_size)

        # 保持 LayerNorm，防止均值坍塌
        self.ln = nn.LayerNorm(hidden_size)

        self.fc = nn.Sequential(
            nn.Linear(hidden_size, 32),
            nn.LeakyReLU(0.1), # 🚀 關鍵：取代 ReLU，防止神經元死亡 (Dying ReLU)
            nn.Dropout(0.2),
            nn.Linear(32, output_size)
        )

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)

        lstm_out, _ = self.lstm(x, (h0, c0))
        context = self.attention(lstm_out)

        out = self.ln(context)
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
    model = AttentionLSTM(input_size=X_train.shape[2]).to(device)

    # 🚀 降低 alpha 值，允許模型自由探索
    criterion = DirectionalL1Loss(alpha=0.5)
    # 取消 weight_decay，使用標準 Adam
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=50)

    print(f"🚀 量化策略 Attention-LSTM 啟動 (標的: {stock_id}, 市場: {market})")
    print(f"🔹 輸入維度: {X_train.shape[2]}, 特徵: {processor.feature_cols}")

    epochs = 325
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

    # -----------------------------------------
    # 預測與量化交易效能評估
    # -----------------------------------------
    model.eval()
    with torch.no_grad():
        preds_scaled = model(X_test.to(device)).cpu().numpy()
        preds_ret = processor.target_scaler.inverse_transform(preds_scaled).flatten()
        actual_ret = processor.target_scaler.inverse_transform(y_test.numpy()).flatten()

        # 提取多維度歷史數據
        actual_close_prev = processed_test_df['Close'].values[processor.window_size-1:-1]
        ma20_prev = processed_test_df['MA20'].values[processor.window_size-1:-1]
        foreign_prev = processed_test_df['Foreign'].values[processor.window_size-1:-1]
        rsi_prev = processed_test_df['RSI'].values[processor.window_size-1:-1]
        macd_prev = processed_test_df['MACD'].values[processor.window_size-1:-1]

        # 提取前一個波段的高點 (近 15 天最高價)
        rolling_high_prev = processed_test_df['High'].rolling(window=15, min_periods=1).max().values[processor.window_size-1:-1]

        # 1. 訊號平滑化
        smoothed_preds = pd.Series(preds_ret).rolling(window=3, min_periods=1).mean().values
        foreign_ma3 = pd.Series(foreign_prev).rolling(window=3, min_periods=1).mean().values

        print("\n🔍 --- 模型輸出診斷 ---")
        print(f"平滑預測 最大值: {smoothed_preds.max():.6f} | 最小值: {smoothed_preds.min():.6f}")

        # 2. 動態閾值 (稍微放寬做空閾值，增加攻擊頻率)
        pos_preds = smoothed_preds[smoothed_preds > 0]
        neg_preds = smoothed_preds[smoothed_preds < 0]
        buy_threshold = np.percentile(pos_preds, 50) if len(pos_preds) > 0 else 0.001
        sell_threshold = np.percentile(neg_preds, 40) if len(neg_preds) > 0 else -0.001

        # 3. 核心策略邏輯
        signals = np.zeros(len(smoothed_preds))

        for i in range(len(smoothed_preds)):
            is_bull_trend = actual_close_prev[i] > ma20_prev[i]
            is_foreign_selling = foreign_ma3[i] < 0
            is_foreign_buying = foreign_ma3[i] > 0
            is_strong_breakout = is_bull_trend and (rsi_prev[i] >= 60) and (macd_prev[i] > 0)

            # 🚀 計算當前價格距離波段高點的跌幅
            wave_drop_pct = (actual_close_prev[i] - rolling_high_prev[i]) / rolling_high_prev[i]

            # 🚀 新增：高檔頭部成型 (從高點回落 > 4%，視為提早轉弱的做空訊號)
            is_peak_reversal = wave_drop_pct < -0.04
            # 🚀 修正：放寬超跌閥值到 -15%，讓做空波段可以一路吃到飽
            is_oversold = wave_drop_pct < -0.15

            # 情況 A：AI 看多
            if smoothed_preds[i] > buy_threshold:
                if is_bull_trend or is_foreign_buying:
                    signals[i] = 1.0
                else:
                    signals[i] = 0.5

            # 情況 B：AI 看空
            elif smoothed_preds[i] < sell_threshold:
                if is_foreign_selling:
                    if is_oversold:
                        signals[i] = 0.0 # 跌幅超過 15% 魚尾太小，才停止追空
                    elif not is_bull_trend or is_peak_reversal:
                        # 💥 解除封印：即使還在月線之上，只要高檔回落超過 4% 且外資賣超，直接狙擊做空！
                        signals[i] = -1.0
                    else:
                        signals[i] = 0.0
                elif is_strong_breakout:
                    signals[i] = 1.0
                else:
                    signals[i] = 0.0

            # 情況 C：AI 覺得是雜訊
            else:
                if is_strong_breakout:
                    signals[i] = 1.0
                elif is_bull_trend:
                    signals[i] = 0.5
                else:
                    signals[i] = 0.0

        # 4. 計算量化策略的真實報酬率
        strategy_returns = signals * actual_ret
        cumulative_market = np.exp(np.cumsum(actual_ret)) * 100
        cumulative_strategy = np.exp(np.cumsum(strategy_returns)) * 100

        days_full = np.sum(signals == 1.0)
        days_half = np.sum(signals == 0.5)
        days_short = np.sum(signals == -1.0)
        days_empty = np.sum(signals == 0.0)

        active_ai_days = (signals != 0) & (signals != 0.5)
        if np.sum(active_ai_days) > 0:
            correct_dir = ((signals[active_ai_days] > 0) == (actual_ret[active_ai_days] > 0)).sum()
            dir_accuracy = (correct_dir / np.sum(active_ai_days)) * 100
        else:
            dir_accuracy = 0.0

        print(f"\n📈 --- 測試集分析報告 (提早做空增強版) ---")
        print(f"總交易天數: {len(actual_ret)} 天")
        print(f"滿倉突破: {days_full} 天 | 半倉跟隨: {days_half} 天 | 狙擊做空: {days_short} 天 | 超跌/避險空手: {days_empty} 天")
        print(f"強勢決策勝率: {dir_accuracy:.2f}%")
        print(f"大盤買進持有最終報酬: {cumulative_market[-1]:.2f}%")
        print(f"AI 量化策略最終報酬: {cumulative_strategy[-1]:.2f}%")

    # -----------------------------------------
    # 繪製量化交易資金曲線
    # -----------------------------------------
    plt.figure(figsize=(14, 8))

    plt.plot(cumulative_market, label=f'{stock_id} 買進持有 (Buy & Hold)', color='#bdc3c7', lw=2, alpha=0.8)
    plt.plot(cumulative_strategy, label='Attention-LSTM (提早狙擊版) 策略報酬', color='#d93025', lw=2.5)

    plot_days = min(150, len(actual_ret))
    recent_signals = signals[-plot_days:]
    recent_strategy = cumulative_strategy[-plot_days:]

    signal_diff = np.diff(recent_signals, prepend=recent_signals[0])

    buy_idx = np.where((recent_signals == 1.0) & (signal_diff > 0))[0] + (len(actual_ret) - plot_days)
    short_idx = np.where((recent_signals == -1.0) & (signal_diff < 0))[0] + (len(actual_ret) - plot_days)
    exit_idx = np.where((recent_signals == 0.0) & (signal_diff < 0))[0] + (len(actual_ret) - plot_days)

    plt.scatter(buy_idx, cumulative_strategy[buy_idx], marker='^', color='#27ae60', s=150, label='滿倉突破/加碼', zorder=5)
    plt.scatter(short_idx, cumulative_strategy[short_idx], marker='v', color='#8e44ad', s=150, label='高檔轉弱/破線放空', zorder=5)
    plt.scatter(exit_idx, cumulative_strategy[exit_idx], marker='x', color='#7f8c8d', s=100, label='超跌/清倉避險', zorder=5)

    plt.title(f'{stock_id} 自動化交易策略：高檔提早狙擊與波段獲利', fontsize=16, fontweight='bold')
    plt.xlabel('測試期天數', fontsize=12)
    plt.ylabel('累積資金成長率 (%)', fontsize=12)
    plt.axhline(100, color='black', linestyle='--', alpha=0.3)
    plt.legend(loc='upper left', fontsize=11)
    plt.grid(True, linestyle=':', alpha=0.6)

    plt.tight_layout()
    save_path = f"ai_research/{stock_id}_equity_curve_early_short.png"
    plt.savefig(save_path, dpi=300)
    plt.show()
    print(f"✅ 回測圖表已儲存至: {save_path}")

if __name__ == "__main__":
    # 範例：訓練 2330 台股
    train_and_predict("2330", market="TW")