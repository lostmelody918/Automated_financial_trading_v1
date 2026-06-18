import pandas as pd
import numpy as np
import os
import json
import time as time_lib
import copy
from data_engine import DayTradingDataEngine
from dotenv import load_dotenv

# 使用絕對路徑確保在不同目錄執行時都能讀取到 .env
env_path = r"F:\Gemini_CLI_Application\finance_v2\.env"
load_dotenv(dotenv_path=env_path)

def vectorized_triple_barrier(df, price_col='Close', vol_col='atr', t_horizon=10, tp_mult=2.0, sl_mult=1.5):
    # 先計算 Rolling Max/Min，再往回 Shift。
    # 這樣可以避免資料集最開頭的 t_horizon-1 筆資料變成 NaN 而被浪費
    future_max = df[price_col].rolling(window=t_horizon).max().shift(-t_horizon)
    future_min = df[price_col].rolling(window=t_horizon).min().shift(-t_horizon)

    p0 = df[price_col]
    atr = df[vol_col].replace(0, 10.0)

    hit_tp = future_max >= (p0 + tp_mult * atr)
    hit_sl = future_min <= (p0 - sl_mult * atr)
    final_p = df[price_col].shift(-t_horizon)

    labels = pd.Series(3, index=df.index)
    # 雙重觸發 (Whip-saw) 洗盤過濾
    hit_both = hit_tp & hit_sl

    # 有在單邊觸發的情況下，才給予強勢買賣訊號
    cond_strong_up = hit_tp & (~hit_both)
    cond_strong_down = hit_sl & (~hit_both)

    cond_med_up = (~hit_tp) & (~hit_sl) & (final_p > p0 + 0.5 * atr)
    cond_med_down = (~hit_tp) & (~hit_sl) & (final_p < p0 - 0.5 * atr)
    cond_weak_up = (~hit_tp) & (~hit_sl) & (final_p > p0) & (final_p <= p0 + 0.5 * atr)
    cond_weak_down = (~hit_tp) & (~hit_sl) & (final_p < p0) & (final_p >= p0 - 0.5 * atr)

    labels[cond_strong_up] = 6
    labels[cond_strong_down] = 0
    labels[cond_med_up] = 5
    labels[cond_med_down] = 1
    labels[cond_weak_up] = 4
    labels[cond_weak_down] = 2

    # 切斷隔夜未來函數
    if 'date' in df.columns:
        # 檢查當前 K 線與未來第 10 根 K 線是否在「同一天」
        is_same_day = df['date'].dt.date == df['date'].shift(-t_horizon).dt.date
        # 若跨日，強制設為 NaN 丟棄，絕不讓 AI 偷看明天開盤
        labels[~is_same_day] = np.nan
    else:
        labels.iloc[-t_horizon:] = np.nan

    # 如果 final_p 是 NaN，表示因為向未來看而超出了數據範圍
    labels[final_p.isna()] = np.nan
    return labels

def prepare_training_data():
    """集中處理資料抓取與預處理，避免重複抓取"""
    engine = DayTradingDataEngine()
    print("📥 正在獲取歷史數據與籌碼...")

    # 恢復為 730 天
    df_raw = engine.fetch_intraday_data(days=730)
    if df_raw.empty:
        print("❌ 無法取得歷史行情數據，訓練終止")
        return pd.DataFrame()
    print(f"✅ 取得行情數據: {df_raw.shape}")

    df_real_chips = engine.fetch_real_historical_chips(days=730)
    if df_real_chips.empty:
        print("⚠️ 無法取得籌碼數據，將僅使用技術指標")
    else:
        print(f"✅ 取得籌碼數據: {df_real_chips.shape}")

    print("🧩 執行籌碼融合與標籤生成...")
    df = engine.integrate_institutional_chips(df_raw, df_real_chips)
    print(f"📊 融合後數據量: {df.shape}")

    df['label'] = vectorized_triple_barrier(df)
    before_drop = len(df)
    df.dropna(subset=['label'], inplace=True)
    after_drop = len(df)
    print(f"🏷️ 標籤生成完成，過濾掉 {before_drop - after_drop} 筆無效末端數據，剩餘 {after_drop} 筆")

    if not df.empty:
        df['label'] = df['label'].astype(int)
    return df

def train_one_run(df, config):
    """
    核心訓練邏輯 (延遲匯入 torch 以避免與 Shioaji 衝突)
    """
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import TensorDataset, DataLoader
    from torch.utils.tensorboard import SummaryWriter
    from composite_ai import MultiTimeframeCompositeAI
    from model_manager import TradingModelManager
    import wandb

    if df.empty:
        print("❌ 傳入的 DataFrame 為空，取消訓練")
        return None

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"💻 使用設備: {device}")

    run_name = f"mtf_trading_{time_lib.strftime('%Y%m%d-%H%M%S')}"
    writer = SummaryWriter(log_dir=f"runs/{run_name}")

    use_wandb = wandb.run is not None

    # 特徵工程與歸一化
    absolute_cols = ['Open', 'High', 'Low', 'Close', 'vwap', 'bb_upper', 'bb_lower', 'Volume']
    exclude_cols = ['date', 'time', 'date_only', 'day_of_week', 'label'] + absolute_cols
    feature_cols = [c for c in df.columns if c not in exclude_cols and not c.startswith('future_')]

    print(f"🔍 特徵欄位數量: {len(feature_cols)}")

    df_feat = df[feature_cols].copy()
    df_numeric = df_feat.select_dtypes(include=[np.number])

    train_split_idx = int(len(df_numeric) * 0.8)
    if train_split_idx < 100:
        print(f"❌ 數據量太少 ({len(df_numeric)})，不足以進行訓練")
        return None

    # --- 數據歸一化 (Normalization) ---
    # 嚴格守則：僅使用訓練集計算統計量 (Median/IQR)，杜絕未來數據洩漏到過去
    median = df_numeric.iloc[:train_split_idx].median()
    iqr = (df_numeric.iloc[:train_split_idx].quantile(0.75) - df_numeric.iloc[:train_split_idx].quantile(0.25)).replace(0, 1.0)

    df_norm = ((df_numeric - median) / iqr).fillna(0)
    input_dim = df_norm.shape[1]

    # MTF 數據準備
    print("📊 準備多時間尺度 (MTF) 張量與驗證集劃分...")
    df_1m = df_norm.copy()
    df_1m['date'] = df['date']

    df_15m_historical = df_1m.set_index('date').resample('15min', closed='right', label='right').last().dropna()

    data_1m = df_1m.drop(columns='date').values
    data_15m_hist = df_15m_historical.values
    label_values = df['label'].values

    print("⚡ 正在執行時序索引優化...")
    indices_15m = df_15m_historical.index.get_indexer(df_1m['date'], method='pad')

    X_1m, X_15m, Y = [], [], []
    win_1m, win_15m = 40, 20

    print("🛠️ 正在合成時序張量 (這可能需要一點時間)...")
    last_p = -1
    for i in range(win_1m - 1, len(df_1m)):
        p = int((i / len(df_1m)) * 10) * 10
        if p != last_p:
            print(f"   ⌛ 已完成 {p}%")
            last_p = p

        curr_time = df_1m['date'].iloc[i]
        idx_15m = indices_15m[i]

        if idx_15m >= win_15m - 1:
            X_1m.append(data_1m[i - win_1m + 1 : i + 1])

            # 修正：使用 idx_15m + 1 以包含距離當下最近的那根 (已完成) 15 分鐘 K 線 (避免切片排除最後一個元素)
            # 策略：取 win_15m - 1 (19) 根歷史 15m K線 + 1 根當前 1m 狀態，湊齊 win_15m (20)
            hist_15m = data_15m_hist[idx_15m - (win_15m - 2) : idx_15m + 1]
            current_1m_state = data_1m[i].reshape(1, -1)

            if len(hist_15m) > 0:
                seq_15m = np.concatenate([hist_15m, current_1m_state], axis=0)
            else:
                seq_15m = np.tile(current_1m_state, (win_15m, 1))

            X_15m.append(seq_15m[-win_15m:])
            Y.append(label_values[i])

    if not X_1m:
        print("❌ 合成後張量列表為空")
        return None

    X_1m = np.array(X_1m)
    X_15m = np.array(X_15m)
    Y = np.array(Y)

    # 嚴格序列切分 (Sequential Split) 並加入「隔離帶 (Gap)」，防止滑動視窗造成的數據重疊洩漏
    total_samples = len(X_1m)
    split_idx = int(total_samples * 0.8)
    # 隔離帶長度設定為 1 分鐘視窗長度 (40)，確保訓練集最後一筆與驗證集第一筆完全無重疊 raw data
    gap = win_1m

    X1_train = torch.tensor(X_1m[:split_idx], dtype=torch.float32)
    X15_train = torch.tensor(X_15m[:split_idx], dtype=torch.float32)
    Y_train = torch.tensor(Y[:split_idx], dtype=torch.long)

    X1_val = torch.tensor(X_1m[split_idx + gap :], dtype=torch.float32)
    X15_val = torch.tensor(X_15m[split_idx + gap :], dtype=torch.float32)
    Y_val = torch.tensor(Y[split_idx + gap :], dtype=torch.long)

    print(f"✅ 數據就緒 (已排除 {gap} 筆重疊樣本): Train={len(X1_train)}, Val={len(X1_val)}")

    model = MultiTimeframeCompositeAI(input_dim=input_dim, d_model=config.d_model, nhead=config.nhead, num_layers=config.num_layers)
    model = model.to(device)

    optimizer = optim.AdamW(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)

    warmup_scheduler = optim.lr_scheduler.LinearLR(optimizer, start_factor=0.1, total_iters=5)
    cosine_scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2, eta_min=1e-6)
    scheduler = optim.lr_scheduler.SequentialLR(optimizer, schedulers=[warmup_scheduler, cosine_scheduler], milestones=[5])

    class_counts = np.bincount(df['label'].iloc[:train_split_idx].dropna().astype(int), minlength=7)
    class_counts = class_counts + 1.0
    weights = 1.0 / np.sqrt(class_counts)
    weights = weights / weights.sum() * len(weights)
    alpha_weights = torch.tensor(weights, dtype=torch.float32).to(device)

    def focal_loss(inputs, targets, alpha, gamma=1.5):
        ce_loss = nn.CrossEntropyLoss(weight=alpha, reduction='none')(inputs, targets)
        pt = torch.exp(-ce_loss)
        return (((1 - pt) ** gamma) * ce_loss).mean()

    train_loader = DataLoader(TensorDataset(X1_train, X15_train, Y_train), batch_size=config.batch_size, shuffle=True, pin_memory=True)
    val_loader = DataLoader(TensorDataset(X1_val, X15_val, Y_val), batch_size=config.batch_size, shuffle=False)

    scaler = torch.amp.GradScaler('cuda') if device.type == 'cuda' else None

    best_val_loss = float('inf')
    best_model_state = None
    early_stop_patience = 50
    patience_counter = 0

    print(f"🚀 開始訓練...")
    for epoch in range(config.epochs):
        model.train()
        train_loss = 0
        for b_1m, b_15m, b_y in train_loader:
            b_1m, b_15m, b_y = b_1m.to(device), b_15m.to(device), b_y.to(device)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast('cuda', enabled=(scaler is not None)):
                outputs = model(b_1m, b_15m)
                loss = focal_loss(outputs, b_y, alpha=alpha_weights)
            if scaler:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()
            train_loss += loss.item()

        avg_train_loss = train_loss / len(train_loader)

        model.eval()
        val_loss = 0
        with torch.no_grad():
            for b_1m, b_15m, b_y in val_loader:
                b_1m, b_15m, b_y = b_1m.to(device), b_15m.to(device), b_y.to(device)
                outputs = model(b_1m, b_15m)
                loss = focal_loss(outputs, b_y, alpha=alpha_weights)
                val_loss += loss.item()

        avg_val_loss = val_loss / len(val_loader)
        scheduler.step()

        writer.add_scalars('Loss', {'train': avg_train_loss, 'val': avg_val_loss}, epoch)
        if use_wandb: wandb.log({"train_loss": avg_train_loss, "val_loss": avg_val_loss, "lr": optimizer.param_groups[0]['lr'], "epoch": epoch})

        if (epoch+1) % 10 == 0:
            print(f"🔥 Epoch [{epoch+1}/{config.epochs}], Train Loss: {avg_train_loss:.6f}, Val Loss: {avg_val_loss:.6f}")

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            best_model_state = copy.deepcopy(model.state_dict())
        else:
            patience_counter += 1
            if patience_counter >= early_stop_patience:
                print(f"🛑 Early Stopping at Epoch {epoch+1}")
                break

    if best_model_state is not None:
        print(f"💾 儲存最佳模型 (Val Loss: {best_val_loss:.6f})...")
        model.load_state_dict(best_model_state)
        manager = TradingModelManager(model_dir=os.path.join(os.path.dirname(__file__), "saved_models"))

        # 修正：確保 config 被乾淨地轉換為 JSON 可序列化的字典
        try:
            if use_wandb:
                # wandb.config 需要特殊的處理來提取純字典
                config_dict = {k: v for k, v in wandb.config.items() if not callable(v)}
            else:
                config_dict = {k: v for k, v in config.__dict__.items() if not k.startswith('_')}
        except:
            config_dict = {"lr": 0.0005, "note": "config serialization failed"}

        manager.save_model(model, optimizer, {"val_loss": best_val_loss}, {"config": config_dict})

    norm_params = {"mean": median.to_dict(), "std": iqr.to_dict(), "feature_cols": feature_cols}
    with open(os.path.join(os.path.dirname(__file__), "saved_models", "norm_params.json"), "w", encoding='utf-8') as f:
        json.dump(norm_params, f)

    writer.close()
    print("✅ 訓練完成！")
    return best_val_loss

class DefaultConfig:
    def __init__(self):
        self.lr = 0.0005
        self.d_model = 256
        self.nhead = 8
        self.num_layers = 3
        self.weight_decay = 0.01
        self.batch_size = 512
        self.epochs = 250

if __name__ == "__main__":
    import traceback
    import wandb
    try:
        # 1. 抓取資料 (Shioaji 先行，避免與 torch 衝突)
        final_df = prepare_training_data()

        if final_df.empty:
            print("❌ 資料集為空，終止訓練。")
            exit(1)

        # 2. 只有資料抓完後，才開始處理 torch/wandb
        import torch
        print(f"PyTorch version: {torch.__version__} | CUDA: {torch.cuda.is_available()}")

        api_key = os.getenv("WANDB_API_KEY")
        login_success = False
        if api_key:
            try:
                wandb.login(key=api_key)
                login_success = True
            except: pass

        if login_success:
            wandb.init(project="trading_mtf_single", config=DefaultConfig().__dict__)
            train_one_run(final_df, wandb.config)
        else:
            train_one_run(final_df, DefaultConfig())

    except BaseException as be:
        print(f"💥 發生嚴重錯誤: {type(be).__name__}: {be}")
        traceback.print_exc()
        exit(1)
