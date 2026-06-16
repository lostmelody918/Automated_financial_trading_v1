import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
pd.options.mode.string_storage = 'python' # Disable pyarrow
import pyarrow
import numpy as np
import os
import json
import time as time_lib
import copy
from data_engine import DayTradingDataEngine
from composite_ai import MultiTimeframeCompositeAI
from model_manager import TradingModelManager
from torch.utils.tensorboard import SummaryWriter
import wandb
from dotenv import load_dotenv

def vectorized_triple_barrier(df, price_col='Close', vol_col='atr', t_horizon=10, tp_mult=2.0, sl_mult=1.5):
    future_max = df[price_col].shift(-t_horizon).rolling(window=t_horizon).max()
    future_min = df[price_col].shift(-t_horizon).rolling(window=t_horizon).min()
    p0 = df[price_col]
    atr = df[vol_col].replace(0, 10.0)
    hit_tp = future_max >= (p0 + tp_mult * atr)
    hit_sl = future_min <= (p0 - sl_mult * atr)
    final_p = df[price_col].shift(-t_horizon)
    labels = pd.Series(3, index=df.index)
    cond_strong_up = hit_tp & (~hit_sl)
    cond_strong_down = hit_sl & (~hit_tp)
    cond_med_up = (~hit_tp) & (~hit_sl) & (final_p > p0 + 0.5 * atr)
    cond_med_down = (~hit_tp) & (~hit_sl) & (final_p < p0 - 0.5 * atr)
    labels[cond_strong_up] = 6
    labels[cond_strong_down] = 0
    labels[cond_med_up] = 5
    labels[cond_med_down] = 1
    return labels

# 使用絕對路徑確保在不同目錄執行時都能讀取到 .env
env_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), ".env")
load_dotenv(dotenv_path=env_path)

def prepare_training_data():
    """集中處理資料抓取與預處理，避免重複抓取"""
    engine = DayTradingDataEngine()
    print("📥 正在獲取歷史數據與籌碼...")
    df_raw = engine.fetch_intraday_data(days=730)
    df_real_chips = engine.fetch_real_historical_chips(days=730)

    print("🧩 執行籌碼融合與標籤生成...")
    df = engine.integrate_institutional_chips(df_raw, df_real_chips)
    df['label'] = vectorized_triple_barrier(df)
    df.dropna(subset=['label'], inplace=True)
    return df

def train_one_run(df, config):
    """
    核心訓練邏輯 (含驗證集、CosineAnnealing、因果卷積)
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    run_name = f"mtf_trading_{time_lib.strftime('%Y%m%d-%H%M%S')}"
    writer = SummaryWriter(log_dir=f"runs/{run_name}")

    use_wandb = wandb.run is not None

    # 特徵工程與歸一化
    absolute_cols = ['Open', 'High', 'Low', 'Close', 'vwap', 'bb_upper', 'bb_lower', 'Volume']
    exclude_cols = ['date', 'time', 'date_only', 'day_of_week', 'label'] + absolute_cols
    feature_cols = [c for c in df.columns if c not in exclude_cols and not c.startswith('future_')]

    df_feat = df[feature_cols].copy()
    df_numeric = df_feat.select_dtypes(include=[np.number])

    # 修正：嚴格區分訓練與驗證 (80/20)
    train_split_idx = int(len(df_numeric) * 0.8)
    df_train_raw = df_numeric.iloc[:train_split_idx]

    median = df_train_raw.median()
    iqr = (df_train_raw.quantile(0.75) - df_train_raw.quantile(0.25)).replace(0, 1.0)
    df_norm = ((df_numeric - median) / iqr).fillna(0)
    input_dim = df_norm.shape[1]

    # MTF 數據準備
    print("📊 準備多時間尺度 (MTF) 張量與驗證集劃分...")
    df_1m = df_norm.copy()
    df_1m['date'] = df['date']
    df_15m_base = df_1m.set_index('date').resample('15min').last().ffill()

    data_1m = df_1m.drop(columns='date').values
    data_15m_all = df_15m_base.values
    label_values = df['label'].values

    X_1m, X_15m, Y = [], [], []
    win_1m, win_15m = 40, 20

    for i in range(win_1m, len(df_1m)):
        curr_time = df_1m['date'].iloc[i]
        idx_15m = df_15m_base.index.get_indexer([curr_time], method='pad')[0]
        if idx_15m >= win_15m:
            X_1m.append(data_1m[i-win_1m:i])
            X_15m.append(data_15m_all[idx_15m-win_15m:idx_15m])
            Y.append(label_values[i])

    X_1m = np.array(X_1m)
    X_15m = np.array(X_15m)
    Y = np.array(Y)

    # 執行訓練/驗證集物理劃分 (確保驗證集是最後 20% 的數據，避免時序洩漏)
    split = int(len(X_1m) * 0.8)
    X1_train, X1_val = torch.tensor(X_1m[:split], dtype=torch.float32), torch.tensor(X_1m[split:], dtype=torch.float32)
    X15_train, X15_val = torch.tensor(X_15m[:split], dtype=torch.float32), torch.tensor(X_15m[split:], dtype=torch.float32)
    Y_train, Y_val = torch.tensor(Y[:split], dtype=torch.long), torch.tensor(Y[split:], dtype=torch.long)

    print(f"✅ 數據就緒: Train={len(X1_train)}, Val={len(X1_val)}")

    model = MultiTimeframeCompositeAI(input_dim=input_dim, d_model=config.d_model, nhead=config.nhead, num_layers=config.num_layers)
    model = model.to(device)

    optimizer = optim.AdamW(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)

    # 修正：更換為 CosineAnnealingWarmRestarts，避免破壞初始權重並提高收斂品質
    # T_0: 週期長度, T_mult: 週期倍率
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2, eta_min=1e-6)

    # 平滑化權重 (Square Root Smoothing)
    obs_counts = df['label'].iloc[:train_split_idx].value_counts()
    full_counts = np.zeros(7)
    for lbl, cnt in obs_counts.items(): full_counts[int(lbl)] = cnt
    weights = np.zeros(7)
    mask = full_counts > 0
    weights[mask] = 1.0 / np.sqrt(full_counts[mask])
    weights = weights / weights.sum() * mask.sum()
    alpha_weights = torch.tensor(weights, dtype=torch.float32).to(device)

    def focal_loss(inputs, targets, alpha, gamma=2.0):
        ce_loss = nn.CrossEntropyLoss(weight=alpha, reduction='none')(inputs, targets)
        pt = torch.exp(-ce_loss)
        return (((1 - pt) ** gamma) * ce_loss).mean()

    from torch.utils.data import TensorDataset, DataLoader
    train_loader = DataLoader(TensorDataset(X1_train, X15_train, Y_train), batch_size=config.batch_size, shuffle=True, pin_memory=True)
    val_loader = DataLoader(TensorDataset(X1_val, X15_val, Y_val), batch_size=config.batch_size, shuffle=False)

    scaler = torch.amp.GradScaler('cuda') if device.type == 'cuda' else None

    best_val_loss = float('inf')
    best_model_state = None
    early_stop_patience = 25
    patience_counter = 0

    print(f"🚀 開始訓練 (含因果卷積與驗證集)...")
    for epoch in range(config.epochs):
        # --- Training Phase ---
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

        # --- Validation Phase ---
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for b_1m, b_15m, b_y in val_loader:
                b_1m, b_15m, b_y = b_1m.to(device), b_15m.to(device), b_y.to(device)
                outputs = model(b_1m, b_15m)
                loss = focal_loss(outputs, b_y, alpha=alpha_weights)
                val_loss += loss.item()

        avg_val_loss = val_loss / len(val_loader)

        # 更新學習率 (CosineAnnealing 每 epoch 更新)
        scheduler.step()

        # 日誌
        writer.add_scalars('Loss', {'train': avg_train_loss, 'val': avg_val_loss}, epoch)
        writer.add_scalar('LR', optimizer.param_groups[0]['lr'], epoch)
        if use_wandb: wandb.log({"train_loss": avg_train_loss, "val_loss": avg_val_loss, "lr": optimizer.param_groups[0]['lr'], "epoch": epoch})

        if (epoch+1) % 5 == 0:
            print(f"🔥 Epoch [{epoch+1}/{config.epochs}], Train Loss: {avg_train_loss:.6f}, Val Loss: {avg_val_loss:.6f} | LR: {optimizer.param_groups[0]['lr']:.8f}")

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            best_model_state = copy.deepcopy(model.state_dict())
        else:
            patience_counter += 1
            if patience_counter >= early_stop_patience:
                print(f"🛑 觸發 Early Stopping 於 Epoch {epoch+1}")
                break

    # --- 訓練結束，執行單次儲存 ---
    if best_model_state is not None:
        print(f"💾 訓練結束，正在儲存驗證集表現最佳模型 (Val Loss: {best_val_loss:.6f})...")
        model.load_state_dict(best_model_state)
        manager = TradingModelManager(model_dir=os.path.join(os.path.dirname(__file__), "saved_models"))
        manager.save_model(model, optimizer, {"val_loss": best_val_loss}, {"config": dict(config)})

    # 儲存歸一化參數
    norm_params = {"mean": median.to_dict(), "std": iqr.to_dict(), "feature_cols": feature_cols}
    with open(os.path.join(os.path.dirname(__file__), "saved_models", "norm_params.json"), "w", encoding='utf-8') as f:
        json.dump(norm_params, f)

    writer.close()
    print("✅ 訓練任務完成！")
    return best_val_loss

class DefaultConfig:
    def __init__(self):
        self.lr = 0.0003
        self.d_model = 256
        self.nhead = 8
        self.num_layers = 3
        self.weight_decay = 0.01
        self.batch_size = 1024
        self.epochs = 150

if __name__ == "__main__":
    # 1. 抓取資料
    try:
        final_df = prepare_training_data()
    except Exception as e:
        print(f"❌ 資料準備失敗: {e}")
        exit(1)

    # 2. WandB 登入
    api_key = os.getenv("WANDB_API_KEY")
    login_success = False
    if api_key:
        try:
            wandb.login(key=api_key)
            login_success = True
        except Exception as e:
            print(f"⚠️ WandB 登入失敗: {e}")

    # 3. 執行訓練
    try:
        if login_success:
            wandb.init(project="trading_mtf_single", config=DefaultConfig().__dict__)
            train_one_run(final_df, wandb.config)
        else:
            raise Exception("Login Failed")
    except Exception as e:
        print(f"ℹ️ 使用本地模式啟動: {e}")
        train_one_run(final_df, DefaultConfig())
