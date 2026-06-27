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

def vectorized_triple_barrier(df, price_col='Close', vol_col='atr', t_horizon=5, tp_mult=3.5, sl_mult=3.5):
    # 先計算 Rolling Max/Min，再往回 Shift。
    future_max = df[price_col].rolling(window=t_horizon).max().shift(-t_horizon)
    future_min = df[price_col].rolling(window=t_horizon).min().shift(-t_horizon)

    p0 = df[price_col]
    atr = df[vol_col].replace(0, 10.0)

    # [FIX] 加入最少點數限制，避免在盤整極低波動時，輕微的 5 點跳動就被誤判為強多或強空
    dynamic_tp = np.maximum(tp_mult * atr, 30.0)
    dynamic_sl = np.maximum(sl_mult * atr, 30.0)

    hit_tp = future_max >= (p0 + dynamic_tp)
    hit_sl = future_min <= (p0 - dynamic_sl)
    final_p = df[price_col].shift(-t_horizon)

    # 【方案 A：5 分類整併】
    # 0: 強空 (Strong Down)
    # 1: 中空 (Med Down)
    # 2: 震盪 (Chop - 包含原本的弱空、盤整、弱多與雙巴)
    # 3: 中多 (Med Up)
    # 4: 強多 (Strong Up)
    
    labels = pd.Series(2, index=df.index) # 預設為 2 (震盪 Chop)
    
    # 雙重觸發 (Whip-saw) 洗盤過濾
    hit_both = hit_tp & hit_sl

    # 有在單邊觸發的情況下，才給予強勢買賣訊號
    cond_strong_up = hit_tp & (~hit_both)
    cond_strong_down = hit_sl & (~hit_both)

    # 中等強度的標準也加入最低點數限制 (至少 10 點)
    med_threshold = np.maximum(1.0 * atr, 10.0)
    
    cond_med_up = (~hit_tp) & (~hit_sl) & (final_p > p0 + med_threshold)
    cond_med_down = (~hit_tp) & (~hit_sl) & (final_p < p0 - med_threshold)

    # 其餘所有介於區間內的跳動，都會維持預設的 2 (Chop)
    labels[cond_strong_up] = 4
    labels[cond_strong_down] = 0
    labels[cond_med_up] = 3
    labels[cond_med_down] = 1

    # 切斷隔夜未來函數
    if 'date' in df.columns:
        # 檢查當前 K 線與未來第 t_horizon 根 K 線是否在「同一天」
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

    # 執行三重屏障標記
    df['label'] = vectorized_triple_barrier(df)
    before_drop = len(df)
    
    df = df.dropna(subset=['label']).copy()
    df['label'] = df['label'].astype(int)
    
    after_drop = len(df)
    print(f"🏷️ 標籤生成完成，過濾掉 {before_drop - after_drop} 筆無效末端數據，剩餘 {after_drop} 筆")

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
    from composite_ai import TriCoreMarketEngine
    from model_manager import TradingModelManager
    import wandb

    if df.empty:
        print("❌ 傳入的 DataFrame 為空，取消訓練")
        return None

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"💻 使用設備: {device}")

    run_name = f"tricore_trading_{time_lib.strftime('%Y%m%d-%H%M%S')}"
    writer = SummaryWriter(log_dir=f"runs/{run_name}")
    use_wandb = wandb.run is not None

    # 定義 Tri-Core 特徵分流
    macro_cols = ['nasdaq_prev_ret', 'nikkei_premarket_momentum', 'pc_ratio', 'us_tw_gap_divergence', 'settlement_type', 'is_settlement_day', 'dte', 'gap_amplitude', 'gap_filled', 'time_sin', 'time_cos']
    meso_cols = ['macd_hist', 'vwap_bias', 'bb_width', 'is_squeeze', 'slope_vwap', 'slope_ma20', 'dist_from_ma20', 'pullback_from_high', 'bounce_from_low', 'intraday_trend', 'spot_futures_proxy']
    micro_cols = ['ret', 'rsi', 'rsi_fast', 'body_length', 'upper_shadow', 'lower_shadow', 'body_avg_5', 'momentum_explosion', 'vol_surge_ratio', 'pv_divergence', 'noise_wavelet', 'vol_range_ratio', 'obv_bias', 'obv_slope', 'orderbook_imbalance', 'volume_delta', 'cvd_bias']

    macro_cols = [c for c in macro_cols if c in df.columns]
    meso_cols = [c for c in meso_cols if c in df.columns]
    micro_cols = [c for c in micro_cols if c in df.columns]
    feature_cols = macro_cols + meso_cols + micro_cols

    print(f"🔍 特徵分流 - Macro: {len(macro_cols)}, Meso: {len(meso_cols)}, Micro: {len(micro_cols)}")

    df_feat = df[feature_cols].copy()
    df_numeric = df_feat.select_dtypes(include=[np.number])
    constant_mask = df_numeric.nunique() <= 1
    if constant_mask.any():
        dropped = constant_mask[constant_mask].index.tolist()
        print(f"⚠️ 移除常數特徵: {dropped}")
        df_numeric = df_numeric.drop(columns=dropped)
        macro_cols = [c for c in macro_cols if c in df_numeric.columns]
        meso_cols = [c for c in meso_cols if c in df_numeric.columns]
        micro_cols = [c for c in micro_cols if c in df_numeric.columns]
        feature_cols = macro_cols + meso_cols + micro_cols

    train_split_idx = int(len(df_numeric) * 0.8)
    if train_split_idx < 100:
        print(f"❌ 數據量太少 ({len(df_numeric)})，不足以進行訓練")
        return None

    # --- 數據歸一化 (Normalization) ---
    median = df_numeric.iloc[:train_split_idx].median()
    iqr = (df_numeric.iloc[:train_split_idx].quantile(0.75) - df_numeric.iloc[:train_split_idx].quantile(0.25)).replace(0, 1.0)

    df_norm = ((df_numeric - median) / iqr).fillna(0)
    df_norm = df_norm.clip(-10, 10)
    df_norm['date'] = df['date']

    # --- Tri-Core 數據提取 ---
    print("📊 準備 Tri-Core 時序張量...")
    df_15m_historical = df_norm.set_index('date').resample('15min', closed='right', label='right').last().dropna()

    data_macro = df_norm[macro_cols].values
    data_micro = df_norm[micro_cols].values
    data_meso_hist = df_15m_historical[meso_cols].values
    data_meso_1m = df_norm[meso_cols].values # Pre-compute to avoid Pandas overhead in loop
    
    # ATR 保留原始數值用於縮放閥值，Regime 保留整數 ID
    data_atr = df['atr'].fillna(10.0).values
    data_regime = df['regime_id'].fillna(7).astype(int).values
    label_values = df['label'].values

    print("⚡ 正在執行時序索引優化...")
    indices_15m = df_15m_historical.index.get_indexer(df_norm['date'], method='pad')

    X_macro, X_meso, X_micro, Regimes, ATRs, Y = [], [], [], [], [], []
    win_micro, win_meso = 10, 15

    print("🛠️ 正在合成時序張量 (這可能需要一點時間)...")
    last_p = -1
    for i in range(win_micro - 1, len(df_norm)):
        p = int((i / len(df_norm)) * 10) * 10
        if p != last_p:
            print(f"   ⌛ 已完成 {p}%")
            last_p = p

        idx_15m = indices_15m[i]
        if idx_15m >= win_meso - 1:
            X_micro.append(data_micro[i - win_micro + 1 : i + 1])
            X_macro.append(data_macro[i])
            Regimes.append(data_regime[i])
            ATRs.append(data_atr[i])

            hist_15m = data_meso_hist[idx_15m - (win_meso - 2) : idx_15m + 1]
            current_1m_state_meso = data_meso_1m[i].reshape(1, -1)

            if len(hist_15m) > 0:
                seq_15m = np.concatenate([hist_15m, current_1m_state_meso], axis=0)
            else:
                seq_15m = np.tile(current_1m_state_meso, (win_meso, 1))

            X_meso.append(seq_15m[-win_meso:])
            Y.append(label_values[i])

    if not X_micro:
        print("❌ 合成後張量列表為空")
        return None

    X_macro = np.array(X_macro)
    X_meso = np.array(X_meso)
    X_micro = np.array(X_micro)
    Regimes = np.array(Regimes)
    ATRs = np.array(ATRs)
    Y = np.array(Y)

    total_samples = len(X_micro)
    split_idx = int(total_samples * 0.8)
    gap = win_micro

    # Tensor 轉換並直接放進 GPU (In-VRAM Training)
    t_mac_tr = torch.tensor(X_macro[:split_idx], dtype=torch.float32).to(device)
    t_mes_tr = torch.tensor(X_meso[:split_idx], dtype=torch.float32).to(device)
    t_mic_tr = torch.tensor(X_micro[:split_idx], dtype=torch.float32).to(device)
    t_reg_tr = torch.tensor(Regimes[:split_idx], dtype=torch.long).to(device)
    t_atr_tr = torch.tensor(ATRs[:split_idx], dtype=torch.float32).to(device)
    Y_train = torch.tensor(Y[:split_idx], dtype=torch.long).to(device)

    t_mac_val = torch.tensor(X_macro[split_idx + gap:], dtype=torch.float32).to(device)
    t_mes_val = torch.tensor(X_meso[split_idx + gap:], dtype=torch.float32).to(device)
    t_mic_val = torch.tensor(X_micro[split_idx + gap:], dtype=torch.float32).to(device)
    t_reg_val = torch.tensor(Regimes[split_idx + gap:], dtype=torch.long).to(device)
    t_atr_val = torch.tensor(ATRs[split_idx + gap:], dtype=torch.float32).to(device)
    Y_val = torch.tensor(Y[split_idx + gap:], dtype=torch.long).to(device)

    print(f"✅ 數據就緒: Train={len(t_mic_tr)}, Val={len(t_mic_val)}")

    model = TriCoreMarketEngine(
        macro_dim=len(macro_cols), meso_dim=len(meso_cols), micro_dim=len(micro_cols),
        d_model=config.d_model, nhead=config.nhead, num_layers=config.num_layers,
        fc_dropout=0.25, transformer_dropout=0.15,
        seq_len_meso=win_meso, seq_len_micro=win_micro
    )
    model = model.to(device)

    optimizer = optim.AdamW(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    warmup = optim.lr_scheduler.LinearLR(optimizer, start_factor=0.1, total_iters=5)
    cosine = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2, eta_min=1e-6)
    scheduler = optim.lr_scheduler.SequentialLR(optimizer, schedulers=[warmup, cosine], milestones=[5])

    class_counts = np.bincount(Y_train.cpu().numpy(), minlength=5)
    weights = np.zeros(5, dtype=np.float32)
    active_classes = class_counts > 0
    weights[active_classes] = 1.0 / np.sqrt(class_counts[active_classes])
    alpha_weights = torch.tensor(weights, dtype=torch.float32).to(device)

    # 定義自訂方向懲罰矩陣 (M[Target, Prediction])
    # 邏輯：方向(多空)占大頭(懲罰4.0~6.0)，強弱才是修正(懲罰0.5)
    penalty_matrix = torch.tensor([
        [0.0, 1.0, 2.0, 1.0, 2.0], # Target 0 (Hold)
        [1.0, 0.0, 0.5, 4.0, 5.0], # Target 1 (Small Long): 猜錯方向重罰4~5，猜大漲輕罰0.5
        [2.0, 0.5, 0.0, 5.0, 6.0], # Target 2 (Big Long): 猜錯方向重罰5~6，猜小漲輕罰0.5
        [1.0, 4.0, 5.0, 0.0, 0.5], # Target 3 (Small Short): 猜錯方向重罰4~5，猜大跌輕罰0.5
        [2.0, 5.0, 6.0, 0.5, 0.0], # Target 4 (Big Short): 猜錯方向重罰5~6，猜小跌輕罰0.5
    ], dtype=torch.float32).to(device)

    def directional_aware_loss(logits, targets, alpha, gamma=2.0):
        import torch.nn.functional as F
        ce_loss = nn.CrossEntropyLoss(weight=alpha, label_smoothing=0.1, reduction='none')(logits, targets)
        pt = torch.exp(-ce_loss)
        f_loss = (((1 - pt + 1e-6) ** gamma) * ce_loss)
        
        probs = F.softmax(logits, dim=1)
        batch_penalties = penalty_matrix[targets] # 取出每個樣本對應的懲罰權重 (batch, 5)
        expected_penalty = torch.sum(probs * batch_penalties, dim=1) # 期望懲罰值
        
        return (f_loss + expected_penalty * 1.5).mean()

    train_ds = TensorDataset(t_mac_tr, t_reg_tr, t_mes_tr, t_mic_tr, t_atr_tr, Y_train)
    val_ds = TensorDataset(t_mac_val, t_reg_val, t_mes_val, t_mic_val, t_atr_val, Y_val)
    # 取消 pin_memory=True 因為已經在 GPU 上
    train_loader = DataLoader(train_ds, batch_size=config.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=config.batch_size, shuffle=False)

    scaler = torch.amp.GradScaler('cuda') if device.type == 'cuda' else None
    best_val_loss = float('inf')
    best_model_state = None
    patience = 0

    model_version = "v19" # 每次訓練可手動改版號
    print(f"🚀 開始 Tri-Core 訓練...")
    for epoch in range(config.epochs):
        model.train()
        train_loss, train_cor, train_tot = 0, 0, 0
        for mac, reg, mes, mic, atr, y in train_loader:
            # 資料已經在 GPU，無需再次 .to(device)
            # Data Augmentation: 注入標準差 0.05 的高斯白雜訊
            noise_std = 0.05
            mac_noisy = mac + torch.randn_like(mac) * noise_std
            mes_noisy = mes + torch.randn_like(mes) * noise_std
            mic_noisy = mic + torch.randn_like(mic) * noise_std
            
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast('cuda', enabled=(scaler is not None)):
                out_main, out_meso, out_mac = model(mac_noisy, reg, mes_noisy, mic_noisy, atr)
                l_main = directional_aware_loss(out_main, y, alpha_weights)
                l_meso = directional_aware_loss(out_meso, y, alpha_weights)
                l_mac = directional_aware_loss(out_mac, y, alpha_weights)
                # Deep Supervision (Auxiliary Tasks)
                loss = l_main + 0.3 * l_meso + 0.1 * l_mac
            
            if scaler:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()
            
            train_loss += loss.item()
            _, pred = torch.max(out_main, 1)
            train_tot += y.size(0)
            train_cor += (pred == y).sum().item()

        avg_t_loss = train_loss / len(train_loader)
        t_acc = train_cor / train_tot

        model.eval()
        val_loss, val_cor, val_tot = 0, 0, 0
        with torch.no_grad():
            for mac, reg, mes, mic, atr, y in val_loader:
                # 資料已經在 GPU，無需再次 .to(device)
                out_main, out_meso, out_mac = model(mac, reg, mes, mic, atr)
                l_main = directional_aware_loss(out_main, y, alpha_weights)
                l_meso = directional_aware_loss(out_meso, y, alpha_weights)
                l_mac = directional_aware_loss(out_mac, y, alpha_weights)
                loss = l_main + 0.3 * l_meso + 0.1 * l_mac
                val_loss += loss.item()
                
                _, pred = torch.max(out_main, 1)
                val_tot += y.size(0)
                val_cor += (pred == y).sum().item()

        avg_v_loss = val_loss / len(val_loader)
        v_acc = val_cor / val_tot
        scheduler.step()

        writer.add_scalars('Loss', {'train': avg_t_loss, 'val': avg_v_loss}, epoch)
        if use_wandb: wandb.log({"train_loss": avg_t_loss, "val_loss": avg_v_loss, "train_acc": t_acc, "val_acc": v_acc, "epoch": epoch})

        if (epoch+1) % 10 == 0:
            print(f"🔥 Epoch [{epoch+1}/{config.epochs}], T-Loss: {avg_t_loss:.4f}, V-Loss: {avg_v_loss:.4f} | T-Acc: {t_acc:.4f}, V-Acc: {v_acc:.4f}", flush=True)

        if avg_v_loss < best_val_loss:
            best_val_loss = avg_v_loss
            patience = 0
            best_model_state = copy.deepcopy(model.state_dict())
        else:
            patience += 1
            if patience >= 50:
                print(f"🛑 Early Stopping at Epoch {epoch+1}")
                break

    if best_model_state is not None:
        print(f"💾 儲存最佳模型 (Val Loss: {best_val_loss:.6f})...")
        model.load_state_dict(best_model_state)
        manager = TradingModelManager(model_dir=os.path.join(os.path.dirname(__file__), "saved_models"))
        
        try:
            config_dict = {k: v for k, v in wandb.config.items() if not callable(v)} if use_wandb else {k: v for k, v in config.__dict__.items() if not k.startswith('_')}
        except:
            config_dict = {"note": "config dict error"}

        manager.save_model(model, optimizer, {"val_loss": best_val_loss}, {"config": config_dict})

    # Save norm params
    meta = {
        "mean": median.to_dict(), "std": iqr.to_dict(), 
        "macro_cols": macro_cols, "meso_cols": meso_cols, "micro_cols": micro_cols
    }
    with open(os.path.join(os.path.dirname(__file__), "saved_models", "norm_params.json"), "w", encoding='utf-8') as f:
        json.dump(meta, f)

    writer.close()
    return best_val_loss

class DefaultConfig:
    def __init__(self):
        self.lr = 0.0004
        self.d_model = 128
        self.nhead = 8
        self.num_layers = 2
        self.weight_decay = 0.03
        self.batch_size = 512
        self.epochs = 250

if __name__ == "__main__":
    import traceback
    import wandb
    try:
        final_df = prepare_training_data()
        if final_df.empty: exit(1)

        import torch
        print(f"PyTorch version: {torch.__version__} | CUDA: {torch.cuda.is_available()}")

        api_key = os.getenv("WANDB_API_KEY")
        if api_key:
            try: wandb.login(key=api_key)
            except: pass
            wandb.init(project="trading_mtf_single", config=DefaultConfig().__dict__)
            train_one_run(final_df, wandb.config)
        else:
            train_one_run(final_df, DefaultConfig())
    except BaseException as be:
        print(f"💥 錯誤: {be}")
        traceback.print_exc()
