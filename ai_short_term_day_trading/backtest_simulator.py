import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import matplotlib as mpl

# 解決 matplotlib 中文字體顯示問題
plt.rcParams['font.sans-serif'] = ['Microsoft JhengHei', 'SimHei', 'Arial'] # 優先使用微軟正黑體
plt.rcParams['axes.unicode_minus'] = False # 正常顯示負號
import torch
from mtf_data_engine import MTFDayTradingDataEngine
from strategy_factory import StrategyFactory
from hexa_core_ai import HexaCoreMarketEngine
from model_manager import TradingModelManager
from delta_gamma_theta import calculate_bs_greeks, get_dynamic_bsm_bounds

def save_trade_plot(df, entry_idx, exit_idx, trade_type, ret, trade_id, entry_features=None, trade_capital=0, position_dir=1, version_str="vUnknown", today_str="19700101"):
    """助手函數：繪製單筆交易的波段圖，並標註特徵與儲存資料"""
    try:
        # 動態調整 X 軸區間 (前後多抓一些 K 線)
        duration = exit_idx - entry_idx
        padding = max(10, int(duration * 0.5))
        start_plot_idx = max(0, entry_idx - padding)
        end_plot_idx = min(len(df)-1, exit_idx + padding)
        plot_df = df.iloc[start_plot_idx:end_plot_idx+1].copy()

        fig, ax = plt.subplots(figsize=(12, 6))
        ax.plot(plot_df['date'], plot_df['Close'], color='gray', alpha=0.5, label='Price')

        # 動態調整 Y 軸區間，避免變成直線
        y_min = plot_df['Low'].min()
        y_max = plot_df['High'].max()
        y_padding = (y_max - y_min) * 0.1
        if y_padding == 0:
            y_padding = y_max * 0.05
        ax.set_ylim(y_min - y_padding, y_max + y_padding)

        entry_row = df.iloc[entry_idx]
        exit_row = df.iloc[exit_idx]

        ax.scatter(entry_row['date'], entry_row['Close'], color='blue', marker='^', s=100, label='Entry')
        ax.scatter(exit_row['date'], exit_row['Close'], color='red', marker='v', s=100, label='Exit')

        ax.plot([entry_row['date'], exit_row['date']], [entry_row['Close'], exit_row['Close']],
                 color='green' if ret > 0 else 'red', linestyle='--', alpha=0.6)

        pnl_amount = trade_capital * ret
        dir_str = "LONG" if position_dir == 1 else "SHORT"
        title_str = f"Trade #{trade_id} | {dir_str} | {trade_type} | Ret: {ret*100:.2f}% | Cap: NT${trade_capital:,.0f} | PnL: NT${pnl_amount:,.0f}"
        ax.set_title(title_str)

        # 新增明顯的圖表內標示：做多/做空、交易本金、獲利/虧損金額
        info_text = f"方向 (Direction): {dir_str}\n本金 (Capital): NT${trade_capital:,.0f}\n損益 (PnL): NT${pnl_amount:,.0f}"
        props = dict(boxstyle='round', facecolor='white' if ret > 0 else 'mistyrose', alpha=0.9, edgecolor='gray')
        ax.text(0.05, 0.95, info_text, transform=ax.transAxes, fontsize=12,
                verticalalignment='top', bbox=props, color='green' if ret > 0 else 'red', weight='bold')

        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.xticks(rotation=45)

        plots_dir = os.path.join(os.path.dirname(__file__), "data_learn", "trade_plots")
        os.makedirs(plots_dir, exist_ok=True)

        # 在圖表上加上特徵文字
        if entry_features:
            feature_text = "\n".join([f"{k}: {v:.4f}" if isinstance(v, float) else f"{k}: {v}" for k, v in entry_features.items() if k != 'entry_date'])
            plt.gcf().text(0.02, 0.5, feature_text, fontsize=8, bbox=dict(facecolor='white', alpha=0.8))

        filename_base = f"trade_{trade_id:03d}_{trade_type}_{'WIN' if ret > 0 else 'LOSS'}_{version_str}_{today_str}"
        plt.tight_layout(rect=[0.15, 0, 1, 1]) # 留空間給左側文字
        plt.savefig(os.path.join(plots_dir, f"{filename_base}.png"))
        plt.close()

        # 儲存詳細資料成 txt
        if entry_features:
            with open(os.path.join(plots_dir, f"{filename_base}.txt"), 'w', encoding='utf-8') as f:
                f.write(f"Trade ID: {trade_id}\n")
                f.write(f"Type: {trade_type}\n")
                f.write(f"Return: {ret*100:.2f}%\n")
                f.write("-" * 20 + "\n")
                for k, v in entry_features.items():
                    f.write(f"{k}: {v}\n")

        # 匯出持倉期間的每一分鐘所有特徵軌跡 (CSV)
        holding_df = df.iloc[entry_idx:exit_idx+1].copy()
        csv_filename = f"{filename_base}_trajectory.csv"
        # 使用 utf-8-sig 確保 Excel 打開不會亂碼
        holding_df.to_csv(os.path.join(plots_dir, csv_filename), index=False, encoding='utf-8-sig')

    except Exception as e:
        print(f"Plot saving failed: {e}")

def run_advanced_simulator(initial_capital=100000, days=120):
    engine = MTFDayTradingDataEngine()
    df_raw = engine.fetch_intraday_data(days=days)
    df_chips = engine.fetch_real_historical_chips(days=days + 15)

    if df_raw.empty or df_chips.empty:
        print("沒有數據。")
        return

    df = engine.integrate_institutional_chips(df_raw, df_chips)
    df = engine.process_mtf_features(df)
    
    # 限制回測資料量避免 VRAM OOM
    bars_needed = int(days * 300) + 135
    if len(df) > bars_needed:
        df = df.tail(bars_needed).reset_index(drop=True)

    if df.empty:
        print("融合後沒有數據。")
        return

    norm_path = os.path.join(os.path.dirname(__file__), "saved_models", "norm_params.json")
    if os.path.exists(norm_path):
        import json
        with open(norm_path, 'r') as f:
            norm_params = json.load(f)

        # 過濾掉無法正規化的欄位 (如 date_x, date_y)
        valid_cols = [c for c in norm_params['mean'].keys() if c in df.columns]

        mean_v = np.array([norm_params['mean'][c] for c in valid_cols])
        std_v = np.array([norm_params['std'][c] for c in valid_cols])
        input_dim = len(valid_cols)
    else:
        mean_v, std_v = 0, 1
        feature_cols = [c for c in df.columns if c not in ['date', 'time', 'date_only', 'day_of_week', 'date_x', 'date_y']]
        input_dim = len(feature_cols)

    # 定義 Hexa-Core 特徵分流
    mac_cols = ['nasdaq_prev_ret', 'nikkei_premarket_momentum', 'pc_ratio', 'us_tw_gap_divergence', 'settlement_type', 'is_settlement_day', 'dte', 'gap_amplitude', 'time_sin', 'time_cos']
    mes_cols = ['macd_hist', 'macd', 'rsi', 'rsi_fast', 'intraday_trend', 'dist_from_ma20', 'pullback_from_high', 'bounce_from_low', 'is_squeeze', 'slope_vwap', 'slope_ma20', 'spot_futures_proxy']
    mic_cols = ['ret', 'tr', 'body_length', 'upper_shadow', 'lower_shadow', 'momentum_explosion', 'vol_surge_ratio', 'pv_divergence', 'obv_bias', 'obv_slope', 'gap_filled', 'noise_wavelet', 'vol_range_ratio', 'orderbook_imbalance', 'volume_delta', 'cvd_bias']

    mac_c = [c for c in mac_cols if c in df.columns]
    m30_c = [f'30m_{c}' for c in mes_cols if f'30m_{c}' in df.columns]
    m15_c = [f'15m_{c}' for c in mes_cols if f'15m_{c}' in df.columns]
    mi3_c = [f'3m_{c}' for c in mic_cols if f'3m_{c}' in df.columns]
    mi1_c = [f'1m_{c}' for c in mic_cols if f'1m_{c}' in df.columns]

    # 必須與 train.py 的參數一致
    ai_model = HexaCoreMarketEngine(
        macro_dim=len(mac_c), meso_dim=len(m30_c), micro_dim=len(mi3_c),
        d_model=128, nhead=8, num_layers=2,
        seq_len_30m=5, seq_len_15m=10, seq_len_3m=3, seq_len_1m=9, num_classes=5
    )
    model_manager = TradingModelManager(model_dir=os.path.join(os.path.dirname(__file__), "saved_models"))

    ai_model, _, current_version = model_manager.load_latest_model(ai_model)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ai_model.to(device)
    ai_model.eval()

    strategy_engine = StrategyFactory.get_strategy("composite")

    # 選擇權實務模式：無額外槓桿，依照選擇權真實點數與乘數計算損益
    LEVERAGE = 1
    # 選擇權一口交易成本約 100 元 (手續費 + 滑價約 1 tick)
    CONTRACT_MULTIPLIER = 50
    FEE_SLIPPAGE_PER_CONTRACT = 100
    MAX_POSITION_CAPITAL = 4000000

    def get_dynamic_cost(num_contracts):
        return num_contracts * FEE_SLIPPAGE_PER_CONTRACT

    print(f"\n🧠 啟動【Hexa-Core 向量化預測】生成全域機率張量...")
    
    # 向量化特徵正規化
    df_norm = df.copy()
    if os.path.exists(norm_path):
        for c in valid_cols:
            df_norm[c] = (df[c] - norm_params['mean'][c]) / norm_params['std'][c]
            
    df_norm = df_norm.fillna(0)
    
    data_mac = df_norm[mac_c].values
    data_30m = df_norm[m30_c].values
    data_15m = df_norm[m15_c].values
    data_3m = df_norm[mi3_c].values
    data_1m = df_norm[mi1_c].values
    data_atr = df['atr'].fillna(10.0).values
    data_regime = df['regime_id'].fillna(7).astype(int).values

    X_mac, X_30m, X_15m, X_3m, X_1m, Regimes, ATRs = [], [], [], [], [], [], []
    arr_30m = np.array([120, 60, 15, 5, 0])
    arr_15m = np.array([135, 90, 45, 30, 20, 15, 10, 5, 2, 0])
    arr_3m = np.array([6, 3, 0])
    arr_1m = np.arange(8, -1, -1)
    
    START_OFFSET = 135
    for i in range(START_OFFSET, len(df_norm)):
        X_mac.append(data_mac[i])
        Regimes.append(data_regime[i])
        ATRs.append(data_atr[i])
        X_30m.append(data_30m[i - arr_30m])
        X_15m.append(data_15m[i - arr_15m])
        X_3m.append(data_3m[i - arr_3m])
        X_1m.append(data_1m[i - arr_1m])
        
    t_mac = torch.tensor(np.array(X_mac), dtype=torch.float32).to(device)
    t_reg = torch.tensor(np.array(Regimes), dtype=torch.long).to(device)
    t_30m = torch.tensor(np.array(X_30m), dtype=torch.float32).to(device)
    t_15m = torch.tensor(np.array(X_15m), dtype=torch.float32).to(device)
    t_3m = torch.tensor(np.array(X_3m), dtype=torch.float32).to(device)
    t_1m = torch.tensor(np.array(X_1m), dtype=torch.float32).to(device)
    t_atr = torch.tensor(np.array(ATRs), dtype=torch.float32).to(device)
    
    # 批次切片預測避免 VRAM OOM
    batch_size = 2048
    all_probs_list = []
    with torch.no_grad():
        for b in range(0, len(t_mac), batch_size):
            out_main, _, _, _, _ = ai_model(
                t_mac[b:b+batch_size], t_reg[b:b+batch_size],
                t_30m[b:b+batch_size], t_15m[b:b+batch_size],
                t_3m[b:b+batch_size], t_1m[b:b+batch_size],
                t_atr[b:b+batch_size]
            )
            all_probs_list.append(torch.softmax(out_main, dim=1).cpu().numpy())
    all_probs = np.concatenate(all_probs_list, axis=0)
    print(f"✅ 全域機率張量預測完成！(總筆數: {len(all_probs)})")
    
    print(f"\n📊 啟動【選擇權實務模式】AI 突破推理 & 交易波段自動繪圖模擬器")
    print(f"💵 初始本金: NT$ {initial_capital:,} | 選擇權乘數: {CONTRACT_MULTIPLIER}")

    import datetime
    today_str = datetime.datetime.now().strftime('%Y%m%d')
    version_str = current_version if current_version else "vUnknown"

    # 控制是否要儲存交易圖表，設為 True 會大幅拖慢回測速度 (若交易次數破千)
    SAVE_PLOTS = False
    if not SAVE_PLOTS:
        print("⚡ 已關閉單筆交易波段圖繪製 (SAVE_PLOTS=False) 以加速回測運行。")

    trade_log = []
    position = 0
    entry_price = 0
    entry_idx = 0
    current_entry_features = {}
    current_capital = initial_capital
    capital_curve = [initial_capital]
    trade_capital_used = 0
    num_contracts = 0
    START_OFFSET = 135

    # 連續波段與追蹤停損變數
    last_trade_win = False
    is_scalp = False
    highest_price_since_entry = 0.0
    hard_tp_price = 0.0
    hard_sl_price = 0.0

    total_bars = len(df)
    for i in range(START_OFFSET, total_bars-1):
        if i % 2000 == 0:
            print(f"⏳ 回測進度: {i} / {total_bars} ({i/total_bars*100:.1f}%) | 當前本金: {current_capital:,.0f}")

        curr_slice = df.iloc[:i+1]
        last_row = curr_slice.iloc[-1]
        next_row = df.iloc[i+1]
        curr_time = last_row['date'].time()

        # --- Hexa-Core AI 向量化機率查表 ---
        if i < START_OFFSET:
            probs = np.zeros(5)
            probs[2] = 1.0 # 預設盤整 (index 2)
        else:
            probs = all_probs[i - START_OFFSET]

        # 1. 平倉邏輯
        if position != 0:
            S_high = next_row['High']
            S_low = next_row['Low']
            S_close = next_row['Close']

            K = current_entry_features['K']
            T = current_entry_features['T']
            opt_type = current_entry_features['opt_type']

            # [真實動態模擬] 平倉時也要考慮動態 IV
            exit_vol_surge = min(max(last_row.get('vol_surge_ratio', 1.0), 1.0), 4.0)
            base_exit_iv = 0.22
            exit_iv = base_exit_iv + (exit_vol_surge * 0.02) if opt_type == 'Put' else base_exit_iv + (exit_vol_surge * 0.005)

            # 使用 BSM 轉換最高最低指數價為權利金 (考量動態 IV)
            _, _, _, opt_price_at_high = calculate_bs_greeks(S_high, K, T, 0.015, exit_iv, opt_type)
            _, _, _, opt_price_at_low = calculate_bs_greeks(S_low, K, T, 0.015, exit_iv, opt_type)
            _, _, _, opt_price_close = calculate_bs_greeks(S_close, K, T, 0.015, exit_iv, opt_type)

            # 對 Call 而言，指數越高選擇權越高；對 Put 而言，指數越低選擇權越高
            if position == 1:
                opt_high = opt_price_at_high
                opt_low = opt_price_at_low
            else:
                opt_high = opt_price_at_low
                opt_low = opt_price_at_high

            # 追蹤歷史最高權利金以啟動 Trailing Stop
            if opt_high > highest_price_since_entry:
                highest_price_since_entry = opt_high

            exit_reason = None
            exec_price = 0.0

            # === 極短線 (Scalping) 出場邏輯 ===
            if is_scalp:
                bars_held = (i + 1) - entry_idx
                is_reverse_k = (next_row['Close'] < next_row['Open']) if position == 1 else (next_row['Close'] > next_row['Open'])
                rsi_exit_long = position == 1 and last_row['rsi_fast'] >= 80 and is_reverse_k
                rsi_exit_short = position == -1 and last_row['rsi_fast'] <= 20 and is_reverse_k
                time_exit = bars_held >= 8

                if rsi_exit_long or rsi_exit_short or time_exit:
                    exit_reason = 'Scalp_RSI_Exit' if (rsi_exit_long or rsi_exit_short) else 'Scalp_Time_Exit'
                    exec_price = opt_price_close
                    is_scalp = False

            # === 當沖強制作業邏輯 (No Overnight) ===
            if not exit_reason:
                # 絕不留倉，若時間達到 13:30，強制市價平倉
                if curr_time.hour == 13 and curr_time.minute >= 30:
                    exit_reason = 'Close_Intraday_EOD'
                    exec_price = opt_price_close

            if not exit_reason:
                # Trailing Stop 檢查
                profit_points = highest_price_since_entry - entry_price
                trailing_sl = hard_sl_price
                if profit_points >= 9:
                    pullback = max(13, profit_points * 0.22) if profit_points >= 20 else max(5, profit_points * 0.16)
                    trailing_sl = max(hard_sl_price, highest_price_since_entry - pullback)
                    trailing_sl = max(trailing_sl, entry_price + 1.0) # 強制保本

                if opt_low <= trailing_sl and trailing_sl > hard_sl_price:
                    exit_reason = 'Trailing_Stop'
                    exec_price = trailing_sl
                elif opt_low <= hard_sl_price:
                    exit_reason = 'Stop_Loss'
                    exec_price = hard_sl_price
                elif opt_high >= hard_tp_price:
                    exit_reason = 'Take_Profit'
                    exec_price = hard_tp_price

            if exit_reason:
                # [真實動態模擬] 平倉滑價：緊急停損時滑價大，正常停利時滑價小
                exit_slippage = 3.0 if exit_reason == 'Stop_Loss' else 1.0
                actual_exec_price = max(exec_price - exit_slippage, 0.1) # 賣出時價格被壓低
                
                points_gained = actual_exec_price - entry_price # 選擇權不分多空，獲利就是賣價減買價
                pnl = (points_gained * CONTRACT_MULTIPLIER * num_contracts) - get_dynamic_cost(num_contracts)
                net_ret = pnl / trade_capital_used if trade_capital_used > 0 else 0
                current_capital += pnl
                last_trade_win = pnl > 0

                trade_log.append({'date': next_row['date'], 'type': exit_reason, 'ret': net_ret, 'capital': current_capital, 'entry_features': current_entry_features})
                save_trade_plot(df, entry_idx, i+1, exit_reason, net_ret, len(trade_log), current_entry_features, trade_capital_used, position, version_str, today_str)

                position = 0
                is_scalp = False
                capital_curve.append(current_capital)
                continue

        capital_curve.append(current_capital)

        # 2. 進場邏輯
        signal = strategy_engine.generate_signal(curr_slice, ai_score=probs, last_win=last_trade_win)
        
        # 僅限日盤交易且預留時間給當沖平倉 (08:45 ~ 13:15)
        import datetime
        is_day_session = (datetime.time(8, 45) <= curr_time <= datetime.time(13, 15))
        
        if signal != 0 and position == 0 and is_day_session:
            is_scalp = (abs(signal) == 10)
            position = 1 if signal > 0 else -1

            # --- 使用 BSM 模擬真實選擇權權利金與風控點 ---
            S = next_row['Open']
            K = round(S / 50) * 50 # 取最接近的價平履約價
            T = 7 / 365.0 # 假設為近週選
            opt_type = 'Call' if position == 1 else 'Put'
            
            # [真實動態模擬] 根據大盤爆量與波動程度，動態膨脹 IV (引發選擇權權利金暴漲)
            vol_surge_factor = min(max(last_row.get('vol_surge_ratio', 1.0), 1.0), 4.0)
            base_iv = 0.22
            if opt_type == 'Put':
                # 跌勢恐慌時 Put IV 狂飆 (Volatility Skew)
                iv = base_iv + (vol_surge_factor * 0.02)
            else:
                iv = base_iv + (vol_surge_factor * 0.005)

            _, _, _, simulated_entry_price = calculate_bs_greeks(S, K, T, 0.015, iv, opt_type)

            # [真實動態模擬] 模擬進場滑價與試探成本 (高波動時買賣價差會被拉開)
            # 在爆量瞬間，通常會買在相對高點 (Slippage 懲罰: 1 ~ 6 點)
            dynamic_slippage = min(vol_surge_factor * 1.5, 6.0)

            # 若計算出權利金小於 5 點，代表太過價外或模型偏差，強制棄單
            if simulated_entry_price < 5:
                position = 0
                continue

            entry_price = simulated_entry_price + dynamic_slippage
            entry_idx = i + 1
            highest_price_since_entry = entry_price

            # 設定動態風控區間
            abs_sig = abs(signal)
            if abs_sig == 3:
                tp_mult, sl_mult = 5.0, 1.5
                expected_hold = 2.0
            elif abs_sig == 2:
                tp_mult, sl_mult = 3.0, 1.0
                expected_hold = 1.0
            else:
                tp_mult, sl_mult = 1.5, 0.5
                expected_hold = 0.5

            hard_tp_price, hard_sl_price, _, _, _ = get_dynamic_bsm_bounds(
                S=S, K=K, T=T, r=0.015, iv=iv, atr=last_row['atr'],
                tp_mult=tp_mult, sl_mult=sl_mult, expected_hold_hours=expected_hold,
                option_type=opt_type, actual_entry_price=entry_price
            )

            # 資金管理
            pos_size_pct = 0.50
            allocated_capital = min(current_capital * pos_size_pct, MAX_POSITION_CAPITAL)
            contract_cost = entry_price * CONTRACT_MULTIPLIER
            if contract_cost > 0:
                num_contracts = int(allocated_capital // contract_cost)
            else:
                num_contracts = 0

            if num_contracts < 1:
                num_contracts = 1

            trade_capital_used = num_contracts * contract_cost

            current_entry_features = {
                'entry_date': next_row['date'],
                'signal': signal,
                'opt_type': opt_type,
                'K': K,
                'T': T,
                'prob_strong_down': probs[0],
                'prob_med_down': probs[1],
                'prob_neutral': probs[2],
                'prob_med_up': probs[3],
                'prob_strong_up': probs[4]
            }
            # 儲存 AI 所看到的所有特徵
            for col in valid_cols:
                current_entry_features[col] = last_row[col]

    # 結算與報告
    df_trades = pd.DataFrame(trade_log)
    if df_trades.empty:
        print("沒有交易次數")
        return

    features_log = []
    for t in trade_log:
        if 'entry_features' in t:
            feat = t['entry_features'].copy()
            feat['exit_date'] = t['date']
            feat['trade_type'] = t['type']
            feat['ret'] = t['ret']
            features_log.append(feat)

    if features_log:
        df_features = pd.DataFrame(features_log)
        out_dir = os.path.join(os.path.dirname(__file__), "data_learn")
        try:
            df_features.to_csv(os.path.join(out_dir, "trade_features_log.csv"), index=False, encoding="utf-8-sig")
            print(f"💾 已儲存交易特徵日誌至 data_learn/trade_features_log.csv (共 {len(df_features)} 筆)")
        except PermissionError:
            print("⚠️ 無法儲存 trade_features_log.csv，檔案可能正被 Excel 開啟。")
        
        # --- 新增: 特徵最佳化分析報告 ---
        print("\n📊 --- 特徵最佳化分析報告 ---")
        win_trades = df_features[df_features['ret'] > 0]
        loss_trades = df_features[df_features['ret'] <= 0]
        if not win_trades.empty and not loss_trades.empty:
            analysis_lines = []
            for col in valid_cols:
                if col in df_features.columns:
                    win_mean = win_trades[col].mean()
                    loss_mean = loss_trades[col].mean()
                    diff_pct = (win_mean - loss_mean) / (abs(loss_mean) + 1e-9) * 100
                    if abs(diff_pct) > 10: # 只列出差異超過 10% 的特徵
                        analysis_lines.append(f"  - {col}: 獲利均值 {win_mean:.4f} | 虧損均值 {loss_mean:.4f} (差異 {diff_pct:+.1f}%)")
            
            if analysis_lines:
                print("💡 發現獲利與虧損交易在以下特徵有顯著差異 (可用於後續優化):")
                for line in analysis_lines:
                    print(line)
                
                with open(os.path.join(out_dir, "feature_optimization_report.txt"), 'w', encoding='utf-8') as f:
                    f.write("發現獲利與虧損交易在以下特徵有顯著差異 (可用於後續優化):\n")
                    f.write("\n".join(analysis_lines))
            else:
                print("💡 目前獲利與虧損交易在各特徵上的平均差異皆不明顯。")

    # 修正 weekly summary: 加入年份避免跨年週次重疊
    df_trades['year_week'] = df_trades['date'].dt.strftime('%Y-W%W')
    weekly_true_ret = {}
    weekly_log = []
    last_week_capital = initial_capital

    # 獲取所有測試天數內的週次 (確保即便沒交易的週也會顯示)
    all_weeks = sorted(df['date'].dt.strftime('%Y-W%W').unique())

    for week in all_weeks:
        group = df_trades[df_trades['year_week'] == week]

        if not group.empty:
            week_end_capital = group['capital'].iloc[-1]
            ret = (week_end_capital - last_week_capital) / last_week_capital

            # 每週最佳與最差出手
            best_trade = group.loc[group['ret'].idxmax()]
            worst_trade = group.loc[group['ret'].idxmin()]

            best_feat = best_trade['entry_features']
            worst_feat = worst_trade['entry_features']

            weekly_log.append({
                'Year_Week': week,
                'Week_End_Total_Capital': week_end_capital,
                'Weekly_Net_Return_%': ret * 100,
                'Weekly_Profit': week_end_capital - last_week_capital,
                'Trade_Count': len(group),
                'Best_Trade_Ret_%': best_trade['ret'] * 100,
                'Best_Trade_Time': best_feat['entry_date'],
                'Best_Trade_Type': best_trade['type'],
                'Best_Trade_Signal': "LONG" if best_feat['signal'] == 1 else "SHORT",
                'Worst_Trade_Ret_%': worst_trade['ret'] * 100,
                'Worst_Trade_Time': worst_feat['entry_date'],
                'Worst_Trade_Type': worst_trade['type'],
                'Worst_Trade_Signal': "LONG" if worst_feat['signal'] == 1 else "SHORT"
            })
            weekly_true_ret[week] = ret
            last_week_capital = week_end_capital
        else:
            # 沒交易的週
            weekly_log.append({
                'Year_Week': week,
                'Week_End_Total_Capital': last_week_capital,
                'Weekly_Net_Return_%': 0.0,
                'Weekly_Profit': 0.0,
                'Trade_Count': 0,
                'Best_Trade_Ret_%': 0.0,
                'Best_Trade_Time': "N/A",
                'Best_Trade_Type': "N/A",
                'Best_Trade_Signal': "N/A",
                'Worst_Trade_Ret_%': 0.0,
                'Worst_Trade_Time': "N/A",
                'Worst_Trade_Type': "N/A",
                'Worst_Trade_Signal': "N/A"
            })
            weekly_true_ret[week] = 0.0

    if weekly_log:
        df_weekly = pd.DataFrame(weekly_log)
        df_weekly.to_csv(os.path.join(out_dir, "weekly_summary.csv"), index=False, encoding="utf-8-sig")
        print(f"💾 已儲存詳細每週報酬報告至 data_learn/weekly_summary.csv (共 {len(df_weekly)} 週)")

    avg_weekly_ret = pd.Series(weekly_true_ret).mean()
    total_ret = (current_capital - initial_capital) / initial_capital

    print("\n" + "="*50)
    print(f"🚀 --- 終極期權當沖模擬器結算報告 ---")
    print(f"累積真實淨報酬率: {total_ret*100:.2f}%")
    print(f"真實平均每週報酬: {avg_weekly_ret*100:.2f}%")
    win_rate = (df_trades['ret']>0).mean() * 100
    print(f"總交易次數: {len(df_trades)} | 勝率: {win_rate:.2f}%")
    print("="*50)
    print(f"\n✅ 回測完成！交易波段圖已儲存至 data_learn/trade_plots/")

    if avg_weekly_ret > 0:
        model_manager.save_model(ai_model, None, {"avg_weekly_ret": avg_weekly_ret}, {"leverage": LEVERAGE})

    filename = f"equity_curve_{version_str}_{today_str}.png"

    plt.figure(figsize=(12, 6))
    plt.plot(capital_curve)
    plt.savefig(os.path.join(os.path.dirname(__file__), "data_learn", filename))

if __name__ == "__main__":
    run_advanced_simulator(initial_capital=120000, days=90)
