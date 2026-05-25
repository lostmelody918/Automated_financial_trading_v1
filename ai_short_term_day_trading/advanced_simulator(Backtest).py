import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import torch
from data_engine import DayTradingDataEngine
from strategy_factory import StrategyFactory
from composite_ai import CompositeDayTradingAI
from model_manager import TradingModelManager

def save_trade_plot(df, entry_idx, exit_idx, trade_type, ret, trade_id, entry_features=None, trade_capital=0, position_dir=1):
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

        filename_base = f"trade_{trade_id:03d}_{trade_type}_{'WIN' if ret > 0 else 'LOSS'}"
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
    except Exception as e:
        print(f"Plot saving failed: {e}")

def run_advanced_simulator(initial_capital=100000, days=5):
    engine = DayTradingDataEngine()
    df, best_contract = engine.fetch_active_option_intraday_data(days=days)

    if df.empty:
        print("沒有數據。")
        return

    # 初始化模型與參數
    feature_cols = [c for c in df.columns if c not in ['date', 'time', 'date_only', 'day_of_week']]
    input_dim = len(feature_cols)

    ai_model = CompositeDayTradingAI(input_dim=input_dim, d_model=128, nhead=8, num_layers=3)
    optimizer = torch.optim.Adam(ai_model.parameters(), lr=0.001)
    model_manager = TradingModelManager(model_dir=os.path.join(os.path.dirname(__file__), "saved_models"))

    ai_model, optimizer, current_version = model_manager.load_latest_model(ai_model, optimizer)
    ai_model.eval()

    norm_path = os.path.join(os.path.dirname(__file__), "saved_models", "norm_params.json")
    if os.path.exists(norm_path):
        import json
        with open(norm_path, 'r') as f:
            norm_params = json.load(f)
        mean_v = np.array([norm_params['mean'][c] for c in norm_params['feature_cols']])
        std_v = np.array([norm_params['std'][c] for c in norm_params['feature_cols']])
    else:
        mean_v, std_v = 0, 1

    strategy_engine = StrategyFactory.get_strategy("composite")

    # 選擇權實務模式：無額外槓桿，依照選擇權真實點數與乘數計算損益
    LEVERAGE = 1
    # 選擇權一口交易成本約 70 元 (手續費 + 滑價約 1 tick)
    CONTRACT_MULTIPLIER = 50
    FEE_SLIPPAGE_PER_CONTRACT = 70
    MAX_POSITION_CAPITAL = 4000000 

    def get_dynamic_cost(num_contracts):
        return num_contracts * FEE_SLIPPAGE_PER_CONTRACT

    print(f"\n📊 啟動【選擇權實務模式】AI 突破推理 & 交易波段自動繪圖模擬器")
    print(f"💵 初始本金: NT$ {initial_capital:,} | 選擇權乘數: {CONTRACT_MULTIPLIER}")

    trade_log = []
    position = 0
    entry_price = 0
    entry_idx = 0
    current_entry_features = {} 
    current_capital = initial_capital
    capital_curve = [initial_capital]
    trade_capital_used = 0
    num_contracts = 0

    # 短停利、寬停損的勝率極大化策略 (以權利金的 % 計算)
    TAKE_PROFIT_PCT = 0.50  # 權利金獲利 50% 停利
    STOP_LOSS_PCT = -0.30   # 權利金虧損 30% 停損
    WINDOW_SIZE = 40

    # 連續波段與追蹤停損變數
    last_trade_win = False
    trailing_stop_active = False
    trailing_stop_price = 0

    for i in range(WINDOW_SIZE, len(df)-1):
        curr_slice = df.iloc[:i+1]
        last_row = curr_slice.iloc[-1]
        next_row = df.iloc[i+1]
        curr_time = last_row['date'].time()

        # AI 推理
        feat_data = curr_slice[norm_params['feature_cols']].tail(WINDOW_SIZE).values
        feat_normalized = (feat_data - mean_v) / std_v
        feat_tensor = torch.tensor(feat_normalized, dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            logits = ai_model(feat_tensor)
            probs = torch.softmax(logits, dim=1).squeeze().cpu().numpy()

        # 1. 平倉邏輯
        if position != 0:
            is_next_day = next_row['date_only'] > current_entry_features['entry_date'].date()
            if is_next_day and curr_time.hour == 13 and curr_time.minute >= 25:
                # EOD 平倉
                exit_price = next_row['Open']
                points_gained = (exit_price - entry_price) if position == 1 else (entry_price - exit_price)
                pnl = (points_gained * CONTRACT_MULTIPLIER * num_contracts) - get_dynamic_cost(num_contracts)
                net_ret = pnl / trade_capital_used
                current_capital += pnl
                last_trade_win = pnl > 0
                trade_log.append({'date': next_row['date'], 'type': 'Close_Next_Day_EOD', 'ret': net_ret, 'capital': current_capital, 'entry_features': current_entry_features})
                save_trade_plot(df, entry_idx, i+1, 'Close_Next_Day_EOD', net_ret, len(trade_log), current_entry_features, trade_capital_used, position)
                position = 0
                capital_curve.append(current_capital)
                continue

            high_ret = (next_row['High'] - entry_price) / entry_price if position == 1 else (entry_price - next_row['Low']) / entry_price
            low_ret = (next_row['Low'] - entry_price) / entry_price if position == 1 else (entry_price - next_row['High']) / entry_price
            current_ret = (next_row['Open'] - entry_price) / entry_price if position == 1 else (entry_price - next_row['Open']) / entry_price
            
            # --- ATR Trailing Stop 實作 ---
            current_atr = last_row['atr']
            if current_ret >= 0.20: # 獲利超過 20% 啟動追蹤
                trailing_stop_active = True
                trail_price = next_row['High'] - (current_atr * 2.5) if position == 1 else next_row['Low'] + (current_atr * 2.5)
                if position == 1:
                    trailing_stop_price = max(trailing_stop_price, trail_price)
                else:
                    if trailing_stop_price == 0: trailing_stop_price = 999999
                    trailing_stop_price = min(trailing_stop_price, trail_price)

            if trailing_stop_active:
                if (position == 1 and next_row['Low'] < trailing_stop_price) or (position == -1 and next_row['High'] > trailing_stop_price):
                    exit_price = trailing_stop_price
                    points_gained = (exit_price - entry_price) if position == 1 else (entry_price - exit_price)
                    pnl = (points_gained * CONTRACT_MULTIPLIER * num_contracts) - get_dynamic_cost(num_contracts)
                    net_ret = pnl / trade_capital_used
                    current_capital += pnl
                    last_trade_win = pnl > 0
                    trade_log.append({'date': next_row['date'], 'type': 'Trailing_Stop', 'ret': net_ret, 'capital': current_capital, 'entry_features': current_entry_features})
                    save_trade_plot(df, entry_idx, i+1, 'Trailing_Stop', net_ret, len(trade_log), current_entry_features, trade_capital_used, position)
                    position = 0
                    capital_curve.append(current_capital)
                    continue

            # 悲觀測試：先檢查停損
            hit_sl = low_ret <= STOP_LOSS_PCT
            hit_tp = high_ret >= TAKE_PROFIT_PCT

            if hit_sl:
                exit_price = entry_price * (1 + STOP_LOSS_PCT) if position == 1 else entry_price * (1 - STOP_LOSS_PCT)
                points_gained = (exit_price - entry_price) if position == 1 else (entry_price - exit_price)
                pnl = (points_gained * CONTRACT_MULTIPLIER * num_contracts) - get_dynamic_cost(num_contracts)
                net_ret = pnl / trade_capital_used
                current_capital += pnl
                last_trade_win = False
                trade_log.append({'date': next_row['date'], 'type': 'Stop_Loss', 'ret': net_ret, 'capital': current_capital, 'entry_features': current_entry_features})
                save_trade_plot(df, entry_idx, i+1, 'Stop_Loss', net_ret, len(trade_log), current_entry_features, trade_capital_used, position)
                position = 0
                capital_curve.append(current_capital)
                continue
            elif hit_tp:
                exit_price = entry_price * (1 + TAKE_PROFIT_PCT) if position == 1 else entry_price * (1 - TAKE_PROFIT_PCT)
                points_gained = (exit_price - entry_price) if position == 1 else (entry_price - exit_price)
                pnl = (points_gained * CONTRACT_MULTIPLIER * num_contracts) - get_dynamic_cost(num_contracts)
                net_ret = pnl / trade_capital_used
                current_capital += pnl
                last_trade_win = True
                trade_log.append({'date': next_row['date'], 'type': 'Take_Profit', 'ret': net_ret, 'capital': current_capital, 'entry_features': current_entry_features})
                save_trade_plot(df, entry_idx, i+1, 'Take_Profit', net_ret, len(trade_log), current_entry_features, trade_capital_used, position)
                position = 0
                capital_curve.append(current_capital)
                continue

        capital_curve.append(current_capital)
        
        # 2. 進場邏輯
        # 帶入 last_trade_win 判斷是否連續進場
        signal = strategy_engine.generate_signal(curr_slice, ai_score=probs, last_win=last_trade_win)
        if signal != 0 and position == 0:
            position = signal
            entry_price = next_row['Open']
            entry_idx = i + 1
            trailing_stop_active = False
            trailing_stop_price = 0
            
            # 資金管理：計算可買口數 (最多使用 50% 資金)
            pos_size_pct = 0.50
            allocated_capital = min(current_capital * pos_size_pct, MAX_POSITION_CAPITAL)
            contract_cost = entry_price * CONTRACT_MULTIPLIER
            if contract_cost > 0:
                num_contracts = int(allocated_capital // contract_cost)
            else:
                num_contracts = 0
                
            if num_contracts < 1:
                num_contracts = 1 # 至少買一口
                
            trade_capital_used = num_contracts * contract_cost

            current_entry_features = {
                'entry_date': next_row['date'],
                'signal': signal,
                'prob_down': probs[0],
                'prob_neutral': probs[1],
                'prob_up': probs[2],
                'atr': last_row['atr'],
                'macd_hist': last_row['macd_hist'],
                'rsi': last_row['rsi'],
                'vwap_bias': last_row['vwap_bias'],
                'vix_ret_1d': last_row.get('vix_ret_1d', 0)
            }

    # 結算與報告 (略，維持原狀)
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
        df_features.to_csv(os.path.join(out_dir, "trade_features_log.csv"), index=False, encoding="utf-8-sig")
        print(f"💾 已儲存交易特徵日誌至 data_learn/trade_features_log.csv (共 {len(df_features)} 筆)")

    df_trades['week'] = df_trades['date'].dt.isocalendar().week
    weekly_true_ret = {}
    weekly_log = []
    last_week_capital = initial_capital
    
    # 獲取所有測試天數內的週次 (確保即便沒交易的週也會顯示)
    all_weeks = sorted(df['date'].dt.isocalendar().week.unique())
    
    for week in all_weeks:
        group = df_trades[df_trades['week'] == week]
        
        if not group.empty:
            week_end_capital = group['capital'].iloc[-1]
            ret = (week_end_capital - last_week_capital) / last_week_capital
            
            # 每週最佳與最差出手
            best_trade = group.loc[group['ret'].idxmax()]
            worst_trade = group.loc[group['ret'].idxmin()]
            
            best_feat = best_trade['entry_features']
            worst_feat = worst_trade['entry_features']
            
            weekly_log.append({
                'Week': week,
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
                'Week': week,
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
        model_manager.save_model(ai_model, optimizer, {"avg_weekly_ret": avg_weekly_ret}, {"leverage": LEVERAGE})

    plt.figure(figsize=(12, 6))
    plt.plot(capital_curve)
    plt.savefig(os.path.join(os.path.dirname(__file__), "data_learn", "equity_curve.png"))

if __name__ == "__main__":
    run_advanced_simulator(initial_capital=120000, days=1)
