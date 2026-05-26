import os
import time
import json
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime, timedelta, time as datetime_time

from data_engine import DayTradingDataEngine
from composite_ai import CompositeDayTradingAI
from model_manager import TradingModelManager
from strategy_factory import StrategyFactory

# ==========================================
# 核心參數設定 (高風險高報酬 + 三大法人籌碼融合版)
# ==========================================
INITIAL_CAPITAL = 120000        # 初始模擬本金
CONTRACT_MULTIPLIER = 50        # 台指選擇權合約乘數 (1點 = 50元)
FEE_SLIPPAGE_PER_CONTRACT = 100 # 單口交易成本 (手續費 + 真實滑價約 1.5-2 ticks)
MAX_POSITION_CAPITAL = 4000000  # 最大部位限制
WINDOW_SIZE = 25                # AI 輸入的時間視窗長度

# 獲利與風險控制 (以權利金變動幅度 % 計算)
TAKE_PROFIT_PCT = 2.50          # 250% 硬停利 (拉高獲利潛力)
STOP_LOSS_PCT = -0.50           # 50% 硬停損 (放寬風險容忍度，避免被洗)
TRAILING_START_PCT = 0.40       # 獲利達 40% 時啟動追蹤停損
TRAILING_ATR_MULTIPLIER = 3.5   # 追蹤停損的 ATR 倍數 (放寬以避免假跌破)

POLL_INTERVAL_SECONDS = 60      # 盤中輪詢間隔 (每 60 秒檢查一次最新 5分K)

def is_market_open(current_time):
    """判斷目前是否為選擇權日盤交易時間"""
    return datetime_time(8, 45) <= current_time <= datetime_time(13, 45)

def is_eod_closing_time(current_time):
    """判斷是否達到日内當沖強制平倉時間 (設定於收盤前 5 分鐘：13:40)"""
    return datetime_time(13, 40) <= current_time < datetime_time(13, 45)

def load_latest_daily_chips_snapshot():
    """
    讀取截至昨日下午 15:00 結算公佈的三大法人留倉快照。
    實務上可透過外部排程腳本每日收盤後自動寫入 JSON/CSV，供今日盤中載入。
    """
    return pd.DataFrame([{
        'date': (datetime.now() - timedelta(days=1)).date(),
        'foreign_net_oi': -12500.0,
        'dealer_net_oi': 4200.0,
        'pc_ratio': 1.15
    }])

def generate_eod_report(daily_trades, df, best_contract, current_capital, today_date):
    print("📊 正在產出今日選擇權損益報告與交易圖表...")

    # 建立保存目錄
    report_dir = os.path.join(os.path.dirname(__file__), "daily_reports")
    os.makedirs(report_dir, exist_ok=True)

    report_filename = os.path.join(report_dir, f"EOD_Report_{today_date.strftime('%Y%m%d')}.png")

    # 準備繪圖資料
    # 我們只取今天的資料
    today_df = df[df['date'].dt.date == today_date].copy()
    if today_df.empty:
        print("⚠️ 無法產出報告：缺少今日 K 線資料。")
        return

    fig = plt.figure(figsize=(16, 12))
    gs = fig.add_gridspec(3, 1, height_ratios=[3, 1, 1], hspace=0.3)

    # 1. 價格與出手時機圖
    ax1 = fig.add_subplot(gs[0])
    ax1.plot(today_df['date'], today_df['Close'], label='Close Price', color='black', linewidth=1.5)
    ax1.set_title(f"[{today_date}] {best_contract.symbol} Intraday Price & Trading Timing", fontsize=16, fontweight='bold')
    ax1.set_ylabel("Price")
    ax1.grid(True, linestyle='--', alpha=0.6)

    # 標記進出場點
    total_pnl = 0
    trade_details = []

    for i, trade in enumerate(daily_trades):
        total_pnl += trade['pnl']
        # 買進標記
        marker_color = 'green' if '多' in trade['direction'] or 'Call' in trade['direction'] else 'red'
        marker_symbol = '^' if '多' in trade['direction'] or 'Call' in trade['direction'] else 'v'

        ax1.scatter(trade['entry_time'], trade['entry_price'], color=marker_color, marker=marker_symbol, s=150, zorder=5, label=f"Entry {i+1}" if i==0 else "")

        # 賣出標記
        exit_color = 'blue' if trade['pnl'] > 0 else 'orange'
        ax1.scatter(trade['exit_time'], trade['exit_price'], color=exit_color, marker='x', s=150, zorder=5, label=f"Exit {i+1}" if i==0 else "")

        # 連接線
        ax1.plot([trade['entry_time'], trade['exit_time']], [trade['entry_price'], trade['exit_price']], color='gray', linestyle=':', linewidth=1)

        trade_details.append(
            f"Trade {i+1}: {trade['direction']} | Entry: {trade['entry_time'].strftime('%H:%M:%S')} @ {trade['entry_price']:.2f} | "
            f"Exit: {trade['exit_time'].strftime('%H:%M:%S')} @ {trade['exit_price']:.2f} | "
            f"PnL: NT$ {trade['pnl']:,.0f}"
        )

    ax1.legend(loc='best')
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))

    # 2. 量能/籌碼圖 (這裡用成交量 Volume 代表)
    ax2 = fig.add_subplot(gs[1], sharex=ax1)
    # 分辨紅黑K量
    colors = ['red' if c >= o else 'green' for c, o in zip(today_df['Close'], today_df['Open'])]
    ax2.bar(today_df['date'], today_df['Volume'], color=colors, alpha=0.7, width=0.001)
    ax2.set_title("Volume (Trading Activity)", fontsize=14)
    ax2.set_ylabel("Volume")
    ax2.grid(True, linestyle='--', alpha=0.6)

    # 3. 損益文字報告區塊
    ax3 = fig.add_subplot(gs[2])
    ax3.axis('off')

    summary_text = (
        f"📅 Date: {today_date}\n"
        f"💰 Total Daily PnL: NT$ {total_pnl:,.0f}\n"
        f"🏦 Current Capital: NT$ {current_capital:,.0f}\n"
        f"📊 Number of Trades: {len(daily_trades)}\n\n"
        f"--- Trade Details ---\n"
    )

    for detail in trade_details:
        summary_text += detail + "\n"

    if not daily_trades:
        summary_text += "No trades executed today.\n"

    ax3.text(0.01, 0.95, summary_text, fontsize=12, family='monospace', verticalalignment='top',
             bbox=dict(boxstyle='round,pad=0.5', facecolor='#f8f9fa', edgecolor='gray', alpha=0.8))

    plt.tight_layout()
    plt.savefig(report_filename, dpi=150)
    plt.close()
    print(f"✅ 今日損益報告已匯出至: {report_filename}")

def run_live_simulator():
    print("=" * 60)
    print("🚀 啟動：三大法人籌碼融合 AI 選擇權即時模擬機 (High Risk Mode) 🚀")
    print(f"💵 初始模擬本金: NT$ {INITIAL_CAPITAL:,} | 選擇權乘數: {CONTRACT_MULTIPLIER}")
    print(f"⏱️ 監控時段: 日盤 08:45 ~ 13:45 (每 {POLL_INTERVAL_SECONDS} 秒輪詢)")
    print("=" * 60)

    engine = DayTradingDataEngine()

    norm_path = os.path.join(os.path.dirname(__file__), "saved_models", "norm_params.json")
    if not os.path.exists(norm_path):
        print("❌ 錯誤：找不到 saved_models/norm_params.json，請先執行 train_model.py。")
        return

    with open(norm_path, 'r', encoding='utf-8') as f:
        norm_params = json.load(f)

    feature_cols = norm_params['feature_cols']
    input_dim = len(feature_cols)
    mean_v = np.array([norm_params['mean'][c] for c in feature_cols])
    std_v = np.array([norm_params['std'][c] for c in feature_cols])

    ai_model = CompositeDayTradingAI(input_dim=input_dim, d_model=128, nhead=8, num_layers=3)
    model_manager = TradingModelManager(model_dir=os.path.join(os.path.dirname(__file__), "saved_models"))
    ai_model, _, version = model_manager.load_latest_model(ai_model)
    ai_model.eval()
    print(f"📦 成功載入 AI 模型版本: v{version}")

    strategy_engine = StrategyFactory.get_strategy("composite")

    current_capital = INITIAL_CAPITAL
    position = 0
    entry_price = 0.0
    entry_time = None
    num_contracts = 0
    trade_capital_used = 0
    last_trade_win = False

    trailing_stop_active = False
    trailing_stop_price = 0.0
    trade_log = []
    eod_report_done = False  # 日終結算報告控制旗標

    print("📡 系統初始化完成，正在接入 Shioaji API 進入即時盤中監控...")

    while True:
        now = datetime.now()
        current_time = now.time()
        time_str = now.strftime('%H:%M:%S')

        # 非交易時間守衛
        if not is_market_open(current_time):
            if position != 0:
                print(f"[{time_str}] ⚠️ 異常警訊：非交易時間仍持有虛擬部位，執行強制清倉。")
                position = 0
                num_contracts = 0
                trade_capital_used = 0

            # 非交易時段自動將報告控制狀態重置，為隔天開盤做準備
            if eod_report_done:
                eod_report_done = False
                trade_log = [] # 重置新一天的日誌緩衝區

            print(f"[{time_str}] 💤 目前為非交易時段，等待日盤開盤 (08:45)...")
            time.sleep(60)
            continue

        try:
            # 抓取盤中即時資料與籌碼快照
            # B. 抓取盤中即時資料
            # 這裡透過更新後的 data_engine，如果選擇權抓不到 5 天 K 線，底層會自動退回抓台指期貨來分析
            df_intraday, best_contract = engine.fetch_active_option_intraday_data(days=5)
            df_chips_daily = load_latest_daily_chips_snapshot()

            # 進行即時整合對齊 (防呆處理：若 data_engine 尚未加上該函數則回退)
            if hasattr(engine, 'integrate_institutional_chips'):
                df = engine.integrate_institutional_chips(df_intraday, df_chips_daily)
            else:
                df = df_intraday

            if df is None or df.empty or len(df) < WINDOW_SIZE:
                print(f"[{time_str}] ⚠️ 期貨底層數據庫更新中或資料不足，等待下一輪...")
                time.sleep(POLL_INTERVAL_SECONDS)
                continue

            last_row = df.iloc[-2]
            latest_row = df.iloc[-1]

            # 實戰報價脫鉤
            try:
                try:
                    snapshot = engine.api.snapshots([best_contract])[0]
                    current_price = snapshot.close
                    opt_high = snapshot.high if snapshot.high > 0 else current_price
                    opt_low = snapshot.low if snapshot.low > 0 else current_price
                except Exception as e:
                    # 避免 snapshots 報錯，退而使用 kbars 的最後一筆
                    df_opt_kbars, _ = engine.fetch_active_option_intraday_data(days=1)
                    if not df_opt_kbars.empty:
                        opt_latest_row = df_opt_kbars.iloc[-1]
                        current_price = opt_latest_row['Close']
                        opt_high = opt_latest_row['High']
                        opt_low = opt_latest_row['Low']
                    else:
                        raise e
            except Exception as e:
                print(f"[{time_str}] ⚠️ 無法取得選擇權即時報價: {e}")
                time.sleep(POLL_INTERVAL_SECONDS)
                continue

            last_row = df.iloc[-2]
            latest_row = df.iloc[-1]

            # 🚀 修正：實戰報價脫鉤
            try:
                # 實務上這裡可能需要檢查 API 型態，若無法 snapshots 可使用其他方法
                try:
                    snapshot = engine.api.snapshots([best_contract])[0]
                    current_price = snapshot.close
                    opt_high = snapshot.high if snapshot.high > 0 else current_price
                    opt_low = snapshot.low if snapshot.low > 0 else current_price
                except Exception as e:
                    # 避免 snapshots 報錯，退而使用 kbars 的最後一筆
                    # 注意：如果 fetch_active_option_intraday_data 回傳的第一個值是 K 線
                    df_opt_kbars, _ = engine.fetch_active_option_intraday_data(days=1)
                    if not df_opt_kbars.empty:
                        opt_latest_row = df_opt_kbars.iloc[-1]
                        current_price = opt_latest_row['Close']
                        opt_high = opt_latest_row['High']
                        opt_low = opt_latest_row['Low']
                    else:
                        raise e
            except Exception as e:
                print(f"[{time_str}] ⚠️ 無法取得選擇權即時報價: {e}")
                time.sleep(POLL_INTERVAL_SECONDS)
                continue

            # AI 特徵正規化與推論
            feat_data = df[feature_cols].tail(WINDOW_SIZE).values
            feat_tensor = torch.tensor((feat_data - mean_v) / std_v, dtype=torch.float32).unsqueeze(0)

            with torch.no_grad():
                logits = ai_model(feat_tensor)
                probs = torch.softmax(logits, dim=1).squeeze().cpu().numpy()

            status_mark = "🟢 持倉中" if position != 0 else "⚪ 空手"
            print(f"[{time_str}] 標的: {best_contract.symbol} | 即時價: {current_price} | AI預測 [跌:{probs[0]:.2f} 平:{probs[1]:.2f} 漲:{probs[2]:.2f}] | {status_mark}")

            # 即時風控與平倉邏輯
            if position != 0:
                points_gained = (current_price - entry_price) if position == 1 else (entry_price - current_price)
                current_pnl = (points_gained * CONTRACT_MULTIPLIER * num_contracts) - (num_contracts * FEE_SLIPPAGE_PER_CONTRACT)
                current_ret = current_pnl / trade_capital_used if trade_capital_used > 0 else 0

                pnl_display = f"NT$ +{current_pnl:,.0f}" if current_pnl > 0 else f"NT$ {current_pnl:,.0f}"
                print(f"   ↳ 帳面未實現損益: {pnl_display} (報酬率: {current_ret * 100:.2f}%)")

                exit_reason = None

                if is_eod_closing_time(current_time):
                    exit_reason = "日内時間截止強制平倉 (EOD)"

                current_atr = last_row['atr']
                if current_ret >= TRAILING_START_PCT:
                    if not trailing_stop_active:
                        trailing_stop_active = True
                        print(f"   ↳ 🔥 獲利達標，啟動 {TRAILING_ATR_MULTIPLIER}xATR 動態追蹤停利機制。")

                    trail_price = current_price - (current_atr * TRAILING_ATR_MULTIPLIER) if position == 1 else current_price + (current_atr * TRAILING_ATR_MULTIPLIER)

                    if position == 1:
                        trailing_stop_price = max(trailing_stop_price, trail_price)
                    else:
                        if trailing_stop_price == 0: trailing_stop_price = 999999.0
                        trailing_stop_price = min(trailing_stop_price, trail_price)

                if trailing_stop_active:
                    if (position == 1 and current_price < trailing_stop_price) or (position == -1 and current_price > trailing_stop_price):
                        exit_reason = f"跌破動態追蹤停損點 (觸發價: {trailing_stop_price:.2f})"

                if not exit_reason:
                    # 改用選擇權的真實快照高低價來衝擊權利金停損利
                    high_ret = (opt_high - entry_price) / entry_price if position == 1 else (entry_price - opt_low) / entry_price
                    low_ret = (opt_low - entry_price) / entry_price if position == 1 else (entry_price - opt_high) / entry_price

                    if low_ret <= STOP_LOSS_PCT:
                        exit_reason = f"觸及權利金硬停損限制 ({STOP_LOSS_PCT * 100}%)"
                    elif high_ret >= TAKE_PROFIT_PCT:
                        exit_reason = f"觸及權利金硬停利目標 (+{TAKE_PROFIT_PCT * 100}%)"

                # 執行平倉結算
                if exit_reason:
                    current_capital += current_pnl
                    last_trade_win = current_pnl > 0

                    trade_log.append({
                        'entry_time': entry_time,
                        'exit_time': now.strftime("%Y-%m-%d %H:%M:%S"),
                        'symbol': best_contract.symbol,
                        'direction': 'LONG' if position == 1 else 'SHORT',
                        'entry_price': entry_price,
                        'exit_price': current_price,
                        'pnl': current_pnl,
                        'ret': current_ret,
                        'reason': exit_reason
                    })

                    print(f"\n{'='*50}")
                    print(f"🔔 【即時平倉執行】: {exit_reason}")
                    print(f"合約: {best_contract.symbol} | 方向: {'LONG 做多' if position == 1 else 'SHORT 做空'}")
                    print(f"進場價: {entry_price:.2f} ➔ 出場價: {current_price:.2f}")
                    print(f"實際結算盈虧: NT$ {current_pnl:,.0f} ({current_ret * 100:.2f}%)")
                    print(f"最新模擬資金: NT$ {current_capital:,.0f}")
                    print(f"{'='*50}\n")

                    position = 0
                    num_contracts = 0
                    trade_capital_used = 0
                    trailing_stop_active = False
                    trailing_stop_price = 0.0

                    if "EOD" in exit_reason:
                        print("💤 已完成本日尾盤結算，暫停盤中交易，等待日終報告輸出...")
                        time.sleep(5)

            # 觸發日終報告 (當時間進入EOD範圍且目前持倉已被強制清空，且尚未生成過報告時)
            if is_eod_closing_time(current_time) and position == 0 and not eod_report_done:
                generate_eod_report(trade_log, INITIAL_CAPITAL, current_capital)
                eod_report_done = True

            # 訊號過濾與進場邏輯
            if position == 0 and not is_eod_closing_time(current_time):
                signal = strategy_engine.generate_signal(df, ai_score=probs, last_win=last_trade_win)

                if signal != 0:
                    position = signal
                    entry_price = current_price
                    entry_time = now.strftime("%Y-%m-%d %H:%M:%S")
                    trailing_stop_active = False
                    trailing_stop_price = 0.0

                    allocated_capital = min(current_capital * 0.50, MAX_POSITION_CAPITAL)
                    contract_cost = entry_price * CONTRACT_MULTIPLIER

                    num_contracts = int(allocated_capital // contract_cost) if contract_cost > 0 else 0
                    if num_contracts < 1:
                        num_contracts = 1

                    trade_capital_used = num_contracts * contract_cost
                    direction_label = "做多 (Buy Call) 🚀" if signal == 1 else "做空 (Buy Put) 📉"

                    print(f"\n{'='*50}")
                    print(f"🔥 【盤中進場訊號觸發】: AI 模型與籌碼技術面共振")
                    print(f"交易合約: {best_contract.symbol}")
                    print(f"操作方向: {direction_label}")
                    print(f"執行價格: {entry_price:.2f} | 預估交易口數: {num_contracts} 口")
                    print(f"建倉動用本金: NT$ {trade_capital_used:,.0f}")
                    print(f"{'='*50}\n")

        except Exception as e:
            print(f"[{time_str}] ❌ 盤中主迴圈執行發生異常錯誤: {e}")

        time.sleep(POLL_INTERVAL_SECONDS)

if __name__ == "__main__":
    run_live_simulator()