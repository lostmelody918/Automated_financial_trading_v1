import os
import time
import json
import torch
import numpy as np
import pandas as pd
from datetime import datetime, time as datetime_time
from data_engine import DayTradingDataEngine
from composite_ai import CompositeDayTradingAI
from model_manager import TradingModelManager
from strategy_factory import StrategyFactory

# ==========================================
# 核心參數設定 (對齊 42,442 點高基期與選擇權實務)
# ==========================================
INITIAL_CAPITAL = 150000        # 初始模擬本金 (因應高價權利金調高)
CONTRACT_MULTIPLIER = 50        # 台指選擇權合約乘數 (1點 = 50元)
FEE_SLIPPAGE_PER_CONTRACT = 70  # 單口交易成本 (手續費 + 滑價約 1 tick)
MAX_POSITION_CAPITAL = 4000000  # 最大部位限制
WINDOW_SIZE = 40                # AI 輸入的時間視窗長度

# 獲利與風險控制 (以權利金變動幅度 % 計算)
TAKE_PROFIT_PCT = 0.50          # 50% 硬停利
STOP_LOSS_PCT = -0.30           # 30% 硬停損
TRAILING_START_PCT = 0.20       # 獲利達 20% 時啟動追蹤停損
TRAILING_ATR_MULTIPLIER = 2.5   # 追蹤停損的 ATR 倍數

POLL_INTERVAL_SECONDS = 60      # 盤中輪詢間隔 (每 60 秒檢查一次最新 5分K)

def is_market_open(current_time):
    """判斷目前是否為選擇權日盤交易時間 (08:45 - 13:45)"""
    return datetime_time(8, 45) <= current_time <= datetime_time(13, 45)

def is_eod_closing_time(current_time):
    """判斷是否達到日内當沖強制平倉時間 (設定於收盤前 5 分鐘：13:40)"""
    return datetime_time(13, 40) <= current_time < datetime_time(13, 45)

def run_live_simulator():
    print("=" * 60)
    print("🚀 啟動獨立版：AI 選擇權盤中即時模擬交易系統 (Live Trader) 🚀")
    print(f"💵 初始模擬本金: NT$ {INITIAL_CAPITAL:,} | 選擇權乘數: {CONTRACT_MULTIPLIER}")
    print(f"⏱️ 監控時段: 日盤 08:45 ~ 13:45 (每 {POLL_INTERVAL_SECONDS} 秒輪詢)")
    print("=" * 60)

    # 1. 初始化資料引擎
    engine = DayTradingDataEngine()

    # 2. 載入正規化參數與特徵欄位
    norm_path = os.path.join(os.path.dirname(__file__), "saved_models", "norm_params.json")
    if not os.path.exists(norm_path):
        print("❌ 錯誤：找不到 saved_models/norm_params.json，請先執行 train_model.py 完成訓練。")
        return

    with open(norm_path, 'r', encoding='utf-8') as f:
        norm_params = json.load(f)

    feature_cols = norm_params['feature_cols']
    input_dim = len(feature_cols)
    mean_v = np.array([norm_params['mean'][c] for c in feature_cols])
    std_v = np.array([norm_params['std'][c] for c in feature_cols])

    # 3. 初始化並載入最新 AI 模型權重
    ai_model = CompositeDayTradingAI(input_dim=input_dim, d_model=128, nhead=8, num_layers=3)
    model_manager = TradingModelManager(model_dir=os.path.join(os.path.dirname(__file__), "saved_models"))
    ai_model, _, version = model_manager.load_latest_model(ai_model)
    ai_model.eval()
    print(f"📦 成功載入 AI 模型版本: v{version}")

    # 4. 載入決策策略工廠
    strategy_engine = StrategyFactory.get_strategy("composite")

    # 5. 即時部位與狀態管理變數 (常駐於記憶體)
    current_capital = INITIAL_CAPITAL
    position = 0             # 0: 空手, 1: 做多買方, -1: 做空買方 (或依策略定義)
    entry_price = 0.0
    num_contracts = 0
    trade_capital_used = 0
    last_trade_win = False

    trailing_stop_active = False
    trailing_stop_price = 0.0

    print("📡 系統初始化完成，正在接入 Shioaji API 進入即時盤中監控...")

    # 6. 即時事件輪詢迴圈 (Event Loop)
    while True:
        now = datetime.now()
        current_time = now.time()
        time_str = now.strftime('%H:%M:%S')

        # A. 非交易時間守衛
        if not is_market_open(current_time):
            if position != 0:
                print(f"[{time_str}] ⚠️ 異常警訊：非交易時間仍持有虛擬部位，執行系統強制清倉。")
                position = 0
                num_contracts = 0
                trade_capital_used = 0
            print(f"[{time_str}] 💤 目前為非交易時段，系統休眠中。等待日盤開盤 (08:45)...")
            time.sleep(60)
            continue

        try:
            # B. 抓取盤中即時資料 (向 Shioaji 請求近 5 天 K 線以確保長週期指標如 100MA、ATR 運算精準)
            df, best_contract = engine.fetch_active_option_intraday_data(days=5)

            if df is None or df.empty or len(df) < WINDOW_SIZE:
                print(f"[{time_str}] ⚠️ 盤中數據庫更新中或資料不足，等待下一輪輪詢...")
                time.sleep(POLL_INTERVAL_SECONDS)
                continue

            last_row = df.iloc[-2]     # 已收定的前一根 K 線 (用於技術指標訊號評估)
            latest_row = df.iloc[-1]   # 盤中最新跳動的 K 線 (提供當前市場最新即時報價)
            current_price = latest_row['Close']

            # C. 擷取時序視窗並進行 AI 特徵正規化
            feat_data = df[feature_cols].tail(WINDOW_SIZE).values
            feat_normalized = (feat_data - mean_v) / std_v
            feat_tensor = torch.tensor(feat_normalized, dtype=torch.float32).unsqueeze(0)

            # D. CNN-Transformer 複合推理
            with torch.no_grad():
                logits = ai_model(feat_tensor)
                probs = torch.softmax(logits, dim=1).squeeze().cpu().numpy()

            status_mark = "🟢 持倉中" if position != 0 else "⚪ 空手"
            print(f"[{time_str}] 標的: {best_contract.symbol} | 即時價: {current_price} | AI預測 [跌:{probs[0]:.2f} 平:{probs[1]:.2f} 漲:{probs[2]:.2f}] | 狀態: {status_mark}")

            # E. 即時風控與平倉邏輯
            if position != 0:
                # 計算當前即時點數損益與真實淨損益 (扣除滑價與手續費)
                points_gained = (current_price - entry_price) if position == 1 else (entry_price - current_price)
                current_pnl = (points_gained * CONTRACT_MULTIPLIER * num_contracts) - (num_contracts * FEE_SLIPPAGE_PER_CONTRACT)
                current_ret = current_pnl / trade_capital_used if trade_capital_used > 0 else 0

                pnl_display = f"NT$ +{current_pnl:,.0f}" if current_pnl > 0 else f"NT$ {current_pnl:,.0f}"
                print(f"   ↳ 帳面未實現損益: {pnl_display} (報酬率: {current_ret * 100:.2f}%)")

                exit_reason = None

                # 1. 檢查是否觸發尾盤強制當沖平倉
                if is_eod_closing_time(current_time):
                    exit_reason = "日内時間截止強制平倉 (EOD)"

                # 2. 檢查與更新 ATR 追蹤停損邏輯
                current_atr = last_row['atr']
                if current_ret >= TRAILING_START_PCT:
                    if not trailing_stop_active:
                        trailing_stop_active = True
                        print(f"   ↳ 🔥 獲利達標，啟動 {TRAILING_ATR_MULTIPLIER}xATR 動態追蹤停利機制。")

                    # 計算動態屏障價
                    trail_price = current_price - (current_atr * TRAILING_ATR_MULTIPLIER) if position == 1 else current_price + (current_atr * TRAILING_ATR_MULTIPLIER)

                    if position == 1:
                        trailing_stop_price = max(trailing_stop_price, trail_price)
                    else:
                        if trailing_stop_price == 0: trailing_stop_price = 999999.0
                        trailing_stop_price = min(trailing_stop_price, trail_price)

                # 觸發追蹤停損
                if trailing_stop_active:
                    if (position == 1 and current_price < trailing_stop_price) or (position == -1 and current_price > trailing_stop_price):
                        exit_reason = f"動態追蹤停損點跌破 (觸發價: {trailing_stop_price:.2f})"

                # 3. 檢查硬停損與硬停利 (雙重防護層)
                if not exit_reason:
                    # 評估當根 K 線極端價格對權利金的衝擊
                    high_ret = (latest_row['High'] - entry_price) / entry_price if position == 1 else (entry_price - latest_row['Low']) / entry_price
                    low_ret = (latest_row['Low'] - entry_price) / entry_price if position == 1 else (entry_price - latest_row['High']) / entry_price

                    if low_ret <= STOP_LOSS_PCT:
                        exit_reason = f"權利金觸及硬停損限制 ({STOP_LOSS_PCT * 100}%)"
                    elif high_ret >= TAKE_PROFIT_PCT:
                        exit_reason = f"權利金觸及硬停利目標 (+{TAKE_PROFIT_PCT * 100}%)"

                # 執行虛擬平倉結算
                if exit_reason:
                    current_capital += current_pnl
                    last_trade_win = current_pnl > 0

                    print("" + "="*50)
                    print(f"🔔 【即時平倉執行】: {exit_reason}")
                    print(f"合約: {best_contract.symbol} | 方向: {'LONG 做多' if position == 1 else 'SHORT 做空'}")
                    print(f"進場價: {entry_price:.2f} ➔ 出場價: {current_price:.2f}")
                    print(f"實際結算盈虧: NT$ {current_pnl:,.0f} ({current_ret * 100:.2f}%)")
                    print(f"帳戶最新模擬可用資金: NT$ {current_capital:,.0f}")
                    print("="*50 + "")

                    # 徹底清空單筆部位狀態狀態
                    position = 0
                    num_contracts = 0
                    trade_capital_used = 0
                    trailing_stop_active = False
                    trailing_stop_price = 0.0

                    # 若為尾盤強制平倉，則直接讓系統休息至收盤
                    if "EOD" in exit_reason:
                        print("💤 已完成本日尾盤結算，暫停盤中交易，等待明日開盤...")
                        time.sleep(300)

            # F. 訊號過濾與進場邏輯 (僅在空手狀態且非尾盤時允許進場)
            if position == 0 and not is_eod_closing_time(current_time):
                # 將當前完整的資料流與 AI 推出的勝率機率傳入決策工廠
                signal = strategy_engine.generate_signal(df, ai_score=probs, last_win=last_trade_win)

                if signal != 0:
                    position = signal
                    entry_price = current_price
                    trailing_stop_active = False
                    trailing_stop_price = 0.0

                    # 實務動態資金控管：最多動用當前模擬總資產的 50%
                    allocated_capital = min(current_capital * 0.50, MAX_POSITION_CAPITAL)
                    contract_cost = entry_price * CONTRACT_MULTIPLIER

                    if contract_cost > 0:
                        num_contracts = int(allocated_capital // contract_cost)
                    else:
                        num_contracts = 0

                    if num_contracts < 1:
                        num_contracts = 1 # 保底機制：資金規模較小時至少買進 1 口

                    trade_capital_used = num_contracts * contract_cost
                    direction_label = "做多 (Buy Call) 🚀" if signal == 1 else "做空 (Buy Put) 📉"

                    print("" + "="*50)
                    print(f"🔥 【🔥 盤中進場訊號觸發】: AI 模型與技術面共振")
                    print(f"交易合約: {best_contract.symbol}")
                    print(f"操作方向: {direction_label}")
                    print(f"執行價格: {entry_price:.2f} | 預估交易口數: {num_contracts} 口")
                    print(f"建倉動用本金: NT$ {trade_capital_used:,.0f}")
                    print("="*50 + "")

        except Exception as e:
            print(f"[{time_str}] ❌ 盤中即時主迴圈執行發生異常錯誤: {e}")

        # G. 執行間隔控制
        time.sleep(POLL_INTERVAL_SECONDS)

if __name__ == "__main__":
    run_live_simulator()
