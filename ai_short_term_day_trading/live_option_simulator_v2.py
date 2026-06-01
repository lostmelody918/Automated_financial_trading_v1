import os
import time
import json
import torch
import numpy as np
import pandas as pd
pd.options.mode.string_storage = 'python' # Disable pyarrow to prevent Shioaji thread crash
import pyarrow # Pre-load pyarrow to prevent access violation with Shioaji threads
from datetime import datetime, timedelta, time as datetime_time
import requests
import io
import re
import shioaji as sj

from data_engine import DayTradingDataEngine
from composite_ai import CompositeDayTradingAI
from model_manager import TradingModelManager
from strategy_factory import StrategyFactory

# ==========================================
# 核心參數設定
# ==========================================
INITIAL_CAPITAL = 120000
CONTRACT_MULTIPLIER = 50
FEE_SLIPPAGE_PER_CONTRACT = 100
MAX_POSITION_CAPITAL = 4000000
WINDOW_SIZE = 40

# 損益參數在下方(多策略)

POLL_INTERVAL_SECONDS = 5

def is_market_open(current_time):
    return datetime_time(8, 45) <= current_time <= datetime_time(13, 45)

def is_eod_closing_time(current_time):
    return datetime_time(13, 40) <= current_time < datetime_time(13, 45)

def load_latest_daily_chips_snapshot():
    print("📥 正在向期交所連線獲取最新籌碼快照 (含外資/投信/自營/PC)...")

    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
        'Accept': 'text/csv,text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
        'Accept-Language': 'zh-TW,zh;q=0.9,en-US;q=0.8,en;q=0.7',
        'Connection': 'keep-alive',
        'Content-Type': 'application/x-www-form-urlencoded',
        'Origin': 'https://www.taifex.com.tw',
        'Referer': 'https://www.taifex.com.tw/cht/3/futContractsDate'
    }

    end_date = datetime.now()
    start_date = end_date - timedelta(days=7)
    s_dt = start_date.strftime("%Y/%m/%d")
    e_dt = end_date.strftime("%Y/%m/%d")

    try:
        session = requests.Session()
        session.headers.update(headers)

        res_pc = session.post("https://www.taifex.com.tw/cht/3/pcRatioDown",
                               data={"queryStartDate": s_dt, "queryEndDate": e_dt},
                               timeout=10)

        if res_pc.status_code != 200: raise ValueError("P/C Ratio 連線失敗")
        content_pc = res_pc.content.decode('cp950', errors='ignore')

        res_oi = session.post("https://www.taifex.com.tw/cht/3/futContractsDateDown",
                               data={
                                   "queryStartDate": s_dt,
                                   "queryEndDate": e_dt,
                                   "commodityId": "TXF"
                               },
                               timeout=10)

        if res_oi.status_code != 200:
            content_oi = ""
        else:
            content_oi = res_oi.content.decode('cp950', errors='ignore')
            if '<html' in content_oi.lower():
                content_oi = ""

        valid_pc_lines = [line for line in content_pc.split('\n') if re.match(r'^20\d{2}/\d{2}/\d{2}', line.strip())]
        if not valid_pc_lines: raise ValueError("找不到 P/C Ratio 數據格式")

        latest_date_str = ""
        pc_ratio = 0.0
        foreign_net_oi = 0.0
        dealer_net_oi = 0.0
        trust_net_oi = 0.0
        found_data = False

        for pc_line in valid_pc_lines:
            line_pc_clean = pc_line.replace('"', '').replace(' ', '')
            parts_pc = line_pc_clean.split(',')
            current_date = parts_pc[0]

            current_content_oi = content_oi
            if not current_content_oi or current_date not in current_content_oi:
                res_single = session.post("https://www.taifex.com.tw/cht/3/futContractsDateDown",
                                         data={
                                             "queryStartDate": current_date,
                                             "queryEndDate": current_date,
                                             "commodityId": "TXF"
                                         },
                                         timeout=10)
                if res_single.status_code == 200:
                    current_content_oi = res_single.content.decode('cp950', errors='ignore')
                    if '<html' in current_content_oi.lower(): current_content_oi = ""
                else:
                    continue

            if '臺股期貨' not in current_content_oi and '台股期貨' not in current_content_oi:
                continue
            if current_date not in current_content_oi:
                continue

            try:
                pc_ratio = float(parts_pc[-1].replace('%', '')) / 100.0
            except (ValueError, IndexError):
                pc_ratio = float(parts_pc[-2].replace('%', '')) / 100.0

            latest_date_str = current_date

            valid_oi_lines = [line.replace('"', '').replace(' ', '') for line in current_content_oi.split('\n')
                              if current_date in line and ('臺股期貨' in line or '台股期貨' in line)]

            for line in valid_oi_lines:
                parts = line.split(',')
                identity = ""
                if '外資' in line: identity = "外資"
                elif '自營商' in line: identity = "自營商"
                elif '投信' in line: identity = "投信"
                else: continue

                numeric_values = [float(p.strip()) for p in parts if re.match(r'^-?\d+$', p.strip())]
                if len(numeric_values) >= 3:
                    net_oi = numeric_values[-2]
                    if identity == "外資": foreign_net_oi = net_oi
                    elif identity == "自營商": dealer_net_oi = net_oi
                    elif identity == "投信": trust_net_oi = net_oi

            found_data = True
            break

        if not found_data:
            raise ValueError("找不到三大法人數據格式")

        parsed_date = datetime.strptime(latest_date_str, "%Y/%m/%d").date()
        print(f"✅ 成功獲取 {parsed_date} 籌碼 -> 外資: {foreign_net_oi:,.0f} | 投信: {trust_net_oi:,.0f} | 自營: {dealer_net_oi:,.0f} | P/C: {pc_ratio:.2f}")

        return pd.DataFrame({
            'date': [pd.to_datetime(parsed_date)],
            'foreign_net_oi': [float(foreign_net_oi)],
            'dealer_net_oi': [float(dealer_net_oi)],
            'trust_net_oi': [float(trust_net_oi)],
            'pc_ratio': [float(pc_ratio)]
        })

    except Exception as e:
        print(f"❌ 即時快照抓取失敗: {e}，啟用安全備用數據...")
        # 備用數據也加上投信
        return pd.DataFrame({
            'date': [pd.to_datetime(datetime.now() - timedelta(days=1))],
            'foreign_net_oi': [-550173.0],
            'dealer_net_oi': [-366972.0],
            'trust_net_oi': [85797.0],
            'pc_ratio': [1267500.0/436152.0]
        })

def generate_eod_report(trade_log, initial_capital, current_capital, out_dir="data_learn"):
    if not os.path.exists(out_dir): os.makedirs(out_dir)
    print("\n" + "="*60 + f"\n📊  --- 今日選擇權即時模擬當沖結算報告 ---\n" + "="*60)
    total_trades = len(trade_log)
    total_pnl = current_capital - initial_capital
    print(f"💰 初始本金 : NT$ {initial_capital:,}\n💵 結算淨值 : NT$ {current_capital:,}\n📈 總淨盈虧 : NT$ {total_pnl:,.0f} ({(total_pnl / initial_capital) * 100:.2f}%)\n" + "-" * 50)
    if total_trades > 0:
        df_trades = pd.DataFrame(trade_log)
        wins = df_trades[df_trades['pnl'] > 0]
        losses = df_trades[df_trades['pnl'] <= 0]
        print(f"⏱️ 總交易次數 : {total_trades} 次 | 🎯 當日勝率 : {(len(wins) / total_trades) * 100:.2f}%")
        csv_path = os.path.join(out_dir, f"daily_trade_report_{datetime.now().strftime('%Y%m%d')}.csv")
        df_trades.to_csv(csv_path, index=False, encoding="utf-8-sig")
        print(f"💾 已儲存今日交易明細至: {csv_path}")
    else: print("❌ 今日無交易紀錄。")
    print("="*60 + "\n")

def calculate_features(df_raw):
    """
    在本地對原始 K 線進行特徵工程
    """
    df = df_raw.copy()
    if df.empty: return df

    df['time'] = df['date'].dt.time
    df['date_only'] = df['date'].dt.date
    df['day_of_week'] = df['date'].dt.dayofweek

    df['ret'] = df['Close'].pct_change()
    df['mock_volume'] = df['Volume'].replace(0, 1)
    df['vol_price'] = (df['High'] + df['Low'] + df['Close']) / 3 * df['mock_volume']

    df['cum_vol_price'] = df.groupby('date_only')['vol_price'].cumsum()
    df['cum_vol'] = df.groupby('date_only')['mock_volume'].cumsum()
    df['vwap'] = df['cum_vol_price'] / df['cum_vol']
    df['vwap_bias'] = (df['Close'] - df['vwap']) / (df['vwap'] + 1e-9)

    exp1 = df['Close'].ewm(span=12, adjust=False).mean()
    exp2 = df['Close'].ewm(span=26, adjust=False).mean()
    df['macd'] = exp1 - exp2
    df['signal'] = df['macd'].ewm(span=9, adjust=False).mean()
    df['macd_hist'] = df['macd'] - df['signal']

    df['h_l'] = df['High'] - df['Low']
    df['h_pc'] = abs(df['High'] - df['Close'].shift(1))
    df['l_pc'] = abs(df['Low'] - df['Close'].shift(1))
    df['tr'] = df[['h_l', 'h_pc', 'l_pc']].max(axis=1)
    df['atr'] = df['tr'].rolling(14).mean()

    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / (loss + 1e-9)
    df['rsi'] = 100 - (100 / (1 + rs))

    df['sma20'] = df['Close'].rolling(window=20).mean()
    df['std20'] = df['Close'].rolling(window=20).std()
    df['bb_upper'] = df['sma20'] + (df['std20'] * 2)
    df['bb_lower'] = df['sma20'] - (df['std20'] * 2)
    df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / (df['sma20'] + 1e-9)
    df['is_squeeze'] = (df['bb_width'] < df['bb_width'].rolling(20).mean()).astype(int)

    df['vwap_5'] = df['mock_volume'] * df['Close'] / df['mock_volume'].rolling(5).sum()
    df['slope_vwap'] = (df['vwap_5'] - df['vwap_5'].shift(3)) / df['vwap_5'].shift(3) * 10000

    df['ma_20'] = df['Close'].rolling(20).mean()
    df['slope_ma20'] = (df['ma_20'] - df['ma_20'].shift(3)) / df['ma_20'].shift(3) * 10000

    df['vol_surge_ratio'] = df['mock_volume'] / (df['mock_volume'].rolling(20).mean() + 1e-9)
    df['price_roc'] = df['Close'].pct_change()
    df['pv_divergence'] = np.where((df['price_roc'] > 0) & (df['vol_surge_ratio'] < 0.8), -1,
                        np.where((df['price_roc'] < 0) & (df['vol_surge_ratio'] < 0.8), 1, 0))

    cols_to_drop = ['vol_price', 'cum_vol_price', 'cum_vol', 'h_l', 'h_pc', 'l_pc', 'tr', 'sma20', 'std20', 'Amount']
    df.drop(columns=[c for c in cols_to_drop if c in df.columns], inplace=True)
    return df.dropna().reset_index(drop=True)

def run_live_simulator():
    print("=" * 60 + "\n🚀 啟動：底層期貨特徵分離版 AI 選擇權即時模擬機\n" + "=" * 60)
    engine = DayTradingDataEngine()

    with open(os.path.join(os.path.dirname(__file__), "saved_models", "norm_params.json"), 'r', encoding='utf-8') as f:
        norm_params = json.load(f)

    feature_cols = [c for c in norm_params['feature_cols'] if c in norm_params['mean']]
    mean_v = np.array([norm_params['mean'][c] for c in feature_cols])
    std_v = np.array([norm_params['std'][c] for c in feature_cols])

    ai_model = CompositeDayTradingAI(input_dim=len(feature_cols), d_model=256, nhead=16, num_layers=4)
    model_manager = TradingModelManager(model_dir=os.path.join(os.path.dirname(__file__), "saved_models"))
    ai_model, _, version = model_manager.load_latest_model(ai_model)
    ai_model.eval()

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
    highest_price_since_entry = 0.0
    trade_log = []
    eod_report_done = False

    active_contract = None # 記錄目前持倉的具體合約 (Call 或 Put)
    entry_features = {} # 記錄進場時的特徵

    df_chips_daily = load_latest_daily_chips_snapshot()

    # === 步驟 1：取得初始 5 天 K 線與基礎設定 ===
    print("📥 正在獲取初始 5 天 K 線歷史資料...")
    today = datetime.now()
    start_date = (today - timedelta(days=5)).strftime("%Y-%m-%d")
    end_date = today.strftime("%Y-%m-%d")

    txf_contract = engine.api.Contracts.Futures.TXF.TXFR1
    kbars = engine.api.kbars(txf_contract, start=start_date, end=end_date)
    df_raw = pd.DataFrame({**kbars})
    df_raw['ts'] = pd.to_datetime(df_raw['ts'])
    df_raw = df_raw.sort_values('ts').reset_index(drop=True)
    df_raw.rename(columns={'ts': 'date'}, inplace=True)
    print(f"✅ 成功載入 {len(df_raw)} 筆初始 K 線！")

    last_total_volume = 0
    if not df_raw.empty:
        try:
            initial_snap = engine.api.snapshots([txf_contract])[0]
            last_total_volume = initial_snap.total_volume
        except:
            pass

    # === 即時報價 Quote Subscribe 變數與回調 ===
    realtime_quotes = {}

    def quote_callback(topic, quote):
        try:
            symbol = topic.split('/')[-1]
            realtime_quotes[symbol] = {
                'close': quote.Close if hasattr(quote, 'Close') else quote.close,
                'high': quote.High if hasattr(quote, 'High') else quote.high,
                'low': quote.Low if hasattr(quote, 'Low') else quote.low
            }
        except:
            pass

    try:
        engine.api.quote.set_on_tick_fnc(quote_callback)
    except AttributeError:
        pass

    while True:
        now = datetime.now()
        current_time = now.time()
        time_str = now.strftime('%H:%M:%S')

        if not is_market_open(current_time):
            if position != 0: position, num_contracts, trade_capital_used, active_contract = 0, 0, 0, None
            if eod_report_done and current_time < datetime_time(8, 45): eod_report_done, trade_log = False, []
            print(f"[{time_str}] 💤 非交易時段，等待日盤開盤...")
            time.sleep(60)
            continue

        if is_eod_closing_time(current_time) and position == 0 and not eod_report_done:
            generate_eod_report(trade_log, INITIAL_CAPITAL, current_capital)
            eod_report_done = True
            time.sleep(300)
            continue

        try:
            # === 本地數據拼裝 (用 snapshots 更新最後一筆 K 線) ===
            snap = engine.api.snapshots([txf_contract])[0]
            current_close = snap.close
            current_total_volume = snap.total_volume

            now_min = now.replace(second=0, microsecond=0)

            if not df_raw.empty:
                last_idx = df_raw.index[-1]
                last_time = df_raw.at[last_idx, 'date']

                added_volume = max(0, current_total_volume - last_total_volume)
                last_total_volume = current_total_volume

                if now_min == last_time:
                    df_raw.at[last_idx, 'Close'] = current_close
                    df_raw.at[last_idx, 'High'] = max(df_raw.at[last_idx, 'High'], current_close)
                    df_raw.at[last_idx, 'Low'] = min(df_raw.at[last_idx, 'Low'], current_close)
                    df_raw.at[last_idx, 'Volume'] += added_volume
                elif now_min > last_time:
                    new_row = pd.DataFrame([{
                        'date': now_min,
                        'Open': current_close,
                        'High': current_close,
                        'Low': current_close,
                        'Close': current_close,
                        'Volume': added_volume,
                        'Amount': 0
                    }])
                    df_raw = pd.concat([df_raw, new_row], ignore_index=True)

            # === 本地計算特徵 ===
            df_intraday = calculate_features(df_raw)
            if df_intraday is None or df_intraday.empty:
                time.sleep(POLL_INTERVAL_SECONDS)
                continue

            df = engine.integrate_institutional_chips(df_intraday, df_chips_daily) if hasattr(engine, 'integrate_institutional_chips') else df_intraday

            if df is None or df.empty or len(df) < WINDOW_SIZE:
                time.sleep(POLL_INTERVAL_SECONDS)
                continue

            df_slice = df[feature_cols].tail(WINDOW_SIZE).copy()
            for col in ['mock_volume', 'macd_hist', 'vwap_bias']:
                if col in df_slice.columns: df_slice[col] = np.sign(df_slice[col]) * np.log1p(np.abs(df_slice[col]))

            feat_tensor = torch.tensor(np.nan_to_num((df_slice.values - mean_v) / np.where(std_v == 0, 1.0, std_v), nan=0.0), dtype=torch.float32).unsqueeze(0)

            with torch.no_grad():
                probs = torch.softmax(ai_model(feat_tensor), dim=1).squeeze().cpu().numpy()

            # --- 狀態印出與風控 (當有持倉時) ---
            if position != 0 and active_contract:
                try:
                    sym = active_contract.symbol
                    if sym in realtime_quotes:
                        current_price = realtime_quotes[sym]['close']
                        day_high = realtime_quotes[sym]['high']
                        day_low = realtime_quotes[sym]['low']
                    else:
                        snapshot = engine.api.snapshots([active_contract])[0]
                        current_price = snapshot.close
                        day_high = snapshot.high
                        day_low = snapshot.low

                    # 更新進場後的最高價
                    if current_price > highest_price_since_entry:
                        highest_price_since_entry = current_price
                except:
                    time.sleep(POLL_INTERVAL_SECONDS)
                    continue

                print(f"[{time_str}] 標的: {active_contract.symbol} | 買入價: {entry_price} | 權利金: {current_price} | 當日最高: {day_high} | 當日最低: {day_low} | 進場後最高價: {highest_price_since_entry} | AI預測 [跌:{probs[0]:.2f} 漲:{probs[2]:.2f}] | 🟢 持倉")

                # 選擇權買方：不論 Call/Put，看對方向權利金都會漲，公式永遠是 (現價 - 進場價)
                points_gained = current_price - entry_price
                current_pnl = (points_gained * CONTRACT_MULTIPLIER * num_contracts) - (num_contracts * FEE_SLIPPAGE_PER_CONTRACT)
                current_ret = current_pnl / trade_capital_used if trade_capital_used > 0 else 0

                print(f"   ↳ 帳面損益: NT$ {'+' if current_pnl>0 else ''}{current_pnl:,.0f} ({current_ret * 100:.2f}%)")
                exit_reason = None

                if is_eod_closing_time(current_time): exit_reason = "尾盤強制平倉 (EOD)"

                # 若獲利達標，啟動動態百分比追蹤停利
                if current_ret >= current_trade_trail_start:
                    if not trailing_stop_active:
                        trailing_stop_active = True
                        print(f"   ↳ 🔥 獲利達標，啟動 {current_trade_trail_retrace*100}% 動態追蹤停利。")

                    # 買方策略：取「歷史最高價」作為基準，跌破一定比例則停利
                    trail_price = highest_price_since_entry * (1 - current_trade_trail_retrace)
                    trailing_stop_price = max(trailing_stop_price, trail_price)

                # [🎯 標註：執行停損停利觸發]
                if not exit_reason:
                    # 改用當前價格判斷是否停損與停利
                    current_price_ret = (current_price - entry_price) / entry_price

                    if current_price_ret <= current_trade_stop_loss:
                        exit_reason = f"觸及硬停損 ({current_trade_stop_loss * 100}%)"
                    elif current_price_ret >= current_trade_take_profit:
                        exit_reason = f"觸及硬停利 (+{current_trade_take_profit * 100}%)"
                    elif trailing_stop_active and current_price < trailing_stop_price:
                        exit_reason = f"跌破動態停利線 (觸發價: {trailing_stop_price:.2f})"

                if exit_reason:
                    current_capital += current_pnl
                    last_trade_win = current_pnl > 0

                    trade_record = {
                        'entry_time': entry_time,
                        'exit_time': now.strftime("%H:%M:%S"),
                        'symbol': active_contract.symbol,
                        'direction': 'Call' if position==1 else 'Put',
                        'entry_price': entry_price,
                        'exit_price': current_price,
                        'pnl': current_pnl,
                        'ret': current_ret,
                        'reason': exit_reason
                    }
                    if entry_features:
                        # 將特徵加入交易紀錄中
                        trade_record.update(entry_features)

                    trade_log.append(trade_record)

                    print(f"\n{'='*40}\n🔔 【平倉】: {exit_reason}\n實際盈虧: NT$ {current_pnl:,.0f} ({current_ret * 100:.2f}%)\n最新資金: NT$ {current_capital:,.0f}\n{'='*40}\n")
                    position, num_contracts, trade_capital_used, trailing_stop_active, trailing_stop_price, highest_price_since_entry, active_contract = 0, 0, 0, False, 0.0, 0.0, None
                    entry_features = {} # 清空特徵

            # --- 訊號過濾與進場 (當空手時) ---
            else:
                if not is_eod_closing_time(current_time):
                    signal = strategy_engine.generate_signal(df, ai_score=probs, last_win=last_trade_win)

                    if signal != 0:
                        # 決定方向
                        opt_type = 'Call' if signal > 0 else 'Put'
                        allocated_capital_limit = min(current_capital * 0.50, MAX_POSITION_CAPITAL)

                        active_contract = engine.get_best_volume_option_contract(option_type=opt_type, allocated_capital=allocated_capital_limit)
                        try:
                            entry_price = engine.api.snapshots([active_contract])[0].close
                        except:
                            active_contract = None; continue

                        # 記錄進場狀態
                        position = 1 if signal > 0 else -1
                        entry_time = now.strftime("%Y-%m-%d %H:%M:%S")
                        trailing_stop_active, trailing_stop_price = False, 0.0
                        highest_price_since_entry = entry_price
                        entry_features = {f"feat_{k}": v for k, v in df.iloc[-1].to_dict().items()}

                        # [🎯 三層級風控動態設定]
                        abs_sig = abs(signal)
                        if abs_sig == 3:
                            # Level 3：極強勢大波段 (抱最緊，看最遠)
                            current_trade_take_profit = 5.50   # 賺 550%
                            current_trade_stop_loss = -0.35    # 扛 35% 回檔
                            current_trade_trail_start = 0.50   # 賺 50% 才啟動追蹤
                            current_trade_trail_retrace = 0.3 # 允許 30% 的大回撤
                            strategy_label = f"🚀 Level 3 極強勢波段 (Buy {opt_type})"

                        elif abs_sig == 2:
                            # Level 2：標準波段 (穩健獲利)
                            current_trade_take_profit = 1.50   # 賺 150%
                            current_trade_stop_loss = -0.20    # 虧 20% 停損
                            current_trade_trail_start = 0.30   # 賺 30% 啟動追蹤
                            current_trade_trail_retrace = 0.2 # 允許 20% 回撤
                            strategy_label = f"📈 Level 2 標準波段 (Buy {opt_type})"

                        else:
                            # Level 1：短線打帶跑 (快進快出)
                            current_trade_take_profit = 0.5   # 賺 50% 就硬停利入袋
                            current_trade_stop_loss = -0.1    # 虧 10% 就無情砍倉
                            current_trade_trail_start = 0.10   # 賺 10% 就啟動保本
                            current_trade_trail_retrace = 0.1 # 回檔 10% 就跑
                            strategy_label = f"⚡ Level 1 短線游擊 (Buy {opt_type})"

                        allocated_capital = allocated_capital_limit
                        num_contracts = max(1, int(allocated_capital // (entry_price * CONTRACT_MULTIPLIER))) if entry_price > 0 else 1
                        trade_capital_used = num_contracts * entry_price * CONTRACT_MULTIPLIER

                        # 訂閱合約的即時報價
                        try:
                            engine.api.quote.subscribe(active_contract, quote_type=sj.constant.QuoteType.Tick)
                        except:
                            pass

                        print(f"\n{'='*40}\n🔥 【盤中進場】: {strategy_label}\n合約: {active_contract.symbol}\n進場價: {entry_price:.2f} | {num_contracts} 口\n{'='*40}\n")

        except Exception as e:
            print(f"[{time_str}] ❌ 異常錯誤: {e}")
        time.sleep(POLL_INTERVAL_SECONDS)

if __name__ == "__main__":
    run_live_simulator()
