import os
import time
import json
import torch
import numpy as np
import pandas as pd
import queue
import sqlite3
import threading

pd.options.mode.string_storage = 'python' # Disable pyarrow to prevent Shioaji thread crash
import pyarrow # Pre-load pyarrow to prevent access violation with Shioaji threads
from datetime import datetime, timedelta, time as datetime_time

import shioaji as sj
from datetime import datetime

from data_engine import DayTradingDataEngine
from composite_ai import CompositeDayTradingAI
from model_manager import TradingModelManager
from strategy_factory import StrategyFactory
from delta_gamma_theta import get_dynamic_bsm_bounds
from get_api_based_dte import get_api_based_dte

# ==========================================
# 核心參數設定
# ==========================================
INITIAL_CAPITAL = 120000
CONTRACT_MULTIPLIER = 50
FEE_SLIPPAGE_PER_CONTRACT = 100
MAX_POSITION_CAPITAL = 4000000
WINDOW_SIZE = 40
BID_ASK_SPREAD_THRESHOLD = 5.0

class PositionManager:
    def __init__(self, db_path="position_state.db"):
        self.db_path = db_path
        self._init_db()
        self.state = self._load_state()

    def _init_db(self):
        with sqlite3.connect(self.db_path) as conn:
            # 開啟 WAL 模式提升寫入效能，避免主迴圈卡頓
            conn.execute("PRAGMA journal_mode=WAL;")
            conn.execute('''
                CREATE TABLE IF NOT EXISTS position_state (
                    id INTEGER PRIMARY KEY,
                    position INTEGER,
                    entry_price REAL,
                    num_contracts INTEGER,
                    highest_price_since_entry REAL,
                    active_contract_symbol TEXT,
                    entry_time TEXT,
                    trade_capital_used REAL,
                    hard_tp_price REAL,
                    hard_sl_price REAL,
                    strategy_label TEXT
                )
            ''')
            cur = conn.execute("SELECT id FROM position_state WHERE id=1")
            if not cur.fetchone():
                conn.execute('''
                    INSERT INTO position_state (
                        id, position, entry_price, num_contracts,
                        highest_price_since_entry, active_contract_symbol,
                        entry_time, trade_capital_used, hard_tp_price, hard_sl_price, strategy_label
                    ) VALUES (1, 0, 0.0, 0, 0.0, NULL, NULL, 0.0, 0.0, 0.0, NULL)
                ''')

    def _load_state(self):
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            row = conn.execute("SELECT * FROM position_state WHERE id=1").fetchone()
            return dict(row)

    def update(self, **kwargs):
        for k, v in kwargs.items():
            self.state[k] = v
        with sqlite3.connect(self.db_path) as conn:
            set_clause = ", ".join([f"{k}=?" for k in kwargs.keys()])
            values = list(kwargs.values()) + [1]
            conn.execute(f"UPDATE position_state SET {set_clause} WHERE id=?", values)

    def clear_position(self):
        self.update(
            position=0, entry_price=0.0, num_contracts=0,
            highest_price_since_entry=0.0, active_contract_symbol=None,
            entry_time=None, trade_capital_used=0.0, hard_tp_price=0.0, hard_sl_price=0.0,
            strategy_label=None
        )

    def get(self, key, default=None):
        return self.state.get(key, default)


class KBarAccumulator:
    def __init__(self):
        self.current_min = None
        self.O = 0
        self.H = 0
        self.L = 0
        self.C = 0
        self.V = 0

    def on_tick(self, timestamp, price, volume):
        tick_min = timestamp.replace(second=0, microsecond=0)
        if self.current_min is None:
            self.current_min = tick_min
            self.O = self.H = self.L = self.C = price
            self.V = volume
            return None

        if tick_min > self.current_min:
            finished_bar = {
                'date': self.current_min,
                'Open': self.O,
                'High': self.H,
                'Low': self.L,
                'Close': self.C,
                'Volume': self.V,
                'Amount': 0
            }
            self.current_min = tick_min
            self.O = self.H = self.L = self.C = price
            self.V = volume
            return finished_bar
        else:
            self.H = max(self.H, price)
            self.L = min(self.L, price)
            self.C = price
            self.V += volume
            return None

def is_market_open(current_time):
    return datetime_time(8, 45) <= current_time <= datetime_time(13, 45)

def is_eod_closing_time(current_time):
    return datetime_time(13, 40) <= current_time < datetime_time(13, 45)

def load_latest_daily_chips_snapshot():
    print("📥 載入本地籌碼快照 (chips_cache.json)...")
    cache_path = os.path.join(os.path.dirname(__file__), "chips_cache.json")
    if os.path.exists(cache_path):
        try:
            with open(cache_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            df_chips = pd.DataFrame(data)
            df_chips['date'] = pd.to_datetime(df_chips['date'])
            print(f"✅ 成功載入 {len(df_chips)} 天籌碼歷史。")
            return df_chips
        except Exception as e:
            print(f"⚠️ 載入 chips_cache.json 失敗: {e}")

    return pd.DataFrame({
        'date': [pd.to_datetime(datetime.now().date() - timedelta(days=1))],
        'foreign_net_oi': [0.0],
        'dealer_net_oi': [0.0],
        'trust_net_oi': [0.0],
        'pc_ratio': [1.0]
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
    print("=" * 60 + "\n🚀 啟動：底層期貨特徵分離版 AI 選擇權即時模擬機 (Event-Driven)\n" + "=" * 60)
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

    # Initialize Position Manager
    pos_manager = PositionManager(os.path.join(os.path.dirname(__file__), "position_state.db"))

    current_capital = INITIAL_CAPITAL
    last_trade_win = False
    trade_log = []
    eod_report_done = False
    entry_features = {}

    df_chips_daily = load_latest_daily_chips_snapshot()

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

    event_queue = queue.Queue()
    accumulator = KBarAccumulator()
    opt_quotes = {}
    current_txf_price = df_raw.iloc[-1]['Close'] if not df_raw.empty else 0.0

    def on_tick_fop_callback(exchange, tick):
        try:
            symbol = getattr(tick, 'code', None)
            if not symbol: return
            now = datetime.now()
            price = getattr(tick, 'close', None)
            if price is None: return

            if "TXF" in symbol:
                volume = getattr(tick, 'volume', 1)
                event_queue.put(('TXF_TICK', now, float(price), int(volume)))
            else:
                bid = 0; ask = 0
                event_queue.put(('OPT_TICK', symbol, now, float(price), float(bid), float(ask)))
        except Exception as e:
            pass

    def quote_callback(topic, quote):
        try:
            symbol = topic.split('/')[-1]
            now = datetime.now()

            # 支援多種 Shioaji 版本屬性
            price = getattr(quote, 'close', getattr(quote, 'Close', None))
            if price is None: return

            if "TXF" in symbol:
                volume = getattr(quote, 'volume', getattr(quote, 'Volume', 1))
                event_queue.put(('TXF_TICK', now, float(price), int(volume)))
            else:
                bid = 0; ask = 0
                bid_prices = getattr(quote, 'BidPrice', getattr(quote, 'bid_price', []))
                ask_prices = getattr(quote, 'AskPrice', getattr(quote, 'ask_price', []))
                if bid_prices: bid = bid_prices[0]
                if ask_prices: ask = ask_prices[0]
                event_queue.put(('OPT_TICK', symbol, now, float(price), float(bid), float(ask)))
        except Exception as e:
            pass

    try:
        # 兼容不同版本的 Shioaji API
        if hasattr(engine.api.quote, 'set_on_tick_fop_v1_callback'):
            engine.api.quote.set_on_tick_fop_v1_callback(on_tick_fop_callback)
        elif hasattr(engine.api.quote, 'set_on_tick_fnc'):
            engine.api.quote.set_on_tick_fnc(quote_callback)
        elif hasattr(engine.api.quote, 'set_quote_callback'):
            engine.api.quote.set_quote_callback(quote_callback)
        elif hasattr(engine.api, 'on_tick'):
            engine.api.on_tick(quote_callback)

        engine.api.quote.subscribe(txf_contract, quote_type=sj.constant.QuoteType.Tick)
    except Exception as e:
        print(f"⚠️ 訂閱失敗或 API 版本不相容: {e}")

    print("✅ 事件驅動引擎啟動，進入主迴圈...")
    last_print_time = datetime.now()

    while True:
        now = datetime.now()
        current_time = now.time()
        time_str = now.strftime('%H:%M:%S')

        if not is_market_open(current_time):
            if pos_manager.get('position') != 0:
                pos_manager.clear_position()
            if eod_report_done and current_time < datetime_time(8, 45):
                eod_report_done = False
                trade_log = []

            # 改進：非交易時段不要卡死 60 秒，讓迴圈能快速響應開盤
            try:
                event = event_queue.get(timeout=5.0)
                # 即使是非交易時段，若有 Tick 也要處理（例如盤後或開盤前五分鐘）
            except queue.Empty:
                if (datetime.now() - last_print_time).total_seconds() > 60:
                    print(f"[{time_str}] 💤 非交易時段，等待日盤開盤 (08:45)...")
                    last_print_time = datetime.now()
                time.sleep(1)
                continue

        if is_eod_closing_time(current_time) and pos_manager.get('position') == 0 and not eod_report_done:
            generate_eod_report(trade_log, INITIAL_CAPITAL, current_capital)
            eod_report_done = True
            time.sleep(300)
            continue

        try:
            event = event_queue.get(timeout=1.0)
            event_type = event[0]

            if event_type == 'TXF_TICK':
                _, tick_time, price, volume = event
                current_txf_price = price
                bar = accumulator.on_tick(tick_time, price, volume)

                if bar:
                    # --- 1-Minute Bar Completed ---
                    new_row = pd.DataFrame([bar])
                    df_raw = pd.concat([df_raw, new_row], ignore_index=True)

                    df_intraday = calculate_features(df_raw)
                    if df_intraday is None or df_intraday.empty:
                        continue

                    df = engine.integrate_institutional_chips(df_intraday, df_chips_daily) if hasattr(engine, 'integrate_institutional_chips') else df_intraday

                    if df is None or df.empty or len(df) < WINDOW_SIZE:
                        continue

                    df_slice = df[feature_cols].tail(WINDOW_SIZE).copy()
                    for col in ['mock_volume', 'macd_hist', 'vwap_bias']:
                        if col in df_slice.columns:
                            df_slice[col] = np.sign(df_slice[col]) * np.log1p(np.abs(df_slice[col]))

                    feat_tensor = torch.tensor(np.nan_to_num((df_slice.values - mean_v) / np.where(std_v == 0, 1.0, std_v), nan=0.0), dtype=torch.float32).unsqueeze(0)

                    with torch.no_grad():
                        probs = torch.softmax(ai_model(feat_tensor), dim=1).squeeze().cpu().numpy()

                    # --- DEBUG AI ---
                    max_idx = int(np.argmax(probs))
                    confidence = probs[max_idx]
                    class_names = {
                        0: "Strong Down (-3)", 1: "Med Down (-2)", 2: "Weak Down (-1)",
                        3: "Hold (0)",
                        4: "Weak Up (1)", 5: "Med Up (2)", 6: "Strong Up (3)"
                    }
                    class_name = class_names.get(max_idx, "Unknown")
                    print(f"[{time_str}] 🤖 AI 預測完成: Class={max_idx-3} ({class_name}), Confidence={confidence:.2%}, Probs={np.round(probs, 3)}")

                    # --- Entry Logic ---
                    if pos_manager.get('position') == 0 and not is_eod_closing_time(current_time):
                        # 提前計算 DTE 判斷是否為結算日
                        dte_days_futures = get_api_based_dte(txf_contract, now)
                        is_settlement_day = dte_days_futures < 1.0

                        # 計算動態持有時間 (針對結算日縮短預期，加速獲利了結或停損)
                        expected_hold_time = 2.0
                        if is_settlement_day:
                            if current_time >= datetime_time(13, 0):
                                expected_hold_time = 0.25
                            elif current_time >= datetime_time(12, 30):
                                expected_hold_time = 0.5
                            else:
                                expected_hold_time = 1.0

                        signal = strategy_engine.generate_signal(df, ai_score=probs, last_win=last_trade_win)

                        # 降低門檻機制：即使 AI 預測最強類別是 Hold (0)，但若有其他趨勢類別達到指定信心度，則強制進場
                        if signal == 0:
                            # 找出非 Hold 類別中機率最高者
                            trend_probs = probs.copy()
                            trend_probs[3] = 0 # 排除 Hold 類別
                            max_trend_idx = int(np.argmax(trend_probs))

                            # 針對不同 Level 設定不同的激進門檻以增加短線與波段的出手次數
                            abs_level = abs(max_trend_idx - 3)
                            if abs_level == 3:
                                threshold = 0.35 # Level 3 極強勢維持較高標準
                            elif abs_level == 2:
                                threshold = 0.25 # Level 2 標準波段降低至 25%
                            else:
                                threshold = 0.20 # Level 1 短線游擊降低至 20% (從0.22微降)
                                
                            # 依據預期持倉時間動態提高門檻 (時間越短，容錯率越低，要求更高的爆發力)
                            if expected_hold_time <= 0.25:
                                threshold += 0.15 # 剩餘 15 分鐘，門檻大幅提高 15%
                            elif expected_hold_time <= 0.5:
                                threshold += 0.10 # 剩餘 30 分鐘，門檻提高 10%
                            elif expected_hold_time <= 1.0:
                                threshold += 0.05 # 剩餘 1 小時，門檻提高 5%

                            if trend_probs[max_trend_idx] > threshold:
                                mapping = {0:-3, 1:-2, 2:-1, 3:0, 4:1, 5:2, 6:3}
                                signal = mapping.get(max_trend_idx, 0)
                                if signal != 0:
                                    print(f"[{time_str}] ⚠️ 觸發激進短線門檻：偵測到 Level {abs_level} 趨勢 (Class {max_trend_idx-3}), Prob={trend_probs[max_trend_idx]:.2%} > 動態門檻 {threshold:.2%} (預期持倉 {expected_hold_time}h)")

                        if signal != 0:
                            opt_type = 'Call' if signal > 0 else 'Put'
                            allocated_capital_limit = min(current_capital * 0.50, MAX_POSITION_CAPITAL)

                            active_contract = engine.get_best_volume_option_contract(option_type=opt_type, allocated_capital=allocated_capital_limit)
                            if not active_contract:
                                continue

                            try:
                                snap = engine.api.snapshots([active_contract])[0]

                                # 吃單成本修正：進場買入 (Long) 時應支付 Best Ask (賣價)
                                # 這裡假設 best_ask 存在於快照中
                                best_bid = snap.buy_price if hasattr(snap, 'buy_price') else (snap.bids[0].price if hasattr(snap, 'bids') and snap.bids else 0)
                                best_ask = snap.sell_price if hasattr(snap, 'sell_price') else (snap.asks[0].price if hasattr(snap, 'asks') and snap.asks else 0)

                                if best_ask <= 0:
                                    print(f"⚠️ {active_contract.symbol} 無效委賣價 (Best Ask={best_ask})，放棄進場！")
                                    continue

                                entry_price = best_ask # 實際成交在賣價

                                if best_ask > 0 and best_bid > 0 and (best_ask - best_bid) > BID_ASK_SPREAD_THRESHOLD:
                                    print(f"⚠️ {active_contract.symbol} 買賣價差過大 ({best_ask} - {best_bid} = {best_ask-best_bid} > {BID_ASK_SPREAD_THRESHOLD})，放棄進場！")
                                    continue

                            except Exception as e:
                                print(f"獲取快照失敗: {e}")
                                continue

                            abs_sig = abs(signal)
                            if abs_sig == 3:
                                tp_mult, sl_mult = 5.0, 1.5
                                strategy_label = f"🚀 Level 3 極強勢波段 (Buy {opt_type})"
                            elif abs_sig == 2:
                                tp_mult, sl_mult = 3.0, 1.0
                                strategy_label = f"📈 Level 2 標準波段 (Buy {opt_type})"
                            else:
                                tp_mult, sl_mult = 1.5, 0.5
                                strategy_label = f"⚡ Level 1 短線游擊 (Buy {opt_type})"

                            current_atr = df_intraday['atr'].iloc[-1]
                            try:
                                strike_p = float(active_contract.strike_price) if hasattr(active_contract, 'strike_price') else current_txf_price
                            except:
                                strike_p = current_txf_price

                            # 改用 API 真實抓取的剩餘天數
                            days_to_expiry = get_api_based_dte(active_contract, now) / 365.0
                            current_iv = 0.22

                            hard_tp_price, hard_sl_price, d, g, t_decay = get_dynamic_bsm_bounds(
                                S=current_txf_price,
                                K=strike_p,
                                T=days_to_expiry,
                                r=0.015,
                                iv=current_iv,
                                atr=current_atr,
                                tp_mult=tp_mult,
                                sl_mult=sl_mult,
                                expected_hold_hours=expected_hold_time,
                                option_type=opt_type,
                                actual_entry_price=entry_price
                            )

                            num_contracts = max(1, int(allocated_capital_limit // (entry_price * CONTRACT_MULTIPLIER))) if entry_price > 0 else 1
                            trade_capital_used = num_contracts * entry_price * CONTRACT_MULTIPLIER

                            pos_manager.update(
                                position=1 if signal > 0 else -1,
                                entry_price=entry_price,
                                num_contracts=num_contracts,
                                highest_price_since_entry=entry_price,
                                active_contract_symbol=active_contract.symbol,
                                entry_time=now.strftime("%Y-%m-%d %H:%M:%S"),
                                trade_capital_used=trade_capital_used,
                                hard_tp_price=hard_tp_price,
                                hard_sl_price=hard_sl_price,
                                strategy_label=strategy_label
                            )
                            entry_features = {f"feat_{k}": v for k, v in df.iloc[-1].to_dict().items()}

                            try:
                                engine.api.quote.subscribe(active_contract, quote_type=sj.constant.QuoteType.Tick)
                            except:
                                pass

                            print(f"\n{'='*40}\n🔥 【盤中進場】: {strategy_label}\n當前本金: NT$ {current_capital:,.0f} | 合約: {active_contract.symbol}\n進場價: {entry_price:.2f} | {num_contracts} 口")
                            print(f"📊 [BSM風控對齊] Delta: {d:.3f} | Gamma: {g:.5f} | Theta 損耗: {t_decay:.2f} 點")
                            print(f"🎯 動態停利點: {hard_tp_price} | 動態停損點: {hard_sl_price}\n{'='*40}\n")

            elif event_type == 'OPT_TICK':
                _, symbol, tick_time, price, bid, ask = event
                opt_quotes[symbol] = {'price': price, 'bid': bid, 'ask': ask}

                position = pos_manager.get('position')
                if position != 0 and pos_manager.get('active_contract_symbol') == symbol:
                    # 離場成本修正：出場平倉 (賣出) 時應對齊 Best Bid (買價)
                    # 這裡的 price 是 Tick 的成交價，我們需要從 opt_quotes 取得最新的買價
                    best_bid = bid # 來自 OPT_TICK 事件
                    if best_bid <= 0:
                        # 若無委買價，暫時無法成交出場（除非尾盤強制）
                        if not is_eod_closing_time(current_time):
                            return
                        else:
                            best_bid = price # 尾盤強制平倉若無買單則用最後成交價

                    exec_price = best_bid
                    entry_price = pos_manager.get('entry_price')
                    hard_tp_price = pos_manager.get('hard_tp_price')
                    hard_sl_price = pos_manager.get('hard_sl_price')
                    highest_price = max(pos_manager.get('highest_price_since_entry'), price)

                    # 計算動態高點回檔停利 (Trailing Stop)
                    # 邏輯：如果最高獲利超過 15 點，則從最高點回檔 30% 或是固定回檔 10 點就停利出場，確保獲利落袋
                    profit_points = highest_price - entry_price
                    trailing_sl = hard_sl_price
                    if profit_points >= 15:
                        pullback = max(10, profit_points * 0.3)
                        trailing_sl = max(hard_sl_price, highest_price - pullback)

                    if highest_price > pos_manager.get('highest_price_since_entry'):
                        pos_manager.update(highest_price_since_entry=highest_price)

                    exit_reason = None
                    if is_eod_closing_time(current_time):
                        exit_reason = "尾盤強制平倉 (EOD)"
                    elif exec_price <= trailing_sl and trailing_sl > hard_sl_price:
                        exit_reason = f"觸及高點回檔動態停利 ({trailing_sl:.1f})"
                    elif exec_price <= hard_sl_price:
                        exit_reason = f"觸及原始動態停損 ({hard_sl_price})"
                    elif exec_price >= hard_tp_price:
                        exit_reason = f"觸及動態停利 ({hard_tp_price})"

                    if exit_reason:
                        num_contracts = pos_manager.get('num_contracts')
                        trade_capital_used = pos_manager.get('trade_capital_used')
                        points_gained = exec_price - entry_price
                        current_pnl = (points_gained * CONTRACT_MULTIPLIER * num_contracts) - (num_contracts * FEE_SLIPPAGE_PER_CONTRACT)
                        current_ret = current_pnl / trade_capital_used if trade_capital_used > 0 else 0

                        current_capital += current_pnl
                        last_trade_win = current_pnl > 0

                        trade_record = {
                            'entry_time': pos_manager.get('entry_time'),
                            'exit_time': now.strftime("%H:%M:%S"),
                            'symbol': symbol,
                            'direction': 'Call' if position==1 else 'Put',
                            'entry_price': entry_price,
                            'exit_price': exec_price,
                            'pnl': current_pnl,
                            'ret': current_ret,
                            'reason': exit_reason
                        }
                        if entry_features:
                            trade_record.update(entry_features)

                        trade_log.append(trade_record)

                        print(f"\n{'='*40}\n🔔 【平倉】: {exit_reason}\n實際盈虧: NT$ {current_pnl:,.0f} ({current_ret * 100:.2f}%)\n最新資金: NT$ {current_capital:,.0f}\n{'='*40}\n")
                        pos_manager.clear_position()
                        entry_features = {}

        except queue.Empty:
            position = pos_manager.get('position')
            if position != 0:
                if (now - last_print_time).total_seconds() >= 10:
                    last_print_time = now
                    sym = pos_manager.get('active_contract_symbol')
                    if sym in opt_quotes:
                        price = opt_quotes[sym]['price']
                        entry_price = pos_manager.get('entry_price')
                        num_contracts = pos_manager.get('num_contracts')
                        hard_sl = pos_manager.get('hard_sl_price')
                        highest = pos_manager.get('highest_price_since_entry')

                        profit_points = highest - entry_price
                        trailing_sl = hard_sl
                        if profit_points >= 15:
                            pullback = max(10, profit_points * 0.3)
                            trailing_sl = max(hard_sl, highest - pullback)

                        points_gained = price - entry_price
                        current_pnl = (points_gained * CONTRACT_MULTIPLIER * num_contracts) - (num_contracts * FEE_SLIPPAGE_PER_CONTRACT)
                        print(f"[{time_str}] 持倉: {sym} | 買入價: {entry_price} | 現價: {price} | 最高: {highest} | 防守價(SL/TSL): {trailing_sl:.1f} | 停利: {pos_manager.get('hard_tp_price')} | 帳面損益: {current_pnl:,.0f}")
        except Exception as e:
            print(f"[{time_str}] ❌ 主迴圈異常錯誤: {e}")

if __name__ == "__main__":
    run_live_simulator()
