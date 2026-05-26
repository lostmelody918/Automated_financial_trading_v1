import os
import pandas as pd
import numpy as np
import shioaji as sj
from datetime import datetime, timedelta
from dotenv import load_dotenv

class DayTradingDataEngine:
    def __init__(self, symbol="2330"):
        self.symbol = symbol
        self.api = sj.Shioaji()

        # 🚀 1. 強制指定載入你電腦中的 .env 絕對路徑 (使用 r 前綴避免斜線跳脫)
        env_path = r"F:\Gemini_CLI_Application\finance_v2\.env"
        load_dotenv(dotenv_path=env_path)

        # 2. 載入後，系統就能抓到環境變數了
        api_key = os.environ.get('SHIOAJI_API_KEY', '')
        secret_key = os.environ.get('SHIOAJI_SECRET_KEY', '')

        # 3. 執行登入與錯誤攔截
        if api_key and secret_key:
            print("🔐 正在登入 Shioaji API 並下載合約檔 (需時幾秒鐘)...")
            try:
                self.api.login(api_key, secret_key, contracts_timeout=10000)
                print("✅ Shioaji 登入成功！合約清單已載入。")
            except Exception as e:
                print(f"❌ Shioaji 登入失敗！請檢查金鑰或憑證: {e}")
                raise SystemExit("終止程式：無法連線至券商伺服器。")
        else:
            print(f"❌ 嚴重錯誤: 找不到 Shioaji API 金鑰！")
            print(f"請確認 {env_path} 檔案存在，且內容包含 SHIOAJI_API_KEY 與 SHIOAJI_SECRET_KEY")
            raise SystemExit("終止程式：缺少 API 金鑰。")


    def fetch_intraday_data(self, days=60):
        """獲取近 60 天 K 線，使用 Shioaji 進行抓取"""
        print(f"📥 下載 {self.symbol} 近 {days} 天 K 線數據...")

        try:
            # 預設抓取加權指數做特徵
            if self.symbol == "2330": # fallback if defaults are kept
                contract = self.api.Contracts.Stocks["2330"]
            else:
                contract = self.api.Contracts.Indices.TSE.TSE01
        except Exception as e:
            # 如果沒有指數權限，改抓台指期近月
            try:
                contract = self.api.Contracts.Futures.TXF.TXFR1
            except:
                print(f"Contract not found: {e}")
                return pd.DataFrame()

        start_date = (datetime.now() - timedelta(days=days)).strftime("%Y-%m-%d")
        end_date = datetime.now().strftime("%Y-%m-%d")

        try:
            kbars = self.api.kbars(contract, start=start_date, end=end_date)
            df = pd.DataFrame({**kbars})
        except Exception as e:
            print(f"Failed to fetch kbars: {e}")
            return pd.DataFrame()

        if df.empty:
            print(f"⚠️ {self.symbol} K 線資料為空")
            return df

        df['ts'] = pd.to_datetime(df['ts'])
        df.rename(columns={'ts': 'date'}, inplace=True)

        df['time'] = df['date'].dt.time
        df['date_only'] = df['date'].dt.date
        df['day_of_week'] = df['date'].dt.dayofweek

        # 基礎技術指標與流動性特徵
        df['ret'] = df['Close'].pct_change()
        df['slope_5'] = (df['Close'] - df['Close'].shift(5)) / 5.0
        df['slope_10'] = (df['Close'] - df['Close'].shift(10)) / 10.0

        df['typical_price'] = (df['High'] + df['Low'] + df['Close']) / 3
        df['mock_volume'] = df['Volume'].replace(0, 1)
        df['vol_price'] = df['typical_price'] * df['mock_volume']
        df['cum_vol_price'] = df.groupby('date_only')['vol_price'].cumsum()
        df['cum_vol'] = df.groupby('date_only')['mock_volume'].cumsum()
        df['vwap'] = df['cum_vol_price'] / (df['cum_vol'] + 1e-9)
        df['vwap_bias'] = (df['Close'] - df['vwap']) / (df['vwap'] + 1e-9)

        # MACD
        exp1 = df['Close'].ewm(span=12, adjust=False).mean()
        exp2 = df['Close'].ewm(span=26, adjust=False).mean()
        df['macd'] = exp1 - exp2
        df['signal'] = df['macd'].ewm(span=9, adjust=False).mean()
        df['macd_hist'] = df['macd'] - df['signal']

        # ATR
        df['h_l'] = df['High'] - df['Low']
        df['h_pc'] = abs(df['High'] - df['Close'].shift(1))
        df['l_pc'] = abs(df['Low'] - df['Close'].shift(1))
        df['tr'] = df[['h_l', 'h_pc', 'l_pc']].max(axis=1)
        df['atr'] = df['tr'].rolling(14).mean()

        # RSI
        delta = df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / (loss + 1e-9)
        df['rsi'] = 100 - (100 / (1 + rs))

        # Bollinger Bands & Squeeze
        df['sma20'] = df['Close'].rolling(window=20).mean()
        df['std20'] = df['Close'].rolling(window=20).std()
        df['bb_upper'] = df['sma20'] + (df['std20'] * 2)
        df['bb_lower'] = df['sma20'] - (df['std20'] * 2)
        df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / df['sma20']
        df['bb_width_ma20'] = df['bb_width'].rolling(20).mean()
        df['is_squeeze'] = (df['bb_width'] < df['bb_width_ma20']).astype(int)

        df['v_ma5'] = df['Volume'].rolling(5).mean()
        df['v_rel'] = df['Volume'] / (df['v_ma5'] + 1e-9)

        # 清理無用中間變數
        df.drop(columns=['typical_price', 'vol_price', 'cum_vol_price', 'cum_vol',
                        'h_l', 'h_pc', 'l_pc', 'tr', 'sma20', 'std20', 'v_ma5', 'bb_width_ma20', 'Amount'],
                inplace=True, errors='ignore')

        return df.dropna()

    def fetch_active_option_intraday_data(self, days=1):
        """
        直接根據當下大盤指數，鎖定最近一週的「價平」選擇權合約。
        """
        print(f"📥 啟動價平精準定位 (ATM Targeting)...")

        try:
            # 1. 先抓取當下大盤/期貨指數位置 (取得靶心)
            txf_contract = self.api.Contracts.Futures.TXF.TXFR1
            snap = self.api.snapshots([txf_contract])[0]
            current_index = snap.close
            print(f"🎯 當下台指期指標點位: {current_index}")

            # 2. 計算最接近的「價平履約價」 (台指選擇權履約價通常間距為 50 或 100)
            # 例如指數 42430 -> 最接近 42450 或 42400。這裡我們以 100 為基距尋找大關卡
            atm_strike = round(current_index / 100) * 100

            # 3. 找出本週到期 (TX1~TX5) 或近月 (TXO) 的合約
            # 實務上我們收集市場所有合約，找出履約價等於 atm_strike 的 Call 與 Put
            all_options = []
            for cat in ['TXO', 'TX1', 'TX2', 'TX4', 'TX5']:
                if hasattr(self.api.Contracts.Options, cat):
                    all_options.extend([c for c in getattr(self.api.Contracts.Options, cat)])

            # 過濾出近期合約 (避免抓到遠月份)
            delivery_months = sorted(list(set(c.delivery_month for c in all_options)))
            if not delivery_months:
                raise ValueError("找不到任何選擇權月份")
            near_month = delivery_months[0] # 最近到期的月份/週

            # 找出屬於該到期日，且履約價等於 ATM 的所有合約
            atm_contracts = [c for c in all_options if c.delivery_month == near_month and c.strike_price == atm_strike]

            if not atm_contracts:
                # 如果剛好沒這個價位，退而求其次抓全部同月份合約，找履約價最接近的
                near_contracts = [c for c in all_options if c.delivery_month == near_month]
                near_contracts.sort(key=lambda x: abs(x.strike_price - current_index))
                best_contract = near_contracts[0] # 抓最接近的一個 (不管是 C 或 P，只是用來抓報價，方向由 AI 決定)
            else:
                # 預設先抓 Call 作為代表 (後續實際下單時再依照 AI 的多空訊號來決定買 C 或買 P)
                calls = [c for c in atm_contracts if c.option_right == 'Call']
                best_contract = calls[0] if calls else atm_contracts[0]

            print(f"⭐ 精準鎖定近期價平合約: {best_contract.symbol} (履約價: {best_contract.strike_price})")

        except Exception as e:
            print(f"❌ 價平精準定位失敗: {e}")
            print("⚠️ 強制切換至 TXFR1 (期貨) 作為備援推論源...")
            best_contract = self.api.Contracts.Futures.TXF.TXFR1

        # --- 下方的 K 線抓取與指標計算邏輯完全維持不變 ---
        start_date = (datetime.now() - timedelta(days=days)).strftime("%Y-%m-%d")
        end_date = datetime.now().strftime("%Y-%m-%d")

        try:
            kbars = self.api.kbars(best_contract, start=start_date, end=end_date)
            df = pd.DataFrame({**kbars})

            if df.empty and best_contract != self.api.Contracts.Futures.TXF.TXFR1:
                print(f"⚠️ {best_contract.symbol} K 線為空，強制切換至 TXFR1 (期貨)...")
                best_contract = self.api.Contracts.Futures.TXF.TXFR1
                kbars = self.api.kbars(best_contract, start=start_date, end=end_date)
                df = pd.DataFrame({**kbars})

        except Exception as e:
            print(f"❌ 抓取 K 線發生錯誤: {e}")
            return pd.DataFrame(), None

        if df.empty:
            print(f"⚠️ {best_contract.symbol} K 線資料依然為空")
            return df, best_contract

        df['ts'] = pd.to_datetime(df['ts'])
        df.rename(columns={'ts': 'date'}, inplace=True)

        df['time'] = df['date'].dt.time
        df['date_only'] = df['date'].dt.date
        df['day_of_week'] = df['date'].dt.dayofweek

        # 基礎技術指標與流動性特徵
        df['ret'] = df['Close'].pct_change()
        df['slope_5'] = (df['Close'] - df['Close'].shift(5)) / 5.0
        df['slope_10'] = (df['Close'] - df['Close'].shift(10)) / 10.0

        df['mock_volume'] = df['Volume'].replace(0, 1)
        df['cum_vol'] = df.groupby('date_only')['mock_volume'].cumsum()

        # MACD (趨勢動能)
        exp1 = df['Close'].ewm(span=12, adjust=False).mean()
        exp2 = df['Close'].ewm(span=26, adjust=False).mean()
        df['macd'] = exp1 - exp2
        df['signal'] = df['macd'].ewm(span=9, adjust=False).mean()
        df['macd_hist'] = df['macd'] - df['signal']

        # ATR (波動率)
        df['h_l'] = df['High'] - df['Low']
        df['h_pc'] = abs(df['High'] - df['Close'].shift(1))
        df['l_pc'] = abs(df['Low'] - df['Close'].shift(1))
        df['tr'] = df[['h_l', 'h_pc', 'l_pc']].max(axis=1)
        df['atr'] = df['tr'].rolling(14).mean()

        # RSI
        delta = df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / (loss + 1e-9)
        df['rsi'] = 100 - (100 / (1 + rs))

        # Bollinger Bands & Squeeze
        df['sma20'] = df['Close'].rolling(window=20).mean()
        df['std20'] = df['Close'].rolling(window=20).std()
        df['bb_upper'] = df['sma20'] + (df['std20'] * 2)
        df['bb_lower'] = df['sma20'] - (df['std20'] * 2)
        df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / df['sma20']
        df['bb_width_ma20'] = df['bb_width'].rolling(20).mean()
        df['is_squeeze'] = (df['bb_width'] < df['bb_width_ma20']).astype(int)

        df['v_ma5'] = df['Volume'].rolling(5).mean()
        df['v_rel'] = df['Volume'] / (df['v_ma5'] + 1e-9)

        # 這裡為了維持與舊版特徵對齊，補上 vwap_bias (如果訓練時有用到)
        # 若需要精確 VWAP 可加回計算公式，或先設為 0 以防報錯
        if 'vwap_bias' not in df.columns:
            df['vwap_bias'] = 0.0

        # 清理無用中間變數
        df.drop(columns=['cum_vol', 'h_l', 'h_pc', 'l_pc', 'tr', 'sma20', 'std20', 'v_ma5', 'bb_width_ma20'],
                inplace=True, errors='ignore')

        return df.dropna(), best_contract

    def integrate_institutional_chips(self, df_intraday, df_daily_chips):
        """
        將三大法人日留倉籌碼特徵，安全、無未來函數地融合至當沖 K 線數據中
        """
        df_chips = df_daily_chips.copy().sort_values('date').reset_index(drop=True)

        # 1. 籌碼特徵工程：滾動 20 日 Z-Score 消除高低基期絕對口數誤差
        df_chips['foreign_oi_zscore'] = (
            df_chips['foreign_net_oi'] - df_chips['foreign_net_oi'].rolling(20).mean()
        ) / (df_chips['foreign_net_oi'].rolling(20).std() + 1e-9)

        df_chips['dealer_oi_zscore'] = (
            df_chips['dealer_net_oi'] - df_chips['dealer_net_oi'].rolling(20).mean()
        ) / (df_chips['dealer_net_oi'].rolling(20).std() + 1e-9)

        # 2. 籌碼一階變動動能
        df_chips['foreign_oi_momentum'] = df_chips['foreign_net_oi'].diff()
        df_chips['dealer_oi_momentum'] = df_chips['dealer_net_oi'].diff()

        if 'pc_ratio' in df_chips.columns:
            df_chips['pc_ratio_momentum'] = df_chips['pc_ratio'].diff()

        # 3. 嚴格防禦未来數據：透過 shift(1) 確保今日盤中只能讀到昨天的最終結算籌碼
        features_to_shift = [
            'foreign_net_oi', 'dealer_net_oi',
            'foreign_oi_zscore', 'dealer_oi_zscore',
            'foreign_oi_momentum', 'dealer_oi_momentum'
        ]
        if 'pc_ratio' in df_chips.columns:
            features_to_shift.extend(['pc_ratio', 'pc_ratio_momentum'])

        df_chips[features_to_shift] = df_chips[features_to_shift].shift(1)
        df_chips.dropna(subset=['foreign_oi_zscore'], inplace=True)

        # 4. 跨時間尺度廣播合併 (Low-freq Broadcast to High-freq)
        df_merged = pd.merge(
            df_intraday,
            df_chips[['date'] + [f for f in features_to_shift if f in df_chips.columns]],
            left_on='date_only',
            right_on='date',
            how='left'
        )

        if 'date_y' in df_merged.columns:
            df_merged.drop(columns=['date_y'], inplace=True)
        df_merged.rename(columns={'date_x': 'date'}, inplace=True, errors='ignore')

        # 順向填補空缺值並剔除歷史早期無籌碼支援的 K 線
        df_merged[features_to_shift] = df_merged[features_to_shift].ffill()
        df_final = df_merged.dropna(subset=['foreign_oi_zscore']).reset_index(drop=True)

        return df_final