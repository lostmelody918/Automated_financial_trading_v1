import os
import pandas as pd
import numpy as np
import requests
import io
import time
import shioaji as sj
from datetime import datetime, timedelta
from dotenv import load_dotenv

class DayTradingDataEngine:
    def __init__(self, symbol="2330"):
        self.symbol = symbol
        self.api = sj.Shioaji()

        env_path = r"F:\Gemini_CLI_Application\finance_v2\.env"
        load_dotenv(dotenv_path=env_path)

        api_key = os.environ.get('SHIOAJI_API_KEY', '')
        secret_key = os.environ.get('SHIOAJI_SECRET_KEY', '')

        if api_key and secret_key:
            print("🔐 正在登入 Shioaji API 並下載合約檔...")
            try:
                self.api.login(api_key, secret_key, contracts_timeout=10000)
                print("✅ Shioaji 登入成功！合約清單已載入。")
            except Exception as e:
                print(f"❌ Shioaji 登入失敗: {e}")
                raise SystemExit("終止程式：無法連線至券商伺服器。")
        else:
            raise SystemExit("終止程式：缺少 API 金鑰。")

    def build_settlement_calendar(self):
        """建立動態交割日曆"""
        settlement_calendar = {}
        today = datetime.now().date()
        for i in range(365 * 3):
            d = today - timedelta(days=i)
            if d.weekday() == 2: # 週三
                if 15 <= d.day <= 21: settlement_calendar[d.strftime('%Y-%m-%d')] = 2
                else: settlement_calendar[d.strftime('%Y-%m-%d')] = 1
        return settlement_calendar

    def add_expiration_features(self, df, settlement_calendar=None):
        """將交割日資訊注入 DataFrame"""
        if df.empty: return df
        if settlement_calendar is None: settlement_calendar = self.build_settlement_calendar()
        df['settlement_type'] = 0
        df['is_settlement_day'] = 0
        df['dte'] = 5.0
        df_dates = df['date_only'].unique()
        for d in df_dates:
            try:
                d_str = d.strftime('%Y-%m-%d')
            except:
                d_str = str(d)
            future_dte, found_type = 5.0, 0
            for i in range(15):
                future_d = d + pd.Timedelta(days=i)
                f_str = future_d.strftime('%Y-%m-%d')
                if f_str in settlement_calendar:
                    future_dte, found_type = float(i), settlement_calendar[f_str]
                    break
            mask = df['date_only'] == d
            df.loc[mask, 'settlement_type'] = found_type
            df.loc[mask, 'is_settlement_day'] = 1 if future_dte == 0.0 else 0
            target_time = pd.to_datetime(d_str + ' 13:30:00')
            time_diff_days = (target_time - df.loc[mask, 'date']).dt.total_seconds() / 86400.0
            df.loc[mask, 'dte'] = (future_dte + time_diff_days).clip(lower=0.001)
        return df

    def fetch_intraday_data(self, days=60):
        """分段式高效資料抓取引擎 (具備本地快取功能)"""
        print(f"🚀 進入 fetch_intraday_data, 請求天數: {days}")
        cache_file = os.path.join(os.path.dirname(__file__), "market_data_cache.parquet")
        today = datetime.now()
        df_cache = pd.DataFrame()
        last_date = None
        first_date = None

        if os.path.exists(cache_file):
            try:
                df_cache = pd.read_parquet(cache_file)
                if not df_cache.empty:
                    df_cache['date'] = pd.to_datetime(df_cache['date'])
                    last_date = df_cache['date'].max()
                    first_date = df_cache['date'].min()
                    print(f"📦 載入本地行情快取，日期範圍: {first_date.date()} ~ {last_date.date()} ({len(df_cache)} 筆)")
            except Exception as e:
                print(f"⚠️ 快取讀取失敗: {e}")

        target_start = (today - timedelta(days=days + 5))

        # 判定是否需要抓取新數據 (包含往未來補與往過去補)
        needs_future = not last_date or last_date.date() < today.date()
        needs_past = not first_date or first_date > target_start

        if not needs_future and not needs_past:
            print("✅ 數據已充足且是最新，無需抓取。")
            df = df_cache
        else:
            all_frames = [df_cache] if not df_cache.empty else []

            try:
                contract = self.api.Contracts.Futures.TXF.TXFR1
            except Exception as e:
                print(f"⏳ 等待合約清單載入... ({e})")
                time.sleep(5)
                contract = self.api.Contracts.Futures.TXF.TXFR1

            # 1. 補過去的缺口
            if needs_past:
                fetch_past_start = target_start
                fetch_past_end = first_date if first_date else today
                print(f"📡 [DATA] 偵測到過去數據缺口，開始分段補抓 ({fetch_past_start.date()} ~ {fetch_past_end.date()})...")

                curr_start = fetch_past_start
                while curr_start < fetch_past_end:
                    curr_end = min(curr_start + timedelta(days=7), fetch_past_end)
                    print(f"   📥 抓取區間(過去): {curr_start.strftime('%Y-%m-%d')} -> {curr_end.strftime('%Y-%m-%d')}...")

                    for retry in range(3):
                        try:
                            kbars = self.api.kbars(contract, start=curr_start.strftime("%Y-%m-%d"), end=curr_end.strftime("%Y-%m-%d"))
                            if kbars and len(kbars.ts) > 0:
                                df_chunk = pd.DataFrame({**kbars})
                                df_chunk['ts'] = pd.to_datetime(df_chunk['ts'])
                                df_chunk.rename(columns={'ts': 'date', 'open':'Open','high':'High','low':'Low','close':'Close','volume':'Volume','amount':'Amount'}, inplace=True)
                                all_frames.append(df_chunk)
                                break
                            else:
                                print("      ℹ️ 此區間無資料")
                                break
                        except Exception as e:
                            wait_time = (retry + 1) * 5
                            print(f"      ❌ 抓取異常 (嘗試 {retry+1}/3): {e}，等待 {wait_time} 秒後重試...")
                            time.sleep(wait_time)

                    curr_start = curr_end + timedelta(days=1)
                    time.sleep(1.2) # 安全間隔

            # 2. 補未來的缺口
            if needs_future:
                fetch_future_start = last_date + timedelta(minutes=1) if last_date else target_start
                print(f"📡 [DATA] 偵測到未來數據缺口，開始分段補抓 ({fetch_future_start.date()} ~ {today.date()})...")

                curr_start = fetch_future_start
                while curr_start < today:
                    curr_end = min(curr_start + timedelta(days=7), today)
                    print(f"   📥 抓取區間(未來): {curr_start.strftime('%Y-%m-%d')} -> {curr_end.strftime('%Y-%m-%d')}...")

                    for retry in range(3):
                        try:
                            kbars = self.api.kbars(contract, start=curr_start.strftime("%Y-%m-%d"), end=curr_end.strftime("%Y-%m-%d"))
                            if kbars and len(kbars.ts) > 0:
                                df_chunk = pd.DataFrame({**kbars})
                                df_chunk['ts'] = pd.to_datetime(df_chunk['ts'])
                                df_chunk.rename(columns={'ts': 'date', 'open':'Open','high':'High','low':'Low','close':'Close','volume':'Volume','amount':'Amount'}, inplace=True)
                                all_frames.append(df_chunk)
                                break
                            else:
                                print("      ℹ️ 此區間無資料")
                                break
                        except Exception as e:
                            wait_time = (retry + 1) * 5
                            print(f"      ❌ 抓取異常 (嘗試 {retry+1}/3): {e}，等待 {wait_time} 秒後重試...")
                            time.sleep(wait_time)

                    curr_start = curr_end + timedelta(days=1)
                    time.sleep(1.2)

            if all_frames:
                df = pd.concat(all_frames, ignore_index=True).drop_duplicates(subset=['date']).sort_values('date').reset_index(drop=True)
                print(f"📊 歷史數據合併完成，總計 {len(df)} 根 K 線")
                # 儲存基本欄位到快取
                base_cols = ['date', 'Open', 'High', 'Low', 'Close', 'Volume', 'Amount']
                df[base_cols].to_parquet(cache_file, compression='snappy')
            else:
                print("⚠️ 警告：完全無法取得任何歷史行情資料！")
                return pd.DataFrame()

        if df.empty:
            print("❌ 最終數據集為空，無法進行後續運算")
            return df

        print("🛠️ 正在執行特徵工程計算...")
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
        exp1, exp2 = df['Close'].ewm(span=12).mean(), df['Close'].ewm(span=26).mean()
        df['macd'] = exp1 - exp2
        df['signal'] = df['macd'].ewm(span=9).mean()
        df['macd_hist'] = df['macd'] - df['signal']
        df['tr'] = df[['High','Low','Close']].max(axis=1) # 簡化
        df['atr'] = df['tr'].rolling(14).mean()
        delta = df['Close'].diff()
        gain, loss = (delta.where(delta > 0, 0)).rolling(14).mean(), (-delta.where(delta < 0, 0)).rolling(14).mean()
        df['rsi'] = 100 - (100 / (1 + gain/(loss+1e-9)))
        df['rsi_fast'] = 100 - (100 / (1 + (delta.where(delta>0,0).rolling(3).mean()/(delta.where(delta<0,0).abs().rolling(3).mean()+1e-9))))
        df['body_length'] = abs(df['Close'] - df['Open'])
        df['upper_shadow'], df['lower_shadow'] = df['High']-df[['Open','Close']].max(axis=1), df[['Open','Close']].min(axis=1)-df['Low']
        df['body_avg_5'] = df['body_length'].rolling(5).mean()
        df['momentum_explosion'] = (df['body_length'] > df['body_avg_5']).astype(int)
        df['daily_open'] = df.groupby('date_only')['Open'].transform('first')
        daily_close = df.groupby('date_only')['Close'].last().shift(1)
        df['yesterday_close'] = df['date_only'].map(daily_close).fillna(df['daily_open'])
        df['gap_amplitude'] = (df['daily_open'] - df['yesterday_close']) / (df['yesterday_close'] + 1e-9)
        df['intraday_trend'] = (df['Close'] - df['daily_open']) / (df['daily_open'] + 1e-9)
        df['dist_from_ma20'] = (df['Close'] - df['Close'].rolling(20).mean()) / (df['Close'].rolling(20).mean() + 1e-9)
        df['pullback_from_high'] = (df['Close'] - df['High'].rolling(20).max()) / (df['High'].rolling(20).max() + 1e-9)
        df['bounce_from_low'] = (df['Close'] - df['Low'].rolling(20).min()) / (df['Low'].rolling(20).min() + 1e-9)
        df['gap_filled'] = 0
        df.loc[(df['gap_amplitude'] > 0) & (df['Close'] <= df['yesterday_close']), 'gap_filled'] = 1
        df.loc[(df['gap_amplitude'] < 0) & (df['Close'] >= df['yesterday_close']), 'gap_filled'] = 1
        df['spot_futures_proxy'] = (df['Close'] - df['vwap']) / (df['Close'].rolling(20).mean() + 1e-9)
        df['sma20'] = df['Close'].rolling(20).mean()
        df['bb_width'] = (df['High'].rolling(20).max() - df['Low'].rolling(20).min()) / (df['sma20'] + 1e-9)
        df['is_squeeze'] = (df['bb_width'] < df['bb_width'].rolling(20).mean()).astype(int)
        df['vwap_5'] = (df['mock_volume'] * df['Close']).rolling(5).sum() / (df['mock_volume'].rolling(5).sum() + 1e-9)
        df['slope_vwap'] = (df['vwap_5'] - df['vwap_5'].shift(3)) / (df['vwap_5'].shift(3) + 1e-9) * 10000
        df['ma_20'] = df['Close'].rolling(20).mean()
        df['slope_ma20'] = (df['ma_20'] - df['ma_20'].shift(3)) / (df['ma_20'].shift(3) + 1e-9) * 10000
        df['vol_surge_ratio'] = df['mock_volume'] / (df['mock_volume'].rolling(20).mean() + 1e-9)
        df['pv_divergence'] = np.where((df['Close'].pct_change() > 0) & (df['vol_surge_ratio'] < 0.8), -1, 0)
        # 小波與分數階簡略，保持特徵維度
        df['close_frac_diff'] = df['Close'].diff().fillna(0)
        df['trend_wavelet'], df['noise_wavelet'] = df['Close'], df['Close'].diff().fillna(0)
        df = self.add_expiration_features(df)

        # --------------------------------------------------
        # 🕰️ 週期性時間編碼 (Cyclical Time Encoding)
        # --------------------------------------------------
        minutes_of_day = df['date'].dt.hour * 60 + df['date'].dt.minute
        df['time_sin'] = np.sin(2 * np.pi * minutes_of_day / 1440.0)
        df['time_cos'] = np.cos(2 * np.pi * minutes_of_day / 1440.0)

        # --------------------------------------------------
        # 🌍 整合美股與日股全域特徵 (Global Features)
        # --------------------------------------------------
        global_df = self.fetch_global_indices()
        if not global_df.empty:
            df['date_only_dt'] = pd.to_datetime(df['date_only'])
            global_df['date_only_dt'] = pd.to_datetime(global_df['date_only'])
            df = pd.merge_asof(
                df.sort_values('date_only_dt'),
                global_df.sort_values('date_only_dt'),
                on='date_only_dt',
                direction='backward'
            )
            df.drop(columns=['date_only_dt', 'date_only_y'], inplace=True, errors='ignore')
            if 'date_only_x' in df.columns:
                df.rename(columns={'date_only_x': 'date_only'}, inplace=True)
            df['nasdaq_prev_ret'] = df['nasdaq_prev_ret'].fillna(0)
            df['nikkei_premarket_momentum'] = df['nikkei_premarket_momentum'].fillna(0)
        else:
            df['nasdaq_prev_ret'] = 0.0
            df['nikkei_premarket_momentum'] = 0.0

        # 計算美台背離 (US-TW Gap Divergence)
        df['us_tw_gap_divergence'] = df['gap_amplitude'] - df['nasdaq_prev_ret']

        # --------------------------------------------------
        # 🛡️ 嚴格日盤遮罩與半週期餘弦衰減 (Half-Cosine Decay)
        # --------------------------------------------------
        # 08:45 = 8 * 60 + 45 = 525, 13:45 = 13 * 60 + 45 = 825
        minutes_from_0845 = minutes_of_day - 525
        is_day_session = (minutes_of_day >= 525) & (minutes_of_day <= 825)
        cosine_decay = np.where(
            (minutes_from_0845 >= 0) & (minutes_from_0845 <= 60),
            np.cos((minutes_from_0845 / 60.0) * (np.pi / 2.0)),
            0.0
        )
        final_weight = cosine_decay * is_day_session
        df['us_tw_gap_divergence'] = df['us_tw_gap_divergence'] * final_weight
        df['nikkei_premarket_momentum'] = df['nikkei_premarket_momentum'] * final_weight

        print(f"✅ 數據就緒，最新時間: {df['date'].iloc[-1]}")
        return df.dropna().reset_index(drop=True)

    def fetch_global_indices(self):
        """抓取美股收盤與日股開盤資訊"""
        cache_file = os.path.join(os.path.dirname(__file__), "global_indices_cache.parquet")
        today = datetime.now()
        df_cache = pd.DataFrame()
        if os.path.exists(cache_file):
            try:
                df_cache = pd.read_parquet(cache_file)
                if not df_cache.empty and pd.to_datetime(df_cache['date_only']).max().date() >= today.date():
                    return df_cache
            except: pass

        print("📡 正在抓取美股(Nasdaq)與日股(Nikkei)數據...")
        try:
            import yfinance as yf
            ndx = yf.download('^IXIC', period='1000d', interval='1d', progress=False)
            if isinstance(ndx.columns, pd.MultiIndex):
                ndx_close = ndx['Close'].iloc[:, 0]
            else:
                ndx_close = ndx['Close']
            ndx_ret = ndx_close.pct_change().shift(1)
            ndx_df = pd.DataFrame({'date_only': ndx_close.index.tz_localize(None).date, 'nasdaq_prev_ret': ndx_ret.values})

            n225 = yf.download('^N225', period='1000d', interval='15m', progress=False)
            if isinstance(n225.columns, pd.MultiIndex):
                n225_open = n225['Open'].iloc[:, 0]
                n225_close = n225['Close'].iloc[:, 0]
            else:
                n225_open = n225['Open']
                n225_close = n225['Close']

            n225_df = pd.DataFrame({'datetime': n225_open.index})
            n225_df['date_only'] = n225_df['datetime'].dt.tz_localize(None).dt.date
            n225_df['time'] = n225_df['datetime'].dt.tz_localize(None).dt.time
            n225_df['open'] = n225_open.values
            n225_df['close'] = n225_close.values

            nikkei_features = []
            for d, grp in n225_df.groupby('date_only'):
                try:
                    open_price = grp.iloc[0]['open']
                    close_0845 = grp.iloc[2]['close'] # JST 15m intervals
                    momentum = (close_0845 / open_price) - 1.0
                    nikkei_features.append({'date_only': d, 'nikkei_premarket_momentum': momentum})
                except IndexError: pass

            nikkei_df = pd.DataFrame(nikkei_features)
            global_df = pd.merge(ndx_df, nikkei_df, on='date_only', how='outer').dropna()
            global_df.to_parquet(cache_file)
            return global_df
        except Exception as e:
            print(f"⚠️ 抓取國際指數失敗: {e}")
            return df_cache

    def fetch_real_historical_chips(self, days=180):
        """自動抓取真實歷史籌碼 (具備本地快取功能)"""
        cache_file = os.path.join(os.path.dirname(__file__), "chips_data_cache.parquet")
        today = datetime.now()
        df_cache = pd.DataFrame()
        last_date = None
        first_date = None

        if os.path.exists(cache_file):
            try:
                df_cache = pd.read_parquet(cache_file)
                if not df_cache.empty:
                    df_cache['date'] = pd.to_datetime(df_cache['date'])
                    last_date = df_cache['date'].max()
                    first_date = df_cache['date'].min()
                    print(f"📦 載入本地籌碼快取，日期範圍: {first_date.date()} ~ {last_date.date()}")
            except: pass

        target_start = (today - timedelta(days=days))

        needs_future = not last_date or last_date.date() < today.date()
        needs_past = not first_date or first_date.date() > target_start.date()

        if not needs_future and not needs_past:
            print("✅ 籌碼數據已充足且是最新")
            return df_cache

        # 如果缺資料，則從缺口開始抓
        fetch_start = target_start if needs_past else (last_date + timedelta(days=1))

        print(f"📡 補抓籌碼: {fetch_start.date()} 至 {today.date()}...")
        twse_data = []
        token = os.environ.get('FINMIND_API_TOKEN', 'eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJ1c2VyX2lkIjoibG9zdG1lbG9keSIsImVtYWlsIjoibGVhdmU5MThAZ21haWwuY29tIiwidG9rZW5fdmVyc2lvbiI6MH0.BmG_w18TAEobmpkAA3BO_9mPvWiVrXwYfey_n7xRUQ4')
        try:
            url = 'https://api.finmindtrade.com/api/v4/data'
            params = {'dataset': 'TaiwanStockTotalInstitutionalInvestors', 'start_date': fetch_start.strftime("%Y-%m-%d"), 'end_date': today.strftime("%Y-%m-%d"), 'token': token}
            res = requests.get(url, params=params, timeout=15)
            data = res.json().get('data', [])
            if data:
                df_fm = pd.DataFrame(data)
                for dt, grp in df_fm.groupby('date'):
                    f_s = grp[grp['name'].str.contains('Foreign_Investor')]['buy'].sum() - grp[grp['name'].str.contains('Foreign_Investor')]['sell'].sum()
                    d_s = grp[grp['name'].str.contains('Dealer')]['buy'].sum() - grp[grp['name'].str.contains('Dealer')]['sell'].sum()
                    twse_data.append({'date': pd.to_datetime(dt), 'foreign_spot_net': f_s, 'dealer_spot_net': d_s})
        except Exception as e: print(f"⚠️ FinMind 抓取失敗: {e}")

        df_new = pd.DataFrame(twse_data)
        if not df_new.empty:
            for col in ['pc_ratio', 'foreign_net_oi', 'dealer_net_oi']:
                if col not in df_new.columns: df_new[col] = 0.0
            df_final = pd.concat([df_cache, df_new], ignore_index=True).drop_duplicates(subset=['date']).sort_values('date')
            df_final.to_parquet(cache_file)
            return df_final
        return df_cache

    def get_best_volume_option_contract(self, option_type='Call', allocated_capital=100000):
        """自動尋找當前成交量最大且符合預算的選擇權合約"""
        try:
            txf_contract = self.api.Contracts.Futures.TXF.TXFR1
            snapshot = self.api.snapshots([txf_contract])
            if not snapshot: return None
            current_price = snapshot[0].close
            txo_list = []
            for cat in ['TXO', 'TX1', 'TX2', 'TX4', 'TX5']:
                if hasattr(self.api.Contracts.Options, cat): txo_list.extend(list(getattr(self.api.Contracts.Options, cat)))
            if not txo_list: return None
            filtered_txo = [c for c in txo_list if c.option_right == option_type and abs(c.strike_price - current_price) <= 500]
            if not filtered_txo: return None
            snapshots = self.api.snapshots(filtered_txo)
            if not snapshots: return filtered_txo[0]
            best_v, best_contract = -1, None
            for s in snapshots:
                # 絕對禁止買入「無賣盤(ask_price <= 0)」或「零成交量」的合約
                if getattr(s, 'ask_price', 0) <= 0 or getattr(s, 'volume', 0) == 0:
                    continue
                if s.ask_price * 50 > allocated_capital: continue
                if s.volume > best_v:
                    best_v = s.volume
                    for c in filtered_txo:
                        if c.code == s.code:
                            best_contract = c
                            break
            if not best_contract: print("⚠️ 找不到具備充足流動性的合約"); return None
            return best_contract
        except Exception as e: print(f"⚠️ get_best_volume_option_contract 異常: {e}"); return None

    def integrate_institutional_chips(self, df_intraday, df_daily_chips):
        """融合籌碼數據"""
        if df_daily_chips.empty: print("⚠️ 籌碼資料為空，跳過融合"); return df_intraday
        df_chips = df_daily_chips.copy()
        df_chips['date_dt'] = pd.to_datetime(df_chips['date']).dt.as_unit('us')
        df_intraday['date_only_dt'] = pd.to_datetime(df_intraday['date_only']).dt.as_unit('us')

        df_merged = pd.merge_asof(
            df_intraday.sort_values('date_only_dt'),
            df_chips.sort_values('date_dt'),
            left_on='date_only_dt',
            right_on='date_dt',
            direction='backward',
            allow_exact_matches=False
        )

        if 'pc_ratio' in df_merged.columns:
            df_merged['pc_ratio'] = df_merged['pc_ratio'].fillna(1.0)

        # 安全填充：清理無用的時間對齊欄位，並針對數值欄位補零
        cols_to_drop = ['date_dt', 'date_only_dt']
        if 'date_y' in df_merged.columns: cols_to_drop.append('date_y')
        df_merged.drop(columns=[c for c in cols_to_drop if c in df_merged.columns], inplace=True, errors='ignore')
        if 'date_x' in df_merged.columns: df_merged.rename(columns={'date_x': 'date'}, inplace=True)

        # 只對數值欄位進行 0.0 填充，避開時間欄位報錯
        numeric_cols = df_merged.select_dtypes(include=[np.number]).columns
        df_merged[numeric_cols] = df_merged[numeric_cols].fillna(0.0)

        return df_merged.reset_index(drop=True)
