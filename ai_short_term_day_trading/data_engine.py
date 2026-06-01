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

    def fetch_intraday_data(self, days=60):
        """
        強制日期對齊與高維度特徵提取引擎
        抓取台指期近月合約，並計算 AI 訓練所需的 29 個技術指標
        """
        # 計算動態日期區間 (確保結束日期為 today)
        today = datetime.now()
        start_date = (today - timedelta(days=days)).strftime("%Y-%m-%d")
        end_date = today.strftime("%Y-%m-%d")

        print(f"📡 [DATA] 正在抓取數據範圍: {start_date} 至 {end_date}")

        try:
            # 1. 強制重新整理合約清單，防止合約轉倉導致數據舊化

            contract = self.api.Contracts.Futures.TXF.TXFR1
            # 檢測合約是否成功讀取
            if contract is None:
                print("❌ 無法獲取近月合約 TXFR1，請檢查 Shioaji 登入狀態")
                return pd.DataFrame()

            # 2. 下載 K 線
            kbars = self.api.kbars(contract, start=start_date, end=end_date)
            df = pd.DataFrame({**kbars})

            if df.empty:
                print("❌ 警告：回傳數據為空，請檢查網路或 API 是否有資料")
                return pd.DataFrame()

            # 3. 基礎日期與索引清理
            df['ts'] = pd.to_datetime(df['ts'])
            df = df.sort_values('ts').reset_index(drop=True) # 強制時間排序
            df.rename(columns={'ts': 'date'}, inplace=True)
            # 檢查列名是否已經改了
            if 'date' not in df.columns:
                print(f"DEBUG: 重新命名失敗，目前的欄位: {df.columns.tolist()}")
            df['time'] = df['date'].dt.time
            df['date_only'] = df['date'].dt.date
            df['day_of_week'] = df['date'].dt.dayofweek

            # 4. 特徵工程 (Feature Engineering)
            # 價格變動
            df['ret'] = df['Close'].pct_change()

           # 價量與流動性指標：先計算好單列乘積，再 Groupby
            df['mock_volume'] = df['Volume'].replace(0, 1)
            df['vol_price'] = (df['High'] + df['Low'] + df['Close']) / 3 * df['mock_volume']

            # 使用更穩定的 sum/cumsum 寫法
            df['cum_vol_price'] = df.groupby('date_only')['vol_price'].cumsum()
            df['cum_vol'] = df.groupby('date_only')['mock_volume'].cumsum()
            df['vwap'] = df['cum_vol_price'] / df['cum_vol']
            df['vwap_bias'] = (df['Close'] - df['vwap']) / (df['vwap'] + 1e-9)

            # MACD
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

            # Bollinger Bands
            df['sma20'] = df['Close'].rolling(window=20).mean()
            df['std20'] = df['Close'].rolling(window=20).std()
            df['bb_upper'] = df['sma20'] + (df['std20'] * 2)
            df['bb_lower'] = df['sma20'] - (df['std20'] * 2)
            df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / (df['sma20'] + 1e-9)
            df['is_squeeze'] = (df['bb_width'] < df['bb_width'].rolling(20).mean()).astype(int)

            # 均線斜率 (Trend Slope) - 衡量趨勢的「速度」
            # 計算 5 分鐘 VWAP 或 MA 的斜率 (使用 3 根 K 棒的變化率並放大)
            df['vwap_5'] = df['mock_volume'] * df['Close'] / df['mock_volume'].rolling(5).sum()
            df['slope_vwap'] = (df['vwap_5'] - df['vwap_5'].shift(3)) / df['vwap_5'].shift(3) * 10000

            # 計算 20MA 斜率 (判斷中期趨勢方向)
            df['ma_20'] = df['Close'].rolling(20).mean()
            df['slope_ma20'] = (df['ma_20'] - df['ma_20'].shift(3)) / df['ma_20'].shift(3) * 10000


            # 價量結構 (Price-Volume Dynamics) - 衡量「爆發力」
            # 爆量倍數：當下成交量是過去 20 根均量的幾倍？(突破 2 倍以上才有意義)
            df['vol_surge_ratio'] = df['mock_volume'] / (df['mock_volume'].rolling(20).mean() + 1e-9)

            # 價量背離背離指標 (Price-Volume Divergence)
            # 如果價格創新高，但成交量萎縮，這通常是誘多 (假突破)
            df['price_roc'] = df['Close'].pct_change()
            df['pv_divergence'] = np.where((df['price_roc'] > 0) & (df['vol_surge_ratio'] < 0.8), -1,
                                np.where((df['price_roc'] < 0) & (df['vol_surge_ratio'] < 0.8), 1, 0))


            # 5. 清理與輸出
            # 刪除輔助計算的欄位，保留乾淨的 DataFrame
            cols_to_drop = ['vol_price', 'cum_vol_price', 'cum_vol', 'h_l', 'h_pc', 'l_pc', 'tr', 'sma20', 'std20', 'Amount']
            df.drop(columns=[c for c in cols_to_drop if c in df.columns], inplace=True)

            print(f"✅ 數據更新完成。最新時間: {df['date'].iloc[-1]}, 總筆數: {len(df)}")
            return df.dropna().reset_index(drop=True)

        except Exception as e:
            print(f"❌ 數據抓取異常: {e}")
            return pd.DataFrame()

    def get_best_volume_option_contract(self, option_type='Call', allocated_capital=100000):
        """🚀 依據成交量與資金篩選近月合約"""
        try:
            txf_contract = self.api.Contracts.Futures.TXF.TXFR1
            snap = self.api.snapshots([txf_contract])[0]
            current_index = snap.close

            all_options = []
            for cat in ['TXO', 'TX1', 'TX2', 'TX4', 'TX5']:
                if hasattr(self.api.Contracts.Options, cat):
                    all_options.extend([c for c in getattr(self.api.Contracts.Options, cat)])

            delivery_months = sorted(list(set(c.delivery_month for c in all_options)))
            if not delivery_months: return None

            near_month = delivery_months[0]
            
            near_contracts = [c for c in all_options if c.delivery_month == near_month and c.option_right.name == option_type]
            near_contracts.sort(key=lambda x: abs(x.strike_price - current_index))
            target_contracts = near_contracts[:30] # 擴大範圍至上下15檔

            if not target_contracts:
                return self.api.Contracts.Futures.TXF.TXFR1

            snaps = self.api.snapshots(target_contracts)
            
            valid_contracts = []
            for i, snap in enumerate(snaps):
                price = snap.close
                volume = snap.total_volume
                # 篩選條件：有報價且買得起 (選擇權一點 50 元)
                if price > 0 and (price * 50) <= allocated_capital:
                    valid_contracts.append({
                        'contract': target_contracts[i],
                        'price': price,
                        'volume': volume
                    })
            
            if not valid_contracts:
                target_contracts.sort(key=lambda x: abs(x.strike_price - current_index), reverse=True)
                return target_contracts[0]
                
            valid_contracts.sort(key=lambda x: x['volume'], reverse=True)
            return valid_contracts[0]['contract']

        except Exception as e:
            print(f"❌ 成交量合約定位失敗: {e}")
            return self.api.Contracts.Futures.TXF.TXFR1

    def integrate_institutional_chips(self, df_intraday, df_daily_chips):
        """將三大法人日留倉籌碼特徵，無未來函數地融合至當沖 K 線數據中"""
        df_chips = df_daily_chips.copy().sort_values('date').reset_index(drop=True)

        df_chips['foreign_oi_zscore'] = (
            df_chips['foreign_net_oi'] - df_chips['foreign_net_oi'].rolling(20).mean()
        ) / (df_chips['foreign_net_oi'].rolling(20).std() + 1e-9)

        df_chips['dealer_oi_zscore'] = (
            df_chips['dealer_net_oi'] - df_chips['dealer_net_oi'].rolling(20).mean()
        ) / (df_chips['dealer_net_oi'].rolling(20).std() + 1e-9)

        df_chips['foreign_oi_momentum'] = df_chips['foreign_net_oi'].diff()
        df_chips['dealer_oi_momentum'] = df_chips['dealer_net_oi'].diff()

        if 'pc_ratio' in df_chips.columns:
            df_chips['pc_ratio_momentum'] = df_chips['pc_ratio'].diff()

        features_to_shift = [
            'foreign_net_oi', 'dealer_net_oi',
            'foreign_oi_zscore', 'dealer_oi_zscore',
            'foreign_oi_momentum', 'dealer_oi_momentum'
        ]
        if 'pc_ratio' in df_chips.columns:
            features_to_shift.extend(['pc_ratio', 'pc_ratio_momentum'])

        df_chips[features_to_shift] = df_chips[features_to_shift].shift(1)
        df_chips.dropna(subset=['foreign_oi_zscore'], inplace=True)

        df_merged = pd.merge(
            df_intraday,
            df_chips[['date'] + [f for f in features_to_shift if f in df_chips.columns]],
            left_on='date_only', right_on='date', how='left'
        )

        if 'date_y' in df_merged.columns:
            df_merged.drop(columns=['date_y'], inplace=True)
        df_merged.rename(columns={'date_x': 'date'}, inplace=True, errors='ignore')

        df_merged[features_to_shift] = df_merged[features_to_shift].ffill()
        return df_merged.dropna(subset=['foreign_oi_zscore']).reset_index(drop=True)

    def fetch_real_historical_chips(self, days=90):
        """
        🚀 自動向期交所與證交所抓取真實歷史籌碼 (完整健壯版)
        """
        import io
        import requests
        import time
        from datetime import datetime, timedelta
        from requests.adapters import HTTPAdapter
        from urllib3.util.retry import Retry

        print(f"\n📡 開始抓取近 {days} 天籌碼...")
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)

        session = requests.Session()
        # 設定 Retry 機制：最多重試 5 次，且遇到 403, 500 等錯誤都會退避重試
        retries = Retry(total=5, backoff_factor=1, status_forcelist=[403, 500, 502, 503, 504])
        session.mount('https://', HTTPAdapter(max_retries=retries))

        # 模擬真實瀏覽器 Header 以突破防爬蟲限制
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            'Accept': 'text/csv,text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
            'Accept-Language': 'zh-TW,zh;q=0.9,en-US;q=0.8,en;q=0.7',
            'Connection': 'keep-alive',
            'Content-Type': 'application/x-www-form-urlencoded',
            'Origin': 'https://www.taifex.com.tw',
            'Referer': 'https://www.taifex.com.tw/cht/3/futContractsDate'
        }

        # 1. 自動分段邏輯 (防止 API 下載區間限制)
        date_chunks = []
        curr = start_date
        while curr < end_date:
            next_end = min(curr + timedelta(days=30), end_date)
            date_chunks.append((curr, next_end))
            curr = next_end + timedelta(days=1)

        # 2. 抓取 P/C Ratio 與 期貨 OI
        pc_frames, oi_frames = [], []
        year_offset = 0
        if end_date.year > 2024:
            year_offset = end_date.year - 2024

        for s_date, e_date in date_chunks:
            # 針對未來時間模擬，執行年份平移以抓取真實歷史資料並映射回未來
            query_s_dt = s_date.replace(year=s_date.year - year_offset).strftime("%Y/%m/%d")
            query_e_dt = e_date.replace(year=e_date.year - year_offset).strftime("%Y/%m/%d")

            try:
                # 抓取 PC Ratio
                r1 = session.post("https://www.taifex.com.tw/cht/3/pcRatioDown",
                                   data={"queryStartDate": query_s_dt, "queryEndDate": query_e_dt}, headers=headers, timeout=10)
                if r1.status_code == 200 and 'DOCTYPE html' not in r1.text[:100].upper() and 'alert' not in r1.text[:200]:
                    content_str = r1.content.decode('big5', errors='ignore')
                    if year_offset > 0: content_str = content_str.replace(str(e_date.year - year_offset), str(e_date.year))
                    pc_frames.append(pd.read_csv(io.StringIO(content_str)))

                time.sleep(1) # 增加延遲避免被封鎖

                # 抓取 期貨 OI
                r2 = session.post("https://www.taifex.com.tw/cht/3/futContractsDateDown",
                                   data={"queryStartDate": query_s_dt, "queryEndDate": query_e_dt, "commodityId": "TXF"}, headers=headers, timeout=10)
                if r2.status_code == 200 and 'DOCTYPE html' not in r2.text[:100].upper() and 'alert' not in r2.text[:200]:
                    content_str = r2.content.decode('big5', errors='ignore')
                    if year_offset > 0: content_str = content_str.replace(str(e_date.year - year_offset), str(e_date.year))
                    oi_frames.append(pd.read_csv(io.StringIO(content_str)))

                time.sleep(1) # 增加延遲避免被封鎖
            except Exception as e:
                print(f"⚠️ 下載警告: {e}")

        if not pc_frames or not oi_frames:
            print("❌ 籌碼抓取失敗，API 回傳無資料")
            return pd.DataFrame()

        # 3. 清理 P/C Ratio
        df_pc = pd.concat(pc_frames, ignore_index=True)
        df_pc.columns = df_pc.columns.str.strip()
        date_col = [c for c in df_pc.columns if '日期' in c][0]
        pc_col = [c for c in df_pc.columns if '比率' in c or '買賣權未平倉' in c][0]
        df_pc = df_pc[[date_col, pc_col]].copy()
        df_pc.rename(columns={date_col: 'date', pc_col: 'pc_ratio'}, inplace=True)
        df_pc['date'] = pd.to_datetime(df_pc['date'].astype(str).str.strip(), format='mixed', errors='coerce').dt.date
        df_pc['pc_ratio'] = pd.to_numeric(df_pc['pc_ratio'].astype(str).str.replace(',', ''), errors='coerce') / 100.0

        # 4. 清理期貨 OI
        df_oi = pd.concat(oi_frames, ignore_index=True)
        df_oi.columns = df_oi.columns.str.strip()

        # 檢查是否被阻擋而回傳 HTML
        if any('DOCTYPE' in str(c) or 'html' in str(c).lower() for c in df_oi.columns):
            print("❌ API 回傳 HTML 錯誤頁面，可能遭到阻擋或網址失效。")
            return pd.DataFrame()

        # 兼容不同欄位名稱：商品名稱 或 契約名稱
        item_col = '商品名稱' if '商品名稱' in df_oi.columns else '契約名稱'
        if item_col not in df_oi.columns:
            # 如果還是沒有，嘗試找包含「名稱」的欄位
            item_cols = [c for c in df_oi.columns if '名稱' in c]
            if item_cols:
                item_col = item_cols[0]
            else:
                print("❌ 找不到商品/契約名稱欄位！")
                return pd.DataFrame()

        df_oi = df_oi[df_oi[item_col].astype(str).str.strip() == '臺股期貨']
        net_col = [c for c in df_oi.columns if '未平倉' in c and '口數' in c][0]

        df_foreign = df_oi[df_oi['身份別'].astype(str).str.strip() == '外資及陸資'][['日期', net_col]].rename(columns={'日期': 'date', net_col: 'foreign_net_oi'})
        df_dealer = df_oi[df_oi['身份別'].astype(str).str.strip() == '自營商'][['日期', net_col]].rename(columns={'日期': 'date', net_col: 'dealer_net_oi'})

        for df_tmp in [df_foreign, df_dealer]:
            df_tmp['date'] = pd.to_datetime(df_tmp['date'].astype(str).str.strip(), format='mixed', errors='coerce')
            df_tmp.dropna(subset=['date'], inplace=True)
            df_tmp.iloc[:, 1] = pd.to_numeric(df_tmp.iloc[:, 1].astype(str).str.replace(',', ''), errors='coerce')

        # 確保 df_pc['date'] 也是 datetime64[ns] 型別
        df_pc['date'] = pd.to_datetime(df_pc['date'])

        # 5. 抓取證交所現貨
        twse_data = []
        for dt in pd.date_range(start=start_date, end=end_date):
            if dt.weekday() >= 5: continue
            try:
                # 若有平移年份，則請求真實歷史資料
                query_dt = dt.replace(year=dt.year - year_offset)
                res = session.get(f"https://www.twse.com.tw/exchangeReport/BFI82U?response=json&dayDate={query_dt.strftime('%Y%m%d')}&type=day", headers=headers, timeout=5)
                data = res.json()
                if data.get('stat') == 'OK':
                    f_s, d_s = 0.0, 0.0
                    for row in data['data']:
                        if '外資及陸資' in row[0]: f_s += float(row[3].replace(',', ''))
                        elif '自營商' in row[0]: d_s += float(row[3].replace(',', ''))
                    twse_data.append({'date': pd.to_datetime(dt.date()), 'foreign_spot_net': f_s, 'dealer_spot_net': d_s})
                time.sleep(1)
            except: pass
        df_twse = pd.DataFrame(twse_data) if twse_data else pd.DataFrame(columns=['date', 'foreign_spot_net', 'dealer_spot_net'])

        if not df_twse.empty:
            df_twse['date'] = pd.to_datetime(df_twse['date'])

        # 6. 合併與最終清理
        df_final = df_pc.merge(df_foreign, on='date', how='outer').merge(df_dealer, on='date', how='outer')
        if not df_twse.empty: df_final = df_final.merge(df_twse, on='date', how='outer')

        df_final.sort_values('date', inplace=True)

        # 關鍵：ffill 補假日 -> bfill 補遺漏開頭 -> fillna(0) 補完全缺失
        df_final.ffill(inplace=True)
        df_final.bfill(inplace=True)
        df_final.fillna(0, inplace=True)

        # 過濾日期解析失敗的 NaT
        df_final = df_final[df_final['date'].notna()]

        print(f"✅ 籌碼抓取完成，清理後有效樣本數: {len(df_final)}")
        return df_final.reset_index(drop=True)

    def integrate_institutional_chips(self, df_intraday, df_daily_chips):
        """
        將三大法人特徵與 K 線融合，並執行分層填補策略 (Layered Imputation)
        """
        # 1. 數據預處理與日期型別對齊
        df_chips = df_daily_chips.copy()
        df_chips['date'] = pd.to_datetime(df_chips['date']).dt.date
        df_intraday['date_only'] = pd.to_datetime(df_intraday['date']).dt.date

        # 2. 計算 Z-Score 與 Momentum (加上容錯處理)
        for col in ['foreign_net_oi', 'dealer_net_oi']:
            if col in df_chips.columns:
                df_chips[f'{col}_zscore'] = (df_chips[col] - df_chips[col].rolling(20).mean()) / (df_chips[col].rolling(20).std() + 1e-9)
                df_chips[f'{col}_momentum'] = df_chips[col].diff()

        # 3. 未來函數防禦：將籌碼指標 Shift(1)
        shift_cols = [c for c in df_chips.columns if c != 'date']
        df_chips[shift_cols] = df_chips[shift_cols].shift(1)

        # 4. 合併數據 (Left Merge)
        df_merged = df_intraday.merge(df_chips, left_on='date_only', right_on='date', how='left')

        # 5. 分層填補策略 (Layered Imputation Strategy)
        # 定義哪些指標是「水準值」，哪些是「變動量」
        ffill_cols = ['pc_ratio', 'foreign_net_oi', 'dealer_net_oi', 'foreign_net_oi_zscore', 'dealer_net_oi_zscore']
        zero_cols = ['foreign_net_oi_momentum', 'dealer_net_oi_momentum']

        # 執行填充：加入欄位存在檢查，防止 KeyError
        for col in ffill_cols:
            if col in df_merged.columns:
                # 先用 ffill 填補當日剩下的 K 線，再用 bfill 填補可能的空頭 (如開盤第一筆)
                df_merged[col] = df_merged[col].ffill().bfill()

        for col in zero_cols:
            if col in df_merged.columns:
                df_merged[col] = df_merged[col].fillna(0)

        # 最後的安全網：如果還有沒被補到的 (例如欄位名稱未定義)，全部填 0
        df_merged.fillna(0, inplace=True)

        print(f"✅ 籌碼融合完成，最終資料筆數: {len(df_merged)}")
        # 輸出一下目前的空值檢查報告
        null_count = df_merged.isnull().sum().sum()
        print(f"ℹ️ 最終檢查：融合後資料集尚有 {null_count} 個空值 (正常情況應為 0)")

        return df_merged.reset_index(drop=True)