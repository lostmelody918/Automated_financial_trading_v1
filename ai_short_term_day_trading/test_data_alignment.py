import pandas as pd
from data_engine import DayTradingDataEngine
from datetime import datetime

def debug_data_pipeline():
    print("🚀 啟動健檢程式...")
    engine = DayTradingDataEngine()

    # 1. 抓取資料 (必須先抓，後面才能診斷)
    print("Step 1: 正在抓取 K 線與籌碼資料...")
    df_raw = engine.fetch_intraday_data(days=60)
    df_chips = engine.fetch_real_historical_chips(days=90)

    # 2. 檢查 K 線狀況
    print(f"\n[K線] 資料筆數: {len(df_raw)}")
    if not df_raw.empty:
        # 強制將欄位改名並解析
        df_raw.rename(columns={'ts': 'date'}, errors='ignore', inplace=True)
        df_raw['date_only'] = pd.to_datetime(df_raw['date']).dt.date
        print(f"[K線] 實際欄位名稱: {df_raw.columns.tolist()}")
        print(f"[K線] 日期解析成功，最新日期: {df_raw['date_only'].iloc[-1]}")
    else:
        print("❌ K 線資料為空！")
        return

    # 3. 檢查籌碼狀況
    print(f"\n[籌碼] 資料筆數: {len(df_chips)}")
    if not df_chips.empty:
        print(f"[籌碼] 欄位: {df_chips.columns.tolist()}")
        print(f"[籌碼] 日期範例: {df_chips['date'].iloc[0]} (Type: {type(df_chips['date'].iloc[0])})")
    else:
        print("❌ 籌碼資料抓取失敗！")
        return

    # 4. 檢查日期重疊
    raw_dates = set(df_raw['date_only'].unique())
    chip_dates = set(df_chips['date'].unique())
    intersection = raw_dates.intersection(chip_dates)

    print(f"\nStep 3: 檢查日期重疊")
    print(f"K 線總天數: {len(raw_dates)}")
    print(f"籌碼總天數: {len(chip_dates)}")
    print(f"重疊日期天數: {len(intersection)}")

    if len(intersection) == 0:
        print("❌ 致命錯誤：K 線與籌碼日期完全沒有重疊！")
        return

    # 5. 模擬合併過程
    print("\nStep 4: 模擬合併結果")
    merged_df = pd.merge(df_raw, df_chips, left_on='date_only', right_on='date', how='left')

    # 檢查是否有籌碼特徵名稱 (請確保欄位名稱符合)
    target_col = 'foreign_net_oi' if 'foreign_net_oi' in merged_df.columns else merged_df.columns[1]
    null_count = merged_df[target_col].isnull().sum()

    print(f"合併後總筆數: {len(merged_df)}")
    print(f"空值籌碼數量: {null_count}")

    # 6. 診斷 dropna 後結果
    print(f"\nStep 5: 檢查 fillna 後是否剩餘資料")
    merged_df[target_col] = merged_df[target_col].ffill().bfill()
    final_df = merged_df.dropna(subset=[target_col])
    print(f"處理後剩餘: {len(final_df)}")

if __name__ == "__main__":
    debug_data_pipeline()