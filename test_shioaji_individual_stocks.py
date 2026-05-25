import os
import sys
from datetime import datetime, timedelta
import pandas as pd
import shioaji as sj
import matplotlib.pyplot as plt
import mplfinance as mpf
from dotenv import load_dotenv

load_dotenv()

def test_advanced_shioaji():
    print("Initializing Shioaji API directly for advanced tests...")
    # 若無真實 API Key 可測試，可加入 simulation=True 進入模擬模式
    api = sj.Shioaji()
    api_key = os.environ.get('SHIOAJI_API_KEY', '')
    secret_key = os.environ.get('SHIOAJI_SECRET_KEY', '')

    if not api_key or not secret_key:
        print("Missing API Keys in .env file.")
        return

    try:
        # 修正 1：加入 contracts_timeout，確保等待合約下載完成
        api.login(api_key, secret_key, contracts_timeout=1)
        print("Login and contracts download successful.")
    except Exception as e:
        print(f"Login failed: {e}")
        return

    # 修正 2：直接使用鍵值取合約，避免使用 .get()
    try:
        contract = api.Contracts.Stocks["2330"]
    except KeyError:
        print("Contract 2330 not found.")
        return

    print("\n--- 1. Testing Snapshot ---")
    try:
        snapshots = api.snapshots([contract])
        if snapshots:
            snap = snapshots[0]
            # 修正 3：Snapshot 僅有最佳一檔報價，非五檔
            print(f"🕒 資料時間 (Time): {snap.ts}")
            print(f"💰 最新成交價 (Close Price): {snap.close}")
            print(f"📊 總成交量 (Total Volume): {snap.total_volume}")
            print(f"📈 最佳委買價 (Best Bid Price): {snap.buy_price}")
            print(f"📦 最佳委買量 (Best Bid Volume): {snap.buy_volume}")
            print(f"📉 最佳委賣價 (Best Ask Price): {snap.sell_price}")
            print(f"📦 最佳委賣量 (Best Ask Volume): {snap.sell_volume}")
        else:
            print("No snapshot data returned.")
    except Exception as e:
        print(f"Snapshot error: {e}")

    print("\n--- 2. Testing Bulk 1-Minute K-bars & Quantitative Features ---")
    try:
        end_date = datetime.today()
        start_date = end_date - timedelta(days=5)

        print(f"📥 Fetching 1-min kbars from {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}...")
        kbars = api.kbars(contract, start=start_date.strftime("%Y-%m-%d"), end=end_date.strftime("%Y-%m-%d"))
        df = pd.DataFrame({**kbars})

        if not df.empty:
            df['ts'] = pd.to_datetime(df['ts'])
            df.set_index('ts', inplace=True)
            print(f"✅ Fetched {len(df)} 1-minute k-bars.")

            # 3. 計算量化特徵
            print("🧮 Calculating quantitative features (SMA, RSI, VWAP)...")
            df['SMA_10'] = df['Close'].rolling(window=10).mean()
            df['SMA_30'] = df['Close'].rolling(window=30).mean()

            # RSI
            delta = df['Close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / (loss + 1e-9)
            df['RSI_14'] = 100 - (100 / (1 + rs))

            # VWAP (日內重置)
            df['Date_Only'] = df.index.date
            df['Typical_Price'] = (df['High'] + df['Low'] + df['Close']) / 3
            df['VP'] = df['Typical_Price'] * df['Volume']
            df['Cumulative_VP'] = df.groupby('Date_Only')['VP'].cumsum()
            df['Cumulative_Vol'] = df.groupby('Date_Only')['Volume'].cumsum()
            df['VWAP'] = df['Cumulative_VP'] / (df['Cumulative_Vol'] + 1e-9)

            print("\n📊 尾部 5 筆量化特徵資料:")
            print(df[['Close', 'SMA_10', 'RSI_14', 'VWAP']].tail())

            # 4. 視覺化
            print("\n--- 3. Visualizing Data ---")
            plot_df = df.tail(200)

            add_plots = [
                mpf.make_addplot(plot_df['SMA_10'], color='blue', width=1),
                mpf.make_addplot(plot_df['VWAP'], color='orange', width=1.5, linestyle='--'),
                mpf.make_addplot(plot_df['RSI_14'], panel=2, color='purple', ylabel='RSI')
            ]

            mc = mpf.make_marketcolors(up='red', down='green', inherit=True)
            s = mpf.make_mpf_style(marketcolors=mc, gridstyle=':')

            output_file = "shioaji_advanced_test.png"
            print(f"🖼️ Rendering and saving plot to {output_file}...")
            mpf.plot(plot_df, type='candle', volume=True, addplot=add_plots, style=s,
                     title=f"2330 Intraday (1-min) Analysis",
                     savefig=dict(fname=output_file, dpi=120, bbox_inches='tight'),
                     figsize=(12, 8))
            print("✅ Visualization saved successfully.")

        else:
            print("⚠️ No k-bar data fetched. (可能是假日休市或超出資料範圍)")
    except Exception as e:
        print(f"K-bars processing error: {e}")

if __name__ == "__main__":
    test_advanced_shioaji()