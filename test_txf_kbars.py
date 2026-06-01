import os
import pandas as pd
import shioaji as sj
from datetime import datetime, timedelta
from dotenv import load_dotenv

def test_futures_kbars():
    print("=" * 60)
    print("🚀 Shioaji API 台指期貨 (TXFR1) K 線連線測試 🚀")
    print("=" * 60)

    # 載入金鑰並登入
    env_path = r"F:\Gemini_CLI_Application\finance_v2\.env"
    load_dotenv(dotenv_path=env_path)
    api_key = os.environ.get('SHIOAJI_API_KEY', '')
    secret_key = os.environ.get('SHIOAJI_SECRET_KEY', '')

    api = sj.Shioaji()
    try:
        api.login(api_key, secret_key, contracts_timeout=10000)
        print("✅ Shioaji 登入成功\n")
    except Exception as e:
        print(f"❌ 登入失敗: {e}")
        return

    # 抓取台指期貨近月連續合約
    try:
        txf_contract = api.Contracts.Futures.TXF.TXFR1
        print(f"📌 測試對象: {txf_contract.symbol} ({txf_contract.name})")
    except Exception as e:
        print(f"❌ 無法定位期貨合約: {e}")
        return

    # 設定抓取過去 5 天的資料
    start_date = (datetime.now() - timedelta(days=5)).strftime("%Y-%m-%d")
    end_date = datetime.now().strftime("%Y-%m-%d")

    print(f"📊 測試抓取區間: {start_date} ~ {end_date} (要求近 5 天)")
    print("-" * 60)

    try:
        # 呼叫 kbars API 獲取資料
        kbars = api.kbars(txf_contract, start=start_date, end=end_date)
        df = pd.DataFrame({**kbars})

        if df.empty:
            print("   ➔ ⚠️ 結果: 抓取成功，但期貨 K 線回傳居然為空！(極度罕見異常)")
        else:
            df['ts'] = pd.to_datetime(df['ts'])
            print(f"   ➔ ✅ 結果: 成功抓取 {len(df)} 根 1 分鐘 K 線！")
            print(f"   ➔ 🕒 資料歷史起點: {df['ts'].iloc[0]}")
            print(f"   ➔ 🕒 資料歷史終點: {df['ts'].iloc[-1]}")

            # 將 1 分鐘 K 降頻為 5 分鐘 K (模擬實盤演算法的處理過程)
            df.set_index('ts', inplace=True)
            df_5m = df['Close'].resample('5min').last().dropna()

            print(f"\n   ➔ 📉 換算為 5 分鐘 K 線後，共獲得: {len(df_5m)} 根")
            print(f"   ➔ 💡 AI 預測只需 40 根，資料量是否達標？: {'✅ 是 (秒速啟動)' if len(df_5m) >= 40 else '❌ 否'}")

            print("\n🔍 擷取最後 5 筆 5分K 收盤價預覽：")
            print(df_5m.tail(5))

    except Exception as e:
        print(f"   ➔ ❌ 抓取失敗，API 發生報錯: {e}")

    print("-" * 60)
    print("🏁 期貨 K 線測試完畢！")

if __name__ == "__main__":
    test_futures_kbars()