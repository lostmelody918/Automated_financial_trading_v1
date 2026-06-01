import os
import pandas as pd
import shioaji as sj
from datetime import datetime, timedelta
from dotenv import load_dotenv

def test_shioaji_options_kbars():
    print("=" * 60)
    print("🚀 Shioaji API 選擇權 K 線支援度極限測試 🚀")
    print("=" * 60)

    # 1. 載入金鑰並登入
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

    # 2. 獲取大盤參考價 (利用台指期近月抓取靶心)
    try:
        txf = api.Contracts.Futures.TXF.TXFR1
        snap = api.snapshots([txf])[0]
        current_idx = snap.close
        atm_strike = round(current_idx / 100) * 100
        print(f"🎯 當下期貨參考點位: {current_idx} -> 預估價平履約價: {atm_strike}\n")
    except Exception as e:
        print(f"❌ 無法獲取期貨報價: {e}")
        return

    # 3. 準備測試名單 (涵蓋三種極端狀況)
    test_contracts = []

    # [狀況A] 近月價平 Call (市場最熱)
    txo_calls = [c for c in api.Contracts.Options.TXO if c.option_right == 'Call' and c.strike_price == atm_strike]
    if txo_calls: test_contracts.append(("近月價平 Call", txo_calls[0]))

    # [狀況B] 近月極度深價外 Call (冷門合約)
    txo_otm_calls = [c for c in api.Contracts.Options.TXO if c.option_right == 'Call' and c.strike_price == atm_strike + 1500]
    if txo_otm_calls: test_contracts.append(("近月極冷深價外 Call", txo_otm_calls[0]))

    # [狀況C] 最新週選擇權 價平 Put (測試歷史長度)
    for cat in ['TX1', 'TX2', 'TX3', 'TX4', 'TX5']:
        if hasattr(api.Contracts.Options, cat):
            weekly_puts = [c for c in getattr(api.Contracts.Options, cat) if c.option_right == 'Put' and c.strike_price == atm_strike]
            if weekly_puts:
                test_contracts.append((f"近期週選({cat})價平 Put", weekly_puts[0]))
                break

    # 4. 開始執行抓取測試
    # 設定抓取過去 5 天的資料
    start_date = (datetime.now() - timedelta(days=5)).strftime("%Y-%m-%d")
    end_date = datetime.now().strftime("%Y-%m-%d")

    print(f"📊 測試抓取區間: {start_date} ~ {end_date} (要求近 5 天)")
    print("-" * 60)

    for label, contract in test_contracts:
        symbol_name = f"{contract.symbol} ({contract.delivery_month} {contract.strike_price}{'C' if contract.option_right=='Call' else 'P'})"
        print(f"📌 測試對象: [{label}] \n   代碼: {symbol_name}")

        try:
            # 呼叫 kbars API
            kbars = api.kbars(contract, start=start_date, end=end_date)
            df = pd.DataFrame({**kbars})

            if df.empty:
                print("   ➔ ⚠️ 結果: 抓取成功，但 DataFrame 回傳為空！(原因：無成交量或永豐尚未寫入歷史)")
            else:
                # Shioaji kbars 預設回傳 1 分鐘 K 線
                df['ts'] = pd.to_datetime(df['ts'])
                print(f"   ➔ ✅ 結果: 成功抓取 {len(df)} 根 1 分鐘 K 線！")
                print(f"   ➔ 🕒 資料範圍: {df['ts'].iloc[0]} ~ {df['ts'].iloc[-1]}")

                # 將 1 分鐘 K 轉換為 5 分鐘 K 看能剩下幾根
                df.set_index('ts', inplace=True)
                df_5m = df['Close'].resample('5min').last().dropna()
                print(f"   ➔ 📉 換算為 5 分 K 線後剩餘: {len(df_5m)} 根")

        except Exception as e:
            print(f"   ➔ ❌ 抓取失敗，API 發生報錯: {e}")

        print("-" * 60)

    print("🏁 測試完畢！")

if __name__ == "__main__":
    test_shioaji_options_kbars()