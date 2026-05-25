import os
import math
import numpy as np
import pandas as pd
import shioaji as sj
from datetime import datetime, timedelta
from scipy.stats import norm
from dotenv import load_dotenv

load_dotenv()

# ==========================================
# 隱含波動率 (IV) 計算引擎
# 使用 Newton-Raphson 方法與 Black-Scholes 模型推導 IV
# ==========================================
def calculate_iv(market_price, S, K, T, r, option_type='Call'):
    """
    market_price: 選擇權市場價格 (權利金)
    S: 標的資產目前價格 (台指期貨近月報價)
    K: 履約價
    T: 剩餘到期年限 (天數 / 365)
    r: 無風險利率 (例如 0.01 代表 1%)
    """
    if T <= 0 or market_price <= 0:
        return np.nan

    MAX_ITERATIONS = 100
    PRECISION = 1.0e-5
    sigma = 0.5 # 初始猜測波動率 50%

    for i in range(MAX_ITERATIONS):
        d1 = (math.log(S / K) + (r + sigma ** 2 / 2) * T) / (sigma * math.sqrt(T))
        d2 = d1 - sigma * math.sqrt(T)

        if option_type.lower() == 'call':
            price = S * norm.cdf(d1) - K * math.exp(-r * T) * norm.cdf(d2)
            vega = S * norm.pdf(d1) * math.sqrt(T)
        else:
            price = K * math.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
            vega = S * norm.pdf(d1) * math.sqrt(T)

        diff = market_price - price
        if abs(diff) < PRECISION:
            return sigma
        if vega == 0.0:
            break
        sigma = sigma + diff / vega

    return sigma

# ==========================================
# 主程式
# ==========================================
def test_advanced_options_shioaji():
    print("Initializing Shioaji API for Advanced Options Testing...")
    api = sj.Shioaji()
    api_key = os.environ.get('SHIOAJI_API_KEY', '')
    secret_key = os.environ.get('SHIOAJI_SECRET_KEY', '')

    if not api_key or not secret_key:
        print("Missing API Keys in .env file.")
        return

    try:
        api.login(api_key, secret_key, contracts_timeout=10000)
        print("Login successful. Contracts downloaded.")
    except Exception as e:
        print(f"Login failed: {e}")
        return

    # ==========================================
    # 步驟 A：動態獲取台指期近月報價
    # ==========================================
    print("\n🔍 正在獲取即時台指期 (TXF) 近月報價...")
    try:
        tx_contract = min(
            [x for x in api.Contracts.Futures.TXF if x.code[-2:] not in ["R1", "R2"]],
            key=lambda x: x.delivery_date
        )
    except AttributeError:
        print("⚠️ 讀取期貨合約失敗。")
        return

    tx_snap = api.snapshots([tx_contract])
    if tx_snap and tx_snap[0].close > 0:
        current_spot_price = tx_snap[0].close
        print(f"📈 抓取成功！目前台指期近月 ({tx_contract.symbol}) 報價為: {current_spot_price}")
    else:
        print("⚠️ 無法獲取報價，切換至預設值...")
        current_spot_price = 21500

    # ==========================================
    # 步驟 B：全自動鎖定選擇權目標合約 (加入防彈過濾機制)
    # ==========================================
    # 1. 強制清除月份字串潛在的空白
    target_month = str(tx_contract.delivery_month).strip()

    # 2. 定義一個「絕對不會出錯」的買權判斷邏輯
    def is_call(contract):
        # 將物件轉為字串並轉小寫，無論是 Enum "OptionRight.Call" 還是 "Call" 都能命中
        val = str(contract.option_right).lower()
        return 'call' in val or val == 'c'

    print(f"\n🎯 系統將選擇權目標設定為: {target_month} 月份合約")

    txo_contracts = api.Contracts.Options.TXO

    # 使用新邏輯過濾出所有的履約價
    available_strikes = sorted(list(set([
        c.strike_price for c in txo_contracts
        if str(c.delivery_month).strip() == target_month and is_call(c)
    ])))

    # 【除錯機制】如果還是找不到，印出市場上實際有的月份來 debug
    if not available_strikes:
        all_months = sorted(list(set([str(c.delivery_month).strip() for c in txo_contracts])))
        print(f"⚠️ 找不到 {target_month} 的 Call 合約。")
        print(f"💡 系統目前抓取到的 TXO 所有可用月份為: {all_months}")
        if len(txo_contracts) > 0:
            print(f"💡 觀察第一個合約的原始屬性 -> 月份: '{txo_contracts[0].delivery_month}', 買賣權: '{txo_contracts[0].option_right}'")
        return

    # 找出最佳價平履約價 (ATM)
    target_strike = min(available_strikes, key=lambda x: abs(x - current_spot_price))
    print(f"🎯 系統根據現價 {current_spot_price}，自動鎖定最佳價平履約價: {target_strike}")

    target_contract_list = [
        c for c in txo_contracts
        if str(c.delivery_month).strip() == target_month
        and c.strike_price == target_strike
        and is_call(c)
    ]

    contract = target_contract_list[0]
    print(f"✅ 成功鎖定合約: {contract.code} ({contract.symbol})")

    # ==========================================
    # 步驟 C：獲取快照與流動性檢查
    # ==========================================
    print("\n--- 1. 獲取快照與流動性檢查 ---")
    snapshots = api.snapshots([contract])
    if snapshots:
        snap = snapshots[0]
        #輸出為奈秒,轉秒單位
        readable_time = datetime.fromtimestamp(snap.ts / 1e9)
        print(f"🕒 資料時間: {readable_time.strftime('%Y-%m-%d %H:%M:%S.%f')}")
        print(f"💰 最新成交價: {snap.close}")
        print(f"📊 總成交量: {snap.total_volume}")
        if snap.total_volume == 0 or snap.close == 0:
            print("🚨 警告：此合約今日無成交量或無報價。")
    else:
        print("No snapshot data returned.")

    # ==========================================
    # 步驟 D & E：獲取 K 線與計算隱含波動率 (IV)
    # ==========================================
    print("\n--- 2. 獲取並處理歷史 K 線 ---")
    end_date = datetime.today()
    start_date = end_date - timedelta(days=5)

    kbars = api.kbars(contract, start=start_date.strftime("%Y-%m-%d"), end=end_date.strftime("%Y-%m-%d"))
    df = pd.DataFrame({**kbars})

    if not df.empty:
        df['ts'] = pd.to_datetime(df['ts'])
        df.set_index('ts', inplace=True)

        print("☀️ 過濾夜盤資料，僅保留日盤 (08:45 ~ 13:45)...")
        df_day = df.between_time('08:45', '13:45').copy()

        if df_day.empty:
            print("⚠️ 過濾後無日盤資料。")
            return

        print(f"✅ 過濾後剩餘 {len(df_day)} 筆日盤 K 線。")

        print("🧮 計算日盤 K 線的隱含波動率 (IV)...")
        RISK_FREE_RATE = 0.01

        try:
            # 確保日期轉換正確
            expiry_date = datetime.strptime(str(tx_contract.delivery_date).strip(), "%Y/%m/%d")
        except ValueError:
            # 備用方案：若 API 的 delivery_date 格式異常，假定為當月第 3 個週三
            expiry_date = end_date + timedelta(days=14)

        def apply_iv(row):
            days_to_expiry = (expiry_date - row.name).days
            T = max(days_to_expiry / 365.0, 0.001)
            return calculate_iv(
                market_price=row['Close'],
                S=current_spot_price,
                K=target_strike,
                T=T,
                r=RISK_FREE_RATE,
                option_type="Call"
            )

        df_day['IV'] = df_day.apply(apply_iv, axis=1)

        print("\n📊 尾部 5 筆期權資料 (包含 IV):")
        df_day['IV_%'] = (df_day['IV'] * 100).round(2).astype(str) + '%'
        print(df_day[['Close', 'Volume', 'IV_%']].tail())

    else:
        print("⚠️ 未抓取到 K 線資料。(可能是非交易時段或該履約價無流動性)")

if __name__ == "__main__":
    test_advanced_options_shioaji()