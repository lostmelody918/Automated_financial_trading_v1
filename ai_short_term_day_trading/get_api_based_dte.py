from datetime import datetime

def get_api_based_dte(active_contract, current_time):
    """
    🚀 利用 Shioaji API 合約屬性精準計算 DTE (完全免疫國定假日與特殊契約)
    """
    try:
        # 1. 從合約物件安全提取交割日字串
        delivery_date_str = getattr(active_contract, 'delivery_date', None)

        if not delivery_date_str:
            raise ValueError("合約物件缺乏 delivery_date 屬性")

        # 2. 統一格式化 (消除斜線，統一變為 YYYYMMDD)
        clean_date_str = delivery_date_str.replace('/', '')

        # 3. 綁定台指期/選擇權的法定結算時間 (13:30:00)
        settlement_time = datetime.strptime(f"{clean_date_str} 13:30:00", "%Y%m%d %H:%M:%S")

        # 4. 計算精確剩餘秒數並轉為天數
        delta = settlement_time - current_time
        dte_days = delta.total_seconds() / 86400.0

        # 5. 防禦機制：如果已經超過結算時間，給予極小值避免 BSM 分母除以 0
        return max(dte_days, 0.001)

    except Exception as e:
        print(f"⚠️ 解析 API 交割日失敗 ({e})，啟動降級防禦 (預設 1 天)")
        return 1.0  # 降級保護