import requests
import re
from datetime import datetime, timedelta

def test_fetch_chips_adaptive():
    print("=" * 50)
    print("🚀 啟動：自適應期交所籌碼解析器 (Adaptive Regex Mode)")
    print("=" * 50)

    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
        'Accept': 'text/csv,text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
        'Accept-Language': 'zh-TW,zh;q=0.9,en-US;q=0.8,en;q=0.7',
        'Connection': 'keep-alive',
        'Content-Type': 'application/x-www-form-urlencoded',
        'Origin': 'https://www.taifex.com.tw',
        'Referer': 'https://www.taifex.com.tw/cht/3/futContractsDate'
    }

    end_date = datetime.now()
    start_date = end_date - timedelta(days=7)
    s_dt = start_date.strftime("%Y/%m/%d")
    e_dt = end_date.strftime("%Y/%m/%d")

    print(f"📅 查詢區間設定: {s_dt} ~ {e_dt}")

    try:
        session = requests.Session()
        session.headers.update(headers)

        # ==========================================
        # 1. 抓取 P/C Ratio
        # ==========================================
        print("\n📡 [1/2] 正在抓取 P/C Ratio 資料...")
        res_pc = session.post("https://www.taifex.com.tw/cht/3/pcRatioDown",
                               data={"queryStartDate": s_dt, "queryEndDate": e_dt},
                               timeout=10)

        if res_pc.status_code != 200: raise ValueError("連線失敗")

        # 移除非必要的空白與換行，把資料變成一整條長字串方便搜尋
        content_pc = res_pc.content.decode('cp950', errors='ignore')

        # ==========================================
        # 2. 抓取三大法人留倉 (使用與 data_engine.py 一致的參數)
        # ==========================================
        print("\n📡 [2/2] 正在抓取三大法人期貨留倉...")
        res_oi = session.post("https://www.taifex.com.tw/cht/3/futContractsDateDown",
                               data={
                                   "queryStartDate": s_dt, 
                                   "queryEndDate": e_dt, 
                                   "commodityId": "TXF"
                               },
                               timeout=10)

        if res_oi.status_code != 200:
            content_oi = ""
        else:
            content_oi = res_oi.content.decode('cp950', errors='ignore')
            if '<html' in content_oi.lower():
                content_oi = ""

        # ==========================================
        # 3. 嘗試尋找兩邊都有資料的最新日期 (自適應回溯)
        # ==========================================
        print("\n🔍 正在尋找最新且完整的籌碼資料...")
        
        valid_pc_lines = [line for line in content_pc.split('\n') if re.match(r'^20\d{2}/\d{2}/\d{2}', line.strip())]
        if not valid_pc_lines: raise ValueError("找不到 P/C Ratio 數據格式")

        latest_date_str = ""
        pc_ratio = 0.0
        foreign_net_oi = 0.0
        dealer_net_oi = 0.0
        trust_net_oi = 0.0
        found_data = False

        # 從最近的日期往回找
        for pc_line in valid_pc_lines:
            line_pc_clean = pc_line.replace('"', '').replace(' ', '')
            parts_pc = line_pc_clean.split(',')
            current_date = parts_pc[0]
            
            print(f"   📡 正在檢查日期: {current_date}...")

            # 檢查 OI 資料中是否有此日期
            # 如果之前的範圍抓取失敗或不含此日期，則嘗試單日抓取
            current_content_oi = content_oi
            if not current_content_oi or current_date not in current_content_oi:
                # 嘗試單日抓取 (有些介面支援 queryDate)
                res_single = session.post("https://www.taifex.com.tw/cht/3/futContractsDateDown",
                                         data={
                                             "queryStartDate": current_date, 
                                             "queryEndDate": current_date,
                                             "commodityId": "TXF"
                                         },
                                         timeout=10)
                if res_single.status_code == 200:
                    current_content_oi = res_single.content.decode('cp950', errors='ignore')
                    if '<html' in current_content_oi.lower(): current_content_oi = ""
                else:
                    continue

            # 檢查是否包含有效資料 (臺股期貨)
            if '臺股期貨' not in current_content_oi or current_date not in current_content_oi:
                continue

            # 找到匹配日期！解析 P/C Ratio
            try:
                pc_ratio = float(parts_pc[-1].replace('%', '')) / 100.0
            except (ValueError, IndexError):
                pc_ratio = float(parts_pc[-2].replace('%', '')) / 100.0
            
            latest_date_str = current_date
            
            # 解析三大法人留倉
            # 注意：這裡要從當前內容中過濾出該日期的行
            valid_oi_lines = [line.replace('"', '').replace(' ', '') for line in current_content_oi.split('\n')
                              if current_date in line and '臺股期貨' in line]
            
            for line in valid_oi_lines:
                parts = line.split(',')
                identity = ""
                if '外資' in line: identity = "外資"
                elif '自營商' in line: identity = "自營商"
                elif '投信' in line: identity = "投信"
                else: continue

                numeric_values = [float(p.strip()) for p in parts if re.match(r'^-?\d+$', p.strip())]
                if len(numeric_values) >= 3:
                    net_oi = numeric_values[-2]
                    if identity == "外資": foreign_net_oi = net_oi
                    elif identity == "自營商": dealer_net_oi = net_oi
                    elif identity == "投信": trust_net_oi = net_oi
            
            found_data = True
            print(f"   ✅ 成功獲取 {latest_date_str} 的完整資料!")
            break
        
        if not found_data:
            raise ValueError(f"在查詢區間內找不到任何完整的三大法人與 P/C Ratio 交叉資料")

        print(f"   ✅ 三大法人解析成功!")
        print("-" * 50)
        print(f"🎯 最終解析結果 ({latest_date_str})")
        print(f"   外資淨未平倉 : {foreign_net_oi:,.0f} 口")
        print(f"   投信淨未平倉 : {trust_net_oi:,.0f} 口")
        print(f"   自營淨未平倉 : {dealer_net_oi:,.0f} 口")
        print(f"   P/C Ratio    : {pc_ratio:.4f}")
        print("-" * 50)
        print("🎉 測試通過！")

    except Exception as e:
        print("\n❌ 測試失敗！發生錯誤:")
        print(f"   {type(e).__name__}: {str(e)}")

if __name__ == "__main__":
    test_fetch_chips_adaptive()