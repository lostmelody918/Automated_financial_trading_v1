import os
import pandas as pd
from datetime import datetime, timedelta
from data_engine import DayTradingDataEngine

class TaifexTickToKbarEngine:
    def __init__(self, start_date: str, end_date: str, target_symbol: str = 'TX', output_dir: str = 'data_learn'):
        """
        重構版本：直接利用 DayTradingDataEngine (Shioaji) 進行歷史 K 線抓取
        解決原版期交所 ZIP 檔僅保留 30 天導致無法抓取長週期的 Bug。
        """
        self.start_date = datetime.strptime(start_date, '%Y-%m-%d')
        self.end_date = datetime.strptime(end_date, '%Y-%m-%d')
        self.target_symbol = target_symbol
        self.output_dir = output_dir

        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

        self.output_file = os.path.join(self.output_dir, f"{self.target_symbol}_1min_continuous.csv")
        self.days_to_fetch = (self.end_date - self.start_date).days

    def run(self):
        """執行主程序：透過 Shioaji 抓取資料並輸出"""
        print(f"🚀 開始抓取 {self.start_date.date()} 至 {self.end_date.date()} 之 {self.target_symbol} 1 分鐘 K 線資料...")
        print(f"使用 Shioaji 引擎抓取過去 {self.days_to_fetch} 天歷史資料...")
        
        engine = DayTradingDataEngine()
        df_kbars = engine.fetch_intraday_data(days=self.days_to_fetch)
        
        if df_kbars.empty:
            print("❌ 未獲取任何有效資料。")
            return
            
        print("\n🧮 正在合併全區間資料與最終排序...")
        
        # 為了相容之前的格式，保留基本的 OHLCV 欄位
        if 'date' in df_kbars.columns:
            df_kbars.rename(columns={'date': 'datetime'}, inplace=True)
            
        # 選擇並排序列
        columns_to_keep = ['datetime', 'Open', 'High', 'Low', 'Close', 'Volume']
        # 確保所有需要的欄位都存在
        for col in columns_to_keep:
            if col not in df_kbars.columns:
                df_kbars[col] = 0
                
        final_df = df_kbars[columns_to_keep].copy()
        final_df.set_index('datetime', inplace=True)
        final_df.sort_index(inplace=True)
        
        # 剔除重複的網格
        final_df = final_df[~final_df.index.duplicated(keep='last')]
        
        # 輸出檔案
        final_df.to_csv(self.output_file)
        print(f"✅ 大功告成！完美 1 分鐘 K 線已儲存至: {self.output_file}")
        print(f"📊 總資料筆數: {len(final_df)} 筆 (足以支撐 CNN-Transformer 的胃口)")

# ================= 執行區塊 =================
if __name__ == "__main__":
    # 設定目標抓取過去 2 年 (730天)
    end_dt = datetime.now()
    start_dt = end_dt - timedelta(days=730)

    crawler = TaifexTickToKbarEngine(
        start_date=start_dt.strftime('%Y-%m-%d'),
        end_date=end_dt.strftime('%Y-%m-%d'),
        target_symbol='TX',  # TX 為大台指
        output_dir='data_learn'
    )

    crawler.run()