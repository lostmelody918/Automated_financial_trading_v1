import os
import sys
import pandas as pd
import torch

# 加入父目錄以引入 DataFetcher
root_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if root_path not in sys.path:
    sys.path.insert(0, root_path)

from data_fetcher import DataFetcher
from ai_short_term.feature_engine import HFTFeatureEngine
from ai_short_term.hft_model import HFT_CNN_LSTM

class ShortTermWorkflow:
    def __init__(self, stock_id="2330"):
        self.stock_id = stock_id
        self.fetcher = DataFetcher()
        self.engine = HFTFeatureEngine()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
    def prepare_data(self, days=5):
        """模擬獲取短線數據 (1分鐘 K線)"""
        print(f"正在獲取 {self.stock_id} 的短線數據...")
        # 使用 fetch_stock_intraday 獲取 1min 數據
        df = self.fetcher.fetch_stock_intraday(self.stock_id, "2024-01-01") 
        
        if df.empty: return None
        
        # 加入特徵
        df = self.engine.add_intraday_momentum(df)
        df['OFI'] = self.engine.calculate_ofi(df.rename(columns={'Open':'bid_p', 'High':'ask_p', 'Volume':'bid_v', 'Low':'ask_v'})) # 示意用
        
        return df.dropna()

    def run_inference(self, df):
        """執行即時推論"""
        # 這裡會實作將 dataframe 轉換為 Tensor 並輸入模型的邏輯
        print("💡 執行 AI 推論中...")
        pass

if __name__ == "__main__":
    workflow = ShortTermWorkflow("2330")
    data = workflow.prepare_data()
    if data is not None:
        print(data.tail())
        print("✅ 短線工作流初始化成功")
