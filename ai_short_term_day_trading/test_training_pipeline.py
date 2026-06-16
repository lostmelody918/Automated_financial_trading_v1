import torch
import pandas as pd
import numpy as np
import os
import sys
import unittest
from datetime import datetime, timedelta

# 加入當前目錄到路徑以匯入模組
sys.path.append(os.path.dirname(__file__))

from data_engine import DayTradingDataEngine
from composite_ai import MultiTimeframeCompositeAI
from train_model import vectorized_triple_barrier

class TestTrainingPipeline(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        print("\n🧪 開始全面性系統測試...")
        cls.engine = DayTradingDataEngine()
        # 建立模擬數據以進行邏輯測試
        # 增加長度以確保 15m 切片有足夠數據
        dates = pd.date_range(start='2026-01-01', periods=5000, freq='1min')
        cls.mock_df = pd.DataFrame({
            'date': dates,
            'Open': np.random.randn(5000).cumsum() + 20000,
            'High': np.random.randn(5000).cumsum() + 20010,
            'Low': np.random.randn(5000).cumsum() + 19990,
            'Close': np.random.randn(5000).cumsum() + 20000,
            'Volume': np.random.randint(100, 1000, 5000),
            'atr': np.random.uniform(5, 20, 5000)
        })
        cls.mock_df['date_only'] = cls.mock_df['date'].dt.date

    def test_data_engine_fetching(self):
        """測試 1: 資料獲取全面性與格式"""
        print("🔎 測試 1: 資料獲取全面性...")
        # 測試抓取少量數據
        df = self.engine.fetch_intraday_data(days=2)
        self.assertFalse(df.empty, "數據抓取不應為空")
        self.assertIn('Close', df.columns, "必須包含價格欄位")
        self.assertIn('vwap_bias', df.columns, "必須包含特徵工程欄位")
        print(f"✅ 資料格式正確，總欄位數: {len(df.columns)}")

    def test_label_correctness(self):
        """測試 2: 三重屏障標籤邏輯正確性"""
        print("🔎 測試 2: 標籤正確性...")
        labels = vectorized_triple_barrier(self.mock_df)
        self.assertEqual(len(labels), len(self.mock_df), "標籤長度需與數據一致")
        unique_labels = labels.unique()
        # 標籤應在 0-6 之間
        for lbl in unique_labels:
            self.assertTrue(0 <= lbl <= 6, f"無效標籤: {lbl}")
        print(f"✅ 標籤分佈正常: {labels.value_counts().to_dict()}")

    def test_mtf_logic(self):
        """測試 3: 多時間尺度 (MTF) 切片邏輯"""
        print("🔎 測試 3: MTF 切片邏輯...")
        df_1m = self.mock_df.copy()
        df_15m_base = df_1m.set_index('date').resample('15min').last().ffill()
        
        win_1m, win_15m = 40, 20
        # 測試單個樣本生成，確保索引足夠大以包含歷史
        i = 1000
        curr_time = df_1m['date'].iloc[i]
        idx_15m = df_15m_base.index.get_indexer([curr_time], method='pad')[0]
        
        self.assertTrue(idx_15m >= 0, "15m 索引定位失敗")
        
        x1m = df_1m.iloc[i-win_1m:i]
        x15m = df_15m_base.iloc[idx_15m-win_15m:idx_15m]
        
        self.assertEqual(len(x1m), 40)
        self.assertEqual(len(x15m), 20)
        print(f"✅ MTF 對齊正確: 1m_idx={i}, 15m_idx={idx_15m}")

    def test_model_architecture(self):
        """測試 4: 模型架構與 Tensor Shape"""
        print("🔎 測試 4: 模型架構與前向傳播...")
        input_dim = 51
        model = MultiTimeframeCompositeAI(input_dim=input_dim, d_model=128, nhead=4, num_layers=2)
        
        # 模擬 Batch Input
        batch_size = 8
        dummy_1m = torch.randn(batch_size, 40, input_dim)
        dummy_15m = torch.randn(batch_size, 20, input_dim)
        
        logits = model(dummy_1m, dummy_15m)
        self.assertEqual(logits.shape, (batch_size, 7), "模型輸出 Shape 錯誤")
        
        # 檢查位置編碼是否存在
        self.assertIsNotNone(model.pos_embed_1m)
        self.assertIsNotNone(model.pos_embed_15m)
        print("✅ 模型前向傳播測試通過")

    def test_training_loop_integration(self):
        """測試 5: 訓練迴圈集成測試 (Small Run)"""
        print("🔎 測試 5: 訓練迴圈整合測試...")
        # 此測試確保 optimizer.step() 與 loss 計算沒有 Runtime Error
        input_dim = 51
        model = MultiTimeframeCompositeAI(input_dim=input_dim, d_model=64, nhead=4, num_layers=1)
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)
        
        x1m = torch.randn(4, 40, input_dim)
        x15m = torch.randn(4, 20, input_dim)
        y = torch.tensor([0, 3, 6, 1], dtype=torch.long)
        
        logits = model(x1m, x15m)
        loss = torch.nn.functional.cross_entropy(logits, y)
        loss.backward()
        optimizer.step()
        
        self.assertGreater(loss.item(), 0, "Loss 應大於 0")
        print(f"✅ 訓練迴圈整合測試通過, 初始 Loss: {loss.item():.4f}")

if __name__ == "__main__":
    unittest.main()
