import torch
import torch.nn as nn
import torch.nn.functional as F

class HFT_CNN_LSTM(nn.Module):
    """
    HFT 專用混合模型:
    1D-CNN 提取局部波段特徵 (Spatial)
    LSTM 捕捉長程訂單流相依性 (Temporal)
    """
    def __init__(self, input_dim, hidden_dim=64):
        super(HFT_CNN_LSTM, self).__init__()
        
        # 1. CNN 層：捕捉微觀波動模式
        self.conv1 = nn.Conv1d(in_channels=input_dim, out_channels=32, kernel_size=3, padding=1)
        self.pool = nn.MaxPool1d(kernel_size=2)
        
        # 2. LSTM 層：處理時間序列
        self.lstm = nn.LSTM(input_size=32, hidden_size=hidden_dim, num_layers=1, batch_first=True)
        
        # 3. 全連接層：輸出買賣信號 (分類：漲/跌/盤)
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, 32),
            nn.LeakyReLU(0.1),
            nn.Dropout(0.2),
            nn.Linear(32, 3) # 輸出: [Down, Neutral, Up]
        )

    def forward(self, x):
        # x shape: [batch, seq_len, features] -> [batch, features, seq_len] for Conv1d
        x = x.transpose(1, 2)
        x = F.leaky_relu(self.conv1(x))
        x = self.pool(x)
        
        # Back to LSTM: [batch, channels, pooled_seq_len] -> [batch, pooled_seq_len, channels]
        x = x.transpose(1, 2)
        lstm_out, _ = self.lstm(x)
        
        # 取最後一個時間點的輸出
        last_out = lstm_out[:, -1, :]
        return self.fc(last_out)
