import torch
import torch.nn as nn
import torch.nn.functional as F

class CompositeDayTradingAI(nn.Module):
    """
    複合 AI 模型：結合 1D-CNN (擷取微觀 K 線型態如雙底/洗盤) 
    與 Transformer Encoder (處理盤中時序記憶與注意力機制)
    
    參考華爾街最新高頻實踐：CNN-Transformer 架構。
    """
    def __init__(self, input_dim, d_model=64, nhead=4, num_layers=2, dropout=0.3): # 增加 dropout
        super().__init__()
        
        # 1D-CNN 特徵萃取器
        self.conv1 = nn.Conv1d(in_channels=input_dim, out_channels=d_model, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(d_model)
        self.dropout_cnn = nn.Dropout(dropout)
        
        # Transformer 編碼器
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dropout=dropout, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # 分類頭 (輸出: 3 個類別)
        self.fc1 = nn.Linear(d_model, 64) # 增加神經元
        self.dropout_fc = nn.Dropout(dropout)
        self.fc2 = nn.Linear(64, 3) 

    def forward(self, x):
        # x shape: [batch, seq_len, features]
        x = x.transpose(1, 2) # [batch, features, seq_len]
        
        # CNN 萃取
        x = self.dropout_cnn(F.relu(self.bn1(self.conv1(x))))
        
        # 轉回 Transformer 輸入格式: [batch, seq_len, d_model]
        x = x.transpose(1, 2)
        
        # Transformer 注意力
        out = self.transformer(x)
        
        # 取最後一個時間步
        last_out = out[:, -1, :]
        
        # 全連接預測
        y = self.dropout_fc(F.relu(self.fc1(last_out)))
        logits = self.fc2(y)
        
        return logits
