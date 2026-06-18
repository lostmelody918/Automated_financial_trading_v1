import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiTimeframeCompositeAI(nn.Module):
    """
    進階多時間尺度 AI 模型：
    1. 雙分支 1D-CNN：使用 Left-Padding 確保嚴格因果律。
    2. 可學習位置編碼 (Learnable Positional Encoding)。
    3. 雙 Transformer 編碼器。
    4. 特徵拼接 (Concatenation) 與多層感知機 (MLP) 決策。
    """
    def __init__(self, input_dim, d_model=256, nhead=8, num_layers=3, dropout=0.3, seq_len_1m=40, seq_len_15m=20):
        super().__init__()
        self.d_model = d_model

        # --- 1分鐘線分支 (微觀) ---
        # 移除 padding=1，改為手動 left padding 確保因果
        self.conv1m = nn.Conv1d(in_channels=input_dim, out_channels=d_model, kernel_size=3)
        self.ln1m = nn.LayerNorm(d_model)
        self.pos_embed_1m = nn.Parameter(torch.zeros(1, seq_len_1m, d_model))

        encoder_layer_1m = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dropout=dropout, batch_first=True)
        self.transformer_1m = nn.TransformerEncoder(encoder_layer_1m, num_layers=num_layers)

        # --- 15分鐘線分支 (全局) ---
        self.conv15m = nn.Conv1d(in_channels=input_dim, out_channels=d_model, kernel_size=3)
        self.ln15m = nn.LayerNorm(d_model)
        self.pos_embed_15m = nn.Parameter(torch.zeros(1, seq_len_15m, d_model))

        encoder_layer_15m = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dropout=dropout, batch_first=True)
        self.transformer_15m = nn.TransformerEncoder(encoder_layer_15m, num_layers=num_layers)

        # --- 融合決策層 ---
        self.fc1 = nn.Linear(d_model * 2, 64)
        self.dropout_fc = nn.Dropout(dropout)
        self.fc2 = nn.Linear(64, 7)

    def forward(self, x_1m, x_15m):
        """
        x_1m: [batch, 40, features]
        x_15m: [batch, 20, features]
        """
        # --- 1分鐘分支 ---
        x1 = F.pad(x_1m.transpose(1, 2), (2, 0))

        # 先卷積，接著立刻翻轉維度，然後才送進 LayerNorm 與 ReLU
        x1 = self.conv1m(x1).transpose(1, 2)  # 翻轉後變為 [batch, 40, d_model]
        x1 = F.relu(self.ln1m(x1))            # 現在 LayerNorm 才能正確處理

        x1 = x1 + self.pos_embed_1m
        out1 = self.transformer_1m(x1)[:, -1, :]

        # 2. 處理 15分鐘分支
        x15 = F.pad(x_15m.transpose(1, 2), (2, 0))
        x15 = self.conv15m(x15).transpose(1, 2) # 🚀 修正翻轉
        x15 = F.relu(self.ln15m(x15))

        x15 = x15 + self.pos_embed_15m
        out15 = self.transformer_15m(x15)[:, -1, :]

        # 3. 特徵拼接與最終決策
        combined = torch.cat([out1, out15], dim=-1)
        y = self.dropout_fc(F.relu(self.fc1(combined)))
        logits = self.fc2(y)

        return logits
