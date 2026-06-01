import torch

print(f"PyTorch 版本: {torch.__version__}")
print(f"CUDA 是否可用: {torch.cuda.is_available()}")

if torch.cuda.is_available():
    # 獲取顯卡算力編號
    major, minor = torch.cuda.get_device_capability(0)
    print(f"顯示卡名稱: {torch.cuda.get_device_name(0)}")
    print(f"CUDA 架構等級: {major}.{minor}")

    # 測試一個簡單的矩陣運算
    x = torch.randn(10, 10).cuda()
    y = torch.matmul(x, x)
    print("✅ GPU 矩陣運算測試成功！")
else:
    print("❌ 錯誤: GPU 仍未被識別，請檢查 NVIDIA 驅動是否為最新版 (建議 560+ 版本)。")