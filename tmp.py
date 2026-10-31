import torch
print(torch.cuda.is_available())  # 应输出 True
print(torch.version.cuda)         # 显示 PyTorch 使用的 CUDA 版本