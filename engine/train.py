"""训练与评估引擎。

为提升 GPU 利用率做的处理：
- AMP 混合精度（torch.amp.autocast + GradScaler），减少显存带宽与部分算子开销
- 计时前调用 torch.cuda.synchronize()，避免异步 kernel 让耗时统计失真
- 数据已在显存中（见 loader/dataLoader.py 的 GpuBatchLoader），CPU 只负责发射 kernel
- 逐 batch 不做 .item() 同步，kernel 发射可连续排队
- 返回吞吐率（img/s）便于观察 GPU 是否真在干活
"""

import time

import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

from utils.metrics import count_params, evaluate


def _sync(device: str) -> None:
    """CUDA 异步执行，计时/测速前先同步。"""
    if torch.device(device).type == "cuda":
        torch.cuda.synchronize()


def train_and_evaluate(model, train_loader, test_loader, epochs: int = 5, lr: float = 1e-3,
                       device: str = "cpu", amp: bool = True, verbose: bool = True) -> dict:
    """训练并评估一个模型，返回统计字典。

    返回字段:
        acc             测试集准确率 (%)
        loss            测试集平均交叉熵
        macro_f1        测试集宏平均 F1
        params_M        总参数量 (M)
        time            训练耗时 (秒, 仅训练阶段)
        samples_per_sec 训练吞吐 (img/s)
    """
    model = model.to(device)
    is_cuda = torch.device(device).type == "cuda"
    amp_enabled = amp and is_cuda

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)

    total_params_M, _ = count_params(model)
    total_train_samples = len(train_loader.dataset)

    running_loss = torch.zeros((), device=device)
    _sync(device)
    start = time.perf_counter()

    model.train()
    for epoch in tqdm(range(epochs), desc="train", leave=False, disable=not verbose):
        for data, target in train_loader:
            data = data.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast(device_type="cuda" if is_cuda else "cpu", enabled=amp_enabled):
                output = model(data)
                loss = criterion(output, target)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            # 用张量累加、全程不调用 .item()，避免每个 batch 同步一次 GPU，
            # 让 kernel 发射能连续排队，显著提升 GPU 利用率
            running_loss += loss.detach()

    _sync(device)
    train_time = time.perf_counter() - start

    test_metrics = evaluate(model, test_loader, device, amp_enabled=amp_enabled)
    return {
        "acc": test_metrics["acc"],
        "loss": test_metrics["loss"],
        "macro_f1": test_metrics["macro_f1"],
        "params_M": total_params_M,
        "train_loss": float((running_loss / max(len(train_loader) * epochs, 1)).item()),
        "time": train_time,
        "samples_per_sec": (epochs * total_train_samples) / train_time if train_time > 0 else 0.0,
    }
