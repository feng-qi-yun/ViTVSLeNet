"""评估与结果统计工具。

包含:
- count_params: 统计模型参数量（单位: M）
- evaluate: 在测试集上评估模型，返回 准确率(%) / 平均损失 / 宏平均F1
- classification_metrics: 由预测与标签直接计算分类指标（不依赖 sklearn）
- save_results / build_summary_table: 把实验记录写成 CSV 汇总表

本模块的所有输出路径都由调用方（main.py）传入，统一写到 results/ 下。
"""

from __future__ import annotations

import os

import numpy as np
import pandas as pd
import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# 基础指标
# ---------------------------------------------------------------------------

def count_params(model: nn.Module) -> tuple[float, float]:
    """返回 (总参数量_M, 可训练参数量_M)。"""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total / 1e6, trainable / 1e6


def classification_metrics(preds, targets, num_classes: int | None = None) -> dict:
    """由预测数组与标签数组计算 准确率(%) 和宏平均 F1。

    纯 numpy 实现，避免额外引入 sklearn 依赖。
    """
    preds = np.asarray(preds)
    targets = np.asarray(targets)
    if num_classes is None:
        num_classes = int(max(targets.max(), preds.max())) + 1

    acc = 100.0 * float((preds == targets).mean())

    f1s = []
    for c in range(num_classes):
        tp = np.sum((preds == c) & (targets == c))
        fp = np.sum((preds == c) & (targets != c))
        fn = np.sum((preds != c) & (targets == c))
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
        f1s.append(f1)

    return {"acc": acc, "macro_f1": float(np.mean(f1s))}


@torch.no_grad()
def evaluate(model: nn.Module, data_loader, device: str, criterion: nn.Module | None = None, amp_enabled: bool = False) -> dict:
    """在 data_loader 上评估模型。

    返回 {"acc": 准确率(%), "loss": 平均交叉熵损失, "macro_f1": 宏平均F1}。
    """
    model.eval()
    if criterion is None:
        criterion = nn.CrossEntropyLoss()

    is_cuda = torch.device(device).type == "cuda"
    # 预测/标签先留在设备上累积，最后只做一次 .cpu()，避免逐 batch 同步
    all_preds: list[torch.Tensor] = []
    all_targets: list[torch.Tensor] = []
    total_loss = torch.zeros((), device=device)
    total_samples = 0

    for data, target in data_loader:
        data = data.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)
        with torch.amp.autocast(device_type="cuda" if is_cuda else "cpu", enabled=amp_enabled):
            output = model(data)
            loss = criterion(output, target)
        total_loss += loss.detach() * target.size(0)
        total_samples += target.size(0)
        all_preds.append(output.argmax(dim=1))
        all_targets.append(target)

    metrics = classification_metrics(
        torch.cat(all_preds).cpu().numpy(),
        torch.cat(all_targets).cpu().numpy(),
    )
    metrics["loss"] = float((total_loss / max(total_samples, 1)).item())
    return metrics


# ---------------------------------------------------------------------------
# 结果持久化（统一写到 results/ 下由调用方指定）
# ---------------------------------------------------------------------------

def save_results(records: list[dict], path: str) -> pd.DataFrame:
    """把实验记录列表保存为 CSV，并返回 DataFrame。"""
    df = pd.DataFrame(records)
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    df.to_csv(path, index=False)
    return df


def build_summary_table(df: pd.DataFrame) -> pd.DataFrame:
    """按 (噪声强度, 数据比例) 聚合，行=配置、列=模型 的准确率汇总表。"""
    summary = df.pivot_table(
        index=["Noise_Std", "Data_Ratio"],
        columns="Model",
        values="Accuracy",
        aggfunc="mean",
    )
    return summary.round(2)
