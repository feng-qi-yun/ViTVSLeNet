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
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
import pandas as pd
import torch
import torch.nn as nn

#: 指标名 -> 数值（acc/loss/macro_f1 等）
Metrics = dict[str, float]

#: 一条实验记录，如 {"Noise_Std": 0.0, "Model": "LeNet", "Accuracy": 99.04, ...}
Record = dict[str, Any]


# ---------------------------------------------------------------------------
# 基础指标
# ---------------------------------------------------------------------------

def count_params(model: nn.Module) -> tuple[float, float]:
    """返回 (总参数量_M, 可训练参数量_M)。"""
    total: int = sum(p.numel() for p in model.parameters())
    trainable: int = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total / 1e6, trainable / 1e6


def classification_metrics(preds: npt.ArrayLike,
                           targets: npt.ArrayLike,
                           num_classes: int | None = None) -> Metrics:
    """由预测数组与标签数组计算 准确率(%) 和宏平均 F1。

    纯 numpy 实现，避免额外引入 sklearn 依赖。
    """
    pred: np.ndarray = np.asarray(preds)
    truth: np.ndarray = np.asarray(targets)
    n_classes: int = num_classes if num_classes is not None else int(max(truth.max(), pred.max())) + 1

    acc: float = 100.0 * float((pred == truth).mean())

    f1s: list[float] = []
    for c in range(n_classes):
        tp: int = int(np.sum((pred == c) & (truth == c)))
        fp: int = int(np.sum((pred == c) & (truth != c)))
        fn: int = int(np.sum((pred != c) & (truth == c)))
        precision: float = tp / (tp + fp) if (tp + fp) else 0.0
        recall: float = tp / (tp + fn) if (tp + fn) else 0.0
        f1: float = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
        f1s.append(f1)

    return {"acc": acc, "macro_f1": float(np.mean(f1s))}


@torch.no_grad()
def evaluate(model: nn.Module,
             data_loader: Iterable[tuple[torch.Tensor, torch.Tensor]],
             device: str | torch.device,
             criterion: nn.Module | None = None,
             amp_enabled: bool = False) -> Metrics:
    """在 data_loader 上评估模型。

    返回 {"acc": 准确率(%), "loss": 平均交叉熵损失, "macro_f1": 宏平均F1}。
    """
    model.eval()
    loss_fn: nn.Module = criterion if criterion is not None else nn.CrossEntropyLoss()

    is_cuda: bool = torch.device(device).type == "cuda"
    # 预测/标签先留在设备上累积，最后只做一次 .cpu()，避免逐 batch 同步
    all_preds: list[torch.Tensor] = []
    all_targets: list[torch.Tensor] = []
    total_loss: torch.Tensor = torch.zeros((), device=device)
    total_samples: int = 0

    for data, target in data_loader:
        data = data.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)
        with torch.amp.autocast(device_type="cuda" if is_cuda else "cpu", enabled=amp_enabled):
            output: torch.Tensor = model(data)
            loss: torch.Tensor = loss_fn(output, target)
        total_loss += loss.detach() * target.size(0)
        total_samples += target.size(0)
        all_preds.append(output.argmax(dim=1))
        all_targets.append(target)

    metrics: Metrics = classification_metrics(
        torch.cat(all_preds).cpu().numpy(),
        torch.cat(all_targets).cpu().numpy(),
    )
    metrics["loss"] = float((total_loss / max(total_samples, 1)).item())
    return metrics


# ---------------------------------------------------------------------------
# 结果持久化（统一写到 results/ 下由调用方指定）
# ---------------------------------------------------------------------------

def save_results(records: Sequence[Record], path: str | Path) -> pd.DataFrame:
    """把实验记录列表保存为 CSV，并返回 DataFrame。"""
    df: pd.DataFrame = pd.DataFrame(records)
    csv_path: str = os.fspath(path)
    os.makedirs(os.path.dirname(os.path.abspath(csv_path)), exist_ok=True)
    df.to_csv(csv_path, index=False)
    return df


def build_summary_table(df: pd.DataFrame) -> pd.DataFrame:
    """按 (噪声强度, 数据比例) 聚合，行=配置、列=模型 的准确率汇总表。"""
    summary: pd.DataFrame = df.pivot_table(
        index=["Noise_Std", "Data_Ratio"],
        columns="Model",
        values="Accuracy",
        aggfunc="mean",
    )
    return summary.round(2)
