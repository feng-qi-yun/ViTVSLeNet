"""课本模型 + 课本优化的定量评估实验入口。

研究问题：在完全相同的训练协议下，Dropout 与 ResNet 元素（残差连接 + BatchNorm）
分别能把准确率、噪声鲁棒性与训练代价改到什么程度？

变体矩阵（6 个）：
    LeNet                     基准（无 Dropout）
    LeNet+Dropout             + Dropout(0.3)
    LeNet+ResNet              + 残差连接 + BatchNorm
    ResLeNet                  + Dropout(0.3) + 残差连接 + BatchNorm
    MiniViT                   基准（无 Dropout）
    MiniViT+Dropout           + Dropout(0.1)

运行方式:
    python main.py

产物 (全部写到 results/ 下):
    results/tables/experiment_results.csv   每条实验记录（含参数量/耗时）
    results/tables/accuracy_summary.csv     按配置聚合的准确率汇总表
    results/figures/*.png                   准确率曲线、参数量、汇总图
    results/logs/experiment.log             运行日志
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Callable

import pandas as pd
import torch
import torch.nn as nn

from loader.dataLoader import BatchLoader, dataLoader
from models.LeNet_5 import LeNet
from models.mini_vit import MiniViT
from models.res_lenet import ResLeNet
from engine.train import TrainStats, train_and_evaluate
from utils.metrics import Record, build_summary_table, save_results
from utils.plot import (plot_accuracy_vs_noise, plot_accuracy_vs_ratio,
                        plot_params_compare, plot_summary)
from utils.seed import set_seed

# ================= 配置参数 =================
DEVICE: str = "cuda" if torch.cuda.is_available() else "cpu"
EPOCHS: int = 30           # 统一训练预算，保证各变体收敛
RATIOS: list[float] = [0.1, 0.5, 1.0]          # 数据量阶梯: 10%, 50%, 100%
NOISE_STDS: list[float] = [0.0, 0.33, 0.67]    # 噪声强度: 0 (无噪声) / 中 / 强
BATCH_SIZE: int = 256
SEED: int = 42

RESULTS_DIR: Path = Path("results")
TABLES_DIR: Path = RESULTS_DIR / "tables"
FIGURES_DIR: Path = RESULTS_DIR / "figures"
LOGS_DIR: Path = RESULTS_DIR / "logs"

#: (显示名, 构造函数) —— 只通过“是否加 Dropout / 是否加 ResNet 元素”区分
MODEL_FACTORIES: list[tuple[str, Callable[[], nn.Module]]] = [
    ("LeNet", lambda: LeNet(dropout=0.0)),
    ("LeNet+Dropout", lambda: LeNet(dropout=0.3)),
    ("LeNet+ResNet", lambda: ResLeNet(dropout=0.0)),
    ("ResLeNet", lambda: ResLeNet(dropout=0.3)),
    ("MiniViT", lambda: MiniViT(dropout=0.0)),
    ("MiniViT+Dropout", lambda: MiniViT(dropout=0.1)),
]

LENET_FAMILY: list[str] = ["LeNet", "LeNet+Dropout", "LeNet+ResNet", "ResLeNet"]
VIT_FAMILY: list[str] = ["MiniViT", "MiniViT+Dropout"]


def setup_logger(log_path: Path) -> logging.Logger:
    """同时输出到控制台与 results/logs/experiment.log。"""
    logger: logging.Logger = logging.getLogger("vit_vs_lenet")
    logger.setLevel(logging.INFO)
    logger.propagate = False
    if not logger.handlers:
        fmt: logging.Formatter = logging.Formatter("%(asctime)s | %(message)s", datefmt="%H:%M:%S")
        fh: logging.FileHandler = logging.FileHandler(log_path, mode="w", encoding="utf-8")
        fh.setFormatter(fmt)
        sh: logging.StreamHandler = logging.StreamHandler(sys.stdout)
        sh.setFormatter(fmt)
        logger.addHandler(fh)
        logger.addHandler(sh)
    return logger


def main() -> None:
    for d in (TABLES_DIR, FIGURES_DIR, LOGS_DIR):
        d.mkdir(parents=True, exist_ok=True)

    logger: logging.Logger = setup_logger(LOGS_DIR / "experiment.log")
    logger.info("设备: %s | epochs=%d | batch_size=%d | 数据比例=%s | 噪声=%s",
                DEVICE, EPOCHS, BATCH_SIZE, RATIOS, NOISE_STDS)

    if DEVICE == "cuda":
        # TF32 提升 fp32 矩阵乘法速度（精度差异可忽略）
        torch.backends.cuda.matmul.allow_tf32 = True

    results: list[Record] = []

    # ================= 核心实验循环 =================
    for noise_std in NOISE_STDS:
        logger.info("========== 正在测试 噪声强度: %s ==========", noise_std)
        for ratio in RATIOS:
            logger.info("---- 数据比例: %.0f%% ----", ratio * 100)

            # 固定种子后再造数据集（保证子集划分/噪声可复现）
            set_seed(SEED)
            train_loader: BatchLoader
            test_loader: BatchLoader
            train_loader, test_loader = dataLoader(ratio=ratio, noise_std=noise_std,
                                                   batch_size=BATCH_SIZE, device=DEVICE)

            for model_name, factory in MODEL_FACTORIES:
                set_seed(SEED)
                model: nn.Module = factory()
                stats: TrainStats = train_and_evaluate(
                    model, train_loader, test_loader, epochs=EPOCHS, device=DEVICE,
                )
                logger.info("%-16s -> 准确率: %.2f%%, 参数量: %.2fM, 训练耗时: %.1fs, 吞吐: %.0f img/s",
                            model_name, stats["acc"], stats["params_M"], stats["time"], stats["samples_per_sec"])
                results.append({
                    "Noise_Std": noise_std,
                    "Data_Ratio": ratio,
                    "Model": model_name,
                    "Accuracy": round(float(stats["acc"]), 4),
                    "Macro_F1": round(float(stats["macro_f1"]), 4),
                    "Params(M)": round(float(stats["params_M"]), 6),
                    "Time(s)": round(float(stats["time"]), 3),
                })

    # ================= 保存结果 =================
    df: pd.DataFrame = save_results(results, TABLES_DIR / "experiment_results.csv")
    summary: pd.DataFrame = build_summary_table(df)
    summary.to_csv(TABLES_DIR / "accuracy_summary.csv")
    logger.info("准确率汇总 (行=配置, 列=模型):\n%s", summary.to_string())

    lenet_df: pd.DataFrame = df[df["Model"].isin(LENET_FAMILY)]
    vit_df: pd.DataFrame = df[df["Model"].isin(VIT_FAMILY)]

    fig_paths: list[str] = [
        plot_accuracy_vs_ratio(lenet_df, FIGURES_DIR, "accuracy_vs_ratio_lenet.png",
                               title="LeNet family: Accuracy vs Data Ratio"),
        plot_accuracy_vs_noise(lenet_df, FIGURES_DIR, "accuracy_vs_noise_lenet.png",
                               title="LeNet family: Robustness vs Noise"),
        plot_accuracy_vs_ratio(vit_df, FIGURES_DIR, "accuracy_vs_ratio_vit.png",
                               title="ViT family: Accuracy vs Data Ratio"),
        plot_accuracy_vs_noise(vit_df, FIGURES_DIR, "accuracy_vs_noise_vit.png",
                               title="ViT family: Robustness vs Noise"),
        plot_params_compare(df, FIGURES_DIR),
        plot_summary(df, FIGURES_DIR),
    ]
    for p in fig_paths:
        logger.info("图表已保存: %s", p)

    logger.info("所有实验完成！%d 条记录 -> %s", len(results), TABLES_DIR / "experiment_results.csv")


if __name__ == "__main__":
    main()
