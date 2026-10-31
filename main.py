"""ViT vs LeNet：不同数据量与噪声强度下的对比实验入口。

运行方式:
    python main.py

产物 (全部写到 results/ 下):
    results/tables/experiment_results.csv   每条实验记录（含参数量/耗时）
    results/tables/accuracy_summary.csv     按配置聚合的准确率汇总表
    results/figures/*.png                   数据比例/噪声强度的准确率曲线、参数量对比
    results/logs/experiment.log             运行日志
"""

import logging
import sys
from pathlib import Path

import torch

from loader.dataLoader import dataLoader
from models.LeNet_5 import LeNet
from models.mini_vit import MiniViT
from engine.train import train_and_evaluate
from utils.seed import set_seed
from utils.metrics import save_results, build_summary_table
from utils.plot import plot_all

# ================= 配置参数 =================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
EPOCHS = 10                # 3 轮对 ViT 不公平（数据饥渴、需要更多步数收敛），提到 10 轮
RATIOS = [0.1, 0.5, 1.0]   # 数据量阶梯: 10%, 50%, 100%
NOISE_STDS = [0.0, 0.5]    # 噪声强度: 0 (无噪声), 0.5 (较强噪声)
BATCH_SIZE = 256           # 越大 GPU 越忙；越小优化步数越多
SEED = 42

RESULTS_DIR = Path("results")
TABLES_DIR = RESULTS_DIR / "tables"
FIGURES_DIR = RESULTS_DIR / "figures"
LOGS_DIR = RESULTS_DIR / "logs"


def setup_logger(log_path: Path) -> logging.Logger:
    """同时输出到控制台与 results/logs/experiment.log。"""
    logger = logging.getLogger("vit_vs_lenet")
    logger.setLevel(logging.INFO)
    logger.propagate = False
    if not logger.handlers:
        fmt = logging.Formatter("%(asctime)s | %(message)s", datefmt="%H:%M:%S")
        fh = logging.FileHandler(log_path, mode="w", encoding="utf-8")
        fh.setFormatter(fmt)
        sh = logging.StreamHandler(sys.stdout)
        sh.setFormatter(fmt)
        logger.addHandler(fh)
        logger.addHandler(sh)
    return logger


def main() -> None:
    for d in (TABLES_DIR, FIGURES_DIR, LOGS_DIR):
        d.mkdir(parents=True, exist_ok=True)

    logger = setup_logger(LOGS_DIR / "experiment.log")
    logger.info("设备: %s | epochs=%d | batch_size=%d | 数据比例=%s | 噪声=%s",
                DEVICE, EPOCHS, BATCH_SIZE, RATIOS, NOISE_STDS)

    if DEVICE == "cuda":
        # TF32 提升 fp32 矩阵乘法速度（精度差异可忽略）
        torch.backends.cuda.matmul.allow_tf32 = True

    results = []
    models = [("LeNet", LeNet), ("MiniViT", MiniViT)]

    # ================= 核心实验循环 =================
    for noise_std in NOISE_STDS:
        logger.info("========== 正在测试 噪声强度: %s ==========", noise_std)
        for ratio in RATIOS:
            logger.info("---- 数据比例: %.0f%% ----", ratio * 100)

            # 固定种子后再造数据集（保证子集划分/噪声可复现）
            set_seed(SEED)
            train_loader, test_loader = dataLoader(ratio=ratio, noise_std=noise_std,
                                                   batch_size=BATCH_SIZE, device=DEVICE)

            for model_name, model_cls in models:
                set_seed(SEED)
                model = model_cls()
                stats = train_and_evaluate(
                    model, train_loader, test_loader, epochs=EPOCHS, device=DEVICE,
                )
                logger.info("%-7s -> 准确率: %.2f%%, 参数量: %.2fM, 训练耗时: %.1fs, 吞吐: %.0f img/s",
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
    df = save_results(results, TABLES_DIR / "experiment_results.csv")
    summary = build_summary_table(df)
    summary.to_csv(TABLES_DIR / "accuracy_summary.csv")
    logger.info("准确率汇总 (行=配置, 列=模型):\n%s", summary.to_string())

    fig_paths = plot_all(df, str(FIGURES_DIR))
    for p in fig_paths:
        logger.info("图表已保存: %s", p)

    logger.info("✅ 所有实验完成！%d 条记录 -> %s", len(results), TABLES_DIR / "experiment_results.csv")


if __name__ == "__main__":
    main()
