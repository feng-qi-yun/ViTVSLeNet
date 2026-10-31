"""实验结果可视化。

生成的图片统一保存到调用方指定的目录（main.py 传 results/figures）。

- plot_accuracy_vs_ratio: 数据比例(x) - 准确率(y)，每个噪声强度一张子图
- plot_accuracy_vs_noise: 噪声强度(x) - 准确率(y)，每个数据比例一张子图
- plot_params_compare: 模型参数量对比柱状图
- plot_summary: 准确率与训练代价汇总（σ=0 与 σ=0.67 的对比柱状图 + 训练耗时）
- plot_all: 一次性生成以上全部图表，返回保存路径列表

对比的是“课本模型 + 课本优化”的若干变体：LeNet 家族（是否加 Dropout / ResNet 元素）
与 ViT 家族（是否加 Dropout）。
"""

from __future__ import annotations

import os
from typing import Any, Final

# 把 matplotlib 缓存目录放到项目内，避免无权限写 C:\Users\...\.matplotlib
os.environ.setdefault(
    "MPLCONFIGDIR",
    os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".matplotlib"),
)

import matplotlib

matplotlib.use("Agg")  # 无界面环境下也能出图

import matplotlib.pyplot as plt
import pandas as pd

# 统一图表风格
plt.rcParams.update({
    "figure.dpi": 130,
    "savefig.dpi": 150,
    "axes.grid": True,
    "grid.alpha": 0.35,
    "axes.spines.top": False,
    "axes.spines.right": False,
})

MODEL_STYLES: Final[dict[str, dict[str, Any]]] = {
    "LeNet": {"color": "#4C72B0", "marker": "o", "linestyle": "-"},
    "LeNet+Dropout": {"color": "#7FA8D4", "marker": "o", "linestyle": ":"},
    "LeNet+ResNet": {"color": "#2E7D32", "marker": "^", "linestyle": "-"},
    "ResLeNet": {"color": "#81C784", "marker": "^", "linestyle": ":"},
    "MiniViT": {"color": "#DD8452", "marker": "s", "linestyle": "-"},
    "MiniViT+Dropout": {"color": "#F0B27A", "marker": "s", "linestyle": ":"},
}

#: 图例/子图线条的优先顺序
MODEL_ORDER: Final[tuple[str, ...]] = (
    "LeNet",
    "LeNet+Dropout",
    "LeNet+ResNet",
    "ResLeNet",
    "MiniViT",
    "MiniViT+Dropout",
)


def _model_order(df: pd.DataFrame) -> list[str]:
    """优先按 MODEL_ORDER 排序，未登记的模型追加在末尾。"""
    present: list[str] = list(df["Model"].unique())
    known: list[str] = [m for m in MODEL_ORDER if m in present]
    unknown: list[str] = [m for m in present if m not in MODEL_ORDER]
    return known + unknown


def _ratios_as_percent(df: pd.DataFrame) -> list[float]:
    return sorted({round(float(r) * 100, 1) for r in df["Data_Ratio"].unique()})


def _annotate_points(ax, xs, ys, model_index: int) -> None:
    """按模型序号交错放置数值标注，避免多条线重叠时标签互相压住。"""
    dy: int = 8 if model_index % 2 == 0 else -13
    for x, y in zip(xs, ys):
        ax.annotate(f"{y:.1f}", (x, y), textcoords="offset points",
                    xytext=(0, dy), ha="center", fontsize=7.5)


def plot_accuracy_vs_ratio(df: pd.DataFrame,
                           out_dir: str | os.PathLike[str],
                           filename: str = "accuracy_vs_ratio.png",
                           title: str = "Accuracy vs Data Ratio") -> str:
    """数据比例 -> 准确率；不同噪声强度分为子图，横轴/纵轴均为数值。"""
    os.makedirs(out_dir, exist_ok=True)
    noises: list[float] = sorted(df["Noise_Std"].unique())
    models: list[str] = _model_order(df)

    fig, axes = plt.subplots(1, len(noises), figsize=(4.6 * len(noises), 3.8), sharey=True)
    if len(noises) == 1:
        axes = [axes]

    for ax, noise in zip(axes, noises):
        sub: pd.DataFrame = df[df["Noise_Std"] == noise]
        for idx, model in enumerate(models):
            g: pd.DataFrame = sub[sub["Model"] == model].sort_values("Data_Ratio")
            if g.empty:
                continue
            xs = g["Data_Ratio"] * 100
            ax.plot(
                xs, g["Accuracy"],
                label=model, linewidth=1.8, markersize=6, **MODEL_STYLES.get(model, {}),
            )
            _annotate_points(ax, xs, g["Accuracy"], idx)
        ax.set_title(f"Noise std = {noise}", fontsize=10)
        ax.set_xlabel("Training data ratio (%)")
        ax.set_ylabel("Test accuracy (%)")
        ax.set_xticks(_ratios_as_percent(sub))

    fig.suptitle(title, fontsize=11)
    axes[0].legend(loc="lower right", fontsize=8)
    fig.tight_layout()
    path: str = os.fspath(os.path.join(out_dir, filename))
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_accuracy_vs_noise(df: pd.DataFrame,
                           out_dir: str | os.PathLike[str],
                           filename: str = "accuracy_vs_noise.png",
                           title: str = "Robustness: Accuracy vs Noise Level") -> str:
    """噪声强度 -> 准确率；不同数据比例分为子图。"""
    os.makedirs(out_dir, exist_ok=True)
    ratios: list[float] = sorted(df["Data_Ratio"].unique())
    models: list[str] = _model_order(df)

    fig, axes = plt.subplots(1, len(ratios), figsize=(4.4 * len(ratios), 3.8), sharey=True)
    if len(ratios) == 1:
        axes = [axes]

    for ax, ratio in zip(axes, ratios):
        sub: pd.DataFrame = df[df["Data_Ratio"] == ratio]
        for idx, model in enumerate(models):
            g: pd.DataFrame = sub[sub["Model"] == model].sort_values("Noise_Std")
            if g.empty:
                continue
            ax.plot(
                g["Noise_Std"], g["Accuracy"],
                label=model, linewidth=1.8, markersize=6, **MODEL_STYLES.get(model, {}),
            )
            _annotate_points(ax, g["Noise_Std"], g["Accuracy"], idx)
        ax.set_title(f"Data ratio = {ratio:.0%}", fontsize=10)
        ax.set_xlabel("Gaussian noise std (test set)")
        ax.set_ylabel("Test accuracy (%)")
        ax.set_xticks(sorted(sub["Noise_Std"].unique()))

    fig.suptitle(title, fontsize=11)
    axes[0].legend(loc="lower left", fontsize=8)
    fig.tight_layout()
    path: str = os.fspath(os.path.join(out_dir, filename))
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_params_compare(df: pd.DataFrame,
                        out_dir: str | os.PathLike[str],
                        filename: str = "params_compare.png") -> str:
    """参数量对比（所有变体）。"""
    os.makedirs(out_dir, exist_ok=True)
    params: pd.Series = df.groupby("Model")["Params(M)"].first()
    models: list[str] = _model_order(df)

    fig, ax = plt.subplots(figsize=(6.6, 3.8))
    colors: list[str] = [str(MODEL_STYLES.get(m, {}).get("color", "#888888")) for m in models]
    bars = ax.bar(models, [float(params.get(m, 0.0)) for m in models], color=colors, width=0.6)
    for bar, model in zip(bars, models):
        ax.annotate(f"{float(params.get(model, 0.0)):.4f} M",
                    (bar.get_x() + bar.get_width() / 2, bar.get_height()),
                    textcoords="offset points", xytext=(0, 4), ha="center", fontsize=8)
    ax.set_ylabel("Trainable parameters (M)")
    ax.set_title("Parameter Count per Variant", fontsize=11)
    plt.setp(ax.get_xticklabels(), rotation=22, ha="right", fontsize=8)
    fig.tight_layout()
    path: str = os.fspath(os.path.join(out_dir, filename))
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_summary(df: pd.DataFrame,
                 out_dir: str | os.PathLike[str],
                 filename: str = "summary.png") -> str:
    """汇总图：左图 100% 数据下 σ=0 与 σ=0.67 的准确率；右图 100% 数据的训练耗时。"""
    os.makedirs(out_dir, exist_ok=True)
    models: list[str] = _model_order(df)
    full: pd.DataFrame = df[df["Data_Ratio"] == 1.0]
    clean: pd.Series = full[full["Noise_Std"] == 0.0].set_index("Model")["Accuracy"]
    noisy: pd.Series = full[full["Noise_Std"] == 0.67].set_index("Model")["Accuracy"]
    times: pd.Series = full.groupby("Model")["Time(s)"].mean()

    x = list(range(len(models)))
    width = 0.38

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.0, 3.8))
    ax1.bar([i - width / 2 for i in x], [float(clean.get(m, 0)) for m in models],
            width=width, label="clean (σ=0)", color="#4C72B0")
    ax1.bar([i + width / 2 for i in x], [float(noisy.get(m, 0)) for m in models],
            width=width, label="noise (σ=0.67)", color="#C44E52")
    for i, m in enumerate(models):
        ax1.annotate(f"{float(clean.get(m, 0)):.1f}", (i - width / 2, float(clean.get(m, 0))),
                     textcoords="offset points", xytext=(0, 3), ha="center", fontsize=7)
        ax1.annotate(f"{float(noisy.get(m, 0)):.1f}", (i + width / 2, float(noisy.get(m, 0))),
                     textcoords="offset points", xytext=(0, 3), ha="center", fontsize=7)
    ax1.set_xticks(x)
    ax1.set_xticklabels(models, rotation=25, ha="right", fontsize=8)
    ax1.set_ylabel("Test accuracy (%) @ 100% data")
    ax1.set_title("Clean vs Noisy Accuracy", fontsize=10)
    ax1.legend(fontsize=8, loc="lower left")

    ax2.bar(x, [float(times.get(m, 0)) for m in models],
            color=[str(MODEL_STYLES.get(m, {}).get("color", "#888888")) for m in models], width=0.6)
    for i, m in enumerate(models):
        ax2.annotate(f"{float(times.get(m, 0)):.1f}s", (i, float(times.get(m, 0))),
                     textcoords="offset points", xytext=(0, 3), ha="center", fontsize=7)
    ax2.set_xticks(x)
    ax2.set_xticklabels(models, rotation=25, ha="right", fontsize=8)
    ax2.set_ylabel("Training time (s) @ 100% data")
    ax2.set_title("Training Cost (30 epochs)", fontsize=10)

    fig.suptitle("Textbook Optimizations: Accuracy, Robustness and Cost", fontsize=11)
    fig.tight_layout()
    path: str = os.fspath(os.path.join(out_dir, filename))
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_all(df: pd.DataFrame, out_dir: str | os.PathLike[str]) -> list[str]:
    """生成全部图表，返回保存路径列表。"""
    return [
        plot_accuracy_vs_ratio(df, out_dir),
        plot_accuracy_vs_noise(df, out_dir),
        plot_params_compare(df, out_dir),
        plot_summary(df, out_dir),
    ]
