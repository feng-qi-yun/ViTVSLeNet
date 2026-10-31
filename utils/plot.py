"""实验结果可视化。

生成的图片统一保存到调用方指定的目录（main.py 传 results/figures）。

- plot_accuracy_vs_ratio: 数据比例(x) - 准确率(y)，每个噪声强度一张子图
- plot_accuracy_vs_noise: 噪声强度(x) - 准确率(y)，每个数据比例一张子图
- plot_params_compare: 模型参数量对比柱状图
- plot_all: 一次性生成以上全部图表，返回保存路径列表
"""

from __future__ import annotations

import os

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

MODEL_STYLES = {
    "LeNet": dict(color="#4C72B0", marker="o", linestyle="-"),
    "MiniViT": dict(color="#DD8452", marker="s", linestyle="--"),
}


def _model_order(df: pd.DataFrame) -> list[str]:
    return [m for m in ["LeNet", "MiniViT"] if m in df["Model"].unique()]


def _ratios_as_percent(df: pd.DataFrame) -> list[float]:
    return sorted({round(float(r) * 100, 1) for r in df["Data_Ratio"].unique()})


def plot_accuracy_vs_ratio(df: pd.DataFrame, out_dir: str, filename: str = "accuracy_vs_ratio.png") -> str:
    """数据比例 -> 准确率；不同噪声强度分为子图，横轴/纵轴均为数值。"""
    os.makedirs(out_dir, exist_ok=True)
    noises = sorted(df["Noise_Std"].unique())
    models = _model_order(df)

    fig, axes = plt.subplots(1, len(noises), figsize=(4.6 * len(noises), 3.8), sharey=True)
    if len(noises) == 1:
        axes = [axes]

    for ax, noise in zip(axes, noises):
        sub = df[df["Noise_Std"] == noise]
        for model in models:
            g = sub[sub["Model"] == model].sort_values("Data_Ratio")
            if g.empty:
                continue
            ax.plot(
                g["Data_Ratio"] * 100, g["Accuracy"],
                label=model, linewidth=1.8, markersize=6, **MODEL_STYLES.get(model, {}),
            )
            for x, y in zip(g["Data_Ratio"] * 100, g["Accuracy"]):
                ax.annotate(f"{y:.1f}", (x, y), textcoords="offset points",
                            xytext=(0, 7), ha="center", fontsize=7.5)
        ax.set_title(f"Noise std = {noise}", fontsize=10)
        ax.set_xlabel("Training data ratio (%)")
        ax.set_ylabel("Test accuracy (%)")
        ax.set_xticks(_ratios_as_percent(sub))

    fig.suptitle("Accuracy vs Data Ratio (ViT vs LeNet)", fontsize=11)
    axes[0].legend(loc="lower right", fontsize=9)
    fig.tight_layout()
    path = os.path.join(out_dir, filename)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_accuracy_vs_noise(df: pd.DataFrame, out_dir: str, filename: str = "accuracy_vs_noise.png") -> str:
    """噪声强度 -> 准确率；不同数据比例分为子图。"""
    os.makedirs(out_dir, exist_ok=True)
    ratios = sorted(df["Data_Ratio"].unique())
    models = _model_order(df)

    fig, axes = plt.subplots(1, len(ratios), figsize=(4.4 * len(ratios), 3.8), sharey=True)
    if len(ratios) == 1:
        axes = [axes]

    for ax, ratio in zip(axes, ratios):
        sub = df[df["Data_Ratio"] == ratio]
        for model in models:
            g = sub[sub["Model"] == model].sort_values("Noise_Std")
            if g.empty:
                continue
            ax.plot(
                g["Noise_Std"], g["Accuracy"],
                label=model, linewidth=1.8, markersize=6, **MODEL_STYLES.get(model, {}),
            )
            for x, y in zip(g["Noise_Std"], g["Accuracy"]):
                ax.annotate(f"{y:.1f}", (x, y), textcoords="offset points",
                            xytext=(0, 7), ha="center", fontsize=7.5)
        ax.set_title(f"Data ratio = {ratio:.0%}", fontsize=10)
        ax.set_xlabel("Gaussian noise std (test set)")
        ax.set_ylabel("Test accuracy (%)")
        ax.set_xticks(sorted(sub["Noise_Std"].unique()))

    fig.suptitle("Robustness: Accuracy vs Noise Level", fontsize=11)
    axes[0].legend(loc="lower left", fontsize=9)
    fig.tight_layout()
    path = os.path.join(out_dir, filename)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_params_compare(df: pd.DataFrame, out_dir: str, filename: str = "params_compare.png") -> str:
    """参数量对比（两种模型）。"""
    os.makedirs(out_dir, exist_ok=True)
    params = df.groupby("Model")["Params(M)"].first()
    models = _model_order(df)

    fig, ax = plt.subplots(figsize=(4.6, 3.6))
    colors = [MODEL_STYLES.get(m, {}).get("color", "#888888") for m in models]
    bars = ax.bar(models, [params.get(m, 0.0) for m in models], color=colors, width=0.5)
    for bar, model in zip(bars, models):
        ax.annotate(f"{params.get(model, 0.0):.3f} M",
                    (bar.get_x() + bar.get_width() / 2, bar.get_height()),
                    textcoords="offset points", xytext=(0, 4), ha="center", fontsize=9)
    ax.set_ylabel("Trainable parameters (M)")
    ax.set_title("Parameter Count Comparison", fontsize=11)
    fig.tight_layout()
    path = os.path.join(out_dir, filename)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_all(df: pd.DataFrame, out_dir: str) -> list[str]:
    """生成全部图表，返回保存路径列表。"""
    return [
        plot_accuracy_vs_ratio(df, out_dir),
        plot_accuracy_vs_noise(df, out_dir),
        plot_params_compare(df, out_dir),
    ]
