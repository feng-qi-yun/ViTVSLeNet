# ViTVSLeNet

**课本模型 + 课本优化到底值多少？** 本文把 Dropout 与 ResNet 元素（残差连接 + BatchNorm）当作变量，
在固定的训练协议下测量它们在准确率、噪声鲁棒性与训练代价上的边际收益。

变体矩阵（6 个）：

| 变体 | Dropout | 残差连接 | BatchNorm | 参数量 |
|---|---|---|---|---|
| LeNet | -- | -- | -- | 44,426 |
| LeNet+Dropout | 0.3 | -- | -- | 44,426 |
| LeNet+ResNet | -- | 2 个 | 6 处 | 49,792 |
| ResLeNet | 0.3 | 2 个 | 6 处 | 49,792 |
| MiniViT | -- | -- | -- | 72,202 |
| MiniViT+Dropout | 0.1 | -- | -- | 72,202 |

实验条件：MNIST，训练数据比例 10% / 50% / 100%，测试集加性高斯噪声 σ = 0 / 0.33 / 0.67，
Adam（lr 1e-3）、batch 256、30 轮、固定种子 42 → 共 3×3×6 = **54 组**。

## 目录结构

```
ViTVSLeNet/
├── main.py              # 实验入口：6 变体 × 3 数据比例 × 3 噪声强度
├── loader/dataLoader.py # MNIST 加载（GPU 驻留张量 / CPU DataLoader 回退）
├── models/
│   ├── LeNet_5.py       # LeNet-5（Dropout 可开关）
│   ├── res_lenet.py     # ResLeNet：残差块 + BatchNorm + 无偏置卷积（Dropout 可开关）
│   └── mini_vit.py      # MiniViT（patch 4×4，dim 64，depth 2，heads 4，Dropout 可开关）
├── engine/train.py      # 训练/评估引擎（AMP、吞吐统计）
├── utils/
│   ├── seed.py          # 随机种子
│   ├── metrics.py       # 参数量、准确率、宏 F1、结果 CSV
│   └── plot.py          # 结果绘图（按模型族出图 + 汇总图）
├── results/             # 输出：tables / figures / logs
└── paper/               # LaTeX 课程报告（NUAA 模板）
```

## 环境

```bash
pip install -r requirements.txt
```

依赖 `torch==2.8.0+cu128`、`torchvision`、`matplotlib`、`pandas`、`tqdm`；CPU 也能跑（自动回退）。

## 运行

```bash
python main.py
```

配置在 `main.py` 顶部：`EPOCHS`、`RATIOS`、`NOISE_STDS`、`BATCH_SIZE`、`SEED` 与 `MODEL_FACTORIES`。

## 结果

| 输出 | 说明 |
|---|---|
| `results/tables/experiment_results.csv` | 54 组记录的准确率 / 宏 F1 / 参数量 / 耗时 |
| `results/tables/accuracy_summary.csv` | 按配置聚合的准确率汇总表 |
| `results/figures/accuracy_vs_ratio_lenet.png` | LeNet 家族：准确率随数据比例变化 |
| `results/figures/accuracy_vs_noise_lenet.png` | LeNet 家族：准确率随噪声强度变化 |
| `results/figures/accuracy_vs_ratio_vit.png` / `..._noise_vit.png` | ViT 家族的同两幅图 |
| `results/figures/summary.png` | 干净 vs 强噪声准确率、训练耗时汇总 |
| `results/figures/params_compare.png` | 各变体参数量 |
| `results/logs/experiment.log` | 完整运行日志 |

关键结果（100% 数据，30 轮）：

| 变体 | 训练时间(s) | 干净准确率 | σ=0.67 准确率 | 鲁棒性下降 |
|---|---|---|---|---|
| LeNet | 18.26 | 98.78 | 98.17 | 0.61 |
| LeNet+Dropout | 18.54 | 99.10 | 98.24 | 0.86 |
| LeNet+ResNet | 27.89 | **99.29** | 96.14 | 3.15 |
| ResLeNet | 28.97 | 99.06 | 95.73 | 3.33 |
| MiniViT | 40.26 | 97.26 | 80.90 | 16.36 |
| MiniViT+Dropout | 41.60 | 98.27 | 85.86 | 12.41 |

要点：
- **Dropout** 是性价比最高的优化：LeNet 上干净数据平均 +0.31（九配置平均 +0.26）、耗时 +1.5%、零参数；ViT 上收益更大（九配置平均 +2.74，强噪声下最多 +10.07）。
- **ResNet 元素** 能把干净准确率推到最高（99.29），但代价是 +12% 参数、+53% 时间，且 BatchNorm 用的滑动统计量在加噪输入上失效，强噪声下反而退化 2.0–7.5 个百分点（九配置平均 −1.26）。
- 两者叠加**没有**额外收益（九配置平均 −1.97）：Dropout 的正则化被 BN 的分布敏感性抵消。

## 论文

`paper/main.tex` 是基于南航课程报告模板（`NUAAReport.sty`）的完整报告，已编译好 `paper/main.pdf`（13 页）。
报告结构：引言（RQ1 准确率 / RQ2 鲁棒性 / RQ3 代价）→ 相关工作 → 方法（模型、三者优化元件的公式、变体矩阵、训练协议）→
结果（Dropout、ResNet 元素、组合、ViT+Dropout、代价核算）→ 讨论（BN 协变量偏移、Dropout 与 BN 的相互作用、实践建议）→ 结论与局限。

| 脚本 | 平台 | 说明 |
|---|---|---|
| `paper/build.py` | 全平台 | **推荐入口**，自动检测操作系统并调用对应脚本 |
| `paper/build.ps1` | Windows | PowerShell |
| `paper/build.sh` | Linux / macOS | bash，两个平台共用 |

**一键编译**（Windows / Linux / macOS 通用，需先装好下面的前置依赖）：

```bash
cd paper
python build.py
```

**前置依赖**：任选其一安装，确保 `xelatex`、`bibtex` 在 `PATH` 中，并装齐 `ctex`、`gbt7714`、`tcolorbox`、`fontawesome`、`pythonhighlight`、`subfigure`、`titlesec` 等宏包。

- **Windows**：安装 [MiKTeX](https://miktex.org/)（或 TeX Live），首次编译会自动补装缺失宏包。
- **macOS**：安装 [MacTeX](https://tug.org/mactex/)。
- **Linux（Debian/Ubuntu）**：
  ```bash
  sudo apt install texlive-xetex texlive-lang-chinese texlive-latex-extra \
      texlive-bibtex-extra fonts-noto-cjk
  # 若仍缺个别宏包，可改装全家桶（约 5 GB）：
  # sudo apt install texlive-full
  ```

> **字体注意**：`main.tex` 第 13 行用 `\setmainfont{Times New Roman}` 指定英文字体，该字体是微软版权字体，Linux/macOS 默认没有，会导致 `xelatex` 报 “font not found”。两种解法二选一：
> 1. 安装该字体：Ubuntu 执行 `sudo apt install ttf-mscorefonts-installer`（macOS 手动安装 Times New Roman 字体文件）；
> 2. 把 `main.tex` 里的 `Times New Roman` 改成系统自带的开源衬线字体（如 `Liberation Serif`、`TeX Gyre Termes`）。
>
> （第 16 行的 `SimHei` 只是定义了一个未被使用的 `\zhhei` 字体族，不会触发加载，可忽略。）

**手动编译**（等价于上面脚本做的事）：

```bash
xelatex main.tex && bibtex main && xelatex main.tex && xelatex main.tex
```

也可以直接把 `paper/` 上传到 Overleaf，选 XeLaTeX 编译。
