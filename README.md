# ViTVSLeNet

对比 **视觉 Transformer（MiniViT）** 与 **LeNet-5** 在 MNIST 上的表现，考察两个因素：
训练数据量（10% / 50% / 100%）和测试集加性高斯噪声（σ = 0 / 0.33 / 0.67）。

## 目录结构

```
ViTVSLeNet/
├── main.py              # 实验入口：2 模型 × 3 数据比例 × 3 噪声强度
├── loader/dataLoader.py # MNIST 加载（GPU 驻留张量 / CPU DataLoader 回退）
├── models/
│   ├── LeNet_5.py       # LeNet-5（含 Dropout）
│   └── mini_vit.py      # MiniViT（patch 4×4，dim 64，depth 2，heads 4）
├── engine/train.py      # 训练/评估引擎（AMP、吞吐统计）
├── utils/
│   ├── seed.py          # 随机种子
│   ├── metrics.py       # 参数量、准确率、宏 F1、结果 CSV
│   └── plot.py          # 结果绘图
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

配置在 `main.py` 顶部：`EPOCHS`、`RATIOS`、`NOISE_STDS`、`BATCH_SIZE`、`SEED`。

## 结果

| 输出 | 说明 |
|---|---|
| `results/tables/experiment_results.csv` | 18 组记录的准确率 / 宏 F1 / 参数量 / 耗时 |
| `results/tables/accuracy_summary.csv` | 按配置聚合的准确率汇总表 |
| `results/figures/accuracy_vs_ratio.png` | 准确率随数据比例变化 |
| `results/figures/accuracy_vs_noise.png` | 准确率随噪声强度变化 |
| `results/figures/params_compare.png` | 参数量对比 |
| `results/logs/experiment.log` | 完整运行日志 |

当前结果：100% 数据、无噪声时 LeNet 99.04% / MiniViT 97.48%；σ=0.67 时 LeNet 98.16% / MiniViT 87.38%。

## 论文

`paper/main.tex` 是基于南航课程报告模板（`NUAAReport.sty`）的完整报告，已编译好 `paper/main.pdf`（12 页）。

仓库内附带了一份 TeX Live 2026（装在 `tools/texlive`，约 670 MB，不需要管理员权限，已装 ctex/gbt7714/tcolorbox 等宏包）。在 `paper/` 下直接编译：

```powershell
cd paper
.\build.ps1        # xelatex -> bibtex -> xelatex x2，输出 main.pdf
```

手动编译则先把 `tools\texlive\2026\bin\windows` 加入 `PATH`，再执行：

```bash
xelatex main.tex && bibtex main && xelatex main.tex && xelatex main.tex
```

若把 `tools/` 删掉（它已在 `.gitignore` 中），可自行安装 TeX Live / MiKTeX 并装齐 `ctex`、`gbt7714`、`tcolorbox`、`fontawesome`、`pythonhighlight`、`subfigure`、`titlesec` 等宏包，或直接把 `paper/` 上传到 Overleaf 选 XeLaTeX 编译。
