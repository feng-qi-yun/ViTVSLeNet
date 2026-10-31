"""在 LeNet-5 骨架上引入 ResNet 元素。

保留 LeNet-5 的经典骨架（两个 5x5 卷积 + 两次 2x2 池化 + 256-120-84-10 全连接头），
在此基础上加入三类 ResNet 设计：

1. **残差连接**：两处“通道不变”的阶段插入 basic residual block（3x3 卷积成对 + 恒等捷径），
   当输入输出形状不一致时自动改用 1x1 卷积 + BN 的投影捷径；
2. **BatchNorm**：每个卷积后接 BN 再接 ReLU，稳定小批量训练的分布；
3. **无偏置卷积**：残差块内的卷积不使用 bias（由紧随其后的 BN 承担平移），与 ResNet 一致。

于是 LeNet 与 ResLeNet 的差别只来自这三类元素，便于做受控对比。
"""

from __future__ import annotations

import torch
import torch.nn as nn


class ResidualBlock(nn.Module):
    """Basic residual block: 2 x (3x3 conv + BN + ReLU) + 恒等/投影捷径。"""

    def __init__(self, in_channels: int, out_channels: int, stride: int = 1) -> None:
        super().__init__()
        self.conv1: nn.Conv2d = nn.Conv2d(in_channels, out_channels, kernel_size=3,
                                          stride=stride, padding=1, bias=False)
        self.bn1: nn.BatchNorm2d = nn.BatchNorm2d(out_channels)
        self.conv2: nn.Conv2d = nn.Conv2d(out_channels, out_channels, kernel_size=3,
                                          stride=1, padding=1, bias=False)
        self.bn2: nn.BatchNorm2d = nn.BatchNorm2d(out_channels)
        self.relu: nn.ReLU = nn.ReLU(inplace=True)

        if stride != 1 or in_channels != out_channels:
            # 形状不一致时用 1x1 卷积 + BN 做投影捷径
            self.shortcut: nn.Module = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity: torch.Tensor = self.shortcut(x)
        out: torch.Tensor = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return self.relu(out + identity)


class ResLeNet(nn.Module):
    """LeNet-5 + ResNet 元素（残差块、BatchNorm、无偏置卷积）。"""

    def __init__(self, num_classes: int = 10, dropout: float = 0.3) -> None:
        super().__init__()
        self.features: nn.Sequential = nn.Sequential(
            # 与 LeNet 相同的 5x5 卷积与池化尺寸，卷积后加 BN 与 ReLU
            nn.Conv2d(1, 6, kernel_size=5, bias=False), nn.BatchNorm2d(6), nn.ReLU(inplace=True),  # 28 -> 24
            ResidualBlock(6, 6),          # 24x24，恒等捷径
            nn.MaxPool2d(2),              # 24 -> 12
            nn.Conv2d(6, 16, kernel_size=5, bias=False), nn.BatchNorm2d(16), nn.ReLU(inplace=True),  # 12 -> 8
            ResidualBlock(16, 16),        # 8x8，恒等捷径
            nn.MaxPool2d(2),              # 8 -> 4
        )
        self.classifier: nn.Sequential = nn.Sequential(
            nn.Flatten(),
            nn.Linear(16 * 4 * 4, 120), nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(120, 84), nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(84, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.classifier(x)
        return x
