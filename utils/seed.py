"""随机种子管理：保证 ViT vs LeNet 实验可复现。

用法:
    from utils.seed import set_seed
    set_seed(42)                      # 固定 python / numpy / torch / cuda 随机源
    set_seed(42, deterministic=True)  # 同时对 cuDNN 开启确定性模式（更慢但完全可复现）
"""

from __future__ import annotations

import os
import random

import numpy as np
import torch


def set_seed(seed: int = 42, deterministic: bool = True) -> int:
    """设置全局随机种子并返回种子值。

    Parameters
    ----------
    seed : int
        要固定的随机种子。
    deterministic : bool
        True 时开启 cuDNN 确定性模式（同一输入每次输出一致，代价是略慢）；
        False 时关闭确定性模式并打开 cuDNN benchmark，吞吐可能略高。
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # DataLoader 多进程 worker 也遵循该哈希种子
    os.environ["PYTHONHASHSEED"] = str(seed)

    torch.backends.cudnn.deterministic = deterministic
    # 确定性模式下 benchmark 必须关闭，否则 cuDNN 可能选择非确定性算法
    torch.backends.cudnn.benchmark = not deterministic
    return seed
