"""数据加载模块。

GPU 利用率的核心思路：MNIST 数据量很小（60k×1×28×28≈47MB uint8），
瓶颈往往不是 GPU 算力，而是 CPU 端“逐 batch 读图/转换/拷贝 + Python 调度”的空转。
因此：

- GPU 环境：把整个训练/测试集一次性归一化后常驻显存，用轻量批迭代器
  （GpuBatchLoader）直接切 batch，CPU 只需发射少量 kernel，GPU 不再等数据；
- CPU 环境：回退为普通 DataLoader + TensorDataset。

测试集高斯噪声作用在“归一化后的特征空间”，noise_std=0.5 才是真正意义上的较强噪声。
"""

import torch
from torchvision import datasets
from torch.utils.data import DataLoader, TensorDataset

MNIST_MEAN, MNIST_STD = 0.1307, 0.3081


class _TensorSubset:
    """轻量容器，只用于让 engine 能通过 len(loader.dataset) 拿到样本数。"""

    def __init__(self, x: torch.Tensor, y: torch.Tensor):
        self.data, self.targets = x, y

    def __len__(self) -> int:
        return self.data.shape[0]


class GpuBatchLoader:
    """显存驻留张量的批迭代器：数据已在 device 上，迭代时只做切片。"""

    def __init__(self, x: torch.Tensor, y: torch.Tensor, batch_size: int, shuffle: bool = False):
        self.x, self.y = x, y
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.dataset = _TensorSubset(x, y)

    def __len__(self) -> int:
        return (len(self.dataset) + self.batch_size - 1) // self.batch_size

    def __iter__(self):
        n = len(self.dataset)
        idx = torch.randperm(n, device=self.x.device) if self.shuffle else torch.arange(n, device=self.x.device)
        for start in range(0, n, self.batch_size):
            sel = idx[start:start + self.batch_size]
            yield self.x[sel], self.y[sel]


def _load_mnist_split(train: bool, device: torch.device):
    """读取 MNIST 并按 ToTensor+Normalize 的等价方式一次性归一化到指定设备。"""
    raw = datasets.MNIST(root="./data", train=train, download=True)
    x = raw.data.to(device=device, dtype=torch.float32).div_(255.0).sub_(MNIST_MEAN).div_(MNIST_STD)
    x = x.unsqueeze_(1).contiguous()  # [N, 28, 28] -> [N, 1, 28, 28]
    y = raw.targets.to(device=device, dtype=torch.long)
    return x, y


def dataLoader(ratio: float = 1.0, noise_std: float = 0.0, batch_size: int = 256,
               device: str | None = None):
    """返回 (train_loader, test_loader)。

    Parameters
    ----------
    ratio : float
        训练集使用比例（0, 1]。
    noise_std : float
        测试集高斯噪声标准差（作用在归一化后的图像上，0 表示不加噪声）。
    batch_size : int
        批大小。
    device : str | None
        数据放置设备；None 且 CUDA 可用时自动驻留显存，否则留在 CPU。
    """
    use_gpu = device is not None and torch.cuda.is_available() and torch.device(device).type == "cuda"
    dev = torch.device(device) if use_gpu else torch.device("cpu")

    x_train, y_train = _load_mnist_split(True, dev)
    x_test, y_test = _load_mnist_split(False, dev)

    if ratio < 1.0:
        num_train = int(len(x_train) * ratio)
        indices = torch.randperm(len(x_train), device=dev)[:num_train]
        x_train, y_train = x_train[indices], y_train[indices]

    # 噪声加在归一化后的特征空间（只在测试集上）
    if noise_std > 0.0:
        x_test = x_test + torch.randn_like(x_test) * noise_std

    if use_gpu:
        return (
            GpuBatchLoader(x_train, y_train, batch_size, shuffle=True),
            GpuBatchLoader(x_test, y_test, batch_size, shuffle=False),
        )

    # CPU 回退：普通 DataLoader（张量已归一化，无需再逐样本做 transform）
    train_loader = DataLoader(TensorDataset(x_train, y_train), batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(TensorDataset(x_test, y_test), batch_size=batch_size, shuffle=False)
    return train_loader, test_loader
