from __future__ import annotations

import torch
import torch.nn as nn


class MiniViT(nn.Module):
    """极简视觉 Transformer：4x4 图块 -> 49 个 token，2 层编码器。"""

    def __init__(
        self,
        image_size: int = 28,
        patch_size: int = 4,
        num_classes: int = 10,
        dim: int = 64,
        depth: int = 2,
        heads: int = 4,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        # 28 / 4 = 7，所以切出来是 7x7 = 49 个 patch
        num_patches: int = (image_size // patch_size) ** 2
        patch_dim: int = 1 * patch_size * patch_size  # 1通道 * 4x4 = 16

        # 1. 切Patch并线性映射
        #    注意: Unfold 输出形状是 [B, patch_dim, num_patches]，
        #    必须先转置成 [B, num_patches, patch_dim] 再进 Linear，
        #    否则 Linear(patch_dim, dim) 会作用在 num_patches 那一维上导致形状不匹配。
        self.unfold: nn.Unfold = nn.Unfold(kernel_size=patch_size, stride=patch_size)  # [B,16,49]
        self.patch_proj: nn.Linear = nn.Linear(patch_dim, dim)  # [B,49,16] -> [B,49,64]
        self.patch_norm: nn.LayerNorm = nn.LayerNorm(dim)
        self.num_patches: int = num_patches

        # 2. 位置编码和Class Token
        self.cls_token: nn.Parameter = nn.Parameter(torch.randn(1, 1, dim))
        self.pos_embed: nn.Parameter = nn.Parameter(torch.randn(1, num_patches + 1, dim))
        self.pos_drop: nn.Dropout = nn.Dropout(dropout)

        # 3. Transformer Encoder (直接用PyTorch现成的)
        #    TransformerEncoderLayer 的 dropout 同时作用于注意力与 FFN；
        #    norm_first=False 是 PyTorch 默认的 post-LN，小模型收敛更快。
        encoder_layer: nn.TransformerEncoderLayer = nn.TransformerEncoderLayer(
            d_model=dim, nhead=heads, dim_feedforward=dim * 2,
            dropout=dropout, batch_first=True,
        )
        self.transformer: nn.TransformerEncoder = nn.TransformerEncoder(encoder_layer, num_layers=depth)

        # 4. 分类头
        self.mlp_head: nn.Sequential = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Dropout(dropout),
            nn.Linear(dim, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x形状: [B, 1, 28, 28]
        x = self.unfold(x)              # 形状变为: [B, patch_dim, num_patches] = [B, 16, 49]
        x = x.transpose(1, 2)           # 转置为: [B, num_patches, patch_dim] = [B, 49, 16]
        x = self.patch_proj(x)          # 映射为: [B, num_patches, dim] = [B, 49, 64]
        x = self.patch_norm(x)

        # 加上Cls Token
        cls_tokens: torch.Tensor = self.cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)  # [B, num_patches+1, dim]

        # 加上位置编码并做 embedding dropout
        x = self.pos_drop(x + self.pos_embed)

        # 送入Transformer
        x = self.transformer(x)

        # 取出Cls Token的输出进行分类
        return self.mlp_head(x[:, 0])
