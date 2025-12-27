# Transformer.py
# ------------------------------------------------------------
# Tab-Patch Transformer for tabular radiomics
# - 将连续特征按 patch_size 分块，线性投影到 d_model
# - 加入可学习 CLS token 与位置编码
# - 使用 nn.TransformerEncoder 堆叠 depth 层
# - 分类头：LayerNorm -> MLP -> (logits)
# ------------------------------------------------------------

from __future__ import annotations
import math
import torch
import torch.nn as nn

__all__ = ["PatchEmbed", "TabPatchTransformer", "build_model"]


class PatchEmbed(nn.Module):
    """
    把连续特征向量 [B, F] 切成大小为 patch_size 的“补丁”，
    再线性投影到 d_model 维度，得到 [B, S, d_model] 的 token 序列。
    S = ceil(F / patch_size)；尾部自动 0 填充以整除。

    Args:
        patch_size (int): 每个 patch 的特征数
        in_feats   (int): 输入特征维度 F
        d_model    (int): 投影后的通道维度
    """
    def __init__(self, patch_size: int, in_feats: int, d_model: int) -> None:
        super().__init__()
        if patch_size <= 0:
            raise ValueError("patch_size must be positive")
        self.patch_size = patch_size
        self.in_feats = in_feats
        self.d_model = d_model

        n_patches = math.ceil(in_feats / patch_size)
        self.n_patches = n_patches
        self.pad_feats = n_patches * patch_size - in_feats  # 需要补零的长度

        # 对每个 patch 做线性投影
        self.proj = nn.Linear(patch_size, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, F] 连续特征
        Returns:
            tokens: [B, S, d_model]
        """
        B, F = x.shape
        if self.pad_feats > 0:
            pad = x.new_zeros((B, self.pad_feats))
            x = torch.cat([x, pad], dim=1)  # [B, F + pad]

        x = x.view(B, self.n_patches, self.patch_size)  # [B, S, patch_size]
        tokens = self.proj(x)                            # [B, S, d_model]
        return tokens


class TabPatchTransformer(nn.Module):
    """
    Tabular Patch Transformer (用于二分类/多分类的头部)

    结构:
      - PatchEmbed: [B,F] -> [B,S,d_model]
      - 加 CLS token 与 learnable 位置编码
      - TransformerEncoder depth 层
      - 取 CLS 向量，经 LN + MLP 输出 logits

    Args:
        in_feats (int): 输入特征维度 F
        num_classes (int): 类别数；二分类用 1（配合 BCEWithLogitsLoss）
        d_model (int): Transformer 通道数
        nhead (int): 多头注意力头数（需整除 d_model）
        depth (int): 编码器层数
        dim_feedforward (int): FFN 隐层维度
        dropout (float): dropout 比例
        patch_size (int): patch 大小
        use_pos_embed (bool): 是否使用可学习位置编码
    """
    def __init__(
        self,
        in_feats: int,
        num_classes: int = 1,
        d_model: int = 128,
        nhead: int = 8,
        depth: int = 3,
        dim_feedforward: int = 256,
        dropout: float = 0.2,
        patch_size: int = 8,
        use_pos_embed: bool = True,
    ) -> None:
        super().__init__()
        if d_model % nhead != 0:
            raise ValueError(f"d_model({d_model}) must be divisible by nhead({nhead})")

        self.embed = PatchEmbed(patch_size, in_feats, d_model)

        # 可学习的 CLS token
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))

        # 可学习的位置编码（包括 CLS 位置）
        self.use_pos_embed = use_pos_embed
        if use_pos_embed:
            self.pos_embed = nn.Parameter(torch.zeros(1, 1 + self.embed.n_patches, d_model))
            nn.init.trunc_normal_(self.pos_embed, std=0.02)

        # Transformer Encoder
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation="gelu",
            batch_first=True,  # 输入输出用 [B, L, C]
            norm_first=False,
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=depth)

        # 归一化 + 线性头
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, num_classes),
        )

        # 参数初始化
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        self.apply(self._init_linear)

    @staticmethod
    def _init_linear(m: nn.Module) -> None:
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, F] float32
        Returns:
            logits: [B] (二分类 num_classes=1) 或 [B, C]
        """
        B = x.size(0)

        # patch embedding
        tokens = self.embed(x)  # [B, S, D]

        # prepend CLS
        cls = self.cls_token.expand(B, -1, -1)  # [B, 1, D]
        tokens = torch.cat([cls, tokens], dim=1)  # [B, 1+S, D]

        # position encoding
        if self.use_pos_embed:
            tokens = tokens + self.pos_embed  # broadcast

        # encoder
        feats = self.encoder(tokens)          # [B, 1+S, D]
        cls_feat = self.norm(feats[:, 0])     # [B, D]

        logits = self.head(cls_feat)          # [B, 1] or [B, C]
        if logits.shape[-1] == 1:
            logits = logits.squeeze(-1)       # [B]
        return logits


def build_model(
    in_feats: int,
    num_classes: int = 1,
    d_model: int = 128,
    nhead: int = 8,
    depth: int = 3,
    dim_feedforward: int = 256,
    dropout: float = 0.2,
    patch_size: int = 8,
    use_pos_embed: bool = True,
) -> TabPatchTransformer:
    """
    便捷构建函数，保持与训练脚本参数一致。
    """
    return TabPatchTransformer(
        in_feats=in_feats,
        num_classes=num_classes,
        d_model=d_model,
        nhead=nhead,
        depth=depth,
        dim_feedforward=dim_feedforward,
        dropout=dropout,
        patch_size=patch_size,
        use_pos_embed=use_pos_embed,
    )
