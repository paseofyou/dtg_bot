"""微观分支：**事件级注意力**编码器。

设计依据（实测，见 AGENTS.md「关口实验结论」，TwiBot-20 / 15 种子）：

    keep vs shuffle    ΔAUC = -0.09  p=0.644   → 推文顺序不携带判别信息
    keep vs bag        ΔAUC = +4.01  p<1e-5    → 事件级注意力极显著
    shuffle vs bag     ΔAUC = +4.10  p<1e-5    → 该增益与顺序完全无关

因此本分支的定位是**在一组推文事件中挑出异常的那几条**，而非建模先后顺序。
关键对比对象是 BotRGCN：它把用户所有推文平均池化成单一向量，
局部异常被平均抹平；本分支保留逐条推文特征并用注意力聚合，AUC 高出 4 个点。

设计选择：

1. **attention pooling** 而非 mean pooling —— 这是 +4 AUC 的直接来源。
2. **位置编码可关闭**（``use_position=False`` 为默认）。既然顺序无信息，
   位置编码只会增加参数与过拟合风险；保留开关是为了可复现顺序诊断实验。

``order_mode`` 现在的用途是**诊断工具**（论文的 Order Sensitivity Diagnosis 小节），
不再是模型机制：
    keep     正常时间正序
    shuffle  每个用户独立随机打乱
    reverse  整体反序
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def permute_sequence(
    x: torch.Tensor, mask: torch.Tensor, mode: str
) -> tuple[torch.Tensor, torch.Tensor]:
    """按 mode 重排每个用户的有效位，**同步重排 mask**。

    同步重排 mask 很重要：若只移动数据而不动 mask，当有效位不在序列前部时
    padding 标记就会与数据错位，得到静默错误的结果。
    ``encode.py`` 产出的 mask 是前对齐的，此时 reverse/shuffle 不改变 mask，
    与旧行为完全一致；但对任意 mask 也保持正确。

    Args:
        x: (B, L, C)
        mask: (B, L) bool，True 为有效位

    Returns:
        (重排后的 x, 重排后的 mask)
    """
    if mode == "keep":
        return x, mask
    if mode not in ("shuffle", "reverse"):
        raise ValueError(f"unknown order_mode: {mode}")

    lengths = mask.sum(dim=1)                                    # (B,)
    positions = torch.arange(x.size(1), device=x.device).expand_as(mask)
    if mode == "reverse":
        # 有效段内 i -> len-1-i
        new_idx = torch.where(mask, lengths.unsqueeze(1) - 1 - positions, positions)
    else:
        # 只在有效段内打乱：给有效位随机 key、无效位 +inf，argsort 后无效位自然留在尾部
        keys = torch.rand_like(x[..., 0])
        keys = torch.where(mask, keys, torch.full_like(keys, float("inf")))
        new_idx = keys.argsort(dim=1)
    return (
        x.gather(1, new_idx.unsqueeze(-1).expand_as(x)),
        mask.gather(1, new_idx),
    )


class AttentionPooling(nn.Module):
    """带 mask 的加性注意力池化。"""

    def __init__(self, dim: int):
        super().__init__()
        self.score = nn.Sequential(nn.Linear(dim, dim), nn.Tanh(), nn.Linear(dim, 1))

    def forward(self, h: torch.Tensor, mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        logits = self.score(h).squeeze(-1)                       # (B, L)
        logits = logits.masked_fill(~mask, float("-inf"))
        # 全 padding 的用户（无推文）：退化成均匀权重，避免 softmax 出 NaN
        empty = ~mask.any(dim=1)
        logits = torch.where(empty.unsqueeze(1), torch.zeros_like(logits), logits)
        weights = torch.softmax(logits, dim=1)
        pooled = (h * weights.unsqueeze(-1)).sum(dim=1)
        pooled = torch.where(empty.unsqueeze(1), torch.zeros_like(pooled), pooled)
        return pooled, weights


class EventAttentionEncoder(nn.Module):
    """事件级注意力编码器：Transformer 交互 + attention pooling。

    Args:
        seq_model: ``transformer`` 事件间自注意力交互；``bag`` 退化为统计池化
            （mean/std → MLP），是"注意力聚合是否真有价值"的对照组。
        use_position: 是否加位置编码。默认 False —— 顺序已实测无信息，
            加位置编码只增加过拟合风险。设为 True 仅用于复现顺序诊断实验。
    """

    def __init__(
        self,
        in_channels: int,
        d_model: int = 64,
        out_dim: int = 64,
        n_layers: int = 2,
        n_heads: int = 4,
        max_len: int = 64,
        dropout: float = 0.1,
        seq_model: str = "transformer",
        order_mode: str = "keep",
        use_position: bool = False,
    ):
        super().__init__()
        self.seq_model = seq_model
        self.order_mode = order_mode
        self.use_position = use_position

        if seq_model == "bag":
            self.mlp = nn.Sequential(
                nn.Linear(in_channels * 2, d_model), nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(d_model, out_dim), nn.GELU(),
            )
            return

        self.input_proj = nn.Linear(in_channels, d_model)
        # 可学习位置编码（序数位置，非物理时间；默认关闭，见类文档）
        self.pos_emb = nn.Embedding(max_len, d_model) if use_position else None
        self.input_norm = nn.LayerNorm(d_model)
        layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=d_model * 4,
            dropout=dropout, activation="gelu", batch_first=True, norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(layer, num_layers=n_layers)
        self.pool = AttentionPooling(d_model)
        self.out_proj = nn.Sequential(nn.LayerNorm(d_model), nn.Linear(d_model, out_dim), nn.GELU())
        if self.pos_emb is not None:
            nn.init.normal_(self.pos_emb.weight, std=0.02)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """x: (B, L, C) float, mask: (B, L) bool → (B, out_dim)"""
        x, mask = permute_sequence(x, mask, self.order_mode)

        if self.seq_model == "bag":
            w = mask.unsqueeze(-1).float()
            denom = w.sum(1).clamp(min=1.0)
            mean = (x * w).sum(1) / denom
            var = (((x - mean.unsqueeze(1)) ** 2) * w).sum(1) / denom
            return self.mlp(torch.cat([mean, var.clamp(min=0).sqrt()], dim=-1))

        h = self.input_norm(self.input_proj(x))
        if self.pos_emb is not None:
            pos = torch.arange(x.size(1), device=x.device)
            h = h + self.pos_emb(pos).unsqueeze(0)
        # 全 padding 的用户若整行都被 mask，TransformerEncoder 会输出 NaN，
        # 故仅对 transformer 放开空用户的第 0 位。
        attn_mask = mask.clone()
        attn_mask[~mask.any(dim=1), 0] = True
        h = self.transformer(h, src_key_padding_mask=~attn_mask)
        h = torch.nan_to_num(h)
        # 注意：这里必须传**原始** mask，不能传 attn_mask。
        # 否则 AttentionPooling 会把空用户误判为非空，其嵌入变成 padding 位的
        # 垃圾值（随输入顺序变化、不可复现），而非干净的零向量。
        pooled, _ = self.pool(h, mask)
        return self.out_proj(pooled)
