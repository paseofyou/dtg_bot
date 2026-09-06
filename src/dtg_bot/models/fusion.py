"""三视图融合与跨视图对比学习。

三个视图：
    micro  事件级注意力（异常推文挑选，顺序无关）
    macro  真实动态图快照演化
    global 当前时刻全景多关系拓扑（RGCN）

``fusion`` 提供完整的消融阶梯，每一档都是论文表格里的一行：
    concat  简单拼接（浅层融合基线）
    gate    逐维门控
    attn    视图间注意力（以全局视图为 query，对三视图做注意力加权）
    none_*  单视图消融
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class ViewAttentionFusion(nn.Module):
    """把 V 个等维视图表示按注意力权重聚合。

    以三视图的拼接作为 query 生成打分，避免把某一视图钦定为"主视图"
    （BotMoE 那类做法会让被钦定的视图主导，其余退化成噪声）。
    """

    def __init__(self, dim: int, n_views: int, dropout: float = 0.1):
        super().__init__()
        self.n_views = n_views
        self.score = nn.Sequential(
            nn.Linear(dim * n_views, dim), nn.GELU(),
            nn.Dropout(dropout), nn.Linear(dim, n_views),
        )
        self.out = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, dim), nn.GELU())

    def forward(self, views: list[torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        stacked = torch.stack(views, dim=1)                    # (N, V, D)
        weights = torch.softmax(self.score(torch.cat(views, dim=-1)), dim=-1)
        fused = (stacked * weights.unsqueeze(-1)).sum(dim=1)
        return self.out(fused), weights


class GatedFusion(nn.Module):
    """逐维门控融合，作为注意力融合的对照组。"""

    def __init__(self, dim: int, n_views: int):
        super().__init__()
        self.gate = nn.Sequential(nn.Linear(dim * n_views, dim * n_views), nn.Sigmoid())
        self.out = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, dim), nn.GELU())
        self.n_views = n_views

    def forward(self, views: list[torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        cat = torch.cat(views, dim=-1)
        gated = (cat * self.gate(cat)).view(cat.size(0), self.n_views, -1)
        return self.out(gated.sum(dim=1)), gated.new_zeros(cat.size(0), self.n_views)


def info_nce(z1: torch.Tensor, z2: torch.Tensor, temperature: float = 0.5,
             max_samples: int = 4096) -> torch.Tensor:
    """跨视图对齐损失。同一用户的两个视图为正样本，其余为负样本。

    子采样以避免超大相似度矩阵（TwiBot-22 有百万节点）。
    """
    if z1.size(0) > max_samples:
        idx = torch.randperm(z1.size(0), device=z1.device)[:max_samples]
        z1, z2 = z1[idx], z2[idx]
    z1 = F.normalize(z1, dim=1)
    z2 = F.normalize(z2, dim=1)
    logits = z1 @ z2.t() / temperature
    labels = torch.arange(z1.size(0), device=z1.device)
    # 对称化：两个方向都算，避免单向退化
    return 0.5 * (F.cross_entropy(logits, labels) + F.cross_entropy(logits.t(), labels))


def multi_view_contrastive(views: list[torch.Tensor], temperature: float = 0.5,
                           max_samples: int = 4096) -> torch.Tensor:
    """所有视图两两之间的对比损失均值。"""
    losses = [
        info_nce(views[i], views[j], temperature, max_samples)
        for i in range(len(views)) for j in range(i + 1, len(views))
    ]
    return torch.stack(losses).mean() if losses else views[0].new_zeros(())
