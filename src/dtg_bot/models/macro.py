"""宏观分支：BotDGT 原装动态图快照演化编码器。

复现 BotDGT 的结构层、位置编码和时序层，输出最后时间步的宏观视图。
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint

from .botdgt_layers import (
    GraphStructuralLayer,
    PositionEncodingBidirectionalLinks,
    PositionEncodingClusteringCoefficient,
)
from .botdgt_temporal import GraphTemporalLayer


class MacroSnapshotEncoder(nn.Module):
    """对 K 个动态图快照跑 BotDGT 结构-时序编码。

    Args:
        emb: 隐藏维 / 节点表示维。
        n_heads: TransformerConv / 时序注意力的头数。
        num_snapshots: 时序长度 K（对应 BotDGT 的 window_size）。
        dropout: 结构层 + 时序层的 dropout。
        temporal: ``attention`` / ``gru`` / ``lstm``。
        checkpoint: 每个快照的结构层是否启用梯度检查点。
    """

    def __init__(self, emb: int = 64, n_heads: int = 4, num_snapshots: int = 8,
                 dropout: float = 0.3, temporal: str = "attention",
                 checkpoint: bool = True):
        super().__init__()
        self.emb = emb
        self.num_snapshots = num_snapshots
        self.temporal = temporal
        self.use_checkpoint = checkpoint

        # BotDGT 原装三层：结构层 + 两类位置编码 + 时序层
        self.structural = GraphStructuralLayer(emb, n_heads, dropout)
        self.cc_pos = PositionEncodingClusteringCoefficient(emb)
        self.blr_pos = PositionEncodingBidirectionalLinks(emb)
        self.temporal_layer = GraphTemporalLayer(
            emb, n_heads, dropout, num_snapshots, temporal,
        )

        self.norm = nn.LayerNorm(emb)

    def _structural_one(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """单个快照的结构层，按需包 checkpoint。"""
        if self.use_checkpoint and self.training and torch.is_grad_enabled():
            return checkpoint(self.structural, x, edge_index, use_reentrant=False)
        return self.structural(x, edge_index)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor,
                snapshot_masks: list[torch.Tensor],
                snapshot_clustering_coefficient: torch.Tensor,
                snapshot_bidirectional_links_ratio: torch.Tensor,
                snapshot_exist_nodes: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (N, emb)
            edge_index: (2, E_full)
            snapshot_masks: K 个 (E,) bool，保留哪些边
            snapshot_clustering_coefficient: (K, N, 1)
            snapshot_bidirectional_links_ratio: (K, N, 1)
            snapshot_exist_nodes: (K, N) 0/1

        Returns:
            (N, emb)
        """
        # 逐快照结构前向，得到 h_k；与 BotDyGNN 一致，不裁剪到 batch，使用全图
        h_list = [
            self._structural_one(x, edge_index[:, m])
            for m in snapshot_masks
        ]
        h = torch.stack(h_list, dim=1)                       # (N, K, emb)

        # 位置编码：从 (K, N, 1) -> (N, K, emb)
        cc = self.cc_pos(snapshot_clustering_coefficient).permute(1, 0, 2)
        blr = self.blr_pos(snapshot_bidirectional_links_ratio).permute(1, 0, 2)

        # snapshot_exist_nodes 形状 (K, N) -> (N, K) 传给时序层
        temporal_out = self.temporal_layer(h, cc, blr, snapshot_exist_nodes.t())
        return self.norm(temporal_out[:, -1])                 # 取末态 (N, emb)
