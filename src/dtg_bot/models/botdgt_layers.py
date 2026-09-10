"""BotDGT 原装位置编码与结构编码层。

直接复现 BotDGT-master/models/GraphStructuralLayer.py 与 PositionEmbeddingLayer.py。
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch_geometric.nn import TransformerConv


class PositionEncodingClusteringCoefficient(nn.Module):
    """把聚类系数标量编码为 hidden_dim 向量。"""

    def __init__(self, hidden_dim: int):
        super().__init__()
        self.clustering_coefficient_linear = nn.Linear(1, hidden_dim)

    def forward(self, clustering_coefficient: torch.Tensor) -> torch.Tensor:
        # 支持 (..., 1) 输入，输出 (..., hidden_dim)
        if clustering_coefficient.shape[-1] != 1:
            clustering_coefficient = clustering_coefficient.unsqueeze(-1)
        return self.clustering_coefficient_linear(clustering_coefficient)


class PositionEncodingBidirectionalLinks(nn.Module):
    """把双向链接比标量编码为 hidden_dim 向量。"""

    def __init__(self, hidden_dim: int):
        super().__init__()
        self.bidirectional_links_ratio_linear = nn.Linear(1, hidden_dim)

    def forward(self, bidirectional_links_ratio: torch.Tensor) -> torch.Tensor:
        if bidirectional_links_ratio.shape[-1] != 1:
            bidirectional_links_ratio = bidirectional_links_ratio.unsqueeze(-1)
        return self.bidirectional_links_ratio_linear(bidirectional_links_ratio)


class GraphStructuralLayer(nn.Module):
    """BotDGT 结构层：两层 TransformerConv + PReLU + 残差。

    注意：与 RGCNEncoder 不同，本层**不使用 edge_type**，仅使用 edge_index。
    """

    def __init__(self, hidden_dim: int, n_heads: int, dropout: float):
        super().__init__()
        self.activation = nn.PReLU()
        self.dropout = nn.Dropout(p=dropout)
        self.layer1 = TransformerConv(
            hidden_dim, hidden_dim // n_heads, heads=n_heads,
            concat=True, dropout=dropout,
        )
        self.layer2 = TransformerConv(
            hidden_dim, hidden_dim // n_heads, heads=n_heads,
            concat=True, dropout=dropout,
        )
        self.init_weights()

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        out1 = self.layer1(x, edge_index)
        out1 = self.activation(out1)
        out1 = self.layer2(out1, edge_index)
        out1 = self.dropout(out1)
        out1 += x
        out1 = self.activation(out1)
        return out1

    def init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.kaiming_normal_(module.weight.data)
                if module.bias is not None:
                    module.bias.data.zero_()
            elif isinstance(module, nn.LayerNorm):
                module.weight.data.fill_(1.0)
                module.bias.data.zero_()
