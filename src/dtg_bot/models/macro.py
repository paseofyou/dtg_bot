"""宏观分支：真实动态图快照演化编码器。

这是本项目**唯一真实可得的时序信号**（微观顺序已实测无信息，见 AGENTS.md）。

数据依据：TwiBot-20 全部标注用户的 `created_at` 缺失率为 0；
按创建时间分位数切成 8 个快照，边覆盖从 3.2% 单调增长到 100%（2009→2020），分布均匀。

建模对象是**邻域结构随时间的演化**，不是用户属性的演化 ——
单快照数据无法重构后者，上一版方法正是在这里被迫合成数据。

实现：K 个快照共享同一套 RGCN 权重（同 BotDGT），
逐快照得到节点表示 h_k，再沿 k 方向做时序聚合。
共享权重的理由：参数量不随 K 增长，且强制模型学"结构模式"而非"某个时刻的特例"。
"""

from __future__ import annotations

import torch
import torch.nn as nn

from .graph import RGCNEncoder


class MacroSnapshotEncoder(nn.Module):
    """对 K 个动态图快照做 RGCN 编码，再沿时间维聚合。

    Args:
        temporal: ``gru`` 单向 GRU（快照有明确时间方向，单向即可）；
            ``transformer`` 带可学习位置编码的自注意力；
            ``last`` 只取最后一个快照（= 静态全图，用作宏观分支的消融对照）。
    """

    def __init__(self, emb: int = 64, num_relations: int = 2, num_snapshots: int = 8,
                 out_dim: int = 64, dropout: float = 0.3, temporal: str = "gru",
                 n_heads: int = 4, n_layers: int = 1):
        super().__init__()
        self.num_snapshots = num_snapshots
        self.temporal = temporal
        # 所有快照共享同一套 RGCN 权重
        self.rgcn = RGCNEncoder(emb, num_relations, dropout)

        if temporal == "gru":
            self.seq = nn.GRU(emb, out_dim, batch_first=True)
        elif temporal == "transformer":
            self.pos_emb = nn.Embedding(num_snapshots, emb)
            layer = nn.TransformerEncoderLayer(
                d_model=emb, nhead=n_heads, dim_feedforward=emb * 4,
                dropout=dropout, activation="gelu", batch_first=True, norm_first=True,
            )
            self.seq = nn.TransformerEncoder(layer, num_layers=n_layers)
            self.proj = nn.Linear(emb, out_dim)
            nn.init.normal_(self.pos_emb.weight, std=0.02)
        elif temporal == "last":
            self.proj = nn.Linear(emb, out_dim)
        else:
            raise ValueError(f"unknown temporal: {temporal}")
        self.norm = nn.LayerNorm(out_dim)

    def forward(self, x, edge_index, edge_type, snapshot_masks) -> torch.Tensor:
        """
        Args:
            x: (N, emb) 静态特征编码后的节点表示
            snapshot_masks: 长度 K 的 bool 张量列表，每个 (E,)，
                第 k 个只保留两端账号在 t_k 前已创建的边

        Returns:
            (N, out_dim)
        """
        if self.temporal == "last":
            h = self.rgcn(x, edge_index, edge_type, snapshot_masks[-1])
            return self.norm(self.proj(h))

        # (K, N, emb) → (N, K, emb)
        states = torch.stack(
            [self.rgcn(x, edge_index, edge_type, m) for m in snapshot_masks], dim=0
        ).transpose(0, 1)

        if self.temporal == "gru":
            out, _ = self.seq(states)
            return self.norm(out[:, -1])          # 取末态：累积了全部演化历史

        pos = torch.arange(states.size(1), device=states.device)
        h = self.seq(states + self.pos_emb(pos).unsqueeze(0))
        return self.norm(self.proj(h.mean(dim=1)))
