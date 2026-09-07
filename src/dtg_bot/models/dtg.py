"""完整模型：事件级注意力 + 宏观动态图演化 + 全局多关系拓扑。

三个分支的定位（每一条都有实测依据，见 AGENTS.md）：

    micro  事件级注意力  —— 在一组推文中挑出异常的几条。
                            相比 BotRGCN 的平均池化 AUC +4.01 (p<1e-5)。
                            **顺序无关**：shuffle 后 ΔAUC=-0.09 (p=0.644)。
    macro  动态图快照演化 —— 唯一真实时序信号，锚点是 created_at（缺失率 0）。
    global 当前全景拓扑   —— RGCN 全图，等价于 BotRGCN 的图分支。

``use_views`` 控制启用哪些分支，用于单视图/双视图消融。
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint

from .fusion import GatedFusion, ViewAttentionFusion, multi_view_contrastive
from .graph import RGCNEncoder, StaticFeatureEncoder
from .micro import EventAttentionEncoder

VIEW_NAMES = ("micro", "macro", "global")


class DTGBot(nn.Module):
    def __init__(
        self,
        micro_channels: int,
        micro_seq_len: int,
        des_size: int = 768,
        tweet_size: int = 768,
        num_prop_size: int = 7,
        cat_prop_size: int = 3,
        emb: int = 64,
        num_relations: int = 2,
        num_snapshots: int = 8,
        dropout: float = 0.3,
        micro_dropout: float = 0.1,
        micro_layers: int = 2,
        micro_heads: int = 4,
        macro_temporal: str = "gru",
        fusion: str = "attn",
        use_views: tuple[str, ...] = VIEW_NAMES,
        micro_seq_model: str = "transformer",
        micro_order_mode: str = "keep",
        micro_use_position: bool = False,
        macro_checkpoint: bool = True,
    ):
        super().__init__()
        bad = set(use_views) - set(VIEW_NAMES)
        if bad:
            raise ValueError(f"unknown views: {bad}")
        if not use_views:
            raise ValueError("至少要启用一个视图")
        self.use_views = tuple(v for v in VIEW_NAMES if v in use_views)
        self.fusion_type = fusion
        self.macro_checkpoint = macro_checkpoint

        # 静态特征编码器由 macro / global 两个图分支共享
        self.static = StaticFeatureEncoder(
            des_size, tweet_size, num_prop_size, cat_prop_size, emb, dropout
        )

        if "micro" in self.use_views:
            self.micro = EventAttentionEncoder(
                in_channels=micro_channels, d_model=emb, out_dim=emb,
                n_layers=micro_layers, n_heads=micro_heads, max_len=micro_seq_len,
                dropout=micro_dropout, seq_model=micro_seq_model,
                order_mode=micro_order_mode, use_position=micro_use_position,
            )
        if "macro" in self.use_views:
            from .macro import MacroSnapshotEncoder
            self.macro = MacroSnapshotEncoder(
                emb=emb, num_relations=num_relations, num_snapshots=num_snapshots,
                out_dim=emb, dropout=dropout, temporal=macro_temporal,
                checkpoint=macro_checkpoint,
            )
        if "global" in self.use_views:
            self.global_rgcn = RGCNEncoder(emb, num_relations, dropout)
            self.global_norm = nn.LayerNorm(emb)

        n_views = len(self.use_views)
        if n_views == 1:
            self.fuse = None
        elif fusion == "attn":
            self.fuse = ViewAttentionFusion(emb, n_views, micro_dropout)
        elif fusion == "gate":
            self.fuse = GatedFusion(emb, n_views)
        elif fusion == "concat":
            self.fuse = None
        else:
            raise ValueError(f"unknown fusion: {fusion}")

        head_in = emb * n_views if (fusion == "concat" and n_views > 1) else emb
        self.head = nn.Sequential(
            nn.Linear(head_in, emb), nn.LeakyReLU(), nn.Dropout(dropout), nn.Linear(emb, 2)
        )

    def _encode_micro(self, batch: dict) -> torch.Tensor:
        """微观分支前向。

        ``batch["micro_index"]`` 给出需要计算的行（标注节点）时只在这些行上跑
        编码器，其余行补零。微观视图不经图传播，未标注节点的输出不进入任何损失、
        梯度恒为 0，故此举与全量前向**数学等价**；但把注意力矩阵从
        n_nodes × heads × L × L（约 3.8GB/层）压到标注节点规模，避免 OOM。
        """
        feat, mask = batch["micro_feat"], batch["micro_mask"]
        rows = batch.get("micro_index")
        if rows is None:
            return self.micro(feat, mask)
        z = self.micro(feat[rows], mask[rows])
        out = z.new_zeros((feat.shape[0], z.shape[1]))
        out[rows] = z
        return out

    def encode_views(self, batch: dict) -> dict[str, torch.Tensor]:
        """返回各视图的节点表示。batch 需含图分支所需的全图张量。"""
        views: dict[str, torch.Tensor] = {}
        need_graph = {"macro", "global"} & set(self.use_views)
        x = self.static(batch["des"], batch["tweet"], batch["num_prop"], batch["cat_prop"]) \
            if need_graph else None

        if "micro" in self.use_views:
            views["micro"] = self._encode_micro(batch)
        if "macro" in self.use_views:
            views["macro"] = self.macro(
                x, batch["edge_index"], batch["edge_type"], batch["snapshot_masks"]
            )
        if "global" in self.use_views:
            # 全局分支同样是一次全图 RGCN，其反向中间量与单个宏观快照同量级。
            # 实测在合成图上它比 8 个宏观快照加起来还多（1.8 vs 0.9 GiB），
            # 故与宏观分支采用同一开关一并检查点化。
            if self.macro_checkpoint and self.training and torch.is_grad_enabled():
                h = checkpoint(self.global_rgcn, x, batch["edge_index"],
                               batch["edge_type"], use_reentrant=False)
            else:
                h = self.global_rgcn(x, batch["edge_index"], batch["edge_type"])
            views["global"] = self.global_norm(h)
        return views

    def forward(self, batch: dict, return_views: bool = False):
        views = self.encode_views(batch)
        ordered = [views[v] for v in self.use_views]

        if len(ordered) == 1:
            fused = ordered[0]
        elif self.fusion_type == "concat":
            fused = torch.cat(ordered, dim=-1)
        else:
            fused, _ = self.fuse(ordered)

        logits = self.head(fused)
        return (logits, views) if return_views else logits

    @staticmethod
    def contrastive_loss(views: dict[str, torch.Tensor], temperature: float = 0.5,
                         max_samples: int = 4096) -> torch.Tensor:
        vs = list(views.values())
        if len(vs) < 2:
            return vs[0].new_zeros(())
        return multi_view_contrastive(vs, temperature, max_samples)
