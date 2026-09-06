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
    ):
        super().__init__()
        bad = set(use_views) - set(VIEW_NAMES)
        if bad:
            raise ValueError(f"unknown views: {bad}")
        if not use_views:
            raise ValueError("至少要启用一个视图")
        self.use_views = tuple(v for v in VIEW_NAMES if v in use_views)
        self.fusion_type = fusion

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

    def encode_views(self, batch: dict) -> dict[str, torch.Tensor]:
        """返回各视图的节点表示。batch 需含图分支所需的全图张量。"""
        views: dict[str, torch.Tensor] = {}
        need_graph = {"macro", "global"} & set(self.use_views)
        x = self.static(batch["des"], batch["tweet"], batch["num_prop"], batch["cat_prop"]) \
            if need_graph else None

        if "micro" in self.use_views:
            views["micro"] = self.micro(batch["micro_feat"], batch["micro_mask"])
        if "macro" in self.use_views:
            views["macro"] = self.macro(
                x, batch["edge_index"], batch["edge_type"], batch["snapshot_masks"]
            )
        if "global" in self.use_views:
            views["global"] = self.global_norm(
                self.global_rgcn(x, batch["edge_index"], batch["edge_type"])
            )
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
