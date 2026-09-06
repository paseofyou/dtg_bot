"""图分支：BotRGCN 风格的多关系拓扑编码器。

`BotRGCN` 类是**忠实复现的基线**，不要在里面加任何本文的新模块 ——
基线必须能独立跑到官方水平（TwiBot-20 F1 87.07 / TwiBot-22 F1 57.50）才有资格做对比。
上一版复现的 TwiBot-22 基线只有 53.60，低了 4 分，导致"超过基线"是弱基线假象。

四路输入各投影到 emb/4 维后拼接，与原文一致：
    description / tweet（均为 RoBERTa 768 维）、num_prop、cat_prop
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import RGCNConv


class StaticFeatureEncoder(nn.Module):
    """把四路异构静态特征编码并拼接成 emb 维输入表示。"""

    def __init__(self, des_size=768, tweet_size=768, num_prop_size=7, cat_prop_size=3,
                 emb=64, dropout=0.3):
        super().__init__()
        quarter = emb // 4
        self.des = nn.Sequential(nn.Linear(des_size, quarter), nn.LeakyReLU())
        self.tweet = nn.Sequential(nn.Linear(tweet_size, quarter), nn.LeakyReLU())
        self.num = nn.Sequential(nn.Linear(num_prop_size, quarter), nn.LeakyReLU())
        self.cat = nn.Sequential(nn.Linear(cat_prop_size, emb - 3 * quarter), nn.LeakyReLU())
        self.fuse = nn.Sequential(nn.Linear(emb, emb), nn.LeakyReLU())
        self.dropout = dropout

    def forward(self, des, tweet, num_prop, cat_prop):
        x = torch.cat(
            [self.des(des), self.tweet(tweet), self.num(num_prop), self.cat(cat_prop)], dim=1
        )
        x = self.fuse(x)
        return F.dropout(x, p=self.dropout, training=self.training)


class RGCNEncoder(nn.Module):
    """两层 RGCN。可传入 edge_mask 只在子图上传播（宏观快照分支复用此点）。"""

    def __init__(self, emb=64, num_relations=2, dropout=0.3):
        super().__init__()
        self.conv1 = RGCNConv(emb, emb, num_relations=num_relations)
        self.conv2 = RGCNConv(emb, emb, num_relations=num_relations)
        self.dropout = dropout

    def forward(self, x, edge_index, edge_type, edge_mask=None):
        if edge_mask is not None:
            edge_index = edge_index[:, edge_mask]
            edge_type = edge_type[edge_mask]
        x = F.leaky_relu(self.conv1(x, edge_index, edge_type))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = F.leaky_relu(self.conv2(x, edge_index, edge_type))
        return x


class BotRGCN(nn.Module):
    """BotRGCN 基线复现。**保持原样，不要添加本文模块。**"""

    def __init__(self, des_size=768, tweet_size=768, num_prop_size=7, cat_prop_size=3,
                 emb=64, num_relations=2, dropout=0.3):
        super().__init__()
        self.static = StaticFeatureEncoder(
            des_size, tweet_size, num_prop_size, cat_prop_size, emb, dropout
        )
        self.rgcn = RGCNEncoder(emb, num_relations, dropout)
        self.out = nn.Sequential(nn.Linear(emb, emb), nn.LeakyReLU())
        self.classifier = nn.Linear(emb, 2)

    def forward(self, des, tweet, num_prop, cat_prop, edge_index, edge_type):
        x = self.static(des, tweet, num_prop, cat_prop)
        x = self.rgcn(x, edge_index, edge_type)
        return self.classifier(self.out(x))
