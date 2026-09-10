"""把预处理产物装配成训练所需的全图张量。

⚠️ 本模块的核心职责之一是**断言 jsonl 行序与图节点序一致**。
上一版项目最大的隐患就是"索引应该一致"这种未经检验的假设
（train_idx.pt 与官方划分不符却被继续使用）。这里逐条比对 user_id，
不一致直接抛错，绝不静默继续。
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from .encode import load_matrix
from .graph import load_graph, load_snapshot_properties, normalize_num_prop


def _read_jsonl_ids(path: Path) -> np.ndarray:
    ids = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            ids.append(json.loads(line)["user_id"])
    return np.array(ids)


def load_twibot20(
    cache: str | Path,
    seq_len: int = 32,
    num_snapshots: int = 8,
    device: str = "cpu",
    need_micro: bool = True,
) -> dict:
    """装配 TwiBot-20 的全图训练数据。

    Returns 字典，含模型 forward 所需的全部张量，以及 train/dev/test 索引。
    """
    cache = Path(cache)
    graph_dir = cache / "graph"
    feat_dir = cache / "features"
    nodes_jsonl = next(cache.glob("nodes_cap*.jsonl"), None)
    if nodes_jsonl is None:
        raise FileNotFoundError(f"未找到 nodes_cap*.jsonl，请先跑 prepare_twibot20_full.py")

    g = load_graph(graph_dir)
    n_nodes = g["meta"]["n_nodes"]

    # --- 一致性断言：jsonl 行序 == 图节点序 ---
    jsonl_ids = _read_jsonl_ids(nodes_jsonl)
    if len(jsonl_ids) != n_nodes:
        raise ValueError(f"节点数不一致: jsonl {len(jsonl_ids)} vs graph {n_nodes}")
    mismatch = np.flatnonzero(jsonl_ids != g["node_ids"])
    if mismatch.size:
        raise ValueError(
            f"jsonl 行序与图节点序不一致，首个不匹配在第 {mismatch[0]} 行："
            f"jsonl={jsonl_ids[mismatch[0]]} graph={g['node_ids'][mismatch[0]]}"
        )

    split = g["node_split"]
    labels = g["node_label"].astype(np.int64)
    idx = {name: np.flatnonzero(split == name) for name in ("train", "dev", "test")}
    for name, ids in idx.items():
        if ids.size == 0:
            raise ValueError(f"{name} split 为空")
        if (labels[ids] < 0).any():
            raise ValueError(f"{name} split 内存在无标签节点")

    num_prop = normalize_num_prop(g["num_prop"], idx["train"])
    des = np.asarray(load_matrix(feat_dir, "description"), dtype=np.float32)
    tweet = np.asarray(load_matrix(feat_dir, "tweet_pooled"), dtype=np.float32)
    for name, mat in (("description", des), ("tweet_pooled", tweet)):
        if mat.shape[0] != n_nodes:
            raise ValueError(f"{name} 行数 {mat.shape[0]} != 节点数 {n_nodes}")

    edge_index = torch.from_numpy(g["edge_index"]).long()
    edge_type = torch.from_numpy(g["edge_type"].astype(np.int64))
    snap = load_snapshot_properties(graph_dir)

    data = {
        "des": torch.from_numpy(des),
        "tweet": torch.from_numpy(tweet),
        "num_prop": torch.from_numpy(num_prop),
        "cat_prop": torch.from_numpy(g["cat_prop"]),
        "edge_index": edge_index,
        "edge_type": edge_type,
        "snapshot_masks": [m for m in snap["snapshot_masks"]],
        "snapshot_clustering_coefficient": snap["snapshot_clustering_coefficient"].float(),
        "snapshot_bidirectional_links_ratio": snap["snapshot_bidirectional_links_ratio"].float(),
        "snapshot_exist_nodes": snap["snapshot_exist_nodes"].float().squeeze(-1),  # (K, N)
        "labels": torch.from_numpy(labels),
        "idx": {k: torch.from_numpy(v) for k, v in idx.items()},
        "snapshot_cutoffs": snap["snapshot_cutoffs"],
        "n_nodes": n_nodes,
        "graph_meta": g["meta"],
    }

    if need_micro:
        micro_dir = cache / f"micro_L{seq_len}"
        enc_dir = cache / f"encoded_L{seq_len}"
        feat = np.load(micro_dir / "micro_feat.npy")
        mmask = np.load(micro_dir / "micro_mask.npy")
        micro_meta = json.loads((micro_dir / "micro_meta.json").read_text(encoding="utf-8"))
        micro_ids = np.load(enc_dir / "user_ids.npy")

        # 微观特征只覆盖标注用户；散射回全图行号，support 节点补零
        id_to_row = {uid: i for i, uid in enumerate(g["node_ids"])}
        rows = np.array([id_to_row[u] for u in micro_ids], dtype=np.int64)
        if len(set(rows.tolist())) != len(rows):
            raise ValueError("微观特征的 user_id 在图中出现重复映射")

        full = np.zeros((n_nodes, feat.shape[1], feat.shape[2]), dtype=np.float32)
        full_mask = np.zeros((n_nodes, mmask.shape[1]), dtype=bool)
        full[rows] = feat
        full_mask[rows] = mmask
        data["micro_feat"] = torch.from_numpy(full)
        data["micro_mask"] = torch.from_numpy(full_mask)
        data["micro_meta"] = micro_meta
        # 微观视图不参与图传播（直接进融合层→分类头），support 节点的微观输出
        # 不被任何损失使用、梯度恒为 0。因此只在标注节点上计算，数学等价而非近似。
        # 不这么做会在全部 n_nodes 行上跑 Transformer：注意力矩阵
        # n_nodes × heads × L × L 约 3.8GB/层，24GB 卡上直接 OOM。
        labeled_rows = np.concatenate([idx[name] for name in ("train", "dev", "test")])
        data["micro_index"] = torch.from_numpy(np.sort(labeled_rows))

    if device != "cpu":
        data = to_device(data, device)
    return data


def to_device(data: dict, device: str) -> dict:
    out = {}
    for k, v in data.items():
        if torch.is_tensor(v):
            out[k] = v.to(device)
        elif isinstance(v, list) and v and torch.is_tensor(v[0]):
            out[k] = [t.to(device) for t in v]
        elif k == "idx":
            out[k] = {kk: vv.to(device) for kk, vv in v.items()}
        else:
            out[k] = v
    return out
