"""TwiBot-20 图结构与静态特征构建。

节点顺序严格约定为 **train → dev → test → support**，与 micro 分支的 jsonl
顺序对齐（前 11826 个节点即标注用户，且顺序一致），避免上一版"索引错位"类的坑。

边关系两类，与 BotRGCN 一致：
    relation 0 = following   (u → v，u 关注 v)
    relation 1 = follower    (u ← v，v 关注 u)

**每个节点的 `created_at` 必须落盘**：它是宏观动态图快照掩码的唯一真实锚点，
也是本项目唯一可用的真实时间信号（见 AGENTS.md）。
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from tqdm import tqdm

from .twibot20 import LABELED_SPLITS, SPLITS, iter_split

RELATION_NAMES = ("following", "follower")


def build_graph(
    raw_dir: str | Path,
    out_dir: str | Path,
    splits: tuple[str, ...] = SPLITS,
    log_every: int = 20000,
) -> dict:
    """单遍扫描全部 split，产出节点表与边表。

    产物（out_dir 下）：
        node_ids.npy      (N,)  str        节点 Twitter ID，顺序 train→dev→test→support
        node_split.npy    (N,)  str
        node_label.npy    (N,)  int8       -1 表示 support（无标签）
        num_prop.npy      (N,7) float32    原始值，未归一化
        cat_prop.npy      (N,3) float32
        created_ts.npy    (N,)  float64    account created_at 的 epoch 秒，缺失为 nan
        edge_index.npy    (2,E) int64
        edge_type.npy     (E,)  int8
        graph_meta.json
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    node_ids: list[str] = []
    node_split: list[str] = []
    node_label: list[int] = []
    num_prop: list[list[float]] = []
    cat_prop: list[list[float]] = []
    created_ts: list[float] = []
    # 先按字符串 ID 收边，全部节点登记完毕后再映射成整数索引
    raw_edges: list[tuple[str, str, int]] = []

    for split in splits:
        n_split = 0
        pbar = tqdm(iter_split(raw_dir, split, seq_len=0), desc=f"scan {split}", unit="user")
        for rec in pbar:
            node_ids.append(rec.user_id)
            node_split.append(split)
            node_label.append(-1 if rec.label is None else rec.label)
            num_prop.append(rec.num_properties)
            cat_prop.append(rec.cat_properties)
            created_ts.append(rec.created_at.timestamp() if rec.created_at else np.nan)

            for target in rec.following:
                raw_edges.append((rec.user_id, target, 0))
            for source in rec.followers:
                raw_edges.append((rec.user_id, source, 1))

            n_split += 1
            if log_every and n_split % log_every == 0:
                pbar.set_postfix(nodes=len(node_ids), edges=len(raw_edges))
        pbar.close()

    id_to_idx = {uid: i for i, uid in enumerate(node_ids)}

    # 只保留两端都在节点表内的边（TwiBot-20 的 neighbor 列表会引用未收录的账号）
    src, dst, rel, dropped = [], [], [], 0
    for u, v, r in raw_edges:
        iu, iv = id_to_idx.get(u), id_to_idx.get(v)
        if iu is None or iv is None:
            dropped += 1
            continue
        src.append(iu)
        dst.append(iv)
        rel.append(r)

    edge_index = np.array([src, dst], dtype=np.int64)
    edge_type = np.array(rel, dtype=np.int8)

    np.save(out_dir / "node_ids.npy", np.array(node_ids))
    np.save(out_dir / "node_split.npy", np.array(node_split))
    np.save(out_dir / "node_label.npy", np.array(node_label, dtype=np.int8))
    np.save(out_dir / "num_prop.npy", np.array(num_prop, dtype=np.float32))
    np.save(out_dir / "cat_prop.npy", np.array(cat_prop, dtype=np.float32))
    np.save(out_dir / "created_ts.npy", np.array(created_ts, dtype=np.float64))
    np.save(out_dir / "edge_index.npy", edge_index)
    np.save(out_dir / "edge_type.npy", edge_type)

    labels = np.array(node_label)
    meta = {
        "n_nodes": len(node_ids),
        "n_edges": int(edge_index.shape[1]),
        "n_edges_dropped_unknown_endpoint": dropped,
        "relations": list(RELATION_NAMES),
        "n_labeled": int((labels >= 0).sum()),
        "n_bot": int((labels == 1).sum()),
        "n_human": int((labels == 0).sum()),
        "split_sizes": {s: int((np.array(node_split) == s).sum()) for s in splits},
        "n_missing_created_at": int(np.isnan(np.array(created_ts)).sum()),
        "node_order": list(splits),
    }
    (out_dir / "graph_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return meta


def load_graph(out_dir: str | Path) -> dict:
    out_dir = Path(out_dir)
    return {
        "node_ids": np.load(out_dir / "node_ids.npy"),
        "node_split": np.load(out_dir / "node_split.npy"),
        "node_label": np.load(out_dir / "node_label.npy"),
        "num_prop": np.load(out_dir / "num_prop.npy"),
        "cat_prop": np.load(out_dir / "cat_prop.npy"),
        "created_ts": np.load(out_dir / "created_ts.npy"),
        "edge_index": np.load(out_dir / "edge_index.npy"),
        "edge_type": np.load(out_dir / "edge_type.npy"),
        "meta": json.loads((out_dir / "graph_meta.json").read_text(encoding="utf-8")),
    }


def normalize_num_prop(num_prop: np.ndarray, fit_rows: np.ndarray) -> np.ndarray:
    """对计数类特征先取 log1p 再 z-score，统计量只用 fit_rows（训练集）。

    计数特征（粉丝数等）跨越多个数量级且长尾极重，直接 z-score 会被极端值主导 ——
    这是 BotRGCN 复现常见的掉点原因。
    """
    x = np.array(num_prop, dtype=np.float32)
    x[:, :5] = np.log1p(np.clip(x[:, :5], 0, None))   # 5 个 count 字段
    mean = x[fit_rows].mean(axis=0)
    std = x[fit_rows].std(axis=0)
    std[std < 1e-6] = 1.0
    return (x - mean) / std


def snapshot_edge_masks(
    edge_index: np.ndarray,
    created_ts: np.ndarray,
    num_snapshots: int = 8,
) -> tuple[list[np.ndarray], np.ndarray]:
    """宏观分支：按账号创建时间生成**真实**的动态图快照掩码序列。

    第 k 个快照只保留两端账号都已在时刻 t_k 之前创建的边，
    即"在 t_k 时点这张图长什么样"。这是本项目唯一真实可得的结构演化信号
    —— 注意它刻画的是**邻域结构的演化**，不是用户属性的演化
    （单快照数据无法重构后者，上一版正是在这里被迫合成数据）。

    Returns:
        masks: 长度 num_snapshots 的 bool 数组列表，每个形状 (E,)
        cutoffs: (num_snapshots,) 各快照的时间切点（epoch 秒）
    """
    ts = np.asarray(created_ts, dtype=np.float64)
    valid = ~np.isnan(ts)
    # 用分位数切点而非等距，保证每个快照都有足量新增节点
    quantiles = np.linspace(1.0 / num_snapshots, 1.0, num_snapshots)
    cutoffs = np.quantile(ts[valid], quantiles)

    # 缺失 created_at 的节点按"最早"处理：始终存在，避免把它们整条排除
    node_ts = np.where(valid, ts, -np.inf)
    src_ts = node_ts[edge_index[0]]
    dst_ts = node_ts[edge_index[1]]
    edge_ts = np.maximum(src_ts, dst_ts)      # 边在两端都创建后才可能存在

    masks = [edge_ts < c for c in cutoffs]
    return masks, cutoffs


def _interval_cutoffs(created_ts: np.ndarray, interval: str = "year"):
    """按 BotDGT 原装思路生成时间区间切点：每年/每月首日的 epoch 秒。"""
    ts = np.asarray(created_ts, dtype=np.float64)
    valid = ~np.isnan(ts)
    if not valid.any():
        return []
    min_t = datetime.fromtimestamp(ts[valid].min(), tz=timezone.utc)
    max_t = datetime.fromtimestamp(ts[valid].max(), tz=timezone.utc)
    start_year = min_t.year
    end_year = max_t.year

    cutoffs = []
    if interval == "year":
        for y in range(start_year, end_year + 1):
            cutoffs.append(datetime(y, 1, 1, tzinfo=timezone.utc).timestamp())
    elif interval == "month":
        from calendar import monthrange
        for y in range(start_year, end_year + 1):
            for m in range(1, 13):
                cutoffs.append(datetime(y, m, 1, tzinfo=timezone.utc).timestamp())
    else:
        raise ValueError(f"unknown interval: {interval}")
    return cutoffs


def build_snapshot_properties(
    edge_index: np.ndarray,
    edge_type: np.ndarray,
    created_ts: np.ndarray,
    num_snapshots: int = 8,
    interval: str = "year",
    following_relation: int = 0,
) -> dict:
    """构建 BotDGT 原装快照属性：节点是否存在、聚类系数、双向链接比。

    Returns:
        masks: K 个 (E,) bool 边掩码
        exist_nodes: (K, N) 0/1
        clustering_coefficient: (K, N, 1) float
        bidirectional_links_ratio: (K, N, 1) float
        cutoffs: (K,) float64
    """
    import networkx as nx

    n_nodes = created_ts.shape[0]
    ts = np.asarray(created_ts, dtype=np.float64)
    valid = ~np.isnan(ts)
    node_ts = np.where(valid, ts, -np.inf)

    all_cutoffs = _interval_cutoffs(created_ts, interval)
    if not all_cutoffs:
        raise ValueError("无法生成快照切点：无有效 created_at")

    # 取最后 num_snapshots 个快照
    cutoffs = all_cutoffs[-num_snapshots:]

    masks = []
    exist_list = []
    cc_list = []
    blr_list = []

    for c in cutoffs:
        # 节点存在：created_at < cutoff（与 BotDGT split_user_by_interval 一致）
        exist = (node_ts < c).astype(np.float32)

        # 边存在：两端都已在 cutoff 前创建
        src_ts = node_ts[edge_index[0]]
        dst_ts = node_ts[edge_index[1]]
        edge_ts = np.maximum(src_ts, dst_ts)
        mask = edge_ts < c
        masks.append(mask)

        sub_ei = edge_index[:, mask]

        # 聚类系数（无向图）
        G = nx.Graph()
        existing = np.flatnonzero(exist).tolist()
        G.add_nodes_from(existing)
        if sub_ei.shape[1] > 0:
            G.add_edges_from(sub_ei.T.tolist())
        cc_dict = nx.clustering(G)
        cc = np.zeros(n_nodes, dtype=np.float32)
        for node in existing:
            cc[node] = float(cc_dict.get(node, 0.0))
        cc_list.append(cc.reshape(-1, 1))

        # 双向链接比（有向图，仅 following 边）
        follow_mask = mask & (edge_type == following_relation)
        follow_ei = edge_index[:, follow_mask]
        D = nx.DiGraph()
        D.add_nodes_from(existing)
        if follow_ei.shape[1] > 0:
            D.add_edges_from(follow_ei.T.tolist())

        blr = np.zeros(n_nodes, dtype=np.float32)
        for i in existing:
            succ = set(D.successors(i))
            if not succ:
                continue
            reciprocal = sum(1 for j in succ if D.has_edge(j, i))
            blr[i] = reciprocal / len(succ)
        blr_list.append(blr.reshape(-1, 1))

        exist_list.append(exist.reshape(-1, 1))

    return {
        "masks": masks,
        "exist_nodes": np.stack(exist_list, axis=0),        # (K, N, 1)
        "clustering_coefficient": np.stack(cc_list, axis=0),  # (K, N, 1)
        "bidirectional_links_ratio": np.stack(blr_list, axis=0),  # (K, N, 1)
        "cutoffs": np.array(cutoffs, dtype=np.float64),
    }


def save_snapshot_properties(
    out_dir: str | Path,
    props: dict,
) -> None:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(out_dir / "snapshot_masks.npy", np.stack(props["masks"], axis=0))
    np.save(out_dir / "snapshot_exist_nodes.npy", props["exist_nodes"].astype(np.int8))
    np.save(out_dir / "snapshot_clustering_coefficient.npy", props["clustering_coefficient"])
    np.save(out_dir / "snapshot_bidirectional_links_ratio.npy", props["bidirectional_links_ratio"])
    np.save(out_dir / "snapshot_cutoffs.npy", props["cutoffs"])


def load_snapshot_properties(out_dir: str | Path) -> dict:
    out_dir = Path(out_dir)
    return {
        "snapshot_masks": [torch.from_numpy(m) for m in np.load(out_dir / "snapshot_masks.npy")],
        "snapshot_exist_nodes": torch.from_numpy(np.load(out_dir / "snapshot_exist_nodes.npy")),
        "snapshot_clustering_coefficient": torch.from_numpy(np.load(out_dir / "snapshot_clustering_coefficient.npy")),
        "snapshot_bidirectional_links_ratio": torch.from_numpy(np.load(out_dir / "snapshot_bidirectional_links_ratio.npy")),
        "snapshot_cutoffs": np.load(out_dir / "snapshot_cutoffs.npy"),
    }
