"""TwiBot-22 原始数据解析、构图与节点序列导出。

数据事实（已核验，见 AGENTS.md）:
- user.json 是 JSON 数组，含 id (u 前缀)、public_metrics、description、created_at。
- label.csv: id, label（human/bot）
- split.csv: id, split（train/valid/test）
- edge.csv: source_id, relation, target_id
- tweet_0..8.json: 每条推文含 author_id（裸整数）、text、created_at。

节点顺序约定与 user.json 一致；split 中的 "valid" 映射为 "dev"，
与 TwiBot-20 的 train/dev/test 协议对齐。
"""

from __future__ import annotations

import array
import csv
import heapq
import json
import random
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator

import ijson
import numpy as np
from tqdm import tqdm

# 采集截止日期。active_days 会做 z-score，整体平移不影响标准化结果。
CRAWL_DATE = datetime(2023, 1, 1, tzinfo=timezone.utc)

# 与 TwiBot-20 num_prop 对齐：5 个计数 + active_days + screen_name_len
_NUM_COUNT_FIELDS = (
    "followers_count",
    "friends_count",
    "statuses_count",
    "favourites_count",
    "listed_count",
)

# TwiBot-22 public_metrics 字段名与内部字段的映射；无对应时填 0
_METRIC_TO_KEY = {
    "followers_count": "followers_count",
    "friends_count": "following_count",
    "statuses_count": "tweet_count",
    "favourites_count": None,
    "listed_count": "listed_count",
}

# 与 TwiBot-20 对齐；default_profile 在 T22 不存在，用 has_url 替代
_CAT_FIELDS = ("protected", "verified", "has_url")

_TWEET_TIME_FORMATS = (
    "%Y-%m-%d %H:%M:%S%z",
    "%Y-%m-%dT%H:%M:%S%z",
    "%Y-%m-%d %H:%M:%S.%f%z",
    "%Y-%m-%dT%H:%M:%S.%f%z",
)


def bare_uid(user_id: str | int) -> str:
    """去掉可能存在的 u 前缀，返回裸数字 id 字符串。"""
    return str(user_id).strip().lstrip("uU")


def with_u_prefix(user_id: str | int) -> str:
    return f"u{bare_uid(user_id)}"


def parse_created_at(value) -> datetime | None:
    """解析 TwiBot-22 的 ISO 时间字符串（含 +00:00 偏移）。"""
    text = str(value).strip() if value is not None else ""
    if not text:
        return None
    try:
        return datetime.fromisoformat(text)
    except ValueError:
        pass
    for fmt in _TWEET_TIME_FORMATS:
        try:
            return datetime.strptime(text, fmt)
        except ValueError:
            continue
    return None


def _clean_text(value) -> str:
    if value is None:
        return ""
    return str(value).strip()


def _to_float(v, default: float = 0.0) -> float:
    try:
        return float(v) if v is not None else default
    except (ValueError, TypeError):
        return default


def static_properties(user: dict) -> tuple[list[float], list[float], str, datetime | None, str]:
    """从 user.json 单条记录提取静态特征；返回 (num, cat, desc, created, uid(u 前缀))。"""
    uid = with_u_prefix(user["id"])
    pm = user.get("public_metrics") or {}

    num: list[float] = []
    for internal, raw_key in _METRIC_TO_KEY.items():
        if raw_key is None:
            num.append(0.0)
        else:
            num.append(_to_float(pm.get(raw_key)))

    created = parse_created_at(user.get("created_at"))
    active_days = float((CRAWL_DATE - created).days) if created else 0.0
    screen_name = _clean_text(user.get("username") or user.get("name") or "")
    num.extend([max(active_days, 0.0), float(len(screen_name))])

    cat = [
        float(bool(user.get("protected"))),
        float(bool(user.get("verified"))),
        float(bool(_clean_text(user.get("url")))),
    ]

    desc = _clean_text(user.get("description"))
    return num, cat, desc, created, uid


def _load_csv_map(path: Path, key_col: str, val_col: str) -> dict[str, str]:
    mapping: dict[str, str] = {}
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            mapping[bare_uid(row[key_col])] = row[val_col]
    return mapping


def _sample_uids(
    split_map: dict[str, str],
    label_map: dict[str, str],
    n_users: int | None,
    seed: int,
) -> set[str] | None:
    """按 (split, label) 分层随机抽样，返回裸 uid 集合。"""
    if n_users is None or n_users <= 0 or n_users >= len(split_map):
        return None

    groups: dict[tuple[str, int], list[str]] = defaultdict(list)
    for uid, split in split_map.items():
        lab = label_map.get(uid)
        if lab is None:
            continue
        label = 0 if lab == "human" else 1
        groups[(split, label)].append(uid)

    n_groups = len(groups)
    base = n_users // n_groups
    extra = n_users - base * n_groups
    chosen: set[str] = set()
    rng = random.Random(seed)
    sorted_groups = sorted(groups.items(), key=lambda x: -len(x[1]))
    for i, ((_, _), uids) in enumerate(sorted_groups):
        k = base + (1 if i < extra else 0)
        k = min(k, len(uids))
        if k > 0:
            chosen.update(rng.sample(uids, k))
    return chosen


def build_user_table(
    raw_dir: str | Path,
    cache: str | Path,
    sample_users: int | None = None,
    sample_seed: int = 42,
) -> tuple[dict[str, int], list[str], list[str], list[int], list[str], np.ndarray]:
    """读取 user.json / label.csv / split.csv，写出图节点表与静态特征。

    Returns:
        (uid_to_idx(裸), node_ids(u 前缀), splits, labels, descriptions, created_ts)
    """
    raw_dir, cache = Path(raw_dir), Path(cache)
    graph_dir = cache / "graph"
    graph_dir.mkdir(parents=True, exist_ok=True)

    label_csv = raw_dir / "label.csv"
    split_csv = raw_dir / "split.csv"
    user_json = raw_dir / "user.json"

    if not user_json.exists():
        raise FileNotFoundError(f"未找到 user.json: {user_json}")
    if not label_csv.exists():
        raise FileNotFoundError(f"未找到 label.csv: {label_csv}")
    if not split_csv.exists():
        raise FileNotFoundError(f"未找到 split.csv: {split_csv}")

    label_map = _load_csv_map(label_csv, "id", "label")
    split_map = _load_csv_map(split_csv, "id", "split")
    keep_uids = _sample_uids(split_map, label_map, sample_users, sample_seed)

    def _split(uid: str) -> str:
        s = split_map.get(uid, "support")
        return "dev" if s == "valid" else s

    def _label(uid: str) -> int:
        lab = label_map.get(uid, "")
        if lab == "human":
            return 0
        if lab == "bot":
            return 1
        return -1

    node_ids: list[str] = []
    splits: list[str] = []
    labels: list[int] = []
    descriptions: list[str] = []
    num_prop: list[list[float]] = []
    cat_prop: list[list[float]] = []
    created_ts: list[float] = []

    total_hint = sample_users if sample_users else len(split_map)
    with user_json.open("r", encoding="utf-8") as f:
        for user in tqdm(ijson.items(f, "item", use_float=True),
                         total=total_hint, desc="scan users", unit="user"):
            buid = bare_uid(user.get("id", ""))
            if not buid:
                continue
            if keep_uids is not None and buid not in keep_uids:
                continue
            try:
                n, c, desc, created, uid = static_properties(user)
            except Exception:
                continue

            idx = len(node_ids)
            node_ids.append(uid)
            splits.append(_split(buid))
            labels.append(_label(buid))
            descriptions.append(desc)
            num_prop.append(n)
            cat_prop.append(c)
            created_ts.append(created.timestamp() if created else np.nan)

    n_nodes = len(node_ids)
    if n_nodes == 0:
        raise ValueError("没有选中的用户，请检查 sample_users 或原始数据")

    uid_to_idx = {bare_uid(uid): i for i, uid in enumerate(node_ids)}
    np.save(graph_dir / "node_ids.npy", np.array(node_ids))
    np.save(graph_dir / "node_split.npy", np.array(splits))
    np.save(graph_dir / "node_label.npy", np.array(labels, dtype=np.int8))
    np.save(graph_dir / "num_prop.npy", np.array(num_prop, dtype=np.float32))
    np.save(graph_dir / "cat_prop.npy", np.array(cat_prop, dtype=np.float32))
    np.save(graph_dir / "created_ts.npy", np.array(created_ts, dtype=np.float64))

    (graph_dir / "uid_to_idx.json").write_text(
        json.dumps(uid_to_idx, ensure_ascii=False), encoding="utf-8"
    )

    meta = {
        "n_nodes": n_nodes,
        "n_labeled": int(np.sum(np.array(labels, dtype=np.int8) >= 0)),
        "n_bot": int(np.sum(np.array(labels, dtype=np.int8) == 1)),
        "n_human": int(np.sum(np.array(labels, dtype=np.int8) == 0)),
        "split_sizes": {s: int(splits.count(s)) for s in set(splits)},
        "n_missing_created_at": int(np.isnan(np.array(created_ts, dtype=np.float64)).sum()),
        "node_order": "user.json",
        "relations": ["following", "followers"],
        "n_edges_dropped_unknown_endpoint": 0,
        "n_edges": 0,
    }
    (graph_dir / "graph_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    return uid_to_idx, node_ids, splits, labels, descriptions, np.array(created_ts, dtype=np.float64)


def load_user_table(
    cache: str | Path,
) -> tuple[dict[str, int], list[str], list[str], list[int], np.ndarray]:
    """加载 build_user_table 已保存的节点表。"""
    graph_dir = Path(cache) / "graph"
    uid_to_idx = json.loads((graph_dir / "uid_to_idx.json").read_text(encoding="utf-8"))
    node_ids = list(np.load(graph_dir / "node_ids.npy", allow_pickle=True))
    splits = list(np.load(graph_dir / "node_split.npy", allow_pickle=True))
    labels = list(np.load(graph_dir / "node_label.npy"))
    created_ts = np.load(graph_dir / "created_ts.npy")
    return uid_to_idx, node_ids, splits, labels, created_ts


def build_twibot22_graph(
    raw_dir: str | Path,
    cache: str | Path,
    uid_to_idx: dict[str, int],
) -> dict:
    """读取 edge.csv，仅保留用户之间的 following/followers 边。"""
    raw_dir, cache = Path(raw_dir), Path(cache)
    graph_dir = cache / "graph"
    edge_csv = raw_dir / "edge.csv"

    if not edge_csv.exists():
        raise FileNotFoundError(f"未找到 edge.csv: {edge_csv}")

    REL_MAP = {"following": 0, "followers": 1}
    src = array.array("q")
    dst = array.array("q")
    rel = array.array("b")
    dropped = 0

    chunksize = 10_000_000
    # pandas 在 requirements.txt 中，用分块读取 6.3GB 边表
    import pandas as pd

    for chunk in tqdm(pd.read_csv(
        edge_csv,
        usecols=["source_id", "relation", "target_id"],
        dtype=str,
        chunksize=chunksize,
    ), desc="scan edge chunks"):
        sub = chunk[chunk["relation"].isin(["following", "followers"])].copy()
        sub["s_bare"] = sub["source_id"].astype(str).str.strip().str.lstrip("uU")
        sub["t_bare"] = sub["target_id"].astype(str).str.strip().str.lstrip("uU")
        mask = sub["s_bare"].isin(uid_to_idx) & sub["t_bare"].isin(uid_to_idx)
        dropped += int((~mask).sum())
        sub = sub[mask]
        if sub.empty:
            continue

        s = sub["s_bare"].map(uid_to_idx).to_numpy().astype(np.int64)
        t = sub["t_bare"].map(uid_to_idx).to_numpy().astype(np.int64)
        r = sub["relation"].map(REL_MAP).to_numpy().astype(np.int8)

        src.frombytes(s.tobytes())
        dst.frombytes(t.tobytes())
        rel.frombytes(r.tobytes())

    edge_index = np.array([src, dst], dtype=np.int64)
    edge_type = np.array(rel, dtype=np.int8)
    np.save(graph_dir / "edge_index.npy", edge_index)
    np.save(graph_dir / "edge_type.npy", edge_type)

    meta = json.loads((graph_dir / "graph_meta.json").read_text(encoding="utf-8"))
    meta["n_edges"] = int(edge_index.shape[1])
    meta["n_edges_dropped_unknown_endpoint"] = dropped
    (graph_dir / "graph_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return meta


def collect_tweets(
    raw_dir: str | Path,
    out_path: str | Path,
    uid_to_idx: dict[str, int],
    node_ids: list[str],
    splits: list[str],
    labels: list[int],
    descriptions: list[str],
    tweet_cap: int = 32,
) -> int:
    """流式读取 tweet_*.json，为每个用户保留最近的 tweet_cap 条推文（含时间戳）。"""
    raw_dir, out_path = Path(raw_dir), Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    n_users = len(node_ids)

    # 每个用户的 (timestamp, text) 小根堆，自动保留时间戳最大的 tweet_cap 条
    heaps: list[list[tuple[float, str]]] = [[] for _ in range(n_users)]
    counts = np.zeros(n_users, dtype=np.int64)

    tweet_files = sorted(raw_dir.glob("tweet_*.json"))
    if not tweet_files:
        raise FileNotFoundError(f"在 {raw_dir} 下未找到 tweet_*.json")

    for tf in tweet_files:
        with tf.open("r", encoding="utf-8") as f:
            for tw in tqdm(ijson.items(f, "item", use_float=True),
                           desc=f"scan {tf.name}", unit="tweet"):
                author = tw.get("author_id")
                if author is None:
                    continue
                buid = bare_uid(author)
                idx = uid_to_idx.get(buid)
                if idx is None:
                    continue

                text = _clean_text(tw.get("text") or tw.get("full_text"))
                if not text:
                    continue
                ts = parse_created_at(tw.get("created_at"))
                if ts is None:
                    continue

                counts[idx] += 1
                ts_f = float(ts.timestamp())
                heapq.heappush(heaps[idx], (ts_f, text))
                if len(heaps[idx]) > tweet_cap:
                    heapq.heappop(heaps[idx])

    with out_path.open("w", encoding="utf-8") as out:
        for idx in tqdm(range(n_users), desc="dump tweets", unit="user"):
            h = sorted(heaps[idx], key=lambda x: x[0])
            kept = h[-tweet_cap:] if len(h) > tweet_cap else h
            tweets = [t for _, t in kept]
            timestamps = [ts for ts, _ in kept]
            out.write(json.dumps({
                "user_id": node_ids[idx],
                "split": splits[idx],
                "label": labels[idx],
                "description": descriptions[idx],
                "tweets": tweets,
                "timestamps": timestamps,
                "n_tweets_total": int(counts[idx]),
            }, ensure_ascii=False) + "\n")

    return n_users
