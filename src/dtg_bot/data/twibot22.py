"""TwiBot-22 原始数据解析。

数据事实（已核验，详见 AGENTS.md）：

- `tweet_0..8.json` 共约 101GB，推文对象含**真实 `created_at`**
  （``"2022-02-27 04:59:35+00:00"``）、`author_id`、`source`、`entities`。
- 同一 `author_id` 的推文在 dump 中按时间**倒序**排列（已抽样验证）。

关键实现技巧
------------
既然 dump 已按时间倒序，只需**每个作者取遇到的前 L 条**即为最近 L 条，
无须缓存该作者的全部推文。这把 101GB 单遍扫描的内存占用压到 O(用户数 × L)。

同时本模块会统计每个作者的推文时间戳是否真的单调递减，
产出 ``order_violation_rate`` —— 这正是「列表顺序 = 真实时序」这一假设的
**直接实证检验**，也是分辨 TwiBot-20 上 `keep≈shuffle` 空结果两种解释的依据：
    若 TwiBot-22 上顺序可靠而仍然 shuffle 不掉点 → 顺序本身确实无信息
    若 TwiBot-20 的列表本就无序                  → 空结果不能否证原假设
"""

from __future__ import annotations

import csv
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator

import ijson
from tqdm import tqdm

# TwiBot-22 的采集截止时间（数据集论文所述采集期为 2022 年初）
CRAWL_DATE = datetime(2022, 4, 1, tzinfo=timezone.utc)

TWEET_FILES = tuple(f"tweet_{i}.json" for i in range(9))

NUM_PROPERTY_FIELDS = ("followers_count", "following_count", "tweet_count", "listed_count")
CAT_PROPERTY_FIELDS = ("protected", "verified")


def _clean(value) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return None if text in ("", "None", "null") else text


def parse_ts(value) -> datetime | None:
    """解析 ``"2022-02-27 04:59:35+00:00"`` 或 ISO8601 变体。"""
    text = _clean(value)
    if text is None:
        return None
    try:
        return datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError:
        return None


def load_splits(data_dir: str | Path) -> dict[str, str]:
    """split.csv → {user_id: split}。TwiBot-22 的 split 值为 train/val/test。"""
    mapping: dict[str, str] = {}
    with (Path(data_dir) / "split.csv").open("r", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            mapping[row["id"].strip()] = row["split"].strip()
    return mapping


def load_labels(data_dir: str | Path) -> dict[str, int]:
    """label.csv → {user_id: 0/1}，``bot`` → 1，``human`` → 0。"""
    mapping: dict[str, int] = {}
    with (Path(data_dir) / "label.csv").open("r", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            mapping[row["id"].strip()] = 1 if row["label"].strip().lower() == "bot" else 0
    return mapping


def iter_users(data_dir: str | Path) -> Iterator[dict]:
    """流式遍历 user.json（约 746MB）。"""
    with (Path(data_dir) / "user.json").open("rb") as fh:
        yield from ijson.items(fh, "item", use_float=True)


def user_static_features(user: dict) -> tuple[list[float], list[float], float | None]:
    metrics = user.get("public_metrics") or {}
    num = [float(metrics.get(f) or 0) for f in NUM_PROPERTY_FIELDS]
    created = parse_ts(user.get("created_at"))
    age_days = float((CRAWL_DATE - created).days) if created else 0.0
    username = _clean(user.get("username")) or ""
    num.extend([max(age_days, 0.0), float(len(username))])

    cat = [
        1.0 if str(user.get(f)).strip().lower() == "true" else 0.0
        for f in CAT_PROPERTY_FIELDS
    ]
    cat.append(1.0 if _clean(user.get("pinned_tweet_id")) else 0.0)
    return num, cat, created.timestamp() if created else None


def collect_recent_tweets(
    data_dir: str | Path,
    out_path: str | Path,
    seq_len: int = 16,
    keep_users: set[str] | None = None,
    tweet_files: tuple[str, ...] = TWEET_FILES,
    stats_only: bool = False,
) -> dict:
    """单遍扫描全部 tweet 文件，为每个作者收集最近 seq_len 条推文。

    利用 dump 已按时间倒序的性质：每作者只保留**遇到的前 seq_len 条**。
    同时统计时间戳单调性违例率，用于检验"列表顺序 = 真实时序"假设。

    Args:
        keep_users: 只收集这些作者（通常是有标签的用户集合），None 表示全收。
        stats_only: 只统计时间戳单调性，**不保存任何文本**。
            100 万作者 × 16 条推文的文本约占 4~5GB Python 对象，云端有 OOM 风险；
            而第一阶段只需要违例率，此模式内存降到约 130MB。

    产物：jsonl，每行
        {"user_id", "tweets": [时间正序文本...], "timestamps": [epoch 秒...],
         "sources": [...], "n_seen": 该作者被扫到的推文总数}
        stats_only=True 时不写 jsonl，仅写 meta。
    """
    data_dir, out_path = Path(data_dir), Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # 解析 tweet 文件真实路径：容忍子目录嵌套与目录/文件软链接。
    # 顶层找不到时按文件名在整个数据目录下递归查找一次（仅扫文件名，开销可忽略）。
    resolved: dict[str, Path] = {}
    for p in data_dir.rglob("tweet_*.json"):
        resolved.setdefault(p.name, p)
    if not resolved:
        raise FileNotFoundError(
            f"在 {data_dir} 及其子目录下未找到任何 tweet_*.json，"
            "请检查数据目录层级或软链接是否指向真实文件。"
        )
    resolved_dirs = sorted({str(p.parent) for p in resolved.values()})
    print(f"[collect] 解析到 {len(resolved)} 个 tweet 文件，所在目录: {resolved_dirs}")

    # author -> {"tw": [(ts, text, source)...], "n_seen": int, "violations": int, "last_ts": float}
    buffers: dict[str, dict] = {}
    total_tweets = 0
    missing_ts = 0

    for fname in tweet_files:
        path = resolved.get(fname, data_dir / fname)
        if not path.exists():
            print(f"  [warn] 缺少 {fname}，跳过")
            continue
        with path.open("rb") as fh:
            pbar = tqdm(ijson.items(fh, "item", use_float=True), desc=fname, unit="tw",
                        mininterval=5.0)
            for tw in pbar:
                author = _clean(tw.get("author_id"))
                if author is None:
                    continue
                if keep_users is not None and author not in keep_users:
                    continue
                total_tweets += 1

                ts_dt = parse_ts(tw.get("created_at"))
                if ts_dt is None:
                    missing_ts += 1
                    continue
                ts = ts_dt.timestamp()

                buf = buffers.get(author)
                if buf is None:
                    buf = buffers[author] = {"tw": [], "n_seen": 0, "violations": 0,
                                             "last_ts": None}
                buf["n_seen"] += 1
                # 单调性检验：dump 应为时间倒序，故 ts 应 <= 前一条
                if buf["last_ts"] is not None and ts > buf["last_ts"]:
                    buf["violations"] += 1
                buf["last_ts"] = ts

                if len(buf["tw"]) < seq_len:
                    # stats_only 下只留时间戳，不持有文本，避免百万级字符串驻留
                    buf["tw"].append(
                        (ts,) if stats_only
                        else (ts, _clean(tw.get("text")) or "", _clean(tw.get("source")) or "")
                    )

    n_users = 0
    n_viol_users = 0
    total_seen = 0
    total_viol = 0

    def _tally(buf: dict) -> None:
        nonlocal n_users, n_viol_users, total_seen, total_viol
        n_users += 1
        total_seen += buf["n_seen"]
        total_viol += buf["violations"]
        if buf["violations"]:
            n_viol_users += 1

    if stats_only:
        for buf in buffers.values():
            _tally(buf)
    else:
        with out_path.open("w", encoding="utf-8") as out:
            for author, buf in buffers.items():
                # dump 是倒序 → 按 ts 升序排成时间正序（同时对违例情形兜底）
                items = sorted(buf["tw"], key=lambda t: t[0])
                out.write(json.dumps({
                    "user_id": author,
                    "tweets": [t[1] for t in items],
                    "timestamps": [t[0] for t in items],
                    "sources": [t[2] for t in items],
                    "n_seen": buf["n_seen"],
                }, ensure_ascii=False) + "\n")
                _tally(buf)

    meta = {
        "stats_only": stats_only,
        "n_authors": n_users,
        "n_tweets_scanned": total_tweets,
        "n_tweets_missing_ts": missing_ts,
        "seq_len": seq_len,
        "mean_tweets_per_author": total_seen / max(n_users, 1),
        # --- 顺序假设的实证检验结果 ---
        # 分母是"相邻推文对"的总数（每作者 n_seen-1 对）
        "n_adjacent_pairs": total_seen - n_users,
        "n_order_violations": total_viol,
        "order_violation_rate": total_viol / max(total_seen - n_users, 1),
        "frac_authors_with_violation": n_viol_users / max(n_users, 1),
    }
    (out_path.parent / f"{out_path.stem}_meta.json").write_text(
        json.dumps(meta, indent=2), encoding="utf-8"
    )
    return meta
