"""TwiBot-20 原始数据解析。

数据事实（已核验，详见 AGENTS.md）：

- `{train,dev,test,support}.json` 是 JSON 数组，support.json 约 5GB，必须流式解析。
- `profile` 内所有字段值**尾部带多余空格**，且缺失值是字符串 ``"None"``，必须清洗。
- `tweet` 是**无时间戳**的推文原文列表，顺序为 Twitter timeline 倒序（新 → 旧）。
  本模块统一 reverse 成**时间正序（旧 → 新）**再截取，这是微观事件驱动序列的输入。
- `neighbor` 可能为 ``null``（无关系信息的用户）。
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator

import ijson

SPLITS = ("train", "dev", "test", "support")
LABELED_SPLITS = ("train", "dev", "test")

# TwiBot-20 的采集截止时间，用于把 created_at 换算成 active_days。
# 固定常量而非 datetime.now()，否则特征会随运行日期漂移、破坏可复现性。
CRAWL_DATE = datetime(2020, 9, 28, tzinfo=timezone.utc)

_PROFILE_DATE_FMT = "%a %b %d %H:%M:%S %z %Y"

NUM_PROPERTY_FIELDS = (
    "followers_count",
    "friends_count",
    "statuses_count",
    "favourites_count",
    "listed_count",
)
CAT_PROPERTY_FIELDS = ("protected", "verified", "default_profile")


def _clean(value) -> str | None:
    """去掉尾部空格，并把字符串 "None" / 空串统一成 None。"""
    if value is None:
        return None
    text = str(value).strip()
    if text in ("", "None", "null"):
        return None
    return text


def _to_int(value, default: int = 0) -> int:
    text = _clean(value)
    if text is None:
        return default
    try:
        return int(float(text))
    except ValueError:
        return default


def _to_bool(value) -> float:
    text = _clean(value)
    if text is None:
        return 0.0
    return 1.0 if text.lower() in ("true", "1") else 0.0


def parse_created_at(value) -> datetime | None:
    """解析 profile.created_at，如 ``"Mon Oct 28 13:39:41 +0000 2019 "``。"""
    text = _clean(value)
    if text is None:
        return None
    try:
        return datetime.strptime(text, _PROFILE_DATE_FMT)
    except ValueError:
        return None


@dataclass
class UserRecord:
    user_id: str
    split: str
    label: int | None
    created_at: datetime | None
    num_properties: list[float]
    cat_properties: list[float]
    description: str
    # 时间正序（旧 → 新）的推文列表，已截断到 seq_len
    tweets: list[str]
    n_tweets_total: int
    following: list[str]
    followers: list[str]


def _static_properties(profile: dict) -> tuple[list[float], list[float]]:
    num = [float(_to_int(profile.get(f))) for f in NUM_PROPERTY_FIELDS]
    created = parse_created_at(profile.get("created_at"))
    active_days = float((CRAWL_DATE - created).days) if created else 0.0
    screen_name = _clean(profile.get("screen_name")) or ""
    num.extend([max(active_days, 0.0), float(len(screen_name))])

    cat = [_to_bool(profile.get(f)) for f in CAT_PROPERTY_FIELDS]
    return num, cat


def iter_split(raw_dir: str | Path, split: str, seq_len: int = 16) -> Iterator[UserRecord]:
    """流式遍历一个 split，内存占用与文件大小无关。

    Args:
        seq_len: 每用户保留的推文条数。取**时间正序后的最后 seq_len 条**，
            即最近 seq_len 条推文，按时间正序排列。
    """
    path = Path(raw_dir) / f"{split}.json"
    if not path.exists():
        raise FileNotFoundError(path)

    with path.open("rb") as fh:
        for entry in ijson.items(fh, "item", use_float=True):
            profile = entry.get("profile") or {}
            num, cat = _static_properties(profile)

            raw_tweets = entry.get("tweet") or []
            # 原始顺序为新 → 旧，reverse 成时间正序（旧 → 新）
            chronological = [t for t in reversed(raw_tweets) if _clean(t)]
            tweets = chronological[-seq_len:] if seq_len else chronological

            neighbor = entry.get("neighbor") or {}
            label = entry.get("label")

            yield UserRecord(
                user_id=str(entry["ID"]).strip(),
                split=split,
                label=int(label) if label is not None else None,
                created_at=parse_created_at(profile.get("created_at")),
                num_properties=num,
                cat_properties=cat,
                description=_clean(profile.get("description")) or "",
                tweets=tweets,
                n_tweets_total=len(chronological),
                following=[str(x).strip() for x in (neighbor.get("following") or [])],
                followers=[str(x).strip() for x in (neighbor.get("follower") or [])],
            )


def dump_ordered_tweets(
    raw_dir: str | Path,
    out_path: str | Path,
    splits: tuple[str, ...] = LABELED_SPLITS,
    seq_len: int = 16,
) -> int:
    """把有序推文导出成 jsonl，供 RoBERTa 编码阶段消费。

    每行: {"user_id", "split", "label", "tweets": [...时间正序...], "n_tweets_total"}
    返回写出的用户数。
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with out_path.open("w", encoding="utf-8") as out:
        for split in splits:
            for rec in iter_split(raw_dir, split, seq_len=seq_len):
                out.write(
                    json.dumps(
                        {
                            "user_id": rec.user_id,
                            "split": rec.split,
                            "label": rec.label,
                            "tweets": rec.tweets,
                            "n_tweets_total": rec.n_tweets_total,
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )
                n += 1
    return n
