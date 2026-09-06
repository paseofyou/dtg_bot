"""微观「事件驱动序列」构建 —— 本文方法的核心。

设计动机
--------
上一版方法的时序特征是**按时间窗聚合的累计计数**（粉丝数曲线、累计发帖量等）。
这类量的整条曲线由静态元数据单调决定，信息上等价于静态特征，
因此 ``flat_static`` 对照一跑就打平，时序分支等于无效。

本模块换一个建模对象：不刻画「某时刻用户处于什么状态」，
而刻画**「相邻两次发帖事件之间发生了什么变化」**。
转移量（语义跳变、模板重叠、RT 连段）无法从单一静态快照恢复，
与静态元数据天然正交，这是时序分支能带来增量信息的前提。

通道分三组：
  1. 逐条通道   —— 单条推文自身的表层形态（长度、hashtag/mention/URL 密度、大小写）
  2. 转移通道   —— 与**前一条**推文的关系（语义相似度、位移、词级重叠、RT 连段长度）
                   这组是"内容拼接导致的短期语义矛盾"的直接观测量
  3. 上下文通道 —— 与用户整体语义中心的偏离、对历史推文的最大自重复度
                   这组是"刻板规律 / 模板化复读"的直接观测量

序列长度保持 L（而非 L-1）：需要前驱的通道在 i=0 处置 0，并由 ``has_prev`` 通道标记。

归一化只用 **train split** 统计量，避免验证/测试集信息泄漏。
"""

from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from tqdm import tqdm

from .encode import load_encoded

CHANNEL_NAMES = (
    # --- 逐条通道 ---
    "log_len",          # log1p(字符数)
    "is_retweet",       # 是否以 "RT @" 开头
    "n_hashtag",        # #tag 数
    "n_mention",        # @user 数
    "n_url",            # http(s) 链接数
    "upper_ratio",      # 大写字母占字母数比例（机器人常见全大写模板）
    "digit_ratio",      # 数字占字符数比例
    # --- 转移通道（需要前驱）---
    "has_prev",         # i>0 且前驱有效
    "cos_prev",         # cos(e_i, e_{i-1})    低值 = 相邻语义生硬跳变
    "l2_prev",          # ||e_i - e_{i-1}||_2  语义位移幅度
    "jaccard_prev",     # 与前一条的词级 Jaccard 重叠，模板化复读的直接证据
    "rt_run",           # 当前连续 RT 段长度（归一化）
    # --- 上下文通道 ---
    "cos_centroid",     # cos(e_i, 用户语义中心)  —— 中心正是 BotRGCN 用的池化向量
    "self_repeat_max",  # max_{j<i} cos(e_i, e_j)  对历史内容的最大自重复度
)

#: 仅在有**真实推文时间戳**时可用（TwiBot-22）。TwiBot-20 无时间戳，不可用。
#: 这组通道是 order-vs-Δt 实验的关键：用来回答"真实发帖时刻是否比
#: 顺序无关的事件内容特征提供额外信息"。
TEMPORAL_CHANNEL_NAMES = (
    "dt_prev",    # log1p(与前一条的间隔秒数)，自动化脚本的间隔分布异于人类
    "hour_sin",   # 发帖时刻的昼夜相位（圆周编码，避免 0 点/23 点被当成远距离）
    "hour_cos",
    "dow",        # 星期（归一化到 [0,1]），机器人常缺少工作日/周末节律
)

N_CHANNELS = len(CHANNEL_NAMES)
N_TEMPORAL_CHANNELS = len(TEMPORAL_CHANNEL_NAMES)


def channel_names(with_temporal: bool = False) -> tuple[str, ...]:
    return CHANNEL_NAMES + TEMPORAL_CHANNEL_NAMES if with_temporal else CHANNEL_NAMES

_RE_HASHTAG = re.compile(r"#\w+")
_RE_MENTION = re.compile(r"@\w+")
_RE_URL = re.compile(r"https?://\S+")
_RE_WORD = re.compile(r"[a-z0-9']+")
_RE_RT = re.compile(r"^\s*RT\s+@")


def _text_channels(text: str) -> tuple[float, float, float, float, float, float, float]:
    letters = [c for c in text if c.isalpha()]
    upper_ratio = sum(c.isupper() for c in letters) / len(letters) if letters else 0.0
    digit_ratio = sum(c.isdigit() for c in text) / len(text) if text else 0.0
    return (
        float(np.log1p(len(text))),
        1.0 if _RE_RT.match(text) else 0.0,
        float(len(_RE_HASHTAG.findall(text))),
        float(len(_RE_MENTION.findall(text))),
        float(len(_RE_URL.findall(text))),
        float(upper_ratio),
        float(digit_ratio),
    )


def _word_set(text: str) -> set[str]:
    """去掉 RT 前缀、@提及和链接后的词集合，避免 Jaccard 被固定模板噪声抬高。"""
    cleaned = _RE_URL.sub(" ", _RE_MENTION.sub(" ", text))
    return set(_RE_WORD.findall(cleaned.lower()))


def _jaccard(a: set[str], b: set[str]) -> float:
    if not a or not b:
        return 0.0
    inter = len(a & b)
    return inter / (len(a) + len(b) - inter)


def build_user_micro_sequence(
    tweets: list[str],
    embeddings: np.ndarray,
    valid: np.ndarray,
    timestamps: list[float] | None = None,
) -> np.ndarray:
    """构建单用户的事件驱动序列。

    Args:
        tweets: 时间正序（旧 → 新）的推文原文，长度 = valid.sum()
        embeddings: (L, 768) 该用户的推文嵌入，padding 位为 0
        valid: (L,) bool，True 表示真实推文
        timestamps: 可选，与 tweets 对应的 epoch 秒（仅 TwiBot-22 有）。
            提供时额外产出 TEMPORAL_CHANNEL_NAMES 四个通道。

    Returns:
        (L, C) float32，padding 位为 0；C = 14 或 18
    """
    seq_len = embeddings.shape[0]
    n_ch = N_CHANNELS + (N_TEMPORAL_CHANNELS if timestamps is not None else 0)
    feat = np.zeros((seq_len, n_ch), dtype=np.float32)
    n_valid = int(valid.sum())
    if n_valid == 0:
        return feat

    # 有效位一定是前 n_valid 个（encode.py 从 pos=0 开始填）
    emb = embeddings[:n_valid].astype(np.float32)
    norms = np.linalg.norm(emb, axis=1, keepdims=True)
    unit = emb / np.clip(norms, 1e-8, None)
    centroid = unit.mean(axis=0)
    centroid /= max(float(np.linalg.norm(centroid)), 1e-8)

    # 一次算出全部两两余弦，L 很小（<=64），开销可忽略
    cos_matrix = unit @ unit.T

    word_sets = [_word_set(t) for t in tweets[:n_valid]]
    rt_flags = [bool(_RE_RT.match(t)) for t in tweets[:n_valid]]

    rt_run = 0
    for i in range(n_valid):
        text = tweets[i]
        (log_len, is_rt, n_hash, n_men, n_url, upper_r, digit_r) = _text_channels(text)

        rt_run = rt_run + 1 if rt_flags[i] else 0

        if i > 0:
            has_prev = 1.0
            cos_prev = float(cos_matrix[i, i - 1])
            l2_prev = float(np.linalg.norm(unit[i] - unit[i - 1]))
            jac_prev = _jaccard(word_sets[i], word_sets[i - 1])
            self_repeat = float(cos_matrix[i, :i].max())
        else:
            has_prev = cos_prev = l2_prev = jac_prev = self_repeat = 0.0

        feat[i, :N_CHANNELS] = (
            log_len, is_rt, n_hash, n_men, n_url, upper_r, digit_r,
            has_prev, cos_prev, l2_prev, jac_prev,
            float(rt_run) / 10.0,
            float(unit[i] @ centroid),
            self_repeat,
        )

        if timestamps is not None:
            ts = float(timestamps[i])
            dt_prev = float(np.log1p(max(ts - float(timestamps[i - 1]), 0.0))) if i > 0 else 0.0
            # UTC 小时与星期，用圆周编码避免 23 点与 0 点被当成远距离
            dt_obj = datetime.fromtimestamp(ts, tz=timezone.utc)
            hour_frac = (dt_obj.hour + dt_obj.minute / 60.0) / 24.0
            feat[i, N_CHANNELS:] = (
                dt_prev,
                float(np.sin(2 * np.pi * hour_frac)),
                float(np.cos(2 * np.pi * hour_frac)),
                dt_obj.weekday() / 6.0,
            )
    return feat


def build_micro_features(
    jsonl_path: str | Path,
    encoded_dir: str | Path,
    out_dir: str | Path,
    fit_split: str = "train",
    with_temporal: bool = False,
) -> dict:
    """为所有用户构建事件驱动序列并归一化。

    Args:
        with_temporal: 是否附加真实时间通道。**要求 jsonl 每行含 ``timestamps``**
            （仅 TwiBot-22 满足）。这是 order-vs-Δt 实验的开关。

    产物：
        micro_feat.npy   float32 (N, L, C)
        micro_mask.npy   bool    (N, L)
        micro_meta.json  含通道名与 scaler
    """
    jsonl_path, out_dir = Path(jsonl_path), Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    enc = load_encoded(encoded_dir)
    emb, mask, splits = enc["emb"], enc["mask"], enc["splits"]
    n_users, seq_len = mask.shape

    names = channel_names(with_temporal)
    feats = np.zeros((n_users, seq_len, len(names)), dtype=np.float32)

    with Path(jsonl_path).open("r", encoding="utf-8") as fh:
        for idx, line in enumerate(tqdm(fh, total=n_users, desc="micro seq", unit="user")):
            user = json.loads(line)
            ts = None
            if with_temporal:
                ts = user.get("timestamps")
                if ts is None:
                    raise KeyError(
                        f"with_temporal=True 但第 {idx} 行缺少 'timestamps' 字段；"
                        "TwiBot-20 无推文时间戳，不能开启该选项"
                    )
                ts = ts[-seq_len:]
            feats[idx] = build_user_micro_sequence(
                user["tweets"][-seq_len:], np.asarray(emb[idx]), mask[idx], ts
            )

    # --- 归一化：只用 fit_split 的有效位统计 ---
    fit_rows = splits == fit_split
    fit_valid = feats[fit_rows][mask[fit_rows]]          # (n_valid_positions, C)
    mean = fit_valid.mean(axis=0)
    std = fit_valid.std(axis=0)
    std[std < 1e-6] = 1.0
    # 二值 / 已在 [0,1] 或 [-1,1] 的通道不做标准化，保持可解释性
    raw_channels = ("is_retweet", "has_prev", "upper_ratio", "digit_ratio",
                    "hour_sin", "hour_cos", "dow")
    keep_raw = [names.index(c) for c in raw_channels if c in names]
    mean[keep_raw] = 0.0
    std[keep_raw] = 1.0

    feats = (feats - mean) / std
    feats[~mask] = 0.0                                   # padding 位归零

    np.save(out_dir / "micro_feat.npy", feats)
    np.save(out_dir / "micro_mask.npy", mask)
    meta = {
        "channel_names": list(names),
        "n_channels": len(names),
        "with_temporal": with_temporal,
        "seq_len": int(seq_len),
        "n_users": int(n_users),
        "fit_split": fit_split,
        "mean": mean.tolist(),
        "std": std.tolist(),
    }
    (out_dir / "micro_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return meta
