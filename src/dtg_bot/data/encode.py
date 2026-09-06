"""逐条推文的 RoBERTa 编码。

与 BotRGCN 的关键区别：BotRGCN 把用户所有推文的嵌入**平均池化**成一个向量，
顺序信息在这一步就被彻底丢弃。本模块保留**每条推文各自的嵌入**，
形状 ``(N, L, 768)``，供 micro.py 计算相邻事件的转移量。

编码器冻结，结果落盘为 fp16 memmap，只需运行一次。
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer

EMB_DIM = 768


def _iter_user_batches(jsonl_path: Path, users_per_batch: int):
    batch = []
    with jsonl_path.open("r", encoding="utf-8") as fh:
        for line in fh:
            batch.append(json.loads(line))
            if len(batch) >= users_per_batch:
                yield batch
                batch = []
    if batch:
        yield batch


def count_lines(path: Path) -> int:
    with path.open("rb") as fh:
        return sum(1 for _ in fh)


@torch.no_grad()
def encode_tweets(
    jsonl_path: str | Path,
    out_dir: str | Path,
    seq_len: int = 32,
    model_name: str = "roberta-base",
    max_tokens: int = 64,
    batch_size: int = 128,
    users_per_batch: int = 256,
    device: str | None = None,
) -> dict:
    """编码 jsonl 中每个用户的有序推文。

    产物（out_dir 下）：
        tweet_emb.memmap  float16 (N, L, 768)  —— 时间正序，padding 位为 0
        tweet_mask.npy    bool    (N, L)       —— True 表示该位是真实推文
        user_ids.npy      str
        labels.npy        int8   (-1 表示无标签)
        splits.npy        str
        meta.json
    """
    jsonl_path = Path(jsonl_path)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    n_users = count_lines(jsonl_path)

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name).to(device).eval()
    if device.startswith("cuda"):
        model = model.half()

    emb_path = out_dir / "tweet_emb.memmap"
    emb = np.memmap(emb_path, dtype=np.float16, mode="w+", shape=(n_users, seq_len, EMB_DIM))
    mask = np.zeros((n_users, seq_len), dtype=bool)
    user_ids: list[str] = []
    labels: list[int] = []
    splits: list[str] = []

    row = 0
    pbar = tqdm(total=n_users, desc="encode tweets", unit="user")
    for users in _iter_user_batches(jsonl_path, users_per_batch):
        # 把这批用户的推文摊平成一个大 batch，避免小 batch 反复启动 GPU
        texts: list[str] = []
        slots: list[tuple[int, int]] = []  # (local_user_idx, position)
        for local_idx, user in enumerate(users):
            tweets = user["tweets"][-seq_len:]
            for pos, text in enumerate(tweets):
                texts.append(text)
                slots.append((local_idx, pos))

        if texts:
            vectors = np.empty((len(texts), EMB_DIM), dtype=np.float16)
            # 按 token 长度分桶：padding=True 只补到 chunk 内最长，
            # 长度相近的放一批可大幅减少 padding 浪费（TwiBot-22 规模下这是数小时的差别）。
            order = np.argsort([len(t) for t in texts])
            for start in range(0, len(order), batch_size):
                sel = order[start : start + batch_size]
                chunk = [texts[i] for i in sel]
                enc = tokenizer(
                    chunk,
                    padding=True,
                    truncation=True,
                    max_length=max_tokens,
                    return_tensors="pt",
                ).to(device)
                out = model(**enc).last_hidden_state  # (B, T, 768)
                # 按 attention_mask 做 mean pooling（不用 CLS：RoBERTa 未针对句向量训练 CLS）
                m = enc["attention_mask"].unsqueeze(-1).to(out.dtype)
                pooled = (out * m).sum(1) / m.sum(1).clamp(min=1e-6)
                vectors[sel] = pooled.float().cpu().numpy().astype(np.float16)

            # 先在内存里聚成连续块，再一次性写 memmap，避免逐元素随机写
            block = np.zeros((len(users), seq_len, EMB_DIM), dtype=np.float16)
            li = np.fromiter((s[0] for s in slots), dtype=np.int64, count=len(slots))
            pi = np.fromiter((s[1] for s in slots), dtype=np.int64, count=len(slots))
            block[li, pi] = vectors
            emb[row : row + len(users)] = block
            mask[row + li, pi] = True

        for user in users:
            user_ids.append(user["user_id"])
            labels.append(-1 if user["label"] is None else int(user["label"]))
            splits.append(user["split"])

        row += len(users)
        pbar.update(len(users))
    pbar.close()

    emb.flush()
    np.save(out_dir / "tweet_mask.npy", mask)
    np.save(out_dir / "user_ids.npy", np.array(user_ids))
    np.save(out_dir / "labels.npy", np.array(labels, dtype=np.int8))
    np.save(out_dir / "splits.npy", np.array(splits))

    meta = {
        "n_users": n_users,
        "seq_len": seq_len,
        "emb_dim": EMB_DIM,
        "model_name": model_name,
        "max_tokens": max_tokens,
        "dtype": "float16",
        "n_users_with_no_tweet": int((~mask.any(axis=1)).sum()),
        "mean_valid_tweets": float(mask.sum(axis=1).mean()),
    }
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return meta


def load_encoded(out_dir: str | Path) -> dict:
    """读回编码产物；嵌入以 memmap 惰性打开，不占内存。"""
    out_dir = Path(out_dir)
    meta = json.loads((out_dir / "meta.json").read_text(encoding="utf-8"))
    emb = np.memmap(
        out_dir / "tweet_emb.memmap",
        dtype=np.float16,
        mode="r",
        shape=(meta["n_users"], meta["seq_len"], meta["emb_dim"]),
    )
    return {
        "emb": emb,
        "mask": np.load(out_dir / "tweet_mask.npy"),
        "user_ids": np.load(out_dir / "user_ids.npy"),
        "labels": np.load(out_dir / "labels.npy"),
        "splits": np.load(out_dir / "splits.npy"),
        "meta": meta,
    }
