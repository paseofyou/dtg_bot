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


def _load_encoder(model_name: str, device: str):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name).to(device).eval()
    if device.startswith("cuda"):
        model = model.half()
    return tokenizer, model


@torch.no_grad()
def _encode_texts(texts: list[str], tokenizer, model, device: str,
                  max_tokens: int, batch_size: int) -> np.ndarray:
    """把一批文本编码成 (len(texts), 768) fp16。按 token 长度分桶以减少 padding 浪费。"""
    out = np.empty((len(texts), EMB_DIM), dtype=np.float16)
    order = np.argsort([len(t) for t in texts])
    for start in range(0, len(order), batch_size):
        sel = order[start : start + batch_size]
        enc = tokenizer(
            [texts[i] for i in sel], padding=True, truncation=True,
            max_length=max_tokens, return_tensors="pt",
        ).to(device)
        hidden = model(**enc).last_hidden_state
        m = enc["attention_mask"].unsqueeze(-1).to(hidden.dtype)
        pooled = (hidden * m).sum(1) / m.sum(1).clamp(min=1e-6)
        out[sel] = pooled.float().cpu().numpy().astype(np.float16)
    return out


class _Progress:
    """断点续跑：记录已完成的行数。云端长任务被打断后可从中途继续。"""

    def __init__(self, path: Path, resume: bool):
        self.path = path
        self.rows_done = 0
        if resume and path.exists():
            self.rows_done = int(json.loads(path.read_text())["rows_done"])

    def update(self, rows_done: int) -> None:
        self.rows_done = rows_done
        self.path.write_text(json.dumps({"rows_done": rows_done}), encoding="utf-8")


def _open_memmap(path: Path, shape: tuple, resume_rows: int) -> np.memmap:
    mode = "r+" if (resume_rows > 0 and path.exists()) else "w+"
    return np.memmap(path, dtype=np.float16, mode=mode, shape=shape)


@torch.no_grad()
def encode_pooled(
    jsonl_path: str | Path,
    out_dir: str | Path,
    pool_cap: int = 20,
    model_name: str = "roberta-base",
    max_tokens: int = 64,
    batch_size: int = 256,
    users_per_batch: int = 512,
    device: str | None = None,
    resume: bool = True,
    out_name: str = "tweet_pooled",
) -> dict:
    """对每个用户的推文编码后**取均值**，只保存池化向量。

    这是 BotRGCN 基线所需的节点文本特征。**不保存逐条嵌入** ——
    TwiBot-20 全节点若保存逐条嵌入需约 49GB，而池化后仅 0.35GB。

    Args:
        pool_cap: 每用户参与池化的推文条数上限（取最近的）。0 表示不设上限。
            这是唯一偏离官方 BotRGCN 实现之处，需用基线分数验证其可接受性。
    """
    jsonl_path, out_dir = Path(jsonl_path), Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    n_users = count_lines(jsonl_path)

    prog = _Progress(out_dir / f"{out_name}_progress.json", resume)
    emb = _open_memmap(out_dir / f"{out_name}.memmap", (n_users, EMB_DIM), prog.rows_done)
    n_tweets_path = out_dir / f"{out_name}_n_tweets.npy"
    n_tweets = (np.load(n_tweets_path) if prog.rows_done and n_tweets_path.exists()
                else np.zeros(n_users, dtype=np.int32))

    tokenizer, model = _load_encoder(model_name, device)
    row = prog.rows_done
    if row:
        print(f"  [resume] 从第 {row}/{n_users} 个用户继续")

    pbar = tqdm(total=n_users, initial=row, desc=f"pool {out_name}", unit="user")
    with jsonl_path.open("r", encoding="utf-8") as fh:
        for _ in range(row):                      # 跳过已完成的行
            fh.readline()
        batch: list[dict] = []
        for line in fh:
            batch.append(json.loads(line))
            if len(batch) < users_per_batch:
                continue
            row = _flush_pooled(batch, row, emb, n_tweets, pool_cap,
                                tokenizer, model, device, max_tokens, batch_size)
            prog.update(row)
            np.save(n_tweets_path, n_tweets)
            pbar.update(len(batch))
            batch = []
        if batch:
            row = _flush_pooled(batch, row, emb, n_tweets, pool_cap,
                                tokenizer, model, device, max_tokens, batch_size)
            prog.update(row)
            np.save(n_tweets_path, n_tweets)
            pbar.update(len(batch))
    pbar.close()
    emb.flush()

    meta = {
        "n_users": n_users, "emb_dim": EMB_DIM, "pool_cap": pool_cap,
        "model_name": model_name, "max_tokens": max_tokens, "dtype": "float16",
        "n_users_with_no_tweet": int((n_tweets == 0).sum()),
        "mean_tweets_pooled": float(n_tweets.mean()),
    }
    (out_dir / f"{out_name}_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return meta


def _flush_pooled(batch, row, emb, n_tweets, pool_cap, tokenizer, model,
                  device, max_tokens, batch_size) -> int:
    texts: list[str] = []
    owner: list[int] = []
    for i, user in enumerate(batch):
        tws = user.get("tweets") or []
        if pool_cap:
            tws = tws[-pool_cap:]
        for t in tws:
            texts.append(t)
            owner.append(i)
    if texts:
        vecs = _encode_texts(texts, tokenizer, model, device, max_tokens, batch_size)
        acc = np.zeros((len(batch), EMB_DIM), dtype=np.float32)
        cnt = np.zeros(len(batch), dtype=np.int32)
        np.add.at(acc, owner, vecs.astype(np.float32))
        np.add.at(cnt, owner, 1)
        acc /= np.clip(cnt, 1, None)[:, None]
        emb[row : row + len(batch)] = acc.astype(np.float16)
        n_tweets[row : row + len(batch)] = cnt
    else:
        emb[row : row + len(batch)] = 0
    return row + len(batch)


@torch.no_grad()
def encode_field(
    jsonl_path: str | Path,
    out_dir: str | Path,
    field: str = "description",
    model_name: str = "roberta-base",
    max_tokens: int = 64,
    batch_size: int = 256,
    users_per_batch: int = 4096,
    device: str | None = None,
    resume: bool = True,
) -> dict:
    """编码每个用户的单条文本字段（如 description），产出 (N, 768)。"""
    jsonl_path, out_dir = Path(jsonl_path), Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    n_users = count_lines(jsonl_path)

    prog = _Progress(out_dir / f"{field}_progress.json", resume)
    emb = _open_memmap(out_dir / f"{field}.memmap", (n_users, EMB_DIM), prog.rows_done)
    tokenizer, model = _load_encoder(model_name, device)

    row = prog.rows_done
    if row:
        print(f"  [resume] {field} 从第 {row}/{n_users} 继续")
    pbar = tqdm(total=n_users, initial=row, desc=f"encode {field}", unit="user")
    with jsonl_path.open("r", encoding="utf-8") as fh:
        for _ in range(row):
            fh.readline()
        batch: list[str] = []
        for line in fh:
            batch.append(json.loads(line).get(field) or "")
            if len(batch) >= users_per_batch:
                emb[row : row + len(batch)] = _encode_texts(
                    batch, tokenizer, model, device, max_tokens, batch_size)
                row += len(batch)
                prog.update(row)
                pbar.update(len(batch))
                batch = []
        if batch:
            emb[row : row + len(batch)] = _encode_texts(
                batch, tokenizer, model, device, max_tokens, batch_size)
            row += len(batch)
            prog.update(row)
            pbar.update(len(batch))
    pbar.close()
    emb.flush()

    meta = {"n_users": n_users, "emb_dim": EMB_DIM, "field": field,
            "model_name": model_name, "dtype": "float16"}
    (out_dir / f"{field}_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return meta


def load_matrix(out_dir: str | Path, name: str) -> np.memmap:
    """读回 (N, 768) 的 fp16 memmap。"""
    out_dir = Path(out_dir)
    meta = json.loads((out_dir / f"{name}_meta.json").read_text(encoding="utf-8"))
    return np.memmap(out_dir / f"{name}.memmap", dtype=np.float16, mode="r",
                     shape=(meta["n_users"], meta["emb_dim"]))


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
