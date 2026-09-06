"""TwiBot-22 预处理（云端运行）。

分三步，每步产物落盘可单独重跑：
    1. collect  单遍扫描 tweet_0..8.json（约 101GB），每作者取最近 L 条 + 真实时间戳
    2. encode   RoBERTa 逐条编码
    3. micro    事件转移序列，可选附加真实时间通道（order-vs-Δt 实验用）

用法（AutoDL）：
    WORK=/root/autodl-tmp/dtg_bot
    python scripts/prepare_twibot22.py --data $WORK/data/twibot22 --cache $WORK/cache/twibot22 \
        --seq-len 16 --batch-size 384 --steps collect encode micro

    # 只跑 collect 先看顺序违例率（这一步就能验证"列表顺序=真实时序"假设）：
    python scripts/prepare_twibot22.py --data ... --cache ... --steps collect

    # 本地小规模冒烟测试（只扫第一个 tweet 文件、只留 2000 个用户）：
    python scripts/prepare_twibot22.py --data D:/dataset/raw_data/TwiBot_22 \
        --cache ./cache/twibot22_smoke --tweet-files tweet_0.json --max-users 2000
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from dtg_bot.data.encode import encode_tweets
from dtg_bot.data.micro import build_micro_features
from dtg_bot.data.twibot22 import (
    TWEET_FILES,
    collect_recent_tweets,
    load_labels,
    load_splits,
)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="含 user.json/label.csv/split.csv/tweet_*.json")
    ap.add_argument("--cache", required=True)
    ap.add_argument("--seq-len", type=int, default=16)
    ap.add_argument("--model", default="roberta-base")
    ap.add_argument("--batch-size", type=int, default=384)
    ap.add_argument("--max-tokens", type=int, default=64)
    ap.add_argument("--tweet-files", nargs="+", default=list(TWEET_FILES))
    ap.add_argument("--max-users", type=int, default=0, help=">0 时只保留前 N 个标注用户（冒烟测试）")
    ap.add_argument("--steps", nargs="+", default=["collect", "encode", "micro"],
                    choices=["collect", "encode", "micro"])
    ap.add_argument("--with-temporal", action="store_true",
                    help="micro 步骤附加真实时间通道（order-vs-Δt 实验）")
    args = ap.parse_args()

    cache = Path(args.cache)
    cache.mkdir(parents=True, exist_ok=True)
    tag = f"L{args.seq_len}"
    raw_jsonl = cache / f"tweets_{tag}.jsonl"
    enc_dir = cache / f"encoded_{tag}"
    micro_dir = cache / f"micro_{tag}" / ("with_dt" if args.with_temporal else "base")

    if "collect" in args.steps:
        labels = load_labels(args.data)
        splits = load_splits(args.data)
        keep = set(labels)
        if args.max_users:
            keep = set(sorted(keep)[: args.max_users])
        print(f"[collect] 标注用户 {len(labels)}，本次收集 {len(keep)}；"
              f"扫描 {len(args.tweet_files)} 个 tweet 文件")

        t0 = time.time()
        meta = collect_recent_tweets(
            args.data, raw_jsonl, seq_len=args.seq_len,
            keep_users=keep, tweet_files=tuple(args.tweet_files),
        )
        print(f"[collect] 完成 ({time.time() - t0:.0f}s)")
        print("          " + json.dumps(meta, indent=2))
        print(f"\n  >>> 顺序假设检验：时间戳单调性违例率 = {meta['order_violation_rate']:.4%}"
              f"，有违例的作者占比 = {meta['frac_authors_with_violation']:.4%}")
        print("      违例率接近 0 → dump 确实按时间倒序，"
              "TwiBot-20 上的顺序空结果才能解释为'顺序本身无信息'\n")

        # 补上 label / split，对齐 encode.py 期望的 jsonl 格式
        tmp = raw_jsonl.with_suffix(".tmp")
        n_kept = 0
        with raw_jsonl.open("r", encoding="utf-8") as fin, tmp.open("w", encoding="utf-8") as fout:
            for line in fin:
                rec = json.loads(line)
                uid = rec["user_id"]
                # TwiBot-22 的 split 值是 train/val/test，统一成本项目的 train/dev/test
                split = splits.get(uid, "train")
                rec["split"] = {"val": "dev", "valid": "dev"}.get(split, split)
                rec["label"] = labels.get(uid)
                fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
                n_kept += 1
        tmp.replace(raw_jsonl)
        print(f"[collect] 已写入 label/split，共 {n_kept} 行 → {raw_jsonl}")

    if "encode" in args.steps:
        t0 = time.time()
        meta = encode_tweets(
            raw_jsonl, enc_dir, seq_len=args.seq_len, model_name=args.model,
            max_tokens=args.max_tokens, batch_size=args.batch_size,
        )
        print(f"[encode] 完成 ({time.time() - t0:.0f}s) {json.dumps(meta, ensure_ascii=False)}")

    if "micro" in args.steps:
        t0 = time.time()
        meta = build_micro_features(
            raw_jsonl, enc_dir, micro_dir, fit_split="train",
            with_temporal=args.with_temporal,
        )
        print(f"[micro] 完成 ({time.time() - t0:.0f}s) → {micro_dir}")
        print(f"        通道 ({meta['n_channels']}): {meta['channel_names']}")


if __name__ == "__main__":
    main()
