"""微观分支消融：事件通道组与池化方式。

用法：
    # 全部组合（4 通道组 × 3 池化 × 5 种子）
    python scripts/ablate_micro.py --cache ./cache/twibot20 --seq-len 32 --seeds 42 123 456 789 2024

    # 单独一组
    python scripts/ablate_micro.py --cache ./cache/twibot20 --seq-len 32 --channel-group event+transition --pool mean
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from diagnose_order import load_data, run_one  # noqa: E402
from dtg_bot.utils.metrics import append_result, summarize  # noqa: E402

#: 通道组名称 → 所包含的微观特征通道名
#: 对应 data/micro.py 中的 CHANNEL_NAMES；with_temporal 时以 base 通道为准
CHANNEL_GROUPS = {
    "event_only": [
        "log_len", "is_retweet", "n_hashtag", "n_mention", "n_url",
        "upper_ratio", "digit_ratio",
    ],
    "event+transition": [
        "log_len", "is_retweet", "n_hashtag", "n_mention", "n_url",
        "upper_ratio", "digit_ratio",
        "has_prev", "cos_prev", "l2_prev", "jaccard_prev", "rt_run",
    ],
    "event+context": [
        "log_len", "is_retweet", "n_hashtag", "n_mention", "n_url",
        "upper_ratio", "digit_ratio",
        "cos_centroid", "self_repeat_max",
    ],
}

POOLS = ["attn", "mean", "max"]


def build_groups(base_names: list[str]) -> dict[str, list[int]]:
    """把通道组定义转换成当前 meta 中的索引。"""
    groups = {}
    for key, want in {
        "event_only": CHANNEL_GROUPS["event_only"],
        "event+transition": CHANNEL_GROUPS["event+transition"],
        "event+context": CHANNEL_GROUPS["event+context"],
    }.items():
        groups[key] = [base_names.index(c) for c in want if c in base_names]
    groups["full"] = list(range(len(base_names)))
    return groups


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default=None)
    ap.add_argument("--dataset", default="TwiBot-20")
    ap.add_argument("--seq-len", type=int, default=32)
    ap.add_argument("--channel-group", default=None)
    ap.add_argument("--pool", default=None, choices=POOLS)
    ap.add_argument("--seeds", nargs="+", type=int,
                    default=[42, 123, 456, 789, 2024])
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--patience", type=int, default=20)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight-decay", type=float, default=0.01)
    ap.add_argument("--d-model", type=int, default=64)
    ap.add_argument("--layers", type=int, default=2)
    ap.add_argument("--heads", type=int, default=4)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--results", default="./experiments/results_ablate_micro.csv")
    args = ap.parse_args()
    if args.cache is None:
        args.cache = f"./cache/twibot{args.dataset.split('-')[1].lower()}"

    feat, mask, labels, idx, meta = load_data(Path(args.cache), args.seq_len)
    base_names = meta["channel_names"]
    groups = build_groups(base_names)

    if args.channel_group:
        if args.channel_group not in groups:
            raise ValueError(f"--channel-group 必须是 {list(groups.keys())} 之一")
        group_keys = [args.channel_group]
    else:
        group_keys = list(groups.keys())

    pools = [args.pool] if args.pool else POOLS

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"base channels: {len(base_names)}  {base_names}")

    for gk in group_keys:
        sel = groups[gk]
        if not sel:
            print(f"跳过 {gk}：当前缓存不含其所需通道")
            continue
        feat_g = feat[..., sel]
        meta_g = dict(meta)
        meta_g["n_channels"] = len(sel)
        meta_g["channel_names"] = [base_names[i] for i in sel]

        for pool in pools:
            variant_name = f"{gk}_{pool}"
            variant = {"seq_model": "transformer", "order_mode": "keep", "pool": pool}
            rows = []
            for seed in args.seeds:
                res = run_one(feat_g, mask, labels, idx, meta_g, variant, seed, args, device)
                res["seed"] = seed
                rows.append(res)
                append_result(args.results, {
                    "dataset": args.dataset, "experiment": "ablate_micro",
                    "variant": variant_name, "channel_group": gk, "pool": pool,
                    "seed": seed, "seq_len": args.seq_len, **res,
                })
                print(f"  {variant_name:<22} seed={seed:<5} "
                      f"f1={res['f1']:.4f} acc={res['accuracy']:.4f} auc={res['auc']:.4f}")
            print(f"  {variant_name:<22} → {summarize(rows)}\n")


if __name__ == "__main__":
    main()
