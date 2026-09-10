"""TwiBot-22 全量预处理（云端 GPU 运行）。

产物结构与 TwiBot-20 一致，供同一套 train.py / ablate*.py 使用：
    cache/twibot22/
      graph/          图结构 + 快照属性
      features/       description / tweet_pooled (BotRGCN 基线用)
      encoded_L{L}/   逐条推文嵌入 (N, L, 768)
      micro_L{L}/     微观事件序列

用法（AutoDL）:
    WORK=/root/autodl-tmp/dtg_bot
    python scripts/prepare_twibot22.py --work-dir $WORK --steps nodes graph desc pooled micro

建议先在小样本上冒烟测试：
    python scripts/prepare_twibot22.py --work-dir $WORK --sample-users 10000 --steps nodes graph
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np

from dtg_bot.data.encode import encode_field, encode_pooled, encode_tweets
from dtg_bot.data.graph import build_snapshot_properties, save_snapshot_properties
from dtg_bot.data.micro import build_micro_features
from dtg_bot.data.twibot22 import (
    build_twibot22_graph,
    build_user_table,
    collect_tweets,
    load_user_table,
)

ALL_STEPS = ("nodes", "graph", "desc", "pooled", "micro")


def _stamp(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--work-dir", required=True,
                    help="根目录；原始数据应在 <work-dir>/data/twibot22")
    ap.add_argument("--raw", default=None, help="覆盖原始数据目录")
    ap.add_argument("--cache", default=None, help="覆盖产物目录")
    ap.add_argument("--steps", nargs="+", default=list(ALL_STEPS), choices=ALL_STEPS)
    ap.add_argument("--seq-len", type=int, default=32,
                    help="微观分支每用户推文条数 L")
    ap.add_argument("--tweet-cap", type=int, default=None,
                    help="每用户保留推文上限，默认等于 seq-len")
    ap.add_argument("--pool-cap", type=int, default=20,
                    help="推文池化条数上限；0 表示不设上限")
    ap.add_argument("--num-snapshots", type=int, default=8)
    ap.add_argument("--interval", default="year", choices=["year", "month"],
                    help="宏观快照时间区间粒度")
    ap.add_argument("--model", default="roberta-base")
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--max-tokens", type=int, default=64)
    ap.add_argument("--sample-users", type=int, default=None,
                    help="分层抽样用户数（按 split × label），用于测试或减少规模")
    ap.add_argument("--sample-seed", type=int, default=42)
    ap.add_argument("--no-resume", action="store_true")
    args = ap.parse_args()

    work = Path(args.work_dir)
    raw = Path(args.raw) if args.raw else work / "data" / "twibot22"
    cache = Path(args.cache) if args.cache else work / "cache" / "twibot22"
    cache.mkdir(parents=True, exist_ok=True)
    tweet_cap = args.tweet_cap or args.seq_len
    resume = not args.no_resume
    graph_dir = cache / "graph"
    feat_dir = cache / "features"
    nodes_jsonl = cache / f"nodes_cap{tweet_cap}.jsonl"
    enc_dir = cache / f"encoded_L{args.seq_len}"
    micro_dir = cache / f"micro_L{args.seq_len}"

    _stamp(f"raw={raw}")
    _stamp(f"cache={cache}")
    _stamp(f"steps={args.steps}  L={args.seq_len}  "
           f"tweet_cap={tweet_cap}  pool_cap={args.pool_cap}  "
           f"sample_users={args.sample_users}")
    if not raw.exists():
        sys.exit(f"原始数据目录不存在: {raw}")

    summary: dict[str, object] = {}
    uid_to_idx: dict[str, int] | None = None
    node_ids: list[str] | None = None
    splits: list[str] | None = None
    labels: list[int] | None = None
    descriptions: list[str] | None = None
    created_ts: np.ndarray | None = None

    # ---------- 1. 节点表 + 用户序列 jsonl ----------
    if "nodes" in args.steps:
        if resume and nodes_jsonl.exists():
            _stamp(f"[nodes] 跳过，已存在 {nodes_jsonl}")
            uid_to_idx, node_ids, splits, labels, created_ts = load_user_table(cache)
            # descriptions 不单独保存；nodes_jsonl 已含全部信息
        else:
            t0 = time.time()
            uid_to_idx, node_ids, splits, labels, descriptions, created_ts = build_user_table(
                raw, cache,
                sample_users=args.sample_users,
                sample_seed=args.sample_seed,
            )
            _stamp(f"[nodes] {len(node_ids)} 个用户 ({time.time() - t0:.0f}s)")
            summary["n_nodes"] = len(node_ids)

            t0 = time.time()
            n = collect_tweets(
                raw, nodes_jsonl, uid_to_idx, node_ids, splits, labels, descriptions,
                tweet_cap=tweet_cap,
            )
            _stamp(f"[tweets] {n} 个用户 → {nodes_jsonl} ({time.time() - t0:.0f}s)")
            summary["n_labeled_jsonl"] = n
    else:
        uid_to_idx, node_ids, splits, labels, created_ts = load_user_table(cache)

    # ---------- 2. 图结构 ----------
    if "graph" in args.steps:
        if resume and (graph_dir / "graph_meta.json").exists() and (
            graph_dir / "edge_index.npy"
        ).exists():
            meta = json.loads((graph_dir / "graph_meta.json").read_text(encoding="utf-8"))
            _stamp("[graph] 跳过，已存在")
        else:
            t0 = time.time()
            meta = build_twibot22_graph(raw, cache, uid_to_idx)
            _stamp(f"[graph] 节点 {meta['n_nodes']}  边 {meta['n_edges']} "
                   f"丢弃 {meta['n_edges_dropped_unknown_endpoint']} ({time.time() - t0:.0f}s)")
        summary["graph"] = meta

        # 快照属性（NetworkX，大样本可能较慢；需要时可采样）
        _stamp("[graph] 构建快照属性（NetworkX；大样本可能较慢，可用 --sample-users 减少）")
        t0 = time.time()
        edge_index = np.load(graph_dir / "edge_index.npy")
        edge_type = np.load(graph_dir / "edge_type.npy")
        created_ts_arr = np.load(graph_dir / "created_ts.npy")
        props = build_snapshot_properties(
            edge_index, edge_type, created_ts_arr,
            num_snapshots=args.num_snapshots,
            interval=args.interval,
            following_relation=0,
        )
        save_snapshot_properties(graph_dir, props)
        cover = [round(float(m.mean()), 4) for m in props["masks"]]
        _stamp(f"[graph] {len(props['masks'])} 个快照边覆盖率: {cover} "
               f"({time.time() - t0:.0f}s)")
        summary["snapshot_coverage"] = cover

    # ---------- 3. description 嵌入 ----------
    if "desc" in args.steps:
        t0 = time.time()
        m = encode_field(nodes_jsonl, feat_dir, field="description",
                         model_name=args.model, max_tokens=args.max_tokens,
                         batch_size=args.batch_size, resume=resume)
        _stamp(f"[desc] 完成 ({time.time() - t0:.0f}s) {m['n_users']} 个节点")
        summary["desc"] = m

    # ---------- 4. 推文池化嵌入 ----------
    if "pooled" in args.steps:
        t0 = time.time()
        m = encode_pooled(nodes_jsonl, feat_dir, pool_cap=args.pool_cap,
                          model_name=args.model, max_tokens=args.max_tokens,
                          batch_size=args.batch_size, resume=resume)
        _stamp(f"[pooled] 完成 ({time.time() - t0:.0f}s) "
               f"均值池化 {m['mean_tweets_pooled']:.1f} 条/节点，"
               f"零推文节点 {m['n_users_with_no_tweet']}")
        summary["pooled"] = m

    # ---------- 5. 微观分支 ----------
    if "micro" in args.steps:
        t0 = time.time()
        m = encode_tweets(nodes_jsonl, enc_dir, seq_len=args.seq_len,
                          model_name=args.model, max_tokens=args.max_tokens,
                          batch_size=args.batch_size)
        _stamp(f"[micro] 逐条编码完成 ({time.time() - t0:.0f}s) "
               f"均值 {m['mean_valid_tweets']:.1f} 条/用户")
        t0 = time.time()
        mmeta = build_micro_features(nodes_jsonl, enc_dir, micro_dir,
                                     fit_split="train", with_temporal=True)
        _stamp(f"[micro] 事件序列完成 ({time.time() - t0:.0f}s) "
               f"{mmeta['n_channels']} 通道")
        summary["micro"] = {**m, "channels": mmeta["n_channels"]}

    out = cache / "prepare_summary.json"
    out.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print("\n" + "=" * 70)
    print("阶段摘要")
    print("=" * 70)
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    _stamp(f"摘要已写入 {out}")


if __name__ == "__main__":
    main()
