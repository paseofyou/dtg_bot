"""TwiBot-20 全量预处理（云端 GPU 运行）。

产出四类特征，每步独立可重跑、支持断点续跑：

    nodes    全部 229,580 个节点的文本 jsonl（顺序与建图一致）
    graph    多关系边表 + 静态属性 + created_at + 快照切点
    desc     description 嵌入 (N, 768)          —— BotRGCN 四路特征之一
    pooled   推文池化嵌入 (N, 768)              —— BotRGCN 四路特征之一
    micro    标注用户的逐条推文嵌入 + 事件序列   —— 本文微观分支

用法（AutoDL）：
    WORK=/root/autodl-tmp/dtg_bot
    python scripts/prepare_twibot20_full.py --work-dir $WORK --pool-cap 20 \
        --batch-size 256 --steps nodes graph desc pooled micro

断点续跑：直接重复执行同一条命令即可，已完成的用户会被跳过。
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
from dtg_bot.data.graph import build_graph, snapshot_edge_masks
from dtg_bot.data.micro import build_micro_features
from dtg_bot.data.twibot20 import LABELED_SPLITS, SPLITS, dump_nodes, dump_ordered_tweets

ALL_STEPS = ("nodes", "graph", "desc", "pooled", "micro")


def _stamp(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--work-dir", required=True,
                    help="根目录；原始数据应在 <work-dir>/data/twibot20")
    ap.add_argument("--raw", default=None, help="覆盖原始数据目录")
    ap.add_argument("--cache", default=None, help="覆盖产物目录")
    ap.add_argument("--steps", nargs="+", default=list(ALL_STEPS), choices=ALL_STEPS)
    ap.add_argument("--splits", nargs="+", default=list(SPLITS), choices=list(SPLITS),
                    help="参与建图与编码的 split。去掉 support 可做小规模自洽冒烟测试，"
                         "但边数会从 227979 降到 16908，不可用于报告基线")
    ap.add_argument("--seq-len", type=int, default=32, help="微观分支每用户推文条数 L")
    ap.add_argument("--tweet-cap", type=int, default=200, help="jsonl 中每用户保留的推文上限")
    ap.add_argument("--pool-cap", type=int, default=20,
                    help="参与池化的推文条数上限；0 = 不设上限（约 12h @3090）")
    ap.add_argument("--num-snapshots", type=int, default=8)
    ap.add_argument("--model", default="roberta-base")
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--max-tokens", type=int, default=64)
    ap.add_argument("--no-resume", action="store_true")
    args = ap.parse_args()

    work = Path(args.work_dir)
    raw = Path(args.raw) if args.raw else work / "data" / "twibot20"
    cache = Path(args.cache) if args.cache else work / "cache" / "twibot20"
    cache.mkdir(parents=True, exist_ok=True)
    resume = not args.no_resume
    splits = tuple(args.splits)

    nodes_jsonl = cache / f"nodes_cap{args.tweet_cap}.jsonl"
    labeled_jsonl = cache / f"labeled_L{args.seq_len}.jsonl"
    graph_dir = cache / "graph"
    feat_dir = cache / "features"
    enc_dir = cache / f"encoded_L{args.seq_len}"
    micro_dir = cache / f"micro_L{args.seq_len}"

    _stamp(f"raw={raw}")
    _stamp(f"cache={cache}")
    _stamp(f"steps={args.steps}  splits={splits}  pool_cap={args.pool_cap}  L={args.seq_len}")
    if not raw.exists():
        sys.exit(f"原始数据目录不存在: {raw}")
    if set(splits) != set(SPLITS):
        _stamp("[WARN] 未使用全部 split，图不完整，结果不可用于报告基线")

    summary: dict[str, object] = {}

    # ---------- 1. 全节点文本 jsonl ----------
    if "nodes" in args.steps:
        if resume and nodes_jsonl.exists():
            _stamp(f"[nodes] 跳过，已存在 {nodes_jsonl}")
        else:
            t0 = time.time()
            n = dump_nodes(raw, nodes_jsonl, splits=splits, tweet_cap=args.tweet_cap)
            _stamp(f"[nodes] {n} 个节点 → {nodes_jsonl}  ({time.time() - t0:.0f}s)")
            summary["n_nodes_jsonl"] = n
        labeled_splits = tuple(s for s in LABELED_SPLITS if s in splits)
        if resume and labeled_jsonl.exists():
            _stamp("[nodes] 跳过标注用户 jsonl，已存在")
        else:
            n = dump_ordered_tweets(raw, labeled_jsonl, splits=labeled_splits,
                                    seq_len=args.seq_len)
            _stamp(f"[nodes] 标注用户 {n} 个 → {labeled_jsonl}")
            summary["n_labeled_jsonl"] = n

    # ---------- 2. 图结构 ----------
    if "graph" in args.steps:
        if resume and (graph_dir / "graph_meta.json").exists():
            meta = json.loads((graph_dir / "graph_meta.json").read_text(encoding="utf-8"))
            _stamp("[graph] 跳过，已存在")
        else:
            t0 = time.time()
            meta = build_graph(raw, graph_dir, splits=splits)
            _stamp(f"[graph] 完成 ({time.time() - t0:.0f}s)")
        _stamp(f"[graph] 节点 {meta['n_nodes']}  边 {meta['n_edges']}  "
               f"标注 {meta['n_labeled']} (bot {meta['n_bot']} / human {meta['n_human']})")
        summary["graph"] = meta

        masks, cutoffs = snapshot_edge_masks(
            np.load(graph_dir / "edge_index.npy"),
            np.load(graph_dir / "created_ts.npy"),
            args.num_snapshots,
        )
        np.save(graph_dir / "snapshot_cutoffs.npy", cutoffs)
        cover = [round(float(m.mean()), 4) for m in masks]
        _stamp(f"[graph] {args.num_snapshots} 个快照的边覆盖率: {cover}")
        summary["snapshot_coverage"] = cover

    # ---------- 3. description 嵌入（全节点） ----------
    if "desc" in args.steps:
        t0 = time.time()
        meta = encode_field(nodes_jsonl, feat_dir, field="description",
                            model_name=args.model, max_tokens=args.max_tokens,
                            batch_size=args.batch_size, resume=resume)
        _stamp(f"[desc] 完成 ({time.time() - t0:.0f}s) {meta['n_users']} 个节点")
        summary["desc"] = meta

    # ---------- 4. 推文池化嵌入（全节点，BotRGCN 基线所需） ----------
    if "pooled" in args.steps:
        t0 = time.time()
        meta = encode_pooled(nodes_jsonl, feat_dir, pool_cap=args.pool_cap,
                             model_name=args.model, max_tokens=args.max_tokens,
                             batch_size=args.batch_size, resume=resume)
        _stamp(f"[pooled] 完成 ({time.time() - t0:.0f}s) "
               f"均值池化 {meta['mean_tweets_pooled']:.1f} 条/节点，"
               f"零推文节点 {meta['n_users_with_no_tweet']}")
        summary["pooled"] = meta

    # ---------- 5. 微观分支（仅标注用户） ----------
    if "micro" in args.steps:
        t0 = time.time()
        meta = encode_tweets(labeled_jsonl, enc_dir, seq_len=args.seq_len,
                             model_name=args.model, max_tokens=args.max_tokens,
                             batch_size=args.batch_size)
        _stamp(f"[micro] 逐条编码完成 ({time.time() - t0:.0f}s) "
               f"均值 {meta['mean_valid_tweets']:.1f} 条/用户")
        t0 = time.time()
        mmeta = build_micro_features(labeled_jsonl, enc_dir, micro_dir, fit_split="train")
        _stamp(f"[micro] 事件序列完成 ({time.time() - t0:.0f}s) "
               f"{mmeta['n_channels']} 通道")
        summary["micro"] = {**meta, "channels": mmeta["n_channels"]}

    out = cache / "prepare_summary.json"
    out.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print("\n" + "=" * 70)
    print("阶段摘要 (可直接贴回)")
    print("=" * 70)
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    _stamp(f"摘要已写入 {out}")


if __name__ == "__main__":
    main()
