"""【闭环实验】TwiBot-22 order-vs-Δt 对照：真实时间通道相对次序无关事件特征的增量。

动机（论文 5.3 节"结论的适用边界"）
--------------------------------
TwiBot-20 的 keep≈shuffle 空结果有两种解释：
    (a) 事件次序本身不含判别信息；
    (b) TwiBot-20 的推文列表在构建时本就未保序，keep 与 shuffle 输入等价。
TwiBot-20 不含时间戳，无法自证，只能在 TwiBot-22 上分辨。

TwiBot-22 已实测 dump 近似保序（全量 88.2M 推文、相邻对违例率 3.29%，
见 experiments/ 下 collect 的 meta）。本脚本在其上做 **2×3 因子对照**：

    特征组   base     : 14 通道次序无关事件特征（与 TwiBot-20 实验完全同构）
             with_dt  : 14 + 4 通道，附加真实时间通道
                        dt_prev（log1p 相邻间隔秒数）、hour_sin/hour_cos、星期
    模型组   keep     : transformer + 位置编码 + 原始顺序
             shuffle  : transformer + 位置编码 + 每用户随机打乱
             bag      : 无自注意力，仅 mean/std 统计池化

关键比较（同一组种子配对 t 检验 + 族内 Holm 校正）：
    keep_dt  vs keep      真实时间通道的净增量（核心问题）
    bag_dt   vs bag       无注意力时的净增量（排除交互效应）
    keep     vs shuffle   T22 上重测次序效应（对照 T20 空结果）
    keep_dt  vs shuffle_dt 有时间特征后次序是否还有效
    keep_dt  vs bag_dt    时间特征下注意力聚合是否仍有效

解释预案：
    keep_dt > keep 显著        → 真实时刻携带 T20 所缺的判别信息，5.3 空结果
                                 更可能是"数据集未保序"所致
    keep_dt ≈ keep 且
    shuffle ≈ keep             → "次序与真实时刻均无信息"的强命题获得外部支持

产物：逐种子结果追加到 results.csv（experiment='order_dt'）。

样本规模：TwiBot-22 的 label.csv 与 split.csv 都覆盖全部 100 万用户
（train/val/test = 70 万/20 万/10 万，无 support 子集），因此"只取标注用户"
无法压缩规模。本实验只需微观分支的独立分类器，故按 (split, label) 分层
抽样 10 万用户——仍是 TwiBot-20 全量（11,826）的 8 倍，统计功效充裕，
而推文量从约 1500 万降到约 160 万。

用法（AutoDL，L=16 与 collect/encode 一致）：
    WORK=/root/autodl-tmp/dtg_bot
    python scripts/prepare_twibot22.py --work-dir $WORK --steps collect \
        --seq-len 16 --sample-users 100000              # 带 timestamps，勿用 --stats-only
    python scripts/prepare_twibot22.py --work-dir $WORK --steps encode \
        --seq-len 16 --batch-size 384
    python scripts/prepare_twibot22.py --work-dir $WORK --steps micro \
        --seq-len 16                                    # → micro_L16/base
    python scripts/prepare_twibot22.py --work-dir $WORK --steps micro \
        --seq-len 16 --with-temporal                    # → micro_L16/with_dt
    python scripts/diagnose_order_dt.py --cache $WORK/cache/twibot22 --seq-len 16 \
        --seeds 42 123 456 789 2024 7 13 99 2025 314 1618 271 577 999 8128
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
# scripts/ 目录已在 sys.path[0]，可直接复用 diagnose_order 的模型与训练循环
from diagnose_order import MicroOnlyClassifier, run_one  # noqa: E402
from dtg_bot.utils.metrics import append_result, summarize  # noqa: E402

#: 因子设计：特征组 × 模型组。variant 名 = f"{order}_{feat}"
FACTORIAL = [
    # (variant 名, 特征组, seq_model, order_mode)
    ("keep_base",      "base",    "transformer", "keep"),
    ("shuffle_base",   "base",    "transformer", "shuffle"),
    ("bag_base",       "base",    "bag",         "keep"),
    ("keep_dt",        "with_dt", "transformer", "keep"),
    ("shuffle_dt",     "with_dt", "transformer", "shuffle"),
    ("bag_dt",         "with_dt", "bag",         "keep"),
]

#: 计划内比较（a − b），全部进入同一个 Holm 校正族
COMPARISONS = [
    ("keep_dt",    "keep_base",    "真实时间通道的净增量（主问题）"),
    ("bag_dt",     "bag_base",     "无注意力时的时间通道净增量"),
    ("keep_base",  "shuffle_base", "T22 上的纯次序效应（对照 T20 空结果）"),
    ("keep_dt",    "shuffle_dt",   "加入时间特征后的次序效应"),
    ("keep_dt",    "bag_dt",       "时间特征下注意力聚合的价值"),
    ("keep_base",  "bag_base",     "无时间特征时注意力聚合的价值"),
]


def load_feature_set(cache: Path, seq_len: int, group: str):
    """加载一组特征。T22 布局：micro_L{L}/{base,with_dt}/。"""
    micro_dir = cache / f"micro_L{seq_len}" / group
    feat = np.load(micro_dir / "micro_feat.npy")
    mask = np.load(micro_dir / "micro_mask.npy")
    meta = json.loads((micro_dir / "micro_meta.json").read_text(encoding="utf-8"))
    return feat, mask, meta


def holm(pvals: np.ndarray) -> np.ndarray:
    m = len(pvals)
    adj = np.empty(m)
    running = 0.0
    for rank, idx in enumerate(np.argsort(pvals)):
        running = max(running, (m - rank) * pvals[idx])
        adj[idx] = min(running, 1.0)
    return adj


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default="./cache/twibot22")
    ap.add_argument("--dataset", default="TwiBot-22")
    ap.add_argument("--seq-len", type=int, default=16)
    ap.add_argument("--variants", nargs="+", default=[v[0] for v in FACTORIAL])
    ap.add_argument("--seeds", nargs="+", type=int,
                    default=[42, 123, 456, 789, 2024, 7, 13, 99, 2025, 314,
                             1618, 271, 577, 999, 8128])
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--patience", type=int, default=20)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight-decay", type=float, default=0.01)
    ap.add_argument("--d-model", type=int, default=64)
    ap.add_argument("--layers", type=int, default=2)
    ap.add_argument("--heads", type=int, default=4)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--results", default="./experiments/results_t22_dt.csv",
                    help="独立存档，避免与 TwiBot-20 的主 results.csv 混杂")
    args = ap.parse_args()

    cache = Path(args.cache)
    enc_dir = cache / f"encoded_L{args.seq_len}"
    labels = np.load(enc_dir / "labels.npy").astype(np.int64)
    splits = np.load(enc_dir / "splits.npy")
    idx = {}
    for s in ("train", "dev", "test"):
        rows = np.where((splits == s) & (labels >= 0))[0]
        if len(rows) == 0:
            raise RuntimeError(f"split '{s}' 无有效标注样本，请检查 label/split 对齐")
        idx[s] = rows
    print(f"train/dev/test = {len(idx['train'])}/{len(idx['dev'])}/{len(idx['test'])}")

    # 两套特征延迟加载，避免两组大数组同时驻留（1M×16×18 fp32 ≈ 1.2GB/组）
    feats: dict[str, tuple[np.ndarray, np.ndarray, dict]] = {}
    groups_needed = {g for name, g, _, _ in FACTORIAL if name in args.variants}
    for g in groups_needed:
        feats[g] = load_feature_set(cache, args.seq_len, g)
        meta = feats[g][2]
        print(f"[{g}] 通道 {meta['n_channels']}  {meta['channel_names']}")
    # 两套特征必须覆盖同一批用户、同一套 mask，否则配对检验不成立
    ref_mask = next(iter(feats.values()))[1]
    for g, (_, m, _) in feats.items():
        assert np.array_equal(m, ref_mask), f"{g} 的 mask 与其他组不一致，无法配对"

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device={device}")

    collected: dict[str, list[dict]] = {}
    for name, group, seq_model, order_mode in FACTORIAL:
        if name not in args.variants:
            continue
        feat, mask, meta = feats[group]
        variant = {"seq_model": seq_model, "order_mode": order_mode}
        rows = []
        for seed in args.seeds:
            res = run_one(feat, mask, labels, idx, meta, variant, seed, args, device)
            rows.append(res)
            append_result(args.results, {
                "dataset": args.dataset, "experiment": "order_dt",
                "variant": name, "feat_group": group,
                "seq_model": seq_model, "order_mode": order_mode,
                "seed": seed, "seq_len": meta["seq_len"], **res,
            })
            print(f"  {name:<13} seed={seed:<5} f1={res['f1']:.4f} "
                  f"acc={res['accuracy']:.4f} auc={res['auc']:.4f}")
        collected[name] = rows
        print(f"  {name:<13} → {summarize(rows)}\n")

    print("=" * 78)
    print("汇总 (mean ± std)")
    for name, rows in collected.items():
        s = summarize(rows)
        print(f"  {name:<13} F1 {s['f1']:<16} Acc {s['accuracy']:<16} AUC {s.get('auc', '-')}")

    # ---- 计划内比较：配对 t 检验 + 族内 Holm 校正 ----
    print("\n" + "=" * 78)
    print(f"配对 t 检验（n={len(args.seeds)}，族内 Holm 校正，m={len(COMPARISONS)}×2）")
    print("=" * 78)
    rows_all = []
    for metric in ("f1", "auc"):
        for a, b, desc in COMPARISONS:
            if a not in collected or b not in collected:
                continue
            xa = np.array([r[metric] for r in collected[a]]) * 100
            xb = np.array([r[metric] for r in collected[b]]) * 100
            t, p = stats.ttest_rel(xa, xb)
            rows_all.append((metric, a, b, desc, xa.mean() - xb.mean(), t, p))
    pvals = np.array([r[6] for r in rows_all])
    padj = holm(pvals)
    for (metric, a, b, desc, d, t, p), pa in zip(rows_all, padj):
        star = "***" if pa < 0.001 else "**" if pa < 0.01 else "*" if pa < 0.05 else "n.s."
        print(f"  [{metric.upper():3}] {a:<13}- {b:<13} Δ={d:+.3f}  "
              f"t={t:+.3f}  p={p:.4f}  p_holm={pa:.4f}  {star}   {desc}")

    print("\n判定：")
    print("  keep_dt - keep_base 显著为正 → 真实时间通道有增量（T20 空结果更可能是")
    print("      '数据集未保序'所致）；不显著且 keep-shuffle 仍不显著 → '次序与真实")
    print("      时刻均无信息'的强命题获得外部数据支持。")


if __name__ == "__main__":
    main()
