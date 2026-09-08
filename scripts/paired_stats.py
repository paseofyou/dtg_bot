"""逐种子配对 t 检验 + Holm-Bonferroni 校正。

输入：
  experiments/stage2_results_seedwise.csv   13 组变体 × 15 种子（f1/auc/best_dev_f1/accuracy）
  experiments/stage1_botrgcn_seedwise.csv   BotRGCN 同种子结果（f1/auc/accuracy，日志 4 位小数）

用法：python scripts/paired_stats.py（在仓库根目录执行）
"""
import numpy as np
import pandas as pd
from scipy import stats

SEEDS = [42, 123, 456, 789, 2024, 7, 13, 99, 2025, 314, 1618, 271, 577, 999, 8128]

df = pd.read_csv("experiments/stage2_results_seedwise.csv")
base = pd.read_csv("experiments/stage1_botrgcn_seedwise.csv")

pool = {}
for v, g in df.groupby("variant"):
    g = g.set_index("seed").reindex(SEEDS)
    assert g["f1"].notna().all(), f"{v} 种子缺失"
    pool[v] = g
pool["botrgcn"] = base.set_index("seed").reindex(SEEDS)


def paired(a, b, metric):
    """a - b 的配对检验，返回 (Δ百分点, t, p, Cohen's dz)。"""
    x = pool[a][metric].to_numpy(float) * 100
    y = pool[b][metric].to_numpy(float) * 100
    d = x - y
    t, p = stats.ttest_rel(x, y)
    dz = d.mean() / d.std(ddof=1) if d.std(ddof=1) > 0 else np.nan
    return d.mean(), t, p, dz


def holm(pvals):
    """Holm-Bonferroni 校正，返回校正后 p（保序）。"""
    m = len(pvals)
    order = np.argsort(pvals)
    adj = np.empty(m)
    running = 0.0
    for rank, idx in enumerate(order):
        val = (m - rank) * pvals[idx]
        running = max(running, val)
        adj[idx] = min(running, 1.0)
    return adj


def star(p):
    return "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "n.s."


FAMILIES = {
    "A 各变体 vs BotRGCN 基线": [
        (v, "botrgcn") for v in [
            "dtg_main", "abl_f_concat", "abl_f_gate", "abl_v_global", "abl_v_macro",
            "abl_v_micro", "abl_v_micro_global", "abl_v_micro_macro", "abl_v_macro_global",
            "abl_t_last", "abl_l_0", "abl_l_0.1", "abl_l_0.5", "abl_l_1.0",
        ]
    ],
    "B 视图消融（相对 global-only）": [
        ("abl_v_micro_global", "abl_v_global"),
        ("abl_v_macro_global", "abl_v_global"),
        ("dtg_main", "abl_v_global"),
        ("abl_f_concat", "abl_v_global"),
        ("abl_v_micro_macro", "abl_v_macro"),
        ("abl_v_micro", "abl_v_global"),
    ],
    "C 融合方式（相对 attn=dtg_main）": [
        ("abl_f_concat", "dtg_main"),
        ("abl_f_gate", "dtg_main"),
        ("abl_f_concat", "abl_f_gate"),
    ],
    "D InfoNCE 权重（相对 lambda=0）": [
        ("abl_l_0.1", "abl_l_0"),
        ("abl_l_0.5", "abl_l_0"),
        ("dtg_main", "abl_l_0"),
        ("abl_l_1.0", "abl_l_0"),
    ],
    "E 宏观时序演化 vs 终态快照": [
        ("dtg_main", "abl_t_last"),
    ],
}

all_rows = []
for metric in ["f1", "auc"]:
    print("=" * 96)
    print(f"指标：{metric.upper()}   n=15 配对 t 检验，Δ 为百分点，p_holm 为族内 Holm 校正")
    print("=" * 96)
    for fam, pairs in FAMILIES.items():
        res = [paired(a, b, metric) for a, b in pairs]
        praw = np.array([r[2] for r in res])
        padj = holm(praw)
        print(f"\n-- {fam}  (族内 m={len(pairs)})")
        print(f"{'对比':<44}{'Δ':>8}{'t':>9}{'p_raw':>11}{'p_holm':>11}{'dz':>8}  判定")
        for (a, b), (d, t, p, dz), pa in zip(pairs, res, padj):
            name = f"{a} - {b}"
            print(f"{name:<44}{d:>+8.3f}{t:>9.3f}{p:>11.2e}{pa:>11.2e}{dz:>8.2f}  {star(pa)}")
            all_rows.append((metric, fam, name, d, t, p, pa, dz))

# 全局 Holm（跨全部 2×28 个确认性对比，最保守）
print("\n" + "=" * 96)
print("全局 Holm 校正（所有指标所有族合并，m={}）：仅列出全局校正后仍显著者".format(len(all_rows)))
print("=" * 96)
gp = holm(np.array([r[5] for r in all_rows]))
for row, pa in sorted(zip(all_rows, gp), key=lambda x: x[1]):
    metric, fam, name, d, t, p, _, dz = row
    if pa < 0.05:
        print(f"[{metric.upper()}] {name:<44}Δ={d:+.3f}  p_global={pa:.2e}  {star(pa)}")

# 验证集正当性：best_dev_f1 上的融合方式对比
print("\n" + "=" * 96)
print("验证集正当性核验（best_dev_f1，配对）")
print("=" * 96)
dev = {v: pool[v]["best_dev_f1"].to_numpy(float) * 100 for v in pool if v != "botrgcn"}
for v in sorted(dev, key=lambda k: -dev[k].mean()):
    print(f"  {v:<24}{dev[v].mean():.3f} ± {dev[v].std(ddof=1):.3f}")
for a, b in [("abl_f_concat", "dtg_main"), ("abl_f_gate", "dtg_main"),
             ("abl_f_concat", "abl_v_global"), ("abl_v_micro_macro", "abl_v_global")]:
    d, t, p, dz = paired(a, b, "best_dev_f1")
    print(f"\n{a} - {b}: Δdev={d:+.3f}  t={t:.3f}  p={p:.4f}  {star(p)}")
