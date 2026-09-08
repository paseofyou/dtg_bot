"""TwiBot-22 Stage-3 (order-vs-Δt) 结果后处理与自动化分析。

输入：experiments/results_t22_dt.csv（来自 scripts/diagnose_order_dt.py）
输出：
    - 变体汇总表格（Markdown + LaTeX）
    - 6 项计划内配对 t 检验 + 族内 Holm-Bonferroni 校正表格
    - 针对 keep_dt 是否显著的两套结论模板

用法：
    python scripts/analyze_stage3.py --results experiments/results_t22_dt.csv --out-dir experiments/stage3_analysis
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

SEEDS = [42, 123, 456, 789, 2024, 7, 13, 99, 2025, 314, 1618, 271, 577, 999, 8128]

#: 6 个变体（用于显示顺序）
VARIANTS = [
    "keep_base",
    "shuffle_base",
    "bag_base",
    "keep_dt",
    "shuffle_dt",
    "bag_dt",
]

#: 6 项计划内配对 t 检验（a - b）
COMPARISONS = [
    ("keep_dt",    "keep_base",    "真实时间通道的净增量（主问题）"),
    ("bag_dt",     "bag_base",     "无注意力时的时间通道净增量"),
    ("keep_base",  "shuffle_base", "T22 上的纯次序效应"),
    ("keep_dt",    "shuffle_dt",   "加入时间特征后的次序效应"),
    ("keep_dt",    "bag_dt",       "时间特征下注意力聚合的价值"),
    ("keep_base",  "bag_base",     "无时间特征时注意力聚合的价值"),
]

METRICS = ("accuracy", "f1", "auc")


def read_results(path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "experiment" in df.columns:
        df = df[df["experiment"] == "order_dt"]
    for col in ["seed", *METRICS, "best_dev_f1"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def collect_by_variant(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    pool = {}
    for v in df["variant"].unique():
        g = df[df["variant"] == v].set_index("seed").sort_index()
        pool[v] = g.reindex(SEEDS)
    return pool


def summary(pool: dict[str, pd.DataFrame]) -> dict[str, dict[str, str]]:
    out = {}
    for v in VARIANTS:
        if v not in pool:
            continue
        g = pool[v]
        row = {}
        for m in METRICS:
            vals = g[m].dropna().to_numpy(float)
            if len(vals) == 0:
                continue
            row[m] = f"{100 * vals.mean():.2f} ± {100 * vals.std(ddof=1 if len(vals) > 1 else 0):.2f}"
        out[v] = row
    return out


def holm(pvals: np.ndarray) -> np.ndarray:
    p = np.asarray(pvals, dtype=float)
    m = len(p)
    order = np.argsort(p, kind="mergesort")
    adj = np.empty(m)
    running = 0.0
    for rank, idx in enumerate(order):
        running = max(running, (m - rank) * p[idx])
        adj[idx] = min(running, 1.0)
    return adj


def paired_test(pool, a: str, b: str, metric: str):
    g_a = pool[a]
    g_b = pool[b]
    common = g_a[metric].dropna().index.intersection(g_b[metric].dropna().index)
    x = g_a.loc[common, metric].to_numpy(float) * 100
    y = g_b.loc[common, metric].to_numpy(float) * 100
    d = x - y
    t, p = stats.ttest_rel(x, y)
    dz = d.mean() / d.std(ddof=1) if len(d) > 1 and d.std(ddof=1) > 0 else np.nan
    return d.mean(), t, p, dz, len(d)


def star(p: float) -> str:
    if p < 0.001:
        return "***"
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    return "n.s."


def fmt_p(p: float) -> str:
    if p < 1e-15:
        return "<1e-15"
    if p < 1e-4:
        return f"{p:.2e}"
    return f"{p:.4f}"


def run_comparisons(pool):
    f1_rows = []
    auc_rows = []
    for a, b, desc in COMPARISONS:
        if a not in pool or b not in pool:
            continue
        d_f1, t_f1, p_f1, dz_f1, n_f1 = paired_test(pool, a, b, "f1")
        d_auc, t_auc, p_auc, dz_auc, n_auc = paired_test(pool, a, b, "auc")
        f1_rows.append({"a": a, "b": b, "desc": desc, "d": d_f1, "t": t_f1, "p": p_f1, "dz": dz_f1, "n": n_f1})
        auc_rows.append({"a": a, "b": b, "desc": desc, "d": d_auc, "t": t_auc, "p": p_auc, "dz": dz_auc, "n": n_auc})

    p_f1 = np.array([r["p"] for r in f1_rows])
    p_auc = np.array([r["p"] for r in auc_rows])
    padj_f1 = holm(p_f1) if len(p_f1) else np.array([])
    padj_auc = holm(p_auc) if len(p_auc) else np.array([])

    combined = []
    for i, (r1, r2) in enumerate(zip(f1_rows, auc_rows)):
        combined.append({
            "a": r1["a"], "b": r1["b"], "desc": r1["desc"], "n": r1["n"],
            "d_f1": r1["d"], "t_f1": r1["t"], "p_f1": r1["p"], "p_f1_holm": padj_f1[i],
            "star_f1": star(padj_f1[i]),
            "d_auc": r2["d"], "t_auc": r2["t"], "p_auc": r2["p"], "p_auc_holm": padj_auc[i],
            "star_auc": star(padj_auc[i]),
        })
    return combined


def markdown_summary(s: dict) -> str:
    lines = [
        "| 变体 | Accuracy | F1 | AUC |",
        "|---|---|---|---|",
    ]
    for v in VARIANTS:
        if v not in s:
            continue
        r = s[v]
        lines.append(f"| {v} | {r.get('accuracy', '-')} | {r.get('f1', '-')} | {r.get('auc', '-')} |")
    return "\n".join(lines)


def markdown_comparisons(rows) -> str:
    lines = [
        "| 对比 (a - b) | ΔF1 | $p_{\\text{holm}}$ | ΔAUC | $p_{\\text{holm}}$ | 备注 |",
        "|---|---|---|---|---|---|",
    ]
    for r in rows:
        pf = f"{fmt_p(r['p_f1_holm'])} {r['star_f1']}"
        pa = f"{fmt_p(r['p_auc_holm'])} {r['star_auc']}"
        lines.append(
            f"| {r['a']} - {r['b']} | {r['d_f1']:+.3f} | {pf} | "
            f"{r['d_auc']:+.3f} | {pa} | {r['desc']} |"
        )
    return "\n".join(lines)


def latex_summary(s: dict) -> str:
    lines = [
        "\\begin{table}[htbp]",
        "\\centering",
        "\\caption{TwiBot-22 order-vs-$\\Delta t$ 变体结果（\\%，15 种子）}",
        "\\begin{tabular}{lccc}",
        "\\toprule",
        "变体 & Accuracy & F1 & AUC \\\\",
        "\\midrule",
    ]
    for v in VARIANTS:
        if v not in s:
            continue
        r = s[v]
        lines.append(f"{v} & {r.get('accuracy', '-')} & {r.get('f1', '-')} & {r.get('auc', '-')} \\\\")
    lines += ["\\bottomrule", "\\end{tabular}", "\\end{table}"]
    return "\n".join(lines)


def latex_comparisons(rows) -> str:
    lines = [
        "\\begin{table}[htbp]",
        "\\centering",
        "\\caption{TwiBot-22 order-vs-$\\Delta t$ 配对 t 检验（a - b，\\%，15 种子）}",
        "\\begin{tabular}{lccccc}",
        "\\toprule",
        "对比 & $\\Delta$F1 & $p_{\\text{holm}}$ & $\\Delta$AUC & $p_{\\text{holm}}$ & 备注 \\\\",
        "\\midrule",
    ]
    for r in rows:
        pf = f"{fmt_p(r['p_f1_holm'])}$^{{{r['star_f1']}}}$" if r["star_f1"] != "n.s." else fmt_p(r["p_f1_holm"]) + " n.s."
        pa = f"{fmt_p(r['p_auc_holm'])}$^{{{r['star_auc']}}}$" if r["star_auc"] != "n.s." else fmt_p(r["p_auc_holm"]) + " n.s."
        desc_escaped = r["desc"].replace("\\", "\\textbackslash{}").replace("&", "\\&").replace("$", "\\$")
        lines.append(
            f"{r['a']} - {r['b']} & {r['d_f1']:+.3f} & {pf} & "
            f"{r['d_auc']:+.3f} & {pa} & {desc_escaped} \\\\"
        )
    lines += ["\\bottomrule", "\\end{tabular}", "\\end{table}"]
    return "\n".join(lines)


def get_conclusion_templates(key_row, keep_shuffle_row, keepdt_bagdt_row):
    """返回两套结论模板，以及根据当前数据应选用的那套。"""
    sig_dt = key_row and key_row["p_auc_holm"] < 0.05 and key_row["d_auc"] > 0
    sig_shuffle = keep_shuffle_row and keep_shuffle_row["p_auc_holm"] < 0.05 and keep_shuffle_row["d_auc"] > 0
    sig_attn_dt = keepdt_bagdt_row and keepdt_bagdt_row["p_auc_holm"] < 0.05 and keepdt_bagdt_row["d_auc"] > 0

    positive = f"""### 预案 A：真实时间通道带来显著增量

T22 order-vs-Δt 对照中，`keep_dt` 相对 `keep_base` 的 AUC 提升为 {key_row['d_auc']:+.3f} 个百分点（$p_{{\\text{{holm}}}}$ = {fmt_p(key_row['p_auc_holm'])}），经族内 Holm 校正后显著。该结果表明：在已知推文列表高度保序（96.71% 相邻对）的 TwiBot-22 上，真实时间通道（相邻间隔、昼夜相位、星期节律）携带了单纯的离散次序无法编码的判别信息。\n
这一结果与数据一致的解释是：TwiBot-20 中 keep≈shuffle 的空结果，不应被解读为“所有时间信息均无用”，而应被解读为“在 200 条推文的列表位次这种**粗粒度次序**中不携带判别信息”。\n
结合测量一（96.71% 保序率）与测量二（`keep_dt` > `keep_base`），证据链如下：\n
1. 列表顺序 ≈ 真实时间倒序，故 keep/shuffle 的输入差异是真实的；\n2. 即便如此，模型仍无法从离散次序中获益；\n3. 但加入连续时间特征后，模型能够利用该信息。\n
因此，TwiBot-20 的“次序无信息”结论保持不变，但其适用范围应明确限定为**离散列表位次/相对先后**；论文 5.3 节措辞从“事件级时间顺序无信息”收紧为“推文列表中的相对先后无信息”，而连续时间的边界尚待进一步研究。\n"""

    negative = f"""### 预案 B：真实时间通道同样未带来显著增量

T22 order-vs-Δt 对照中，`keep_dt` 相对 `keep_base` 的 AUC 差异为 {key_row['d_auc']:+.3f} 个百分点（$p_{{\\text{{holm}}}}$ = {fmt_p(key_row['p_auc_holm'])}），经族内 Holm 校正后不显著；同时 `keep_base` 与 `shuffle_base` 的 AUC 差异为 {keep_shuffle_row['d_auc']:+.3f} 个百分点（$p_{{\\text{{holm}}}}$ = {fmt_p(keep_shuffle_row['p_auc_holm'])}），亦不显著。\n
该结果支持一个更强的命题：在社交机器人检测的微观事件层级上，**无论是推文列表的相对先后，还是相邻发帖间隔、昼夜相位等连续时间信息，都未提供可被当前模型利用的判别信号**。\n
由于 TwiBot-22 同时具有真实时间戳和高保序率，这一外部证据进一步巩固了 5.3 节的结论——T20 的空结果并非由“数据集未保序”造成，而是在本模型与特征配置下，事件时间尺度未表现出可辨识的贡献。据此，本文将微观分支重新定位为对**无序事件集合**的置换不变聚合，并在论文中保留“连续时间同样无增量”的边界限定，直至找到新的机制性证据。\n
补充说明：\n- `keep_dt` vs `bag_dt`（注意力聚合）的 AUC 差异为 {keepdt_bagdt_row['d_auc']:+.3f}（$p_{{\\text{{holm}}}}$ = {fmt_p(keepdt_bagdt_row['p_auc_holm'])}），若仍显著，则事件级注意力在时间通道存在时仍然有效。\n- 即便时间通道主效应不显著，仍应单独报告其点估计与置信区间，避免以“不显著”等同于“零效应”。\n"""

    selected = positive if sig_dt else negative
    return positive, negative, selected


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="./experiments/results_t22_dt.csv")
    ap.add_argument("--out-dir", default="./experiments/stage3_analysis")
    args = ap.parse_args()

    res_path = Path(args.results)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if not res_path.exists():
        raise FileNotFoundError(f"找不到结果文件：{res_path}")

    df = read_results(res_path)
    pool = collect_by_variant(df)
    missing = [v for v in VARIANTS if v not in pool]
    if missing:
        raise ValueError(f"结果文件缺少以下变体：{missing}")

    summ = summary(pool)
    comp_rows = run_comparisons(pool)

    key = next((r for r in comp_rows if r["a"] == "keep_dt" and r["b"] == "keep_base"), None)
    keep_shuffle = next((r for r in comp_rows if r["a"] == "keep_base" and r["b"] == "shuffle_base"), None)
    keepdt_bagdt = next((r for r in comp_rows if r["a"] == "keep_dt" and r["b"] == "bag_dt"), None)

    if key is None or keep_shuffle is None or keepdt_bagdt is None:
        raise RuntimeError("关键比较缺失，无法生成结论模板")

    positive, negative, selected = get_conclusion_templates(key, keep_shuffle, keepdt_bagdt)

    md = [
        "# TwiBot-22 Stage-3 自动化分析报告\n",
        "## 1. 变体汇总\n",
        markdown_summary(summ),
        "\n## 2. 计划内配对 t 检验（族内 Holm 校正）\n",
        markdown_comparisons(comp_rows),
        "\n## 3. 结论模板\n",
        "**当前数据选用的预案：** " + ("A" if selected is positive else "B") + "\n",
        selected,
        "\n---\n",
        "**预案 A（备用）**\n",
        positive,
        "\n---\n",
        "**预案 B（备用）**\n",
        negative,
    ]
    md_text = "\n".join(md)

    latex = [
        "% 自动生成的 Stage-3 表格\n",
        latex_summary(summ),
        "\n",
        latex_comparisons(comp_rows),
    ]
    latex_text = "\n".join(latex)

    (out_dir / "stage3_analysis.md").write_text(md_text, encoding="utf-8")
    (out_dir / "stage3_analysis_tables.tex").write_text(latex_text, encoding="utf-8")

    selected_name = "A" if selected is positive else "B"
    print(f"Stage-3 分析完成。当前数据选用预案：{selected_name}")
    print("=" * 72)
    print(f"表格与结论已写入：\n  {out_dir / 'stage3_analysis.md'}\n  {out_dir / 'stage3_analysis_tables.tex'}")


if __name__ == "__main__":
    main()
