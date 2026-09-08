"""绘制 Stage-2 消融柱状图与 InfoNCE λ 敏感性折线图。

输入：
    experiments/stage1_botrgcn_seedwise.csv
    experiments/stage2_results_seedwise.csv

输出：
    figures/stage2_ablation_bars.{pdf,png}
    figures/stage2_lambda_auc.{pdf,png}

用法：
    python scripts/plot_ablation.py --data-dir experiments --out-dir figures
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

try:
    import seaborn as sns
    HAS_SEABORN = True
except ImportError:
    HAS_SEABORN = False

mpl.use("Agg")

#: 论文固定使用的 15 个种子
SEEDS = [42, 123, 456, 789, 2024, 7, 13, 99, 2025, 314, 1618, 271, 577, 999, 8128]

#: 消融柱状图顺序与标签
BAR_VARIANTS = [
    "botrgcn",
    "abl_v_micro",
    "abl_v_macro",
    "abl_v_global",
    "dtg_main",
    "abl_f_concat",
]
BAR_LABELS = [
    "BotRGCN",
    "micro",
    "macro",
    "global",
    "3-view (attn)",
    "3-view (concat)",
]

#: InfoNCE λ 敏感性：variant → λ 值
LAMBDA_VARIANTS = ["abl_l_0", "abl_l_0.1", "dtg_main", "abl_l_0.5", "abl_l_1.0"]
LAMBDA_VALUES = [0.0, 0.1, 0.3, 0.5, 1.0]


def set_style():
    """顶会风格：无衬线字体、黑色边框、嵌入字体。"""
    if HAS_SEABORN:
        sns.set_style("whitegrid")
    else:
        mpl.rcParams["axes.grid"] = True
        mpl.rcParams["grid.linestyle"] = "--"
        mpl.rcParams["grid.alpha"] = 0.3
    mpl.rcParams["font.family"] = "sans-serif"
    mpl.rcParams["font.sans-serif"] = ["Helvetica", "Arial", "DejaVu Sans"]
    mpl.rcParams["axes.linewidth"] = 1.0
    mpl.rcParams["xtick.major.width"] = 1.0
    mpl.rcParams["ytick.major.width"] = 1.0
    mpl.rcParams["pdf.fonttype"] = 42
    mpl.rcParams["ps.fonttype"] = 42
    mpl.rcParams["figure.dpi"] = 300


def load_data(data_dir: Path) -> pd.DataFrame:
    bot = pd.read_csv(data_dir / "stage1_botrgcn_seedwise.csv")
    main = pd.read_csv(data_dir / "stage2_results_seedwise.csv")
    df = pd.concat([bot, main], ignore_index=True)
    df["seed"] = df["seed"].astype(int)
    df = df[df["variant"].isin(BAR_VARIANTS + LAMBDA_VARIANTS)]
    return df


def reindex_variant(df: pd.DataFrame, variant: str) -> pd.DataFrame:
    """按固定种子对齐，缺失为 NaN。"""
    g = df[df["variant"] == variant].set_index("seed")
    return g.reindex(SEEDS)


def mean_std_percent(vals: pd.Series):
    v = vals.dropna().to_numpy(float) * 100
    return v.mean(), v.std(ddof=1 if len(v) > 1 else 0)


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


def paired_stats(base: pd.DataFrame, target: pd.DataFrame, metric: str):
    """target - base 的逐种子差异与（双侧）p 值。"""
    common = base[metric].dropna().index.intersection(target[metric].dropna().index)
    x = target.loc[common, metric].to_numpy(float)
    y = base.loc[common, metric].to_numpy(float)
    if len(x) < 2:
        return 1.0, 0.0
    _, p = stats.ttest_rel(x, y)
    return p, (x - y).mean() * 100


def star(p: float) -> str:
    if p < 0.001:
        return "***"
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    return ""


def save_fig(fig, out_dir: Path, name: str):
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_dir / f"{name}.pdf", bbox_inches="tight", transparent=True)
    fig.savefig(out_dir / f"{name}.png", dpi=300, bbox_inches="tight", transparent=True)
    plt.close(fig)


def plot_ablation_bars(df: pd.DataFrame, out_dir: Path):
    base = reindex_variant(df, "botrgcn")
    groups = {v: reindex_variant(df, v) for v in BAR_VARIANTS}

    f1_means, f1_stds, auc_means, auc_stds = [], [], [], []
    p_f1, p_auc, d_f1, d_auc = [], [], [], []
    for v in BAR_VARIANTS:
        g = groups[v]
        m1, s1 = mean_std_percent(g["f1"])
        ma, sa = mean_std_percent(g["auc"])
        f1_means.append(m1)
        f1_stds.append(s1)
        auc_means.append(ma)
        auc_stds.append(sa)

        if v == "botrgcn":
            p_f1.append(1.0)
            p_auc.append(1.0)
            d_f1.append(0.0)
            d_auc.append(0.0)
        else:
            pf, df1 = paired_stats(base, g, "f1")
            pa, da = paired_stats(base, g, "auc")
            p_f1.append(pf)
            p_auc.append(pa)
            d_f1.append(df1)
            d_auc.append(da)

    # 族内 Holm 校正（仅非基线条目）
    mask = np.array([v != "botrgcn" for v in BAR_VARIANTS])
    padj_f1 = np.ones(len(BAR_VARIANTS))
    padj_auc = np.ones(len(BAR_VARIANTS))
    padj_f1[mask] = holm(np.array(p_f1)[mask])
    padj_auc[mask] = holm(np.array(p_auc)[mask])

    stars_f1 = [star(p) if d > 0 else "" for p, d in zip(padj_f1, d_f1)]
    stars_auc = [star(p) if d > 0 else "" for p, d in zip(padj_auc, d_auc)]

    # 高对比度配色：基线深灰，单视图浅灰，三视图蓝/橙
    palette = ["#4C4C4C", "#9E9E9E", "#9E9E9E", "#9E9E9E", "#2E75B6", "#ED7D31"]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4.5))
    x = np.arange(len(BAR_LABELS))
    width = 0.6

    for ax, means, stds, title, ylabel, stars in [
        (ax1, f1_means, f1_stds, "TwiBot-20 Ablation (F1)", "F1 (%)", stars_f1),
        (ax2, auc_means, auc_stds, "TwiBot-20 Ablation (AUC)", "AUC (%)", stars_auc),
    ]:
        bars = ax.bar(
            x, means, width, yerr=stds, color=palette,
            edgecolor="black", linewidth=0.5, capsize=3, error_kw={"linewidth": 1.0},
            zorder=2,
        )
        ax.axhline(y=means[0], color="#4C4C4C", linestyle="--", linewidth=1.2, zorder=1, label="BotRGCN")
        ax.set_xticks(x)
        ax.set_xticklabels(BAR_LABELS, rotation=30, ha="right", fontsize=10)
        ax.set_ylabel(ylabel, fontsize=11)
        ax.set_title(title, fontsize=12, fontweight="bold")
        ax.tick_params(axis="y", labelsize=10)
        ax.grid(axis="y", linestyle="--", alpha=0.3, zorder=0)
        margin = max(stds) * 0.6 + 0.3
        ax.set_ylim(min(means) - margin, max(means) + margin + 0.8)

        for i, (m, s, st) in enumerate(zip(means, stds, stars)):
            if st:
                ax.text(i, m + s + 0.15, st, ha="center", va="bottom", fontsize=12, color="black", zorder=3)

    handles, labels = ax2.get_legend_handles_labels()
    ax2.legend(handles, labels, loc="lower right", frameon=False, fontsize=9)
    plt.tight_layout()
    save_fig(fig, out_dir, "stage2_ablation_bars")


def plot_lambda_sensitivity(df: pd.DataFrame, out_dir: Path):
    base = reindex_variant(df, "abl_l_0")
    auc_means, auc_stds = [], []
    p_vals, d_vals = [], []

    for v in LAMBDA_VARIANTS:
        g = reindex_variant(df, v)
        m, s = mean_std_percent(g["auc"])
        auc_means.append(m)
        auc_stds.append(s)
        if v == "abl_l_0":
            p_vals.append(1.0)
            d_vals.append(0.0)
        else:
            p, d = paired_stats(base, g, "auc")
            p_vals.append(p)
            d_vals.append(d)

    # 对 4 个非零 λ vs λ=0 做 Holm 校正
    non_zero = LAMBDA_VALUES[1:]
    padj = holm(np.array(p_vals[1:]))
    stars = [""] + [star(p) if d < 0 else "" for p, d in zip(padj, d_vals[1:])]

    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    x = np.array(LAMBDA_VALUES)
    y = np.array(auc_means)
    yerr = np.array(auc_stds)

    ax.plot(x, y, marker="o", markersize=7, linewidth=2.2, color="#C44E52", zorder=3, label="AUC")
    ax.errorbar(x, y, yerr=yerr, fmt="none", ecolor="#C44E52", capsize=4, capthick=1.5, elinewidth=1.5, zorder=2)
    for xi, yi, err, st in zip(x, y, yerr, stars):
        if st:
            ax.text(xi, yi + err + 0.05, st, ha="center", va="bottom", fontsize=12, color="#C44E52", zorder=4)

    ax.set_xlabel(r"InfoNCE weight $\lambda$", fontsize=11)
    ax.set_ylabel("AUC (%)", fontsize=11)
    ax.set_title("InfoNCE Alignment Sensitivity (TwiBot-20)", fontsize=12, fontweight="bold")
    ax.set_xlim(-0.08, 1.08)
    margin = max(yerr) * 0.8 + 0.15
    ax.set_ylim(min(y) - margin, max(y) + margin + 0.5)
    ax.set_xticks(x)
    ax.tick_params(axis="both", labelsize=10)
    ax.grid(axis="y", linestyle="--", alpha=0.3, zorder=0)

    # 标注 λ=0.3 对应 dtg_main
    ax.axvline(x=0.3, color="gray", linestyle=":", linewidth=1.0, alpha=0.7, zorder=1)
    ax.text(0.3, max(y) + margin * 0.4, "default", ha="center", va="bottom", fontsize=9, color="gray")

    plt.tight_layout()
    save_fig(fig, out_dir, "stage2_lambda_auc")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="./experiments")
    ap.add_argument("--out-dir", default="./figures")
    args = ap.parse_args()

    data_dir = Path(args.data_dir)
    out_dir = Path(args.out_dir)
    set_style()

    df = load_data(data_dir)
    missing = set(BAR_VARIANTS + LAMBDA_VARIANTS) - set(df["variant"].unique())
    if missing:
        raise ValueError(f"缺少以下变体的数据：{sorted(missing)}")

    plot_ablation_bars(df, out_dir)
    plot_lambda_sensitivity(df, out_dir)
    print(f"图表已保存至：{out_dir.resolve()}")


if __name__ == "__main__":
    main()
