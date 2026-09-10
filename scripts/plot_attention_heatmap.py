"""从 attention_vis.json 绘制事件级注意力 + 转移特征热力图。

用法：
    python scripts/plot_attention_heatmap.py \
        --in experiments/attention_vis.json --out-dir figures \
        --events top --n-users 10 --sort-by attention
"""

from __future__ import annotations

import argparse
import json
import re
import textwrap
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

mpl.use("Agg")

FEATURES = [
    ("weight", "Attention"),
    ("cos_prev", "Cosine"),
    ("l2_prev", "L2"),
    ("jaccard", "Jaccard"),
    ("rt_run", "RT run"),
]


def set_style():
    """顶会 Figure 板式：无衬线字体、细边框、嵌入字体。"""
    mpl.rcParams["font.family"] = "sans-serif"
    mpl.rcParams["font.sans-serif"] = ["Helvetica", "Arial", "DejaVu Sans"]
    mpl.rcParams["axes.linewidth"] = 1.0
    mpl.rcParams["xtick.major.width"] = 1.0
    mpl.rcParams["ytick.major.width"] = 1.0
    mpl.rcParams["pdf.fonttype"] = 42
    mpl.rcParams["ps.fonttype"] = 42
    mpl.rcParams["figure.dpi"] = 300


def _short_text(text: str, width: int = 34) -> str:
    return textwrap.shorten(str(text), width=width, placeholder="...")


def _sort_events(events: list[dict], sort_by: str) -> tuple[list[dict], np.ndarray]:
    """按 sort_by 重排事件并返回 (events, matrix)。"""
    if sort_by == "attention":
        order = np.argsort([-e["weight"] for e in events])
    else:
        order = np.argsort([e["pos"] for e in events])
    events = [events[int(i)] for i in order]

    keys = [c[0] for c in FEATURES]
    mat = np.array([[float(e[k]) for k in keys] for e in events], dtype=float)
    # rt_run 在事件通道里是 "连续 RT 段长度 / 10"，图中还原为整数段长更直观
    mat[:, -1] *= 10.0
    return events, mat


def plot_user(record: dict, out_dir: Path, events: str, sort_by: str, cmap: str = "RdYlBu_r"):
    """为单个用户绘制一张热力图并输出 pdf/png。"""
    if events == "top":
        ev = record.get("top_k_events", [])
    else:
        ev = record.get("all_events") or record.get("top_k_events", [])
    if not ev:
        print(f"  跳过 {record.get('user_id')}：无事件")
        return

    ev, mat = _sort_events(ev, sort_by)

    # 按列做 min-max 归一化，使不同量纲的特征可在同一色板下比较
    mn = mat.min(axis=0, keepdims=True)
    mx = mat.max(axis=0, keepdims=True)
    denom = np.where(mx > mn, mx - mn, 1.0)
    scaled = (mat - mn) / denom
    scaled = np.where(np.isfinite(scaled), scaled, 0.0)

    n_rows, n_cols = scaled.shape
    ylabels = [f"[{int(e['pos'])}] {_short_text(e['text'])}" for e in ev]

    fig, ax = plt.subplots(figsize=(0.9 * n_cols + 2.0, 0.55 * n_rows + 2.0))
    im = ax.imshow(scaled, aspect="auto", cmap=cmap, vmin=0, vmax=1)
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Relative intensity (per column)", fontsize=10)

    # 单元格数值标注
    for i in range(n_rows):
        for j in range(n_cols):
            text_color = "white" if scaled[i, j] > 0.6 else "black"
            ax.text(j, i, f"{mat[i, j]:.2f}", ha="center", va="center",
                    color=text_color, fontsize=8)

    ax.set_xticks(np.arange(n_cols))
    ax.set_xticklabels([c[1] for c in FEATURES], fontsize=10)
    ax.set_yticks(np.arange(n_rows))
    ax.set_yticklabels(ylabels, fontsize=8)

    # 网格线
    ax.set_xticks(np.arange(-0.5, n_cols, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, n_rows, 1), minor=True)
    ax.grid(which="minor", color="gray", linestyle="-", linewidth=0.5)
    ax.tick_params(which="minor", bottom=False, left=False)

    ax.set_xlabel("Event-level signals", fontsize=10)
    ax.set_ylabel("Tweets", fontsize=10)
    ax.tick_params(axis="x", labelsize=10)
    ax.tick_params(axis="y", labelsize=8)

    title = (
        f"User {record.get('user_id')}  "
        f"label={record.get('label')}  pred={record.get('pred')}  "
        f"n_tweets={record.get('n_tweets')}"
    )
    ax.set_title(title, fontsize=11, fontweight="bold")

    out_dir.mkdir(parents=True, exist_ok=True)
    uid = re.sub(r"[^A-Za-z0-9_-]", "_", str(record.get("user_id") or "unknown"))[:50]
    fig.savefig(out_dir / f"attention_heatmap_{uid}.pdf", bbox_inches="tight", transparent=True)
    fig.savefig(out_dir / f"attention_heatmap_{uid}.png", dpi=300, bbox_inches="tight", transparent=True)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="input", default="./experiments/attention_vis.json")
    ap.add_argument("--out-dir", default="./figures")
    ap.add_argument("--events", choices=["top", "all"], default="top",
                    help="top: 只画 top-k 事件；all: 画出所有事件（如果 JSON 提供）")
    ap.add_argument("--sort-by", choices=["attention", "position"], default="attention",
                    help="attention: 按注意力权重降序；position: 按推文位置（时间正序）")
    ap.add_argument("--n-users", type=int, default=None, help="最多绘制用户数，默认全部")
    ap.add_argument("--cmap", default="RdYlBu_r")
    args = ap.parse_args()

    set_style()
    data = json.loads(Path(args.input).read_text(encoding="utf-8"))
    if args.n_users is not None:
        data = data[:args.n_users]

    out_dir = Path(args.out_dir)
    for rec in data:
        plot_user(rec, out_dir, args.events, args.sort_by, args.cmap)

    print(f"热力图已保存至：{out_dir.resolve()} 共 {len(data)} 张")


if __name__ == "__main__":
    main()
