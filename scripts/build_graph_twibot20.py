"""构建 TwiBot-20 的图结构与静态特征（纯 CPU，需扫过 support.json，约 5GB）。

用法：
    python scripts/build_graph_twibot20.py --raw D:/project/dataset/TwiBot-20/raw \
        --cache ./cache/twibot20
    # 只用标注用户建子图（快速验证管线）：
    python scripts/build_graph_twibot20.py --splits train dev test
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from dtg_bot.data.graph import build_graph, snapshot_edge_masks


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", default="D:/project/dataset/TwiBot-20/raw")
    ap.add_argument("--cache", default="./cache/twibot20")
    ap.add_argument("--splits", nargs="+", default=["train", "dev", "test", "support"])
    ap.add_argument("--num-snapshots", type=int, default=8)
    args = ap.parse_args()

    out_dir = Path(args.cache) / "graph"
    t0 = time.time()
    meta = build_graph(args.raw, out_dir, splits=tuple(args.splits))
    print(f"\n建图完成 ({time.time() - t0:.1f}s) → {out_dir}")
    print(json.dumps(meta, indent=2, ensure_ascii=False))

    # 宏观快照掩码：顺带报告每个快照保留多少边，用于确认切分是否合理
    edge_index = np.load(out_dir / "edge_index.npy")
    created_ts = np.load(out_dir / "created_ts.npy")
    masks, cutoffs = snapshot_edge_masks(edge_index, created_ts, args.num_snapshots)
    print(f"\n宏观动态图快照 (num_snapshots={args.num_snapshots}):")
    total = edge_index.shape[1]
    for k, (mask, cut) in enumerate(zip(masks, cutoffs)):
        from datetime import datetime, timezone
        stamp = datetime.fromtimestamp(cut, tz=timezone.utc).strftime("%Y-%m-%d")
        print(f"  t{k}: cutoff={stamp}  edges={int(mask.sum()):>8} / {total}  "
              f"({100 * mask.mean():.1f}%)")
    np.save(out_dir / "snapshot_cutoffs.npy", cutoffs)


if __name__ == "__main__":
    main()
