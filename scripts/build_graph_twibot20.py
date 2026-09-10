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

from dtg_bot.data.graph import build_graph, build_snapshot_properties, save_snapshot_properties


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", default="D:/project/dataset/TwiBot-20/raw")
    ap.add_argument("--cache", default="./cache/twibot20")
    ap.add_argument("--splits", nargs="+", default=["train", "dev", "test", "support"])
    ap.add_argument("--num-snapshots", type=int, default=8)
    ap.add_argument("--interval", default="year", choices=["year", "month"])
    args = ap.parse_args()

    out_dir = Path(args.cache) / "graph"
    t0 = time.time()
    meta = build_graph(args.raw, out_dir, splits=tuple(args.splits))
    print(f"\n建图完成 ({time.time() - t0:.1f}s) → {out_dir}")
    print(json.dumps(meta, indent=2, ensure_ascii=False))

    # 宏观动态图快照：按 BotDGT 原装时间区间 + 聚类/双向链接位置编码
    print("\n构建 BotDGT 宏观快照属性...")
    t0 = time.time()
    edge_index = np.load(out_dir / "edge_index.npy")
    edge_type = np.load(out_dir / "edge_type.npy")
    created_ts = np.load(out_dir / "created_ts.npy")
    props = build_snapshot_properties(
        edge_index, edge_type, created_ts,
        num_snapshots=args.num_snapshots, interval=args.interval,
        following_relation=0,
    )
    save_snapshot_properties(out_dir, props)
    total = edge_index.shape[1]
    for k, cut in enumerate(props["cutoffs"]):
        from datetime import datetime, timezone
        stamp = datetime.fromtimestamp(float(cut), tz=timezone.utc).strftime("%Y-%m-%d")
        print(f"  t{k}: cutoff={stamp}  edges={int(props['masks'][k].sum()):>8} / {total}  "
              f"({100 * props['masks'][k].mean():.1f}%)")
    print(f"快照属性保存完成 ({time.time() - t0:.1f}s) → {out_dir}")


if __name__ == "__main__":
    main()
