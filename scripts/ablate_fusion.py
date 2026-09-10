"""融合方式消融：concat / simple attention (attn) / gated cross-view attention (gate)。

用法：
    python scripts/ablate_fusion.py --work-dir $WORK --seeds 42 123 456 789 2024
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

FUSIONS = ["concat", "attn", "gate"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--work-dir", required=True)
    ap.add_argument("--cache", default=None)
    ap.add_argument("--dataset", default="TwiBot-20")
    ap.add_argument("--views", nargs="+", default=["micro", "macro", "global"])
    ap.add_argument("--seeds", nargs="+", type=int,
                    default=[42, 123, 456, 789, 2024])
    ap.add_argument("--results", default=None)
    args = ap.parse_args()

    work = Path(args.work_dir)
    results = Path(args.results) if args.results else work / "experiments" / "ablate_fusion.csv"
    train_py = Path(__file__).resolve().parents[1] / "scripts" / "train.py"
    base_cmd = [
        sys.executable, str(train_py),
        "--work-dir", str(work),
        "--cache", str(args.cache) if args.cache else str(work / "cache" / "twibot20"),
        "--dataset", args.dataset,
        "--views", *args.views,
        "--seeds", *map(str, args.seeds),
        "--results", str(results),
    ]

    for fusion in FUSIONS:
        tag = f"fusion_{fusion}"
        cmd = base_cmd + ["--fusion", fusion, "--tag", tag]
        print(f"\n{'='*60}")
        print(f"开始跑融合方式: {fusion}  (tag={tag})")
        print(f"{'='*60}\n")
        subprocess.run(cmd, check=True)

    print(f"\n全部 {len(FUSIONS)} 种融合方式完成，结果写入 {results}")


if __name__ == "__main__":
    main()
