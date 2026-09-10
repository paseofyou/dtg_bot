"""微观分支序列长度敏感性（8/16/32/64）。

用法：
    python scripts/ablate_seq_len.py --raw D:/project/dataset/TwiBot-20/raw \
        --cache ./cache/twibot20 --seeds 42 123 456 789 2024
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

SEQ_LENS = [8, 16, 32, 64]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", default="D:/project/dataset/TwiBot-20/raw")
    ap.add_argument("--cache", default="./cache/twibot20")
    ap.add_argument("--splits", nargs="+", default=["train", "dev", "test"])
    ap.add_argument("--seeds", nargs="+", type=int,
                    default=[42, 123, 456, 789, 2024])
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--patience", type=int, default=20)
    ap.add_argument("--results", default="./experiments/results_seq_len.csv")
    args = ap.parse_args()

    base = [
        sys.executable, str(Path(__file__).resolve().parents[1] / "scripts" / "prepare_twibot20.py"),
        "--raw", str(args.raw), "--cache", str(args.cache), "--splits", *args.splits,
    ]

    for L in SEQ_LENS:
        print(f"\n{'='*60}")
        print(f"序列长度 L={L}")
        print(f"{'='*60}")
        subprocess.run(base + ["--seq-len", str(L)], check=True)
        subprocess.run([
            sys.executable,
            str(Path(__file__).resolve().parents[1] / "scripts" / "ablate_micro.py"),
            "--cache", str(args.cache),
            "--seq-len", str(L),
            "--channel-group", "full",
            "--pool", "attn",
            "--seeds", *map(str, args.seeds),
            "--results", str(args.results),
            "--epochs", str(args.epochs),
            "--patience", str(args.patience),
        ], check=True)

    print(f"\n全部长度完成，结果写入 {args.results}")


if __name__ == "__main__":
    main()
