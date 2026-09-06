"""TwiBot-20 预处理：raw json → 有序推文 jsonl → RoBERTa 逐条嵌入 → 事件驱动序列。

用法（本地）:
    python scripts/prepare_twibot20.py --raw D:/project/dataset/TwiBot-20/raw \
        --cache ./cache/twibot20 --splits test          # 冒烟测试
    python scripts/prepare_twibot20.py --splits train dev test

用法（AutoDL）:
    python scripts/prepare_twibot20.py --raw $WORK/data/twibot20 --cache $WORK/cache/twibot20
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from dtg_bot.data.encode import encode_tweets
from dtg_bot.data.micro import build_micro_features
from dtg_bot.data.twibot20 import dump_ordered_tweets


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", default="D:/project/dataset/TwiBot-20/raw")
    ap.add_argument("--cache", default="./cache/twibot20")
    ap.add_argument("--splits", nargs="+", default=["train", "dev", "test"])
    ap.add_argument("--seq-len", type=int, default=32)
    ap.add_argument("--model", default="roberta-base")
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--max-tokens", type=int, default=64)
    ap.add_argument("--fit-split", default="train")
    ap.add_argument("--skip-existing", action="store_true")
    args = ap.parse_args()

    cache = Path(args.cache)
    cache.mkdir(parents=True, exist_ok=True)
    jsonl = cache / f"tweets_L{args.seq_len}.jsonl"
    enc_dir = cache / f"encoded_L{args.seq_len}"
    micro_dir = cache / f"micro_L{args.seq_len}"

    # --- 1. 有序推文导出 ---
    if args.skip_existing and jsonl.exists():
        print(f"[1/3] 跳过，已存在 {jsonl}")
    else:
        t0 = time.time()
        n = dump_ordered_tweets(args.raw, jsonl, splits=tuple(args.splits), seq_len=args.seq_len)
        print(f"[1/3] 导出 {n} 个用户的有序推文 → {jsonl}  ({time.time() - t0:.1f}s)")

    # --- 2. RoBERTa 逐条编码 ---
    if args.skip_existing and (enc_dir / "meta.json").exists():
        print(f"[2/3] 跳过，已存在 {enc_dir}")
        meta = json.loads((enc_dir / "meta.json").read_text(encoding="utf-8"))
    else:
        t0 = time.time()
        meta = encode_tweets(
            jsonl, enc_dir,
            seq_len=args.seq_len,
            model_name=args.model,
            max_tokens=args.max_tokens,
            batch_size=args.batch_size,
        )
        print(f"[2/3] 编码完成 ({time.time() - t0:.1f}s)")
    print("      ", json.dumps(meta, ensure_ascii=False))

    # --- 3. 事件驱动序列 ---
    t0 = time.time()
    micro_meta = build_micro_features(jsonl, enc_dir, micro_dir, fit_split=args.fit_split)
    print(f"[3/3] 事件序列完成 ({time.time() - t0:.1f}s) → {micro_dir}")
    print(f"       通道 ({micro_meta['n_channels']}): {micro_meta['channel_names']}")


if __name__ == "__main__":
    main()
