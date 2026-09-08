"""TwiBot-22 预处理（云端运行）。

分三步，每步产物落盘可单独重跑：
    1. collect  单遍扫描 tweet_0..8.json（约 101GB），每作者取最近 L 条 + 真实时间戳
    2. encode   RoBERTa 逐条编码
    3. micro    事件转移序列，可选附加真实时间通道（order-vs-Δt 实验用）

用法（AutoDL）：
    WORK=/root/autodl-tmp/dtg_bot
    python scripts/prepare_twibot22.py --data $WORK/data/twibot22 --cache $WORK/cache/twibot22 \
        --seq-len 16 --batch-size 384 --steps collect encode micro

    # 只跑 collect 先看顺序违例率（这一步就能验证"列表顺序=真实时序"假设）：
    python scripts/prepare_twibot22.py --data ... --cache ... --steps collect

    # 本地小规模冒烟测试（只扫第一个 tweet 文件、只留 2000 个用户）：
    python scripts/prepare_twibot22.py --data D:/dataset/raw_data/TwiBot_22 \
        --cache ./cache/twibot22_smoke --tweet-files tweet_0.json --max-users 2000
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from dtg_bot.data.encode import encode_tweets
from dtg_bot.data.micro import build_micro_features
from dtg_bot.data.twibot22 import (
    TWEET_FILES,
    collect_recent_tweets,
    load_labels,
    load_splits,
)


def stratified_sample(
    keep: set[str],
    labels: dict[str, int],
    splits: dict[str, str],
    n_target: int,
    seed: int,
) -> set[str]:
    """按 (split, label) 分层抽样，保持各层比例与总体一致。

    为什么必须抽样而不是"只取标注用户"：TwiBot-22 的 ``label.csv`` 与 ``split.csv``
    都恰好覆盖全部 100 万用户（已核验：两者交集 100 万、各自差集为 0，
    train/val/test = 70 万/20 万/10 万），**不存在无标签的 support 子集**。
    因此 ``keep = set(labels)`` 已是全量，规模压缩只能靠抽样实现。

    分层而非均匀随机：bot 仅占 14%（139,943/1,000,000），
    小样本下均匀抽样会让正类比例产生可观漂移，进而改变 F1 的基线水平，
    使 TwiBot-22 的结果无法与 TwiBot-20 对照。
    """
    strata: dict[tuple[str, int], list[str]] = {}
    for uid in keep:
        key = (splits.get(uid, "train"), labels[uid])
        strata.setdefault(key, []).append(uid)

    rng = np.random.default_rng(seed)
    frac = n_target / len(keep)
    sampled: set[str] = set()
    print(f"[sample] 分层抽样 {len(keep)} → 目标 {n_target}（比例 {frac:.4f}，seed={seed}）")
    for key in sorted(strata):
        members = sorted(strata[key])                     # 排序保证跨机器可复现
        take = max(1, int(round(len(members) * frac)))
        take = min(take, len(members))
        chosen = rng.choice(len(members), size=take, replace=False)
        sampled.update(members[i] for i in chosen)
        print(f"         {key[0]:<6} label={key[1]}  {len(members):>7} → {take:>6}")
    return sampled


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--work-dir", default=None,
                    help="根目录；数据默认在 <work-dir>/data/twibot22，产物在 <work-dir>/cache/twibot22")
    ap.add_argument("--data", default=None, help="含 user.json/label.csv/split.csv/tweet_*.json")
    ap.add_argument("--cache", default=None)
    ap.add_argument("--seq-len", type=int, default=16)
    ap.add_argument("--model", default="roberta-base")
    ap.add_argument("--batch-size", type=int, default=384)
    ap.add_argument("--max-tokens", type=int, default=64)
    ap.add_argument("--tweet-files", nargs="+", default=list(TWEET_FILES))
    ap.add_argument("--max-users", type=int, default=0, help=">0 时只保留前 N 个标注用户（冒烟测试）")
    ap.add_argument("--sample-users", type=int, default=0,
                    help=">0 时按 (split, label) 分层抽样 N 个标注用户。"
                         "注意 TwiBot-22 的 label.csv/split.csv 都覆盖全部 100 万用户，"
                         "不存在无标签子集，故'只取标注用户'无法减小规模，只能抽样。")
    ap.add_argument("--sample-seed", type=int, default=42, help="分层抽样随机种子（可复现）")
    ap.add_argument("--steps", nargs="+", default=["collect", "encode", "micro"],
                    choices=["collect", "encode", "micro"])
    ap.add_argument("--with-temporal", action="store_true",
                    help="micro 步骤附加真实时间通道（order-vs-Δt 实验）")
    ap.add_argument("--stats-only", action="store_true",
                    help="collect 只统计时间戳单调性、不保存文本（内存 5GB→130MB）")
    args = ap.parse_args()

    if args.work_dir:
        work = Path(args.work_dir)
        args.data = args.data or str(work / "data" / "twibot22")
        args.cache = args.cache or str(work / "cache" / "twibot22")
    if not args.data or not args.cache:
        ap.error("需提供 --work-dir，或同时提供 --data 与 --cache")
    if not Path(args.data).exists():
        ap.error(f"数据目录不存在: {args.data}")

    cache = Path(args.cache)
    cache.mkdir(parents=True, exist_ok=True)
    tag = f"L{args.seq_len}"
    raw_jsonl = cache / f"tweets_{tag}.jsonl"
    enc_dir = cache / f"encoded_{tag}"
    micro_dir = cache / f"micro_{tag}" / ("with_dt" if args.with_temporal else "base")

    if "collect" in args.steps:
        labels = load_labels(args.data)
        splits = load_splits(args.data)
        keep = set(labels)
        if args.max_users:
            keep = set(sorted(keep)[: args.max_users])
        if args.sample_users and args.sample_users < len(keep):
            keep = stratified_sample(keep, labels, splits, args.sample_users, args.sample_seed)
            # 抽样名单落盘：encode/micro 依赖 collect 产出的 jsonl，本身自动一致；
            # 单独存一份是为了让论文结果可追溯到确切的用户集合。
            (cache / f"sampled_users_{tag}.json").write_text(
                json.dumps({"n": len(keep), "seed": args.sample_seed,
                            "ids": sorted(keep)}, indent=2), encoding="utf-8")
        elif not args.max_users:
            print("[collect] ⚠️ 未指定 --sample-users，将收集全部标注用户。"
                  "TwiBot-22 的标注集即全量 100 万用户（无 support 子集），"
                  "文本驻留内存约 6~10GB，且后续编码约 1500 万条推文。"
                  "如只需微观诊断实验，建议加 --sample-users 100000。")
        print(f"[collect] 标注用户 {len(labels)}，本次收集 {len(keep)}；"
              f"扫描 {len(args.tweet_files)} 个 tweet 文件")

        t0 = time.time()
        meta = collect_recent_tweets(
            args.data, raw_jsonl, seq_len=args.seq_len,
            keep_users=keep, tweet_files=tuple(args.tweet_files),
            stats_only=args.stats_only,
        )
        print(f"[collect] 完成 ({time.time() - t0:.0f}s)")
        print("          " + json.dumps(meta, indent=2))

        print("\n" + "=" * 70)
        print("顺序假设实证检验（论文 5.5.1 节 / 3.3 节结论所依赖）")
        print("=" * 70)
        print(f"  扫描推文        : {meta['n_tweets_scanned']:,}")
        print(f"  作者数          : {meta['n_authors']:,}")
        print(f"  相邻推文对      : {meta['n_adjacent_pairs']:,}")
        print(f"  时间倒序违例    : {meta['n_order_violations']:,}")
        print(f"  违例率          : {meta['order_violation_rate']:.6%}")
        print(f"  有违例的作者占比: {meta['frac_authors_with_violation']:.4%}")
        print()
        rate = meta["order_violation_rate"]
        if rate < 0.001:
            print("  [ORDER-PRESERVED] 违例率 < 0.1%，dump 严格按时间倒序，列表保序假设成立。")
            print("     TwiBot-20 的 keep~shuffle 空结果可解释为'顺序本身无判别信息'。")
        elif rate < 0.05:
            print("  [ORDER-MOSTLY] 违例率偏高但整体有序，假设基本成立，论文中需报告该比率。")
        else:
            print("  [ORDER-BROKEN] 违例率显著，dump 未保序。TwiBot-20 上的顺序空结果")
            print("     不能否证'顺序含信息'，论文 3.3/5.5 必须相应降级表述。")
        print("=" * 70 + "\n")

        if args.stats_only:
            print("[collect] stats-only 模式，未写出 jsonl；"
                  "拿到违例率后请去掉 --stats-only 再跑完整 collect。")
            return

        # 补上 label / split，对齐 encode.py 期望的 jsonl 格式
        tmp = raw_jsonl.with_suffix(".tmp")
        n_kept = 0
        with raw_jsonl.open("r", encoding="utf-8") as fin, tmp.open("w", encoding="utf-8") as fout:
            for line in fin:
                rec = json.loads(line)
                uid = rec["user_id"]
                # TwiBot-22 的 split 值是 train/val/test，统一成本项目的 train/dev/test
                split = splits.get(uid, "train")
                rec["split"] = {"val": "dev", "valid": "dev"}.get(split, split)
                rec["label"] = labels.get(uid)
                fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
                n_kept += 1
        tmp.replace(raw_jsonl)
        print(f"[collect] 已写入 label/split，共 {n_kept} 行 → {raw_jsonl}")

    if "encode" in args.steps:
        t0 = time.time()
        meta = encode_tweets(
            raw_jsonl, enc_dir, seq_len=args.seq_len, model_name=args.model,
            max_tokens=args.max_tokens, batch_size=args.batch_size,
        )
        print(f"[encode] 完成 ({time.time() - t0:.0f}s) {json.dumps(meta, ensure_ascii=False)}")

    if "micro" in args.steps:
        t0 = time.time()
        meta = build_micro_features(
            raw_jsonl, enc_dir, micro_dir, fit_split="train",
            with_temporal=args.with_temporal,
        )
        print(f"[micro] 完成 ({time.time() - t0:.0f}s) → {micro_dir}")
        print(f"        通道 ({meta['n_channels']}): {meta['channel_names']}")


if __name__ == "__main__":
    main()
