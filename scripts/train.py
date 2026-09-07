"""训练与评估主脚本。BotRGCN 严格基线与 DTG-Bot 共用同一套数据与评估协议。

用法：
    # BotRGCN 基线复现（目标 TwiBot-20 test F1 ≈ 87.07）
    python scripts/train.py --work-dir $WORK --model botrgcn --seeds 42 123 456 789 2024

    # 本文模型（三视图）
    python scripts/train.py --work-dir $WORK --model dtg --views micro macro global

    # 单视图消融
    python scripts/train.py --work-dir $WORK --model dtg --views global --tag graph_only

评估协议（见 AGENTS.md）：按 dev F1 选检查点，test 只在最后评估一次；
关键对比用 15 个种子并做配对 t 检验。
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from dtg_bot.data.dataset import load_twibot20
from dtg_bot.models.dtg import DTGBot
from dtg_bot.models.graph import BotRGCN
from dtg_bot.utils.metrics import append_result, compute_metrics, summarize
from dtg_bot.utils.seed import set_seed


def build_model(args, data) -> nn.Module:
    num_prop_size = data["num_prop"].shape[1]
    cat_prop_size = data["cat_prop"].shape[1]
    if args.model == "botrgcn":
        return BotRGCN(
            num_prop_size=num_prop_size, cat_prop_size=cat_prop_size,
            emb=args.emb, num_relations=args.num_relations, dropout=args.dropout,
        )
    return DTGBot(
        micro_channels=data["micro_meta"]["n_channels"],
        micro_seq_len=data["micro_meta"]["seq_len"],
        num_prop_size=num_prop_size, cat_prop_size=cat_prop_size,
        emb=args.emb, num_relations=args.num_relations,
        num_snapshots=len(data["snapshot_masks"]), dropout=args.dropout,
        micro_dropout=args.micro_dropout, micro_layers=args.micro_layers,
        micro_heads=args.micro_heads, macro_temporal=args.macro_temporal,
        fusion=args.fusion, use_views=tuple(args.views),
        micro_seq_model=args.micro_seq_model, micro_order_mode=args.micro_order_mode,
        micro_use_position=args.micro_use_position,
        macro_checkpoint=not args.no_macro_checkpoint,
    )


def forward(model, args, data):
    """统一两种模型的调用方式。返回 (logits, views 或 None)。"""
    if args.model == "botrgcn":
        return model(data["des"], data["tweet"], data["num_prop"], data["cat_prop"],
                     data["edge_index"], data["edge_type"]), None
    return model(data, return_views=True)


def run_one_seed(args, data, seed: int, device: str) -> dict:
    set_seed(seed)
    model = build_model(args, data).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    crit = nn.CrossEntropyLoss()

    y = data["labels"]
    tr, dv, te = data["idx"]["train"], data["idx"]["dev"], data["idx"]["test"]

    # 对比损失的采样范围：train 内**有真实发帖事件**的节点。
    # 未标注节点本就不在 tr 中（其 z_micro 为补零行，绝不参与）；但零推文的标注
    # 用户其 z_micro 是 out_proj(0) 的同一常量向量，彼此逐元素恒等。若让它们进入
    # InfoNCE，非对角位置上会出现数值恒等的"负样本"，交叉熵强行推开恒等向量，
    # 梯度只能倾泻到 macro/global 分支，构成虚假的对齐压力。故一并排除。
    cl_idx = tr
    if "micro" in args.views and args.beta > 0 and "micro_mask" in data:
        has_event = data["micro_mask"][tr].any(dim=1)
        cl_idx = tr[has_event]
        n_drop = int(tr.numel() - cl_idx.numel())
        if n_drop:
            print(f"  [对比损失] train 内零推文用户 {n_drop} 个已从 InfoNCE 采样中排除",
                  flush=True)

    best_dev, best_state, bad = -1.0, None, 0
    for epoch in range(args.epochs):
        model.train()
        opt.zero_grad()
        logits, views = forward(model, args, data)
        loss = crit(logits[tr], y[tr])
        if views is not None and args.beta > 0 and len(views) > 1:
            loss = loss + args.beta * DTGBot.contrastive_loss(
                {k: v[cl_idx] for k, v in views.items()},
                temperature=args.temperature, max_samples=args.cl_max_samples,
            )
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        opt.step()
        # 训练前向的 logits/views 在紧随其后的评估前向期间仍被引用（三视图各 (N,emb)），
        # 显式释放以免两次前向的峰值叠加。
        del logits, views, loss

        model.eval()
        with torch.no_grad():
            logits, _ = forward(model, args, data)
            dev_f1 = compute_metrics(
                y[dv].cpu().numpy(), logits[dv].argmax(1).cpu().numpy()
            )["f1"]
        if dev_f1 > best_dev:
            best_dev, bad = dev_f1, 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= args.patience:
                break

    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        logits, _ = forward(model, args, data)
        prob = torch.softmax(logits[te], dim=1)[:, 1]
    res = compute_metrics(y[te].cpu().numpy(), logits[te].argmax(1).cpu().numpy(),
                          prob.cpu().numpy())
    res["best_dev_f1"] = best_dev
    res["epochs_run"] = epoch + 1
    return res


#: BotRGCN 官方报告值，用于自动校验基线复现质量（见 AGENTS.md 核心原则 5）
OFFICIAL_BASELINE = {"TwiBot-20": {"f1": 0.8707, "accuracy": 0.8575}}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--work-dir", required=True)
    ap.add_argument("--cache", default=None)
    ap.add_argument("--dataset", default="TwiBot-20")
    ap.add_argument("--model", default="dtg", choices=["botrgcn", "dtg"])
    ap.add_argument("--tag", default="", help="结果表里的变体名，默认自动生成")
    ap.add_argument("--views", nargs="+", default=["micro", "macro", "global"])
    ap.add_argument("--fusion", default="attn", choices=["attn", "gate", "concat"])
    ap.add_argument("--macro-temporal", default="gru", choices=["gru", "transformer", "last"])
    ap.add_argument("--micro-seq-model", default="transformer", choices=["transformer", "bag"])
    ap.add_argument("--micro-order-mode", default="keep", choices=["keep", "shuffle", "reverse"])
    ap.add_argument("--micro-use-position", action="store_true")
    ap.add_argument("--no-macro-checkpoint", action="store_true",
                    help="关闭宏观分支的梯度检查点（省一次重算但显存翻 K 倍，24GB 卡会 OOM）")
    ap.add_argument("--seq-len", type=int, default=32)
    ap.add_argument("--num-snapshots", type=int, default=8)
    ap.add_argument("--emb", type=int, default=64)
    ap.add_argument("--num-relations", type=int, default=2)
    ap.add_argument("--dropout", type=float, default=0.3)
    ap.add_argument("--micro-dropout", type=float, default=0.1)
    ap.add_argument("--micro-layers", type=int, default=2)
    ap.add_argument("--micro-heads", type=int, default=4)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight-decay", type=float, default=5e-3)
    ap.add_argument("--beta", "--lambda", dest="beta", type=float, default=0.0,
                    help="对比损失权重（论文记号 λ；--beta 与 --lambda 等价）")
    ap.add_argument("--temperature", type=float, default=0.5)
    ap.add_argument("--cl-max-samples", type=int, default=4096)
    ap.add_argument("--epochs", type=int, default=300)
    ap.add_argument("--patience", type=int, default=40)
    ap.add_argument("--grad-clip", type=float, default=1.0)
    ap.add_argument("--seeds", nargs="+", type=int,
                    default=[42, 123, 456, 789, 2024, 7, 13, 99, 2025, 314,
                             1618, 271, 577, 999, 8128])
    ap.add_argument("--results", default=None)
    args = ap.parse_args()

    work = Path(args.work_dir)
    cache = Path(args.cache) if args.cache else work / "cache" / "twibot20"
    results = Path(args.results) if args.results else work / "experiments" / "results.csv"
    device = "cuda" if torch.cuda.is_available() else "cpu"

    tag = args.tag or (
        "botrgcn" if args.model == "botrgcn"
        else f"{'+'.join(args.views)}_{args.fusion}"
        + (f"_beta{args.beta}" if args.beta else "")
    )
    print(f"model={args.model}  tag={tag}  device={device}", flush=True)

    t0 = time.time()
    data = load_twibot20(cache, seq_len=args.seq_len, num_snapshots=args.num_snapshots,
                         device=device, need_micro=(args.model != "botrgcn"))
    print(f"数据装载完成 ({time.time() - t0:.0f}s) 节点 {data['n_nodes']} "
          f"边 {data['edge_index'].shape[1]} "
          f"train/dev/test = {len(data['idx']['train'])}/"
          f"{len(data['idx']['dev'])}/{len(data['idx']['test'])}", flush=True)

    rows = []
    for seed in args.seeds:
        t0 = time.time()
        res = run_one_seed(args, data, seed, device)
        rows.append(res)
        append_result(results, {
            "dataset": args.dataset, "experiment": "main", "variant": tag,
            "model": args.model, "views": "+".join(args.views), "fusion": args.fusion,
            "macro_temporal": args.macro_temporal, "micro_seq_model": args.micro_seq_model,
            "micro_order_mode": args.micro_order_mode, "beta": args.beta,
            "seed": seed, "seq_len": args.seq_len, "emb": args.emb, "lr": args.lr,
            "weight_decay": args.weight_decay, **res,
        })
        print(f"  seed={seed:<5} f1={res['f1']:.4f} acc={res['accuracy']:.4f} "
              f"auc={res['auc']:.4f} ep={res['epochs_run']:<4} ({time.time() - t0:.0f}s)",
              flush=True)

    s = summarize(rows)
    print("\n" + "=" * 70)
    print(f"{tag}  (n={len(args.seeds)} seeds)")
    for k, v in s.items():
        print(f"  {k:<10} {v}")

    # --- 基线复现质量自动校验 ---
    ref = OFFICIAL_BASELINE.get(args.dataset)
    if args.model == "botrgcn" and ref:
        f1 = float(np.mean([r["f1"] for r in rows]))
        delta = 100 * (f1 - ref["f1"])
        print("\n" + "-" * 70)
        print(f"基线复现校验：本文 F1 {100 * f1:.2f} vs 官方 {100 * ref['f1']:.2f} "
              f"(Δ={delta:+.2f})")
        if f1 >= ref["f1"] - 0.006:
            print("  [BASELINE-PASS] 落在 86.5~87.6 区间，基线可信，可继续后续实验")
        elif f1 >= ref["f1"] - 0.011:
            print("  [BASELINE-MARGINAL] 偏低但可接受，建议加大 --pool-cap 后复核")
        else:
            print("  [BASELINE-FAIL] 低于 86.0。按顺序排查：")
            print("     1) --pool-cap 加大到 50 或 0")
            print("     2) 检查 num_prop 归一化（log1p + z-score，仅用 train 拟合）")
            print("     3) 超参：--lr 1e-3/1e-2, --weight-decay 5e-3/5e-2, --emb 64/128")
            print("     4) 检查边构造：relation 数、是否漏掉 support 端点")
            print("     基线不达标前不要采信任何'超过基线'的结论")
    print("=" * 70, flush=True)


if __name__ == "__main__":
    main()
