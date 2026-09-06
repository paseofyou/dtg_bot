"""【关口实验】顺序诊断：推文的隐式先后顺序是否真的携带判别信息？

论文的整条微观分支押在"用户推文列表顺序 = 真实时间顺序，且该顺序含判别信息"上。
本脚本用**仅微观分支**的分类器（不含图、不含静态元数据）做四组对照：

    transformer + keep      正常时间正序
    transformer + shuffle   每用户独立打乱有效位  ← 若与 keep 持平，顺序无信息
    transformer + reverse   整体反序             ← 检验方向性
    bag                     去掉自注意力与位置编码，仅 mean/std 池化 → MLP
                                                 ← 若与 keep 持平，序列建模无价值

判定标准（必须在动手做融合模块之前得到结论）：
    keep 显著优于 shuffle 且显著优于 bag  → 顺序建模成立，继续
    keep ≈ shuffle                        → 顺序无信息，微观分支的叙事不成立，停下换方向
    keep ≈ bag 但优于 shuffle             → 顺序有信息但无需序列模型，应简化为统计特征

用法：
    python scripts/diagnose_order.py --cache ./cache/twibot20 --seq-len 32
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from dtg_bot.models.micro import EventAttentionEncoder
from dtg_bot.utils.metrics import append_result, compute_metrics, summarize
from dtg_bot.utils.seed import set_seed


class MicroOnlyClassifier(nn.Module):
    """仅用微观事件序列的分类器，用于隔离测量顺序信息量。

    ⚠️ ``use_position=True`` 是**本诊断实验的必要条件**：
    Transformer 自注意力与 attention pooling 都是置换等变的，
    若不加位置编码，整个编码器对输入顺序完全不变，
    shuffle 在构造上就不可能改变任何结果 —— 顺序检验会退化成空检验。
    只有加上位置编码，模型才**有能力**利用顺序，此时 shuffle 不掉点
    才能说明"顺序确实不含判别信息"。
    """

    def __init__(self, in_channels: int, seq_len: int, d_model=64, out_dim=64,
                 n_layers=2, n_heads=4, dropout=0.1, seq_model="transformer",
                 order_mode="keep"):
        super().__init__()
        self.encoder = EventAttentionEncoder(
            in_channels=in_channels, d_model=d_model, out_dim=out_dim,
            n_layers=n_layers, n_heads=n_heads, max_len=seq_len,
            dropout=dropout, seq_model=seq_model, order_mode=order_mode,
            use_position=True,
        )
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(out_dim, 2))

    def forward(self, x, mask):
        return self.head(self.encoder(x, mask))


def load_data(cache: Path, seq_len: int):
    micro_dir = cache / f"micro_L{seq_len}"
    enc_dir = cache / f"encoded_L{seq_len}"
    feat = np.load(micro_dir / "micro_feat.npy")
    mask = np.load(micro_dir / "micro_mask.npy")
    labels = np.load(enc_dir / "labels.npy").astype(np.int64)
    splits = np.load(enc_dir / "splits.npy")
    meta = json.loads((micro_dir / "micro_meta.json").read_text(encoding="utf-8"))
    idx = {s: np.where(splits == s)[0] for s in ("train", "dev", "test")}
    return feat, mask, labels, idx, meta


def run_one(feat, mask, labels, idx, meta, variant: dict, seed: int, args, device) -> dict:
    set_seed(seed)
    x = torch.from_numpy(feat).float().to(device)
    m = torch.from_numpy(mask).to(device)
    y = torch.from_numpy(labels).to(device)

    model = MicroOnlyClassifier(
        in_channels=meta["n_channels"], seq_len=meta["seq_len"],
        d_model=args.d_model, n_layers=args.layers, n_heads=args.heads,
        dropout=args.dropout, seq_model=variant["seq_model"],
        order_mode=variant["order_mode"],
    ).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    crit = nn.CrossEntropyLoss()

    tr = torch.from_numpy(idx["train"]).to(device)
    dv = torch.from_numpy(idx["dev"]).to(device)
    te = torch.from_numpy(idx["test"]).to(device)

    best_dev, best_state, bad = -1.0, None, 0
    for epoch in range(args.epochs):
        model.train()
        perm = tr[torch.randperm(len(tr), device=device)]
        for start in range(0, len(perm), args.batch_size):
            b = perm[start : start + args.batch_size]
            opt.zero_grad()
            loss = crit(model(x[b], m[b]), y[b])
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

        model.eval()
        with torch.no_grad():
            dev_pred = model(x[dv], m[dv]).argmax(1)
        dev_f1 = compute_metrics(y[dv].cpu().numpy(), dev_pred.cpu().numpy())["f1"]
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
        logits = model(x[te], m[te])
        prob = torch.softmax(logits, dim=1)[:, 1]
    res = compute_metrics(y[te].cpu().numpy(), logits.argmax(1).cpu().numpy(), prob.cpu().numpy())
    res["best_dev_f1"] = best_dev
    return res


VARIANTS = {
    "keep":     {"seq_model": "transformer", "order_mode": "keep"},
    "shuffle":  {"seq_model": "transformer", "order_mode": "shuffle"},
    "reverse":  {"seq_model": "transformer", "order_mode": "reverse"},
    "bag":      {"seq_model": "bag",         "order_mode": "keep"},
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default="./cache/twibot20")
    ap.add_argument("--dataset", default="TwiBot-20")
    ap.add_argument("--seq-len", type=int, default=32)
    ap.add_argument("--variants", nargs="+", default=list(VARIANTS))
    ap.add_argument("--seeds", nargs="+", type=int, default=[42, 123, 456, 789, 2024])
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--patience", type=int, default=20)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight-decay", type=float, default=0.01)
    ap.add_argument("--d-model", type=int, default=64)
    ap.add_argument("--layers", type=int, default=2)
    ap.add_argument("--heads", type=int, default=4)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--results", default="./experiments/results.csv")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    feat, mask, labels, idx, meta = load_data(Path(args.cache), args.seq_len)
    print(f"通道 {meta['n_channels']}  L={meta['seq_len']}  "
          f"train/dev/test = {len(idx['train'])}/{len(idx['dev'])}/{len(idx['test'])}  device={device}")

    collected: dict[str, list[dict]] = {}
    for name in args.variants:
        variant = VARIANTS[name]
        rows = []
        for seed in args.seeds:
            res = run_one(feat, mask, labels, idx, meta, variant, seed, args, device)
            rows.append(res)
            append_result(args.results, {
                "dataset": args.dataset, "experiment": "diagnose_order",
                "variant": name, "seq_model": variant["seq_model"],
                "order_mode": variant["order_mode"], "seed": seed,
                "seq_len": meta["seq_len"], **res,
            })
            print(f"  {name:<9} seed={seed:<5} f1={res['f1']:.4f} acc={res['accuracy']:.4f} auc={res['auc']:.4f}")
        collected[name] = rows
        print(f"  {name:<9} → {summarize(rows)}")
        print()

    print("=" * 72)
    print("汇总 (mean ± std, %)")
    for name, rows in collected.items():
        s = summarize(rows)
        print(f"  {name:<9} F1 {s['f1']:<16} Acc {s['accuracy']:<16} AUC {s.get('auc','-')}")

    if "keep" in collected:
        print(f"\n配对 t 检验 (keep vs 对照, n={len(args.seeds)} seeds)")
        for name in ("shuffle", "reverse", "bag"):
            if name not in collected:
                continue
            parts = []
            for metric in ("f1", "auc"):
                base = [r[metric] for r in collected["keep"]]
                other = [r[metric] for r in collected[name]]
                t, p = stats.ttest_rel(base, other)
                star = "*" if p < 0.05 else " "
                parts.append(
                    f"Δ{metric.upper()}={100 * (np.mean(base) - np.mean(other)):+.2f} "
                    f"t={t:+.3f} p={p:.4f}{star}"
                )
            print(f"  keep vs {name:<8} " + " | ".join(parts))
        print("\n判定：若 keep vs shuffle 不显著 → 顺序无判别信息；"
              "若 keep vs bag 显著 → 事件级注意力有效。")


if __name__ == "__main__":
    main()
