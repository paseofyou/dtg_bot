"""微观分支事件级注意力可视化。

用法：
    python scripts/visualize_attention.py --cache ./cache/twibot20 --seq-len 32 \
        --n-users 20 --top-k 5 --out experiments/attention_vis.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from diagnose_order import load_data, MicroOnlyClassifier  # noqa: E402
from dtg_bot.utils.metrics import compute_metrics  # noqa: E402
from dtg_bot.utils.seed import set_seed  # noqa: E402


def train_micro(feat, mask, labels, idx, meta, args, device):
    set_seed(args.seed)
    x = torch.from_numpy(feat).float().to(device)
    m = torch.from_numpy(mask).to(device)
    y = torch.from_numpy(labels).to(device)
    tr = torch.from_numpy(idx["train"]).to(device)
    dv = torch.from_numpy(idx["dev"]).to(device)

    model = MicroOnlyClassifier(
        in_channels=meta["n_channels"], seq_len=meta["seq_len"],
        d_model=args.d_model, n_layers=args.layers, n_heads=args.heads,
        dropout=args.dropout, seq_model="transformer", order_mode="keep",
        pool="attn",
    ).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    crit = nn.CrossEntropyLoss()

    best_dev, best_state, bad = -1.0, None, 0
    for epoch in range(args.epochs):
        model.train()
        perm = tr[torch.randperm(len(tr), device=device)]
        for start in range(0, len(perm), args.batch_size):
            b = perm[start:start + args.batch_size]
            opt.zero_grad()
            loss = crit(model(x[b], m[b]), y[b])
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

        model.eval()
        with torch.no_grad():
            dev_pred = model(x[dv], m[dv]).argmax(1)
        dev_acc = compute_metrics(y[dv].cpu().numpy(), dev_pred.cpu().numpy())["accuracy"]
        if dev_acc > best_dev:
            best_dev, bad = dev_acc, 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= args.patience:
                break

    model.load_state_dict(best_state)
    return model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default="./cache/twibot20")
    ap.add_argument("--dataset", default="TwiBot-20")
    ap.add_argument("--seq-len", type=int, default=32)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--patience", type=int, default=20)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight-decay", type=float, default=0.01)
    ap.add_argument("--d-model", type=int, default=64)
    ap.add_argument("--layers", type=int, default=2)
    ap.add_argument("--heads", type=int, default=4)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--n-users", type=int, default=20)
    ap.add_argument("--top-k", type=int, default=5)
    ap.add_argument("--out", default="./experiments/attention_vis.json")
    args = ap.parse_args()

    feat, mask, labels, idx, meta = load_data(Path(args.cache), args.seq_len)
    mean = np.array(meta["mean"])
    std = np.array(meta["std"])
    ch_names = list(meta["channel_names"])
    trans_idx = {c: ch_names.index(c) for c in ["cos_prev", "l2_prev", "jaccard_prev", "rt_run"]}
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"训练微观注意力模型（seed={args.seed}）...")
    model = train_micro(feat, mask, labels, idx, meta, args, device)

    te = idx["test"]
    te_has = mask[te].any(axis=1)
    sel = te[te_has][:args.n_users]

    x = torch.from_numpy(feat[sel]).float().to(device)
    m = torch.from_numpy(mask[sel]).to(device)
    y = labels[sel]

    model.eval()
    with torch.no_grad():
        logits = model(x, m)
        _, w = model.encoder(x, m, return_weights=True)
    preds = logits.argmax(1).cpu().numpy()
    w = w.cpu().numpy()

    jsonl = Path(args.cache) / f"tweets_L{args.seq_len}.jsonl"
    records = [json.loads(line) for line in Path(jsonl).read_text(encoding="utf-8").splitlines()]

    raw = feat * std + mean                 # 反标准化为原始事件通道
    raw_sel = raw[sel]

    def _event(pos_in_batch: int, i: int):
        return {
            "pos": int(i),
            "weight": float(w[pos_in_batch, i]),
            "text": rec["tweets"][i] if i < len(rec["tweets"]) else "",
            "cos_prev": float(raw_sel[pos_in_batch, i, trans_idx["cos_prev"]]),
            "l2_prev": float(raw_sel[pos_in_batch, i, trans_idx["l2_prev"]]),
            "jaccard": float(raw_sel[pos_in_batch, i, trans_idx["jaccard_prev"]]),
            "rt_run": float(raw_sel[pos_in_batch, i, trans_idx["rt_run"]]),
        }

    out = []
    for pos, uid in enumerate(sel):
        n_valid = int(mask[uid].sum())
        rec = records[uid]
        all_events = [_event(pos, i) for i in range(n_valid)]
        top_idx = np.argsort(w[pos, :n_valid])[-args.top_k:][::-1]
        top_k_events = [all_events[int(i)] for i in top_idx]
        out.append({
            "user_id": rec.get("user_id"),
            "split": rec.get("split"),
            "label": int(y[pos]),
            "pred": int(preds[pos]),
            "n_tweets": n_valid,
            "n_tweets_total": rec.get("n_tweets_total"),
            "top_k_events": top_k_events,
            "all_events": all_events,
        })

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"已保存 {len(out)} 个用户的注意力可视化 → {out_path}")


if __name__ == "__main__":
    main()
