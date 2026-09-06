"""评估指标与结果记录。"""

from __future__ import annotations

import csv
import subprocess
from datetime import datetime
from pathlib import Path

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)

METRIC_NAMES = ("accuracy", "precision", "recall", "f1", "mcc", "auc")


def compute_metrics(y_true, y_pred, y_score=None) -> dict[str, float]:
    out = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "mcc": matthews_corrcoef(y_true, y_pred),
    }
    out["auc"] = roc_auc_score(y_true, y_score) if y_score is not None else float("nan")
    return {k: float(v) for k, v in out.items()}


def git_commit() -> str:
    """返回当前 commit，工作区有未提交改动时加 -dirty 后缀。

    带 -dirty 的结果不可复现，不得写入论文（见 AGENTS.md 实验协议）。
    """
    try:
        rev = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], stderr=subprocess.DEVNULL, text=True
        ).strip()
        dirty = subprocess.check_output(
            ["git", "status", "--porcelain"], stderr=subprocess.DEVNULL, text=True
        ).strip()
        return f"{rev}-dirty" if dirty else rev
    except Exception:
        return "nogit"


def append_result(csv_path: str | Path, row: dict) -> None:
    csv_path = Path(csv_path)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    row = {"time": datetime.now().isoformat(timespec="seconds"), "commit": git_commit(), **row}
    exists = csv_path.exists()
    with csv_path.open("a", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(row.keys()))
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def summarize(rows: list[dict], keys=METRIC_NAMES) -> dict[str, str]:
    """把多个种子的结果汇总成 mean±std。"""
    out = {}
    for k in keys:
        vals = np.array([r[k] for r in rows if k in r], dtype=float)
        if len(vals):
            out[k] = f"{100 * vals.mean():.2f} ± {100 * vals.std(ddof=1 if len(vals) > 1 else 0):.2f}"
    return out
