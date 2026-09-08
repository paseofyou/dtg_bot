# 论文归档

本目录为 `论文/` 工作区的安全归档副本（`论文/` 目录未纳入任何 Git 仓库，
为避免丢失按用户指示归档于此）。**原始工作副本仍在 `D:\project\trace-小论文\论文\`，
后续修改请改那里，再同步覆盖本目录。**

- `初稿最新版.md` —— 定稿（截至 commit 26d105c 之后）
- 图 1：内嵌 ASCII 框架图（4.1 节）
- 图 2/图 3：`../../figures/stage2_ablation_bars.pdf`、
  `../../figures/stage2_lambda_auc.pdf`（相对路径已按本目录位置改写）

## 数据-结论追溯（所有关键数字均可复算）

| 论文结论 | 数据来源 | 分析脚本 |
|---|---|---|
| 表 4/5 T20 顺序诊断 | `results.csv`（diagnose_order） | `scripts/diagnose_order.py` |
| 表 6 T22 有序性 | collect meta（stats-only 全量扫描） | `scripts/prepare_twibot22.py` |
| 表 7/8 T22 order-vs-Δt | `experiments/results_t22_dt.csv` | `scripts/analyze_stage3.py` |
| 表 2,3,9,10,11 T20 主实验+消融 | `experiments/stage{1,2}_*_seedwise.csv` | `scripts/paired_stats.py` |
| 图 2/图 3 | 同上逐种子数据 | `scripts/plot_ablation.py` |
