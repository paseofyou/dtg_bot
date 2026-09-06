# DTG-Bot 项目说明

融合宏微观双时序与图结构的社交机器人检测。**全新项目，不复用 `../my_model` 的任何代码。**

对应论文：`../论文/初稿最新版.md`

## 核心设计原则（这些是上一版失败的直接教训，不要违反）

1. **禁止合成时序数据。** 上一版用 `np.random` 从静态特征生成"伪时间序列"当时序证据，
   这是研究诚信红线。本项目所有时序信号必须可追溯到原始数据中的真实字段
   （推文列表顺序、`created_at`、TwiBot-22 的推文 `created_at`）。
2. **时序特征不能是累计计数。** 累计计数序列的信息等价于静态元数据的单调变换，
   会被 `flat_static` 对照打平。本项目微观分支建模的是**相邻发帖事件之间的转移量**
   （语义跳变、模板重叠），这类量无法从静态快照恢复。
3. **先跑对照，再堆模块。** `shuffle` 诊断（打乱每用户序列顺序）是全文生命线。
   若 shuffle 后性能不掉，说明模型没用到顺序，立即停止并换方向，
   不要继续叠加融合层和对比学习。
4. **时序编码器必须端到端训练。** 预计算成固定嵌入会让时序退化为静态特征。
   这是为什么微观特征要压到低维（`(L-1)×14`）——为了能全量驻留显存联合训练。
5. **baseline 必须复现到官方水平**才有资格做对比：
   BotRGCN TwiBot-20 F1 = 87.07，TwiBot-22 F1 = 57.50。
   上一版复现的 T22 baseline 只有 53.60（低 4 分），导致"超过 baseline"是弱基线假象。

## 关口实验结论（2026-09，TwiBot-20，15 种子，`scripts/diagnose_order.py`）

仅微观分支的分类器（无图、无静态元数据），test split：

| 变体 | F1 | Acc | AUC |
|---|---|---|---|
| `keep` 正常时间正序 | 74.56 ± 0.76 | 70.37 ± 1.20 | 77.56 ± 0.57 |
| `shuffle` 每用户打乱 | 74.50 ± 0.90 | 70.25 ± 1.03 | 77.65 ± 0.43 |
| `reverse` 整体反序 | 74.31 ± 0.67 | 70.07 ± 0.88 | 77.31 ± 0.76 |
| `bag` 去注意力，仅统计池化 | 72.46 ± 0.58 | 67.57 ± 1.06 | 73.55 ± 0.69 |

配对 t 检验：

```
keep vs shuffle    ΔF1 = +0.05  p=0.877  |  ΔAUC = -0.09  p=0.644   → 无差异
keep vs bag        ΔF1 = +2.10  p<1e-5   |  ΔAUC = +4.01  p<1e-5    → 极显著
shuffle vs bag     ΔF1 = +2.05  p<1e-5   |  ΔAUC = +4.10  p<1e-5    → 同等显著
```

**结论（论文叙事必须据此调整）：**

1. **推文列表顺序不携带判别信息。** 打乱顺序性能不变，AUC 甚至略升。
   "利用隐式先后顺序捕获短期语义矛盾"这一论点在 TwiBot-20 上**不成立**，不得写入论文。
2. **事件级注意力有极强价值。** 逐条推文特征 + 注意力聚合，相比统计池化 AUC +4.01。
   且该增益与顺序完全无关（shuffle 同样拿到 +4.10）。
   即：**哪几条推文异常**很重要，**它们何时发布**不重要。
3. 5 种子不足以判定 1 分以内的差异（5 种子时 keep vs shuffle 曾报 +0.59/p=0.088，
   15 种子后归零）。**所有关键对比一律 ≥15 种子 + 配对 t 检验。**

⚠️ **未排除的混淆**：上述空结果也可能是因为 TwiBot-20 的 `tweet` 列表在数据集构建时
就未保持时序（只在 TwiBot-22 上验证过 dump 是时间倒序，TwiBot-20 无时间戳可验证）。
若如此，`keep≈shuffle` 是预期结果、不能否证原假设。
**须在 TwiBot-22 上做 order-vs-Δt 实验来分辨这两种解释。**

## 数据事实（已逐项核验，不要再猜）

### TwiBot-20 — 本地 `D:\project\dataset\TwiBot-20\raw\`

`{train,dev,test,support}.json`，每条记录结构：

```
{ "ID": str,
  "profile": {...},          # 含 created_at ("Mon Oct 28 13:39:41 +0000 2019")，字段值尾部有多余空格，必须 strip
  "tweet": [str, ...] | null,# 推文原文列表，无时间戳；顺序 = Twitter timeline 倒序（新→旧）
  "neighbor": {"following": [id...], "follower": [id...]} | null,
  "domain": [str...],
  "label": "0" | "1" }
```

- **推文无时间戳，列表顺序被假定为 timeline 倒序**（新→旧），
  使用时 reverse 成时间正序（旧→新）。该假设的依据仅是 TwiBot-22 dump 呈时间倒序，
  TwiBot-20 本身**无法自证**。
  ⚠️ 且关口实验已证明：无论该假设是否成立，**顺序都不带来性能增益**
  （keep vs shuffle ΔAUC=-0.09, p=0.644）。不要再基于"顺序"设计模型机制。
- `profile.created_at` 是真实账号创建时间 → 宏观动态图快照掩码的唯一真实锚点。
- **单快照无法重构用户自身属性的演化曲线**（上一版就是在这里被迫造假）。
  宏观分支只能建模「邻域结构随 created_at 的真实演化」，不要写成"用户属性演化"。
- 官方划分 = 四个 json 文件本身；节点顺序约定为 train → dev → test → support。
- 已实测统计：
  - 标注用户共 **11826**（train 8278 / dev 2365 / test 1183）；test split 640 bot / 543 human。
  - 推文数中位数 200（上限 200），均值约 169；**test split 有 10 个零推文用户**，
    micro 分支必须处理全 padding 行（`AttentionPooling` 与 `TransformerEncoder` 都会出 NaN）。
  - test split 有 109 个用户完全没有 neighbor 信息。
  - **support 用户也有推文**（抽样 3000 个，均值 136.5，12.9% 零推文），
    所以忠实复现 BotRGCN 需要为 217754 个 support 节点也编码推文 → 约 650 万条，
    本地 GTX 1650 不可行（约 48h），**必须在云上做**。
  - support 用户中仅约 **0.5%** 带 neighbor 列表，图的边基本由标注用户的
    following/follower 列表定义。
- 也存在 format22 版本 `D:\project\dataset\TwiBot-20\Twibot-20-format22\`（node.json / edge.csv），
  本项目以 `raw/*.json` 为准，因为只有它保留了 `tweet` 的原始列表顺序。

### TwiBot-22 — 本地 `D:\dataset\raw_data\TwiBot_22\`，云上已有副本

```
user.json 746MB, label.csv, split.csv, edge.csv 6.3GB,
tweet_0..8.json 共 ~101GB, hashtag.json, list.json
```

推文对象含真实 `created_at`（`"2022-02-27 04:59:35+00:00"`）、`author_id`、`source`、
`entities`、`public_metrics`。**同一 author_id 的推文在 dump 中已按时间倒序排列。**

→ TwiBot-22 同时拥有真实时间戳和列表顺序，因此它是「仅用顺序 vs 用真实时间戳」
信息差实验的验证台，这是论文"无需精确时间戳"论点的正面证据来源。

## 环境

本地（仅开发调试，GTX 1650 / 4.3GB 显存，跑不了 TwiBot-22 全量）：

```
C:\Users\20996\.conda\envs\torch_pyg\python.exe    # torch 2.5.1, PyG 2.6.1, transformers 4.57.0
```

注意：`python` 不在 PATH（只有 Windows Store stub），必须用绝对路径。

云端：AutoDL，约定 `--work-dir /root/autodl-tmp/dtg_bot`，数据在 `$WORK/data/{twibot20,twibot22}`。
**所有脚本路径必须通过 `--work-dir` / config 参数化，禁止硬编码本地路径。**

### RoBERTa 编码吞吐（实测，用于云端排期）

GTX 1650 上 roberta-base fp16、batch 64、max_tokens 64 → **约 37 条推文/秒**
（该卡 TU117 无 tensor core，fp16 不提速；瓶颈在 GPU 而非 IO）。

| 任务 | 推文量 | GTX 1650 | 预计 3090（约 20x） |
|---|---|---|---|
| TwiBot-20 标注用户 (L=32) | 35 万 | 2.6 h | ~8 min |
| TwiBot-20 全节点 (含 support) | 约 700 万 | 约 52 h（不可行） | ~2.6 h |
| TwiBot-22 (100 万用户, L=16) | 约 1600 万 | 约 120 h（不可行） | ~6 h |

`encode.py` 已做长度分桶 + 连续写盘优化；云端应把 `--batch-size` 提到 256~512。

## 实验协议

- **关键对比一律 15 个种子**（`42 123 456 789 2024 7 13 99 2025 314 1618 271 577 999 8128`）。
  实测 5 个种子无法判定 1 分以内的差异，会产出假阳性（见上文关口实验第 3 条）。
  仅粗筛超参时可用 5 个种子，且结论不得写入论文。
- 同一组超参用于所有变体
- 按 val F1 选 checkpoint，test 只在最后评估一次
- 报 Accuracy / Precision / Recall / F1 / MCC / AUC 的 mean±std，
  并对关键对比做**配对 t 检验**。AUC 对阈值不敏感，比 F1 更稳，应同时报告。
- 每次运行追加一行到 `experiments/results.csv`，记录 git commit；dirty commit 的结果不得写入论文

## 目录约定

```
configs/            数据集与模型配置 (yaml)
src/dtg_bot/
  data/             twibot20.py twibot22.py encode.py micro.py macro.py
  models/           micro.py macro.py graph.py fusion.py dtg.py
  utils/            seed / metrics / results
scripts/            prepare_*.py, diagnose_order.py, train.py
experiments/        results.csv
```
