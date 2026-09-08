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

最终结果以干净 commit `02f1db0`（`gate_final.log`）为准：

| 变体 | F1 | Acc | AUC |
|---|---|---|---|
| `keep` 正常时间正序 | 74.46 ± 0.89 | 70.32 ± 1.11 | 77.65 ± 0.72 |
| `shuffle` 每用户打乱 | 74.58 ± 0.78 | 70.13 ± 1.07 | 77.83 ± 0.48 |
| `reverse` 整体反序 | 74.43 ± 1.21 | 70.09 ± 1.53 | 77.59 ± 0.91 |
| `bag` 去注意力，仅统计池化 | 72.46 ± 0.58 | 67.57 ± 1.06 | 73.55 ± 0.69 |

配对 t 检验（keep vs 各对照，n = 15）：

```
keep vs shuffle    ΔF1 = -0.12  t=-0.369  p=0.718  |  ΔAUC = -0.18  t=-0.789  p=0.443   → 无差异
keep vs reverse    ΔF1 = +0.03  t=+0.122  p=0.905  |  ΔAUC = +0.06  t=+0.270  p=0.791   → 无差异
keep vs bag        ΔF1 = +2.01  t=+7.289  p<1e-5   |  ΔAUC = +4.10  t=+14.936 p<1e-5    → 极显著
```

**结论（论文叙事必须据此调整）：**

1. **推文列表顺序不携带判别信息。** 打乱顺序性能不变，AUC 甚至略升。
   "利用隐式先后顺序捕获短期语义矛盾"这一论点在 TwiBot-20 上**不成立**，不得写入论文。
2. **事件级注意力有极强价值。** 逐条推文特征 + 注意力聚合，相比统计池化 AUC +4.10。
   且该增益与顺序完全无关（shuffle 相对 bag 同样显著）。
   即：**哪几条推文异常**很重要，**它们何时发布**不重要。
3. 5 种子不足以判定 1 分以内的差异（5 种子时 keep vs shuffle 曾报 +0.59/p=0.088，
   15 种子后最终 ΔF1 = -0.12 / p = 0.718）。**所有关键对比一律 ≥15 种子 + 配对 t 检验。**

⚠️ **未排除的混淆**：上述空结果也可能是因为 TwiBot-20 的 `tweet` 列表在数据集构建时
就未保持时序（只在 TwiBot-22 上验证过 dump 是时间倒序，TwiBot-20 无时间戳可验证）。
若如此，`keep≈shuffle` 是预期结果、不能否证原假设。
**须在 TwiBot-22 上做 order-vs-Δt 实验来分辨这两种解释。**

## Stage-2 消融结论（2026-09，TwiBot-20，13 组变体 × 15 种子，commit 5ed258d）

BotRGCN 基线复现 **F1 87.72 ± 0.25 / AUC 93.12 ± 0.18**（高于官方 87.07，[BASELINE-PASS]）。

统计方法：逐种子**配对 t 检验**（n=15），5 个对比族共 56 项检验，
**族内 Holm-Bonferroni 校正**，判定一律以 p_holm 为准。复算脚本 `_paired_stats.py`。

**获得支持：**
1. 微观视图叠加到图视图上，AUC 一致显著提升，两条独立路径：
   global→micro+global +0.28 (p_holm=0.027)；macro→micro+macro +0.20 (p_holm=0.0031)。
   对应 F1 增益均**不显著**（+0.25 / +0.08）。
2. vs BotRGCN 的 AUC 提升：attn +0.28 (p_holm=0.0079)、gate +0.43 (0.0079)、
   concat +0.49 (3.2e-6)。
3. 仅全局视图 vs BotRGCN：ΔF1 -0.03 (p=0.82)、ΔAUC -0.04 (p=0.64)
   → **实现正确性检验通过**（全局分支确实等价于 BotRGCN 图通路）。

**未获得支持（负面结果，论文已如实报告）：**
4. **宏观快照演化无价值**：macro+global vs global ΔAUC **-0.17**（点估计为负）；
   GRU(K=8) vs last ΔAUC -0.10 (p=0.27)。但 **last vs BotRGCN ΔAUC +0.38
   (p_holm=6.2e-4)** → 宏观分支的有效成分是**第二条静态图通路，不是演化**。
   主因：边形成时间不可观测，`max(c_u,c_v)<=t_k` 只反映注册次序。
5. **attn 融合无优势**：concat/gate vs attn ΔF1 +0.39/+0.35 (p_holm=0.041)，
   但 AUC 上 concat vs attn 校正后**不显著**(p_holm=0.054)，且 concat vs gate 无差异。
   → 只能说"没有证据表明 attn 更优"，不能断言 concat 更优。
6. **InfoNCE 无增益**：AUC 随 λ 单调下降，λ=1.0 vs λ=0 ΔAUC -0.32 (p_holm=0.0052)；
   默认的 λ=0.3 也未优于 λ=0。剂量依赖负效应是"强制对齐有害"的最直接证据。

→ 统一解释：**视图判别力悬殊时（micro 的 F1 比 global 低 13 分），
强制对齐或加权求和会损害强视图。** 不要再往融合层加复杂度。

### ⚠️ F1 结论一律不可信，只报 AUC

- 全局 Holm（56 项合并）后仍显著的正向结论 11 项，**10 项是 AUC，只有 1 项是 F1**。
- `best_dev_f1` 核验：13 组变体全挤在 88.39~88.62（0.23 分区间），
  **concat 在验证集排第 10**，且 concat vs global **验证集显著更差**(Δ=-0.13, p=0.049)，
  而测试集显著更好(+0.52)。dev/test 方向不一致 → 按 test F1 选 concat 是选择偏差。
- 因此：**保留 attn 为默认配置**，concat/gate 只作并列报告，不得称"最优"。
- 缺口：训练只记了 `best_dev_f1`，**没记 dev AUC**，主结论无法在验证集上核验。
  后续跑实验务必把 dev AUC 一并写入 results.csv。

## TwiBot-22 列表有序性（已实测，commit 7e56228 修复后）

```
扫描推文 88,217,457   作者 933,872   相邻对 87,283,585
违例 2,875,870   违例率 3.29%   有违例的作者占比 57.40%
```

→ dump 近似时间倒序（96.71% 相邻对保序），但并非严格有序。
⚠️ **不能外推到 TwiBot-20**（不同采集流程、无时间戳），
所以 5.3 节的顺序空结果**仍未闭环**，还需 T22 上的 order-vs-Δt 增量实验。



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
  （keep vs shuffle ΔAUC=-0.18, p=0.443）。不要再基于"顺序"设计模型机制。
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
`entities`、`public_metrics`。**同一 author_id 的推文在 dump 中已按时间倒序排列**
（全量实测相邻对违例率 3.29%，见上文）。

⚠️ **T22 没有 support / 无标签子集**（已核验，不要再假设它像 T20 那样有）：

```
label.csv  1,000,000 行   human 860,057 / bot 139,943
split.csv  1,000,000 行   train 700,000 / val 200,000 / test 100,000
label ∩ split = 1,000,000    label-only = 0    split-only = 0
```

推论：`keep_users = set(labels)` **已经是全量 100 万**，
"只保留标注用户以减小规模"是空操作（上次扫描得到 93.4 万作者就是这个原因）。
要压缩规模只能**分层抽样**（`--sample-users`）。

⚠️ **bot 比例在各 split 间差异极大**：train 7.8% / val 28.0% / test 29.4%。
所以抽样必须按 `(split, label)` 分层，均匀随机抽样会改变正类比例、
使 F1 基线漂移，无法与 TwiBot-20 对照。

→ TwiBot-22 同时拥有真实时间戳和列表顺序，因此它是「仅用顺序 vs 用真实时间戳」
信息差实验的验证台，这是论文"无需精确时间戳"论点的正面证据来源。

⚠️ **id 形式在文件之间不一致**（已核验，踩过坑）：

```
label.csv / split.csv / user.json  →  "u1217628182611927040"   带 u 前缀
tweet_*.json 的 author_id          →  1304855289208819713      裸整数
tweet_*.json 的 id                 →  "t1497798545872588801"   带 t 前缀
```

未归一化时 `keep_users`（u 前缀）与 `author_id`（裸整数）永远匹配不上，
8.2M+ 条推文被全部过滤，统计为 0/0；而 **0/0 的违例率是 0.0，会通过
"< 0.1% 即保序"的判据**，打印出完全虚假的 `[ORDER-PRESERVED]` 结论
（该结论一度即将写入论文）。现已由 `bare_uid()` 归一化 + 零匹配硬失败守卫兜住。
**教训：任何"通过判据"的统计结论，必须先检查分母是否为 0。**

⚠️ **T22 全量违例率仍未测得。** 论文 3.4 节表 1 的"抽样观察为时间倒序"仅来自
人工抽样，全量违例率待 `prepare_twibot22.py --steps collect --stats-only` 重跑。
在此之前不得在论文中主张 T22 列表保序。

## 算力需求评估（Hardware Sizing）规则

交付任何 `run_stageX.sh` / 实验脚本前，必须在交付报告开头注明三行：

```
[推荐卡型]   ...
[预期显存占用] ...
[瓶颈类型]    GPU密集型 / CPU密集型 / IO密集型
```

分类规则：

**A. 纯微观 / 小样本实验（不跑全图快照）**
- 特征：仅评估文本/序列分支，显存 < 4GB；瓶颈在 CPU 端指标计算与数据搬运。
- 推荐：低成本实例（RTX 3060 / 2080Ti / 3080 等 12GB 卡）+ 高主频 CPU。
  用 4090 属算力浪费。

**B. 宏观快照 / 全图 GNN 实验**
- 特征：全图 N 节点消息传递、跨快照演化（K ≥ 4）或多视图联合优化，须配梯度检查点。
- 推荐：先测算显存理论峰值，明确标记必须 24GB 卡（RTX 3090 / 4090）；
  若涉 TwiBot-22 全图（百万级节点）则提示考虑 A100。

历史实测参照：Stage-2（T20 全图三视图 + 梯度检查点）在 3090（23.5GB）上通过；
未加检查点前曾在 backward 处 OOM（申请 9.10GB 时仅剩 4.87GB）。

## 环境

本地（仅开发调试，GTX 1650 / 4.3GB 显存，跑不了 TwiBot-22 全量）：

```
C:\Users\20996\.conda\envs\torch_pyg\python.exe    # torch 2.5.1, PyG 2.6.1, transformers 4.57.0
```

注意：`python` 不在 PATH（只有 Windows Store stub），必须用绝对路径。

⚠️ **绝对不要用 PowerShell 的 `Set-Content` / `Out-File` 改写含中文的源文件。**
它会按本地代码页有损重编码，把部分 UTF-8 字节替换成 `?`，文件将无法解析且无法还原
（已踩过：`prepare_twibot20_full.py` 被整体损坏，只能重写）。
改文件一律用编辑器工具或 Python 显式指定 `encoding='utf-8'`。

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
- ⚠️ **正式记录运行期间不要修改 `src/` `scripts/` `configs/`**。
  commit 标记是在每行结果写入时求值的，中途改代码会让同一次运行的前后行
  带上不同的 commit（已踩过：一次 60 行的运行里 12 行 clean、48 行 dirty）。
  等运行结束再改，或改文档（AGENTS.md / 论文）——文档不影响 dirty 判定。

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
