# AutoDL 执行清单

可直接复制粘贴。每步都会把日志落盘，跑完贴 `stage1_summary.log` 回来即可。

## 0. 一次性：上传代码与数据

```bash
WORK=/root/autodl-tmp/dtg_bot
mkdir -p $WORK/{code,data/twibot20}

# 代码（三种任选）
#   a) git clone <你的仓库> $WORK/code
#   b) 本地 rsync -avz --exclude cache --exclude .git dtg_bot/ root@<host>:$WORK/code/
#   c) AutoDL 网盘上传后解压到 $WORK/code
```

数据放置（若云上已有，只需软链接过来）：

```
$WORK/data/twibot20/{train,dev,test,support}.json

# 已有数据在别处时用软链，避免重复占盘：
# ln -s /root/autodl-tmp/twibot20/data/* $WORK/data/twibot20/
```

## 1. 环境准备（约 15 分钟）

```bash
cd $WORK/code
bash deploy/setup_autodl.sh $WORK 2>&1 | tee $WORK/logs/setup.log
```

**必须看到 `环境准备完成` 才继续。** 该脚本会前置检查依赖、GPU、
`roberta-base` 是否能从 hf-mirror 下载、数据文件是否齐全、磁盘是否够用 ——
目的是避免 12 小时的任务跑到一半才失败。

## 2. 阶段 1：全量编码 + 基线复现 + TwiBot-22 违例率

无人值守启动（GPU 任务串行，TwiBot-22 的 CPU 任务并行）：

```bash
cd $WORK/code
source $WORK/env.sh
nohup bash deploy/run_stage1.sh $WORK > /root/stage1.out 2>&1 &
echo "PID=$!"
```

监控：

```bash
tail -f $WORK/logs/stage1_summary.log     # 阶段摘要（贴回给我这个）
tail -f $WORK/logs/t20_prepare.log        # 编码进度条
tail -f $WORK/logs/t20_baseline.log       # 基线训练
tail -f $WORK/logs/t22_violation.log      # TwiBot-22 扫描
nvidia-smi -l 5                           # GPU 利用率
```

预计耗时（3090 参考）：

| 步骤 | 内容 | 耗时 |
|---|---|---|
| A1-nodes | 全节点文本 jsonl（4GB） | ~10 min |
| A1-graph | 建图，应得节点 229580 / 边 227979 | ~5 min |
| A1-desc | 229580 条 description 编码 | ~5 min |
| A1-pooled | 约 440 万条推文池化（cap=20） | ~1.7 h |
| A1-micro | 38 万条逐条编码 + 事件序列 | ~10 min |
| A2 | BotRGCN 基线 15 种子 | ~1 h |
| B | TwiBot-22 违例率（并行） | ~1.5–2.5 h |

**中断了直接重跑同一条命令**，编码步骤会从断点继续。

## 3. 关键判据

### 3.1 建图正确性（自动打印）

```
n_nodes = 229580      n_edges = 227979      n_edges_dropped = 0
split_sizes = {train: 8278, dev: 2365, test: 1183, support: 217754}
```

对不上说明数据文件有问题，**不要继续**。

### 3.2 基线复现（自动校验）

```
[BASELINE-PASS]      F1 ∈ [86.5, 87.6]   基线可信，进入阶段 2
[BASELINE-MARGINAL]  F1 ∈ [86.0, 86.5)   建议 POOL_CAP=50 复核
[BASELINE-FAIL]      F1 < 86.0           暂停，按日志提示排查
```

不达标时重跑并加大 pool-cap（已编码的 desc 会复用，只重跑 pooled）：

```bash
rm -f $WORK/cache/twibot20/features/tweet_pooled*     # 清掉旧池化向量
POOL_CAP=50 nohup bash deploy/run_stage1.sh $WORK > /root/stage1b.out 2>&1 &
```

### 3.3 TwiBot-22 顺序假设（自动判定）

```
[ORDER-PRESERVED]  违例率 < 0.1%   → TwiBot-20 的顺序空结果可解释为"顺序无信息"
[ORDER-MOSTLY]     < 5%            → 假设基本成立，论文需报告该比率
[ORDER-BROKEN]     ≥ 5%            → 论文 3.3/5.5 需降级表述
```

## 4. 阶段 2（基线达标后再执行）

```bash
source $WORK/env.sh && cd $WORK/code

# 单视图消融
for V in micro macro global; do
  python scripts/train.py --work-dir $WORK --model dtg --views $V --tag only_$V \
    >> $WORK/logs/stage2_views.log 2>&1
done

# 双视图 + 三视图
python scripts/train.py --work-dir $WORK --model dtg --views micro global --tag mi_gl \
  >> $WORK/logs/stage2_views.log 2>&1
python scripts/train.py --work-dir $WORK --model dtg --views macro global --tag ma_gl \
  >> $WORK/logs/stage2_views.log 2>&1
python scripts/train.py --work-dir $WORK --model dtg --views micro macro global --tag full \
  >> $WORK/logs/stage2_views.log 2>&1

# 融合方式消融
for F in attn gate concat; do
  python scripts/train.py --work-dir $WORK --model dtg --fusion $F --tag fusion_$F \
    >> $WORK/logs/stage2_fusion.log 2>&1
done

# 宏观分支必要性：快照演化 vs 静态全图
python scripts/train.py --work-dir $WORK --model dtg --macro-temporal last --tag macro_last \
  >> $WORK/logs/stage2_macro.log 2>&1

# 对比学习权重敏感性
for B in 0 0.05 0.1 0.3 1.0; do
  python scripts/train.py --work-dir $WORK --model dtg --beta $B --tag beta_$B \
    >> $WORK/logs/stage2_beta.log 2>&1
done
```

汇总：

```bash
python - <<'PY'
import pandas as pd, os
d = pd.read_csv(os.environ['DTG_WORK'] + '/experiments/results.csv')
d = d[~d.commit.str.endswith('-dirty')]
g = d.groupby('variant')[['f1','auc','accuracy']].agg(['mean','std','count'])
print((100*g.xs('mean',level=1,axis=1)).round(2).to_string())
PY
```


