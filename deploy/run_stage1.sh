#!/usr/bin/env bash
# 阶段 1：TwiBot-20 全量编码 + BotRGCN 基线复现（GPU 任务）
#         并行启动 TwiBot-22 违例率统计（CPU/IO 任务）
#
# 无人值守设计：
#   - 全部输出重定向到 $WORK/logs/*.log
#   - 每步结束打印状态摘要，可直接抓日志贴回
#   - 编码步骤支持断点续跑：重复执行本脚本即可从中断处继续
#   - GPU 任务串行、CPU 任务并行，避免互相拖慢
#
# 用法：
#   source /root/autodl-tmp/dtg_bot/env.sh
#   nohup bash deploy/run_stage1.sh /root/autodl-tmp/dtg_bot > /root/stage1.out 2>&1 &
#   tail -f /root/autodl-tmp/dtg_bot/logs/stage1_summary.log

set -uo pipefail

WORK="${1:-${DTG_WORK:-/root/autodl-tmp/dtg_bot}}"
CODE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LOGS="$WORK/logs"
mkdir -p "$LOGS"

POOL_CAP="${POOL_CAP:-20}"
BATCH_SIZE="${BATCH_SIZE:-256}"
SEQ_LEN="${SEQ_LEN:-32}"
SEEDS="${SEEDS:-42 123 456 789 2024 7 13 99 2025 314 1618 271 577 999 8128}"
RUN_T22="${RUN_T22:-1}"

SUMMARY="$LOGS/stage1_summary.log"
: > "$SUMMARY"

log() { echo "[$(date '+%F %T')] $*" | tee -a "$SUMMARY"; }

log "=============================================================="
log " DTG-Bot 阶段 1"
log " WORK=$WORK  POOL_CAP=$POOL_CAP  BATCH_SIZE=$BATCH_SIZE  L=$SEQ_LEN"
log "=============================================================="

cd "$CODE_DIR"
export PYTHONPATH="$CODE_DIR/src:${PYTHONPATH:-}"

# ---------------------------------------------------------------
# 任务 B（并行，CPU/IO）：TwiBot-22 时间戳违例率
# ---------------------------------------------------------------
T22_PID=""
if [ "$RUN_T22" = "1" ] && [ -d "$WORK/data/twibot22" ]; then
  log "[B] 后台启动 TwiBot-22 违例率统计 (stats-only) → $LOGS/t22_violation.log"
  nohup python scripts/prepare_twibot22.py \
      --work-dir "$WORK" --steps collect --stats-only --seq-len 16 \
      > "$LOGS/t22_violation.log" 2>&1 &
  T22_PID=$!
  log "[B] PID=$T22_PID （预计 1.5~2.5h，与 GPU 任务并行）"
else
  log "[B] 跳过 TwiBot-22（未找到数据或 RUN_T22=0）"
fi

# ---------------------------------------------------------------
# 任务 A1（GPU）：TwiBot-20 全量预处理
# ---------------------------------------------------------------
log "[A1] TwiBot-20 全量预处理开始 → $LOGS/t20_prepare.log"
python scripts/prepare_twibot20_full.py \
    --work-dir "$WORK" \
    --steps nodes graph desc pooled micro \
    --seq-len "$SEQ_LEN" --pool-cap "$POOL_CAP" --batch-size "$BATCH_SIZE" \
    > "$LOGS/t20_prepare.log" 2>&1
A1_RC=$?
if [ $A1_RC -ne 0 ]; then
  log "[A1] ❌ 失败 (rc=$A1_RC)。日志末尾："
  tail -30 "$LOGS/t20_prepare.log" | tee -a "$SUMMARY"
  log "[A1] 预处理支持断点续跑，修复后重新执行本脚本即可从中断处继续。"
  exit $A1_RC
fi
log "[A1] ✅ 完成。摘要："
sed -n '/阶段摘要/,$p' "$LOGS/t20_prepare.log" | tee -a "$SUMMARY"

# ---------------------------------------------------------------
# 任务 A2（GPU）：BotRGCN 基线复现 —— 成败关口
# ---------------------------------------------------------------
log "[A2] BotRGCN 基线复现开始（目标 F1 ≈ 87.07）→ $LOGS/t20_baseline.log"
python scripts/train.py \
    --work-dir "$WORK" --model botrgcn --tag botrgcn_cap${POOL_CAP} \
    --seeds $SEEDS \
    > "$LOGS/t20_baseline.log" 2>&1
A2_RC=$?
if [ $A2_RC -ne 0 ]; then
  log "[A2] ❌ 失败 (rc=$A2_RC)。日志末尾："
  tail -40 "$LOGS/t20_baseline.log" | tee -a "$SUMMARY"
  exit $A2_RC
fi
log "[A2] 结果："
tail -25 "$LOGS/t20_baseline.log" | tee -a "$SUMMARY"

if grep -q "BASELINE-FAIL" "$LOGS/t20_baseline.log"; then
  log "[A2] 基线未达标。按 AGENTS.md 核心原则 5，暂停后续实验。"
  log "     建议下一步： POOL_CAP=50 bash deploy/run_stage1.sh $WORK"
  BASELINE_OK=0
elif grep -q "BASELINE-MARGINAL" "$LOGS/t20_baseline.log"; then
  log "[A2] 基线偏低但可接受，建议 POOL_CAP=50 复核后再进入阶段 2"
  BASELINE_OK=marginal
else
  log "[A2] 基线达标，可进入阶段 2（本文模型与消融）"
  BASELINE_OK=1
fi

# ---------------------------------------------------------------
# 等待任务 B
# ---------------------------------------------------------------
if [ -n "$T22_PID" ]; then
  log "[B] 等待 TwiBot-22 违例率统计完成 (PID=$T22_PID) ..."
  wait "$T22_PID"
  log "[B] 结果："
  sed -n '/顺序假设实证检验/,$p' "$LOGS/t22_violation.log" | tee -a "$SUMMARY"
fi

log "=============================================================="
log " 阶段 1 结束。BASELINE_OK=$BASELINE_OK"
log " 完整摘要： $SUMMARY"
log " 请把该文件内容贴回以便继续。"
log "=============================================================="
