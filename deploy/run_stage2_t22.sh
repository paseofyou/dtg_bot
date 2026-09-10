#!/usr/bin/env bash
# 阶段 2：TwiBot-22 跨数据集重跑实验（主模型 + 视图/融合/宏观/微观 消融）
#
# 前置条件：T22 预处理已完成（$WORK/cache/twibot22 含 graph/、micro_L32/、encoded_L32/）
#
# 无人值守用法：
#   source /root/autodl-tmp/dtg_bot/env.sh
#   nohup bash deploy/run_stage2_t22.sh /root/autodl-tmp/dtg_bot > /root/stage2_t22.out 2>&1 &
#   tail -f /root/autodl-tmp/dtg_bot/logs/stage2_t22_summary.log

set -uo pipefail

WORK="${1:-${DTG_WORK:-/root/autodl-tmp/dtg_bot}}"
CODE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LOGS="$WORK/logs"
mkdir -p "$LOGS"

SEQ_LEN="${SEQ_LEN:-32}"
SEEDS="${SEEDS:-42 123 456 789 2024}"
DATASET="TwiBot-22"
CACHE="$WORK/cache/twibot22"

SUMMARY="$LOGS/stage2_t22_summary.log"
: > "$SUMMARY"

log() { echo "[$(date '+%F %T')] $*" | tee -a "$SUMMARY"; }

run() {  # run <tag> <extra train.py args...>
  local tag="$1"; shift
  log "----- $tag -----"
  python scripts/train.py --work-dir "$WORK" --cache "$CACHE" --dataset "$DATASET" \
      --model dtg --seq-len "$SEQ_LEN" \
      --tag "$tag" --seeds $SEEDS --select-metric accuracy "$@" \
      > "$LOGS/${tag}_t22.log" 2>&1
  local rc=$?
  if [ $rc -ne 0 ]; then
    log "  ❌ $tag 失败 (rc=$rc)，日志末尾："
    tail -20 "$LOGS/${tag}_t22.log" | tee -a "$SUMMARY"
    return $rc
  fi
  tail -12 "$LOGS/${tag}_t22.log" | tee -a "$SUMMARY"
  return 0
}

run_py() {  # run_py <tag> <python command and args...>
  local tag="$1"; shift
  log "----- $tag -----"
  "$@" > "$LOGS/${tag}_t22.log" 2>&1
  local rc=$?
  if [ $rc -ne 0 ]; then
    log "  ❌ $tag 失败 (rc=$rc)，日志末尾："
    tail -20 "$LOGS/${tag}_t22.log" | tee -a "$SUMMARY"
    return $rc
  fi
  tail -12 "$LOGS/${tag}_t22.log" | tee -a "$SUMMARY"
  return 0
}

log "=============================================================="
log " DTG-Bot 阶段 2  TwiBot-22 跨数据集重跑"
log " WORK=$WORK  DATASET=$DATASET  L=$SEQ_LEN  SEEDS=($SEEDS)"
log "=============================================================="

cd "$CODE_DIR"
export PYTHONPATH="$CODE_DIR/src:${PYTHONPATH:-}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

# ---------- M1 主模型 ----------
run main_concat --views micro macro global --fusion concat

# ---------- V 视图消融（7 种非空组合） ----------
run abl_v_micro        --views micro        --fusion concat
run abl_v_macro        --views macro        --fusion concat
run abl_v_global       --views global       --fusion concat
run abl_v_micro_macro  --views micro macro  --fusion concat
run abl_v_micro_global --views micro global --fusion concat
run abl_v_macro_global --views macro global --fusion concat

# ---------- F 融合方式消融 ----------
run_py ablate_fusion \
    python scripts/ablate_fusion.py \
    --work-dir "$WORK" --cache "$CACHE" \
    --dataset "$DATASET" --views micro macro global \
    --seeds $SEEDS \
    --results "$WORK/experiments/ablate_fusion.csv"

# ---------- T 宏观时序模块消融 ----------
run main_macro_attention --views micro macro global --fusion concat --macro-temporal attention
run main_macro_gru       --views micro macro global --fusion concat --macro-temporal gru
run main_macro_lstm      --views micro macro global --fusion concat --macro-temporal lstm

# ---------- 微观通道 + 池化消融 ----------
run_py ablate_micro \
    python scripts/ablate_micro.py \
    --cache "$CACHE" --dataset "$DATASET" \
    --seq-len "$SEQ_LEN" --seeds $SEEDS \
    --results "$WORK/experiments/results_ablate_micro.csv"

log "=============================================================="
log " 阶段 2（TwiBot-22）结束。各实验结果："
log "   主/视图/宏观：         experiments/results.csv（与 TwiBot-20 共用，以 dataset 列区分）"
log "   融合方式消融：         experiments/ablate_fusion.csv（与 TwiBot-20 共用）"
log "   微观通道/池化：        experiments/results_ablate_micro.csv（与 TwiBot-20 共用）"
log "   完整摘要： $SUMMARY"
log "=============================================================="
