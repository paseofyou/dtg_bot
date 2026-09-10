#!/usr/bin/env bash
# 阶段 2：TwiBot-20 全部重跑实验（主模型 + 视图/融合/宏观/微观 消融）
#
# 前置条件：阶段 1 已完成（$WORK/cache/twibot20 含 graph/、micro_L32/、encoded_L32/）
#
# 无人值守用法：
#   source /root/autodl-tmp/dtg_bot/env.sh
#   nohup bash deploy/run_stage2.sh /root/autodl-tmp/dtg_bot > /root/stage2.out 2>&1 &
#   tail -f /root/autodl-tmp/dtg_bot/logs/stage2_summary.log

set -uo pipefail

WORK="${1:-${DTG_WORK:-/root/autodl-tmp/dtg_bot}}"
CODE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LOGS="$WORK/logs"
mkdir -p "$LOGS"

SEQ_LEN="${SEQ_LEN:-32}"
SEEDS="${SEEDS:-42 123 456 789 2024}"
RAW="${RAW:-$WORK/data/twibot20/raw}"

SUMMARY="$LOGS/stage2_summary.log"
: > "$SUMMARY"

log() { echo "[$(date '+%F %T')] $*" | tee -a "$SUMMARY"; }

run() {  # run <tag> <extra train.py args...>
  local tag="$1"; shift
  log "----- $tag -----"
  python scripts/train.py --work-dir "$WORK" --cache "$WORK/cache/twibot20" \
      --model dtg --seq-len "$SEQ_LEN" \
      --tag "$tag" --seeds $SEEDS --select-metric accuracy "$@" \
      > "$LOGS/${tag}.log" 2>&1
  local rc=$?
  if [ $rc -ne 0 ]; then
    log "  ❌ $tag 失败 (rc=$rc)，日志末尾："
    tail -20 "$LOGS/${tag}.log" | tee -a "$SUMMARY"
    return $rc
  fi
  tail -12 "$LOGS/${tag}.log" | tee -a "$SUMMARY"
  return 0
}

run_py() {  # run_py <tag> <python command and args...>
  local tag="$1"; shift
  log "----- $tag -----"
  "$@" > "$LOGS/${tag}.log" 2>&1
  local rc=$?
  if [ $rc -ne 0 ]; then
    log "  ❌ $tag 失败 (rc=$rc)，日志末尾："
    tail -20 "$LOGS/${tag}.log" | tee -a "$SUMMARY"
    return $rc
  fi
  tail -12 "$LOGS/${tag}.log" | tee -a "$SUMMARY"
  return 0
}

log "=============================================================="
log " DTG-Bot 阶段 2  TwiBot-20 重跑"
log " WORK=$WORK  L=$SEQ_LEN  SEEDS=($SEEDS)"
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
    --work-dir "$WORK" --cache "$WORK/cache/twibot20" \
    --dataset TwiBot-20 --views micro macro global \
    --seeds $SEEDS

# ---------- T 宏观时序模块消融 ----------
run main_macro_attention --views micro macro global --fusion concat --macro-temporal attention
run main_macro_gru       --views micro macro global --fusion concat --macro-temporal gru
run main_macro_lstm      --views micro macro global --fusion concat --macro-temporal lstm

# ---------- 微观通道 + 池化消融 ----------
run_py ablate_micro \
    python scripts/ablate_micro.py \
    --cache "$WORK/cache/twibot20" --dataset TwiBot-20 \
    --seq-len "$SEQ_LEN" --seeds $SEEDS \
    --results "$WORK/experiments/results_ablate_micro.csv"

# ---------- Temporal Order Sensitivity ----------
run_py diagnose_order \
    python scripts/diagnose_order.py \
    --cache "$WORK/cache/twibot20" --dataset TwiBot-20 \
    --seq-len "$SEQ_LEN" --seeds $SEEDS \
    --results "$WORK/experiments/results.csv"

# ---------- Sequence Length 敏感性 ----------
run_py ablate_seq_len \
    python scripts/ablate_seq_len.py \
    --raw "$RAW" --cache "$WORK/cache/twibot20" \
    --seeds $SEEDS --skip-existing \
    --results "$WORK/experiments/results_seq_len.csv"

# ---------- Event-level Attention Visualization ----------
run_py visualize_attention \
    python scripts/visualize_attention.py \
    --cache "$WORK/cache/twibot20" --seq-len "$SEQ_LEN" \
    --out "$WORK/experiments/attention_vis.json"

log "=============================================================="
log " 阶段 2 结束。各实验结果："
log "   主/视图/融合/宏观： experiments/results.csv"
log "   融合方式消融：       experiments/ablate_fusion.csv"
log "   微观通道/池化：      experiments/results_ablate_micro.csv"
log "   顺序敏感性：         experiments/results.csv"
log "   序列长度：           experiments/results_seq_len.csv"
log "   注意力可视化：       experiments/attention_vis.json"
log " 完整摘要： $SUMMARY"
log "=============================================================="
