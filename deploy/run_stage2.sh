#!/usr/bin/env bash
# 阶段 2：DTG-Bot 主实验 + 消融（TwiBot-20，GPU 任务）
#
# 前置条件：阶段 1 已完成（A1 编码产物在 $WORK/cache/twibot20，
#           A2 BotRGCN 基线已 PASS）。
# 覆盖论文 5.2 主结果与 5.4 消融：
#   M1  主模型  : micro+macro+global, fusion=attn, lambda=0.3
#   V   视图消融: 7 种非空视图组合
#   F   融合消融: attn / gate / concat
#   L   λ 敏感性: {0, 0.1, 0.3, 0.5, 1.0}
#   T   宏观时序消融: gru vs last（"演化轨迹 vs 终态结构"对照）
#
# 用法：
#   source /root/autodl-tmp/dtg_bot/env.sh
#   nohup bash deploy/run_stage2.sh /root/autodl-tmp/dtg_bot > /root/stage2.out 2>&1 &
#   tail -f /root/autodl-tmp/dtg_bot/logs/stage2_summary.log

set -uo pipefail

WORK="${1:-${DTG_WORK:-/root/autodl-tmp/dtg_bot}}"
CODE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LOGS="$WORK/logs"
mkdir -p "$LOGS"

SEQ_LEN="${SEQ_LEN:-32}"
LAMBDA="${LAMBDA:-0.3}"
SEEDS="${SEEDS:-42 123 456 789 2024 7 13 99 2025 314 1618 271 577 999 8128}"

SUMMARY="$LOGS/stage2_summary.log"
: > "$SUMMARY"

log() { echo "[$(date '+%F %T')] $*" | tee -a "$SUMMARY"; }

run() {  # run <tag> <extra args...>
  local tag="$1"; shift
  log "----- $tag -----"
  python scripts/train.py --work-dir "$WORK" --model dtg --seq-len "$SEQ_LEN" \
      --tag "$tag" --seeds $SEEDS "$@" \
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

log "=============================================================="
log " DTG-Bot 阶段 2   WORK=$WORK  L=$SEQ_LEN  λ=$LAMBDA"
log "=============================================================="

cd "$CODE_DIR"
export PYTHONPATH="$CODE_DIR/src:${PYTHONPATH:-}"
# 阶段 2 首次运行曾在 backward 处 OOM，其中 7.95GB 为 reserved-but-unallocated 碎片。
# 可伸缩段能显著缓解全图训练下大块张量反复申请/释放造成的碎片。
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

# ---------- M1 主模型 ----------
run dtg_main --views micro macro global --fusion attn --lambda "$LAMBDA"

# ---------- V 视图消融（7 种非空组合，含主模型已是全集故只跑其余 6 种） ----------
run abl_v_micro        --views micro                 --fusion attn --lambda "$LAMBDA"
run abl_v_macro        --views macro                 --fusion attn --lambda "$LAMBDA"
run abl_v_global       --views global                --fusion attn --lambda "$LAMBDA"
run abl_v_micro_macro  --views micro macro           --fusion attn --lambda "$LAMBDA"
run abl_v_micro_global --views micro global          --fusion attn --lambda "$LAMBDA"
run abl_v_macro_global --views macro global          --fusion attn --lambda "$LAMBDA"

# ---------- F 融合方式消融 ----------
run abl_f_concat --views micro macro global --fusion concat --lambda "$LAMBDA"
run abl_f_gate   --views micro macro global --fusion gate   --lambda "$LAMBDA"

# ---------- L λ 敏感性（0.3 已由 M1 覆盖） ----------
for lam in 0 0.1 0.5 1.0; do
  run "abl_l_${lam}" --views micro macro global --fusion attn --lambda "$lam"
done

# ---------- T 宏观时序消融：终态结构 vs 演化轨迹 ----------
run abl_t_last --views micro macro global --fusion attn --lambda "$LAMBDA" \
    --macro-temporal last

log "=============================================================="
log " 阶段 2 结束。汇总各 run 结果请查看 experiments/results.csv"
log " 完整摘要： $SUMMARY"
log "=============================================================="
