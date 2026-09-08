#!/usr/bin/env bash
# 阶段 3：TwiBot-22 order-vs-Δt 闭环对照（论文 5.3 节"测量二"）
#
# 要回答的问题：TwiBot-20 上 keep≈shuffle 的空结果，是"次序确实无信息"，
# 还是"TwiBot-20 列表本就未保序"？TwiBot-20 无时间戳无法自证，只能外部验证。
# TwiBot-22 已实测 dump 近似保序（相邻对违例率 3.29%），因此是合格的验证台。
#
# 分四步，每步产物落盘可单独重跑（用 STEPS 环境变量控制）：
#   P1 collect  单遍扫 tweet_0..8.json（约 101GB），每作者取最近 L 条 + 真实时间戳
#   P2 encode   RoBERTa 逐条编码（约 1600 万条推文，3090 约 6h）
#   P3 micro    构建两套特征：base（14 通道，次序无关）与 with_dt（18 通道，加真实时间）
#   P4 diag     2×3 因子对照 × 15 种子 + 配对 t 检验 + Holm 校正
#
# ⚠️ P1 必须**不带** --stats-only：之前只跑违例率统计时用了该开关，
#    它不写 jsonl；本阶段需要文本与时间戳落盘。
#
# 用法：
#   source /root/autodl-tmp/dtg_bot/env.sh
#   nohup bash deploy/run_stage3.sh /root/autodl-tmp/dtg_bot > /root/stage3.out 2>&1 &
#   tail -f /root/autodl-tmp/dtg_bot/logs/stage3_summary.log
#
#   只重跑最后的诊断（前三步产物已在）：
#   STEPS=diag bash deploy/run_stage3.sh /root/autodl-tmp/dtg_bot

set -uo pipefail

WORK="${1:-${DTG_WORK:-/root/autodl-tmp/dtg_bot}}"
CODE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LOGS="$WORK/logs"
CACHE="$WORK/cache/twibot22"
mkdir -p "$LOGS"

SEQ_LEN="${SEQ_LEN:-16}"
BATCH_SIZE="${BATCH_SIZE:-384}"
SEEDS="${SEEDS:-42 123 456 789 2024 7 13 99 2025 314 1618 271 577 999 8128}"
STEPS="${STEPS:-collect encode micro diag}"

SUMMARY="$LOGS/stage3_summary.log"
: > "$SUMMARY"
log() { echo "[$(date '+%F %T')] $*" | tee -a "$SUMMARY"; }
has_step() { [[ " $STEPS " == *" $1 "* ]]; }

cd "$CODE_DIR"
export PYTHONPATH="$CODE_DIR/src:${PYTHONPATH:-}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

log "=============================================================="
log " DTG-Bot 阶段 3 (TwiBot-22 order-vs-Δt)  WORK=$WORK  L=$SEQ_LEN"
log " STEPS=$STEPS"
log "=============================================================="

# ---------- P1 collect：抽取最近 L 条推文 + 真实时间戳 ----------
if has_step collect; then
  log "----- P1 collect（预计 3~6h，101GB 单遍扫描）-----"
  python scripts/prepare_twibot22.py --work-dir "$WORK" \
      --seq-len "$SEQ_LEN" --steps collect \
      > "$LOGS/t22_collect.log" 2>&1
  rc=$?
  if [ $rc -ne 0 ]; then
    log "  ❌ collect 失败 (rc=$rc)"; tail -30 "$LOGS/t22_collect.log" | tee -a "$SUMMARY"; exit $rc
  fi
  # 顺序违例率是论文表 6 的数据来源，单独摘出来
  grep -A 10 "顺序假设实证检验" "$LOGS/t22_collect.log" | tee -a "$SUMMARY"
fi

# ---------- P2 encode：RoBERTa 逐条编码 ----------
if has_step encode; then
  log "----- P2 encode（约 1600 万条推文，3090 约 6h）-----"
  python scripts/prepare_twibot22.py --work-dir "$WORK" \
      --seq-len "$SEQ_LEN" --steps encode --batch-size "$BATCH_SIZE" \
      > "$LOGS/t22_encode.log" 2>&1
  rc=$?
  if [ $rc -ne 0 ]; then
    log "  ❌ encode 失败 (rc=$rc)"; tail -30 "$LOGS/t22_encode.log" | tee -a "$SUMMARY"; exit $rc
  fi
  tail -3 "$LOGS/t22_encode.log" | tee -a "$SUMMARY"
fi

# ---------- P3 micro：两套特征，唯一差别是那 4 个时间通道 ----------
if has_step micro; then
  for grp in base with_dt; do
    extra=""; [ "$grp" = "with_dt" ] && extra="--with-temporal"
    log "----- P3 micro/$grp -----"
    python scripts/prepare_twibot22.py --work-dir "$WORK" \
        --seq-len "$SEQ_LEN" --steps micro $extra \
        > "$LOGS/t22_micro_${grp}.log" 2>&1
    rc=$?
    if [ $rc -ne 0 ]; then
      log "  ❌ micro/$grp 失败 (rc=$rc)"; tail -30 "$LOGS/t22_micro_${grp}.log" | tee -a "$SUMMARY"; exit $rc
    fi
    tail -2 "$LOGS/t22_micro_${grp}.log" | tee -a "$SUMMARY"
  done
fi

# ---------- P4 diag：2×3 因子对照 ----------
if has_step diag; then
  log "----- P4 order-vs-Δt 对照（6 变体 × 15 种子）-----"
  python scripts/diagnose_order_dt.py --cache "$CACHE" --seq-len "$SEQ_LEN" \
      --seeds $SEEDS --results "$CODE_DIR/experiments/results.csv" \
      > "$LOGS/t22_order_dt.log" 2>&1
  rc=$?
  if [ $rc -ne 0 ]; then
    log "  ❌ diag 失败 (rc=$rc)"; tail -30 "$LOGS/t22_order_dt.log" | tee -a "$SUMMARY"; exit $rc
  fi
  # 汇总与检验结果直接进 summary，便于回填论文
  sed -n '/汇总 (mean/,$p' "$LOGS/t22_order_dt.log" | tee -a "$SUMMARY"
fi

log "=============================================================="
log " 阶段 3 结束。逐种子结果见 experiments/results.csv (experiment=order_dt)"
log " 完整摘要： $SUMMARY"
log "=============================================================="
