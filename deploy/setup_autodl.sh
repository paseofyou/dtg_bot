#!/usr/bin/env bash
# AutoDL 环境准备与前置校验。
#
# 设计原则：把所有"会让 12 小时任务跑到一半才失败"的因素在这里全部前置检查掉
# —— 依赖缺失、HF 无法访问、数据缺文件、磁盘不足。任一不满足立即退出。
#
# 用法：
#   bash deploy/setup_autodl.sh /root/autodl-tmp/dtg_bot

set -euo pipefail

WORK="${1:-/root/autodl-tmp/dtg_bot}"
CODE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

echo "=============================================================="
echo " DTG-Bot AutoDL 环境准备"
echo " WORK = $WORK"
echo " CODE = $CODE_DIR"
echo "=============================================================="

mkdir -p "$WORK"/{data,cache,logs,experiments,hf_cache}

# ---------- 1. HuggingFace 镜像（AutoDL 直连 huggingface.co 常超时） ----------
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export HF_HOME="$WORK/hf_cache"
export TRANSFORMERS_OFFLINE=0
echo "[1/6] HF_ENDPOINT=$HF_ENDPOINT   HF_HOME=$HF_HOME"

# 写入 profile，后续所有 shell 自动继承，避免忘记 export 导致任务失败
PROFILE="$WORK/env.sh"
cat > "$PROFILE" <<EOF
export HF_ENDPOINT=$HF_ENDPOINT
export HF_HOME=$WORK/hf_cache
export PYTHONPATH=$CODE_DIR/src:\${PYTHONPATH:-}
export DTG_WORK=$WORK
export TOKENIZERS_PARALLELISM=false
EOF
echo "      环境变量已写入 $PROFILE （后续用 'source $PROFILE'）"
# shellcheck disable=SC1090
source "$PROFILE"

# ---------- 2. Python 依赖 ----------
echo "[2/6] 检查 Python 依赖"
python - <<'PY'
import importlib, sys
need = {"torch": "torch", "torch_geometric": "torch-geometric",
        "transformers": "transformers", "numpy": "numpy", "pandas": "pandas",
        "sklearn": "scikit-learn", "scipy": "scipy", "yaml": "pyyaml",
        "tqdm": "tqdm", "ijson": "ijson"}
missing = [pkg for mod, pkg in need.items() if not importlib.util.find_spec(mod)]
if missing:
    print("MISSING:" + " ".join(missing)); sys.exit(3)
print("  全部依赖已就绪")
PY
rc=$?
if [ "$rc" = "3" ]; then
  echo "      检测到缺失依赖，开始安装 ..."
  pip install -q -r "$CODE_DIR/requirements.txt" ijson
fi

# ---------- 3. GPU 与显存 ----------
echo "[3/6] GPU 信息"
python - <<'PY'
import torch
if not torch.cuda.is_available():
    raise SystemExit("❌ CUDA 不可用，编码与训练无法进行")
p = torch.cuda.get_device_properties(0)
gb = p.total_memory / 1e9
print(f"  {p.name}  {gb:.1f} GB  torch {torch.__version__}")
if gb < 10:
    print(f"  ⚠️  显存仅 {gb:.1f}GB，建议把 --batch-size 降到 128 以下")
PY

# ---------- 4. 预下载 roberta-base（失败则立即退出） ----------
echo "[4/6] 预下载 roberta-base"
python - <<'PY'
import os, sys
from transformers import AutoModel, AutoTokenizer
try:
    AutoTokenizer.from_pretrained("roberta-base")
    AutoModel.from_pretrained("roberta-base")
except Exception as e:
    sys.exit(f"❌ roberta-base 下载失败: {type(e).__name__}: {e}\n"
             f"   当前 HF_ENDPOINT={os.environ.get('HF_ENDPOINT')}\n"
             f"   请检查网络或手动上传模型到 $HF_HOME")
print("  roberta-base 就绪")
PY

# ---------- 5. 数据完整性 ----------
echo "[5/6] 数据完整性检查"
T20="$WORK/data/twibot20"
ok=1
for f in train.json dev.json test.json support.json; do
  if [ -f "$T20/$f" ]; then
    printf "  T20 %-14s %s\n" "$f" "$(du -h "$T20/$f" | cut -f1)"
  else
    echo "  ❌ 缺少 $T20/$f"; ok=0
  fi
done
[ "$ok" = "1" ] || { echo "❌ TwiBot-20 数据不完整，退出"; exit 1; }

# ---------- 6. 磁盘空间 ----------
echo "[6/6] 磁盘空间"
df -h "$WORK" | tail -1 | awk '{print "  可用 " $4 " / 总计 " $2 " (已用 " $5 ")"}'
avail_gb=$(df -BG "$WORK" | tail -1 | awk '{gsub("G","",$4); print $4}')
if [ "$avail_gb" -lt 30 ]; then
  echo "  ⚠️  可用空间不足 30GB，TwiBot-20 产物约需 5GB"
fi

echo
echo "=============================================================="
echo " ✅ 环境准备完成"
echo " 下一步： source $WORK/env.sh && bash deploy/run_stage1.sh $WORK"
echo "=============================================================="
