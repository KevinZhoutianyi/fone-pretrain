#!/usr/bin/env bash

# FoNE Pretraining: Llama 0.5B locally (no sbatch/modules)
# Usage:
#   bash scripts/pretrain_0p5b_local.sh
#   OUT_DIR=outputs/custom bash scripts/pretrain_0p5b_local.sh
#   NUM_PROCESSES=2 bash scripts/pretrain_0p5b_local.sh

set -e

# Optional: activate local conda env if available
if [ -f "/home/nvidia/miniconda3/etc/profile.d/conda.sh" ]; then
  . "/home/nvidia/miniconda3/etc/profile.d/conda.sh" || true
  conda activate fone || true
fi

# Performance/env settings
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-8}

# Resolve repo root (this script lives in scripts/)
SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
REPO_ROOT=$(cd "$SCRIPT_DIR/.." && pwd)
cd "$REPO_ROOT"

# Load environment variables from .env if present (for HF token access)
if [ -f .env ]; then
  export $(grep -v '^#' .env | xargs)
  echo "✅ Loaded environment variables from .env"
fi

# Choose number of processes (default: min(8, gpu_count) or 1 if unknown)
GPU_COUNT=$(python - <<'PY' 2>/dev/null || true
try:
    import torch
    print(torch.cuda.device_count())
except Exception:
    print(0)
PY
)
if [ -z "$NUM_PROCESSES" ]; then
  if [ "$GPU_COUNT" -ge 8 ]; then
    NUM_PROCESSES=8
  elif [ "$GPU_COUNT" -ge 4 ]; then
    NUM_PROCESSES=4
  elif [ "$GPU_COUNT" -ge 2 ]; then
    NUM_PROCESSES=2
  elif [ "$GPU_COUNT" -ge 1 ]; then
    NUM_PROCESSES=1
  else
    NUM_PROCESSES=1
  fi
fi

# Experiment configuration
MODEL_SIZE="0p5b"
MODEL_CONFIG="configs/llama_${MODEL_SIZE}.json"
DATA_CONFIG="${DATA_CONFIG:-configs/data_local_sanity.json}"  # Default to local sanity check
OUT_DIR_DEFAULT="outputs/pretrain_${MODEL_SIZE}_$(date +%Y%m%d_%H%M%S)"
OUT_DIR="${OUT_DIR:-$OUT_DIR_DEFAULT}"

echo "🚀 Starting FoNE Pretraining (local)"
echo "Model: Llama ${MODEL_SIZE}"
echo "GPUs (processes): ${NUM_PROCESSES} (detected GPUs: ${GPU_COUNT})"
echo "Data Config: ${DATA_CONFIG}"
echo "Output: ${OUT_DIR}"
echo "Config: ${MODEL_CONFIG}"

# Prepare output directory and copy configs for reproducibility
mkdir -p "$OUT_DIR"
cp "$MODEL_CONFIG" "$OUT_DIR/"
cp "$DATA_CONFIG" "$OUT_DIR/"

# Allow customization via environment variables
BATCH_SIZE="${PER_DEVICE_BATCH_SIZE:-2}"
GRAD_ACCUM="${GRADIENT_ACCUMULATION_STEPS:-1}"
RUN_NAME="${WANDB_RUN_NAME:-my_run_$(date +%Y%m%d_%H%M%S)}"
LR="${LEARNING_RATE:-3e-4}"
WARMUP="${WARMUP_STEPS:-1000}"
WD="${WEIGHT_DECAY:-0.1}"

# Read total_tokens_target from data config
TOTAL_TOKENS=$(python3 -c "import json; print(json.load(open('$DATA_CONFIG'))['total_tokens_target'])")
echo "Total tokens target: ${TOTAL_TOKENS}"
echo "Batch size: ${BATCH_SIZE} per GPU"
echo "Gradient accumulation: ${GRAD_ACCUM} steps"
echo "Effective batch size: $((NUM_PROCESSES * BATCH_SIZE * GRAD_ACCUM))"

# Launch training with Accelerate DDP (no DeepSpeed)
accelerate launch \
  --num_processes "$NUM_PROCESSES" \
  --mixed_precision bf16 \
  src/train/pretrain.py \
  --model_config "$MODEL_CONFIG" \
  --dataset "$DATA_CONFIG" \
  --output_dir "$OUT_DIR" \
  --lr "$LR" \
  --warmup_steps "$WARMUP" \
  --weight_decay "$WD" \
  --save_every_steps 10000 \
  --log_every_steps 1 \
  --wandb_project fone-pretraining \
  --wandb_run_name "$RUN_NAME" \
  --total_tokens "$TOTAL_TOKENS" \
  --dataloader_num_workers 0 \
  --per_device_train_batch_size "$BATCH_SIZE" \
  --gradient_accumulation_steps "$GRAD_ACCUM"
  \
  # Disable activation checkpointing by default to avoid DDP reentrant backward conflicts
  # --activation_checkpointing \

echo "✅ Pretraining launched (local). Check: ${OUT_DIR}"


