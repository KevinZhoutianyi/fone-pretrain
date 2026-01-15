#!/usr/bin/env bash

# Unified Pretraining Script
# Usage:
#   bash scripts/pretrain.sh fone              # Run in foreground
#   bash scripts/pretrain.sh baseline          # Run in foreground
#   bash scripts/pretrain.sh fone --background # Run in background with nohup
#   bash scripts/pretrain.sh baseline --bg     # Run in background (shorthand)

set -e

# Colors
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

# Check if config file is provided
if [ $# -eq 0 ]; then
    echo -e "${RED}❌ Error: No config file provided${NC}"
    echo ""
    echo "Usage:"
    echo "  bash scripts/pretrain.sh fone              # Run in foreground"
    echo "  bash scripts/pretrain.sh baseline          # Run in foreground"
    echo "  bash scripts/pretrain.sh fone --background # Run in background"
    echo "  bash scripts/pretrain.sh baseline --bg     # Run in background (shorthand)"
    exit 1
fi

CONFIG_ARG="$1"
BACKGROUND_MODE=false

# Check for background flag
if [ $# -ge 2 ]; then
    if [ "$2" = "--background" ] || [ "$2" = "--bg" ]; then
        BACKGROUND_MODE=true
    fi
fi

# Get script directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# Handle shorthand: "fone" or "baseline"
case "$CONFIG_ARG" in
    fone)
        CONFIG_FILE="$SCRIPT_DIR/pretrain_fone.yaml"
        ;;
    baseline)
        CONFIG_FILE="$SCRIPT_DIR/pretrain_baseline.yaml"
        ;;
    *)
        CONFIG_FILE="$CONFIG_ARG"
        ;;
esac

# Check if config file exists
if [ ! -f "$CONFIG_FILE" ]; then
    echo -e "${RED}❌ Error: Config file not found: $CONFIG_FILE${NC}"
    echo ""
    echo "Available options:"
    echo "  bash scripts/pretrain.sh fone       # FoNE pretraining"
    echo "  bash scripts/pretrain.sh baseline   # Baseline pretraining"
    exit 1
fi

echo -e "${BLUE}"
echo "╔══════════════════════════════════════════════════════════════╗"
echo "║              LLaMA 1.5B Pretraining                          ║"
echo "╚══════════════════════════════════════════════════════════════╝"
echo -e "${NC}"
echo -e "${YELLOW}Config: $CONFIG_FILE${NC}"
echo ""

# Navigate to repo root
cd "$SCRIPT_DIR/.."

# Load environment variables from .env
if [ -f .env ]; then
    export $(grep -v '^#' .env | xargs)
    echo -e "${GREEN}✅ Loaded .env file${NC}"
fi

# Activate conda
echo -e "${BLUE}🔧 Activating environment...${NC}"
if [ -f /home/nvidia/miniconda3/etc/profile.d/conda.sh ]; then
    . /home/nvidia/miniconda3/etc/profile.d/conda.sh
    conda activate fone
    echo -e "${GREEN}✅ Environment activated${NC}"
fi

# Performance/env settings
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-8}

# Detect GPU count
GPU_COUNT=$(python - <<'PY' 2>/dev/null || true
try:
    import torch
    print(torch.cuda.device_count())
except Exception:
    print(0)
PY
)

# Parse YAML config using Python
echo -e "${BLUE}📋 Parsing config...${NC}"
eval $(python3 <<EOF
import yaml
import sys

with open('$CONFIG_FILE', 'r') as f:
    config = yaml.safe_load(f)

# Print bash variable assignments
for key, value in config.items():
    if value is None or value == 'null':
        print(f"{key.upper()}=")
    elif isinstance(value, bool):
        print(f"{key.upper()}={str(value).lower()}")
    elif isinstance(value, (int, float)):
        print(f"{key.upper()}={value}")
    else:
        # Escape quotes in string values
        value_str = str(value).replace("'", "'\\''")
        print(f"{key.upper()}='{value_str}'")
EOF
)

# Generate output directory
OUTPUT_DIR="outputs/pretrain_$(basename $CONFIG_FILE .yaml)_$(date +%Y%m%d_%H%M%S)"

# Create output directory and copy configs
mkdir -p "$OUTPUT_DIR"
cp "$CONFIG_FILE" "$OUTPUT_DIR/"
cp "$MODEL_CONFIG" "$OUTPUT_DIR/" 2>/dev/null || true
cp "$DATA_CONFIG" "$OUTPUT_DIR/" 2>/dev/null || true

# Read total tokens from data config
TOTAL_TOKENS=$(python3 -c "import json; print(json.load(open('$DATA_CONFIG'))['total_tokens_target'])")

# Display configuration
echo ""
echo -e "${YELLOW}Configuration Summary:${NC}"
echo "  Model config: $MODEL_CONFIG"
echo "  Data config: $DATA_CONFIG"
echo "  Output dir: $OUTPUT_DIR"
echo "  Batch size: $PER_DEVICE_TRAIN_BATCH_SIZE per GPU"
echo "  Gradient accumulation: $GRADIENT_ACCUMULATION_STEPS"
echo "  Effective batch: $((NUM_PROCESSES * PER_DEVICE_TRAIN_BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS))"
echo "  Learning rate: $LEARNING_RATE"
echo "  Warmup steps: $WARMUP_STEPS"
echo "  Weight decay: $WEIGHT_DECAY"
echo "  Total tokens: $TOTAL_TOKENS"
echo "  GPUs: $NUM_PROCESSES (detected: $GPU_COUNT)"
echo "  FoNE hi: ${FONE_HI:-default}"
echo "  Wandb: $WANDB_PROJECT / $WANDB_RUN_NAME"
echo "  HF Repo: ${HF_REPO_ID:-none (local only)}"
echo ""
echo -e "${YELLOW}💾 Checkpoint Strategy:${NC}"
if [ -n "$HF_REPO_ID" ]; then
  echo "  ✓ Upload to HuggingFace: $HF_REPO_ID"
  echo "  ✓ Auto-cleanup local checkpoints after upload"
  echo "  (Saves disk space - models are backed up on HuggingFace)"
else
  echo "  ⚠️  Local checkpoints only (no HuggingFace upload)"
fi
echo ""

sleep 2

# Build accelerate launch command
ACCEL_CMD="accelerate launch \
  --num_processes $NUM_PROCESSES \
  --mixed_precision $MIXED_PRECISION \
  src/train/pretrain.py \
  --model_config $MODEL_CONFIG \
  --dataset $DATA_CONFIG \
  --output_dir $OUTPUT_DIR \
  --lr $LEARNING_RATE \
  --warmup_steps $WARMUP_STEPS \
  --weight_decay $WEIGHT_DECAY \
  --save_every_steps $SAVE_EVERY_STEPS \
  --log_every_steps $LOG_EVERY_STEPS \
  --wandb_project $WANDB_PROJECT \
  --wandb_run_name $WANDB_RUN_NAME \
  --total_tokens $TOTAL_TOKENS \
  --dataloader_num_workers $DATALOADER_NUM_WORKERS \
  --per_device_train_batch_size $PER_DEVICE_TRAIN_BATCH_SIZE \
  --gradient_accumulation_steps $GRADIENT_ACCUMULATION_STEPS"

# Add fone_hi flag if specified
if [ -n "$FONE_HI" ]; then
  ACCEL_CMD="$ACCEL_CMD --fone_hi $FONE_HI"
fi

# Add HuggingFace repo ID if specified
if [ -n "$HF_REPO_ID" ]; then
  ACCEL_CMD="$ACCEL_CMD --hf_repo_id $HF_REPO_ID"
fi

# Launch training
if [ "$BACKGROUND_MODE" = true ]; then
    LOG_FILE="$OUTPUT_DIR/training.log"
    echo -e "${GREEN}🚀 Starting training in BACKGROUND mode...${NC}"
    echo -e "${YELLOW}📝 Logs: $LOG_FILE${NC}"
    echo -e "${YELLOW}💡 Monitor with: tail -f $LOG_FILE${NC}"
    echo ""
    
    # Run in background with nohup
    nohup bash -c "$ACCEL_CMD" > "$LOG_FILE" 2>&1 &
    PID=$!
    echo $PID > "$OUTPUT_DIR/training.pid"
    
    echo -e "${GREEN}✅ Training started in background (PID: $PID)${NC}"
    echo -e "${YELLOW}📊 Check status: ps -p $PID${NC}"
    echo -e "${YELLOW}🛑 Stop training: kill $PID${NC}"
    echo ""
else
    echo -e "${GREEN}🚀 Starting training...${NC}"
    echo ""
    eval $ACCEL_CMD
    
    echo ""
    echo -e "${GREEN}✅ Training complete!${NC}"
    echo -e "${YELLOW}Model saved to: $OUTPUT_DIR${NC}"
    echo ""
fi
