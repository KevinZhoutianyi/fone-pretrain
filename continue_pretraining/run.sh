#!/usr/bin/env bash

# Continue Pretraining Script
# Usage:
#   bash run.sh fone              # Continue pretrain FoNE model
#   bash run.sh baseline          # Continue pretrain baseline model
#   bash run.sh fone --background # Run in background
#   bash run.sh baseline --bg     # Run in background (shorthand)

set -e

GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

if [ $# -eq 0 ]; then
    echo -e "${RED}❌ Error: No config provided${NC}"
    echo ""
    echo "Usage:"
    echo "  bash run.sh fone              # Continue pretrain FoNE model"
    echo "  bash run.sh baseline          # Continue pretrain baseline model"
    echo "  bash run.sh fone --background # Run in background"
    exit 1
fi

CONFIG_ARG="$1"
BACKGROUND_MODE=false

if [ $# -ge 2 ]; then
    if [ "$2" = "--background" ] || [ "$2" = "--bg" ]; then
        BACKGROUND_MODE=true
    fi
fi

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

# Handle shorthand configs
case "$CONFIG_ARG" in
    fone)
        CONFIG_FILE="$SCRIPT_DIR/continue_fone.yaml"
        ;;
    baseline)
        CONFIG_FILE="$SCRIPT_DIR/continue_baseline.yaml"
        ;;
    *)
        CONFIG_FILE="$CONFIG_ARG"
        ;;
esac

if [ ! -f "$CONFIG_FILE" ]; then
    echo -e "${RED}❌ Error: Config file not found: $CONFIG_FILE${NC}"
    exit 1
fi

echo -e "${BLUE}"
echo "╔══════════════════════════════════════════════════════════════╗"
echo "║           Continue Pretraining on MegaMath                   ║"
echo "╚══════════════════════════════════════════════════════════════╝"
echo -e "${NC}"
echo -e "${YELLOW}Config: $CONFIG_FILE${NC}"
echo ""

# Load environment variables from .env (contains WANDB_API_KEY)
REPO_ROOT="$SCRIPT_DIR/.."
if [ -f "$REPO_ROOT/.env" ]; then
    export $(grep -v '^#' "$REPO_ROOT/.env" | xargs)
    echo -e "${GREEN}✅ Loaded .env file${NC}"
fi

# Activate conda
if [ -f /home/nvidia/miniconda3/etc/profile.d/conda.sh ]; then
    . /home/nvidia/miniconda3/etc/profile.d/conda.sh
    conda activate fone
    echo -e "${GREEN}✅ Environment activated${NC}"
fi

export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-8}

# Use /data for all caching to avoid filling root filesystem
export HF_HOME=/data/.cache/huggingface
export HF_DATASETS_CACHE=/data/.cache/huggingface/datasets
export TRANSFORMERS_CACHE=/data/.cache/huggingface/transformers
mkdir -p $HF_HOME $HF_DATASETS_CACHE $TRANSFORMERS_CACHE

# Parse YAML config
echo -e "${BLUE}📋 Parsing config...${NC}"
eval $(python3 <<EOF
import yaml

with open('$CONFIG_FILE', 'r') as f:
    config = yaml.safe_load(f)

for key, value in config.items():
    if value is None or value == 'null':
        print(f"{key.upper()}=")
    elif isinstance(value, bool):
        print(f"{key.upper()}={str(value).lower()}")
    elif isinstance(value, (int, float)):
        print(f"{key.upper()}={value}")
    else:
        value_str = str(value).replace("'", "'\\''")
        print(f"{key.upper()}='{value_str}'")
EOF
)

OUTPUT_DIR="$SCRIPT_DIR/outputs/continue_$(basename $CONFIG_FILE .yaml)_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$OUTPUT_DIR"
cp "$CONFIG_FILE" "$OUTPUT_DIR/"

# Read total tokens from data config (like pretrain.sh)
TOTAL_TOKENS=$(python3 -c "import json; print(json.load(open('$DATA_CONFIG'))['total_tokens_target'])")

echo ""
echo -e "${YELLOW}Configuration Summary:${NC}"
echo "  Model: $MODEL_NAME_OR_PATH"
echo "  Data config: $DATA_CONFIG"
echo "  Output dir: $OUTPUT_DIR"
echo "  Batch size: $PER_DEVICE_TRAIN_BATCH_SIZE per GPU"
echo "  Gradient accumulation: $GRADIENT_ACCUMULATION_STEPS"
echo "  Learning rate: $LEARNING_RATE"
echo "  Total tokens: $TOTAL_TOKENS (from data config)"
echo "  GPUs: $NUM_PROCESSES"
echo "  FoNE freeze: ${FONE_HI:--1} (0-N number embeddings frozen)"
echo "  Wandb: $WANDB_PROJECT / $WANDB_RUN_NAME"
echo "  HF Repo: ${HF_REPO_ID:-none} (final checkpoint only)"
echo ""

sleep 2

# Build command
ACCEL_CMD="accelerate launch \
  --num_processes $NUM_PROCESSES \
  --mixed_precision $MIXED_PRECISION \
  continue_pretrain.py \
  --model_name_or_path $MODEL_NAME_OR_PATH \
  --dataset $DATA_CONFIG \
  --output_dir $OUTPUT_DIR \
  --lr $LEARNING_RATE \
  --warmup_steps $WARMUP_STEPS \
  --weight_decay $WEIGHT_DECAY \
  --log_every_steps $LOG_EVERY_STEPS \
  --wandb_project $WANDB_PROJECT \
  --wandb_run_name $WANDB_RUN_NAME \
  --total_tokens $TOTAL_TOKENS \
  --dataloader_num_workers $DATALOADER_NUM_WORKERS \
  --per_device_train_batch_size $PER_DEVICE_TRAIN_BATCH_SIZE \
  --gradient_accumulation_steps $GRADIENT_ACCUMULATION_STEPS"

# Add FoNE parameters if specified
if [ -n "$FONE_HI" ]; then
  ACCEL_CMD="$ACCEL_CMD --fone_hi $FONE_HI"
fi

if [ -n "$HF_REPO_ID" ]; then
  ACCEL_CMD="$ACCEL_CMD --hf_repo_id $HF_REPO_ID"
fi

# Launch
if [ "$BACKGROUND_MODE" = true ]; then
    LOG_FILE="$OUTPUT_DIR/training.log"
    echo -e "${GREEN}🚀 Starting training in BACKGROUND mode...${NC}"
    echo -e "${YELLOW}📝 Logs: $LOG_FILE${NC}"
    echo -e "${YELLOW}💡 Monitor with: tail -f $LOG_FILE${NC}"
    
    nohup bash -c "$ACCEL_CMD" > "$LOG_FILE" 2>&1 &
    PID=$!
    echo $PID > "$OUTPUT_DIR/training.pid"
    
    echo -e "${GREEN}✅ Training started (PID: $PID)${NC}"
else
    echo -e "${GREEN}🚀 Starting training...${NC}"
    eval $ACCEL_CMD
    echo -e "${GREEN}✅ Training complete!${NC}"
fi

