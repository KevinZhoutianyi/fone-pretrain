#!/bin/bash
#
# FoNE GSM8K Finetuning Script
# Prepares data and finetunes on GSM8K with Chain-of-Thought
#

set -e

# Colors
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

echo -e "${BLUE}"
echo "╔══════════════════════════════════════════════════════════════╗"
echo "║                   FoNE GSM8K Finetuning                      ║"
echo "╚══════════════════════════════════════════════════════════════╝"
echo -e "${NC}"

# Parse arguments
MODEL=""
EPOCHS=3
BATCH_SIZE=4
GRAD_ACCUM=4
LR=1e-4
LORA_R=16
SKIP_PREP=false
OUTPUT_DIR=""
NO_WANDB=false
HF_REPO=""
KEEP_LOCAL=false

while [[ $# -gt 0 ]]; do
    case $1 in
        --model) MODEL="$2"; shift 2 ;;
        --epochs) EPOCHS="$2"; shift 2 ;;
        --batch-size) BATCH_SIZE="$2"; shift 2 ;;
        --grad-accum) GRAD_ACCUM="$2"; shift 2 ;;
        --lr) LR="$2"; shift 2 ;;
        --lora-r) LORA_R="$2"; shift 2 ;;
        --skip-data-prep) SKIP_PREP=true; shift ;;
        --output-dir) OUTPUT_DIR="$2"; shift 2 ;;
        --no-wandb) NO_WANDB=true; shift ;;
        --hf-repo) HF_REPO="$2"; shift 2 ;;
        --keep-local) KEEP_LOCAL=true; shift ;;
        --help|-h)
            echo -e "${YELLOW}Usage:${NC} $0 --model <model_id> [options]"
            echo ""
            echo -e "${YELLOW}Required:${NC}"
            echo "  --model ID         HuggingFace model ID or local path"
            echo ""
            echo -e "${YELLOW}Options:${NC}"
            echo "  --epochs N         Training epochs (default: 3)"
            echo "  --batch-size N     Batch size per GPU (default: 4)"
            echo "  --grad-accum N     Gradient accumulation (default: 4)"
            echo "  --lr RATE          Learning rate (default: 1e-4)"
            echo "  --lora-r N         LoRA rank (default: 16)"
            echo "  --skip-data-prep   Skip data preparation"
            echo "  --output-dir DIR   Custom output directory"
            echo "  --no-wandb         Disable W&B logging"
            echo "  --hf-repo REPO     Upload to HuggingFace repo (e.g., user/model-name)"
            echo "  --keep-local       Keep local checkpoints after upload"
            echo ""
            echo -e "${YELLOW}Examples:${NC}"
            echo "  $0 --model Onlydrinkwater/fone-1p5b-step-76000"
            echo "  $0 --model Onlydrinkwater/fone-1p5b-step-76000 --epochs 5 --lr 2e-4"
            echo "  $0 --model Onlydrinkwater/fone-1p5b-step-76000 --hf-repo user/my-model"
            exit 0
            ;;
        *) echo -e "${RED}❌ Unknown: $1${NC}"; exit 1 ;;
    esac
done

if [ -z "$MODEL" ]; then
    echo -e "${RED}❌ --model required. Use --help for info${NC}"
    exit 1
fi

# Get script directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR/.."

# Activate conda
echo -e "${BLUE}🔧 Activating environment...${NC}"
if [ -f /home/nvidia/miniconda3/etc/profile.d/conda.sh ]; then
    . /home/nvidia/miniconda3/etc/profile.d/conda.sh
    conda activate fone
    echo -e "${GREEN}✅ Environment activated${NC}"
else
    echo -e "${RED}❌ Conda not found${NC}"; exit 1
fi

# Prepare data
DATA_DIR="data/gsm8k_cot"
if [ "$SKIP_PREP" = false ]; then
    echo ""
    echo -e "${BLUE}📥 Preparing GSM8K dataset...${NC}"
    python src/finetune/prepare_gsm8k_data.py --output_dir "$DATA_DIR"
    echo -e "${GREEN}✅ Data ready${NC}"
else
    echo -e "${YELLOW}⏭️  Skipping data prep${NC}"
fi

# Check data
if [ ! -f "$DATA_DIR/train.jsonl" ]; then
    echo -e "${RED}❌ No data at $DATA_DIR/train.jsonl${NC}"
    exit 1
fi

# Finetune
echo ""
echo -e "${BLUE}🚀 Finetuning...${NC}"
echo -e "${YELLOW}   Model: $MODEL${NC}"
echo -e "${YELLOW}   Epochs: $EPOCHS | Batch: $BATCH_SIZE | Accum: $GRAD_ACCUM | LR: $LR${NC}"
echo ""

sleep 3

CMD="python src/finetune/train_gsm8k.py \
    --model $MODEL \
    --data_dir $DATA_DIR \
    --num_epochs $EPOCHS \
    --batch_size $BATCH_SIZE \
    --grad_accum $GRAD_ACCUM \
    --learning_rate $LR \
    --use_lora \
    --lora_r $LORA_R"

[ -n "$OUTPUT_DIR" ] && CMD="$CMD --output_dir $OUTPUT_DIR"
[ "$NO_WANDB" = true ] && CMD="$CMD --no_wandb"
[ -n "$HF_REPO" ] && CMD="$CMD --hf_repo $HF_REPO"
[ "$KEEP_LOCAL" = true ] && CMD="$CMD --keep_local"

eval $CMD

echo ""
echo -e "${GREEN}✅ Finetuning complete!${NC}"
echo ""
echo -e "${YELLOW}To evaluate, run:${NC}"
echo "  bash scripts/eval_gsm8k.sh --model <finetuned_model_path>"

