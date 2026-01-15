#!/bin/bash
#
# Easy wrapper to run the addition dataset experiment
#

set -e

# Colors
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m'

echo -e "${BLUE}"
echo "╔════════════════════════════════════════════════════════════════╗"
echo "║      Addition Dataset: Baseline vs FoNE Comparison            ║"
echo "║          Dataset: Onlydrinkwater/1000addition                  ║"
echo "╚════════════════════════════════════════════════════════════════╝"
echo -e "${NC}"

# Default values
BASELINE="Onlydrinkwater/baseline-1p5b-step-76294"
FONE="Onlydrinkwater/fone-1p5b-step-76294"
TRAIN_SAMPLES=50000
EPOCHS=50
BATCH_SIZE=64
LR="5e-5"

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --baseline) BASELINE="$2"; shift 2 ;;
        --fone) FONE="$2"; shift 2 ;;
        --train-samples) TRAIN_SAMPLES="$2"; shift 2 ;;
        --epochs) EPOCHS="$2"; shift 2 ;;
        --batch-size) BATCH_SIZE="$2"; shift 2 ;;
        --lr) LR="$2"; shift 2 ;;
        --help|-h)
            echo -e "${YELLOW}Usage:${NC} $0 [options]"
            echo ""
            echo -e "${YELLOW}Options:${NC}"
            echo "  --baseline MODEL       Baseline model (default: Onlydrinkwater/baseline-1p5b-step-76294)"
            echo "  --fone MODEL          FoNE model (default: Onlydrinkwater/fone-1p5b-step-76294)"
            echo "  --train-samples N     Training samples (default: 1000)"
            echo "  --epochs N            Training epochs (default: 3)"
            echo "  --batch-size N        Batch size (default: 8)"
            echo "  --lr RATE             Learning rate (default: 5e-5)"
            echo ""
            echo -e "${YELLOW}Example:${NC}"
            echo "  $0"
            echo "  $0 --epochs 5 --train-samples 2000"
            exit 0
            ;;
        *) echo -e "${RED}❌ Unknown option: $1${NC}"; exit 1 ;;
    esac
done

# Get script directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

echo -e "${GREEN}Configuration:${NC}"
echo "  Baseline:      $BASELINE"
echo "  FoNE:          $FONE"
echo "  Train samples: $TRAIN_SAMPLES"
echo "  Epochs:        $EPOCHS"
echo "  Batch size:    $BATCH_SIZE"
echo "  Learning rate: $LR"
echo ""

# Run the Python script
python "$PROJECT_DIR/finetune/finetune_and_eval_addition.py" \
    --baseline_model "$BASELINE" \
    --fone_model "$FONE" \
    --train_samples $TRAIN_SAMPLES \
    --epochs $EPOCHS \
    --batch_size $BATCH_SIZE \
    --learning_rate $LR

echo ""
echo -e "${GREEN}✅ Experiment complete!${NC}"

