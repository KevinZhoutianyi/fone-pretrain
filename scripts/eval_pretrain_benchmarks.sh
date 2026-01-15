#!/bin/bash
#
# Evaluate pretrained models on easy benchmarks
# Compares baseline vs FoNE models
#

set -e

# Colors
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

echo -e "${BLUE}"
echo "╔════════════════════════════════════════════════════════════════╗"
echo "║          Pretraining Benchmark Evaluation (Easy)               ║"
echo "║  HellaSwag | PIQA | WinoGrande | ARC-easy | BoolQ             ║"
echo "╚════════════════════════════════════════════════════════════════╝"
echo -e "${NC}"

# Default values
BASELINE="Onlydrinkwater/baseline-1p5b-step-76294"
FONE="Onlydrinkwater/fone-1p5b-step-76294"
TASKS="easy"
BATCH_SIZE=8
DEVICE="cuda:0"
OUTPUT_DIR=""

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --baseline) BASELINE="$2"; shift 2 ;;
        --fone) FONE="$2"; shift 2 ;;
        --tasks) TASKS="$2"; shift 2 ;;
        --batch-size) BATCH_SIZE="$2"; shift 2 ;;
        --device) DEVICE="$2"; shift 2 ;;
        --output-dir) OUTPUT_DIR="$2"; shift 2 ;;
        --help|-h)
            echo -e "${YELLOW}Usage:${NC} $0 [options]"
            echo ""
            echo -e "${YELLOW}Options:${NC}"
            echo "  --baseline MODEL    Baseline model (default: Onlydrinkwater/baseline-1p5b-step-76294)"
            echo "  --fone MODEL        FoNE model (default: Onlydrinkwater/fone-1p5b-step-76294)"
            echo "  --tasks TYPE        Benchmark set: easy/medium/all (default: easy)"
            echo "  --batch-size N      Batch size (default: 8)"
            echo "  --device DEVICE     Device (default: cuda:0)"
            echo "  --output-dir DIR    Output directory"
            echo ""
            echo -e "${YELLOW}Easy benchmarks:${NC}"
            echo "  - HellaSwag: Common sense reasoning"
            echo "  - PIQA: Physical commonsense"
            echo "  - WinoGrande: Pronoun resolution"
            echo "  - ARC-easy: Simple science questions"
            echo "  - BoolQ: Yes/No questions"
            echo ""
            echo -e "${YELLOW}Examples:${NC}"
            echo "  $0"
            echo "  $0 --tasks medium"
            echo "  $0 --baseline my-model --fone my-fone-model"
            exit 0
            ;;
        *) echo -e "${RED}❌ Unknown option: $1${NC}"; exit 1 ;;
    esac
done

# Get script directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
EVAL_SCRIPT="$PROJECT_DIR/eval/eval_pretraining_benchmarks.py"

# Check if eval script exists
if [ ! -f "$EVAL_SCRIPT" ]; then
    echo -e "${RED}❌ Evaluation script not found: $EVAL_SCRIPT${NC}"
    exit 1
fi

echo -e "${GREEN}Configuration:${NC}"
echo "  Baseline: $BASELINE"
echo "  FoNE:     $FONE"
echo "  Tasks:    $TASKS"
echo "  Batch:    $BATCH_SIZE"
echo "  Device:   $DEVICE"
echo ""

# Build command
CMD="python $EVAL_SCRIPT --baseline $BASELINE --fone $FONE --tasks $TASKS --batch_size $BATCH_SIZE --device $DEVICE"

if [ -n "$OUTPUT_DIR" ]; then
    CMD="$CMD --output_dir $OUTPUT_DIR"
fi

echo -e "${BLUE}Running evaluation...${NC}"
echo ""

# Run evaluation
eval $CMD

echo ""
echo -e "${GREEN}✅ Evaluation complete!${NC}"

