#!/bin/bash
#
# Evaluate FoNE model on GSM8K
# Can compare base model vs finetuned model
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
echo "║                  GSM8K Evaluation                            ║"
echo "╚══════════════════════════════════════════════════════════════╝"
echo -e "${NC}"

# Parse arguments
MODEL=""
COMPARE_WITH=""
NUM_SAMPLES=""
MAX_TOKENS=256

while [[ $# -gt 0 ]]; do
    case $1 in
        --model) MODEL="$2"; shift 2 ;;
        --compare-with) COMPARE_WITH="$2"; shift 2 ;;
        --num-samples) NUM_SAMPLES="$2"; shift 2 ;;
        --max-tokens) MAX_TOKENS="$2"; shift 2 ;;
        --help|-h)
            echo -e "${YELLOW}Usage:${NC} $0 --model <model> [options]"
            echo ""
            echo -e "${YELLOW}Required:${NC}"
            echo "  --model ID         Model to evaluate"
            echo ""
            echo -e "${YELLOW}Options:${NC}"
            echo "  --compare-with ID  Compare with another model"
            echo "  --num-samples N    Number of samples (default: all 1319)"
            echo "  --max-tokens N     Max generation tokens (default: 256)"
            echo ""
            echo -e "${YELLOW}Examples:${NC}"
            echo "  # Single model"
            echo "  $0 --model Onlydrinkwater/fone-1p5b-step-76000"
            echo ""
            echo "  # Quick test"
            echo "  $0 --model Onlydrinkwater/fone-1p5b-step-76000 --num-samples 100"
            echo ""
            echo "  # Compare base vs finetuned"
            echo "  $0 --model Onlydrinkwater/fone-1p5b-step-76000 \\"
            echo "     --compare-with outputs/gsm8k_*/merged_model"
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

# Evaluate main model
echo ""
echo -e "${BLUE}📊 Evaluating: $MODEL${NC}"
echo ""

CMD="python eval/eval_gsm8k.py --model $MODEL --max_new_tokens $MAX_TOKENS --temperature 0.0"
[ -n "$NUM_SAMPLES" ] && CMD="$CMD --num_samples $NUM_SAMPLES"

eval $CMD

# Get result directory
RESULT_DIR=$(ls -td eval/results/gsm8k_*$(basename $MODEL)* 2>/dev/null | head -1)

# Compare if requested
if [ -n "$COMPARE_WITH" ]; then
    echo ""
    echo -e "${BLUE}📊 Evaluating: $COMPARE_WITH${NC}"
    echo ""
    
    CMD2="python eval/eval_gsm8k.py --model $COMPARE_WITH --max_new_tokens $MAX_TOKENS --temperature 0.0"
    [ -n "$NUM_SAMPLES" ] && CMD2="$CMD2 --num_samples $NUM_SAMPLES"
    
    eval $CMD2
    
    RESULT_DIR2=$(ls -td eval/results/gsm8k_*$(basename $COMPARE_WITH | sed 's/merged_model//g')* 2>/dev/null | head -1)
    
    # Show comparison
    echo ""
    echo -e "${BLUE}╔══════════════════════════════════════════════════════════════╗${NC}"
    echo -e "${BLUE}║                    COMPARISON SUMMARY                        ║${NC}"
    echo -e "${BLUE}╚══════════════════════════════════════════════════════════════╝${NC}"
    echo ""
    
    if [ -n "$RESULT_DIR" ] && [ -f "$RESULT_DIR/summary.txt" ]; then
        echo -e "${YELLOW}MODEL 1: $MODEL${NC}"
        cat "$RESULT_DIR/summary.txt"
        echo ""
    fi
    
    if [ -n "$RESULT_DIR2" ] && [ -f "$RESULT_DIR2/summary.txt" ]; then
        echo -e "${YELLOW}MODEL 2: $COMPARE_WITH${NC}"
        cat "$RESULT_DIR2/summary.txt"
        echo ""
    fi
fi

echo ""
echo -e "${GREEN}✅ Evaluation complete!${NC}"
if [ -n "$RESULT_DIR" ]; then
    echo -e "${YELLOW}Results: $RESULT_DIR${NC}"
fi

