#!/bin/bash
#
# Evaluate FoNE model with text generation
# Tests various generation capabilities
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
echo "║                  Text Generation Evaluation                  ║"
echo "╚══════════════════════════════════════════════════════════════╝"
echo -e "${NC}"

# Parse arguments
MODEL=""
MAX_TOKENS=100
TEMP=0.7
NUM_SAMPLES=3

while [[ $# -gt 0 ]]; do
    case $1 in
        --model) MODEL="$2"; shift 2 ;;
        --max-tokens) MAX_TOKENS="$2"; shift 2 ;;
        --temperature) TEMP="$2"; shift 2 ;;
        --num-samples) NUM_SAMPLES="$2"; shift 2 ;;
        --help|-h)
            echo -e "${YELLOW}Usage:${NC} $0 --model <model> [options]"
            echo ""
            echo -e "${YELLOW}Required:${NC}"
            echo "  --model ID         Model to evaluate"
            echo ""
            echo -e "${YELLOW}Options:${NC}"
            echo "  --max-tokens N     Max tokens to generate (default: 100)"
            echo "  --temperature T    Sampling temperature (default: 0.7)"
            echo "  --num-samples N    Samples per prompt (default: 3)"
            echo ""
            echo -e "${YELLOW}Examples:${NC}"
            echo "  $0 --model Onlydrinkwater/fone-1p5b-step-76000"
            echo "  $0 --model outputs/gsm8k_*/merged_model --max-tokens 200"
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

# Run evaluation
echo ""
echo -e "${BLUE}🎯 Evaluating generation capabilities...${NC}"
echo -e "${YELLOW}   Model: $MODEL${NC}"
echo -e "${YELLOW}   Max tokens: $MAX_TOKENS | Temperature: $TEMP | Samples: $NUM_SAMPLES${NC}"
echo ""

python eval/eval_generation.py \
    --model "$MODEL" \
    --max_new_tokens "$MAX_TOKENS" \
    --temperature "$TEMP" \
    --num_samples "$NUM_SAMPLES"

echo ""
echo -e "${GREEN}✅ Evaluation complete!${NC}"
echo -e "${YELLOW}Check eval/results/ for detailed outputs${NC}"

