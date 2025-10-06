#!/bin/bash

# FoNE Evaluation: Llama 0.5B on GSM8K with Few-Shot
# Usage: bash scripts/eval_0p5b_gsm8k.sh /path/to/model [/path/to/lora/adapter]

set -e

# Check arguments
if [ $# -eq 0 ]; then
    echo "❌ Error: Please provide path to model"
    echo "Usage: bash scripts/eval_0p5b_gsm8k.sh /path/to/model [/path/to/lora/adapter]"
    exit 1
fi

MODEL_PATH="$1"
LORA_PATH="${2:-}"

# Environment setup
module load anaconda3_gpu cuda
eval "$(conda shell.bash hook)"
conda activate fone

# Load project environment variables
export PROJECT_ID="bfdj"
export PROJECT_DIR="/projects/${PROJECT_ID}/${USER}"
export WORK_NVME_DIR="/work/nvme/${PROJECT_ID}/${USER}"
export WORK_HDD_DIR="/work/hdd/${PROJECT_ID}/${USER}"

# Load environment variables from .env file
if [ -f .env ]; then
    export $(cat .env | grep -v '^#' | xargs)
    echo "✅ Loaded HuggingFace token from .env"
fi

# Performance settings
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=8

# Experiment configuration
MODEL_SIZE="0p5b"
SHOTS_LIST="0 1 3 5"  # Different few-shot settings
BATCH_SIZE=8
MAX_LENGTH=1024
TEMPERATURE=0.1

# Create output directory - use PROJECT_DIR for evaluation results (persistent storage)
if [ -n "$PROJECT_DIR" ] && [ -d "$(dirname "$PROJECT_DIR")" ]; then
    OUTPUT_DIR="${PROJECT_DIR}/outputs/eval_${MODEL_SIZE}_gsm8k_$(date +%Y%m%d_%H%M%S)"
else
    OUTPUT_DIR="outputs/eval_${MODEL_SIZE}_gsm8k_$(date +%Y%m%d_%H%M%S)"
fi
mkdir -p "$OUTPUT_DIR"

echo "🚀 Starting FoNE GSM8K Evaluation"
echo "Model: Llama ${MODEL_SIZE}"
echo "Model Path: ${MODEL_PATH}"
if [ -n "$LORA_PATH" ]; then
    echo "LoRA Path: ${LORA_PATH}"
fi
echo "Output: ${OUTPUT_DIR}"

# Run evaluation for different few-shot settings
for SHOTS in $SHOTS_LIST; do
    echo ""
    echo "📊 Evaluating with ${SHOTS}-shot prompting..."
    
    LORA_ARG=""
    if [ -n "$LORA_PATH" ]; then
        LORA_ARG="--lora_path $LORA_PATH"
    fi
    
    python src/eval/gsm8k_eval.py \
        --model_path "$MODEL_PATH" \
        $LORA_ARG \
        --dataset_name "gsm8k" \
        --dataset_config "main" \
        --split "test" \
        --num_shots "$SHOTS" \
        --batch_size "$BATCH_SIZE" \
        --max_length "$MAX_LENGTH" \
        --temperature "$TEMPERATURE" \
        --output_file "${OUTPUT_DIR}/results_${SHOTS}shot.json" \
        --save_predictions \
        --verbose
done

echo ""
echo "📈 Summary Results:"
echo "=================="

# Display summary of all results
for SHOTS in $SHOTS_LIST; do
    RESULT_FILE="${OUTPUT_DIR}/results_${SHOTS}shot.json"
    if [ -f "$RESULT_FILE" ]; then
        ACCURACY=$(python -c "import json; data=json.load(open('$RESULT_FILE')); print(f\"{data.get('accuracy', 0)*100:.1f}%\")")
        echo "${SHOTS}-shot: $ACCURACY"
    fi
done

echo ""
echo "✅ Evaluation completed!"
echo "Results saved in: $OUTPUT_DIR"
