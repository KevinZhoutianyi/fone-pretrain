#!/bin/bash
#SBATCH --job-name=fone_sft_0p5b_gsm8k
#SBATCH --account=bfdj-delta-gpu
#SBATCH --partition=gpuA100x8
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:8
#SBATCH --mem=200G
#SBATCH --time=8:00:00
#SBATCH --output=logs/sft_0p5b_gsm8k_%j.out
#SBATCH --error=logs/sft_0p5b_gsm8k_%j.err

# FoNE SFT: Llama 0.5B on GSM8K with 8 GPUs (SLURM)
# Usage: sbatch --export=PRETRAINED_MODEL=/path/to/model scripts/sft_0p5b_8gpu.sbatch

set -e

# Check arguments
if [ -z "$PRETRAINED_MODEL" ]; then
    echo "❌ Error: Please provide PRETRAINED_MODEL environment variable"
    echo "Usage: sbatch --export=PRETRAINED_MODEL=/path/to/model scripts/sft_0p5b_4gpu.sbatch"
    exit 1
fi

# Create logs directory
mkdir -p logs

# Environment setup
module load anaconda3_gpu cuda
eval "$(conda shell.bash hook)"
conda activate fone

# Load environment variables from .env file
if [ -f .env ]; then
    export $(cat .env | grep -v '^#' | xargs)
    echo "✅ Loaded HuggingFace token from .env"
fi

# Load project environment variables
export PROJECT_ID="bfdj"
export PROJECT_DIR="/projects/${PROJECT_ID}/${USER}"
export WORK_NVME_DIR="/work/nvme/${PROJECT_ID}/${USER}"
export WORK_HDD_DIR="/work/hdd/${PROJECT_ID}/${USER}"

# DeepSpeed environment variables (prevent compilation)
export DS_BUILD_OPS=0
export DS_SKIP_CUDA_CHECK=1
export DS_BUILD_CPU_ADAM=0
export DS_BUILD_FUSED_ADAM=0
export DS_BUILD_FUSED_LAMB=0
export DS_BUILD_SPARSE_ATTN=0
export DS_BUILD_TRANSFORMER=0
export DS_BUILD_STOCHASTIC_TRANSFORMER=0
export DS_BUILD_UTILS=0

# Performance settings
export TOKENIZERS_PARALLELISM=false
export TRITON_CACHE_DIR=/tmp/triton_cache_${SLURM_JOB_ID}
export OMP_NUM_THREADS=8

# Experiment configuration
MODEL_SIZE="0p5b"
NUM_GPUS=8
BATCH_SIZE=4
GRAD_ACCUM=1  # Reduced for 8 GPUs
LEARNING_RATE=1e-4
NUM_EPOCHS=3
SAVE_STEPS=1000
LOG_STEPS=50

# Paths
ACCELERATE_CONFIG="configs/accelerate_8gpu.yaml"
# Use HDD storage for checkpoints (recommended scratch volume)
if [ -n "$WORK_HDD_DIR" ] && [ -d "$(dirname "$WORK_HDD_DIR")" ]; then
    OUTPUT_DIR="${WORK_HDD_DIR}/outputs/sft_${MODEL_SIZE}_gsm8k_$(date +%Y%m%d_%H%M%S)"
else
    OUTPUT_DIR="outputs/sft_${MODEL_SIZE}_gsm8k_$(date +%Y%m%d_%H%M%S)"
fi

echo "🚀 Starting FoNE SFT on GSM8K (SLURM Job: $SLURM_JOB_ID)"
echo "Model: Llama ${MODEL_SIZE}"
echo "GPUs: ${NUM_GPUS}"
echo "Pretrained: ${PRETRAINED_MODEL}"
echo "Output: ${OUTPUT_DIR}"
echo "Node: $SLURMD_NODENAME"

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Launch SFT
accelerate launch \
    --config_file "$ACCELERATE_CONFIG" \
    src/train/sft_lora.py \
    --model_name_or_path "$PRETRAINED_MODEL" \
    --dataset "gsm8k" \
    --train_split "train" \
    --output_dir "$OUTPUT_DIR" \
    --learning_rate "$LEARNING_RATE" \
    --per_device_train_batch_size "$BATCH_SIZE" \
    --gradient_accumulation_steps "$GRAD_ACCUM" \
    --num_train_epochs "$NUM_EPOCHS" \
    --save_steps "$SAVE_STEPS" \
    --logging_steps "$LOG_STEPS" \
    --warmup_ratio 0.03 \
    --weight_decay 0.01 \
    --max_seq_length 1024 \
    --bf16 \
    --lora_r 16 \
    --lora_alpha 32 \
    --lora_dropout 0.05 \
    --target_modules "q_proj,v_proj,k_proj,o_proj,gate_proj,up_proj,down_proj" \
    --wandb_run_name "llama${MODEL_SIZE}_fone_sft_gsm8k_${SLURM_JOB_ID}"

echo "✅ SFT completed!"
echo "LoRA adapter saved in: $OUTPUT_DIR"

# Cleanup
rm -rf /tmp/triton_cache_${SLURM_JOB_ID}
