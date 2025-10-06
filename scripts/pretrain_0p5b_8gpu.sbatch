#!/bin/bash
#SBATCH --job-name=fone_pretrain_0p5b
#SBATCH --account=bfdj-delta-gpu
#SBATCH --partition=gpuA100x8
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:8
#SBATCH --mem=200G
#SBATCH --time=1-00:00:00
#SBATCH --output=logs/pretrain_0p5b_%j.out
#SBATCH --error=logs/pretrain_0p5b_%j.err

# FoNE Pretraining: Llama 0.5B on 8 GPUs (SLURM)
# Usage: sbatch scripts/pretrain_0p5b_8gpu.sbatch

set -e

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
BATCH_SIZE_PER_GPU=24  # per-device micro-batch size
GRAD_ACCUM=4         # gradient accumulation steps
LEARNING_RATE=2e-4
MAX_STEPS=50000
SAVE_STEPS=10000
LOG_STEPS=100

# Paths
MODEL_CONFIG="configs/llama_${MODEL_SIZE}.json"
ACCELERATE_CONFIG="configs/accelerate_8gpu.yaml"
DATA_CONFIG="configs/data_mixture.hf.json"
# Use HDD storage for checkpoints (recommended scratch volume)
if [ -n "$WORK_HDD_DIR" ] && [ -d "$(dirname "$WORK_HDD_DIR")" ]; then
    OUTPUT_DIR="${WORK_HDD_DIR}/outputs/pretrain_${MODEL_SIZE}_$(date +%Y%m%d_%H%M%S)"
else
    OUTPUT_DIR="outputs/pretrain_${MODEL_SIZE}_$(date +%Y%m%d_%H%M%S)"
fi

echo "🚀 Starting FoNE Pretraining (SLURM Job: $SLURM_JOB_ID)"
echo "Model: Llama ${MODEL_SIZE}"
echo "GPUs: ${NUM_GPUS}"
echo "Output: ${OUTPUT_DIR}"
echo "Config: ${MODEL_CONFIG}"
echo "Node: $SLURMD_NODENAME"

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Copy configs for reproducibility
cp "$MODEL_CONFIG" "$OUTPUT_DIR/"
cp "$ACCELERATE_CONFIG" "$OUTPUT_DIR/"
cp "$DATA_CONFIG" "$OUTPUT_DIR/"

# Launch training
accelerate launch \
    --config_file "$ACCELERATE_CONFIG" \
    src/train/pretrain.py \
    --model_config "$MODEL_CONFIG" \
    --dataset "$DATA_CONFIG" \
    --output_dir "$OUTPUT_DIR" \
    --lr "$LEARNING_RATE" \
    --warmup_steps 1000 \
    --weight_decay 0.1 \
    --per_device_train_batch_size "$BATCH_SIZE_PER_GPU" \
    --gradient_accumulation_steps "$GRAD_ACCUM" \
    --save_every_steps "$SAVE_STEPS" \
    --log_every_steps "$LOG_STEPS" \
    --total_tokens 10000000000 \
    --dataloader_num_workers 4 \
    --activation_checkpointing \
    --wandb_run_name "llama${MODEL_SIZE}_fone_pretrain_${SLURM_JOB_ID}"

echo "✅ Pretraining completed!"
echo "Model saved in: $OUTPUT_DIR"

# Cleanup
rm -rf /tmp/triton_cache_${SLURM_JOB_ID}
