# FoNE Pretraining Guide

Complete guide for pretraining both **FoNE** (with Fourier number embeddings) and **Baseline** (standard) models.

## 🎯 Quick Start

### Option 1: Using the Pretraining Script (Recommended)

```bash
# Activate environment
conda activate fone

# Quick sanity check (2 min, ~137M tokens)
bash scripts/pretrain_0p5b_local.sh

# Small test (15 min, ~7B tokens)
DATA_CONFIG=configs/data_local_small.json bash scripts/pretrain_0p5b_local.sh

# Full pretraining (1.5B model, 30B tokens)
DATA_CONFIG=configs/data_local_30b.json \
MODEL_SIZE=1p5b \
NUM_PROCESSES=8 \
bash scripts/pretrain_0p5b_local.sh
```

### Option 2: Direct Python Script

```bash
# FoNE model (1.5B with Fourier number embeddings)
accelerate launch \
  --num_processes 8 \
  --mixed_precision bf16 \
  src/train/pretrain.py \
  --model_config configs/llama_1p5b.json \
  --dataset configs/data_local_30b.json \
  --output_dir outputs/fone_1p5b \
  --lr 3e-4 \
  --warmup_steps 3000 \
  --weight_decay 0.1 \
  --save_every_steps 10000 \
  --wandb_project fone-pretraining \
  --wandb_run_name fone_1p5b_30b

# Baseline model (1.5B without FoNE, standard embeddings)
accelerate launch \
  --num_processes 8 \
  --mixed_precision bf16 \
  src/train/pretrain.py \
  --model_config configs/llama_1p5b_baseline.json \
  --dataset configs/data_local_30b.json \
  --output_dir outputs/baseline_1p5b \
  --lr 3e-4 \
  --warmup_steps 3000 \
  --weight_decay 0.1 \
  --save_every_steps 10000 \
  --wandb_project fone-pretraining \
  --wandb_run_name baseline_1p5b_30b
```

## 📊 Model Configurations

### FoNE vs Baseline

| Feature | FoNE | Baseline |
|---------|------|----------|
| Config | `llama_1p5b.json` | `llama_1p5b_baseline.json` |
| Number Embeddings | Frozen Fourier features (0-999) | Standard learned embeddings |
| Architecture | 26 layers, 2048 hidden | 26 layers, 2048 hidden |
| Parameters | ~1.5B | ~1.5B |
| Purpose | Better numerical reasoning | Standard comparison |

### Model Sizes Available

| Size | Config | Layers | Hidden | Parameters |
|------|--------|--------|--------|------------|
| 0.5B | `llama_0p5b.json` | 16 | 1024 | ~500M |
| 1.5B | `llama_1p5b.json` | 26 | 2048 | ~1.5B |

## 📁 Data Configurations

| Config | Tokens | Time (8xGPU) | Use Case |
|--------|--------|--------------|----------|
| `data_local_sanity.json` | 137M | 2 min | Quick test |
| `data_local_small.json` | 7B | 15 min | Fast iteration |
| `data_local_1p5b_1x.json` | 15B | 30 min | Medium |
| `data_local_30b.json` | 30B | 1 hour | Full pretraining |
| `data_local_1p5b_2x.json` | 30B | 1 hour | Extended |

## 🚀 Full Pretraining Examples

### 1. FoNE Model (Full 30B tokens)

```bash
# With auto-upload to HuggingFace
accelerate launch \
  --num_processes 8 \
  --mixed_precision bf16 \
  src/train/pretrain.py \
  --model_config configs/llama_1p5b.json \
  --dataset configs/data_local_30b.json \
  --output_dir outputs/fone_1p5b_30b \
  --lr 3e-4 \
  --warmup_steps 3000 \
  --weight_decay 0.1 \
  --per_device_train_batch_size 6 \
  --gradient_accumulation_steps 4 \
  --save_every_steps 10000 \
  --log_every_steps 10 \
  --wandb_project fone-pretraining \
  --wandb_run_name fone_1p5b_30b_$(date +%Y%m%d)
```

### 2. Baseline Model (Full 30B tokens)

```bash
# Standard model without FoNE
accelerate launch \
  --num_processes 8 \
  --mixed_precision bf16 \
  src/train/pretrain.py \
  --model_config configs/llama_1p5b_baseline.json \
  --dataset configs/data_local_30b.json \
  --output_dir outputs/baseline_1p5b_30b \
  --lr 3e-4 \
  --warmup_steps 3000 \
  --weight_decay 0.1 \
  --per_device_train_batch_size 6 \
  --gradient_accumulation_steps 4 \
  --save_every_steps 10000 \
  --log_every_steps 10 \
  --wandb_project fone-pretraining \
  --wandb_run_name baseline_1p5b_30b_$(date +%Y%m%d)
```

### 3. Run Both in Background

```bash
# Start FoNE training
nohup accelerate launch --num_processes 8 --mixed_precision bf16 \
  src/train/pretrain.py \
  --model_config configs/llama_1p5b.json \
  --dataset configs/data_local_30b.json \
  --output_dir outputs/fone_1p5b_30b \
  --lr 3e-4 --warmup_steps 3000 --weight_decay 0.1 \
  --save_every_steps 10000 \
  --wandb_project fone-pretraining \
  --wandb_run_name fone_1p5b_30b \
  > fone_train.log 2>&1 &

# Monitor progress
tail -f fone_train.log

# Check if running
ps aux | grep pretrain
```

## 🔧 Environment Variables

Control training via environment variables:

```bash
# Quick pretraining with custom settings
DATA_CONFIG=configs/data_local_small.json \
NUM_PROCESSES=4 \
PER_DEVICE_BATCH_SIZE=4 \
GRADIENT_ACCUMULATION_STEPS=2 \
LEARNING_RATE=5e-4 \
WANDB_RUN_NAME=my_custom_run \
bash scripts/pretrain_0p5b_local.sh
```

Available variables:
- `DATA_CONFIG` - Data configuration file
- `NUM_PROCESSES` - Number of GPUs (default: auto-detect)
- `PER_DEVICE_BATCH_SIZE` - Batch size per GPU (default: 2)
- `GRADIENT_ACCUMULATION_STEPS` - Gradient accumulation (default: 1)
- `LEARNING_RATE` - Learning rate (default: 3e-4)
- `WARMUP_STEPS` - Warmup steps (default: 1000)
- `WEIGHT_DECAY` - Weight decay (default: 0.1)
- `WANDB_RUN_NAME` - WandB run name
- `OUT_DIR` - Output directory

## 📊 Monitoring Training

### 1. Check Progress

```bash
# View live logs
tail -f outputs/pretrain_*/training.log

# Or if running with nohup
tail -f fone_train.log
```

### 2. WandB Dashboard

Training automatically logs to Weights & Biases:
- Project: `fone-pretraining`
- View at: https://wandb.ai/your-username/fone-pretraining

### 3. Check Checkpoints

```bash
# List saved checkpoints
ls -lh outputs/fone_1p5b_30b/checkpoint-*

# Check latest checkpoint
ls -lhtr outputs/fone_1p5b_30b/checkpoint-* | tail -1
```

## 💾 Checkpoints

Checkpoints are saved every `save_every_steps` (default: 10000):

```
outputs/fone_1p5b_30b/
├── checkpoint-10000/
│   ├── pytorch_model.bin
│   ├── config.json
│   ├── optimizer.pt
│   └── trainer_state.json
├── checkpoint-20000/
└── final/
```

## 🧮 Computing Requirements

### Memory Requirements

| Model | GPUs | Batch Size | Memory per GPU |
|-------|------|------------|----------------|
| 0.5B | 1 | 2 | ~12 GB |
| 1.5B | 4 | 4 | ~20 GB |
| 1.5B | 8 | 6 | ~24 GB |

### Training Time Estimates (30B tokens)

| Setup | Time |
|-------|------|
| 1x A100 (80GB) | ~8 hours |
| 4x A100 (80GB) | ~2 hours |
| 8x A100 (80GB) | ~1 hour |

## 🔍 What is FoNE?

FoNE (Fourier Number Embedding) replaces standard token embeddings for numbers (0-999) with:

1. **Fourier Features**: 6-dimensional features based on digit decomposition
   - `[cos(x0/10), sin(x0/10), cos(x1/10), sin(x1/10), cos(x2/10), sin(x2/10)]`
   - where x0, x1, x2 are units, tens, hundreds digits

2. **Zero Padding**: Fourier features are padded with zeros to match hidden dimension

3. **Frozen Embeddings**: Number embeddings stay fixed during training (no gradient updates)

**Why?** Better structural understanding of numbers for improved numerical reasoning tasks.

## 🐛 Troubleshooting

### Out of Memory

```bash
# Reduce batch size
PER_DEVICE_BATCH_SIZE=2 bash scripts/pretrain_0p5b_local.sh

# Or use gradient accumulation
PER_DEVICE_BATCH_SIZE=2 GRADIENT_ACCUMULATION_STEPS=4 bash scripts/pretrain_0p5b_local.sh
```

### Data Not Found

Make sure you have the data in the right location:
```bash
# Check data config
cat configs/data_local_30b.json

# Verify data paths exist
ls -lh data/
```

### CUDA Out of Memory

```bash
# Use smaller model
MODEL_SIZE=0p5b bash scripts/pretrain_0p5b_local.sh

# Or fewer GPUs
NUM_PROCESSES=4 bash scripts/pretrain_0p5b_local.sh
```

## 📈 Best Practices

1. **Start Small**: Test with `data_local_sanity.json` first
2. **Monitor WandB**: Check loss curves and learning rate
3. **Save Often**: Use `--save_every_steps 5000` for important runs
4. **Baseline Comparison**: Always train a baseline model for fair comparison
5. **Log Everything**: Keep detailed logs for reproducibility

## 🎓 Research Workflow

```bash
# 1. Quick sanity check
DATA_CONFIG=configs/data_local_sanity.json bash scripts/pretrain_0p5b_local.sh

# 2. Small-scale experiment
DATA_CONFIG=configs/data_local_small.json bash scripts/pretrain_0p5b_local.sh

# 3. Full pretraining (FoNE)
nohup accelerate launch --num_processes 8 --mixed_precision bf16 \
  src/train/pretrain.py \
  --model_config configs/llama_1p5b.json \
  --dataset configs/data_local_30b.json \
  --output_dir outputs/fone_1p5b \
  > fone_train.log 2>&1 &

# 4. Full pretraining (Baseline for comparison)
nohup accelerate launch --num_processes 8 --mixed_precision bf16 \
  src/train/pretrain.py \
  --model_config configs/llama_1p5b_baseline.json \
  --dataset configs/data_local_30b.json \
  --output_dir outputs/baseline_1p5b \
  > baseline_train.log 2>&1 &

# 5. Compare results
python src/eval/compare_models.py \
  --model1 outputs/fone_1p5b/final \
  --model2 outputs/baseline_1p5b/final
```

## 📚 Next Steps

After pretraining:
1. **Evaluate**: Test on numerical reasoning benchmarks
2. **Finetune**: Fine-tune on GSM8K (see `UPLOAD_AND_CLEANUP.md`)
3. **Upload**: Share your model on HuggingFace
4. **Compare**: Analyze FoNE vs Baseline performance

## 🔗 Related Files

- Model configs: `configs/llama_*.json`
- Data configs: `configs/data_local_*.json`
- Training script: `src/train/pretrain.py`
- FoNE implementation: `src/modeling/fone_init.py`

