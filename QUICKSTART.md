# FoNE Quick Start Guide

Simple guide to pretrain, finetune, and evaluate FoNE models.

## 🎯 Four Simple Commands

### 1. Pretrain a Model

**FoNE Model** (with number embeddings):
```bash
# Quick sanity check
bash scripts/pretrain.sh sanity

# Full 1.5B model training (compute-optimal)
bash scripts/pretrain.sh 1p5b-1x --gpus 8

# Data-rich training (2x data)
bash scripts/pretrain.sh 1p5b-2x --gpus 8
```

**Baseline Model** (standard LLaMA, no FoNE):
```bash
# For comparison - same architecture, same data, NO number embeddings
bash scripts/pretrain_baseline.sh 1p5b-1x
bash scripts/pretrain_baseline.sh sanity  # Quick test
```

### 2. Finetune on GSM8K
```bash
# Finetune your pretrained model
bash scripts/finetune.sh \
    --model Onlydrinkwater/fone-1p5b-step-76000

# Custom settings
bash scripts/finetune.sh \
    --model Onlydrinkwater/fone-1p5b-step-76000 \
    --epochs 5 \
    --lr 2e-4
```
```bash scripts/finetune.sh \
    --model Onlydrinkwater/fone-1p5b-step-76000 \
    --epochs 1 \
    --lr 5e-5 \
    --lora-r 32 \
    --batch-size 4 \
    --grad-accum 4```
### 3. Evaluate on GSM8K
```bash
# Evaluate single model
bash scripts/eval_gsm8k.sh \
    --model Onlydrinkwater/fone-1p5b-step-76000

# Quick test (100 samples)
bash scripts/eval_gsm8k.sh \
    --model Onlydrinkwater/fone-1p5b-step-76000 \
    --num-samples 100
``` bash scripts/eval_gsm8k.sh     --model finetune/outputs/gsm8k_fone-1p5b-step-76000_20251021_010833/merged_model     --num-samples 10```
# Compare base vs finetuned
bash scripts/eval_gsm8k.sh \
    --model Onlydrinkwater/fone-1p5b-step-76000 \
    --compare-with outputs/gsm8k_*/merged_model
```

### 4. Evaluate Text Generation
```bash
# Test general generation capabilities
bash scripts/eval_generation.sh \
    --model Onlydrinkwater/fone-1p5b-step-76000

# Custom settings
bash scripts/eval_generation.sh \
    --model Onlydrinkwater/fone-1p5b-step-76000 \
    --max-tokens 200 \
    --temperature 0.8
```

## 📁 Project Structure

```
fone-pretrain/
├── scripts/                    # All executable scripts (.sh only)
│   ├── pretrain.sh            # Download data & pretrain
│   ├── finetune.sh            # Finetune on GSM8K
│   ├── eval_gsm8k.sh          # Evaluate on GSM8K
│   ├── eval_generation.sh     # Evaluate text generation
│   ├── pretrain_baseline.sh   # Pretrain baseline (no FoNE)
│   ├── pretrain_0p5b_local.sh # (internal)
│   └── pretrain_1p5b_local.sh # (internal)
│
├── src/                        # Python library code
│   ├── data/                  # Data processing
│   ├── finetune/              # Finetuning code
│   ├── modeling/              # Model architecture (FoNE)
│   ├── train/                 # Training utilities
│   └── eval/                  # Evaluation utilities
│
├── eval/                       # Evaluation scripts & results
│   ├── eval_gsm8k.py          # GSM8K evaluation
│   ├── eval_generation.py     # Generation evaluation
│   └── results/               # Evaluation outputs
│
├── configs/                    # Model and data configs
├── outputs/                    # Training outputs
│   ├── fone_*/                # Pretrained FoNE models
│   ├── baseline_*/            # Baseline models (no FoNE)
│   └── gsm8k_*/               # Finetuned models
└── data/                       # Processed datasets
```

## 🔄 Typical Workflow

### Option 1: Use Existing Pretrained Model
```bash
# 1. Evaluate base model (optional)
bash scripts/eval_gsm8k.sh \
    --model Onlydrinkwater/fone-1p5b-step-76000 \
    --num-samples 100

# 2. Finetune on GSM8K
bash scripts/finetune.sh \
    --model Onlydrinkwater/fone-1p5b-step-76000

# 3. Compare before/after
bash scripts/eval_gsm8k.sh \
    --model Onlydrinkwater/fone-1p5b-step-76000 \
    --compare-with outputs/gsm8k_*/merged_model
```

### Option 2: Pretrain Your Own Model
```bash
# 1. Pretrain
bash scripts/pretrain.sh 1p5b-1x --gpus 8

# 2. Finetune
bash scripts/finetune.sh \
    --model outputs/fone_1p5b_*/step-XXXXX

# 3. Evaluate
bash scripts/eval_gsm8k.sh \
    --model outputs/gsm8k_*/merged_model
```

## 📊 Results

### Evaluation Outputs

**GSM8K Evaluation** (`eval/results/gsm8k_*/`)
- `results.json`: Full results with all generations
- `summary.txt`: Accuracy summary
- `examples.txt`: Sample correct/incorrect outputs

**Generation Evaluation** (`eval/results/eval_*/`)
- `generation_results.json`: All generations in JSON
- `generation_results.txt`: Human-readable results
- `summary.txt`: Quick summary

### Training Outputs

**Pretraining** (`outputs/fone_1p5b_*/`)
- Checkpoints saved every N steps
- Final model ready for finetuning or inference

**Finetuning** (`outputs/gsm8k_*/`)
- `merged_model/`: Full model (use for inference) ⭐
- `final_model/`: LoRA adapters only
- `checkpoint-*/`: Training checkpoints
- `training_info.json`: Config and hyperparameters

## 💡 Tips

**Quick Testing**
- Use `--num-samples 100` for fast GSM8K evaluation
- Use `sanity` scale for pretrain testing

**Memory Issues**
- Reduce `--batch-size` 
- Increase `--grad-accum` to maintain effective batch size

**Better Results**
- Increase `--lora-r` to 32 or 64 for finetuning
- Try `--lr 2e-4` or `--lr 5e-5` for different learning rates

**Multi-GPU**
- Scripts automatically detect and use all available GPUs
- Specify with `--gpus N` for pretrain

## 📖 More Help

Each script has detailed help:
```bash
bash scripts/pretrain.sh --help
bash scripts/finetune.sh --help
bash scripts/eval_gsm8k.sh --help
bash scripts/eval_generation.sh --help
```

## 🎓 Expected Performance

**Base Pretrained Model (Before Finetuning)**
- GSM8K: 0-5% accuracy (no reasoning)
- Generation: Can produce fluent text, but struggles with math

**After GSM8K Finetuning**
- GSM8K: 30-50% accuracy (learns step-by-step reasoning)
- Generation: Improved at showing reasoning steps

**Reference**: GPT-3.5 achieves ~57% on GSM8K

## 🚀 Quick Examples

```bash
# Complete workflow
bash scripts/pretrain.sh sanity
bash scripts/eval_generation.sh --model outputs/fone_*/step-1000
bash scripts/finetune.sh --model outputs/fone_*/step-1000
bash scripts/eval_gsm8k.sh --model outputs/gsm8k_*/merged_model --num-samples 100

# Use existing checkpoint
bash scripts/finetune.sh --model Onlydrinkwater/fone-1p5b-step-76000
bash scripts/eval_gsm8k.sh \
    --model Onlydrinkwater/fone-1p5b-step-76000 \
    --compare-with outputs/gsm8k_*/merged_model
```

