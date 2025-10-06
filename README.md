# FoNE Pretraining Project

This repository implements **FoNE (Fourier Number Embedding)** for pretraining a 1.5B parameter Llama-style language model from scratch. FoNE replaces token embeddings for numbers 0-999 with frozen Fourier-based features passed through a learnable projection layer, designed to improve numerical reasoning capabilities.

## 🌟 Key Features

- **FoNE Integration**: Frozen Fourier number embeddings for numbers 0-999
- **Llama 1.5B Architecture**: Custom 26-layer, 2048-hidden Llama model
- **DeepSpeed ZeRO-3**: Memory-efficient training with CPU offloading
- **FlashAttention-2**: Optional fast attention implementation
- **Comprehensive Validation**: Perplexity + text generation quality assessment
- **Delta/SLURM Support**: Ready-to-use NCSA Delta cluster scripts

## 📁 Repository Structure

```
fone-pretrain/
├── README.md                    # This file
├── requirements.txt             # Python dependencies
├── setup_env.sh                 # Environment variables and outputs roots
├── configs/
│   ├── accelerate_2gpu.yaml     # Accelerate/DeepSpeed config (2 GPUs)
│   ├── accelerate_8gpu.yaml     # Accelerate/DeepSpeed config (8 GPUs)
│   ├── llama_0p5b.json          # 0.5B model config
│   ├── llama_1p5b.json          # 1.5B model config
│   ├── data_mixture.hf.json     # Training data mixture
│   ├── validation_data.json     # Perplexity eval data config
│   └── validation_prompts.json  # Text generation prompts
├── src/
│   ├── data/                    # Data loading and processing
│   ├── modeling/                # FoNE model implementation
│   ├── train/                   # Training entrypoints
│   └── eval/                    # Validation / evaluation entrypoints
├── eval/
│   └── consolidate.py           # Consolidate ZeRO-3 ckpt and run text-gen eval
└── scripts/
    ├── pretrain_0p5b_2gpu.sh    # 0.5B pretraining (interactive)
    ├── pretrain_1p5b_2gpu.sh    # 1.5B pretraining (interactive)
    ├── pretrain_0p5b_8gpu.sbatch # 0.5B pretraining (SLURM)
    ├── pretrain_1p5b_8gpu.sbatch # 1.5B pretraining (SLURM)
    └── consolidate_and_eval.py  # Consolidate + run text-gen eval helper
```

## 🚀 Quick Start

### 1. Setup Environment

```bash
# Create conda environment
conda create -n fone python=3.10
conda activate fone

# Install requirements
pip install -r requirements.txt
```

### 2. Run Training

Interactive (2 GPUs):
```bash
# 0.5B
bash scripts/pretrain_0p5b_2gpu.sh

# 1.5B
bash scripts/pretrain_1p5b_2gpu.sh
```

SLURM (8 GPUs):
```bash
# 0.5B
sbatch scripts/pretrain_0p5b_8gpu.sbatch

# 1.5B
sbatch scripts/pretrain_1p5b_8gpu.sbatch
```

### 3. Validate Your Model (Perplexity + Text Generation)

Text generation quick-check (consolidate ZeRO-3 -> single model, then run eval):
```bash
python eval/consolidate.py \
  --ckpt /path/to/outputs/pretrain_*/step_150000 \
  --config configs/llama_1p5b.json \
  --prompts_file configs/validation_prompts.json \
  --num_samples 1 \
  --eval_root /path/to/eval_root   # or export FONE_EVAL_DIR
```

Perplexity or text-gen evaluation directly:
```bash
# Perplexity
python src/eval/perplexity_eval.py \
  --model /path/to/model_or_ckpt_dir \
  --validation_data configs/validation_data.json \
  --seq_len 2048 --max_samples 1000 \
  --output_file eval/perplexity.json

# Text generation
python src/eval/text_generation_eval.py \
  --model /path/to/model_or_ckpt_dir \
  --prompts_file configs/validation_prompts.json \
  --num_samples 1 --max_new_tokens 200 \
  --output_file eval/text_gen.json
```

## 🔍 Validation System

The training includes comprehensive validation to ensure your model learns to generate coherent language:

- **Perplexity Evaluation**: Quantitative measure on held-out data
- **Text Generation Quality**: 20 diverse prompts testing different capabilities
- **Progress Tracking**: Comparison with previous checkpoints
- **Manual Inspection**: Sample generations displayed for human review

See [VALIDATION.md](VALIDATION.md) for complete details.

## 📊 What to Look For

**✅ Good Signs (Model is Learning):**
- Decreasing perplexity over time
- Coherent, on-topic text completions  
- Low repetition in generated text
- Quality scores improving

**❌ Warning Signs:**
- Highly repetitive text (same phrases repeated)
- Nonsensical or incoherent responses
- Consistently increasing perplexity

## 🔧 Configuration

### Training Parameters (from scripts/configs)

- Sequence length: 2048 (default; configurable via `--seq_len`)
- Optimizer: AdamW (`weight_decay` 0.1)
- Warmup steps: 1000
- Logging every: 100 steps (`--log_every_steps`)
- Checkpoint every: 10000 steps (`--save_every_steps`)
- Mixed precision: bfloat16 (Accelerate configs)

Interactive 2-GPU defaults (configs/accelerate_2gpu.yaml):
- Per-device micro-batch: 8
- Grad accumulation: 4
- Effective micro-batch per step: 8 × 4 × 2 GPUs = 64 sequences

8-GPU SLURM defaults:
- 0.5B (`scripts/pretrain_0p5b_8gpu.sbatch`): `--per_device_train_batch_size 24`, `--gradient_accumulation_steps 4`
- 1.5B (`scripts/pretrain_1p5b_8gpu.sbatch`): per-device micro-batch 2, grad accumulation 2 (from `configs/accelerate_8gpu.yaml`)

Total tokens targets (approximate):
- 0.5B: 10B tokens (`--total_tokens 10000000000`)
- 1.5B: 20B tokens (`--total_tokens 20000000000`)

Learning rate:
- 0.5B 8-GPU: `2e-4`
- Other scripts: `3e-4`

### Model Architecture

0.5B (`configs/llama_0p5b.json`):
- Layers: 16
- Hidden size: 1024
- Attention heads: 16
- Intermediate size: 2752

1.5B (`configs/llama_1p5b.json`):
- Layers: 26
- Hidden size: 2048
- Attention heads: 16
- Intermediate size: 5504

Tokenizer: default `meta-llama/Llama-3.1-8B` (configurable via `--tokenizer`); its vocab size is 128256.

## 🏗️ Technical Details

### FoNE Implementation

FoNE replaces standard embeddings for numbers 0-999 with:
1. **Fourier Features**: `[cos(2πkn/1000), sin(2πkn/1000)]` for k=0,1,2,...
2. **Frozen Weights**: Fourier features are not updated during training
3. **Learnable Projection**: Maps Fourier features to model dimension

### Training Infrastructure

- **DeepSpeed ZeRO-3**: Enabled via Accelerate configs (CPU offloading enabled on 8-GPU config)
- **FlashAttention-2**: Optional fast attention (full training only)
- **Gradient Accumulation**: 16 steps
- **Mixed Precision**: bfloat16
- **Activation Checkpointing**: Enabled for memory savings

## 📈 Monitoring

Training automatically logs to:
- **Weights & Biases**: Loss, perplexity, validation metrics
- **Console**: Progress, ETA, sample generations
- **Files**: Detailed validation results and checkpoints

## 🔬 Evaluation

This repo focuses on: (1) pretraining and (2) perplexity + text-generation evaluation. SFT and GSM8K scripts were removed to keep scope minimal.

## 🐛 Troubleshooting

### Common Issues

1. **"CUDA out of memory"**: Use debug training first, or reduce batch size
2. **"Validation data not found"**: Check that config files exist
3. **"Model generates empty text"**: Verify tokenizer and model checkpoint

### Performance Tips

- Start with debug training to test the full pipeline
- Monitor sample generations manually - they're more intuitive than metrics
- Use validation recommendations to guide training decisions

## 📝 Citation

If you use this code, please cite:

```bibtex
@misc{fone-pretraining,
  title={FoNE: Fourier Number Embeddings for Language Model Pretraining},
  author={Zhou, Tianyi},
  year={2025},
  url={https://github.com/KevinZhoutianyi/fone-pretrain}
}
```
