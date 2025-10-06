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
├── accelerate_config.yaml       # DeepSpeed ZeRO-3 configuration
├── VALIDATION.md                # Validation system documentation
├── configs/
│   ├── llama_1p5b_fone.json    # Model configuration
│   ├── data_mixture.json       # Training data mixture
│   ├── validation_data.json    # Validation data configuration
│   └── validation_prompts.json # Text generation prompts
├── src/
│   ├── data/                   # Data loading and processing
│   ├── modeling/               # FoNE model implementation
│   ├── train/                  # Training scripts
│   └── eval/                   # Validation and evaluation
└── scripts/
    ├── train_debug.sh/.sbatch  # Debug training (quick test)
    ├── train_full.sh/.sbatch   # Full training (production)
    ├── manual_validation.sh    # Validate existing checkpoints
    └── eval_gsm8k.sbatch       # Math reasoning evaluation
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

**Debug Training** (recommended first - 10M tokens, 2 hours):
```bash
# Local/interactive
./scripts/train_debug.sh

# Or submit to SLURM
sbatch scripts/train_debug.sbatch
```

**Full Training** (production - 30B tokens, 48 hours):
```bash
# Local/interactive  
./scripts/train_full.sh

# Or submit to SLURM
sbatch scripts/train_full.sbatch
```

### 3. Validate Your Model

```bash
# Test any checkpoint
./scripts/manual_validation.sh /path/to/checkpoint step_number
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

### Training Parameters

| Parameter | Debug | Full |
|-----------|-------|------|
| Total Tokens | 10M | 30B |
| Sequence Length | 512 | 2048 |
| Learning Rate | 2e-4 | 2e-4 |
| Batch Size | 2 | 2 |
| GPUs | 1 | 4 |
| Time | ~2h | ~48h |
| Validation | Every 200 steps | Every 2000 steps |

### Model Architecture

- **Layers**: 26
- **Hidden Size**: 2048  
- **Attention Heads**: 16
- **Vocab Size**: 128,256
- **FoNE Numbers**: 0-999 (frozen Fourier embeddings)
- **Parameters**: ~1.5B total

## 🏗️ Technical Details

### FoNE Implementation

FoNE replaces standard embeddings for numbers 0-999 with:
1. **Fourier Features**: `[cos(2πkn/1000), sin(2πkn/1000)]` for k=0,1,2,...
2. **Frozen Weights**: Fourier features are not updated during training
3. **Learnable Projection**: Maps Fourier features to model dimension

### Training Infrastructure

- **DeepSpeed ZeRO-3**: CPU offloading for memory efficiency
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

After training, evaluate mathematical reasoning:

```bash
sbatch scripts/eval_gsm8k.sbatch
```

This runs GSM8K evaluation with self-consistency for robust math performance assessment.

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
  author={Your Name},
  year={2024},
  url={https://github.com/yourusername/fone-pretrain}
}
```
