# FoNE Refined Pipeline Guide

This guide provides a walkthrough for FoNE pretraining (0.5B and 1.5B) and evaluation via perplexity and text generation.

## 🎯 Experiment Overview

The pipeline consists of two stages:

1. **Pretrain** (0.5B or 1.5B)
2. **Evaluate** (perplexity + text generation)  

Each stage can be run in two modes:
- **Interactive (2-GPU)**: For development and debugging
- **SLURM (4-GPU)**: For production runs

## 📁 Pipeline Structure

```
fone-pretrain/
├── configs/
│   ├── llama_0p5b.json          # 0.5B model config
│   ├── llama_1p5b.json          # 1.5B model config
│   ├── accelerate_2gpu.yaml     # 2-GPU DeepSpeed config
│   ├── accelerate_4gpu.yaml     # 4-GPU DeepSpeed config
│   └── data_mixture.hf.json     # Training data mixture
├── scripts/
│   ├── pretrain_0p5b_2gpu.sh    # 0.5B pretrain (interactive)
│   ├── pretrain_0p5b_4gpu.sbatch # 0.5B pretrain (SLURM)
│   ├── pretrain_1p5b_2gpu.sh    # 1.5B pretrain (interactive)
│   ├── pretrain_1p5b_4gpu.sbatch # 1.5B pretrain (SLURM)
│   └── consolidate_and_eval.py  # Consolidate + text-gen eval helper
└── outputs/                     # All experiment outputs
```

## 🚀 Quick Start

### Prerequisites

```bash
# Setup environment
module load anaconda3_gpu cuda
eval "$(conda shell.bash hook)"
conda activate fone
cd /path/to/fone-pretrain
```

### Experiment 1: Llama 0.5B

#### Step 1: Pretraining

**Interactive (2-GPU):**
```bash
bash scripts/pretrain_0p5b_2gpu.sh
```

**SLURM (4-GPU):**
```bash
sbatch scripts/pretrain_0p5b_4gpu.sbatch
```

#### Step 2: Evaluation

Perplexity:
```bash
python src/eval/perplexity_eval.py \
  --model /work/hdd/.../outputs/pretrain_0p5b_YYYYMMDD_HHMMSS/step_150000 \
  --validation_data configs/validation_data.json \
  --seq_len 2048 --max_samples 1000 \
  --output_file eval/pretrain_0p5b_.../step_150000/perplexity.json
```

Text generation:
```bash
python eval/consolidate.py \
  --ckpt /work/hdd/.../outputs/pretrain_0p5b_YYYYMMDD_HHMMSS/step_150000 \
  --config configs/llama_0p5b.json \
  --prompts_file configs/validation_prompts.json \
  --num_samples 1 \
  --eval_root eval
```

### Experiment 2: Llama 1.5B

#### Step 1: Pretraining

**Interactive (2-GPU):**
```bash
bash scripts/pretrain_1p5b_2gpu.sh
```

**SLURM (4-GPU):**
```bash
sbatch scripts/pretrain_1p5b_4gpu.sbatch
```

#### Step 2: Evaluation

Perplexity and text generation: see 0.5B examples above (swap configs/paths for 1.5B).

## ⚙️ Configuration Details

### Model Architectures

| Parameter | 0.5B Model | 1.5B Model |
|-----------|------------|------------|
| Layers | 16 | 26 |
| Hidden Size | 1024 | 2048 |
| Attention Heads | 16 | 16 |
| Intermediate Size | 2752 | 5504 |
| Parameters | ~0.5B | ~1.5B |

### Training Settings (pretraining)

#### Pretraining
- **Dataset**: FineWeb (65%) + OpenWebMath (25%) + The Stack v2 (10%)
- **Sequence Length**: 2048 tokens
- **Learning Rate**: 3e-4
- **Training Steps**: 50,000
- **Warmup Steps**: 1,000

#### Evaluation
- Perplexity: configs/validation_data.json (or HF datasets)
- Text generation: configs/validation_prompts.json (diverse prompts)

### Resource Requirements

| Configuration | GPUs | Memory/GPU | Time (Est.) | Batch Size |
|---------------|------|------------|-------------|------------|
| 0.5B Pretrain (2-GPU) | 2 | ~15GB | 24h | 2 |
| 0.5B Pretrain (4-GPU) | 4 | ~12GB | 12h | 2 |
| 1.5B Pretrain (2-GPU) | 2 | ~20GB | 48h | 2 |
| 1.5B Pretrain (4-GPU) | 4 | ~15GB | 24h | 2 |
| 0.5B SFT (2-GPU) | 2 | ~8GB | 4h | 4 |
| 0.5B SFT (4-GPU) | 4 | ~6GB | 2h | 4 |
| 1.5B SFT (2-GPU) | 2 | ~12GB | 8h | 2 |
| 1.5B SFT (4-GPU) | 4 | ~8GB | 4h | 2 |

## 📊 Expected Results

### Performance Baselines

| Model | Setting | GSM8K 0-shot | GSM8K 5-shot |
|-------|---------|--------------|--------------|
| 0.5B Pretrained | Baseline | 5-15% | 10-20% |
| 0.5B + SFT | Expected | 20-35% | 25-40% |
| 1.5B Pretrained | Baseline | 10-25% | 15-30% |
| 1.5B + SFT | Expected | 30-50% | 35-55% |

*Note: These are rough estimates. FoNE's number embeddings may provide additional benefits for mathematical reasoning.*

## 🔍 Monitoring and Debugging

### Key Metrics to Watch

1. **Pretraining**:
   - Training loss should decrease steadily
   - Perplexity should improve over time
   - Watch for loss spikes (may indicate learning rate issues)

2. **SFT**:
   - Loss should decrease quickly (supervised learning)
   - Monitor for overfitting after epoch 2-3

3. **Evaluation**:
   - Compare 0-shot vs few-shot performance
   - Look for consistent improvement with more examples

### Troubleshooting

**Common Issues**:

1. **Out of Memory**: Reduce batch size or use gradient checkpointing
2. **DeepSpeed Errors**: Check environment variables are set correctly
3. **Dataset Loading**: Ensure data_mixture.hf.json is accessible
4. **Slow Training**: Monitor GPU utilization and data loading

**Log Files**:
- Interactive: Output printed to terminal
- SLURM: Check `logs/` directory for job outputs

## 📈 Analysis and Comparison

After running all experiments, compare:

1. **Model Size Impact**: How much does 3x parameters (0.5B → 1.5B) improve performance?
2. **SFT Effectiveness**: How much does fine-tuning improve each model?
3. **Few-Shot Learning**: Do larger models benefit more from examples?
4. **FoNE Benefits**: How do number embeddings help mathematical reasoning?

## 🎯 Next Steps

After completing the pipeline:

1. **Error Analysis**: Examine failed GSM8K problems
2. **Ablation Studies**: Try different LoRA configurations
3. **Additional Datasets**: Evaluate on MATH, ARC, etc.
4. **Scaling**: Test larger models or longer training

## 📝 Notes

- All outputs are timestamped and saved in `outputs/`
- Configs are automatically copied to output directories for reproducibility
- WandB logging is enabled for experiment tracking
- Models are saved with DeepSpeed-compatible checkpoints
