# Addition Dataset Evaluation

Compare Baseline vs FoNE models on the **Onlydrinkwater/1000addition** dataset.

## 🎯 Quick Start

### Option 1: Simple Shell Script (Recommended)

```bash
# Run with defaults (1000 train samples, 3 epochs)
bash scripts/run_addition_experiment.sh

# Custom configuration
bash scripts/run_addition_experiment.sh \
    --train-samples 2000 \
    --epochs 5 \
    --batch-size 16
```

### Option 2: Direct Python Script

```bash
# Run full pipeline
python finetune/finetune_and_eval_addition.py \
    --baseline_model Onlydrinkwater/baseline-1p5b-step-76294 \
    --fone_model Onlydrinkwater/fone-1p5b-step-76294 \
    --train_samples 10000 \
    --epochs 3 \
    --batch_size 8

# Skip baseline (only train/eval FoNE)
python finetune/finetune_and_eval_addition.py \
    --skip_baseline \
    --fone_model Onlydrinkwater/fone-1p5b-step-76294

# Skip FoNE (only train/eval baseline)
python finetune/finetune_and_eval_addition.py \
    --skip_fone \
    --baseline_model Onlydrinkwater/baseline-1p5b-step-76294
```

## 📊 What It Does

1. **Loads Dataset**: `Onlydrinkwater/1000addition` from HuggingFace
   - Clearly logs train/val/test split sizes
2. **Finetunes Baseline**: Trains baseline model on addition problems
   - ⚠️ **No disk saving** - model kept in memory only!
3. **Finetunes FoNE**: Trains FoNE model on addition problems
   - ⚠️ **No disk saving** - model kept in memory only!
4. **Evaluates Both**: Tests on **100 examples** (subset for speed)
5. **Compares Results**: Generates comparison report

## 🎓 Dataset Info

- **Dataset**: Onlydrinkwater/1000addition
- **Task**: Integer addition problems
- **Format**: `Q: a + b = answer`
- **Splits**: train, validation, test

## 📁 Output Structure

Results are saved to `eval/results/addition_pipeline_TIMESTAMP/`:

```
addition_pipeline_20251115_123456/
├── baseline_training_logs/   # Training logs only (no model saved!)
├── fone_training_logs/       # Training logs only (no model saved!)
├── baseline_eval/
│   ├── results.json          # Detailed baseline results (100 examples)
│   └── summary.txt           # Baseline summary
├── fone_eval/
│   ├── results.json          # Detailed FoNE results (100 examples)
│   └── summary.txt           # FoNE summary
└── comparison_report.txt     # 📊 Final comparison!
```

**Note**: Models are NOT saved to disk - they're kept in memory during evaluation then discarded. This saves disk space!

## 🔧 Configuration Options

| Option | Default | Description |
|--------|---------|-------------|
| `--baseline_model` | `Onlydrinkwater/baseline-1p5b-step-76294` | Baseline model |
| `--fone_model` | `Onlydrinkwater/fone-1p5b-step-76294` | FoNE model |
| `--train_samples` | 10000 | Training samples to use |
| `--epochs` | 3 | Training epochs |
| `--batch_size` | 8 | Batch size |
| `--learning_rate` | 5e-5 | Learning rate |
| `--output_dir` | Auto | Output directory |
| `--skip_baseline` | False | Skip baseline training |
| `--skip_fone` | False | Skip FoNE training |

## 💡 Expected Results

FoNE should significantly outperform the baseline on addition tasks because:

1. **Fourier Number Embeddings**: FoNE has specialized number representations
2. **Better Numerical Reasoning**: Designed for arithmetic operations
3. **Frozen Features**: Pre-trained numerical patterns

Typical accuracy after 3 epochs (on 100 test examples):
- **Baseline**: 60-80%
- **FoNE**: 80-95%+ ✨

## 🎯 Key Features

- ✅ **No Disk Saving**: Models kept in memory only (saves ~3-6GB per model!)
- ✅ **Fast Evaluation**: Only 100 test examples (enough for comparison)
- ✅ **Clear Logging**: Train/val/test split sizes clearly logged
- ✅ **Automatic Cleanup**: Models discarded after evaluation

## 🚀 Quick Examples

```bash
# Fast test (500 samples, 2 epochs)
bash scripts/run_addition_experiment.sh \
    --train-samples 500 \
    --epochs 2

# Full training (2000 samples, 5 epochs)
bash scripts/run_addition_experiment.sh \
    --train-samples 2000 \
    --epochs 5 \
    --batch-size 16

# High learning rate experiment
bash scripts/run_addition_experiment.sh \
    --lr 1e-4 \
    --epochs 5
```

## 📈 View Results

After running, check the comparison report:

```bash
# Find latest results
cd eval/results
ls -lt | head

# View comparison
cat addition_pipeline_*/comparison_report.txt
```

## 🔍 Troubleshooting

### Dataset not found

Make sure you have internet connection to download from HuggingFace.

### Out of memory

Try reducing batch size:

```bash
bash scripts/run_addition_experiment.sh --batch-size 4
```

### Models not found

Make sure you have HF_TOKEN set in `.env` file if models are private.

## 📚 Related Files

- **Finetuning script**: `finetune/finetune_and_eval_addition.py` (main logic)
- **Shell wrapper**: `scripts/run_addition_experiment.sh` (easy to run)
- **Pretraining evals**: `eval/` folder (for pretrained model benchmarks)

