# Continue Pretraining on MegaMath

Continue pretraining FoNE and baseline models on the MegaMath-web-pro dataset (15B tokens).

## Quick Start

### 1. Download MegaMath Data

```bash
cd /home/nvidia/fone-pretrain/continue_pretraining
python download_megamath.py
```

### 2. Run Continue Pretraining

**FoNE model (with frozen number embeddings):**
```bash
bash run.sh fone --bg
```

**Baseline model:**
```bash
bash run.sh baseline --bg
```

## Models

- **FoNE**: `Onlydrinkwater/fone-1p5b-step-76294`
- **Baseline**: `Onlydrinkwater/baseline-1p5b-step-76294`

## FoNE Embedding Freezing

**Important:** The FoNE model uses frozen Fourier embeddings for numbers 0-999. During continue pretraining, these embeddings are **frozen by default** to preserve the number representations learned during initial pretraining.

### Configuration

In `continue_fone.yaml`:
```yaml
fone_hi: 999  # Freeze embeddings for numbers 0-999
# no_freeze_fone: false  # Set to true to allow number embeddings to update
```

### How It Works

1. **Number Detection**: Finds all number tokens (0-999) in the tokenizer
2. **Gradient Hook**: Registers a hook that zeros out gradients for these embedding rows
3. **Preservation**: Number embeddings remain fixed while all other parameters (attention, FFN, etc.) continue training

### To Disable Freezing

If you want to allow number embeddings to update (not recommended for FoNE):
```bash
python continue_pretrain.py --no_freeze_fone ...
```

Or in the YAML config:
```yaml
no_freeze_fone: true
```

## Behavior

- No local checkpoints saved during training
- Only final model uploaded to HuggingFace
- 15B tokens training target
- FoNE: Number embeddings (0-999) frozen by default
- Baseline: No special embedding handling
