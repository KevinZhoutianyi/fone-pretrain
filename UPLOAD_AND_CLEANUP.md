# Automatic Model Upload & Cleanup

This guide explains how to use the automatic HuggingFace upload and local cleanup feature.

## 🚀 Overview

The training script now automatically uploads your model to HuggingFace Hub and **removes local checkpoints** to save disk space. Your model is safely backed up in the cloud, so you don't need to keep large local copies.

## 📋 Quick Start

### Basic Usage

```bash
bash scripts/finetune.sh \
  --model Onlydrinkwater/fone-1p5b-step-76000 \
  --hf-repo YourUsername/your-model-name
```

This will:
1. ✅ Train your model
2. ✅ Upload to HuggingFace: `YourUsername/your-model-name`
3. ✅ **Automatically delete local checkpoints** (saves ~3-4GB per run)
4. ✅ Only keep the `training_info.json` file locally

### Keep Local Copies (Not Recommended)

If you really want to keep local checkpoints:

```bash
bash scripts/finetune.sh \
  --model Onlydrinkwater/fone-1p5b-step-76000 \
  --hf-repo YourUsername/your-model-name \
  --keep-local
```

## 🧹 What Gets Removed

After successful upload, the script automatically removes:

1. **All intermediate checkpoints** (`checkpoint-500`, `checkpoint-1000`, etc.)
   - These are just training snapshots, not needed once training is complete
   
2. **LoRA adapter files** (`final_model/`)
   - Redundant since they're merged into the final model
   
3. **The final merged model** (`merged_model/`)
   - Uploaded to HuggingFace, so no need to keep locally

**What's kept:**
- `training_info.json` - Small metadata file about your training run

## 🔑 Setup Required

Make sure you have your HuggingFace token set:

```bash
# In your .env file:
HF_TOKEN=hf_your_token_here
```

Get your token from: https://huggingface.co/settings/tokens

## 📊 Space Savings

Typical savings per training run:
- Intermediate checkpoints: **~2-3 GB**
- Final models: **~3.5 GB**
- **Total saved: ~5-6 GB per run**

## 🛡️ Safety

- Local files are only deleted **after successful upload**
- If upload fails, all files are kept locally
- You can always download your model from HuggingFace later

## 📥 Downloading Your Model Later

To use your uploaded model:

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

# Load directly from HuggingFace
model = AutoModelForCausalLM.from_pretrained("YourUsername/your-model-name")
tokenizer = AutoTokenizer.from_pretrained("YourUsername/your-model-name")
```

Or use it in a new training run:

```bash
bash scripts/finetune.sh --model YourUsername/your-model-name
```

## 🔧 Advanced Options

### Python Script Direct Usage

```bash
python src/finetune/train_gsm8k.py \
  --model Onlydrinkwater/fone-1p5b-step-76000 \
  --data_dir data/gsm8k_cot \
  --hf_repo YourUsername/your-model-name \
  --num_epochs 3 \
  --batch_size 4 \
  --use_lora \
  --lora_r 64
```

### Disable Cleanup (Keep Everything)

```bash
python src/finetune/train_gsm8k.py \
  --model Onlydrinkwater/fone-1p5b-step-76000 \
  --hf_repo YourUsername/your-model-name \
  --keep_local
```

## ✅ Benefits

1. **Save Disk Space**: Free up 5-6 GB per training run
2. **Automatic Backup**: Models safely stored in HuggingFace
3. **Easy Sharing**: Share models with collaborators via HuggingFace
4. **No Manual Work**: Everything happens automatically after training

## 📝 Example Output

```
================================================================================
✅ TRAINING COMPLETE!
================================================================================
Output directory: finetune/outputs/gsm8k_fone-1p5b-step-76000_20251024_123456

================================================================================
📤 UPLOADING TO HUGGINGFACE HUB
================================================================================
Repository: YourUsername/your-model-name
✅ Repository ready: https://huggingface.co/YourUsername/your-model-name
📤 Uploading model from merged_model...
✅ Model uploaded successfully!
🔗 View at: https://huggingface.co/YourUsername/your-model-name

================================================================================
🧹 CLEANING UP LOCAL CHECKPOINTS
================================================================================
  ✓ Removed: checkpoint-500 (0.80 GB)
  ✓ Removed: checkpoint-1000 (0.80 GB)
  ✓ Removed: checkpoint-1404 (0.80 GB)
  ✓ Removed: final_model (0.26 GB)
  ✓ Removed: merged_model (3.50 GB)

📊 Summary:
  Items removed: 5
  Space freed: 6.16 GB
  Model safely backed up at: https://huggingface.co/YourUsername/your-model-name
================================================================================
```

