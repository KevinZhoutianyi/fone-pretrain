# FoNE Quick Start Guide

## 🎯 Three Main Operations

### 1. Pretraining (From Scratch)
Train a FoNE or Baseline model from scratch on your data.

### 2. Finetuning (Adapt Pretrained Model)
Fine-tune a pretrained model on a specific task (e.g., GSM8K math problems).

### 3. Evaluation
Test your model's performance.

---

## 📋 1. PRETRAINING

### Start Pretraining FoNE Model

```bash
# Activate environment
conda activate fone

# Quick test (2 minutes, 137M tokens)
bash scripts/pretrain_0p5b_local.sh

# Full FoNE pretraining (1.5B, 30B tokens, with auto-upload)
accelerate launch --num_processes 8 --mixed_precision bf16 \
  src/train/pretrain.py \
  --model_config configs/llama_1p5b.json \
  --dataset configs/data_local_30b.json \
  --output_dir outputs/fone_1p5b_30b \
  --lr 3e-4 \
  --warmup_steps 3000 \
  --save_every_steps 10000 \
  --hf_repo_id YourUsername/fone-1p5b \
  --wandb_project fone-pretraining \
  --wandb_run_name fone_1p5b_30b
```

### Start Pretraining Baseline Model (For Comparison)

```bash
# Baseline model (same architecture, no FoNE)
accelerate launch --num_processes 8 --mixed_precision bf16 \
  src/train/pretrain.py \
  --model_config configs/llama_1p5b_baseline.json \
  --dataset configs/data_local_30b.json \
  --output_dir outputs/baseline_1p5b_30b \
  --lr 3e-4 \
  --warmup_steps 3000 \
  --save_every_steps 10000 \
  --hf_repo_id YourUsername/baseline-1p5b \
  --wandb_project fone-pretraining \
  --wandb_run_name baseline_1p5b_30b
```

### Run in Background

```bash
nohup accelerate launch --num_processes 8 --mixed_precision bf16 \
  src/train/pretrain.py \
  --model_config configs/llama_1p5b.json \
  --dataset configs/data_local_30b.json \
  --output_dir outputs/fone_1p5b \
  --hf_repo_id YourUsername/fone-1p5b \
  > pretrain.log 2>&1 &

# Monitor progress
tail -f pretrain.log
```

---

## 📋 2. FINETUNING

### Fine-tune on GSM8K (Math Problems)

```bash
# With auto-upload and cleanup
bash scripts/finetune.sh \
  --model Onlydrinkwater/fone-1p5b-step-76000 \
  --hf-repo YourUsername/fone-1p5b-gsm8k

# The script will:
# ✅ Train the model
# ✅ Upload to HuggingFace
# ✅ Delete local checkpoints automatically (saves ~5-6GB)
```

### Keep Local Checkpoints (Not Recommended)

```bash
bash scripts/finetune.sh \
  --model Onlydrinkwater/fone-1p5b-step-76000 \
  --hf-repo YourUsername/fone-1p5b-gsm8k \
  --keep-local
```

---

## 📊 3. EVALUATION

### Evaluate on GSM8K

```bash
bash scripts/eval_gsm8k.sh --model outputs/gsm8k_*/merged_model
```

---

## 🔑 Key Differences: FoNE vs Baseline

| Feature | FoNE | Baseline |
|---------|------|----------|
| **Config** | `llama_1p5b.json` | `llama_1p5b_baseline.json` |
| **Number Embeddings** | Frozen Fourier features (0-999) | Standard learned embeddings |
| **Purpose** | Better numerical reasoning | Standard comparison |
| **Use When** | Training on math/numbers | General text tasks |

---

## 🚀 Complete Workflow Example

```bash
# 1. Activate environment
conda activate fone

# 2. Pretrain FoNE model
nohup accelerate launch --num_processes 8 --mixed_precision bf16 \
  src/train/pretrain.py \
  --model_config configs/llama_1p5b.json \
  --dataset configs/data_local_30b.json \
  --output_dir outputs/fone_1p5b \
  --hf_repo_id YourUsername/fone-1p5b \
  --save_every_steps 10000 \
  > pretrain_fone.log 2>&1 &

# 3. Pretrain Baseline (for comparison)
nohup accelerate launch --num_processes 8 --mixed_precision bf16 \
  src/train/pretrain.py \
  --model_config configs/llama_1p5b_baseline.json \
  --dataset configs/data_local_30b.json \
  --output_dir outputs/baseline_1p5b \
  --hf_repo_id YourUsername/baseline-1p5b \
  --save_every_steps 10000 \
  > pretrain_baseline.log 2>&1 &

# 4. Monitor progress
tail -f pretrain_fone.log
# Check WandB: https://wandb.ai

# 5. Fine-tune on GSM8K (when pretraining done)
bash scripts/finetune.sh \
  --model YourUsername/fone-1p5b \
  --hf-repo YourUsername/fone-1p5b-gsm8k

# 6. Evaluate
bash scripts/eval_gsm8k.sh --model YourUsername/fone-1p5b-gsm8k
```

---

## 🧹 Auto-Cleanup Feature

**Both pretraining and finetuning now auto-delete local checkpoints after uploading to HuggingFace!**

### What Gets Deleted:
- ✅ Intermediate checkpoints (`checkpoint-*`, `step_*`)
- ✅ LoRA adapter files
- ✅ Merged models (already uploaded)

### What's Kept:
- ✅ Your model on HuggingFace (safe backup)
- ✅ Training metadata (`training_info.json`)

### Space Saved:
- **Pretraining**: ~3-5 GB per checkpoint
- **Finetuning**: ~5-6 GB per run
- **Total**: Can save 15-20 GB easily!

### To Disable Cleanup:

```bash
# Pretraining
--keep_local_checkpoints

# Finetuning
--keep-local
```

---

## 📁 Data Configurations

| Config | Tokens | Time (8xGPU) | Use Case |
|--------|--------|--------------|----------|
| `data_local_sanity.json` | 137M | 2 min | Quick test |
| `data_local_small.json` | 7B | 15 min | Fast iteration |
| `data_local_30b.json` | 30B | 1 hour | Full pretraining |

---

## 📊 Monitoring

### Live Logs
```bash
tail -f pretrain.log
tail -f finetune.log
```

### WandB Dashboard
- Project: `fone-pretraining`
- View: https://wandb.ai/your-username/fone-pretraining

### Disk Space
```bash
du -sh outputs/
du -sh /home/nvidia
```

---

## 🔧 Environment Setup (First Time Only)

```bash
# Create environment
conda create -n fone python=3.10
conda activate fone

# Install dependencies
cd /home/nvidia/fone-pretrain
pip install -r requirements.txt

# Set HuggingFace token in .env
echo "HF_TOKEN=hf_your_token_here" > .env
```

---

## 🆘 Troubleshooting

### Out of Memory
```bash
# Reduce batch size
PER_DEVICE_BATCH_SIZE=2 bash scripts/pretrain_0p5b_local.sh
```

### Check GPU Usage
```bash
nvidia-smi
watch -n 1 nvidia-smi
```

### Check Training Progress
```bash
# See latest checkpoint
ls -lhtr outputs/fone_1p5b/step_* | tail -1

# Check logs
tail -100 pretrain.log | grep "loss="
```

---

## 📚 More Information

- **Full Pretraining Guide**: See `PRETRAIN_GUIDE.md`
- **Upload & Cleanup Details**: See `UPLOAD_AND_CLEANUP.md`
- **Main README**: See `README.md`

---

## 🎓 Common Commands Reference

```bash
# Quick sanity check
bash scripts/pretrain_0p5b_local.sh

# Full FoNE pretraining with upload
accelerate launch --num_processes 8 --mixed_precision bf16 src/train/pretrain.py \
  --model_config configs/llama_1p5b.json --dataset configs/data_local_30b.json \
  --output_dir outputs/fone_1p5b --hf_repo_id YourUsername/fone-1p5b

# Fine-tune with upload
bash scripts/finetune.sh --model MODEL_NAME --hf-repo YourUsername/model-name

# Evaluate
bash scripts/eval_gsm8k.sh --model MODEL_PATH

# Monitor
tail -f *.log

# Check space
du -sh outputs/
```

