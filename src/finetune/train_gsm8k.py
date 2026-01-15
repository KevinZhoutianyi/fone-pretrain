#!/usr/bin/env python3
"""
Finetune FoNE model on GSM8K with Chain-of-Thought reasoning
Uses LoRA for efficient finetuning
"""

import argparse
import json
import os
import shutil
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
)
from peft import LoraConfig, get_peft_model, TaskType, PeftModel
from datasets import load_dataset
from dotenv import load_dotenv
from huggingface_hub import HfApi, create_repo
import wandb

load_dotenv()


@dataclass
class DataCollatorForCompletionOnlyLM:
    """
    Data collator for causal language modeling that pads sequences dynamically.
    """
    tokenizer: Any
    pad_to_multiple_of: Optional[int] = None
    return_tensors: str = "pt"

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        # Extract input_ids and labels
        input_ids = [f["input_ids"] for f in features]
        labels = [f["labels"] for f in features]
        
        # Find max length in batch
        max_length = max(len(ids) for ids in input_ids)
        
        # Pad to multiple if specified
        if self.pad_to_multiple_of is not None:
            max_length = ((max_length + self.pad_to_multiple_of - 1) 
                         // self.pad_to_multiple_of * self.pad_to_multiple_of)
        
        # Pad sequences
        padded_input_ids = []
        padded_labels = []
        attention_mask = []
        
        for ids, lbls in zip(input_ids, labels):
            padding_length = max_length - len(ids)
            
            # Pad input_ids with tokenizer.pad_token_id
            padded_ids = ids + [self.tokenizer.pad_token_id] * padding_length
            padded_input_ids.append(padded_ids)
            
            # Pad labels with -100 (ignored in loss)
            padded_lbls = lbls + [-100] * padding_length
            padded_labels.append(padded_lbls)
            
            # Create attention mask (1 for real tokens, 0 for padding)
            mask = [1] * len(ids) + [0] * padding_length
            attention_mask.append(mask)
        
        return {
            "input_ids": torch.tensor(padded_input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "labels": torch.tensor(padded_labels, dtype=torch.long),
        }


def load_jsonl_dataset(file_path):
    """Load dataset from JSONL file"""
    dataset = load_dataset('json', data_files=str(file_path), split='train')
    return dataset


def tokenize_function(examples, tokenizer, max_length=512):
    """Tokenize examples for causal language modeling"""
    # Use the 'text' field which contains prompt + completion
    texts = examples['text']
    
    # Tokenize
    tokenized = tokenizer(
        texts,
        truncation=True,
        max_length=max_length,
        padding=False,
        return_tensors=None,
    )
    
    # For causal LM, labels are the same as input_ids
    # Create proper copies as plain lists
    tokenized['labels'] = [list(ids) for ids in tokenized['input_ids']]
    
    return tokenized


def main():
    parser = argparse.ArgumentParser(description="Finetune FoNE on GSM8K")
    
    # Model arguments
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Base model to finetune (HF repo ID or local path)"
    )
    
    parser.add_argument(
        "--data_dir",
        type=str,
        default="finetune/data/gsm8k_cot",
        help="Directory containing train.jsonl and test.jsonl"
    )
    
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Output directory for finetuned model"
    )
    
    # Training arguments
    parser.add_argument(
        "--num_epochs",
        type=int,
        default=3,
        help="Number of training epochs"
    )
    
    parser.add_argument(
        "--batch_size",
        type=int,
        default=4,
        help="Per-device batch size"
    )
    
    parser.add_argument(
        "--grad_accum",
        type=int,
        default=4,
        help="Gradient accumulation steps"
    )
    
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-4,
        help="Learning rate"
    )
    
    parser.add_argument(
        "--max_length",
        type=int,
        default=512,
        help="Maximum sequence length"
    )
    
    parser.add_argument(
        "--warmup_ratio",
        type=float,
        default=0.1,
        help="Warmup ratio"
    )
    
    # LoRA arguments
    parser.add_argument(
        "--use_lora",
        action="store_true",
        default=True,
        help="Use LoRA for efficient finetuning"
    )
    
    parser.add_argument(
        "--lora_r",
        type=int,
        default=16,
        help="LoRA rank"
    )
    
    parser.add_argument(
        "--lora_alpha",
        type=int,
        default=32,
        help="LoRA alpha"
    )
    
    parser.add_argument(
        "--lora_dropout",
        type=float,
        default=0.05,
        help="LoRA dropout"
    )
    
    # Other arguments
    parser.add_argument(
        "--wandb_project",
        type=str,
        default="fone-gsm8k",
        help="W&B project name"
    )
    
    parser.add_argument(
        "--wandb_run_name",
        type=str,
        default=None,
        help="W&B run name"
    )
    
    parser.add_argument(
        "--no_wandb",
        action="store_true",
        help="Disable W&B logging"
    )
    
    parser.add_argument(
        "--save_steps",
        type=int,
        default=500,
        help="Save checkpoint every N steps"
    )
    
    parser.add_argument(
        "--hf_repo",
        type=str,
        default=None,
        help="HuggingFace repository ID to upload model (e.g., username/model-name)"
    )
    
    parser.add_argument(
        "--cleanup_after_upload",
        action="store_true",
        default=True,
        help="Remove local checkpoints after uploading to HuggingFace (default: True)"
    )
    
    parser.add_argument(
        "--keep_local",
        action="store_true",
        help="Keep local checkpoints even after upload"
    )
    
    args = parser.parse_args()
    
    # Setup output directory
    if args.output_dir is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_name = args.model.split('/')[-1]
        args.output_dir = f"finetune/outputs/gsm8k_{model_name}_{timestamp}"
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    
    print("=" * 80)
    print("GSM8K FINETUNING WITH CHAIN-OF-THOUGHT")
    print("=" * 80)
    print(f"Base model: {args.model}")
    print(f"Data directory: {args.data_dir}")
    print(f"Output directory: {output_dir}")
    print(f"Use LoRA: {args.use_lora}")
    if args.use_lora:
        print(f"  LoRA rank: {args.lora_r}")
        print(f"  LoRA alpha: {args.lora_alpha}")
    print("=" * 80)
    print()
    
    # Setup W&B
    if not args.no_wandb:
        if args.wandb_run_name is None:
            args.wandb_run_name = f"gsm8k_{args.model.split('/')[-1]}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        wandb.init(
            project=args.wandb_project,
            name=args.wandb_run_name,
            config=vars(args),
        )
    
    # Load tokenizer
    print("Loading tokenizer...")
    hf_token = os.getenv("HF_TOKEN")
    tokenizer = AutoTokenizer.from_pretrained(args.model, token=hf_token)
    
    # Set pad token if not set
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    print(f"✅ Tokenizer loaded")
    print()
    
    # Load datasets
    print("Loading datasets...")
    data_dir = Path(args.data_dir)
    train_file = data_dir / "train.jsonl"
    test_file = data_dir / "test.jsonl"
    
    if not train_file.exists():
        raise FileNotFoundError(
            f"Training data not found at {train_file}. "
            f"Please run prepare_gsm8k_data.py first."
        )
    
    train_dataset = load_jsonl_dataset(train_file)
    eval_dataset = load_jsonl_dataset(test_file) if test_file.exists() else None
    
    print(f"✅ Loaded {len(train_dataset)} training examples")
    if eval_dataset:
        print(f"✅ Loaded {len(eval_dataset)} evaluation examples")
    print()
    
    # Tokenize datasets
    print("Tokenizing datasets...")
    train_dataset = train_dataset.map(
        lambda x: tokenize_function(x, tokenizer, args.max_length),
        batched=True,
        remove_columns=train_dataset.column_names,
        desc="Tokenizing train",
    )
    
    if eval_dataset:
        eval_dataset = eval_dataset.map(
            lambda x: tokenize_function(x, tokenizer, args.max_length),
            batched=True,
            remove_columns=eval_dataset.column_names,
            desc="Tokenizing eval",
        )
    
    print(f"✅ Tokenization complete")
    print()
    
    # Load model
    print("Loading model...")
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        token=hf_token,
    )
    
    print(f"✅ Model loaded: {sum(p.numel() for p in model.parameters()) / 1e9:.2f}B parameters")
    
    # Apply LoRA
    if args.use_lora:
        print("\nApplying LoRA...")
        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
            bias="none",
        )
        
        model = get_peft_model(model, lora_config)
        
        # Enable gradient checkpointing for LoRA (required for training)
        model.enable_input_require_grads()
        
        model.print_trainable_parameters()
        print()
    
    # Data collator
    # Use custom collator that properly handles padding for causal LM
    data_collator = DataCollatorForCompletionOnlyLM(
        tokenizer=tokenizer,
        pad_to_multiple_of=8,  # For better GPU efficiency
    )
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=args.num_epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.learning_rate,
        warmup_ratio=args.warmup_ratio,
        warmup_steps=50,  # Fixed warmup steps for consistency
        logging_steps=5,  # More frequent logging to monitor convergence
        logging_first_step=True,  # Log the first step
        save_steps=args.save_steps,
        eval_strategy="steps" if eval_dataset else "no",
        eval_steps=args.save_steps if eval_dataset else None,
        save_total_limit=3,
        bf16=True,
        gradient_checkpointing=True,
        report_to="wandb" if not args.no_wandb else "none",
        load_best_model_at_end=eval_dataset is not None,
        metric_for_best_model="loss" if eval_dataset else None,
        greater_is_better=False,
        ddp_find_unused_parameters=False,
        # Learning rate schedule - use cosine for better convergence
        lr_scheduler_type="cosine",
        # Weight decay for regularization
        weight_decay=0.01,
    )
    
    # Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
    )
    
    # Train
    print("=" * 80)
    print("STARTING TRAINING")
    print("=" * 80)
    print(f"Total epochs: {args.num_epochs}")
    print(f"Batch size per device: {args.batch_size}")
    print(f"Gradient accumulation: {args.grad_accum}")
    print(f"Effective batch size: {args.batch_size * args.grad_accum * torch.cuda.device_count()}")
    print(f"Learning rate: {args.learning_rate}")
    print("=" * 80)
    print()
    
    trainer.train()
    
    # Save final model
    print("\n" + "=" * 80)
    print("SAVING FINAL MODEL")
    print("=" * 80)
    
    final_model_dir = output_dir / "final_model"
    
    if args.use_lora:
        # Save LoRA adapters
        model.save_pretrained(final_model_dir)
        tokenizer.save_pretrained(final_model_dir)
        print(f"✅ LoRA adapters saved to: {final_model_dir}")
        
        # Also save merged model
        print("\nMerging LoRA weights into base model...")
        merged_model_dir = output_dir / "merged_model"
        
        # Merge and save
        merged_model = model.merge_and_unload()
        merged_model.save_pretrained(merged_model_dir)
        tokenizer.save_pretrained(merged_model_dir)
        print(f"✅ Merged model saved to: {merged_model_dir}")
    else:
        # Save full model
        trainer.save_model(final_model_dir)
        tokenizer.save_pretrained(final_model_dir)
        print(f"✅ Model saved to: {final_model_dir}")
    
    # Save training info
    info_file = output_dir / "training_info.json"
    with open(info_file, 'w') as f:
        json.dump({
            'base_model': args.model,
            'dataset': 'gsm8k',
            'num_epochs': args.num_epochs,
            'batch_size': args.batch_size,
            'grad_accum': args.grad_accum,
            'learning_rate': args.learning_rate,
            'use_lora': args.use_lora,
            'lora_r': args.lora_r if args.use_lora else None,
            'lora_alpha': args.lora_alpha if args.use_lora else None,
            'output_dir': str(output_dir),
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        }, f, indent=2)
    
    print()
    print("=" * 80)
    print("✅ TRAINING COMPLETE!")
    print("=" * 80)
    print(f"Output directory: {output_dir}")
    if args.use_lora:
        print(f"  - {final_model_dir.name}: LoRA adapters")
        print(f"  - {merged_model_dir.name}: Merged model (ready for inference)")
    else:
        print(f"  - {final_model_dir.name}: Finetuned model")
    print(f"  - {info_file.name}: Training info")
    print("=" * 80)
    
    # Upload to HuggingFace if requested
    if args.hf_repo:
        print("\n" + "=" * 80)
        print("📤 UPLOADING TO HUGGINGFACE HUB")
        print("=" * 80)
        print(f"Repository: {args.hf_repo}")
        
        try:
            hf_token = os.getenv("HF_TOKEN")
            if not hf_token:
                print("⚠️  Warning: HF_TOKEN not found in environment. Upload may fail.")
            
            # Create repo if it doesn't exist
            create_repo(args.hf_repo, token=hf_token, exist_ok=True)
            print(f"✅ Repository ready: https://huggingface.co/{args.hf_repo}")
            
            # Upload the merged model (or final model if not using LoRA)
            api = HfApi()
            upload_dir = merged_model_dir if args.use_lora else final_model_dir
            
            print(f"\n📤 Uploading model from {upload_dir.name}...")
            api.upload_folder(
                folder_path=str(upload_dir),
                repo_id=args.hf_repo,
                repo_type="model",
                token=hf_token,
            )
            print(f"✅ Model uploaded successfully!")
            print(f"🔗 View at: https://huggingface.co/{args.hf_repo}")
            
            # Cleanup local checkpoints if requested
            if args.cleanup_after_upload and not args.keep_local:
                print("\n" + "=" * 80)
                print("🧹 CLEANING UP LOCAL CHECKPOINTS")
                print("=" * 80)
                
                space_freed = 0
                items_removed = []
                
                # Remove intermediate checkpoints
                for item in output_dir.iterdir():
                    if item.is_dir() and item.name.startswith("checkpoint-"):
                        size = sum(f.stat().st_size for f in item.rglob('*') if f.is_file())
                        shutil.rmtree(item)
                        space_freed += size
                        items_removed.append(item.name)
                        print(f"  ✓ Removed: {item.name} ({size / 1e9:.2f} GB)")
                
                # Remove LoRA adapters if we have merged model
                if args.use_lora and final_model_dir.exists() and merged_model_dir.exists():
                    size = sum(f.stat().st_size for f in final_model_dir.rglob('*') if f.is_file())
                    shutil.rmtree(final_model_dir)
                    space_freed += size
                    items_removed.append(final_model_dir.name)
                    print(f"  ✓ Removed: {final_model_dir.name} ({size / 1e9:.2f} GB)")
                
                # Remove the merged model too since it's uploaded
                if merged_model_dir.exists():
                    size = sum(f.stat().st_size for f in merged_model_dir.rglob('*') if f.is_file())
                    shutil.rmtree(merged_model_dir)
                    space_freed += size
                    items_removed.append(merged_model_dir.name)
                    print(f"  ✓ Removed: {merged_model_dir.name} ({size / 1e9:.2f} GB)")
                
                print(f"\n📊 Summary:")
                print(f"  Items removed: {len(items_removed)}")
                print(f"  Space freed: {space_freed / 1e9:.2f} GB")
                print(f"  Model safely backed up at: https://huggingface.co/{args.hf_repo}")
                print("=" * 80)
            
        except Exception as e:
            print(f"❌ Upload failed: {e}")
            print("⚠️  Local files kept due to upload failure")
    
    if not args.no_wandb:
        wandb.finish()


if __name__ == "__main__":
    main()

