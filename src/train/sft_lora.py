"""
LoRA fine-tuning script for FoNE Llama models.

This script performs supervised fine-tuning using LoRA (Low-Rank Adaptation)
on mathematical reasoning tasks like GSM8K.
"""

import os
import sys
import argparse
import logging
import json
from pathlib import Path
from typing import Dict, Any, List, Optional
import torch
from transformers import (
    AutoTokenizer,
    TrainingArguments,
    DataCollatorForLanguageModeling
)
from datasets import load_dataset, Dataset
from peft import LoraConfig, get_peft_model, TaskType
from trl import SFTTrainer
from accelerate import Accelerator
import wandb

# Add src to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from modeling.llama_1p5b import LlamaWithFoNE

logger = logging.getLogger(__name__)


def setup_logging(rank: int = 0):
    """Set up logging configuration."""
    logging.basicConfig(
        level=logging.INFO if rank == 0 else logging.WARNING,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
        ]
    )


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="FoNE Llama LoRA SFT")
    
    # Model arguments
    parser.add_argument("--base_model", type=str, required=True,
                       help="Path to base FoNE model checkpoint")
    parser.add_argument("--tokenizer", type=str, default=None,
                       help="Tokenizer name or path (defaults to base_model)")
    
    # Data arguments
    parser.add_argument("--dataset", type=str, default="gsm8k_train",
                       help="Dataset name or path")
    parser.add_argument("--max_seq_length", type=int, default=2048,
                       help="Maximum sequence length")
    parser.add_argument("--train_split", type=str, default="train",
                       help="Training split name")
    
    # LoRA arguments
    parser.add_argument("--lora_r", type=int, default=16,
                       help="LoRA rank")
    parser.add_argument("--lora_alpha", type=int, default=32,
                       help="LoRA alpha parameter")
    parser.add_argument("--lora_dropout", type=float, default=0.1,
                       help="LoRA dropout")
    parser.add_argument("--lora_target_modules", type=str, nargs="+",
                       default=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
                       help="Target modules for LoRA")
    
    # Training arguments
    parser.add_argument("--output_dir", type=str, required=True,
                       help="Output directory for checkpoints")
    parser.add_argument("--num_train_epochs", type=int, default=3,
                       help="Number of training epochs")
    parser.add_argument("--per_device_train_batch_size", type=int, default=1,
                       help="Training batch size per device")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=16,
                       help="Gradient accumulation steps")
    parser.add_argument("--learning_rate", type=float, default=2e-4,
                       help="Learning rate")
    parser.add_argument("--warmup_ratio", type=float, default=0.1,
                       help="Warmup ratio")
    parser.add_argument("--weight_decay", type=float, default=0.01,
                       help="Weight decay")
    parser.add_argument("--lr_scheduler_type", type=str, default="cosine",
                       help="Learning rate scheduler type")
    
    # Logging and evaluation
    parser.add_argument("--logging_steps", type=int, default=10,
                       help="Logging interval")
    parser.add_argument("--save_steps", type=int, default=500,
                       help="Save checkpoint interval")
    parser.add_argument("--eval_steps", type=int, default=500,
                       help="Evaluation interval")
    parser.add_argument("--save_total_limit", type=int, default=3,
                       help="Maximum number of checkpoints to keep")
    
    # Experiment tracking
    parser.add_argument("--wandb_project", type=str, default="fone-sft",
                       help="Weights & Biases project name")
    parser.add_argument("--wandb_run_name", type=str, default=None,
                       help="Weights & Biases run name")
    parser.add_argument("--no_wandb", action="store_true",
                       help="Disable Weights & Biases logging")
    
    # Technical arguments
    parser.add_argument("--bf16", action="store_true", default=True,
                       help="Use bfloat16 training")
    parser.add_argument("--fp16", action="store_true",
                       help="Use float16 training")
    parser.add_argument("--dataloader_num_workers", type=int, default=4,
                       help="Number of dataloader workers")
    
    return parser.parse_args()


def format_gsm8k_example(example: Dict[str, Any]) -> str:
    """Format GSM8K example into CoT format."""
    question = example["question"]
    answer = example["answer"]
    
    # Extract the numeric answer
    if "####" in answer:
        parts = answer.split("####")
        reasoning = parts[0].strip()
        numeric_answer = parts[1].strip()
    else:
        reasoning = answer.strip()
        numeric_answer = "Unknown"
    
    # Format as instruction-following conversation
    formatted = f"<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n{question}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n{reasoning}\n\n#### {numeric_answer}<|eot_id|>"
    
    return formatted


def format_math_example(example: Dict[str, Any]) -> str:
    """Format mathematical reasoning example."""
    if "question" in example and "answer" in example:
        return format_gsm8k_example(example)
    elif "problem" in example and "solution" in example:
        # Handle MATH dataset format
        problem = example["problem"]
        solution = example["solution"]
        
        formatted = f"<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n{problem}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n{solution}<|eot_id|>"
        return formatted
    elif "input" in example and "output" in example:
        # Generic input-output format
        input_text = example["input"]
        output_text = example["output"]
        
        formatted = f"<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n{input_text}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n{output_text}<|eot_id|>"
        return formatted
    else:
        # Fallback: use text field if available
        return example.get("text", str(example))


def load_and_format_dataset(dataset_name: str, split: str = "train") -> Dataset:
    """Load and format dataset for SFT."""
    logger.info(f"Loading dataset: {dataset_name}")
    
    if dataset_name == "gsm8k_train":
        # Load GSM8K training set
        dataset = load_dataset("gsm8k", "main", split="train")
        
        def format_example(example):
            return {"text": format_gsm8k_example(example)}
            
        dataset = dataset.map(format_example, remove_columns=dataset.column_names)
        
    elif dataset_name == "gsm8k":
        # Load GSM8K with specified split
        dataset = load_dataset("gsm8k", "main", split=split)
        
        def format_example(example):
            return {"text": format_gsm8k_example(example)}
            
        dataset = dataset.map(format_example, remove_columns=dataset.column_names)
        
    elif os.path.exists(dataset_name):
        # Load from local file
        if dataset_name.endswith(".jsonl"):
            dataset = load_dataset("json", data_files=dataset_name, split="train")
        elif dataset_name.endswith(".json"):
            dataset = load_dataset("json", data_files=dataset_name, split="train")
        else:
            raise ValueError(f"Unsupported file format: {dataset_name}")
            
        # Format examples
        def format_example(example):
            return {"text": format_math_example(example)}
            
        dataset = dataset.map(format_example)
        
    else:
        # Try loading as HuggingFace dataset
        try:
            dataset = load_dataset(dataset_name, split=split)
            
            # Format examples
            def format_example(example):
                return {"text": format_math_example(example)}
                
            dataset = dataset.map(format_example)
            
        except Exception as e:
            raise ValueError(f"Could not load dataset {dataset_name}: {e}")
    
    logger.info(f"Loaded {len(dataset)} examples")
    return dataset


def create_lora_config(args: argparse.Namespace) -> LoraConfig:
    """Create LoRA configuration."""
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=args.lora_target_modules,
        bias="none",
        inference_mode=False,
    )
    
    logger.info(f"LoRA configuration:")
    logger.info(f"  Rank (r): {args.lora_r}")
    logger.info(f"  Alpha: {args.lora_alpha}")
    logger.info(f"  Dropout: {args.lora_dropout}")
    logger.info(f"  Target modules: {args.lora_target_modules}")
    
    return lora_config


def main():
    args = parse_args()
    
    # Set up logging
    setup_logging()
    
    # Initialize wandb
    if not args.no_wandb:
        wandb.init(
            project=args.wandb_project,
            name=args.wandb_run_name,
            config=vars(args)
        )
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load tokenizer
    tokenizer_path = args.tokenizer or args.base_model
    logger.info(f"Loading tokenizer from: {tokenizer_path}")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        
    # Load base model
    logger.info(f"Loading base model from: {args.base_model}")
    
    try:
        # Try loading as FoNE model first
        model_wrapper = LlamaWithFoNE.from_pretrained(
            args.base_model,
            torch_dtype=torch.bfloat16 if args.bf16 else torch.float16
        )
        model = model_wrapper.model
        logger.info("Loaded FoNE model successfully")
        
    except Exception as e:
        logger.warning(f"Could not load as FoNE model: {e}")
        # Fallback to regular loading
        from transformers import LlamaForCausalLM
        model = LlamaForCausalLM.from_pretrained(
            args.base_model,
            torch_dtype=torch.bfloat16 if args.bf16 else torch.float16
        )
        logger.info("Loaded as regular Llama model")
    
    # Apply LoRA
    lora_config = create_lora_config(args)
    model = get_peft_model(model, lora_config)
    
    # Print trainable parameters
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    
    logger.info(f"Trainable parameters: {trainable_params:,} ({100 * trainable_params / total_params:.2f}%)")
    logger.info(f"Total parameters: {total_params:,}")
    
    # Load and format dataset
    train_dataset = load_and_format_dataset(args.dataset, args.train_split)
    
    # Create training arguments
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        warmup_ratio=args.warmup_ratio,
        weight_decay=args.weight_decay,
        lr_scheduler_type=args.lr_scheduler_type,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        eval_steps=args.eval_steps,
        save_total_limit=args.save_total_limit,
        bf16=args.bf16,
        fp16=args.fp16,
        dataloader_num_workers=args.dataloader_num_workers,
        remove_unused_columns=False,
        report_to="wandb" if not args.no_wandb else None,
        run_name=args.wandb_run_name,
        gradient_checkpointing=True,
        optim="adamw_torch",
    )
    
    # Create data collator
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,  # Causal LM, not masked LM
        pad_to_multiple_of=8,
    )
    
    # Create trainer
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
        max_seq_length=args.max_seq_length,
        dataset_text_field="text",
        packing=False,  # We handle packing ourselves
    )
    
    # Start training
    logger.info("Starting LoRA fine-tuning...")
    trainer.train()
    
    # Save final model
    logger.info("Saving final LoRA adapter...")
    trainer.save_model()
    
    # Save tokenizer
    tokenizer.save_pretrained(args.output_dir)
    
    # Save training configuration
    config = {
        "base_model": args.base_model,
        "lora_config": {
            "r": args.lora_r,
            "alpha": args.lora_alpha,
            "dropout": args.lora_dropout,
            "target_modules": args.lora_target_modules,
        },
        "training_args": training_args.to_dict(),
        "dataset": args.dataset,
        "trainable_params": trainable_params,
        "total_params": total_params,
    }
    
    with open(os.path.join(args.output_dir, "training_config.json"), "w") as f:
        json.dump(config, f, indent=2)
    
    logger.info(f"LoRA fine-tuning completed. Model saved to {args.output_dir}")
    
    if not args.no_wandb:
        wandb.finish()


if __name__ == "__main__":
    main()
