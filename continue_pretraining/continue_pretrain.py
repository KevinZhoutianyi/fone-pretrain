"""
Continue pretraining script - loads from HuggingFace Hub checkpoint and continues training.

This script is adapted from the main pretrain.py to support loading pretrained weights
from HuggingFace Hub models.
"""

import os
import sys
import argparse
import logging
import math
import json
import time
from pathlib import Path
from typing import Dict, Any, Optional
import torch
import torch.nn.functional as F
from torch.optim import AdamW
from transformers import (
    AutoModelForCausalLM, 
    AutoTokenizer,
    get_cosine_schedule_with_warmup
)
from accelerate import Accelerator
from accelerate.utils import set_seed, DistributedDataParallelKwargs
import wandb

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from data.mixture import DataMixture
from data.packing import StreamingPackedDataset
from modeling.fone_init import find_number_tokens, verify_fone_overrides

logger = logging.getLogger(__name__)


def setup_logging(rank: int = 0):
    """Set up logging configuration."""
    logging.basicConfig(
        level=logging.INFO if rank == 0 else logging.WARNING,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[logging.StreamHandler()]
    )
    logging.getLogger("datasets").setLevel(logging.ERROR)
    logging.getLogger("data.mixture").setLevel(logging.ERROR)
    logging.getLogger("data.packing").setLevel(logging.ERROR)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Continue Pretraining from HF Checkpoint")
    
    # Model - load from HuggingFace
    parser.add_argument("--model_name_or_path", type=str, required=True,
                       help="HuggingFace model ID or local path (e.g., 'Onlydrinkwater/fone-1p5b-step-76294')")
    parser.add_argument("--tokenizer", type=str, default=None,
                       help="Tokenizer name (defaults to model_name_or_path)")
    parser.add_argument("--dataset", type=str, required=True,
                       help="Path to data mixture configuration")
    
    # Training arguments
    parser.add_argument("--seq_len", type=int, default=2048,
                       help="Sequence length for training")
    parser.add_argument("--lr", type=float, default=1e-5,
                       help="Learning rate (lower for continue pretrain)")
    parser.add_argument("--warmup_steps", type=int, default=500,
                       help="Number of warmup steps")
    parser.add_argument("--total_tokens", type=int, default=5_000_000_000,
                       help="Total tokens to train on")
    parser.add_argument("--weight_decay", type=float, default=0.1,
                       help="Weight decay")
    parser.add_argument("--beta1", type=float, default=0.9,
                       help="Adam beta1")
    parser.add_argument("--beta2", type=float, default=0.95,
                       help="Adam beta2")
    parser.add_argument("--grad_clip", type=float, default=1.0,
                       help="Gradient clipping norm")
    parser.add_argument("--per_device_train_batch_size", type=int, default=4,
                       help="Per-GPU micro-batch size")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4,
                       help="Gradient accumulation steps")
    
    # Logging
    parser.add_argument("--output_dir", type=str, required=True,
                       help="Output directory for logs")
    parser.add_argument("--log_every_steps", type=int, default=10,
                       help="Log metrics every N steps")
    
    # FoNE arguments
    parser.add_argument("--fone_hi", type=int, default=999,
                       help="Maximum number for FoNE freezing (0 to fone_hi). Set to -1 to disable.")
    parser.add_argument("--no_freeze_fone", action="store_true",
                       help="Don't freeze FoNE number embeddings (allow them to be updated)")
    
    # Technical arguments
    parser.add_argument("--dataloader_num_workers", type=int, default=0,
                       help="Number of dataloader workers")
    
    # Experiment tracking
    parser.add_argument("--wandb_project", type=str, default="fone-continue-pretrain",
                       help="Weights & Biases project name")
    parser.add_argument("--wandb_run_name", type=str, default=None,
                       help="Weights & Biases run name")
    parser.add_argument("--no_wandb", action="store_true",
                       help="Disable Weights & Biases logging")
    
    # Hugging Face Hub upload
    parser.add_argument("--hf_repo_id", type=str, default=None,
                       help="HuggingFace repo ID to upload checkpoints")
    
    return parser.parse_args()


def compute_training_steps(
    total_tokens: int,
    micro_batch_size: int,
    sequence_length: int,
    gradient_accumulation_steps: int,
    num_processes: int = 1
) -> int:
    """Compute total number of training steps."""
    tokens_per_update = micro_batch_size * sequence_length * gradient_accumulation_steps * num_processes
    total_steps = math.ceil(total_tokens / tokens_per_update)
    
    logger.info(f"Training configuration:")
    logger.info(f"  Total tokens: {total_tokens:,}")
    logger.info(f"  Batch size per GPU: {micro_batch_size}")
    logger.info(f"  Sequence length: {sequence_length}")
    logger.info(f"  Gradient accumulation steps: {gradient_accumulation_steps}")
    logger.info(f"  Number of GPUs: {num_processes}")
    logger.info(f"  Tokens per update (all GPUs): {tokens_per_update:,}")
    logger.info(f"  Total optimizer steps: {total_steps:,}")
    
    return total_steps


def create_optimizer(model: torch.nn.Module, args: argparse.Namespace) -> AdamW:
    """Create AdamW optimizer with weight decay."""
    decay_params = []
    no_decay_params = []
    
    for name, param in model.named_parameters():
        if param.requires_grad:
            if any(nd in name for nd in ["bias", "norm", "embed"]):
                no_decay_params.append(param)
            else:
                decay_params.append(param)
                
    optimizer_grouped_parameters = [
        {"params": decay_params, "weight_decay": args.weight_decay},
        {"params": no_decay_params, "weight_decay": 0.0},
    ]
    
    optimizer = AdamW(
        optimizer_grouped_parameters,
        lr=args.lr,
        betas=(args.beta1, args.beta2),
        eps=1e-8,
    )
    
    logger.info(f"Created optimizer with {len(decay_params)} decay params, "
                f"{len(no_decay_params)} no-decay params")
    
    return optimizer


def save_final_to_hf(
    accelerator: Accelerator,
    model: torch.nn.Module,
    tokenizer,
    step: int,
    hf_repo_id: str
):
    """Save final model directly to HuggingFace (no local save)."""
    if not accelerator.is_main_process:
        return
    
    import tempfile
    from huggingface_hub import HfApi
    
    # Save to temp dir, upload, then delete
    with tempfile.TemporaryDirectory() as tmpdir:
        logger.info(f"Saving final model to temp dir for HF upload...")
        
        unwrapped_model = accelerator.unwrap_model(model)
        unwrapped_model.save_pretrained(tmpdir)
        tokenizer.save_pretrained(tmpdir)
        
        try:
            api = HfApi()
            repo_id = f"{hf_repo_id}-step-{step}"
            
            logger.info(f"Uploading final model to {repo_id}...")
            api.create_repo(repo_id=repo_id, exist_ok=True, private=False)
            api.upload_folder(
                folder_path=tmpdir,
                repo_id=repo_id,
                commit_message=f"Continue pretrain on MegaMath - final model at step {step}"
            )
            
            logger.info(f"✅ Uploaded to {repo_id}")
            logger.info(f"🔗 View at: https://huggingface.co/{repo_id}")
            
        except Exception as e:
            logger.error(f"Failed to upload to HuggingFace: {e}")
            raise


def freeze_number_embeddings(
    model: torch.nn.Module,
    tokenizer,
    hi: int = 999
) -> int:
    """
    Freeze number embeddings for tokens 0 to hi.
    Returns the number of frozen embedding rows.
    """
    if hi < 0:
        logger.info("FoNE freezing disabled (hi < 0)")
        return 0
    
    logger.info(f"Freezing number embeddings for 0-{hi}...")
    
    # Find number tokens
    token_info = find_number_tokens(tokenizer, hi)
    found_tokens = token_info['found']
    
    if not found_tokens:
        logger.warning("No number tokens found to freeze!")
        return 0
    
    # Get embedding layer
    embed_tokens = model.get_input_embeddings()
    
    # Collect indices to freeze
    frozen_indices = set()
    for token_str, token_id in found_tokens:
        try:
            num_value = int(token_str.strip())
            if 0 <= num_value <= hi:
                frozen_indices.add(token_id)
        except ValueError:
            continue
    
    if not frozen_indices:
        logger.warning("No valid number token indices found!")
        return 0
    
    # Create gradient hook to freeze these embeddings
    def freeze_number_embeddings_hook(grad):
        """Zero out gradients for number embedding rows."""
        grad_clone = grad.clone()
        for idx in frozen_indices:
            grad_clone[idx] = 0.0
        return grad_clone
    
    # Register the hook
    embed_tokens.weight.register_hook(freeze_number_embeddings_hook)
    
    logger.info(f"✅ Froze {len(frozen_indices)} number embedding rows (0-{hi})")
    logger.info(f"   Number embeddings will remain fixed during continue pretraining")
    
    return len(frozen_indices)


def train_step(
    model: torch.nn.Module,
    batch: Dict[str, torch.Tensor],
    accelerator: Accelerator
) -> Dict[str, float]:
    """Perform a single training step."""
    model.train()
    outputs = model(**batch)
    loss = outputs.loss
    accelerator.backward(loss)
    return {"loss": loss.detach().float()}


def main():
    args = parse_args()
    
    # Initialize accelerator
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        log_with="wandb" if not args.no_wandb else None,
        project_dir=args.output_dir,
        step_scheduler_with_optimizer=False,
        kwargs_handlers=[DistributedDataParallelKwargs(find_unused_parameters=True)]
    )
    
    setup_logging(accelerator.process_index)
    set_seed(42)
    
    # Initialize wandb
    if accelerator.is_main_process and not args.no_wandb:
        wandb.init(
            project=args.wandb_project,
            name=args.wandb_run_name,
            config=vars(args)
        )
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load pretrained model from HuggingFace
    logger.info(f"Loading pretrained model from {args.model_name_or_path}...")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True
    )
    
    tokenizer_name = args.tokenizer or args.model_name_or_path
    logger.info(f"Loading tokenizer from {tokenizer_name}...")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, trust_remote_code=True)
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Model loaded: {total_params/1e9:.2f}B params ({trainable_params/1e9:.2f}B trainable)")
    
    # Freeze number embeddings if requested
    num_frozen = 0
    if not args.no_freeze_fone and args.fone_hi >= 0:
        num_frozen = freeze_number_embeddings(model, tokenizer, args.fone_hi)
        if num_frozen > 0:
            hidden_size = model.config.hidden_size
            frozen_params = num_frozen * hidden_size
            logger.info(f"   Frozen embedding parameters: {frozen_params:,} ({frozen_params/1e6:.2f}M)")
    elif args.no_freeze_fone:
        logger.info("FoNE freezing disabled by --no_freeze_fone flag")
    
    # Set up data loading
    logger.info("Setting up data loading...")
    data_mixture = DataMixture(
        mixture_config=args.dataset,
        sequence_length=args.seq_len,
        shuffle_buffer_size=10000,
        num_workers=args.dataloader_num_workers
    )
    
    text_stream = data_mixture.get_streaming_iterator()
    dataset = StreamingPackedDataset(
        text_iterator=text_stream,
        tokenizer=tokenizer,
        sequence_length=args.seq_len
    )
    
    # Create optimizer
    optimizer = create_optimizer(model, args)
    
    # Compute training steps
    micro_batch_size = args.per_device_train_batch_size
    gradient_accumulation_steps = args.gradient_accumulation_steps
    
    total_steps = compute_training_steps(
        total_tokens=args.total_tokens,
        micro_batch_size=micro_batch_size,
        sequence_length=args.seq_len,
        gradient_accumulation_steps=gradient_accumulation_steps,
        num_processes=accelerator.num_processes
    )
    
    # Create learning rate scheduler
    effective_warmup_steps = min(args.warmup_steps, int(total_steps * 0.1))
    logger.info(f"Creating LR scheduler with {total_steps} total steps and {effective_warmup_steps} warmup steps")
    
    lr_scheduler = get_cosine_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=effective_warmup_steps,
        num_training_steps=total_steps
    )
    
    # Prepare with accelerator
    model, optimizer, lr_scheduler = accelerator.prepare(model, optimizer, lr_scheduler)
    
    # Training loop
    logger.info("Starting continue pretraining...")
    
    step = 0
    data_iterator = iter(dataset)
    start_time = time.time()
    
    model.train()
    
    while step < total_steps:
        try:
            with accelerator.accumulate(model):
                # Build micro-batch
                samples = []
                for _ in range(micro_batch_size):
                    example = next(data_iterator)
                    example = {k: (v.unsqueeze(0) if isinstance(v, torch.Tensor) and v.dim() == 1 else v)
                               for k, v in example.items()}
                    samples.append(example)
                
                keys = samples[0].keys()
                batch = {key: torch.cat([s[key] for s in samples], dim=0) for key in keys}
                batch = {k: v.to(accelerator.device) for k, v in batch.items()}
                
                # Training step
                metrics = train_step(model, batch, accelerator)
                
                # Gradient clipping and optimizer step
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(model.parameters(), args.grad_clip)
                    optimizer.step()
                    lr_scheduler.step()
                    optimizer.zero_grad()
                
                # Update step counter
                if accelerator.sync_gradients:
                    step += 1
                    
                    # Logging
                    if step % args.log_every_steps == 0:
                        current_time = time.time()
                        elapsed_time = current_time - start_time
                        progress_pct = (step / total_steps) * 100
                        
                        if step > 0:
                            avg_step_time = elapsed_time / step
                            remaining_steps = total_steps - step
                            eta_seconds = remaining_steps * avg_step_time
                            eta_hours = eta_seconds / 3600
                            eta_str = f"{eta_hours:.1f}h" if eta_hours >= 1 else f"{eta_seconds/60:.0f}m"
                        else:
                            eta_str = "calculating..."
                        
                        lr = lr_scheduler.get_last_lr()[0]
                        loss_val = float(metrics["loss"])
                        ppl = math.exp(min(20.0, max(-20.0, loss_val)))
                        
                        tokens_processed = step * micro_batch_size * args.seq_len * gradient_accumulation_steps * accelerator.num_processes
                        
                        log_metrics = {
                            "step": step,
                            "loss": loss_val,
                            "ppl": ppl,
                            "learning_rate": lr,
                            "tokens_processed": tokens_processed,
                            "progress_pct": progress_pct
                        }
                        
                        if accelerator.is_main_process:
                            logger.info(f"Step {step}/{total_steps} ({progress_pct:.1f}%): "
                                       f"loss={loss_val:.4f}, ppl={ppl:.2f}, lr={lr:.2e}, ETA={eta_str}")
                            
                            if not args.no_wandb:
                                wandb.log(log_metrics)
                        
        except StopIteration:
            logger.info("Data iterator exhausted")
            break
        except Exception as e:
            logger.error(f"Error during training step {step}: {e}")
            raise
    
    # Upload final model to HuggingFace (no local save)
    if args.hf_repo_id:
        save_final_to_hf(accelerator, model, tokenizer, step, args.hf_repo_id)
    else:
        logger.warning("No hf_repo_id specified - final model not saved!")
    
    if accelerator.is_main_process:
        logger.info(f"Continue pretraining completed after {step} steps")
        
        if not args.no_wandb:
            wandb.finish()


if __name__ == "__main__":
    main()

