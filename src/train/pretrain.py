"""
FoNE Llama pretraining script with Accelerate + DeepSpeed integration.

This script handles the full pretraining pipeline including:
- Model initialization with FoNE overrides
- Data loading and packing
- Training loop with gradient accumulation
- Checkpointing and logging
- FoNE embedding freezing
"""

import os
import sys
import argparse
import logging
import math
import json
import time
import subprocess
from pathlib import Path
from typing import Dict, Any, Optional
import torch
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from transformers import get_cosine_schedule_with_warmup
from accelerate import Accelerator
from accelerate.utils import set_seed
import wandb

# Add src to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from modeling.llama_1p5b import setup_llama_with_fone
from data.mixture import DataMixture
from data.packing import StreamingPackedDataset

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
    
    # Reduce noise from datasets library
    logging.getLogger("datasets").setLevel(logging.ERROR)
    logging.getLogger("data.mixture").setLevel(logging.ERROR)
    logging.getLogger("data.packing").setLevel(logging.ERROR)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="FoNE Llama Pretraining")
    
    # Model and data arguments
    parser.add_argument("--model_config", type=str, required=True,
                       help="Path to model configuration JSON")
    parser.add_argument("--tokenizer", type=str, default="meta-llama/Llama-3.1-8B",
                       help="Tokenizer name or path")
    parser.add_argument("--dataset", type=str, required=True,
                       help="Path to data mixture configuration")
    
    # Training arguments
    parser.add_argument("--seq_len", type=int, default=2048,
                       help="Sequence length for training")
    parser.add_argument("--lr", type=float, default=2e-4,
                       help="Learning rate")
    parser.add_argument("--warmup_steps", type=int, default=3000,
                       help="Number of warmup steps")
    parser.add_argument("--total_tokens", type=int, default=30_000_000_000,
                       help="Total tokens to train on")
    parser.add_argument("--total_tokens_scope", type=str, default="per_device",
                       choices=["global", "per_device"],
                       help="Interpretation scope for total_tokens: global across all GPUs or per_device")
    parser.add_argument("--weight_decay", type=float, default=0.1,
                       help="Weight decay")
    parser.add_argument("--beta1", type=float, default=0.9,
                       help="Adam beta1")
    parser.add_argument("--beta2", type=float, default=0.95,
                       help="Adam beta2")
    parser.add_argument("--grad_clip", type=float, default=1.0,
                       help="Gradient clipping norm")
    parser.add_argument("--per_device_train_batch_size", type=int, default=None,
                       help="Per-GPU micro-batch size (overrides accelerate/deepspeed config if set)")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=None,
                       help="Override gradient accumulation steps")
    
    # Logging and checkpointing
    parser.add_argument("--output_dir", type=str, required=True,
                       help="Output directory for checkpoints")
    parser.add_argument("--save_every_steps", type=int, default=2000,
                       help="Save checkpoint every N steps")
    parser.add_argument("--log_every_steps", type=int, default=100,
                       help="Log metrics every N steps")
    parser.add_argument("--eval_every_steps", type=int, default=0,
                       help="Evaluate every N steps (0 to disable)")
    parser.add_argument("--validation_data", type=str, default="configs/validation_data.json",
                       help="Path to validation data configuration")
    parser.add_argument("--validation_prompts", type=str, default="configs/validation_prompts.json",
                       help="Path to validation prompts file")
    parser.add_argument("--quick_validation", action="store_true",
                       help="Run quick validation with fewer samples")
    
    # FoNE arguments
    parser.add_argument("--fone_hi", type=int, default=999,
                       help="Maximum number for FoNE overrides")
    parser.add_argument("--no_freeze_fone", action="store_true",
                       help="Don't freeze FoNE embedding rows")
    
    # Technical arguments
    parser.add_argument("--flash_attn", action="store_true",
                       help="Use FlashAttention-2")
    parser.add_argument("--activation_checkpointing", action="store_true", default=True,
                       help="Use activation checkpointing")
    parser.add_argument("--dataloader_num_workers", type=int, default=0,
                       help="Number of dataloader workers")
    
    # Experiment tracking
    parser.add_argument("--wandb_project", type=str, default="fone-pretraining",
                       help="Weights & Biases project name")
    parser.add_argument("--wandb_run_name", type=str, default=None,
                       help="Weights & Biases run name")
    parser.add_argument("--no_wandb", action="store_true",
                       help="Disable Weights & Biases logging")
    
    # Resume training
    parser.add_argument("--resume_from", type=str, default=None,
                       help="Resume training from checkpoint")
    
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
    logger.info(f"  Tokens per update: {tokens_per_update:,}")
    logger.info(f"  Total steps: {total_steps:,}")
    logger.info(f"  Micro batch size: {micro_batch_size}")
    logger.info(f"  Sequence length: {sequence_length}")
    logger.info(f"  Gradient accumulation steps: {gradient_accumulation_steps}")
    logger.info(f"  Number of processes: {num_processes}")
    
    return total_steps


def create_optimizer(model: torch.nn.Module, args: argparse.Namespace) -> AdamW:
    """Create AdamW optimizer with weight decay."""
    # Separate parameters that should and shouldn't have weight decay
    decay_params = []
    no_decay_params = []
    
    for name, param in model.named_parameters():
        if param.requires_grad:
            # Don't apply weight decay to biases, layer norms, and embeddings
            if any(nd in name for nd in ["bias", "norm", "embed"]):
                no_decay_params.append(param)
            else:
                decay_params.append(param)
                
    optimizer_grouped_parameters = [
        {
            "params": decay_params,
            "weight_decay": args.weight_decay,
        },
        {
            "params": no_decay_params,
            "weight_decay": 0.0,
        },
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


def save_checkpoint(
    accelerator: Accelerator,
    model: torch.nn.Module,
    tokenizer,
    step: int,
    output_dir: str,
    fone_info: Dict[str, Any]
):
    """Save model checkpoint."""
    checkpoint_dir = os.path.join(output_dir, f"step_{step}")
    
    # Save with accelerator (handles DeepSpeed state)
    accelerator.save_state(checkpoint_dir)
    
    if accelerator.is_main_process:
        # Save model and tokenizer
        unwrapped_model = accelerator.unwrap_model(model)
        unwrapped_model.save_pretrained(checkpoint_dir)
        tokenizer.save_pretrained(checkpoint_dir)
        
        # Save FoNE information
        fone_config = {
            'overridden_indices': list(fone_info['overridden_indices']),
            'added_tokens': fone_info['added_tokens'],
            'num_overridden': fone_info['num_overridden']
        }
        
        with open(os.path.join(checkpoint_dir, 'fone_config.json'), 'w') as f:
            json.dump(fone_config, f, indent=2)
            
        logger.info(f"Saved checkpoint at step {step} to {checkpoint_dir}")


def load_checkpoint(
    accelerator: Accelerator,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    lr_scheduler,
    checkpoint_path: str
) -> int:
    """Load checkpoint and return the step number."""
    logger.info(f"Loading checkpoint from {checkpoint_path}")
    
    # Load accelerator state
    accelerator.load_state(checkpoint_path)
    
    # Extract step number from checkpoint path
    if "step_" in checkpoint_path:
        step = int(checkpoint_path.split("step_")[-1])
    else:
        step = 0
        
    logger.info(f"Resumed training from step {step}")
    return step


def train_step(
    model: torch.nn.Module,
    batch: Dict[str, torch.Tensor],
    accelerator: Accelerator
) -> Dict[str, float]:
    """Perform a single training step."""
    model.train()
    
    # Forward pass
    outputs = model(**batch)
    loss = outputs.loss
    
    # Backward pass
    accelerator.backward(loss)
    
    return {"loss": loss.detach().float()}


def run_validation(
    model_path: str,
    step: int,
    args: argparse.Namespace,
    accelerator: Accelerator
) -> Optional[Dict[str, Any]]:
    """Run validation during training."""
    if not args.eval_every_steps or step % args.eval_every_steps != 0:
        return None
    
    if not accelerator.is_main_process:
        return None
    
    logger.info(f"Running validation at step {step}")
    
    try:
        # Build validation command
        validation_cmd = [
            "python", "src/eval/run_validation.py",
            "--model", model_path,
            "--step", str(step),
            "--output_dir", os.path.join(args.output_dir, "validation_results")
        ]
        
        # Add validation data if it exists
        if os.path.exists(args.validation_data):
            validation_cmd.extend(["--validation_data", args.validation_data])
        else:
            validation_cmd.append("--skip_perplexity")
            logger.warning(f"Validation data not found: {args.validation_data}")
        
        # Add prompts file if it exists
        if os.path.exists(args.validation_prompts):
            validation_cmd.extend(["--prompts_file", args.validation_prompts])
        
        # Add quick validation flag if specified
        if args.quick_validation:
            validation_cmd.append("--quick")
        
        # Run validation
        result = subprocess.run(
            validation_cmd, 
            capture_output=True, 
            text=True, 
            timeout=1800  # 30 minute timeout
        )
        
        if result.returncode == 0:
            logger.info("Validation completed successfully")
            
            # Try to load and parse results
            results_file = os.path.join(
                args.output_dir, 
                "validation_results", 
                f"validation_summary_step_{step}.json"
            )
            
            if os.path.exists(results_file):
                with open(results_file, 'r') as f:
                    validation_results = json.load(f)
                
                # Extract key metrics for logging
                metrics_to_log = {}
                
                if "perplexity" in validation_results:
                    ppl_metrics = validation_results["perplexity"].get("metrics", {})
                    if "perplexity" in ppl_metrics:
                        metrics_to_log["validation_perplexity"] = ppl_metrics["perplexity"]
                    if "loss" in ppl_metrics:
                        metrics_to_log["validation_loss"] = ppl_metrics["loss"]
                
                if "text_generation" in validation_results:
                    gen_metrics = validation_results["text_generation"].get("overall_metrics", {})
                    if "avg_quality_score" in gen_metrics:
                        metrics_to_log["validation_text_quality"] = gen_metrics["avg_quality_score"]
                    if "successful_prompts" in gen_metrics and "total_prompts" in gen_metrics:
                        success_rate = gen_metrics["successful_prompts"] / gen_metrics["total_prompts"]
                        metrics_to_log["validation_success_rate"] = success_rate
                
                # Log to wandb if available
                if not args.no_wandb and metrics_to_log:
                    wandb.log(metrics_to_log)
                
                # Log key metrics
                if "validation_perplexity" in metrics_to_log:
                    logger.info(f"Validation perplexity: {metrics_to_log['validation_perplexity']:.2f}")
                if "validation_text_quality" in metrics_to_log:
                    logger.info(f"Validation text quality: {metrics_to_log['validation_text_quality']:.3f}")
                
                return validation_results
            else:
                logger.warning("Validation results file not found")
        else:
            logger.error(f"Validation failed with return code {result.returncode}")
            logger.error(f"stdout: {result.stdout}")
            logger.error(f"stderr: {result.stderr}")
    
    except subprocess.TimeoutExpired:
        logger.error("Validation timed out after 30 minutes")
    except Exception as e:
        logger.error(f"Error running validation: {e}")
    
    return None


def main():
    args = parse_args()
    
    # Initialize accelerator
    accelerator = Accelerator(
        gradient_accumulation_steps=(args.gradient_accumulation_steps or 1),
        log_with="wandb" if not args.no_wandb else None,
        project_dir=args.output_dir
    )
    
    # Set DeepSpeed micro batch size/grad accumulation if using DeepSpeed
    if accelerator.state.deepspeed_plugin is not None:
        ds_cfg = accelerator.state.deepspeed_plugin.deepspeed_config
        if args.per_device_train_batch_size is not None:
            ds_cfg['train_micro_batch_size_per_gpu'] = int(args.per_device_train_batch_size)
        # Respect explicit override for gradient accumulation
        if args.gradient_accumulation_steps is not None:
            ds_cfg['gradient_accumulation_steps'] = int(args.gradient_accumulation_steps)
    
    # Set up logging
    setup_logging(accelerator.process_index)
    
    # Set seed for reproducibility
    set_seed(42)
    
    # Set FlashAttention environment variable
    if args.flash_attn:
        os.environ["FLASH_ATTENTION"] = "1"
    
    # Initialize wandb
    if accelerator.is_main_process and not args.no_wandb:
        wandb.init(
            project=args.wandb_project,
            name=args.wandb_run_name,
            config=vars(args)
        )
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load model and tokenizer
    logger.info("Loading model and tokenizer...")
    model, tokenizer, fone_info = setup_llama_with_fone(
        config_path=args.model_config,
        tokenizer_name=args.tokenizer,
        flash_attention=args.flash_attn,
        fone_hi=args.fone_hi,
        freeze_fone_rows=not args.no_freeze_fone
    )
    
    # Enable activation checkpointing
    if args.activation_checkpointing:
        model.gradient_checkpointing_enable()
        logger.info("Enabled activation checkpointing")
    
    # Set up data loading
    logger.info("Setting up data loading...")
    data_mixture = DataMixture(
        mixture_config=args.dataset,
        sequence_length=args.seq_len,
        shuffle_buffer_size=10000,
        num_workers=args.dataloader_num_workers
    )
    
    # Create streaming dataset
    text_stream = data_mixture.get_streaming_iterator()
    dataset = StreamingPackedDataset(
        text_iterator=text_stream,
        tokenizer=tokenizer,
        sequence_length=args.seq_len
    )
    
    # Create optimizer
    optimizer = create_optimizer(model, args)
    
    # Compute training steps
    # Resolve effective micro batch size per GPU
    if accelerator.state.deepspeed_plugin is not None:
        ds_cfg = accelerator.state.deepspeed_plugin.deepspeed_config
        micro_batch_size = int(ds_cfg.get('train_micro_batch_size_per_gpu', 1))
    else:
        micro_batch_size = int(args.per_device_train_batch_size or 1)

    # Resolve accumulation steps
    gradient_accumulation_steps = int(args.gradient_accumulation_steps or (accelerator.gradient_accumulation_steps or 1))

    # Determine whether to count tokens per update globally or per-device
    processes_factor = accelerator.num_processes if args.total_tokens_scope == "global" else 1

    # Log breakdown for clarity
    tokens_per_update_per_gpu = micro_batch_size * args.seq_len * gradient_accumulation_steps
    tokens_per_update_effective = tokens_per_update_per_gpu * processes_factor
    logger.info(
        "Batching breakdown: per_gpu_micro_batch=%d, grad_accum=%d, num_processes=%d (scope=%s)",
        micro_batch_size,
        gradient_accumulation_steps,
        accelerator.num_processes,
        args.total_tokens_scope,
    )
    logger.info(
        "Tokens/update: per_gpu=%d, effective=%d; total_tokens=%d (%s)",
        tokens_per_update_per_gpu,
        tokens_per_update_effective,
        args.total_tokens,
        args.total_tokens_scope,
    )

    total_steps = compute_training_steps(
        total_tokens=args.total_tokens,
        micro_batch_size=micro_batch_size,
        sequence_length=args.seq_len,
        gradient_accumulation_steps=gradient_accumulation_steps,
        num_processes=processes_factor
    )
    
    # Create learning rate scheduler
    lr_scheduler = get_cosine_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=args.warmup_steps,
        num_training_steps=total_steps
    )
    
    # Prepare with accelerator
    model, optimizer, lr_scheduler = accelerator.prepare(model, optimizer, lr_scheduler)
    
    # Resume from checkpoint if specified
    start_step = 0
    if args.resume_from:
        start_step = load_checkpoint(
            accelerator, model, optimizer, lr_scheduler, args.resume_from
        )
    
    # Training loop
    logger.info("Starting training...")
    
    step = start_step
    data_iterator = iter(dataset)
    
    # Time tracking
    start_time = time.time()
    step_times = []
    
    model.train()
    
    while step < total_steps:
        try:
            with accelerator.accumulate(model):
                # Build per-GPU micro-batch by stacking examples
                samples = []
                for _ in range(micro_batch_size):
                    example = next(data_iterator)
                    # Ensure batch dimension exists for each example
                    example = {k: (v.unsqueeze(0) if isinstance(v, torch.Tensor) and v.dim() == 1 else v)
                               for k, v in example.items()}
                    samples.append(example)

                # Collate stacked batch
                keys = samples[0].keys()
                batch = {key: torch.cat([s[key] for s in samples], dim=0) for key in keys}

                # Move to device (accelerator handles this)
                batch = {k: v.to(accelerator.device) for k, v in batch.items()}
                
                # Training step
                metrics = train_step(model, batch, accelerator)
                
                # Gradient clipping
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(model.parameters(), args.grad_clip)
                
                # Optimizer step
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
                
                # Update step counter
                if accelerator.sync_gradients:
                    step += 1
                    
                    # Logging
                    if step % args.log_every_steps == 0:
                        current_time = time.time()
                        step_time = current_time - start_time if len(step_times) == 0 else current_time - step_times[-1]
                        step_times.append(current_time)
                        
                        # Calculate progress and time estimates
                        progress_pct = (step / total_steps) * 100
                        elapsed_time = current_time - start_time
                        
                        if step > start_step:
                            avg_step_time = elapsed_time / (step - start_step)
                            remaining_steps = total_steps - step
                            eta_seconds = remaining_steps * avg_step_time
                            eta_hours = eta_seconds / 3600
                            eta_mins = (eta_seconds % 3600) / 60
                            
                            if eta_hours >= 1:
                                eta_str = f"{eta_hours:.0f}h {eta_mins:.0f}m"
                            else:
                                eta_str = f"{eta_mins:.0f}m {eta_seconds % 60:.0f}s"
                        else:
                            eta_str = "calculating..."
                        
                        lr = lr_scheduler.get_last_lr()[0]
                        # Convert loss to float and compute perplexity (exp(loss))
                        loss_val = float(metrics["loss"]) if not isinstance(metrics["loss"], float) else metrics["loss"]
                        # Guard against overflow
                        ppl = math.exp(min(20.0, max(-20.0, loss_val)))
                        
                        tokens_processed_per_gpu = step * micro_batch_size * args.seq_len * gradient_accumulation_steps
                        tokens_processed_global = tokens_processed_per_gpu * accelerator.num_processes
                        log_metrics = {
                            "step": step,
                            "loss": loss_val,
                            "ppl": ppl,
                            "learning_rate": lr,
                            "tokens_processed_per_gpu": tokens_processed_per_gpu,
                            "tokens_processed_global": tokens_processed_global,
                            "progress_pct": progress_pct,
                            "eta_hours": eta_seconds / 3600 if step > start_step else 0,
                            "step_time": step_time
                        }
                        
                        if accelerator.is_main_process:
                            logger.info(f"Step {step}/{total_steps} ({progress_pct:.1f}%): "
                                       f"loss={loss_val:.4f}, ppl={ppl:.2f}, lr={lr:.2e}, ETA={eta_str}")
                            
                            if not args.no_wandb:
                                wandb.log(log_metrics)
                    
                    # Save checkpoint
                    if step % args.save_every_steps == 0:
                        checkpoint_path = os.path.join(args.output_dir, f"step_{step}")
                        save_checkpoint(
                            accelerator, model, tokenizer, step, args.output_dir, fone_info
                        )
                        
                        # Run validation after saving checkpoint
                        if args.eval_every_steps > 0:
                            run_validation(checkpoint_path, step, args, accelerator)
                        
        except StopIteration:
            logger.info("Data iterator exhausted")
            break
        except Exception as e:
            logger.error(f"Error during training step {step}: {e}")
            raise
    
    # Final checkpoint
    if accelerator.is_main_process:
        save_checkpoint(
            accelerator, model, tokenizer, step, args.output_dir, fone_info
        )
        
        logger.info(f"Training completed after {step} steps")
        
        # Final statistics
        dataset_stats = dataset.get_stats()
        logger.info(f"Dataset statistics: {dataset_stats}")
        
        if not args.no_wandb:
            wandb.log({
                "final_step": step,
                "total_tokens_processed": dataset_stats.get('total_tokens_processed', 0),
                "packing_efficiency": dataset_stats.get('packing_efficiency', 0)
            })
            wandb.finish()


if __name__ == "__main__":
    main()
