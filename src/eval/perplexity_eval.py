"""
Perplexity evaluation script for FoNE models during pretraining.

This script evaluates perplexity on validation datasets to track learning progress:
- Computes perplexity on held-out text data
- Supports multiple validation datasets
- Tracks perplexity trends over training
- Provides early stopping signals
"""

import os
import sys
import argparse
import logging
import json
import math
import time
from pathlib import Path
from typing import Dict, List, Optional, Any, Iterator
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer
from datasets import load_dataset
import numpy as np

# Add src to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from modeling.llama_1p5b import LlamaWithFoNE
from data.mixture import DataMixture

logger = logging.getLogger(__name__)


def setup_logging():
    """Set up logging configuration."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
        ]
    )


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Perplexity Evaluation for FoNE Models")
    
    # Model arguments
    parser.add_argument("--model", type=str, required=True,
                       help="Path to model checkpoint")
    parser.add_argument("--tokenizer", type=str, default=None,
                       help="Tokenizer path (defaults to model path)")
    
    # Data arguments
    parser.add_argument("--validation_data", type=str, required=True,
                       help="Path to validation data configuration or dataset name")
    parser.add_argument("--seq_len", type=int, default=2048,
                       help="Sequence length for evaluation")
    parser.add_argument("--max_samples", type=int, default=1000,
                       help="Maximum number of samples to evaluate")
    
    # Evaluation arguments
    parser.add_argument("--batch_size", type=int, default=1,
                       help="Batch size for evaluation")
    parser.add_argument("--stride", type=int, default=None,
                       help="Stride for sliding window evaluation (default: seq_len)")
    
    # Output arguments
    parser.add_argument("--output_file", type=str, default=None,
                       help="Output file for results")
    parser.add_argument("--step", type=int, default=None,
                       help="Training step (for tracking progress)")
    
    # Technical arguments
    parser.add_argument("--device", type=str, default="auto",
                       help="Device to use (auto, cpu, cuda)")
    
    return parser.parse_args()


def load_model_and_tokenizer(args: argparse.Namespace) -> tuple:
    """Load model and tokenizer."""
    tokenizer_path = args.tokenizer or args.model
    logger.info(f"Loading tokenizer from: {tokenizer_path}")
    
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    logger.info(f"Loading model from: {args.model}")
    
    try:
        # Try loading as FoNE model
        model_wrapper = LlamaWithFoNE.from_pretrained(args.model)
        model = model_wrapper.model
        logger.info("Loaded as FoNE model")
    except Exception as e:
        logger.warning(f"Could not load as FoNE model: {e}")
        # Fallback to standard Llama
        from transformers import LlamaForCausalLM
        model = LlamaForCausalLM.from_pretrained(args.model)
        logger.info("Loaded as standard Llama model")
    
    # Set device
    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device
        
    model = model.to(device)
    model.eval()
    
    logger.info(f"Model loaded on device: {device}")
    return model, tokenizer


def load_validation_data(data_path: str, tokenizer, seq_len: int, max_samples: int) -> Iterator[Dict[str, torch.Tensor]]:
    """Load validation data."""
    if os.path.exists(data_path) and data_path.endswith('.json'):
        # Assume it's a data mixture configuration
        logger.info(f"Loading data mixture from: {data_path}")
        data_mixture = DataMixture(
            mixture_config=data_path,
            sequence_length=seq_len,
            shuffle_buffer_size=1000,  # Smaller for validation
            num_workers=0
        )
        
        text_stream = data_mixture.get_streaming_iterator()
        
        # Convert to tokenized batches
        sample_count = 0
        current_batch = []
        
        for text in text_stream:
            if sample_count >= max_samples:
                break
                
            # Tokenize text
            tokens = tokenizer(text, max_length=seq_len, truncation=True, return_tensors="pt")
            
            # Only use sequences that are reasonably long
            if tokens.input_ids.shape[1] >= seq_len // 2:
                current_batch.append(tokens)
                sample_count += 1
                
                if len(current_batch) >= 1:  # Yield one at a time for memory efficiency
                    yield current_batch[0]
                    current_batch = []
        
        # Yield remaining batch
        if current_batch:
            yield current_batch[0]
            
    else:
        # Try loading as HuggingFace dataset
        logger.info(f"Loading HuggingFace dataset: {data_path}")
        try:
            if "/" in data_path:
                dataset_name, split = data_path.split("/", 1) if "/" in data_path else (data_path, "validation")
            else:
                dataset_name, split = data_path, "validation"
            
            dataset = load_dataset(dataset_name, split=split, streaming=True)
            
            sample_count = 0
            for example in dataset:
                if sample_count >= max_samples:
                    break
                
                # Extract text (try common field names)
                text = None
                for field in ["text", "content", "article", "document"]:
                    if field in example:
                        text = example[field]
                        break
                
                if text is None:
                    logger.warning(f"Could not find text field in example: {list(example.keys())}")
                    continue
                
                # Tokenize
                tokens = tokenizer(text, max_length=seq_len, truncation=True, return_tensors="pt")
                
                if tokens.input_ids.shape[1] >= seq_len // 2:
                    yield tokens
                    sample_count += 1
                    
        except Exception as e:
            logger.error(f"Failed to load dataset {data_path}: {e}")
            raise


def compute_perplexity_batch(
    model: torch.nn.Module,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor = None,
    stride: int = None
) -> Dict[str, float]:
    """Compute perplexity for a batch of sequences."""
    device = next(model.parameters()).device
    input_ids = input_ids.to(device)
    
    if attention_mask is not None:
        attention_mask = attention_mask.to(device)
    
    seq_len = input_ids.shape[1]
    if stride is None:
        stride = seq_len
    
    total_loss = 0.0
    total_tokens = 0
    
    with torch.no_grad():
        # Use sliding window if stride < seq_len
        for start_idx in range(0, seq_len, stride):
            end_idx = min(start_idx + seq_len, input_ids.shape[1])
            
            if end_idx - start_idx < 10:  # Skip very short segments
                continue
            
            # Get segment
            segment_ids = input_ids[:, start_idx:end_idx]
            segment_mask = attention_mask[:, start_idx:end_idx] if attention_mask is not None else None
            
            # Forward pass
            if segment_mask is not None:
                outputs = model(input_ids=segment_ids, attention_mask=segment_mask)
            else:
                outputs = model(input_ids=segment_ids)
            
            logits = outputs.logits
            
            # Compute loss (shift by 1 for causal LM)
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = segment_ids[:, 1:].contiguous()
            
            # Compute cross-entropy loss
            loss = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
                reduction='none'
            )
            
            # Mask out padding tokens if attention mask is provided
            if segment_mask is not None:
                shift_mask = segment_mask[:, 1:].contiguous()
                loss = loss * shift_mask.view(-1)
                valid_tokens = shift_mask.sum().item()
            else:
                valid_tokens = shift_labels.numel()
            
            total_loss += loss.sum().item()
            total_tokens += valid_tokens
            
            # If no stride, we're done
            if stride >= seq_len:
                break
    
    if total_tokens == 0:
        return {"loss": float('inf'), "perplexity": float('inf'), "tokens": 0}
    
    avg_loss = total_loss / total_tokens
    perplexity = math.exp(avg_loss)
    
    return {
        "loss": avg_loss,
        "perplexity": perplexity,
        "tokens": total_tokens
    }


def evaluate_perplexity(
    model: torch.nn.Module,
    tokenizer,
    validation_data: Iterator[Dict[str, torch.Tensor]],
    args: argparse.Namespace
) -> Dict[str, Any]:
    """Evaluate perplexity on validation data."""
    model.eval()
    
    total_loss = 0.0
    total_tokens = 0
    sample_count = 0
    sample_perplexities = []
    
    start_time = time.time()
    
    for batch in validation_data:
        try:
            input_ids = batch["input_ids"]
            attention_mask = batch.get("attention_mask", None)
            
            # Compute perplexity for this batch
            batch_metrics = compute_perplexity_batch(
                model=model,
                input_ids=input_ids,
                attention_mask=attention_mask,
                stride=args.stride
            )
            
            if batch_metrics["tokens"] > 0:
                total_loss += batch_metrics["loss"] * batch_metrics["tokens"]
                total_tokens += batch_metrics["tokens"]
                sample_perplexities.append(batch_metrics["perplexity"])
                sample_count += 1
                
                if sample_count % 100 == 0:
                    current_ppl = math.exp(total_loss / total_tokens) if total_tokens > 0 else float('inf')
                    logger.info(f"Processed {sample_count} samples, current perplexity: {current_ppl:.2f}")
            
        except Exception as e:
            logger.warning(f"Error processing sample {sample_count}: {e}")
            continue
    
    evaluation_time = time.time() - start_time
    
    # Compute final metrics
    if total_tokens > 0:
        avg_loss = total_loss / total_tokens
        perplexity = math.exp(avg_loss)
    else:
        avg_loss = float('inf')
        perplexity = float('inf')
    
    # Additional statistics
    sample_perplexities = [p for p in sample_perplexities if not math.isinf(p)]
    
    metrics = {
        "perplexity": perplexity,
        "loss": avg_loss,
        "total_tokens": total_tokens,
        "total_samples": sample_count,
        "evaluation_time": evaluation_time,
        "tokens_per_second": total_tokens / evaluation_time if evaluation_time > 0 else 0,
    }
    
    if sample_perplexities:
        metrics.update({
            "median_perplexity": np.median(sample_perplexities),
            "std_perplexity": np.std(sample_perplexities),
            "min_perplexity": min(sample_perplexities),
            "max_perplexity": max(sample_perplexities),
        })
    
    return metrics


def main():
    args = parse_args()
    setup_logging()
    
    # Load model and tokenizer
    model, tokenizer = load_model_and_tokenizer(args)
    
    # Load validation data
    logger.info("Loading validation data...")
    validation_data = load_validation_data(
        args.validation_data, 
        tokenizer, 
        args.seq_len, 
        args.max_samples
    )
    
    # Evaluate perplexity
    logger.info("Starting perplexity evaluation...")
    metrics = evaluate_perplexity(model, tokenizer, validation_data, args)
    
    # Log results
    logger.info(f"Evaluation completed!")
    logger.info(f"Perplexity: {metrics['perplexity']:.2f}")
    logger.info(f"Loss: {metrics['loss']:.4f}")
    logger.info(f"Total tokens: {metrics['total_tokens']:,}")
    logger.info(f"Total samples: {metrics['total_samples']}")
    logger.info(f"Evaluation time: {metrics['evaluation_time']:.2f}s")
    
    if "median_perplexity" in metrics:
        logger.info(f"Median perplexity: {metrics['median_perplexity']:.2f}")
        logger.info(f"Std perplexity: {metrics['std_perplexity']:.2f}")
    
    # Save results
    output_file = args.output_file or f"perplexity_eval_step_{args.step or 'unknown'}.json"
    
    output_data = {
        "args": vars(args),
        "timestamp": time.time(),
        "step": args.step,
        "metrics": metrics
    }
    
    with open(output_file, "w") as f:
        json.dump(output_data, f, indent=2)
    
    logger.info(f"Results saved to {output_file}")
    
    # Print summary
    print(f"\n{'='*60}")
    print(f"Perplexity Evaluation Results")
    if args.step:
        print(f"Training Step: {args.step}")
    print(f"{'='*60}")
    print(f"Model: {args.model}")
    print(f"Validation Data: {args.validation_data}")
    print(f"Perplexity: {metrics['perplexity']:.2f}")
    print(f"Loss: {metrics['loss']:.4f}")
    print(f"Tokens: {metrics['total_tokens']:,}")
    print(f"Samples: {metrics['total_samples']}")
    print(f"Time: {metrics['evaluation_time']:.2f}s")
    
    if "median_perplexity" in metrics:
        print(f"Median PPL: {metrics['median_perplexity']:.2f}")
        print(f"PPL Range: {metrics['min_perplexity']:.2f} - {metrics['max_perplexity']:.2f}")
    
    print(f"{'='*60}")
    
    return metrics["perplexity"]


if __name__ == "__main__":
    main()
