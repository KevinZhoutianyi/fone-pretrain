"""
GSM8K evaluation script for FoNE models.

This script evaluates mathematical reasoning performance on GSM8K with support for:
- Deterministic decoding
- Self-consistency decoding (K samples)
- Exact match scoring
- Chain-of-thought answer extraction
"""

import os
import sys
import argparse
import logging
import json
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import torch
from transformers import AutoTokenizer, GenerationConfig
from datasets import load_dataset
from collections import Counter
import numpy as np
from tqdm import tqdm

# Add src to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from modeling.llama_1p5b import LlamaWithFoNE

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
    parser = argparse.ArgumentParser(description="GSM8K Evaluation for FoNE Models")
    
    # Model arguments
    parser.add_argument("--model", type=str, required=True,
                       help="Path to model checkpoint (base model or LoRA)")
    parser.add_argument("--tokenizer", type=str, default=None,
                       help="Tokenizer path (defaults to model path)")
    parser.add_argument("--is_lora", action="store_true",
                       help="Model is a LoRA adapter")
    parser.add_argument("--base_model", type=str, default=None,
                       help="Base model path (required if --is_lora)")
    
    # Evaluation arguments
    parser.add_argument("--dataset", type=str, default="gsm8k",
                       help="Dataset to evaluate on")
    parser.add_argument("--split", type=str, default="test",
                       help="Dataset split to use")
    parser.add_argument("--num_examples", type=int, default=None,
                       help="Number of examples to evaluate (None for all)")
    
    # Generation arguments
    parser.add_argument("--max_new_tokens", type=int, default=512,
                       help="Maximum number of tokens to generate")
    parser.add_argument("--temperature", type=float, default=0.0,
                       help="Generation temperature (0.0 for deterministic)")
    parser.add_argument("--top_p", type=float, default=1.0,
                       help="Top-p sampling parameter")
    parser.add_argument("--k", type=int, default=1,
                       help="Number of samples for self-consistency (1 for deterministic)")
    
    # Output arguments
    parser.add_argument("--output_file", type=str, default=None,
                       help="Output file for detailed results")
    parser.add_argument("--save_predictions", action="store_true",
                       help="Save predictions to output file")
    
    # Technical arguments
    parser.add_argument("--batch_size", type=int, default=1,
                       help="Batch size for evaluation")
    parser.add_argument("--device", type=str, default="auto",
                       help="Device to use (auto, cpu, cuda)")
    
    return parser.parse_args()


def extract_numeric_answer(text: str) -> Optional[float]:
    """
    Extract numeric answer from generated text.
    
    Looks for patterns like "#### 42" or "The answer is 42"
    """
    # First, look for #### pattern (GSM8K standard)
    pattern = r"####\s*([+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?)"
    match = re.search(pattern, text)
    if match:
        try:
            return float(match.group(1).replace(",", ""))
        except ValueError:
            pass
    
    # Look for "The answer is X" pattern
    pattern = r"(?:the answer is|answer:|final answer:)\s*([+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?)"
    match = re.search(pattern, text, re.IGNORECASE)
    if match:
        try:
            return float(match.group(1).replace(",", ""))
        except ValueError:
            pass
    
    # Look for numbers at the end of the text
    pattern = r"([+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?)(?:\s*(?:\.|$))"
    matches = re.findall(pattern, text)
    if matches:
        try:
            return float(matches[-1].replace(",", ""))
        except ValueError:
            pass
    
    return None


def extract_gsm8k_answer(answer_text: str) -> Optional[float]:
    """Extract the ground truth numeric answer from GSM8K answer text."""
    if "####" in answer_text:
        parts = answer_text.split("####")
        if len(parts) > 1:
            try:
                return float(parts[-1].strip().replace(",", ""))
            except ValueError:
                pass
    
    # Fallback: look for numbers in the text
    numbers = re.findall(r"([+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?)", answer_text)
    if numbers:
        try:
            return float(numbers[-1].replace(",", ""))
        except ValueError:
            pass
    
    return None


def format_prompt(question: str) -> str:
    """Format question into chat template."""
    return f"<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n{question}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"


def load_model_and_tokenizer(args: argparse.Namespace) -> Tuple[torch.nn.Module, AutoTokenizer]:
    """Load model and tokenizer based on arguments."""
    tokenizer_path = args.tokenizer or args.model
    logger.info(f"Loading tokenizer from: {tokenizer_path}")
    
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    if args.is_lora:
        if not args.base_model:
            raise ValueError("--base_model required when --is_lora is specified")
            
        logger.info(f"Loading LoRA model from: {args.model}")
        logger.info(f"Base model: {args.base_model}")
        
        # Load base model
        try:
            base_model_wrapper = LlamaWithFoNE.from_pretrained(args.base_model)
            base_model = base_model_wrapper.model
        except Exception as e:
            logger.warning(f"Could not load as FoNE model: {e}")
            from transformers import LlamaForCausalLM
            base_model = LlamaForCausalLM.from_pretrained(args.base_model)
        
        # Load LoRA adapter
        from peft import PeftModel
        model = PeftModel.from_pretrained(base_model, args.model)
        
    else:
        logger.info(f"Loading model from: {args.model}")
        
        try:
            model_wrapper = LlamaWithFoNE.from_pretrained(args.model)
            model = model_wrapper.model
        except Exception as e:
            logger.warning(f"Could not load as FoNE model: {e}")
            from transformers import LlamaForCausalLM
            model = LlamaForCausalLM.from_pretrained(args.model)
    
    # Set device
    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device
        
    model = model.to(device)
    model.eval()
    
    logger.info(f"Model loaded on device: {device}")
    return model, tokenizer


def generate_response(
    model: torch.nn.Module,
    tokenizer: AutoTokenizer,
    prompt: str,
    max_new_tokens: int = 512,
    temperature: float = 0.0,
    top_p: float = 1.0,
    device: str = "cuda"
) -> str:
    """Generate response from model."""
    # Tokenize input
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    
    # Set up generation config
    generation_config = GenerationConfig(
        max_new_tokens=max_new_tokens,
        temperature=temperature if temperature > 0 else None,
        top_p=top_p if temperature > 0 else None,
        do_sample=temperature > 0,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )
    
    # Generate
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            generation_config=generation_config,
        )
    
    # Decode response (excluding input)
    response = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
    return response.strip()


def evaluate_example(
    model: torch.nn.Module,
    tokenizer: AutoTokenizer,
    question: str,
    ground_truth: str,
    args: argparse.Namespace
) -> Dict[str, Any]:
    """Evaluate a single example."""
    prompt = format_prompt(question)
    device = next(model.parameters()).device
    
    # Generate K responses
    responses = []
    predictions = []
    
    for _ in range(args.k):
        response = generate_response(
            model=model,
            tokenizer=tokenizer,
            prompt=prompt,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
            device=device
        )
        
        responses.append(response)
        
        # Extract numeric prediction
        pred = extract_numeric_answer(response)
        predictions.append(pred)
    
    # Extract ground truth
    gt_value = extract_gsm8k_answer(ground_truth)
    
    # Determine final prediction
    if args.k == 1:
        # Deterministic
        final_prediction = predictions[0]
    else:
        # Self-consistency: majority vote among valid predictions
        valid_preds = [p for p in predictions if p is not None]
        if valid_preds:
            # Count occurrences and take majority
            pred_counts = Counter(valid_preds)
            final_prediction = pred_counts.most_common(1)[0][0]
        else:
            final_prediction = None
    
    # Check if correct
    is_correct = False
    if final_prediction is not None and gt_value is not None:
        # Allow small numerical differences
        is_correct = abs(final_prediction - gt_value) < 1e-6
    
    return {
        "question": question,
        "ground_truth": ground_truth,
        "gt_value": gt_value,
        "responses": responses,
        "predictions": predictions,
        "final_prediction": final_prediction,
        "is_correct": is_correct,
        "k": args.k
    }


def main():
    args = parse_args()
    setup_logging()
    
    # Load model and tokenizer
    model, tokenizer = load_model_and_tokenizer(args)
    
    # Load dataset
    logger.info(f"Loading dataset: {args.dataset}")
    if args.dataset == "gsm8k":
        dataset = load_dataset("gsm8k", "main", split=args.split)
    else:
        dataset = load_dataset(args.dataset, split=args.split)
    
    # Limit examples if specified
    if args.num_examples:
        dataset = dataset.select(range(min(args.num_examples, len(dataset))))
    
    logger.info(f"Evaluating on {len(dataset)} examples")
    
    # Evaluate
    results = []
    correct_count = 0
    
    for i, example in enumerate(tqdm(dataset, desc="Evaluating")):
        question = example["question"]
        answer = example["answer"]
        
        try:
            result = evaluate_example(model, tokenizer, question, answer, args)
            results.append(result)
            
            if result["is_correct"]:
                correct_count += 1
                
            # Log progress
            if (i + 1) % 50 == 0:
                accuracy = correct_count / (i + 1)
                logger.info(f"Progress: {i+1}/{len(dataset)}, Accuracy: {accuracy:.3f}")
                
        except Exception as e:
            logger.error(f"Error evaluating example {i}: {e}")
            continue
    
    # Compute final metrics
    total_evaluated = len(results)
    accuracy = correct_count / total_evaluated if total_evaluated > 0 else 0.0
    
    # Log results
    logger.info(f"Evaluation completed!")
    logger.info(f"Total examples: {total_evaluated}")
    logger.info(f"Correct: {correct_count}")
    logger.info(f"Accuracy: {accuracy:.4f}")
    
    if args.k > 1:
        logger.info(f"Self-consistency with k={args.k}")
    
    # Detailed analysis
    valid_predictions = sum(1 for r in results if r["final_prediction"] is not None)
    valid_gt = sum(1 for r in results if r["gt_value"] is not None)
    
    logger.info(f"Valid predictions: {valid_predictions}/{total_evaluated} ({valid_predictions/total_evaluated:.3f})")
    logger.info(f"Valid ground truth: {valid_gt}/{total_evaluated} ({valid_gt/total_evaluated:.3f})")
    
    # Save results if requested
    if args.output_file or args.save_predictions:
        output_file = args.output_file or f"gsm8k_results_{args.k}shot.json"
        
        output_data = {
            "args": vars(args),
            "metrics": {
                "accuracy": accuracy,
                "correct": correct_count,
                "total": total_evaluated,
                "valid_predictions": valid_predictions,
                "valid_ground_truth": valid_gt,
            },
            "results": results if args.save_predictions else []
        }
        
        with open(output_file, "w") as f:
            json.dump(output_data, f, indent=2)
            
        logger.info(f"Results saved to {output_file}")
    
    # Print summary
    print(f"\n{'='*50}")
    print(f"GSM8K Evaluation Results")
    print(f"{'='*50}")
    print(f"Model: {args.model}")
    print(f"Examples: {total_evaluated}")
    print(f"Correct: {correct_count}")
    print(f"Accuracy: {accuracy:.4f}")
    if args.k > 1:
        print(f"Self-consistency: k={args.k}")
    print(f"{'='*50}")


if __name__ == "__main__":
    main()
