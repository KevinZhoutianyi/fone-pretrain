"""
Text generation evaluation script for FoNE models during pretraining.

This script provides qualitative assessment of text generation capabilities:
- Tests on diverse prompts (stories, conversations, instructions, etc.)
- Evaluates coherence, fluency, and relevance
- Provides samples for manual inspection
- Tracks generation quality over training steps
"""

import os
import sys
import argparse
import logging
import json
import time
from pathlib import Path
from typing import Dict, List, Optional, Any
import torch
from transformers import AutoTokenizer, GenerationConfig
import numpy as np

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
    parser = argparse.ArgumentParser(description="Text Generation Evaluation for FoNE Models")
    
    # Model arguments
    parser.add_argument("--model", type=str, required=True,
                       help="Path to model checkpoint")
    parser.add_argument("--tokenizer", type=str, default=None,
                       help="Tokenizer path (defaults to model path)")
    
    # Generation arguments
    parser.add_argument("--max_new_tokens", type=int, default=200,
                       help="Maximum number of tokens to generate")
    parser.add_argument("--temperature", type=float, default=0.8,
                       help="Generation temperature")
    parser.add_argument("--top_p", type=float, default=0.9,
                       help="Top-p sampling parameter")
    parser.add_argument("--repetition_penalty", type=float, default=1.1,
                       help="Repetition penalty")
    
    # Evaluation arguments
    parser.add_argument("--prompts_file", type=str, default=None,
                       help="Custom prompts file (JSON)")
    parser.add_argument("--num_samples", type=int, default=3,
                       help="Number of samples per prompt")
    
    # Output arguments
    parser.add_argument("--output_file", type=str, default=None,
                       help="Output file for results")
    parser.add_argument("--step", type=int, default=None,
                       help="Training step (for tracking progress)")
    
    # Technical arguments
    parser.add_argument("--device", type=str, default="auto",
                       help="Device to use (auto, cpu, cuda)")
    
    return parser.parse_args()


def get_default_prompts() -> List[Dict[str, str]]:
    """Get default validation prompts covering various text types."""
    return [
        {
            "category": "story_beginning",
            "prompt": "Once upon a time in a small village",
            "description": "Story continuation"
        },
        {
            "category": "conversation",
            "prompt": "Alice: How was your day?\nBob:",
            "description": "Dialogue continuation"
        },
        {
            "category": "instruction",
            "prompt": "Explain how to make a paper airplane:",
            "description": "Instructional text"
        },
        {
            "category": "factual",
            "prompt": "The capital of France is",
            "description": "Factual completion"
        },
        {
            "category": "creative",
            "prompt": "Write a short poem about the ocean:",
            "description": "Creative writing"
        },
        {
            "category": "reasoning",
            "prompt": "If it takes 5 machines 5 minutes to make 5 widgets, how long would it take 100 machines to make 100 widgets?",
            "description": "Logical reasoning"
        },
        {
            "category": "description",
            "prompt": "Describe a sunset over the mountains:",
            "description": "Descriptive writing"
        },
        {
            "category": "list",
            "prompt": "Here are 5 benefits of reading books:\n1.",
            "description": "List generation"
        },
        {
            "category": "question_answering",
            "prompt": "What is photosynthesis?",
            "description": "Question answering"
        },
        {
            "category": "code",
            "prompt": "Write a Python function to calculate the factorial of a number:",
            "description": "Code generation"
        },
        {
            "category": "email",
            "prompt": "Subject: Meeting Tomorrow\n\nDear Team,\n\nI hope this email finds you well.",
            "description": "Email continuation"
        },
        {
            "category": "news",
            "prompt": "BREAKING: Scientists discover",
            "description": "News article beginning"
        }
    ]


def load_custom_prompts(prompts_file: str) -> List[Dict[str, str]]:
    """Load custom prompts from JSON file."""
    with open(prompts_file, 'r') as f:
        prompts = json.load(f)
    
    # Validate format
    for prompt in prompts:
        if not all(key in prompt for key in ["category", "prompt", "description"]):
            raise ValueError("Each prompt must have 'category', 'prompt', and 'description' fields")
    
    return prompts


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
        # Fallbacks to load a standard Llama model
        from transformers import LlamaForCausalLM, AutoConfig
        try:
            model = LlamaForCausalLM.from_pretrained(args.model)
            logger.info("Loaded as standard Llama model")
        except Exception as e2:
            logger.warning(f"Standard from_pretrained failed: {e2}")
            # Final fallback: manually load from a local pytorch_model.bin inside the directory
            bin_path = os.path.join(args.model, "pytorch_model.bin")
            if not os.path.isfile(bin_path):
                raise
            logger.info(f"Manual load from state_dict: {bin_path}")
            config = AutoConfig.from_pretrained(args.model)
            model = LlamaForCausalLM.from_config(config)
            import torch as _torch
            state_dict = _torch.load(bin_path, map_location="cpu")
            missing, unexpected = model.load_state_dict(state_dict, strict=False)
            if missing or unexpected:
                logger.warning(f"State dict load warnings. Missing: {len(missing)}, Unexpected: {len(unexpected)}")
    
    # Set device
    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device
        
    model = model.to(device)
    model.eval()
    
    logger.info(f"Model loaded on device: {device}")
    return model, tokenizer


def generate_text(
    model: torch.nn.Module,
    tokenizer: AutoTokenizer,
    prompt: str,
    max_new_tokens: int = 200,
    temperature: float = 0.8,
    top_p: float = 0.9,
    repetition_penalty: float = 1.1,
    device: str = "cuda"
) -> str:
    """Generate text from model."""
    # Tokenize input
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    
    # Set up generation config
    generation_config = GenerationConfig(
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_p=top_p,
        repetition_penalty=repetition_penalty,
        do_sample=True,
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


def evaluate_generation_quality(text: str) -> Dict[str, Any]:
    """Simple heuristic evaluation of generation quality."""
    lines = text.split('\n')
    words = text.split()
    
    # Basic metrics
    metrics = {
        "length": len(text),
        "num_words": len(words),
        "num_lines": len(lines),
        "avg_word_length": np.mean([len(word) for word in words]) if words else 0,
    }
    
    # Repetition detection (simple)
    word_counts = {}
    for word in words:
        word_lower = word.lower().strip('.,!?;:"')
        word_counts[word_lower] = word_counts.get(word_lower, 0) + 1
    
    if words:
        # Most frequent word repetition
        max_repetition = max(word_counts.values()) if word_counts else 0
        metrics["max_word_repetition"] = max_repetition
        metrics["repetition_ratio"] = max_repetition / len(words)
        
        # Unique words ratio
        metrics["unique_words_ratio"] = len(word_counts) / len(words)
    else:
        metrics["max_word_repetition"] = 0
        metrics["repetition_ratio"] = 0
        metrics["unique_words_ratio"] = 0
    
    # Line repetition
    line_counts = {}
    for line in lines:
        line_stripped = line.strip()
        if line_stripped:
            line_counts[line_stripped] = line_counts.get(line_stripped, 0) + 1
    
    metrics["max_line_repetition"] = max(line_counts.values()) if line_counts else 0
    
    # Simple quality score (lower is better for repetition)
    quality_score = 1.0 - min(metrics["repetition_ratio"], 0.5) * 2  # 0 to 1 scale
    if metrics["length"] < 10:  # Penalize very short outputs
        quality_score *= 0.5
    
    metrics["quality_score"] = quality_score
    
    return metrics


def evaluate_prompt(
    model: torch.nn.Module,
    tokenizer: AutoTokenizer,
    prompt_data: Dict[str, str],
    args: argparse.Namespace
) -> Dict[str, Any]:
    """Evaluate a single prompt."""
    prompt = prompt_data["prompt"]
    category = prompt_data["category"]
    description = prompt_data["description"]
    
    device = next(model.parameters()).device
    
    # Generate multiple samples
    samples = []
    sample_metrics = []
    
    for i in range(args.num_samples):
        try:
            generated_text = generate_text(
                model=model,
                tokenizer=tokenizer,
                prompt=prompt,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
                top_p=args.top_p,
                repetition_penalty=args.repetition_penalty,
                device=device
            )
            
            # Evaluate quality
            metrics = evaluate_generation_quality(generated_text)
            
            samples.append({
                "sample_id": i,
                "generated_text": generated_text,
                "metrics": metrics
            })
            sample_metrics.append(metrics)
            
        except Exception as e:
            logger.warning(f"Error generating sample {i} for prompt '{category}': {e}")
            continue
    
    # Aggregate metrics across samples
    if sample_metrics:
        avg_metrics = {}
        for key in sample_metrics[0].keys():
            values = [m[key] for m in sample_metrics if key in m]
            avg_metrics[f"avg_{key}"] = np.mean(values) if values else 0
            avg_metrics[f"std_{key}"] = np.std(values) if len(values) > 1 else 0
    else:
        avg_metrics = {}
    
    return {
        "prompt": prompt,
        "category": category,
        "description": description,
        "samples": samples,
        "num_successful_samples": len(samples),
        "aggregate_metrics": avg_metrics
    }


def main():
    args = parse_args()
    setup_logging()
    
    # Load model and tokenizer
    model, tokenizer = load_model_and_tokenizer(args)
    
    # Load prompts
    if args.prompts_file:
        logger.info(f"Loading custom prompts from: {args.prompts_file}")
        prompts = load_custom_prompts(args.prompts_file)
    else:
        logger.info("Using default prompts")
        prompts = get_default_prompts()
    
    logger.info(f"Evaluating on {len(prompts)} prompts with {args.num_samples} samples each")
    
    # Evaluate each prompt
    results = []
    overall_metrics = {
        "total_prompts": len(prompts),
        "successful_prompts": 0,
        "total_samples": 0,
        "successful_samples": 0,
        "avg_quality_score": 0,
        "categories": {}
    }
    
    start_time = time.time()
    
    for i, prompt_data in enumerate(prompts):
        logger.info(f"Evaluating prompt {i+1}/{len(prompts)}: {prompt_data['category']}")
        
        try:
            result = evaluate_prompt(model, tokenizer, prompt_data, args)
            results.append(result)
            
            # Update overall metrics
            if result["num_successful_samples"] > 0:
                overall_metrics["successful_prompts"] += 1
                overall_metrics["successful_samples"] += result["num_successful_samples"]
                
                # Track by category
                category = result["category"]
                if category not in overall_metrics["categories"]:
                    overall_metrics["categories"][category] = {
                        "count": 0,
                        "successful_samples": 0,
                        "avg_quality": 0
                    }
                
                cat_metrics = overall_metrics["categories"][category]
                cat_metrics["count"] += 1
                cat_metrics["successful_samples"] += result["num_successful_samples"]
                
                if "avg_quality_score" in result["aggregate_metrics"]:
                    cat_metrics["avg_quality"] += result["aggregate_metrics"]["avg_quality_score"]
            
            overall_metrics["total_samples"] += args.num_samples
            
        except Exception as e:
            logger.error(f"Error evaluating prompt {i}: {e}")
            continue
    
    # Finalize metrics
    evaluation_time = time.time() - start_time
    
    if overall_metrics["successful_prompts"] > 0:
        # Calculate average quality score across all samples
        total_quality = 0
        quality_count = 0
        
        for result in results:
            if "avg_quality_score" in result["aggregate_metrics"]:
                total_quality += result["aggregate_metrics"]["avg_quality_score"] * result["num_successful_samples"]
                quality_count += result["num_successful_samples"]
        
        overall_metrics["avg_quality_score"] = total_quality / quality_count if quality_count > 0 else 0
        
        # Finalize category metrics
        for category, cat_metrics in overall_metrics["categories"].items():
            if cat_metrics["count"] > 0:
                cat_metrics["avg_quality"] /= cat_metrics["count"]
    
    # Log results
    logger.info(f"Evaluation completed in {evaluation_time:.2f} seconds")
    logger.info(f"Successful prompts: {overall_metrics['successful_prompts']}/{overall_metrics['total_prompts']}")
    logger.info(f"Successful samples: {overall_metrics['successful_samples']}/{overall_metrics['total_samples']}")
    logger.info(f"Average quality score: {overall_metrics['avg_quality_score']:.3f}")
    
    # Save results
    output_file = args.output_file or f"text_generation_eval_step_{args.step or 'unknown'}.json"
    
    output_data = {
        "args": vars(args),
        "evaluation_time": evaluation_time,
        "timestamp": time.time(),
        "step": args.step,
        "overall_metrics": overall_metrics,
        "results": results
    }
    
    with open(output_file, "w") as f:
        json.dump(output_data, f, indent=2)
    
    logger.info(f"Results saved to {output_file}")
    
    # Print summary
    print(f"\n{'='*60}")
    print(f"Text Generation Evaluation Results")
    if args.step:
        print(f"Training Step: {args.step}")
    print(f"{'='*60}")
    print(f"Model: {args.model}")
    print(f"Prompts: {overall_metrics['successful_prompts']}/{overall_metrics['total_prompts']}")
    print(f"Samples: {overall_metrics['successful_samples']}/{overall_metrics['total_samples']}")
    print(f"Average Quality Score: {overall_metrics['avg_quality_score']:.3f}")
    print(f"Evaluation Time: {evaluation_time:.2f}s")
    
    # Category breakdown
    if overall_metrics["categories"]:
        print(f"\nCategory Breakdown:")
        for category, metrics in overall_metrics["categories"].items():
            print(f"  {category}: {metrics['successful_samples']} samples, "
                  f"quality: {metrics['avg_quality']:.3f}")
    
    print(f"\nDetailed results saved to: {output_file}")
    print(f"{'='*60}")
    
    # Print a few sample generations for manual inspection
    print(f"\nSample Generations:")
    print(f"{'='*60}")
    
    for result in results[:3]:  # Show first 3 prompts
        if result["samples"]:
            print(f"\nPrompt ({result['category']}): {result['prompt']}")
            print(f"Generated: {result['samples'][0]['generated_text'][:200]}...")
            print(f"Quality Score: {result['samples'][0]['metrics']['quality_score']:.3f}")
    
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
