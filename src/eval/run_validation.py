"""
Comprehensive validation runner for FoNE models during pretraining.

This script runs multiple validation checks:
1. Perplexity evaluation on held-out data
2. Text generation quality assessment
3. Progress tracking and comparison
4. Early stopping recommendations
"""

import os
import sys
import argparse
import logging
import json
import time
from pathlib import Path
from typing import Dict, Any, Optional
import subprocess

# Add src to path for imports
sys.path.append(str(Path(__file__).parent.parent))

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
    parser = argparse.ArgumentParser(description="Comprehensive Validation for FoNE Models")
    
    # Model arguments
    parser.add_argument("--model", type=str, required=True,
                       help="Path to model checkpoint")
    parser.add_argument("--tokenizer", type=str, default=None,
                       help="Tokenizer path (defaults to model path)")
    parser.add_argument("--step", type=int, default=None,
                       help="Training step (for tracking progress)")
    
    # Validation data
    parser.add_argument("--validation_data", type=str, default=None,
                       help="Path to validation data for perplexity evaluation")
    parser.add_argument("--prompts_file", type=str, default="configs/validation_prompts.json",
                       help="Path to validation prompts file")
    
    # Evaluation settings
    parser.add_argument("--seq_len", type=int, default=2048,
                       help="Sequence length for evaluation")
    parser.add_argument("--max_samples", type=int, default=500,
                       help="Maximum samples for perplexity evaluation")
    parser.add_argument("--num_text_samples", type=int, default=2,
                       help="Number of text samples per prompt")
    parser.add_argument("--max_new_tokens", type=int, default=150,
                       help="Maximum tokens for text generation")
    
    # Output settings
    parser.add_argument("--output_dir", type=str, default="validation_results",
                       help="Output directory for results")
    parser.add_argument("--save_generations", action="store_true",
                       help="Save all generated text samples")
    
    # Control flags
    parser.add_argument("--skip_perplexity", action="store_true",
                       help="Skip perplexity evaluation")
    parser.add_argument("--skip_generation", action="store_true",
                       help="Skip text generation evaluation")
    parser.add_argument("--quick", action="store_true",
                       help="Run quick validation (fewer samples)")
    
    # Technical arguments
    parser.add_argument("--device", type=str, default="auto",
                       help="Device to use (auto, cpu, cuda)")
    
    return parser.parse_args()


def run_perplexity_evaluation(
    model_path: str,
    validation_data: str,
    output_dir: str,
    step: Optional[int] = None,
    **kwargs
) -> Dict[str, Any]:
    """Run perplexity evaluation."""
    logger.info("Running perplexity evaluation...")
    
    cmd = [
        "python", "src/eval/perplexity_eval.py",
        "--model", model_path,
        "--validation_data", validation_data,
        "--output_file", os.path.join(output_dir, f"perplexity_step_{step or 'unknown'}.json")
    ]
    
    # Add optional arguments
    for key, value in kwargs.items():
        if value is not None:
            if isinstance(value, bool) and value:
                cmd.append(f"--{key}")
            elif not isinstance(value, bool):
                cmd.extend([f"--{key}", str(value)])
    
    if step is not None:
        cmd.extend(["--step", str(step)])
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        logger.info("Perplexity evaluation completed successfully")
        
        # Load and return results
        output_file = os.path.join(output_dir, f"perplexity_step_{step or 'unknown'}.json")
        if os.path.exists(output_file):
            with open(output_file, 'r') as f:
                return json.load(f)
        
    except subprocess.CalledProcessError as e:
        logger.error(f"Perplexity evaluation failed: {e}")
        logger.error(f"stdout: {e.stdout}")
        logger.error(f"stderr: {e.stderr}")
    
    return {}


def run_text_generation_evaluation(
    model_path: str,
    prompts_file: str,
    output_dir: str,
    step: Optional[int] = None,
    **kwargs
) -> Dict[str, Any]:
    """Run text generation evaluation."""
    logger.info("Running text generation evaluation...")
    
    cmd = [
        "python", "src/eval/text_generation_eval.py",
        "--model", model_path,
        "--prompts_file", prompts_file,
        "--output_file", os.path.join(output_dir, f"text_generation_step_{step or 'unknown'}.json")
    ]
    
    # Add optional arguments
    for key, value in kwargs.items():
        if value is not None:
            if isinstance(value, bool) and value:
                cmd.append(f"--{key}")
            elif not isinstance(value, bool):
                cmd.extend([f"--{key}", str(value)])
    
    if step is not None:
        cmd.extend(["--step", str(step)])
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        logger.info("Text generation evaluation completed successfully")
        
        # Load and return results
        output_file = os.path.join(output_dir, f"text_generation_step_{step or 'unknown'}.json")
        if os.path.exists(output_file):
            with open(output_file, 'r') as f:
                return json.load(f)
        
    except subprocess.CalledProcessError as e:
        logger.error(f"Text generation evaluation failed: {e}")
        logger.error(f"stdout: {e.stdout}")
        logger.error(f"stderr: {e.stderr}")
    
    return {}


def load_previous_results(output_dir: str) -> Dict[int, Dict[str, Any]]:
    """Load previous validation results for comparison."""
    results = {}
    
    if not os.path.exists(output_dir):
        return results
    
    # Look for previous validation summary files
    for filename in os.listdir(output_dir):
        if filename.startswith("validation_summary_step_") and filename.endswith(".json"):
            try:
                step = int(filename.replace("validation_summary_step_", "").replace(".json", ""))
                with open(os.path.join(output_dir, filename), 'r') as f:
                    results[step] = json.load(f)
            except (ValueError, json.JSONDecodeError) as e:
                logger.warning(f"Could not load previous result {filename}: {e}")
    
    return results


def analyze_progress(
    current_step: int,
    current_results: Dict[str, Any],
    previous_results: Dict[int, Dict[str, Any]]
) -> Dict[str, Any]:
    """Analyze progress compared to previous evaluations."""
    analysis = {
        "step": current_step,
        "improvements": {},
        "regressions": {},
        "trends": {},
        "recommendations": []
    }
    
    if not previous_results:
        analysis["recommendations"].append("First validation - no previous results to compare")
        return analysis
    
    # Find most recent previous result
    previous_steps = sorted(previous_results.keys())
    if not previous_steps:
        return analysis
    
    recent_step = previous_steps[-1]
    recent_results = previous_results[recent_step]
    
    # Compare perplexity
    if "perplexity" in current_results and "perplexity" in recent_results:
        current_ppl = current_results["perplexity"].get("metrics", {}).get("perplexity", float('inf'))
        previous_ppl = recent_results["perplexity"].get("metrics", {}).get("perplexity", float('inf'))
        
        if current_ppl < previous_ppl:
            improvement = ((previous_ppl - current_ppl) / previous_ppl) * 100
            analysis["improvements"]["perplexity"] = {
                "current": current_ppl,
                "previous": previous_ppl,
                "improvement_pct": improvement
            }
        elif current_ppl > previous_ppl:
            regression = ((current_ppl - previous_ppl) / previous_ppl) * 100
            analysis["regressions"]["perplexity"] = {
                "current": current_ppl,
                "previous": previous_ppl,
                "regression_pct": regression
            }
    
    # Compare text generation quality
    if "text_generation" in current_results and "text_generation" in recent_results:
        current_quality = current_results["text_generation"].get("overall_metrics", {}).get("avg_quality_score", 0)
        previous_quality = recent_results["text_generation"].get("overall_metrics", {}).get("avg_quality_score", 0)
        
        if current_quality > previous_quality:
            improvement = ((current_quality - previous_quality) / previous_quality) * 100 if previous_quality > 0 else 0
            analysis["improvements"]["text_quality"] = {
                "current": current_quality,
                "previous": previous_quality,
                "improvement_pct": improvement
            }
        elif current_quality < previous_quality:
            regression = ((previous_quality - current_quality) / previous_quality) * 100 if previous_quality > 0 else 0
            analysis["regressions"]["text_quality"] = {
                "current": current_quality,
                "previous": previous_quality,
                "regression_pct": regression
            }
    
    # Generate recommendations
    if analysis["improvements"]:
        analysis["recommendations"].append("Model is showing improvements - continue training")
    
    if analysis["regressions"]:
        if "perplexity" in analysis["regressions"]:
            ppl_regression = analysis["regressions"]["perplexity"]["regression_pct"]
            if ppl_regression > 5:
                analysis["recommendations"].append("Significant perplexity regression - consider reducing learning rate")
            elif ppl_regression > 10:
                analysis["recommendations"].append("Large perplexity regression - possible overfitting, consider early stopping")
    
    if len(previous_steps) >= 3:
        # Analyze trends over multiple steps
        recent_steps = previous_steps[-3:] + [current_step]
        perplexities = []
        
        for step in recent_steps[:-1]:
            ppl = previous_results[step].get("perplexity", {}).get("metrics", {}).get("perplexity", None)
            if ppl is not None:
                perplexities.append(ppl)
        
        # Add current perplexity
        current_ppl = current_results.get("perplexity", {}).get("metrics", {}).get("perplexity", None)
        if current_ppl is not None:
            perplexities.append(current_ppl)
        
        if len(perplexities) >= 3:
            # Simple trend analysis
            recent_trend = perplexities[-1] - perplexities[-2]
            overall_trend = perplexities[-1] - perplexities[0]
            
            if recent_trend > 0 and overall_trend > 0:
                analysis["trends"]["perplexity"] = "increasing"
                analysis["recommendations"].append("Perplexity trend is increasing - monitor closely")
            elif recent_trend < 0 and overall_trend < 0:
                analysis["trends"]["perplexity"] = "decreasing"
                analysis["recommendations"].append("Perplexity trend is decreasing - good progress")
            else:
                analysis["trends"]["perplexity"] = "mixed"
    
    return analysis


def main():
    args = parse_args()
    setup_logging()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Adjust settings for quick mode
    if args.quick:
        args.max_samples = min(args.max_samples, 100)
        args.num_text_samples = 1
        args.max_new_tokens = 100
        logger.info("Running in quick mode with reduced samples")
    
    logger.info(f"Starting comprehensive validation for model: {args.model}")
    if args.step:
        logger.info(f"Training step: {args.step}")
    
    start_time = time.time()
    results = {
        "timestamp": time.time(),
        "step": args.step,
        "model": args.model,
        "args": vars(args)
    }
    
    # Run perplexity evaluation
    if not args.skip_perplexity and args.validation_data:
        perplexity_results = run_perplexity_evaluation(
            model_path=args.model,
            validation_data=args.validation_data,
            output_dir=args.output_dir,
            step=args.step,
            seq_len=args.seq_len,
            max_samples=args.max_samples,
            tokenizer=args.tokenizer,
            device=args.device
        )
        results["perplexity"] = perplexity_results
    elif not args.validation_data:
        logger.warning("No validation data provided - skipping perplexity evaluation")
    
    # Run text generation evaluation
    if not args.skip_generation:
        generation_results = run_text_generation_evaluation(
            model_path=args.model,
            prompts_file=args.prompts_file,
            output_dir=args.output_dir,
            step=args.step,
            num_samples=args.num_text_samples,
            max_new_tokens=args.max_new_tokens,
            tokenizer=args.tokenizer,
            device=args.device
        )
        results["text_generation"] = generation_results
    
    # Load previous results for comparison
    previous_results = load_previous_results(args.output_dir)
    
    # Analyze progress
    if args.step:
        progress_analysis = analyze_progress(args.step, results, previous_results)
        results["progress_analysis"] = progress_analysis
    
    # Save comprehensive results
    total_time = time.time() - start_time
    results["total_evaluation_time"] = total_time
    
    summary_file = os.path.join(args.output_dir, f"validation_summary_step_{args.step or 'unknown'}.json")
    with open(summary_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    # Print summary
    print(f"\n{'='*80}")
    print(f"COMPREHENSIVE VALIDATION RESULTS")
    if args.step:
        print(f"Training Step: {args.step}")
    print(f"{'='*80}")
    print(f"Model: {args.model}")
    print(f"Total Evaluation Time: {total_time:.2f}s")
    
    # Perplexity results
    if "perplexity" in results and results["perplexity"]:
        ppl_metrics = results["perplexity"].get("metrics", {})
        print(f"\nPerplexity Evaluation:")
        print(f"  Perplexity: {ppl_metrics.get('perplexity', 'N/A'):.2f}")
        print(f"  Loss: {ppl_metrics.get('loss', 'N/A'):.4f}")
        print(f"  Tokens: {ppl_metrics.get('total_tokens', 'N/A'):,}")
    
    # Text generation results
    if "text_generation" in results and results["text_generation"]:
        gen_metrics = results["text_generation"].get("overall_metrics", {})
        print(f"\nText Generation Evaluation:")
        print(f"  Average Quality Score: {gen_metrics.get('avg_quality_score', 'N/A'):.3f}")
        print(f"  Successful Prompts: {gen_metrics.get('successful_prompts', 'N/A')}/{gen_metrics.get('total_prompts', 'N/A')}")
        print(f"  Successful Samples: {gen_metrics.get('successful_samples', 'N/A')}/{gen_metrics.get('total_samples', 'N/A')}")
    
    # Progress analysis
    if "progress_analysis" in results:
        analysis = results["progress_analysis"]
        print(f"\nProgress Analysis:")
        
        if analysis["improvements"]:
            print(f"  Improvements:")
            for metric, data in analysis["improvements"].items():
                print(f"    {metric}: {data['improvement_pct']:.1f}% better")
        
        if analysis["regressions"]:
            print(f"  Regressions:")
            for metric, data in analysis["regressions"].items():
                print(f"    {metric}: {data['regression_pct']:.1f}% worse")
        
        if analysis["recommendations"]:
            print(f"  Recommendations:")
            for rec in analysis["recommendations"]:
                print(f"    - {rec}")
    
    print(f"\nDetailed results saved to: {summary_file}")
    print(f"{'='*80}")
    
    logger.info(f"Comprehensive validation completed in {total_time:.2f}s")


if __name__ == "__main__":
    main()
