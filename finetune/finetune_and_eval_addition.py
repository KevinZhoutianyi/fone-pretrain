#!/usr/bin/env python3
"""
Complete pipeline to finetune and evaluate models on Onlydrinkwater/1000addition dataset.
This tests arithmetic ability - where FoNE should excel!
"""

import argparse
import json
import random
import re
import torch
from pathlib import Path
from datetime import datetime
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
)
from datasets import load_dataset, Dataset
from tqdm import tqdm
import os
from dotenv import load_dotenv

load_dotenv()


def load_addition_dataset(split="train", max_samples=None):
    """Load the 1000addition dataset from HuggingFace."""
    print(f"Loading Onlydrinkwater/1000addition dataset ({split} split)...")
    dataset = load_dataset("Onlydrinkwater/1000addition", split=split)
    
    if max_samples:
        dataset = dataset.select(range(min(max_samples, len(dataset))))
    
    print(f"✅ Loaded {len(dataset)} examples")
    return dataset


def prepare_dataset_for_training(dataset, tokenizer, max_length=128):
    """
    Prepare dataset for finetuning.
    Format: "Q: a + b = answer"
    """
    def format_example(example):
        # The dataset should have 'input' and 'output' or similar fields
        # Adapt based on actual dataset structure
        if 'equation' in example and 'result' in example:
            text = f"Q: {example['equation']} = {example['result']}"
        elif 'input' in example and 'output' in example:
            text = f"Q: {example['input']} = {example['output']}"
        elif 'question' in example and 'answer' in example:
            text = f"Q: {example['question']} = {example['answer']}"
        else:
            # Fallback: just use the first two fields
            keys = list(example.keys())
            text = f"Q: {example[keys[0]]} = {example[keys[1]]}"
        
        return {"text": text}
    
    # Format examples
    formatted_dataset = dataset.map(format_example, remove_columns=dataset.column_names)
    
    # Tokenize
    def tokenize_function(example):
        outputs = tokenizer(
            example["text"],
            max_length=max_length,
            truncation=True,
            padding="max_length",
        )
        outputs["labels"] = outputs["input_ids"]
        return outputs
    
    tokenized_dataset = formatted_dataset.map(
        tokenize_function,
        remove_columns=["text"]
    )
    
    return tokenized_dataset


def finetune_model(
    model_name,
    train_dataset,
    val_dataset,
    output_dir,
    epochs=3,
    batch_size=8,
    learning_rate=5e-5,
):
    """Finetune a model on the addition dataset."""
    print(f"\n{'='*80}")
    print(f"FINETUNING: {model_name}")
    print(f"{'='*80}\n")
    
    # Load model and tokenizer
    hf_token = os.getenv("HF_TOKEN")
    
    tokenizer = AutoTokenizer.from_pretrained(model_name, token=hf_token)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        token=hf_token
    )
    
    print(f"✅ Model loaded: {sum(p.numel() for p in model.parameters()) / 1e9:.2f}B parameters\n")
    
    # Training arguments (no model saving to disk!)
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        learning_rate=learning_rate,
        weight_decay=0.01,
        warmup_steps=50,
        logging_steps=50,
        eval_strategy="steps",
        eval_steps=100,
        save_strategy="no",  # Don't save checkpoints to disk!
        bf16=True,
        report_to="none",
    )
    
    # Data collator
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,
    )
    
    # Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=data_collator,
    )
    
    # Train
    print("Starting training...")
    trainer.train()
    
    print(f"\n✅ Finetuning complete! Model kept in memory (not saved to disk)\n")
    
    # Return the model and tokenizer directly (no disk saving!)
    return model, tokenizer


def extract_number(text):
    """Extract the answer number from generated text (first number before any '=' or at start)."""
    text = text.replace(',', '').strip()
    
    # Strategy: The answer is usually the FIRST number in the generation
    # Example: "1470 =  = 1533 = 1513" → answer is 1470
    # Example: "579" → answer is 579
    # Example: "the answer is 579" → answer is 579
    
    # Pattern 1: Number at the very beginning (before any "=")
    # This handles: "1470 =  = 1533" → extracts 1470
    match = re.search(r'^[^=]*?(-?\d+(?:\.\d+)?)', text)
    if match:
        return match.group(1)
    
    # Pattern 2: If generation starts with non-numeric text, find first number
    # This handles: "the answer is 579" → extracts 579
    match = re.search(r'(-?\d+(?:\.\d+)?)', text)
    if match:
        return match.group(1)
    
    return None


def evaluate_model(model, tokenizer, model_name, test_dataset, output_dir, max_new_tokens=20, max_eval_samples=100):
    """Evaluate a model on the test set."""
    print(f"\n{'='*80}")
    print(f"EVALUATING: {model_name}")
    print(f"{'='*80}\n")
    
    # Model is already loaded and finetuned - just set to eval mode
    model.eval()
    
    print(f"✅ Model ready: {sum(p.numel() for p in model.parameters()) / 1e9:.2f}B parameters\n")
    
    # Limit evaluation samples
    eval_dataset = test_dataset.select(range(min(max_eval_samples, len(test_dataset))))
    
    # Evaluate
    results = []
    correct = 0
    total = 0
    
    print(f"Evaluating on {len(eval_dataset)} examples (limited for speed)...\n")
    
    # Track examples for logging
    example_generations = []
    
    for idx, example in enumerate(tqdm(eval_dataset, desc="Evaluating")):
        # Extract question and answer based on dataset structure
        if 'equation' in example and 'result' in example:
            question = example['equation']
            ground_truth = str(example['result'])
        elif 'input' in example and 'output' in example:
            question = example['input']
            ground_truth = str(example['output'])
        elif 'question' in example and 'answer' in example:
            question = example['question']
            ground_truth = str(example['answer'])
        else:
            keys = list(example.keys())
            question = example[keys[0]]
            ground_truth = str(example[keys[1]])
        
        # Create prompt
        prompt = f"Q: {question} = "
        
        # Generate
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=1.0,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )
        
        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        generated_answer_text = generated_text[len(prompt):].strip()
        predicted_answer = extract_number(generated_answer_text)
        
        # Check if correct (compare as floats for decimal numbers)
        if predicted_answer is None:
            is_correct = False
        else:
            try:
                # Try to compare as floats (handles decimals properly)
                pred_float = float(predicted_answer.strip())
                truth_float = float(ground_truth.strip())
                # Use small tolerance for floating point comparison
                is_correct = abs(pred_float - truth_float) < 1e-6
            except (ValueError, TypeError):
                # Fallback to string comparison if conversion fails
                pred_normalized = predicted_answer.strip().lstrip('0') or '0'
                truth_normalized = ground_truth.strip().lstrip('0') or '0'
                is_correct = (pred_normalized == truth_normalized)
        
        if is_correct:
            correct += 1
        total += 1
        
        results.append({
            'idx': idx,
            'question': question,
            'ground_truth': ground_truth,
            'predicted': predicted_answer,
            'generated_text': generated_answer_text,
            'correct': is_correct,
        })
        
        # Save first 5 examples for logging
        if idx < 5:
            example_generations.append({
                'idx': idx,
                'input': f"Q: {question} = ",
                'expected': ground_truth,
                'generated': generated_answer_text,
                'predicted': predicted_answer,
                'correct': is_correct,
            })
        
        if (idx + 1) % 50 == 0:
            current_acc = correct / total * 100
            print(f"Progress: {idx + 1}/{len(eval_dataset)} | Current Accuracy: {current_acc:.2f}%")
    
    accuracy = correct / total * 100 if total > 0 else 0
    
    # Save results
    os.makedirs(output_dir, exist_ok=True)
    
    results_data = {
        'model': model_name,
        'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        'dataset': 'Onlydrinkwater/1000addition',
        'eval_samples': max_eval_samples,
        'accuracy': accuracy,
        'correct': correct,
        'total': total,
        'results': results,
    }
    
    results_file = Path(output_dir) / "results.json"
    with open(results_file, 'w') as f:
        json.dump(results_data, f, indent=2)
    
    # Save summary
    summary_file = Path(output_dir) / "summary.txt"
    with open(summary_file, 'w') as f:
        f.write(f"Addition Dataset Evaluation\n")
        f.write(f"=" * 80 + "\n")
        f.write(f"Model: {model_name}\n")
        f.write(f"Dataset: Onlydrinkwater/1000addition\n")
        f.write(f"Eval Samples: {max_eval_samples} (subset for speed)\n")
        f.write(f"Accuracy: {accuracy:.2f}%\n")
        f.write(f"Correct: {correct}/{total}\n")
    
    print(f"\n{'='*80}")
    print(f"EVALUATION COMPLETE")
    print(f"{'='*80}")
    print(f"Accuracy: {accuracy:.2f}% ({correct}/{total})")
    print(f"{'='*80}\n")
    
    # Show example generations
    print("\n" + "🔍" * 40)
    print("EXAMPLE GENERATIONS (First 5)")
    print("🔍" * 40 + "\n")
    
    for ex in example_generations:
        status = "✅ CORRECT" if ex['correct'] else "❌ WRONG"
        print(f"Example {ex['idx']+1} {status}:")
        print(f"  Input:      {ex['input']}")
        print(f"  Expected:   {ex['expected']}")
        print(f"  Generated:  {ex['generated'][:100]}{'...' if len(ex['generated']) > 100 else ''}")
        print(f"  Extracted:  {ex['predicted']}")
        print(f"  Match:      {ex['predicted']} {'==' if ex['correct'] else '!='} {ex['expected']}")
        print()
    
    print("=" * 80 + "\n")
    
    return results_data


def compare_results(baseline_results, fone_results, output_file):
    """Generate comparison report."""
    baseline_acc = baseline_results['accuracy']
    fone_acc = fone_results['accuracy']
    diff = fone_acc - baseline_acc
    
    report = []
    report.append("=" * 80)
    report.append("ADDITION DATASET COMPARISON")
    report.append("=" * 80)
    report.append(f"Dataset: Onlydrinkwater/1000addition")
    report.append(f"Baseline Model: {baseline_results['model']}")
    report.append(f"FoNE Model: {fone_results['model']}")
    report.append("=" * 80)
    report.append("")
    report.append(f"{'Metric':<25} {'Baseline':>12} {'FoNE':>12} {'Δ':>12} {'Winner':<12}")
    report.append("-" * 80)
    
    winner = "FoNE ✅" if diff > 0 else "Baseline ✅" if diff < 0 else "Tie"
    report.append(
        f"{'Accuracy':<25} {baseline_acc:>11.2f}% {fone_acc:>11.2f}% "
        f"{diff:>+11.2f}% {winner:<12}"
    )
    
    report.append("=" * 80)
    report.append("")
    report.append("SUMMARY:")
    report.append(f"  Baseline: {baseline_results['correct']}/{baseline_results['total']} correct ({baseline_acc:.2f}%)")
    report.append(f"  FoNE:     {fone_results['correct']}/{fone_results['total']} correct ({fone_acc:.2f}%)")
    report.append(f"  Improvement: {diff:+.2f}%")
    report.append("")
    
    if diff > 5:
        report.append("  🎉 FoNE performs significantly better!")
    elif diff > 1:
        report.append("  ✅ FoNE performs better")
    elif diff < -5:
        report.append("  ⚠️  Baseline performs significantly better")
    elif diff < -1:
        report.append("  ⚠️  Baseline performs slightly better")
    else:
        report.append("  ➖ Models perform similarly")
    
    report.append("")
    report.append("=" * 80)
    
    report_text = "\n".join(report)
    
    with open(output_file, 'w') as f:
        f.write(report_text)
    
    return report_text


def main():
    parser = argparse.ArgumentParser(
        description="Finetune and evaluate models on Onlydrinkwater/1000addition"
    )
    
    parser.add_argument(
        "--baseline_model",
        type=str,
        default="Onlydrinkwater/baseline-1p5b-step-76294",
        help="Baseline model to finetune"
    )
    
    parser.add_argument(
        "--fone_model",
        type=str,
        default="Onlydrinkwater/fone-1p5b-step-76294",
        help="FoNE model to finetune"
    )
    
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Output directory for all results"
    )
    
    parser.add_argument(
        "--train_samples",
        type=int,
        default=10000,
        help="Number of training samples to use"
    )
    
    parser.add_argument(
        "--epochs",
        type=int,
        default=3,
        help="Number of training epochs"
    )
    
    parser.add_argument(
        "--batch_size",
        type=int,
        default=8,
        help="Training batch size"
    )
    
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=5e-5,
        help="Learning rate"
    )
    
    parser.add_argument(
        "--skip_baseline",
        action="store_true",
        help="Skip baseline training/eval (reuse existing)"
    )
    
    parser.add_argument(
        "--skip_fone",
        action="store_true",
        help="Skip FoNE training/eval (reuse existing)"
    )
    
    args = parser.parse_args()
    
    # Setup output directory
    if args.output_dir is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.output_dir = f"/home/nvidia/fone-pretrain/eval/results/addition_pipeline_{timestamp}"
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    print("\n" + "=" * 80)
    print("ADDITION DATASET PIPELINE")
    print("=" * 80)
    print(f"Dataset: Onlydrinkwater/1000addition")
    print(f"Baseline: {args.baseline_model}")
    print(f"FoNE: {args.fone_model}")
    print(f"Train samples: {args.train_samples}")
    print(f"Epochs: {args.epochs}")
    print(f"Output: {args.output_dir}")
    print("=" * 80 + "\n")
    
    # Load datasets with clear logging
    print("\n" + "📊" * 40)
    print("DATASET INFORMATION")
    print("📊" * 40 + "\n")
    
    train_dataset_raw = load_addition_dataset("train", args.train_samples)
    print(f"  → Train: {len(train_dataset_raw)} examples")
    
    val_dataset_raw = load_addition_dataset("validation", 200)
    print(f"  → Validation: {len(val_dataset_raw)} examples")
    
    test_dataset_raw = load_addition_dataset("test")
    print(f"  → Test (full): {len(test_dataset_raw)} examples")
    print(f"  → Test (eval): 100 examples (subset for speed)")
    print()
    
    # Show example data format
    print("📝 DATA FORMAT EXAMPLES:")
    print("-" * 80)
    for i in range(min(3, len(train_dataset_raw))):
        example = train_dataset_raw[i]
        print(f"Example {i+1}:")
        print(f"  Raw data: {example}")
        # Show what it looks like when formatted
        if 'equation' in example and 'result' in example:
            formatted = f"Q: {example['equation']} = {example['result']}"
        elif 'input' in example and 'output' in example:
            formatted = f"Q: {example['input']} = {example['output']}"
        elif 'question' in example and 'answer' in example:
            formatted = f"Q: {example['question']} = {example['answer']}"
        else:
            keys = list(example.keys())
            formatted = f"Q: {example[keys[0]]} = {example[keys[1]]}"
        print(f"  Formatted: {formatted}")
        print()
    print("-" * 80)
    print()
    
    # Process baseline
    baseline_results = None
    if not args.skip_baseline:
        print("\n" + "🔷" * 40)
        print("BASELINE MODEL PIPELINE")
        print("🔷" * 40 + "\n")
        
        # Load tokenizer for dataset preparation
        baseline_tokenizer = AutoTokenizer.from_pretrained(
            args.baseline_model, 
            token=os.getenv("HF_TOKEN")
        )
        if baseline_tokenizer.pad_token is None:
            baseline_tokenizer.pad_token = baseline_tokenizer.eos_token
        
        # Prepare datasets
        print("Preparing training data for baseline...")
        train_dataset = prepare_dataset_for_training(
            train_dataset_raw, baseline_tokenizer
        )
        val_dataset = prepare_dataset_for_training(
            val_dataset_raw, baseline_tokenizer
        )
        
        # Finetune (returns model in memory, no disk save!)
        baseline_ft_dir = f"{args.output_dir}/baseline_training_logs"
        baseline_model, baseline_tokenizer = finetune_model(
            args.baseline_model,
            train_dataset,
            val_dataset,
            baseline_ft_dir,
            args.epochs,
            args.batch_size,
            args.learning_rate,
        )
        
        # Evaluate (model in memory, 100 examples)
        baseline_eval_dir = f"{args.output_dir}/baseline_eval"
        baseline_results = evaluate_model(
            baseline_model,
            baseline_tokenizer,
            args.baseline_model,
            test_dataset_raw,
            baseline_eval_dir,
            max_eval_samples=100,
        )
    
    # Process FoNE
    fone_results = None
    if not args.skip_fone:
        print("\n" + "🟢" * 40)
        print("FONE MODEL PIPELINE")
        print("🟢" * 40 + "\n")
        
        # Load tokenizer for dataset preparation
        fone_tokenizer = AutoTokenizer.from_pretrained(
            args.fone_model, 
            token=os.getenv("HF_TOKEN")
        )
        if fone_tokenizer.pad_token is None:
            fone_tokenizer.pad_token = fone_tokenizer.eos_token
        
        # Prepare datasets
        print("Preparing training data for FoNE...")
        train_dataset = prepare_dataset_for_training(
            train_dataset_raw, fone_tokenizer
        )
        val_dataset = prepare_dataset_for_training(
            val_dataset_raw, fone_tokenizer
        )
        
        # Finetune (returns model in memory, no disk save!)
        fone_ft_dir = f"{args.output_dir}/fone_training_logs"
        fone_model, fone_tokenizer = finetune_model(
            args.fone_model,
            train_dataset,
            val_dataset,
            fone_ft_dir,
            args.epochs,
            args.batch_size,
            args.learning_rate,
        )
        
        # Evaluate (model in memory, 100 examples)
        fone_eval_dir = f"{args.output_dir}/fone_eval"
        fone_results = evaluate_model(
            fone_model,
            fone_tokenizer,
            args.fone_model,
            test_dataset_raw,
            fone_eval_dir,
            max_eval_samples=100,
        )
    
    # Compare results
    if baseline_results and fone_results:
        print("\n" + "📊" * 40)
        print("COMPARISON REPORT")
        print("📊" * 40 + "\n")
        
        comparison_file = f"{args.output_dir}/comparison_report.txt"
        report = compare_results(baseline_results, fone_results, comparison_file)
        print(report)
        print(f"\nComparison saved to: {comparison_file}")
    
    print("\n" + "=" * 80)
    print(f"✅ PIPELINE COMPLETE!")
    print(f"All results saved to: {args.output_dir}")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()

