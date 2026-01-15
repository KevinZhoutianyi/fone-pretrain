#!/usr/bin/env python3
"""
Prepare GSM8K dataset for finetuning with Chain-of-Thought
Formats data in instruction-following format
"""

import argparse
import json
from pathlib import Path
from datasets import load_dataset
from tqdm import tqdm


def format_gsm8k_cot(example):
    """
    Format GSM8K example with Chain-of-Thought
    
    Input format:
    - question: The math problem
    - answer: Step-by-step solution ending with #### <answer>
    
    Output format (instruction-following):
    - prompt: "Question: <question>\nAnswer: Let's think step by step."
    - completion: "<reasoning>\nThe answer is <answer>."
    """
    question = example['question']
    answer_text = example['answer']
    
    # Extract the reasoning and final answer
    # GSM8K format: "reasoning text\n#### final_answer"
    if '####' in answer_text:
        reasoning, final_answer = answer_text.split('####')
        reasoning = reasoning.strip()
        final_answer = final_answer.strip()
    else:
        # Fallback if format is different
        reasoning = answer_text
        final_answer = ""
    
    # Create prompt and completion
    prompt = f"Question: {question}\nAnswer: Let's think step by step."
    
    # Format completion with reasoning and clear final answer
    if final_answer:
        completion = f" {reasoning}\nThe answer is {final_answer}."
    else:
        completion = f" {reasoning}"
    
    return {
        'prompt': prompt,
        'completion': completion,
        'text': prompt + completion,  # Full text for causal LM training
    }


def prepare_dataset(output_dir, max_train_samples=None, max_test_samples=None):
    """
    Download and prepare GSM8K dataset
    
    Args:
        output_dir: Directory to save processed data
        max_train_samples: Maximum number of training samples (None for all)
        max_test_samples: Maximum number of test samples (None for all)
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    
    print("=" * 80)
    print("PREPARING GSM8K DATASET FOR FINETUNING")
    print("=" * 80)
    print()
    
    # Load dataset
    print("Loading GSM8K dataset from HuggingFace...")
    try:
        train_dataset = load_dataset("gsm8k", "main", split="train")
        test_dataset = load_dataset("gsm8k", "main", split="test")
    except:
        # Fallback to openai/gsm8k
        train_dataset = load_dataset("openai/gsm8k", "main", split="train")
        test_dataset = load_dataset("openai/gsm8k", "main", split="test")
    
    print(f"✅ Loaded {len(train_dataset)} training examples")
    print(f"✅ Loaded {len(test_dataset)} test examples")
    print()
    
    # Limit samples if specified
    if max_train_samples:
        train_dataset = train_dataset.select(range(min(max_train_samples, len(train_dataset))))
        print(f"Limited to {len(train_dataset)} training samples")
    
    if max_test_samples:
        test_dataset = test_dataset.select(range(min(max_test_samples, len(test_dataset))))
        print(f"Limited to {len(test_dataset)} test samples")
    
    print()
    
    # Process training data
    print("Processing training data...")
    train_formatted = []
    for example in tqdm(train_dataset, desc="Formatting train"):
        formatted = format_gsm8k_cot(example)
        train_formatted.append(formatted)
    
    # Process test data
    print("Processing test data...")
    test_formatted = []
    for example in tqdm(test_dataset, desc="Formatting test"):
        formatted = format_gsm8k_cot(example)
        test_formatted.append(formatted)
    
    # Save as JSONL files
    train_file = output_dir / "train.jsonl"
    test_file = output_dir / "test.jsonl"
    
    print(f"\nSaving formatted data...")
    with open(train_file, 'w') as f:
        for item in train_formatted:
            f.write(json.dumps(item) + '\n')
    
    with open(test_file, 'w') as f:
        for item in test_formatted:
            f.write(json.dumps(item) + '\n')
    
    # Save a few examples for inspection
    examples_file = output_dir / "examples.txt"
    with open(examples_file, 'w') as f:
        f.write("GSM8K CoT Training Examples\n")
        f.write("=" * 80 + "\n\n")
        
        for i, example in enumerate(train_formatted[:5], 1):
            f.write(f"Example {i}:\n")
            f.write("-" * 80 + "\n")
            f.write(f"FULL TEXT:\n{example['text']}\n")
            f.write("\n" + "=" * 80 + "\n\n")
    
    # Save dataset info
    info_file = output_dir / "dataset_info.json"
    with open(info_file, 'w') as f:
        json.dump({
            'dataset': 'gsm8k',
            'format': 'instruction_following_cot',
            'train_samples': len(train_formatted),
            'test_samples': len(test_formatted),
            'prompt_template': 'Question: {question}\\nAnswer: Let\'s think step by step.',
            'completion_template': ' {reasoning}\\nThe answer is {answer}.',
        }, f, indent=2)
    
    print()
    print("=" * 80)
    print("✅ Dataset preparation complete!")
    print("=" * 80)
    print(f"Output directory: {output_dir}")
    print(f"  - {train_file.name}: {len(train_formatted)} examples")
    print(f"  - {test_file.name}: {len(test_formatted)} examples")
    print(f"  - {examples_file.name}: Sample examples")
    print(f"  - {info_file.name}: Dataset info")
    print()


def main():
    parser = argparse.ArgumentParser(description="Prepare GSM8K dataset for finetuning")
    
    parser.add_argument(
        "--output_dir",
        type=str,
        default="finetune/data/gsm8k_cot",
        help="Output directory for processed data"
    )
    
    parser.add_argument(
        "--max_train_samples",
        type=int,
        default=None,
        help="Maximum number of training samples (default: all)"
    )
    
    parser.add_argument(
        "--max_test_samples",
        type=int,
        default=None,
        help="Maximum number of test samples (default: all)"
    )
    
    args = parser.parse_args()
    
    prepare_dataset(
        output_dir=args.output_dir,
        max_train_samples=args.max_train_samples,
        max_test_samples=args.max_test_samples,
    )


if __name__ == "__main__":
    main()

