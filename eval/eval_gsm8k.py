"""
GSM8K Evaluation with Few-shot Chain-of-Thought Prompting

Evaluates language models on GSM8K using the standard few-shot CoT evaluation protocol.
"""

import argparse
import json
import os
import re
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.generation.stopping_criteria import StoppingCriteria, StoppingCriteriaList


class NewQuestionStoppingCriteria(StoppingCriteria):
    """Stop generation when model starts generating a new question (Q:)"""
    
    def __init__(self, tokenizer, start_length: int):
        self.tokenizer = tokenizer
        self.start_length = start_length
        # Pre-encode the stop sequence
        self.stop_sequence = tokenizer.encode("\nQ:", add_special_tokens=False)
        
    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        # Only check the newly generated tokens
        if input_ids.shape[1] <= self.start_length:
            return False
        
        # Get the generated tokens
        generated_ids = input_ids[0, self.start_length:]
        
        # Check if the stop sequence appears in generated tokens
        # Convert to list for easier searching
        generated_list = generated_ids.tolist()
        stop_list = self.stop_sequence
        
        # Check if stop_sequence appears as a subsequence
        for i in range(len(generated_list) - len(stop_list) + 1):
            if generated_list[i:i+len(stop_list)] == stop_list:
                return True
        
        return False


# Few-shot CoT exemplars (standard GSM8K examples)
FEW_SHOT_EXEMPLARS = [
    {
        "question": "There are 15 trees in the grove. Grove workers will plant trees in the grove today. After they are done, there will be 21 trees. How many trees did the grove workers plant today?",
        "reasoning": "There are 15 trees originally. Then there were 21 trees after some more were planted. So there must have been 21 - 15 = 6.",
        "answer": "6"
    },
    {
        "question": "If there are 3 cars in the parking lot and 2 more cars arrive, how many cars are in the parking lot?",
        "reasoning": "There are originally 3 cars. 2 more cars arrive. 3 + 2 = 5.",
        "answer": "5"
    },
    {
        "question": "Leah had 32 chocolates and her sister had 42. If they ate 35, how many pieces do they have left in total?",
        "reasoning": "Originally, Leah had 32 chocolates. Her sister had 42. So in total they had 32 + 42 = 74. After eating 35, they had 74 - 35 = 39.",
        "answer": "39"
    },
    {
        "question": "Jason had 20 lollipops. He gave Denny some lollipops. Now Jason has 12 lollipops. How many lollipops did Jason give to Denny?",
        "reasoning": "Jason started with 20 lollipops. Then he had 12 after giving some to Denny. So he gave Denny 20 - 12 = 8.",
        "answer": "8"
    },
    {
        "question": "Shawn has five toys. For Christmas, he got two toys each from his mom and dad. How many toys does he have now?",
        "reasoning": "Shawn started with 5 toys. If he got 2 toys each from his mom and dad, then that is 2 * 2 = 4 more toys. 5 + 4 = 9.",
        "answer": "9"
    },
    {
        "question": "There were nine computers in the server room. Five more computers were installed each day, from monday to thursday. How many computers are now in the server room?",
        "reasoning": "There were originally 9 computers. For each of 4 days, 5 more computers were added. So 5 * 4 = 20 computers were added. 9 + 20 = 29.",
        "answer": "29"
    },
    {
        "question": "Michael had 58 golf balls. On tuesday, he lost 23 golf balls. On wednesday, he lost 2 more. How many golf balls did he have at the end of wednesday?",
        "reasoning": "Michael started with 58 golf balls. After losing 23 on tuesday, he had 58 - 23 = 35. After losing 2 more, he had 35 - 2 = 33 golf balls.",
        "answer": "33"
    },
    {
        "question": "Olivia has $23. She bought five bagels for $3 each. How much money does she have left?",
        "reasoning": "Olivia had 23 dollars. 5 bagels for 3 dollars each will be 5 * 3 = 15 dollars. So she has 23 - 15 = 8 dollars left.",
        "answer": "8"
    }
]


def construct_prompt(question: str, use_few_shot: bool = True) -> str:
    """
    Construct few-shot CoT prompt for GSM8K.
    
    Args:
        question: The test question
        use_few_shot: Whether to include few-shot exemplars
        
    Returns:
        Formatted prompt string
    """
    if use_few_shot:
        prompt_parts = []
        
        # Add few-shot exemplars
        for exemplar in FEW_SHOT_EXEMPLARS:
            prompt_parts.append(f"Q: {exemplar['question']}")
            prompt_parts.append(f"A: Let's think step by step.")
            prompt_parts.append(exemplar['reasoning'])
            prompt_parts.append(f"#### {exemplar['answer']}")
            prompt_parts.append("")
        
        # Add clear separator to distinguish test question from exemplars
        prompt_parts.append("Now solve this problem:")
        prompt_parts.append("")
        
        # Add test question with #### prompt
        prompt_parts.append(f"Q: {question}")
        prompt_parts.append("A: Let's think step by step.")
        prompt_parts.append("####")  # Prompt model to output answer after ####
        
        return "\n".join(prompt_parts)
    else:
        # Zero-shot CoT
        return f"Q: {question}\nA: Let's think step by step.\n####"


def extract_answer(text: str) -> tuple[Optional[str], str]:
    """
    Extract the final numeric answer from model output.
    
    Priority:
    1. Last #### delimiter (GSM8K format)
    2. Last number in the text
    
    Args:
        text: Generated text from the model
        
    Returns:
        Tuple of (extracted answer, parsing context for debugging)
    """
    # Remove commas from numbers for parsing
    text_normalized = text.replace(",", "")
    
    # Pattern 1: GSM8K format "#### X" - take the LAST occurrence
    matches = list(re.finditer(r'####\s*(-?\d+\.?\d*)', text_normalized))
    if matches:
        last_match = matches[-1]
        answer = last_match.group(1).rstrip('.')  # Remove trailing period
        # Get context around the match for debugging
        start = max(0, last_match.start() - 30)
        end = min(len(text), last_match.end() + 30)
        context = text[start:end].replace('\n', ' ')
        return answer, f"#### match: ...{context}..."
    
    # Pattern 2: Last number in the text (fallback)
    # Use more precise regex to avoid matching parts of words
    numbers = re.findall(r'(?<!\w)(-?\d+\.?\d*)(?!\w)', text_normalized)
    if numbers:
        answer = numbers[-1].rstrip('.')
        # Find where this number appears
        number_pos = text.rfind(numbers[-1])
        start = max(0, number_pos - 30)
        end = min(len(text), number_pos + len(numbers[-1]) + 30)
        context = text[start:end].replace('\n', ' ')
        return answer, f"last number: ...{context}..."
    
    return None, "no number found"


def normalize_answer(answer: str) -> str:
    """Normalize answer for comparison."""
    if answer is None:
        return ""
    
    # Remove commas, extra spaces, and trailing periods
    answer = str(answer).replace(",", "").strip().rstrip('.')
    
    # Try to convert to float then back to remove trailing zeros
    try:
        num = float(answer)
        # If it's a whole number, return as int
        if num.is_integer():
            return str(int(num))
        return str(num)
    except (ValueError, AttributeError):
        return answer


def evaluate_model(
    model_name: str,
    num_samples: Optional[int] = None,
    max_new_tokens: int = 256,
    temperature: float = 0.0,
    use_few_shot: bool = True,
    device: str = "cuda",
    output_dir: Optional[str] = None
) -> Dict:
    """
    Evaluate a model on GSM8K.
    
    Args:
        model_name: HuggingFace model ID or local path
        num_samples: Number of samples to evaluate (None = all)
        max_new_tokens: Max tokens to generate
        temperature: Sampling temperature (0.0 = greedy)
        use_few_shot: Use few-shot CoT prompting
        device: Device to run on
        output_dir: Directory to save results
        
    Returns:
        Dictionary with evaluation metrics
    """
    print(f"Loading model: {model_name}")
    
    # Load model and tokenizer
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    
    # Important: Do NOT apply chat template for base models
    # Set padding side to left for generation
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    model.eval()
    
    print(f"Loading GSM8K dataset...")
    dataset = load_dataset("gsm8k", "main", split="test")
    
    if num_samples is not None:
        dataset = dataset.select(range(min(num_samples, len(dataset))))
    
    print(f"Evaluating on {len(dataset)} samples...")
    print(f"{'Few-shot' if use_few_shot else 'Zero-shot'} CoT prompting")
    print(f"Max tokens: {max_new_tokens}, Temperature: {temperature}")
    print()
    
    results = []
    correct = 0
    total = 0
    parsing_logs = []  # Store parsing context for first 10 samples
    
    for idx, example in enumerate(tqdm(dataset, desc="Evaluating")):
        question = example["question"]
        gold_answer = example["answer"]
        
        # Extract gold answer (GSM8K format: "#### answer")
        gold_match = re.search(r'####\s*(-?\d+\.?\d*)', gold_answer.replace(",", ""))
        if gold_match:
            gold_numeric = normalize_answer(gold_match.group(1))
        else:
            # Fallback: take last number
            numbers = re.findall(r'(-?\d+\.?\d*)', gold_answer.replace(",", ""))
            gold_numeric = normalize_answer(numbers[-1]) if numbers else ""
        
        # Construct prompt (NO chat template - plain text only)
        prompt = construct_prompt(question, use_few_shot=use_few_shot)
        
        # Tokenize
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        input_length = inputs.input_ids.shape[1]
        
        # Create stopping criteria to stop at "\nQ:"
        stopping_criteria = StoppingCriteriaList([
            NewQuestionStoppingCriteria(tokenizer, input_length)
        ])
        
        # Generate
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature if temperature > 0 else None,
                do_sample=temperature > 0,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
                stopping_criteria=stopping_criteria,
            )
        
        # Decode
        full_output = tokenizer.decode(outputs[0], skip_special_tokens=True)
        generated_text = full_output[len(prompt):].strip()
        
        # Extract predicted answer with context
        pred_answer, parse_context = extract_answer(generated_text)
        pred_numeric = normalize_answer(pred_answer)
        
        # Check correctness
        is_correct = (pred_numeric == gold_numeric) and (pred_numeric != "")
        
        if is_correct:
            correct += 1
        total += 1
        
        # Log parsing context for first 10 samples
        if idx < 10:
            parsing_logs.append({
                "idx": idx,
                "predicted": pred_numeric,
                "gold": gold_numeric,
                "parse_context": parse_context,
                "generated_preview": generated_text[:200] + "..." if len(generated_text) > 200 else generated_text
            })
        
        # Store result
        results.append({
            "question": question,
            "gold_answer": gold_answer,
            "gold_numeric": gold_numeric,
            "generated_text": generated_text,
            "predicted_answer": pred_answer,
            "predicted_numeric": pred_numeric,
            "correct": is_correct,
            "parse_context": parse_context
        })
    
    accuracy = correct / total if total > 0 else 0.0
    
    metrics = {
        "model": model_name,
        "total_samples": total,
        "correct": correct,
        "incorrect": total - correct,
        "accuracy": accuracy,
        "use_few_shot": use_few_shot,
        "max_new_tokens": max_new_tokens,
        "temperature": temperature
    }
    
    print()
    print("=" * 60)
    print("RESULTS")
    print("=" * 60)
    print(f"Model: {model_name}")
    print(f"Total samples: {total}")
    print(f"Correct: {correct}")
    print(f"Incorrect: {total - correct}")
    print(f"Accuracy: {accuracy:.2%}")
    print("=" * 60)
    
    # Print parsing logs for debugging
    if parsing_logs:
        print()
        print("=" * 60)
        print("PARSING DEBUG (first 10 samples)")
        print("=" * 60)
        for log in parsing_logs:
            print(f"\nSample {log['idx']}:")
            print(f"  Predicted: {log['predicted']}")
            print(f"  Gold: {log['gold']}")
            print(f"  Parse method: {log['parse_context']}")
            print(f"  Generated preview: {log['generated_preview'][:150]}...")
        print("=" * 60)
    
    # Save results
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        
        # Save detailed results
        with open(os.path.join(output_dir, "results.json"), "w") as f:
            json.dump({"metrics": metrics, "results": results, "parsing_logs": parsing_logs}, f, indent=2)
        
        # Save summary
        with open(os.path.join(output_dir, "summary.txt"), "w") as f:
            f.write(f"Model: {model_name}\n")
            f.write(f"Total samples: {total}\n")
            f.write(f"Correct: {correct}\n")
            f.write(f"Incorrect: {total - correct}\n")
            f.write(f"Accuracy: {accuracy:.2%}\n")
        
        # Save parsing debug logs
        with open(os.path.join(output_dir, "parsing_debug.txt"), "w") as f:
            f.write("PARSING DEBUG LOG (first 10 samples)\n")
            f.write("=" * 80 + "\n\n")
            for log in parsing_logs:
                f.write(f"Sample {log['idx']}:\n")
                f.write(f"  Predicted: {log['predicted']}\n")
                f.write(f"  Gold: {log['gold']}\n")
                f.write(f"  Match: {'✓' if log['predicted'] == log['gold'] else '✗'}\n")
                f.write(f"  Parse method: {log['parse_context']}\n")
                f.write(f"  Generated preview:\n    {log['generated_preview']}\n")
                f.write("\n" + "-" * 80 + "\n\n")
        
        # Save first 10 examples for inspection
        sample_results = results[:10]
        with open(os.path.join(output_dir, "sample_outputs.txt"), "w") as f:
            for i, result in enumerate(sample_results, 1):
                f.write(f"{'='*60}\n")
                f.write(f"Example {i}\n")
                f.write(f"{'='*60}\n")
                f.write(f"Question: {result['question']}\n\n")
                f.write(f"Generated: {result['generated_text']}\n\n")
                f.write(f"Predicted: {result['predicted_numeric']}\n")
                f.write(f"Gold: {result['gold_numeric']}\n")
                f.write(f"Correct: {result['correct']}\n\n")
        
        print(f"\nResults saved to: {output_dir}")
    
    return metrics


def main():
    parser = argparse.ArgumentParser(description="Evaluate model on GSM8K")
    parser.add_argument("--model", type=str, required=True,
                       help="Model name or path")
    parser.add_argument("--num_samples", type=int, default=None,
                       help="Number of samples to evaluate (default: all)")
    parser.add_argument("--max_new_tokens", type=int, default=256,
                       help="Max tokens to generate")
    parser.add_argument("--temperature", type=float, default=0.0,
                       help="Sampling temperature (0.0 = greedy)")
    parser.add_argument("--zero_shot", action="store_true",
                       help="Use zero-shot instead of few-shot")
    parser.add_argument("--output_dir", type=str, default=None,
                       help="Output directory (auto-generated if not specified)")
    
    args = parser.parse_args()
    
    # Auto-generate output directory
    if args.output_dir is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_name_safe = args.model.replace("/", "_").replace("\\", "_")
        shot_type = "zeroshot" if args.zero_shot else "fewshot"
        args.output_dir = f"eval/results/gsm8k_{shot_type}_{model_name_safe}_{timestamp}"
    
    # Run evaluation
    evaluate_model(
        model_name=args.model,
        num_samples=args.num_samples,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        use_few_shot=not args.zero_shot,
        output_dir=args.output_dir
    )


if __name__ == "__main__":
    main()

