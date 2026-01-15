# GSM8K Evaluation

Evaluate FoNE and Baseline models on GSM8K using few-shot Chain-of-Thought prompting.

## Quick Start

### Compare MegaMath Continue-Pretrained Models

Compare the FoNE and Baseline models after continue pretraining on MegaMath:

```bash
cd /home/nvidia/fone-pretrain

# Full evaluation (all 1319 samples)
bash eval/compare_megamath_models.sh

# Quick test (100 samples)
bash eval/compare_megamath_models.sh 100
```

This will evaluate:
- **Baseline**: `Onlydrinkwater/baseline-1p5b-megamath-step-57221`
- **FoNE**: `Onlydrinkwater/fone-1p5b-megamath-step-57221`

## Models Being Compared

| Model | Description |
|-------|-------------|
| `baseline-1p5b-megamath-step-57221` | Baseline 1.5B model continued on 15B MegaMath tokens |
| `fone-1p5b-megamath-step-57221` | FoNE 1.5B model continued on 15B MegaMath tokens (without frozen embeddings) |

## Evaluation Protocol

### Few-shot Chain-of-Thought (CoT)

The standard GSM8K evaluation uses **8-shot CoT prompting**:

1. **Prompt Structure**:
   - 8 exemplar questions with step-by-step reasoning
   - Test question
   - Prompt: "Let's think step by step."

2. **Generation**:
   - Model generates reasoning trace
   - Model produces final numeric answer

3. **Scoring**:
   - Extract final numeric answer from generation
   - Compare with gold answer
   - Accuracy = correct answers / total questions

### Answer Extraction

The script extracts answers using multiple patterns:
- `"The answer is X"`
- `"#### X"` (GSM8K format)
- Last number in generated text (fallback)

## Individual Model Evaluation

Evaluate a single model:

```bash
python eval/eval_gsm8k.py --model MODEL_NAME
```

Options:
- `--model`: Model name or path (required)
- `--num_samples`: Number of samples (default: all 1319)
- `--max_new_tokens`: Max tokens to generate (default: 256)
- `--temperature`: Sampling temperature (default: 0.0 = greedy)
- `--zero_shot`: Use zero-shot instead of few-shot CoT
- `--output_dir`: Custom output directory

Examples:

```bash
# Full evaluation
python eval/eval_gsm8k.py --model Onlydrinkwater/fone-1p5b-megamath-step-57221

# Quick test on 50 samples
python eval/eval_gsm8k.py --model Onlydrinkwater/fone-1p5b-megamath-step-57221 --num_samples 50

# Zero-shot evaluation
python eval/eval_gsm8k.py --model Onlydrinkwater/fone-1p5b-megamath-step-57221 --zero_shot
```

## Output Structure

Results are saved to `eval/results/gsm8k_[shot]_[model]_[timestamp]/`:

```
gsm8k_fewshot_MODEL_20251214_120000/
├── results.json          # Full results with all predictions
├── summary.txt           # Brief accuracy summary
└── sample_outputs.txt    # First 10 examples for inspection
```

### results.json

```json
{
  "metrics": {
    "model": "MODEL_NAME",
    "total_samples": 1319,
    "correct": 450,
    "incorrect": 869,
    "accuracy": 0.341,
    "use_few_shot": true,
    "max_new_tokens": 256,
    "temperature": 0.0
  },
  "results": [
    {
      "question": "...",
      "gold_answer": "...",
      "gold_numeric": "42",
      "generated_text": "...",
      "predicted_answer": "42",
      "predicted_numeric": "42",
      "correct": true
    },
    ...
  ]
}
```

### summary.txt

```
Model: Onlydrinkwater/fone-1p5b-megamath-step-57221
Total samples: 1319
Correct: 450
Incorrect: 869
Accuracy: 34.14%
```

## Dataset

- **Dataset**: GSM8K test set (1319 grade-school math problems)
- **Source**: HuggingFace `gsm8k` dataset
- **Task**: Multi-step arithmetic reasoning

## Expected Performance

Reference benchmarks for GSM8K (8-shot CoT):

| Model Size | Typical Accuracy |
|------------|------------------|
| 1.3B - 1.5B | 15-25% |
| 7B | 30-50% |
| 13B | 45-60% |
| 70B+ | 60-80% |

Note: Models trained specifically on math data (like MegaMath) typically perform better than general pretrained models.

## Tips

### Quick Testing

Start with a small sample to test setup:

```bash
bash eval/compare_megamath_models.sh 10
```

### Memory Usage

- Models are loaded in bfloat16
- Requires ~4GB GPU memory per model
- Only one model loaded at a time

### Generation Length

GSM8K solutions typically require 100-200 tokens. The default 256 tokens should be sufficient. If answers are being cut off, increase with `--max_new_tokens`.

