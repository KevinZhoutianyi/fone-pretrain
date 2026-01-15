#!/bin/bash
#
# Complete arithmetic training and evaluation pipeline
# 1. Finetune baseline on arithmetic
# 2. Finetune FoNE on arithmetic  
# 3. Evaluate both
# 4. Compare results
#

set -e

# Colors
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

echo -e "${BLUE}"
echo "╔════════════════════════════════════════════════════════════════╗"
echo "║        Arithmetic Training & Evaluation Pipeline              ║"
echo "║    Train both models on arithmetic, then test which is better  ║"
echo "╚════════════════════════════════════════════════════════════════╝"
echo -e "${NC}"

# Default values
BASELINE="Onlydrinkwater/baseline-1p5b-step-76294"
FONE="Onlydrinkwater/fone-1p5b-step-76294"
NUM_SAMPLES=1000
EPOCHS=3
BATCH_SIZE=8
EVAL_SAMPLES=100

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --baseline) BASELINE="$2"; shift 2 ;;
        --fone) FONE="$2"; shift 2 ;;
        --num-samples) NUM_SAMPLES="$2"; shift 2 ;;
        --epochs) EPOCHS="$2"; shift 2 ;;
        --batch-size) BATCH_SIZE="$2"; shift 2 ;;
        --eval-samples) EVAL_SAMPLES="$2"; shift 2 ;;
        --help|-h)
            echo -e "${YELLOW}Usage:${NC} $0 [options]"
            echo ""
            echo -e "${YELLOW}Options:${NC}"
            echo "  --baseline MODEL       Baseline model (default: Onlydrinkwater/baseline-1p5b-step-76294)"
            echo "  --fone MODEL          FoNE model (default: Onlydrinkwater/fone-1p5b-step-76294)"
            echo "  --num-samples N       Training samples per operation (default: 1000)"
            echo "  --epochs N            Training epochs (default: 3)"
            echo "  --batch-size N        Batch size (default: 8)"
            echo "  --eval-samples N      Evaluation samples per operation (default: 100)"
            echo ""
            echo -e "${YELLOW}Example:${NC}"
            echo "  $0 --num-samples 500 --epochs 2"
            exit 0
            ;;
        *) echo -e "${RED}❌ Unknown option: $1${NC}"; exit 1 ;;
    esac
done

# Get directories
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

# Create output directory
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
OUTPUT_DIR="$PROJECT_DIR/eval/results/arithmetic_pipeline_${TIMESTAMP}"
mkdir -p "$OUTPUT_DIR"

echo -e "${GREEN}Configuration:${NC}"
echo "  Baseline:      $BASELINE"
echo "  FoNE:          $FONE"
echo "  Train samples: $NUM_SAMPLES per operation"
echo "  Epochs:        $EPOCHS"
echo "  Batch size:    $BATCH_SIZE"
echo "  Eval samples:  $EVAL_SAMPLES per operation"
echo "  Output:        $OUTPUT_DIR"
echo ""

# Step 1: Finetune baseline
echo -e "${BLUE}╔════════════════════════════════════════════════════════════════╗${NC}"
echo -e "${BLUE}║  STEP 1: Finetuning Baseline Model on Arithmetic              ║${NC}"
echo -e "${BLUE}╚════════════════════════════════════════════════════════════════╝${NC}"
echo ""

BASELINE_FT_DIR="$OUTPUT_DIR/baseline_finetuned"
python $PROJECT_DIR/finetune/finetune_arithmetic.py \
    --model "$BASELINE" \
    --output_dir "$BASELINE_FT_DIR" \
    --num_samples $NUM_SAMPLES \
    --epochs $EPOCHS \
    --batch_size $BATCH_SIZE

echo ""
echo -e "${GREEN}✅ Baseline finetuning complete!${NC}"
echo ""

# Step 2: Finetune FoNE
echo -e "${BLUE}╔════════════════════════════════════════════════════════════════╗${NC}"
echo -e "${BLUE}║  STEP 2: Finetuning FoNE Model on Arithmetic                  ║${NC}"
echo -e "${BLUE}╚════════════════════════════════════════════════════════════════╝${NC}"
echo ""

FONE_FT_DIR="$OUTPUT_DIR/fone_finetuned"
python $PROJECT_DIR/finetune/finetune_arithmetic.py \
    --model "$FONE" \
    --output_dir "$FONE_FT_DIR" \
    --num_samples $NUM_SAMPLES \
    --epochs $EPOCHS \
    --batch_size $BATCH_SIZE

echo ""
echo -e "${GREEN}✅ FoNE finetuning complete!${NC}"
echo ""

# Step 3: Evaluate baseline
echo -e "${BLUE}╔════════════════════════════════════════════════════════════════╗${NC}"
echo -e "${BLUE}║  STEP 3: Evaluating Finetuned Baseline                        ║${NC}"
echo -e "${BLUE}╚════════════════════════════════════════════════════════════════╝${NC}"
echo ""

BASELINE_EVAL_DIR="$OUTPUT_DIR/baseline_eval"
python $PROJECT_DIR/eval/eval_arithmetic.py \
    --model "$BASELINE_FT_DIR/final_model" \
    --output_dir "$BASELINE_EVAL_DIR" \
    --num_samples $EVAL_SAMPLES \
    --seed 123

echo ""
echo -e "${GREEN}✅ Baseline evaluation complete!${NC}"
echo ""

# Step 4: Evaluate FoNE
echo -e "${BLUE}╔════════════════════════════════════════════════════════════════╗${NC}"
echo -e "${BLUE}║  STEP 4: Evaluating Finetuned FoNE                            ║${NC}"
echo -e "${BLUE}╚════════════════════════════════════════════════════════════════╝${NC}"
echo ""

FONE_EVAL_DIR="$OUTPUT_DIR/fone_eval"
python $PROJECT_DIR/eval/eval_arithmetic.py \
    --model "$FONE_FT_DIR/final_model" \
    --output_dir "$FONE_EVAL_DIR" \
    --num_samples $EVAL_SAMPLES \
    --seed 123

echo ""
echo -e "${GREEN}✅ FoNE evaluation complete!${NC}"
echo ""

# Step 5: Generate comparison
echo -e "${BLUE}╔════════════════════════════════════════════════════════════════╗${NC}"
echo -e "${BLUE}║  STEP 5: Generating Comparison Report                         ║${NC}"
echo -e "${BLUE}╚════════════════════════════════════════════════════════════════╝${NC}"
echo ""

# Create comparison script
python - <<EOF
import json
from pathlib import Path

baseline_file = Path("$BASELINE_EVAL_DIR/results.json")
fone_file = Path("$FONE_EVAL_DIR/results.json")

with open(baseline_file) as f:
    baseline = json.load(f)

with open(fone_file) as f:
    fone = json.load(f)

# Print comparison
print("=" * 80)
print("ARITHMETIC FINETUNING RESULTS COMPARISON")
print("=" * 80)
print(f"Baseline: {baseline['model']}")
print(f"FoNE:     {fone['model']}")
print("=" * 80)
print()

baseline_acc = baseline['overall_accuracy']
fone_acc = fone['overall_accuracy']
diff = fone_acc - baseline_acc

print(f"{'Metric':<25} {'Baseline':>12} {'FoNE':>12} {'Δ':>12} {'Winner':<12}")
print("-" * 80)
print(f"{'Overall Accuracy':<25} {baseline_acc:>11.2f}% {fone_acc:>11.2f}% {diff:>+11.2f}% {'FoNE ✅' if diff > 0 else 'Baseline ✅' if diff < 0 else 'Tie':<12}")
print()

print("By Operation:")
print("-" * 80)

operations = sorted(set(baseline['accuracy_by_operation'].keys()) | set(fone['accuracy_by_operation'].keys()))

wins = {"baseline": 0, "fone": 0, "tie": 0}

for op in operations:
    b_acc = baseline['accuracy_by_operation'].get(op, 0)
    f_acc = fone['accuracy_by_operation'].get(op, 0)
    diff_op = f_acc - b_acc
    
    if diff_op > 1:
        winner = "FoNE ✅"
        wins["fone"] += 1
    elif diff_op < -1:
        winner = "Baseline ✅"
        wins["baseline"] += 1
    else:
        winner = "Tie"
        wins["tie"] += 1
    
    print(f"  {op:<23} {b_acc:>11.2f}% {f_acc:>11.2f}% {diff_op:>+11.2f}% {winner:<12}")

print("=" * 80)
print()
print("SUMMARY:")
print(f"  Overall: FoNE is {diff:+.2f}% vs Baseline")
print(f"  FoNE wins:     {wins['fone']}/{len(operations)} operations")
print(f"  Baseline wins: {wins['baseline']}/{len(operations)} operations")
print(f"  Ties:          {wins['tie']}/{len(operations)} operations")
print()

if diff > 5:
    print("  🎉 FoNE performs significantly better on arithmetic!")
elif diff > 1:
    print("  ✅ FoNE performs better on arithmetic")
elif diff < -5:
    print("  ⚠️  Baseline performs significantly better on arithmetic")
elif diff < -1:
    print("  ⚠️  Baseline performs slightly better on arithmetic")
else:
    print("  ➖ Models perform similarly on arithmetic")

print()
print("=" * 80)

# Save comparison
comparison_file = Path("$OUTPUT_DIR/final_comparison.txt")
with open(comparison_file, 'w') as f:
    f.write(f"ARITHMETIC FINETUNING RESULTS COMPARISON\n")
    f.write(f"=" * 80 + "\n")
    f.write(f"Baseline: {baseline['model']}\n")
    f.write(f"FoNE:     {fone['model']}\n")
    f.write(f"Overall: FoNE {diff:+.2f}% vs Baseline\n")
    f.write(f"=" * 80 + "\n")

print(f"\n✅ Comparison saved to: {comparison_file}")
EOF

echo ""
echo -e "${GREEN}╔════════════════════════════════════════════════════════════════╗${NC}"
echo -e "${GREEN}║  ✅ COMPLETE! All results saved to:${NC}"
echo -e "${GREEN}║  ${OUTPUT_DIR}${NC}"
echo -e "${GREEN}╚════════════════════════════════════════════════════════════════╝${NC}"
echo ""

