#!/usr/bin/env python3
import os
import sys
import json
import argparse
import subprocess
from pathlib import Path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser("Consolidate ZeRO-3 checkpoint and run text-generation eval")
    p.add_argument("--ckpt", required=True, help="Path to step_XXXX checkpoint directory")
    p.add_argument("--config", required=True, help="Path to model config JSON used in pretraining")
    p.add_argument("--out_dir", default=None, help="Output directory for consolidated model (defaults to <ckpt>_consolidated)")
    p.add_argument("--prompts_file", default="configs/validation_prompts.json", help="Prompts JSON for generation eval")
    p.add_argument("--num_samples", type=int, default=1)
    p.add_argument("--max_new_tokens", type=int, default=200)
    p.add_argument("--temperature", type=float, default=0.7)
    p.add_argument("--device", default="auto")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    src_dir = repo_root / "src"
    eval_runs = repo_root / "eval_runs"
    eval_runs.mkdir(parents=True, exist_ok=True)

    ckpt = Path(args.ckpt).resolve()
    if not ckpt.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt}")

    model_cfg = Path(args.config).resolve()
    if not model_cfg.exists():
        raise FileNotFoundError(f"Model config not found: {model_cfg}")

    # Consolidated output folder
    out_dir = Path(args.out_dir) if args.out_dir else Path(str(ckpt) + "_consolidated")
    out_dir = out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    # Build a small python snippet to consolidate using Accelerate API and setup_llama_with_fone
    consolidate_py = f"""
import os, json, torch
from accelerate import Accelerator
from pathlib import Path
import sys
sys.path.append(str(Path('{src_dir}')))
from modeling.llama_1p5b import setup_llama_with_fone

CKPT = r"{ckpt}"
CFG  = r"{model_cfg}"
OUT  = r"{out_dir}"

acc = Accelerator()
model, tokenizer, fone_info = setup_llama_with_fone(
    config_path=CFG,
    tokenizer_name="meta-llama/Llama-3.1-8B",
    flash_attention=False,
    fone_hi=999,
    freeze_fone_rows=True,
    torch_dtype=torch.bfloat16,
)
model = acc.prepare(model)
acc.load_state(CKPT)
state_dict = acc.get_state_dict(model)
unwrapped = acc.unwrap_model(model)

os.makedirs(OUT, exist_ok=True)
unwrapped.save_pretrained(OUT, state_dict=state_dict)
tokenizer.save_pretrained(OUT)
with open(os.path.join(OUT, "fone_config.json"), "w") as f:
    json.dump({{"fone_hi": 999, "freeze_fone_rows": True}}, f, indent=2)
print("Consolidated to:", OUT)
"""

    # Run consolidation
    env = os.environ.copy()
    env["PYTHONPATH"] = f"{src_dir}:{env.get('PYTHONPATH','')}"
    subprocess.run([sys.executable, "-c", consolidate_py], check=True, env=env)

    # Prepare eval output paths
    step_name = ckpt.name
    run_dir = eval_runs / step_name
    run_dir.mkdir(parents=True, exist_ok=True)
    gen_json = run_dir / f"text_gen_eval_{step_name}.json"

    # Run text-generation eval on consolidated model
    cmd = [
        sys.executable, str(src_dir / "eval" / "text_generation_eval.py"),
        "--model", str(out_dir),
        "--prompts_file", str(Path(args.prompts_file).resolve()),
        "--num_samples", str(args.num_samples),
        "--max_new_tokens", str(args.max_new_tokens),
        "--temperature", str(args.temperature),
        "--output_file", str(gen_json),
    ]
    if args.device:
        cmd.extend(["--device", args.device])

    subprocess.run(cmd, check=True, cwd=repo_root)

    # Write a small run manifest
    manifest = {
        "checkpoint": str(ckpt),
        "consolidated_model": str(out_dir),
        "eval_json": str(gen_json),
        "prompts_file": str(Path(args.prompts_file).resolve()),
        "num_samples": args.num_samples,
        "max_new_tokens": args.max_new_tokens,
        "temperature": args.temperature,
    }
    with open(run_dir / "manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"\nSaved eval to: {gen_json}")
    print(f"Run folder:    {run_dir}")


if __name__ == "__main__":
    main()


