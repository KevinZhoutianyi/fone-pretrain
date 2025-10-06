#!/usr/bin/env python3
import os
import sys
import json
import argparse
import subprocess
from datetime import datetime
from pathlib import Path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser("Consolidate ZeRO-3 checkpoint and run text-generation eval, logging to <eval_root>/<run>/<step>/run.log")
    p.add_argument("--ckpt", required=True, help="Path to step_XXXX checkpoint directory")
    p.add_argument("--config", required=True, help="Path to model config JSON used in pretraining")
    p.add_argument("--prompts_file", default="configs/validation_prompts.json", help="Prompts JSON for generation eval")
    p.add_argument("--num_samples", type=int, default=1)
    p.add_argument("--max_new_tokens", type=int, default=200)
    p.add_argument("--temperature", type=float, default=0.7)
    p.add_argument("--device", default="auto")
    p.add_argument(
        "--eval_root",
        default=None,
        help=(
            "Root directory to store evaluation logs/results. "
            "Defaults to environment variable FONE_EVAL_DIR if set, else <repo>/eval"
        ),
    )
    return p.parse_args()


def sh(cmd, cwd: Path, log_file: Path):
    with log_file.open("a") as lf:
        lf.write(f"\n$ {' '.join(cmd)}\n")
        lf.flush()
        subprocess.run(cmd, check=True, cwd=str(cwd), stdout=lf, stderr=lf)


def main() -> None:
    args = parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    src_dir = repo_root / "src"

    # Resolve output roots from environment
    project_dir = os.environ.get("PROJECT_DIR")
    work_hdd_dir = os.environ.get("WORK_HDD_DIR")
    project_outputs = Path(project_dir, "outputs") if project_dir else None
    work_outputs = Path(work_hdd_dir, "outputs") if work_hdd_dir else None

    ckpt = Path(args.ckpt).resolve()
    if not ckpt.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt}")

    model_cfg = Path(args.config).resolve()
    if not model_cfg.exists():
        raise FileNotFoundError(f"Model config not found: {model_cfg}")

    # Derive run name from checkpoint parent directory (e.g., pretrain_.../step_XXXXX)
    run_name = ckpt.parent.name
    step_name = ckpt.name  # e.g., step_150000

    # Evaluation logs/results root: CLI > env > repo eval/
    eval_root = args.eval_root or os.environ.get("FONE_EVAL_DIR") or (repo_root / "eval")
    eval_root = Path(eval_root)
    run_dir = eval_root / run_name / step_name
    run_dir.mkdir(parents=True, exist_ok=True)
    log_file = run_dir / "run.log"
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with log_file.open("a") as lf:
        lf.write(f"Run started: {ts}\n")
        lf.write(f"Checkpoint: {ckpt}\n")
        lf.write(f"Config:     {model_cfg}\n")

    # Consolidated model directory: prefer path anchored to the checkpoint's outputs root
    def find_outputs_root(p: Path) -> Path | None:
        parts = list(p.resolve().parts)
        if "outputs" in parts:
            idx = parts.index("outputs")
            return Path(*parts[: idx + 1])
        return None

    ckpt_outputs_root = find_outputs_root(ckpt)
    if ckpt_outputs_root and ckpt_outputs_root.exists():
        consolidated_base = ckpt_outputs_root / "consolidated"
    elif work_outputs and work_outputs.exists():
        consolidated_base = work_outputs / "consolidated"
    elif project_outputs and project_outputs.exists():
        consolidated_base = project_outputs / "consolidated"
    else:
        consolidated_base = run_dir / "consolidated"
    model_out_dir = consolidated_base / run_name / step_name / "model"
    model_out_dir.mkdir(parents=True, exist_ok=True)

    # Prefer Deepspeed zero_to_fp32 if available
    ds_script = ckpt / "zero_to_fp32.py"
    if ds_script.exists():
        # Ensure target file path inside model_out_dir
        out_bin = model_out_dir / "pytorch_model.bin"
        # Clean up if path exists as file or directory from previous runs
        if out_bin.exists():
            if out_bin.is_dir():
                import shutil
                shutil.rmtree(out_bin)
            else:
                out_bin.unlink()
        # Run DeepSpeed consolidation directly into model_out_dir
        sh([sys.executable, str(ds_script), str(ckpt), str(out_bin)], cwd=repo_root, log_file=log_file)
        # If file not created where expected, look for common fallbacks and any shard patterns
        if not out_bin.exists():
            patterns = [
                "pytorch_model.bin",
                "pytorch_model-*.bin",
                "pytorch_model.bin.index.json",
                "model.safetensors",
                "model-*.safetensors",
                "model.safetensors.index.json",
            ]
            search_dirs = [ckpt, run_dir, repo_root]
            import glob
            copied_any = False
            for d in search_dirs:
                for pat in patterns:
                    for src_path in glob.glob(str((d / pat))):
                        sp = Path(src_path)
                        if sp.is_file():
                            (model_out_dir / sp.name).write_bytes(sp.read_bytes())
                            copied_any = True
            if not copied_any:
                with log_file.open("a") as lf:
                    lf.write("WARNING: DeepSpeed did not produce recognizable weight files in expected locations.\n")
        # Copy minimal HF files into model dir
        for name in [
            "config.json", "tokenizer_config.json", "tokenizer.json",
            "special_tokens_map.json", "generation_config.json", "fone_config.json",
        ]:
            src = ckpt / name
            if src.exists():
                data = src.read_bytes()
                (model_out_dir / name).write_bytes(data)

        # If still no weights in model_out_dir, fall back to Accelerate gather to produce them here
        if not any((model_out_dir / fn).exists() for fn in [
            "pytorch_model.bin",
            "pytorch_model.bin.index.json",
            "pytorch_model-00001-of-00001.bin",
            "model.safetensors",
            "model.safetensors.index.json",
            "model-00001-of-00001.safetensors",
        ]):
            with log_file.open("a") as lf:
                lf.write("Falling back to Accelerate consolidation to create weights in model_out_dir...\n")
        consolidate_py = f"""
import os, json, torch
from accelerate import Accelerator
from pathlib import Path
import sys
sys.path.append(str(Path('{src_dir}')))
from modeling.llama_1p5b import setup_llama_with_fone

CKPT = r"{ckpt}"
CFG  = r"{model_cfg}"
OUT  = r"{model_out_dir}"

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
        tmp_script = run_dir / "consolidate_tmp.py"
        tmp_script.write_text(consolidate_py)
        sh([sys.executable, str(tmp_script)], cwd=repo_root, log_file=log_file)
        # Log the final contents of the model directory
    with log_file.open("a") as lf:
        lf.write("\nModel directory contents:\n")
        try:
            for p in sorted(model_out_dir.iterdir()):
                lf.write(f"  - {p.name}\n")
        except Exception:
            lf.write("  <failed to list directory>\n")
        else:
            # Fallback: try Accelerate gather path
            consolidate_py = f"""
import os, json, torch
from accelerate import Accelerator
from pathlib import Path
import sys
sys.path.append(str(Path('{src_dir}')))
from modeling.llama_1p5b import setup_llama_with_fone

CKPT = r"{ckpt}"
CFG  = r"{model_cfg}"
OUT  = r"{model_out_dir}"

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
        tmp_script = run_dir / "consolidate_tmp.py"
        tmp_script.write_text(consolidate_py)
        sh([sys.executable, str(tmp_script)], cwd=repo_root, log_file=log_file)

    # Run generation eval on the consolidated model, logs to run.log
    gen_json = run_dir / f"text_gen_eval_{step_name}.json"
    cmd = [
        sys.executable, str(src_dir / "eval" / "text_generation_eval.py"),
        "--model", str(model_out_dir),
        "--prompts_file", str(Path(args.prompts_file).resolve()),
        "--num_samples", str(args.num_samples),
        "--max_new_tokens", str(args.max_new_tokens),
        "--temperature", str(args.temperature),
        "--output_file", str(gen_json),
    ]
    if args.device:
        cmd.extend(["--device", args.device])
    sh(cmd, cwd=repo_root, log_file=log_file)

    # Save manifest
    manifest = {
        "checkpoint": str(ckpt),
        "consolidated_model": str(model_out_dir),
        "eval_json": str(gen_json),
        "prompts_file": str(Path(args.prompts_file).resolve()),
        "num_samples": args.num_samples,
        "max_new_tokens": args.max_new_tokens,
        "temperature": args.temperature,
        "device": args.device,
    }
    with (run_dir / "manifest.json").open("w") as f:
        json.dump(manifest, f, indent=2)

    with log_file.open("a") as lf:
        lf.write("\nRun completed successfully.\n")


if __name__ == "__main__":
    main()


