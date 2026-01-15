#!/usr/bin/env python3
"""
Upload finetuned model to HuggingFace Hub and cleanup local checkpoints
"""

import argparse
import os
import shutil
from pathlib import Path
from huggingface_hub import HfApi, create_repo
from transformers import AutoTokenizer, AutoModelForCausalLM
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


def upload_model(model_dir: Path, repo_id: str, token: str = None):
    """Upload model to HuggingFace Hub"""
    print(f"\n{'='*80}")
    print(f"📤 Uploading model to HuggingFace Hub")
    print(f"{'='*80}")
    print(f"Model directory: {model_dir}")
    print(f"Repository: {repo_id}")
    
    # Get token from environment if not provided
    if token is None:
        token = os.getenv("HF_TOKEN")
        if not token:
            raise ValueError("HF_TOKEN not found. Please set it in .env or pass --token")
    
    # Create repo if it doesn't exist
    print("\n🔧 Creating repository...")
    try:
        create_repo(repo_id, token=token, exist_ok=True)
        print(f"✅ Repository ready: https://huggingface.co/{repo_id}")
    except Exception as e:
        print(f"⚠️  Repository might already exist: {e}")
    
    # Upload model files
    print("\n📤 Uploading model files...")
    api = HfApi()
    
    try:
        api.upload_folder(
            folder_path=str(model_dir),
            repo_id=repo_id,
            repo_type="model",
            token=token,
        )
        print(f"✅ Model uploaded successfully!")
        print(f"🔗 View at: https://huggingface.co/{repo_id}")
        return True
    except Exception as e:
        print(f"❌ Upload failed: {e}")
        return False


def cleanup_checkpoints(output_dir: Path, keep_merged: bool = True):
    """Remove checkpoint directories to free up space"""
    print(f"\n{'='*80}")
    print(f"🧹 Cleaning up checkpoints")
    print(f"{'='*80}")
    print(f"Output directory: {output_dir}")
    
    if not output_dir.exists():
        print(f"❌ Directory not found: {output_dir}")
        return
    
    total_freed = 0
    dirs_removed = []
    
    # Find all checkpoint directories
    for item in output_dir.iterdir():
        if item.is_dir():
            # Remove checkpoint-* directories
            if item.name.startswith("checkpoint-"):
                size = sum(f.stat().st_size for f in item.rglob('*') if f.is_file())
                shutil.rmtree(item)
                total_freed += size
                dirs_removed.append(item.name)
                print(f"  ✓ Removed: {item.name} ({size / 1e9:.2f} GB)")
            
            # Optionally remove final_model (LoRA adapters)
            elif item.name == "final_model" and not keep_merged:
                size = sum(f.stat().st_size for f in item.rglob('*') if f.is_file())
                shutil.rmtree(item)
                total_freed += size
                dirs_removed.append(item.name)
                print(f"  ✓ Removed: {item.name} ({size / 1e9:.2f} GB)")
    
    print(f"\n📊 Summary:")
    print(f"  Directories removed: {len(dirs_removed)}")
    print(f"  Space freed: {total_freed / 1e9:.2f} GB")
    
    if not dirs_removed:
        print("  ℹ️  No checkpoints found to remove")


def main():
    parser = argparse.ArgumentParser(
        description="Upload model to HuggingFace and cleanup checkpoints"
    )
    
    parser.add_argument(
        "--model-dir",
        type=str,
        required=True,
        help="Path to model directory (e.g., outputs/gsm8k_xxx/merged_model)"
    )
    
    parser.add_argument(
        "--repo-id",
        type=str,
        required=True,
        help="HuggingFace repository ID (e.g., username/model-name)"
    )
    
    parser.add_argument(
        "--token",
        type=str,
        default=None,
        help="HuggingFace token (defaults to HF_TOKEN env var)"
    )
    
    parser.add_argument(
        "--cleanup",
        action="store_true",
        help="Remove checkpoint directories after upload"
    )
    
    parser.add_argument(
        "--cleanup-only",
        action="store_true",
        help="Only cleanup checkpoints, skip upload"
    )
    
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory containing checkpoints (auto-detected from model-dir)"
    )
    
    args = parser.parse_args()
    
    model_dir = Path(args.model_dir)
    
    if not model_dir.exists():
        print(f"❌ Model directory not found: {model_dir}")
        return 1
    
    # Auto-detect output directory (parent of merged_model)
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        # If model_dir ends with merged_model or final_model, use parent
        if model_dir.name in ["merged_model", "final_model"]:
            output_dir = model_dir.parent
        else:
            output_dir = model_dir
    
    print(f"\n{'='*80}")
    print(f"🚀 Model Upload & Cleanup Tool")
    print(f"{'='*80}")
    
    success = True
    
    # Upload model
    if not args.cleanup_only:
        success = upload_model(model_dir, args.repo_id, args.token)
    
    # Cleanup checkpoints
    if (args.cleanup or args.cleanup_only) and success:
        cleanup_checkpoints(output_dir)
    
    print(f"\n{'='*80}")
    if success:
        print("✅ All tasks completed successfully!")
    else:
        print("⚠️  Some tasks failed. Check the output above.")
    print(f"{'='*80}\n")
    
    return 0 if success else 1


if __name__ == "__main__":
    exit(main())

