#!/usr/bin/env python3
"""
Download specific DCLM-RefinedWeb shards for FoNE experiments.

Follows DCLM's official compute scales:
- 1B-1x (sanity): global-shard_03_of_10 / local-shard_1_of_10
- 1.5B-1x (~30B tokens): ~20 shards
- 1.5B-2x (~60B tokens): ~40 shards
- 7B-1x: global-shard_03_of_10 / local-shards {1,3,6,8,9}
"""

import argparse
import os
from pathlib import Path
from huggingface_hub import hf_hub_download, snapshot_download
from datasets import load_dataset
import json


def estimate_tokens_per_shard():
    """DCLM has ~3.8T tokens across 27838 shards → ~137M tokens/shard"""
    return 137_000_000


def download_for_sanity_check(output_dir: Path):
    """
    Download 1B-1x scale: just 1 shard for quick iteration.
    Target: ~137M tokens (1 shard)
    """
    print("=" * 60)
    print("📦 SANITY CHECK MODE: 1B-1x scale")
    print("=" * 60)
    print(f"Target: ~137M tokens (1 shard)")
    print(f"Output: {output_dir}")
    print()
    
    # Download just the first shard for testing
    ds = load_dataset(
        "mlfoundations/dclm-baseline-1.0",
        split="train",
        streaming=True,
    )
    
    # Take just one shard worth of data
    print("⬇️  Downloading first shard (~137M tokens)...")
    examples = []
    
    # Estimate: each shard ~1000-2000 examples
    shard_size = 1500
    
    for i, example in enumerate(ds):
        examples.append(example)
        if i >= shard_size:
            break
        if i % 100 == 0:
            print(f"   Downloaded {i} examples...", end='\r')
    
    # Save as parquet
    output_dir.mkdir(parents=True, exist_ok=True)
    shard_file = output_dir / "shard_00000.parquet"
    
    import pandas as pd
    import pyarrow.parquet as pq
    import pyarrow as pa
    
    df = pd.DataFrame(examples)
    table = pa.Table.from_pandas(df)
    pq.write_table(table, shard_file)
    
    print(f"\n✅ Saved: {shard_file}")
    print(f"   Examples: {len(examples)}")
    print(f"   Estimated tokens: ~137M")
    
    return 1


def download_for_1p5b_1x(output_dir: Path):
    """
    Download 1.5B-1x scale: ~30B tokens
    Target: ~220 shards (30B / 137M per shard)
    """
    print("=" * 60)
    print("📦 1.5B-1x SCALE (Compute-Optimal)")
    print("=" * 60)
    print(f"Target: ~30B tokens (~220 shards)")
    print(f"Output: {output_dir}")
    print()
    
    num_shards = 220
    return download_n_shards(output_dir, num_shards)


def download_for_1p5b_2x(output_dir: Path):
    """
    Download 1.5B-2x scale: ~60B tokens
    Target: ~440 shards
    """
    print("=" * 60)
    print("📦 1.5B-2x SCALE (Data-Rich)")
    print("=" * 60)
    print(f"Target: ~60B tokens (~440 shards)")
    print(f"Output: {output_dir}")
    print()
    
    num_shards = 440
    return download_n_shards(output_dir, num_shards)


def download_n_shards(output_dir: Path, num_shards: int):
    """Download N shards efficiently"""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    ds = load_dataset(
        "mlfoundations/dclm-baseline-1.0",
        split="train",
        streaming=True,
    )
    
    print(f"⬇️  Downloading {num_shards} shards...")
    print(f"   Estimated size: ~{num_shards * 100 / 1024:.1f} GB")
    print()
    
    examples_per_shard = 1500
    total_examples = num_shards * examples_per_shard
    
    current_shard = []
    shard_idx = 0
    
    import pandas as pd
    import pyarrow.parquet as pq
    import pyarrow as pa
    from tqdm import tqdm
    
    for i, example in enumerate(tqdm(ds, total=total_examples, desc="Downloading")):
        current_shard.append(example)
        
        if len(current_shard) >= examples_per_shard:
            # Save shard
            shard_file = output_dir / f"shard_{shard_idx:05d}.parquet"
            df = pd.DataFrame(current_shard)
            table = pa.Table.from_pandas(df)
            pq.write_table(table, shard_file)
            
            current_shard = []
            shard_idx += 1
            
            if shard_idx >= num_shards:
                break
    
    # Save remaining
    if current_shard and shard_idx < num_shards:
        shard_file = output_dir / f"shard_{shard_idx:05d}.parquet"
        df = pd.DataFrame(current_shard)
        table = pa.Table.from_pandas(df)
        pq.write_table(table, shard_file)
        shard_idx += 1
    
    print(f"\n✅ Download complete!")
    print(f"   Shards saved: {shard_idx}")
    print(f"   Location: {output_dir}")
    print(f"   Estimated tokens: ~{shard_idx * 137:.0f}M = ~{shard_idx * 0.137:.1f}B")
    
    return shard_idx


def create_data_config(output_dir: Path, num_shards: int, target_tokens: int):
    """Create a data config file pointing to local shards"""
    config_path = Path("/home/nvidia/fone-pretrain/configs") / f"data_local_{target_tokens // 1_000_000_000}b.json"
    
    config = {
        "streams": [
            {
                "name": "dclm_local",
                "path": str(output_dir / "*.parquet"),
                "weight": 1.0
            }
        ],
        "total_tokens_target": target_tokens,
        "sequence_length": 2048,
        "shuffle_buffer_size": 1000,
        "num_workers": 0
    }
    
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)
    
    print(f"\n📝 Created config: {config_path}")
    return config_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download DCLM data for FoNE experiments")
    parser.add_argument(
        "--scale",
        type=str,
        choices=["sanity", "1p5b-1x", "1p5b-2x", "custom"],
        default="sanity",
        help="Which scale to download",
    )
    parser.add_argument(
        "--num_shards",
        type=int,
        help="For custom: number of shards to download",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="/home/nvidia/data/dclm_local",
        help="Where to save shards",
    )
    
    args = parser.parse_args()
    output_dir = Path(args.output_dir)
    
    if args.scale == "sanity":
        num_shards = download_for_sanity_check(output_dir)
        target_tokens = 137_000_000  # ~137M
        
    elif args.scale == "1p5b-1x":
        num_shards = download_for_1p5b_1x(output_dir)
        target_tokens = 30_000_000_000  # 30B
        
    elif args.scale == "1p5b-2x":
        num_shards = download_for_1p5b_2x(output_dir)
        target_tokens = 60_000_000_000  # 60B
        
    elif args.scale == "custom":
        if not args.num_shards:
            print("❌ Error: --num_shards required for custom scale")
            exit(1)
        num_shards = download_n_shards(output_dir, args.num_shards)
        target_tokens = num_shards * 137_000_000
    
    # Create config file
    config_path = create_data_config(output_dir, num_shards, target_tokens)
    
    print("\n" + "=" * 60)
    print("✅ ALL DONE!")
    print("=" * 60)
    print(f"Shards downloaded: {num_shards}")
    print(f"Data location: {output_dir}")
    print(f"Config file: {config_path}")
    print()
    print("📝 Next steps:")
    print(f"   1. export DATA_CONFIG={config_path}")
    print(f"   2. bash scripts/pretrain_0p5b_local.sh")
    print()

