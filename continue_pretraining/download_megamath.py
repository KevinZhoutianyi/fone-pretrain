#!/usr/bin/env python3
"""Download MegaMath-web-pro dataset from HuggingFace."""

from huggingface_hub import snapshot_download
import os

def main():
    data_dir = "/data/megamath-web-pro"
    os.makedirs(data_dir, exist_ok=True)
    
    print(f"Downloading MegaMath-web-pro to {data_dir}...")
    
    snapshot_download(
        repo_id="LLM360/MegaMath",
        local_dir=data_dir,
        repo_type="dataset",
        allow_patterns=["megamath-web-pro/*"]
    )
    
    print(f"✅ Download complete! Data saved to: {data_dir}")

if __name__ == "__main__":
    main()

