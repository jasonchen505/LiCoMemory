#!/bin/bash
# Download LOCOMO dataset from Hugging Face

echo "Downloading LOCOMO dataset..."

# Create dataset directory
mkdir -p /home/chenyizhou/LiCoMemory/dataset/locomo_raw

# Try to download from Hugging Face
pip install huggingface_hub

python3 << 'EOF'
from huggingface_hub import hf_hub_download
import json
import os

# Download LOCOMO dataset
try:
    # Try to download the dataset
    file_path = hf_hub_download(
        repo_id="snap-research/locomo",
        filename="locomo10.json",
        repo_type="dataset",
        cache_dir="/home/chenyizhou/.cache/huggingface/hub"
    )
    print(f"Downloaded to: {file_path}")
    
    # Copy to our dataset directory
    import shutil
    dest = "/home/chenyizhou/LiCoMemory/dataset/locomo_raw/locomo10.json"
    shutil.copy(file_path, dest)
    print(f"Copied to: {dest}")
    
except Exception as e:
    print(f"Error downloading from Hugging Face: {e}")
    print("\nAlternative: Please download manually from https://github.com/snap-research/locomo")
    print("And place locomo10.json in /home/chenyizhou/LiCoMemory/dataset/locomo_raw/")
EOF

echo "Download complete!"
