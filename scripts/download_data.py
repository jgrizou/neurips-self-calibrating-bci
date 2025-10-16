#!/usr/bin/env python3
"""
Download large data files from Hugging Face Hub.
This script automatically downloads all necessary data files for the project.
"""

import os
import sys
from pathlib import Path

# Get project root
SCRIPT_DIR = Path(__file__).parent
PROJECT_ROOT = SCRIPT_DIR.parent

# Files to download from Hugging Face
# Format: (hf_path, local_path)
FILES_TO_DOWNLOAD = [
    # GAN checkpoint
    ("checkpoints/pggan_celebahq1024.pth", "checkpoints/pggan_celebahq1024.pth"),

    # Analysis results
    ("final_dfs/optim_ablation.parquet", "src/analysis/final_dfs/optim_ablation.parquet"),
    ("final_dfs/optim_rmse_y_true_vs_y_reconstructed.parquet", "src/analysis/final_dfs/optim_rmse_y_true_vs_y_reconstructed.parquet"),
    ("final_dfs/optim.parquet", "src/analysis/final_dfs/optim.parquet"),
    ("final_dfs/results_ablation.parquet", "src/analysis/final_dfs/results_ablation.parquet"),
    ("final_dfs/results_dimensionality_eeg.parquet", "src/analysis/final_dfs/results_dimensionality_eeg.parquet"),
    ("final_dfs/results_dimensionality_face.parquet", "src/analysis/final_dfs/results_dimensionality_face.parquet"),
    ("final_dfs/results.parquet", "src/analysis/final_dfs/results.parquet"),

    # Plot files
    ("plots/diagonal.eps", "src/analysis/plots/faces/diagonal.eps"),
    ("plots/diagonal.png", "src/analysis/plots/faces/diagonal.png"),
    ("plots/diagonal.svg", "src/analysis/plots/faces/diagonal.svg"),
    ("plots/random.eps", "src/analysis/plots/faces/random.eps"),
    ("plots/random.png", "src/analysis/plots/faces/random.png"),
    ("plots/random.svg", "src/analysis/plots/faces/random.svg"),
]

# Hugging Face repo ID
HF_REPO_ID = "jgrizou/neurips-self-calibrating-bci-data"


def download_file(hf_path, local_path):
    """Download a single file from Hugging Face Hub."""
    from huggingface_hub import hf_hub_download

    local_full_path = PROJECT_ROOT / local_path

    # Skip if file already exists
    if local_full_path.exists():
        print(f"✓ Already exists: {local_path}")
        return

    # Create directory if needed
    local_full_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"⬇ Downloading: {local_path} ...", end=" ", flush=True)

    try:
        downloaded_path = hf_hub_download(
            repo_id=HF_REPO_ID,
            filename=hf_path,
            repo_type="dataset",
            local_dir=PROJECT_ROOT,
            local_dir_use_symlinks=False
        )

        # Move to correct location if needed
        if not local_full_path.exists():
            downloaded_path = Path(downloaded_path)
            downloaded_path.rename(local_full_path)

        print("✓ Done")
    except Exception as e:
        print(f"✗ Failed: {e}")
        return False

    return True


def main():
    print("=" * 60)
    print("Downloading data files from Hugging Face Hub")
    print("=" * 60)
    print()

    # Check if huggingface_hub is installed
    try:
        import huggingface_hub
    except ImportError:
        print("ERROR: huggingface_hub is not installed.")
        print()
        print("Please install it:")
        print("  pip install huggingface_hub")
        print()
        sys.exit(1)

    print(f"Repository: {HF_REPO_ID}")
    print(f"Downloading {len(FILES_TO_DOWNLOAD)} files...")
    print()

    failed = []
    for hf_path, local_path in FILES_TO_DOWNLOAD:
        success = download_file(hf_path, local_path)
        if success is False:
            failed.append(local_path)

    print()
    print("=" * 60)
    if failed:
        print(f"⚠ {len(failed)} file(s) failed to download:")
        for f in failed:
            print(f"  - {f}")
        print()
        print("Please contact jonathan.grizou@grizai.com for assistance.")
        sys.exit(1)
    else:
        print("✓ All files downloaded successfully!")
        print()
        print("You can now run the experiments and analysis.")
    print("=" * 60)


if __name__ == "__main__":
    main()
